from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
import pandas as pd

from .config import Config
from .logger import get_logger
from .data_yf import download_batched
from .strategy import check_touch_and_confirm_5m
from .sheets import open_sheet, ensure_worksheet, read_worksheet_df, append_df


SIGNAL_HEADERS = [
    "timestamp_utc",
    "symbol",
    "ref_time",
    "ref_low",
    "atr",
    "touch_level",
    "rejection_time",
    "rejection_high",
    "confirm_time",
    "confirm_close",
    "note",
]

CACHE_DIR = "data"
FIRED_CACHE_FILE = os.path.join(CACHE_DIR, "fired_signals.json")


def _utcnow_iso() -> str:
    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()


def _load_fired_cache() -> dict:
    os.makedirs(CACHE_DIR, exist_ok=True)
    if not os.path.exists(FIRED_CACHE_FILE):
        return {"fired": []}
    try:
        with open(FIRED_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"fired": []}


def _save_fired_cache(cache: dict) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(FIRED_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _is_duplicate(symbol: str, ref_time: str, cache: dict) -> bool:
    key = f"{symbol}__{ref_time}"
    return key in set(cache.get("fired", []))


def _mark_fired(symbol: str, ref_time: str, cache: dict) -> None:
    key = f"{symbol}__{ref_time}"
    fired = cache.get("fired", [])
    if key not in fired:
        fired.append(key)
    # keep cache from growing forever
    cache["fired"] = fired[-2000:]


def _parse_iso(dt_str: str) -> datetime | None:
    if not dt_str:
        return None
    try:
        # Example: 2026-01-10T12:34:56.123456
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None


def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _clean_candidates(df: pd.DataFrame, logger) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    # Normalize col names we expect
    required = ["symbol", "ref_time", "ref_low", "atr", "expires_utc", "status"]
    for c in required:
        if c not in df.columns:
            logger.warning(f"Candidates sheet missing column: {c}")
            return pd.DataFrame()

    # Trim and normalize
    df = df.copy()
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df["status"] = df["status"].astype(str).str.upper().str.strip()

    # Filter ACTIVE
    df = df[df["status"] == "ACTIVE"]

    # Filter not expired
    now = datetime.utcnow()
    def not_expired(row) -> bool:
        exp = _parse_iso(str(row.get("expires_utc", "")))
        if exp is None:
            return True
        return exp >= now

    df = df[df.apply(not_expired, axis=1)]

    # Keep only needed columns
    return df[required].drop_duplicates(subset=["symbol", "ref_time"])


def run_once(cfg: Config, logger) -> int:
    """
    One monitoring tick:
    - read candidates
    - pull 5m bars for active tickers
    - confirm signals
    - append signals to sheet
    """
    if not cfg.gsheet_id:
        raise RuntimeError("GSHEET_ID missing in env/.env")

    ss = open_sheet(cfg.gsheet_id, cfg.google_service_account_json)
    ws_candidates = ensure_worksheet(ss, "Candidates", [])
    ws_signals = ensure_worksheet(ss, "Signals", SIGNAL_HEADERS)
    ws_logs = ensure_worksheet(ss, "Logs", ["timestamp_utc", "component", "level", "message"])

    cand_df = read_worksheet_df(ws_candidates)
    cand_df = _clean_candidates(cand_df, logger)

    if cand_df.empty:
        logger.info("No ACTIVE, non-expired candidates. Nothing to monitor.")
        ws_logs.append_rows([[_utcnow_iso(), "monitor_5m", "INFO", "No active candidates"]])
        return 0

    # Duplicate protection cache (local)
    cache = _load_fired_cache()

    tickers = cand_df["symbol"].tolist()
    logger.info(f"Monitoring 5m confirmations for {len(tickers)} tickers.")

    # Download 5m data (short period to reduce load)
    data_5m = download_batched(
        tickers=tickers,
        interval="5m",
        period="5d",
        batch_size=min(cfg.yf_batch_size, 50),
        sleep_sec=cfg.yf_sleep_between_batch_sec,
        logger=logger,
        cache_key_prefix="yf5m",
    )

    signal_rows = []
    fired_now = 0

    for _, row in cand_df.iterrows():
        sym = row["symbol"]
        ref_time = str(row["ref_time"])
        if _is_duplicate(sym, ref_time, cache):
            continue

        df5 = data_5m.get(sym)
        if df5 is None or df5.empty:
            continue

        ref_low = _safe_float(row["ref_low"])
        atr_val = _safe_float(row["atr"])
        if atr_val <= 0 or ref_low <= 0:
            continue

        result = check_touch_and_confirm_5m(
            df_5m=df5,
            ref_low=ref_low,
            atr_val=atr_val,
            touch_buffer_atr_mult=cfg.touch_buffer_atr_mult,
            confirm_break_buffer_atr_mult=cfg.confirm_break_buffer_atr_mult,
        )

        if not result:
            continue

        # Mark fired (local cache)
        _mark_fired(sym, ref_time, cache)
        fired_now += 1

        signal_rows.append({
            "timestamp_utc": _utcnow_iso(),
            "symbol": sym,
            "ref_time": ref_time,
            "ref_low": ref_low,
            "atr": atr_val,
            "touch_level": result.get("touch_level", ""),
            "rejection_time": result.get("rejection_time", ""),
            "rejection_high": result.get("rejection_high", ""),
            "confirm_time": result.get("confirm_time", ""),
            "confirm_close": result.get("confirm_close", ""),
            "note": "5m confirm: touch+rejection then confirm close above rejection high",
        })

    # Persist cache
    _save_fired_cache(cache)

    if signal_rows:
        out_df = pd.DataFrame(signal_rows, columns=SIGNAL_HEADERS)
        append_df(ws_signals, out_df, SIGNAL_HEADERS)
        msg = f"Signals appended={len(out_df)} (new fired={fired_now})"
        logger.info(msg)
        ws_logs.append_rows([[_utcnow_iso(), "monitor_5m", "INFO", msg]])
        return len(out_df)

    msg = "No confirmations this tick."
    logger.info(msg)
    ws_logs.append_rows([[_utcnow_iso(), "monitor_5m", "INFO", msg]])
    return 0


def loop_forever(poll_seconds: int = 60):
    cfg = Config()
    logger = get_logger("monitor_5m", cfg.log_level)

    logger.info("Starting monitor_5m loop...")
    while True:
        try:
            run_once(cfg, logger)
        except Exception as e:
            logger.exception(f"monitor_5m tick failed: {e}")
        time.sleep(poll_seconds)
