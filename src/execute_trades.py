from __future__ import annotations

import os
import time
from datetime import datetime, timezone
import requests
import pandas as pd

from .config import Config
from .logger import get_logger
from .sheets import open_sheet, ensure_worksheet, read_worksheet_df, clear_and_write, append_rows


LOG_HEADERS = ["timestamp_utc", "component", "level", "message"]

# Must match your Trades sheet header (code will also tolerate missing columns by adding them)
TRADES_BASE_HEADERS = [
    "timestamp_utc",
    "symbol",
    "ref_time",
    "entry",
    "stop_loss",
    "take_profit",
    "risk_per_share",
    "r_mult",
    "shares",
    "notional",
    "cost_est",
    "cost_in_r",
    "expected_net_r",
    "score",
    "priority_rank",
    "selected",
    "order_qty",
    "status",
    "note",
]

TRADES_EXTRA_HEADERS = [
    "alpaca_order_id",
    "alpaca_client_order_id",
    "submitted_at_utc",
    "filled_avg_price",
]

TRADES_HEADERS = TRADES_BASE_HEADERS + TRADES_EXTRA_HEADERS


def utc_iso_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _env(name: str, default: str = "") -> str:
    v = os.getenv(name, default)
    return v.strip() if isinstance(v, str) else default


def _to_float(x, default=0.0) -> float:
    try:
        if x is None:
            return float(default)
        s = str(x).strip()
        if s == "" or s.lower() == "nan":
            return float(default)
        return float(s)
    except Exception:
        return float(default)


def _alpaca_base_url() -> str:
    base = _env("ALPACA_BASE_URL")
    if base:
        return base.rstrip("/")
    paper = _env("ALPACA_PAPER", "true").lower() in ("true", "1", "yes", "y")
    return ("https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets").rstrip("/")


def _alpaca_headers() -> dict:
    key = _env("ALPACA_API_KEY")
    secret = _env("ALPACA_API_SECRET")
    if not key or not secret:
        raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_API_SECRET in env")
    return {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def _alpaca_post(path: str, payload: dict) -> dict:
    url = f"{_alpaca_base_url()}{path}"
    r = requests.post(url, headers=_alpaca_headers(), json=payload, timeout=30)
    if r.status_code >= 400:
        raise RuntimeError(f"Alpaca POST {path} failed: {r.status_code} {r.text[:300]}")
    return r.json()


def _client_order_id(symbol: str, ref_time: str) -> str:
    # Alpaca max length for client_order_id is limited; keep it short and stable.
    # Example: VG-SPY-20260109T1930
    base = (ref_time or "").replace(" ", "").replace(":", "").replace("-", "")
    base = base[:12]  # trim
    return f"VG-{symbol}-{base}"[:48]


def execute_selected_trades():
    cfg = Config()
    logger = get_logger("execute_trades", cfg.log_level)

    if not cfg.gsheet_id:
        raise RuntimeError("GSHEET_ID missing")

    ss = open_sheet(cfg.gsheet_id, cfg.google_service_account_json)
    ws_trades = ensure_worksheet(ss, "Trades", TRADES_HEADERS)
    ws_logs = ensure_worksheet(ss, "Logs", LOG_HEADERS)

    df = read_worksheet_df(ws_trades)
    if df is None or df.empty:
        msg = "Trades sheet is empty."
        logger.info(msg)
        append_rows(ws_logs, [[utc_iso_z(), "execute_trades", "INFO", msg]])
        return

    # Ensure required columns exist
    for c in TRADES_HEADERS:
        if c not in df.columns:
            df[c] = ""

    # Normalize selected/status
    df["selected"] = df["selected"].astype(str).str.upper().str.strip()
    df["status"] = df["status"].astype(str).str.upper().str.strip()

    # Only execute selected NEW trades
    todo = df[(df["selected"] == "TRUE") & (df["status"] == "NEW")].copy()

    if todo.empty:
        msg = "No selected NEW trades to execute."
        logger.info(msg)
        append_rows(ws_logs, [[utc_iso_z(), "execute_trades", "INFO", msg]])
        # still persist normalization
        df = df[TRADES_HEADERS]
        clear_and_write(ws_trades, TRADES_HEADERS, df)
        return

    # Safety: optional max per run (avoid blasting)
    max_orders = int(float(_env("MAX_EXEC_ORDERS", "5") or "5"))
    todo = todo.head(max_orders)

    executed = 0
    errors = 0

    for idx, r in todo.iterrows():
        symbol = str(r.get("symbol", "")).strip().upper()
        ref_time = str(r.get("ref_time", "")).strip()

        qty = int(_to_float(r.get("order_qty", 0), 0))
        tp = _to_float(r.get("take_profit", 0), 0)
        sl = _to_float(r.get("stop_loss", 0), 0)

        if not symbol or qty <= 0 or tp <= 0 or sl <= 0:
            errors += 1
            msg = f"Skip idx={idx} invalid fields symbol={symbol} qty={qty} tp={tp} sl={sl}"
            logger.error(msg)
            append_rows(ws_logs, [[utc_iso_z(), "execute_trades", "ERROR", msg]])
            continue

        cid = _client_order_id(symbol, ref_time)

        # Bracket order (market entry)
        payload = {
            "symbol": symbol,
            "qty": str(qty),
            "side": "buy",
            "type": "market",
            "time_in_force": "day",
            "order_class": "bracket",
            "take_profit": {"limit_price": str(round(tp, 4))},
            "stop_loss": {"stop_price": str(round(sl, 4))},
            "client_order_id": cid,
        }

        try:
            resp = _alpaca_post("/v2/orders", payload)
            oid = str(resp.get("id", ""))
            submitted_at = str(resp.get("submitted_at", ""))

            df.at[idx, "alpaca_order_id"] = oid
            df.at[idx, "alpaca_client_order_id"] = cid
            df.at[idx, "submitted_at_utc"] = utc_iso_z()
            df.at[idx, "status"] = "SENT"
            df.at[idx, "note"] = (str(df.at[idx, "note"]) + " | " if str(df.at[idx, "note"]).strip() else "") + "Bracket sent"

            executed += 1
            msg = f"Submitted bracket: {symbol} qty={qty} oid={oid}"
            logger.info(msg)
            append_rows(ws_logs, [[utc_iso_z(), "execute_trades", "INFO", msg]])

            # tiny pause to reduce API burst
            time.sleep(0.4)

        except Exception as e:
            errors += 1
            df.at[idx, "note"] = (str(df.at[idx, "note"]) + " | " if str(df.at[idx, "note"]).strip() else "") + f"EXEC_ERROR: {e}"
            msg = f"Failed submit: {symbol} idx={idx} err={e}"
            logger.error(msg)
            append_rows(ws_logs, [[utc_iso_z(), "execute_trades", "ERROR", msg]])

    # Write back updated Trades sheet
    df = df[TRADES_HEADERS]
    clear_and_write(ws_trades, TRADES_HEADERS, df)

    summary = f"Executed={executed} Errors={errors} (paper={_env('ALPACA_PAPER','true')})"
    logger.info(summary)
    append_rows(ws_logs, [[utc_iso_z(), "execute_trades", "INFO", summary]])


if __name__ == "__main__":
    execute_selected_trades()
