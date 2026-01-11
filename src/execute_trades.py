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


def _to_int(x, default=0) -> int:
    try:
        return int(float(str(x).strip()))
    except Exception:
        return int(default)


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


def _alpaca_get(path: str, params: dict | None = None) -> dict | list:
    url = f"{_alpaca_base_url()}{path}"
    r = requests.get(url, headers=_alpaca_headers(), params=params, timeout=30)
    if r.status_code >= 400:
        raise RuntimeError(f"Alpaca GET {path} failed: {r.status_code} {r.text[:300]}")
    return r.json()


def _alpaca_post(path: str, payload: dict) -> dict:
    url = f"{_alpaca_base_url()}{path}"
    r = requests.post(url, headers=_alpaca_headers(), json=payload, timeout=30)
    if r.status_code >= 400:
        raise RuntimeError(f"Alpaca POST {path} failed: {r.status_code} {r.text[:300]}")
    return r.json()


def _alpaca_symbol(symbol: str) -> str:
    s = (symbol or "").strip().upper()
    # common yfinance -> alpaca symbol mapping
    if s == "BRK-B":
        return "BRK.B"
    if s == "BF-B":
        return "BF.B"
    return s


def _round_price_for_us_equity(price: float) -> float:
    """
    Alpaca enforces tick sizes:
    - >= $1: $0.01 increments (2 decimals)
    - <  $1: allow 4 decimals
    """
    if price <= 0:
        return price
    if price >= 1.0:
        return round(price, 2)
    return round(price, 4)


def _client_order_id(symbol: str, ref_time: str) -> str:
    base = (ref_time or "").replace(" ", "").replace(":", "").replace("-", "")
    base = base[:12]
    return f"VG-{symbol}-{base}"[:48]


def _get_qty(row: pd.Series) -> int:
    """
    Accept common header variations:
    - order_qty
    - order qty
    - qty
    - shares
    """
    for key in ("order_qty", "order qty", "qty", "shares"):
        if key in row.index:
            q = _to_int(row.get(key, 0), 0)
            if q > 0:
                return q
    return 0


def execute_selected_trades():
    cfg = Config()
    logger = get_logger("execute_trades", cfg.log_level)

    if not cfg.gsheet_id:
        raise RuntimeError("GSHEET_ID missing")

    ss = open_sheet(cfg.gsheet_id, cfg.google_service_account_json)
    ws_trades = ensure_worksheet(ss, "Trades", TRADES_HEADERS)
    ws_logs = ensure_worksheet(ss, "Logs", LOG_HEADERS)

    # ---- HARD PROOF: confirm Alpaca auth + endpoint ----
    try:
        base = _alpaca_base_url()
        acct = _alpaca_get("/v2/account")
        msg = f"Alpaca OK base={base} status={acct.get('status')} cash={acct.get('cash')}"
        logger.info(msg)
        append_rows(ws_logs, [[utc_iso_z(), "execute_trades", "INFO", msg]])
    except Exception as e:
        msg = f"Alpaca auth/check failed: {e}"
        logger.error(msg)
        append_rows(ws_logs, [[utc_iso_z(), "execute_trades", "ERROR", msg]])
        raise

    df = read_worksheet_df(ws_trades)
    if df is None or df.empty:
        msg = "Trades sheet is empty."
        logger.info(msg)
        append_rows(ws_logs, [[utc_iso_z(), "execute_trades", "INFO", msg]])
        return

    # Ensure expected columns exist
    for c in TRADES_HEADERS:
        if c not in df.columns:
            df[c] = ""

    # Normalize flags
    df["selected"] = df["selected"].astype(str).str.upper().str.strip()
    df["status"] = df["status"].astype(str).str.upper().str.strip()

    todo = df[(df["selected"] == "TRUE") & (df["status"] == "NEW")].copy()

    msg = f"Found selected NEW trades: {len(todo)}"
    logger.info(msg)
    append_rows(ws_logs, [[utc_iso_z(), "execute_trades", "INFO", msg]])

    if todo.empty:
        df = df[TRADES_HEADERS]
        clear_and_write(ws_trades, TRADES_HEADERS, df)
        return

    max_orders = _to_int(_env("MAX_EXEC_ORDERS", "5"), 5)
    todo = todo.head(max_orders)

    executed = 0
    skipped = 0
    errors = 0

    for idx, r in todo.iterrows():
        symbol_raw = str(r.get("symbol", "")).strip().upper()
        symbol = _alpaca_symbol(symbol_raw)
        ref_time = str(r.get("ref_time", "")).strip()

        qty = _get_qty(r)

        tp_raw = _to_float(r.get("take_profit", 0), 0)
        sl_raw = _to_float(r.get("stop_loss", 0), 0)

        tp = _round_price_for_us_equity(tp_raw)
        sl = _round_price_for_us_equity(sl_raw)

        if not symbol or qty <= 0 or tp <= 0 or sl <= 0:
            skipped += 1
            msg = f"SKIP idx={idx} {symbol_raw}->{symbol} qty={qty} tp={tp_raw}->{tp} sl={sl_raw}->{sl}"
            logger.info(msg)
            append_rows(ws_logs, [[utc_iso_z(), "execute_trades", "INFO", msg]])
            continue

        cid = _client_order_id(symbol, ref_time)

        payload = {
            "symbol": symbol,
            "qty": str(qty),
            "side": "buy",
            "type": "market",
            "time_in_force": "day",
            "order_class": "bracket",
            "take_profit": {"limit_price": str(tp)},
            "stop_loss": {"stop_price": str(sl)},
            "client_order_id": cid,
        }

        try:
            resp = _alpaca_post("/v2/orders", payload)
            oid = str(resp.get("id", ""))
            submitted_at = str(resp.get("submitted_at", ""))

            # Read-back verify
            confirm = _alpaca_get(f"/v2/orders/{oid}")
            c_status = confirm.get("status")

            df.at[idx, "alpaca_order_id"] = oid
            df.at[idx, "alpaca_client_order_id"] = cid
            df.at[idx, "submitted_at_utc"] = utc_iso_z()
            df.at[idx, "filled_avg_price"] = ""
            df.at[idx, "status"] = "SENT"
            df.at[idx, "note"] = (str(df.at[idx, "note"]) + " | " if str(df.at[idx, "note"]).strip() else "") + "Bracket sent"

            executed += 1
            msg = f"EXECUTED {symbol_raw}->{symbol} qty={qty} oid={oid} alpaca_status={c_status} submitted_at={submitted_at}"
            logger.info(msg)
            append_rows(ws_logs, [[utc_iso_z(), "execute_trades", "INFO", msg]])

            time.sleep(0.4)

        except Exception as e:
            errors += 1
            df.at[idx, "note"] = (str(df.at[idx, "note"]) + " | " if str(df.at[idx, "note"]).strip() else "") + f"EXEC_ERROR: {e}"
            msg = f"ERROR submit {symbol_raw}->{symbol} idx={idx}: {e}"
            logger.error(msg)
            append_rows(ws_logs, [[utc_iso_z(), "execute_trades", "ERROR", msg]])

    # Persist back to sheet
    df = df[TRADES_HEADERS]
    clear_and_write(ws_trades, TRADES_HEADERS, df)

    summary = f"Done. executed={executed} skipped={skipped} errors={errors} paper={_env('ALPACA_PAPER','true')}"
    logger.info(summary)
    append_rows(ws_logs, [[utc_iso_z(), "execute_trades", "INFO", summary]])


if __name__ == "__main__":
    execute_selected_trades()
