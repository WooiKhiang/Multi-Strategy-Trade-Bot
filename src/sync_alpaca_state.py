from __future__ import annotations

import os
import requests
from datetime import datetime, timezone
import pandas as pd

from .config import Config
from .logger import get_logger
from .sheets import open_sheet, ensure_worksheet, read_worksheet_df, clear_and_write, append_rows


CAPITAL_HEADERS = [
    "timestamp_utc",
    "mode",
    "endpoint",
    "cash",
    "buying_power",
    "portfolio_value",
    "equity",
    "last_equity",
    "currency",
    "status",
]

ORDERS_HEADERS = [
    "timestamp_utc",
    "mode",
    "order_id",
    "client_order_id",
    "symbol",
    "side",
    "type",
    "qty",
    "filled_qty",
    "limit_price",
    "stop_price",
    "status",
    "submitted_at",
    "filled_at",
    "filled_avg_price",
    "updated_at",
]

LOG_HEADERS = ["timestamp_utc", "component", "level", "message"]


def utc_iso_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _env(name: str, default: str = "") -> str:
    v = os.getenv(name, default)
    return v.strip() if isinstance(v, str) else default


def _alpaca_headers() -> dict:
    key = _env("ALPACA_API_KEY")
    secret = _env("ALPACA_API_SECRET")
    if not key or not secret:
        raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_API_SECRET in env")
    return {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
        "Accept": "application/json",
    }


def _alpaca_base_url() -> str:
    # Prefer explicit base url; otherwise infer from ALPACA_PAPER
    base = _env("ALPACA_BASE_URL")
    if base:
        return base.rstrip("/")
    paper = _env("ALPACA_PAPER", "true").lower() in ("true", "1", "yes", "y")
    return ("https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets").rstrip("/")


def _alpaca_get(path: str, params: dict | None = None) -> dict | list:
    base = _alpaca_base_url()
    url = f"{base}{path}"
    r = requests.get(url, headers=_alpaca_headers(), params=params, timeout=30)
    if r.status_code >= 400:
        raise RuntimeError(f"Alpaca GET {path} failed: {r.status_code} {r.text[:300]}")
    return r.json()


def _mode_label() -> str:
    paper = _env("ALPACA_PAPER", "true").lower() in ("true", "1", "yes", "y")
    return "PAPER" if paper else "LIVE"


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


def sync_alpaca_state():
    cfg = Config()
    logger = get_logger("sync_alpaca", cfg.log_level)

    if not cfg.gsheet_id:
        raise RuntimeError("GSHEET_ID missing")

    # --- Open Sheets
    ss = open_sheet(cfg.gsheet_id, cfg.google_service_account_json)

    ws_capital = ensure_worksheet(ss, "Capital", CAPITAL_HEADERS)
    ws_orders = ensure_worksheet(ss, "Orders", ORDERS_HEADERS)
    ws_logs = ensure_worksheet(ss, "Logs", LOG_HEADERS)

    now = utc_iso_z()
    mode = _mode_label()
    endpoint = _alpaca_base_url()

    # --- Fetch account snapshot
    acct = _alpaca_get("/v2/account")
    cash = _to_float(acct.get("cash", 0))
    buying_power = _to_float(acct.get("buying_power", 0))
    portfolio_value = _to_float(acct.get("portfolio_value", 0))
    equity = _to_float(acct.get("equity", 0))
    last_equity = _to_float(acct.get("last_equity", 0))
    currency = str(acct.get("currency", "") or "")
    status = str(acct.get("status", "") or "")

    cap_row = [[
        now, mode, endpoint,
        cash, buying_power, portfolio_value, equity, last_equity, currency, status
    ]]

    # Append to capital history (keeps timeline)
    append_rows(ws_capital, cap_row)

    # --- Fetch orders snapshot (recent-ish)
    # Keep it simple: get all orders (up to 500) newest first; you can tighten later.
    orders = _alpaca_get("/v2/orders", params={"status": "all", "direction": "desc", "limit": 200})

    # Normalize orders into dataframe
    order_rows = []
    for o in orders:
        order_rows.append({
            "timestamp_utc": now,
            "mode": mode,
            "order_id": str(o.get("id", "")),
            "client_order_id": str(o.get("client_order_id", "")),
            "symbol": str(o.get("symbol", "")),
            "side": str(o.get("side", "")),
            "type": str(o.get("type", "")),
            "qty": _to_float(o.get("qty", 0)),
            "filled_qty": _to_float(o.get("filled_qty", 0)),
            "limit_price": _to_float(o.get("limit_price", 0)),
            "stop_price": _to_float(o.get("stop_price", 0)),
            "status": str(o.get("status", "")),
            "submitted_at": str(o.get("submitted_at", "")),
            "filled_at": str(o.get("filled_at", "")),
            "filled_avg_price": _to_float(o.get("filled_avg_price", 0)),
            "updated_at": str(o.get("updated_at", "")),
        })

    odf = pd.DataFrame(order_rows) if order_rows else pd.DataFrame(columns=ORDERS_HEADERS)

    # Overwrite Orders tab with latest snapshot (keeps sheet small + fast)
    # If you prefer append-only history later, we can change this.
    clear_and_write(ws_orders, ORDERS_HEADERS, odf[ORDERS_HEADERS] if not odf.empty else odf)

    msg = f"Synced account + orders. cash={round(cash,2)} buying_power={round(buying_power,2)} orders={len(order_rows)}"
    logger.info(msg)
    append_rows(ws_logs, [[now, "sync_alpaca", "INFO", msg]])


if __name__ == "__main__":
    sync_alpaca_state()
