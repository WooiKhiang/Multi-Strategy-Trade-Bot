from __future__ import annotations

import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import requests
import pandas as pd

from .config import Config
from .logger import get_logger
from .sheets import open_sheet, ensure_worksheet, read_worksheet_df, clear_and_write, append_rows

LOG_HEADERS = ["timestamp_utc", "component", "level", "message"]

ORDERS_HEADERS = [
    "timestamp_utc",
    "id",
    "client_order_id",
    "symbol",
    "side",
    "type",
    "order_class",
    "qty",
    "filled_qty",
    "filled_avg_price",
    "status",
    "submitted_at",
    "filled_at",
    "canceled_at",
    "replaced_at",
    "time_in_force",
    "limit_price",
    "stop_price",
    "take_profit_price",
    "stop_loss_price",
    "parent_order_id",
]

DAILY_HEADERS = [
    "date_ny",
    "regime",
    "orders_sent",
    "entries_filled",
    "exits_filled",
    "closed_trades",
    "wins",
    "losses",
    "winrate",
    "net_pnl_usd",
]


def utc_iso_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _env(name: str, default: str = "") -> str:
    v = os.getenv(name, default)
    return v.strip() if isinstance(v, str) else default


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
        raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_API_SECRET")
    return {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def _alpaca_get(path: str, params: dict | None = None):
    url = f"{_alpaca_base_url()}{path}"
    r = requests.get(url, headers=_alpaca_headers(), params=params, timeout=30)
    if r.status_code >= 400:
        raise RuntimeError(f"Alpaca GET {path} failed: {r.status_code} {r.text[:300]}")
    return r.json()


def _ny_now() -> datetime:
    return datetime.now(ZoneInfo("America/New_York"))


def _is_market_window() -> bool:
    now = _ny_now()
    if now.weekday() >= 5:
        return False
    minutes = now.hour * 60 + now.minute
    return (9 * 60 + 25) <= minutes <= (16 * 60 + 20)


def _read_latest_regime(ss) -> str:
    ws = ensure_worksheet(ss, "Market", ["date", "spy_close", "spy_ema20", "spy_ema50", "vix", "regime", "confidence"])
    df = read_worksheet_df(ws)
    if df is None or df.empty:
        return "NEUTRAL"
    col = None
    for c in df.columns:
        if c.strip().lower() == "regime":
            col = c
            break
    if not col:
        return "NEUTRAL"
    v = str(df.iloc[-1][col]).strip().upper()
    return v if v in ("BULL", "NEUTRAL", "DEFENSIVE") else "NEUTRAL"


def _flatten_order(o: dict) -> dict:
    tp = o.get("take_profit") or {}
    sl = o.get("stop_loss") or {}

    return {
        "timestamp_utc": utc_iso_z(),
        "id": o.get("id", ""),
        "client_order_id": o.get("client_order_id", ""),
        "symbol": o.get("symbol", ""),
        "side": o.get("side", ""),
        "type": o.get("type", ""),
        "order_class": o.get("order_class", ""),
        "qty": o.get("qty", ""),
        "filled_qty": o.get("filled_qty", ""),
        "filled_avg_price": o.get("filled_avg_price", ""),
        "status": o.get("status", ""),
        "submitted_at": o.get("submitted_at", ""),
        "filled_at": o.get("filled_at", ""),
        "canceled_at": o.get("canceled_at", ""),
        "replaced_at": o.get("replaced_at", ""),
        "time_in_force": o.get("time_in_force", ""),
        "limit_price": o.get("limit_price", ""),
        "stop_price": o.get("stop_price", ""),
        "take_profit_price": tp.get("limit_price", ""),
        "stop_loss_price": sl.get("stop_price", ""),
        "parent_order_id": o.get("parent_order_id", ""),
    }


def _is_vg_parent(o: dict) -> bool:
    cid = str(o.get("client_order_id") or "")
    return cid.startswith("VG-") or cid.startswith("VG_")


def _parse_dt(dt_str: str) -> datetime | None:
    if not dt_str:
        return None
    try:
        # Alpaca uses Z
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except Exception:
        return None


def sync_orders_and_daily():
    cfg = Config()
    logger = get_logger("alpaca_sync", cfg.log_level)

    if not cfg.gsheet_id:
        raise RuntimeError("GSHEET_ID missing")

    ss = open_sheet(cfg.gsheet_id, cfg.google_service_account_json)
    ws_orders = ensure_worksheet(ss, "Orders", ORDERS_HEADERS)
    ws_daily = ensure_worksheet(ss, "Daily", DAILY_HEADERS)
    ws_logs = ensure_worksheet(ss, "Logs", LOG_HEADERS)

    if not _is_market_window():
        msg = "Outside NY market window; skip sync."
        logger.info(msg)
        append_rows(ws_logs, [[utc_iso_z(), "alpaca_sync", "INFO", msg]])
        return

    acct = _alpaca_get("/v2/account")
    msg = f"Alpaca OK base={_alpaca_base_url()} status={acct.get('status')} equity={acct.get('equity')}"
    logger.info(msg)
    append_rows(ws_logs, [[utc_iso_z(), "alpaca_sync", "INFO", msg]])

    limit = int(float(_env("ALPACA_SYNC_LIMIT", "300")))

    raw = _alpaca_get(
        "/v2/orders",
        params={
            "status": "all",
            "nested": "true",
            "limit": str(limit),
            "direction": "desc",
        },
    )
    if not isinstance(raw, list):
        raise RuntimeError("Unexpected Alpaca orders response")

    # 1) Identify VG parent entry orders (bracket buy parents)
    vg_parents = [o for o in raw if _is_vg_parent(o)]

    # 2) Collect all legs for each parent (nested=true usually includes 'legs')
    collected = []
    seen_ids = set()

    def add_order(o: dict):
        oid = o.get("id")
        if oid and oid in seen_ids:
            return
        if oid:
            seen_ids.add(oid)
        collected.append(o)

    for p in vg_parents:
        add_order(p)
        legs = p.get("legs") or []
        if isinstance(legs, list):
            for leg in legs:
                # make sure parent_order_id is present
                if not leg.get("parent_order_id"):
                    leg["parent_order_id"] = p.get("id", "")
                add_order(leg)

    # 3) Also include any standalone child orders returned top-level (some accounts show this)
    parent_ids = {p.get("id") for p in vg_parents if p.get("id")}
    for o in raw:
        if o.get("parent_order_id") in parent_ids:
            add_order(o)

    # Write Orders snapshot
    rows = [_flatten_order(o) for o in collected]
    odf = pd.DataFrame(rows) if rows else pd.DataFrame(columns=ORDERS_HEADERS)
    odf = odf.reindex(columns=ORDERS_HEADERS)
    clear_and_write(ws_orders, ORDERS_HEADERS, odf)

    # Daily summary from Orders snapshot (robust)
    regime = _read_latest_regime(ss)
    ny_date = _ny_now().date().isoformat()

    # Define "today" in NY time by submitted_at/filled_at timestamps
    def in_today(dt_str: str) -> bool:
        dt = _parse_dt(str(dt_str))
        if not dt:
            return False
        dt_ny = dt.astimezone(ZoneInfo("America/New_York"))
        return dt_ny.date().isoformat() == ny_date

    parents_today = [p for p in vg_parents if in_today(p.get("submitted_at", ""))]
    orders_sent = len(parents_today)

    entries_filled = sum(1 for p in parents_today if str(p.get("status", "")).lower() == "filled")

    # exits are SELL legs filled today
    exits_filled = 0
    closed_trades = 0
    wins = 0
    losses = 0
    net_pnl = 0.0

    # build map parent_id -> (entry fill price, qty)
    entry_map = {}
    for p in vg_parents:
        if str(p.get("status", "")).lower() == "filled":
            try:
                entry_map[p.get("id")] = (
                    float(p.get("filled_avg_price") or 0),
                    float(p.get("filled_qty") or 0),
                )
            except Exception:
                entry_map[p.get("id")] = (0.0, 0.0)

    # compute pnl using filled exit legs (TP/SL)
    for p in parents_today:
        pid = p.get("id")
        legs = p.get("legs") or []
        filled_sell_legs = [l for l in legs if str(l.get("side", "")).lower() == "sell" and str(l.get("status", "")).lower() == "filled" and in_today(l.get("filled_at", ""))]
        if not filled_sell_legs:
            continue

        # bracket should have at most one filled exit
        leg = sorted(filled_sell_legs, key=lambda x: str(x.get("filled_at") or ""))[-1]
        exits_filled += 1
        closed_trades += 1

        exit_price = float(leg.get("filled_avg_price") or 0.0)
        exit_qty = float(leg.get("filled_qty") or 0.0)

        entry_price, entry_qty = entry_map.get(pid, (0.0, 0.0))
        qty = exit_qty if exit_qty > 0 else entry_qty

        pnl = (exit_price - entry_price) * qty if (entry_price > 0 and exit_price > 0 and qty > 0) else 0.0
        net_pnl += pnl
        if pnl > 0:
            wins += 1
        elif pnl < 0:
            losses += 1

    winrate = (wins / closed_trades) if closed_trades else 0.0

    # upsert daily row
    ddf = read_worksheet_df(ws_daily)
    if ddf is None or ddf.empty:
        ddf = pd.DataFrame(columns=DAILY_HEADERS)

    if "date_ny" not in ddf.columns:
        ddf["date_ny"] = ""

    ddf = ddf[ddf["date_ny"].astype(str) != ny_date].copy()
    ddf = pd.concat([ddf, pd.DataFrame([{
        "date_ny": ny_date,
        "regime": regime,
        "orders_sent": int(orders_sent),
        "entries_filled": int(entries_filled),
        "exits_filled": int(exits_filled),
        "closed_trades": int(closed_trades),
        "wins": int(wins),
        "losses": int(losses),
        "winrate": round(float(winrate), 3),
        "net_pnl_usd": round(float(net_pnl), 2),
    }])], ignore_index=True)

    clear_and_write(ws_daily, DAILY_HEADERS, ddf.reindex(columns=DAILY_HEADERS))

    msg = f"Sync OK. OrdersRows={len(odf)} parents_today={orders_sent} exits_filled={exits_filled} net_pnl={round(net_pnl,2)}"
    logger.info(msg)
    append_rows(ws_logs, [[utc_iso_z(), "alpaca_sync", "INFO", msg]])


if __name__ == "__main__":
    sync_orders_and_daily()
