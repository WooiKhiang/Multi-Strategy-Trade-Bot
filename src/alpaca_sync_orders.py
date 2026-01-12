from __future__ import annotations

import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import requests
import pandas as pd

from .config import Config
from .logger import get_logger
from .sheets import (
    open_sheet,
    ensure_worksheet,
    read_worksheet_df,
    clear_and_write,
    append_rows,
)

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

# We will update Trades.status + note + filled_avg_price.
# (We won't change your Trades schema beyond what you already have.)


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
        raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_API_SECRET in env")
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


def _safe_float(x, default=0.0) -> float:
    try:
        if x is None:
            return float(default)
        s = str(x).strip()
        if s == "" or s.lower() == "nan":
            return float(default)
        return float(s)
    except Exception:
        return float(default)


def _safe_int(x, default=0) -> int:
    try:
        return int(float(str(x).strip()))
    except Exception:
        return int(default)


def _ny_now() -> datetime:
    return datetime.now(ZoneInfo("America/New_York"))


def _is_market_window() -> bool:
    """
    We sync frequently on weekdays, but only do real work during NY market window
    to reduce noise & rate usage.

    Window: 09:25–16:20 NY time (covers open/close + bracket fills)
    """
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
    # find regime column
    regime_col = None
    for c in df.columns:
        if c.strip().lower() == "regime":
            regime_col = c
            break
    if not regime_col:
        return "NEUTRAL"
    v = str(df.iloc[-1][regime_col]).strip().upper()
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


def _order_is_entry_parent(o: dict) -> bool:
    # Parent entry in bracket usually has order_class="bracket" and side="buy"
    return (o.get("order_class") == "bracket") and (o.get("side") == "buy")


def _order_is_exit_leg(o: dict) -> bool:
    # Child legs are sells with parent_order_id set
    return (o.get("side") == "sell") and bool(o.get("parent_order_id"))


def _exit_reason(o: dict) -> str:
    # Take profit legs are typically type="limit" with limit_price set
    # Stop loss legs are typically type="stop" with stop_price set
    t = (o.get("type") or "").lower()
    if t == "limit" and o.get("limit_price"):
        return "TP"
    if t in ("stop", "stop_limit") and o.get("stop_price"):
        return "SL"
    # fallback if ambiguous
    if o.get("limit_price"):
        return "TP"
    if o.get("stop_price"):
        return "SL"
    return "EXIT"


def sync_orders_and_update_sheets():
    cfg = Config()
    logger = get_logger("alpaca_sync", cfg.log_level)

    if not cfg.gsheet_id:
        raise RuntimeError("GSHEET_ID missing")

    ss = open_sheet(cfg.gsheet_id, cfg.google_service_account_json)
    ws_orders = ensure_worksheet(ss, "Orders", ORDERS_HEADERS)
    ws_daily = ensure_worksheet(ss, "Daily", DAILY_HEADERS)
    ws_logs = ensure_worksheet(ss, "Logs", LOG_HEADERS)

    if not _is_market_window():
        msg = "Outside NY market sync window; skip."
        logger.info(msg)
        append_rows(ws_logs, [[utc_iso_z(), "alpaca_sync", "INFO", msg]])
        return

    # Prove auth + endpoint
    acct = _alpaca_get("/v2/account")
    base = _alpaca_base_url()
    msg = f"Alpaca OK base={base} status={acct.get('status')} cash={acct.get('cash')}"
    logger.info(msg)
    append_rows(ws_logs, [[utc_iso_z(), "alpaca_sync", "INFO", msg]])

    # Pull recent orders (nested includes bracket legs)
    # status=all includes filled/canceled/etc
    limit = int(float(_env("ALPACA_SYNC_LIMIT", "200")))
    orders = _alpaca_get(
        "/v2/orders",
        params={
            "status": "all",
            "nested": "true",
            "limit": str(limit),
            "direction": "desc",
        },
    )

    if not isinstance(orders, list):
        raise RuntimeError("Unexpected Alpaca orders response (not a list)")

    # Filter to our bot orders only (client_order_id prefix VG-)
    bot_orders = [o for o in orders if str(o.get("client_order_id", "")).startswith("VG-") or str(o.get("client_order_id", "")).startswith("VG_")]
    # Still keep nested legs even if their client_order_id differs (some legs inherit, some don't)
    # So also include legs of our parent orders:
    parent_ids = {o.get("id") for o in bot_orders if o.get("id")}
    for o in orders:
        if o.get("parent_order_id") in parent_ids and o not in bot_orders:
            bot_orders.append(o)

    # Write Orders tab snapshot
    rows = [_flatten_order(o) for o in bot_orders]
    odf = pd.DataFrame(rows) if rows else pd.DataFrame(columns=ORDERS_HEADERS)
    odf = odf.reindex(columns=ORDERS_HEADERS)
    clear_and_write(ws_orders, ORDERS_HEADERS, odf)

    # Update Trades statuses (light-touch)
    ws_trades = ensure_worksheet(ss, "Trades", ["timestamp_utc", "symbol", "ref_time", "entry", "stop_loss", "take_profit",
                                               "risk_per_share", "r_mult", "shares", "notional", "cost_est", "cost_in_r",
                                               "expected_net_r", "score", "priority_rank", "selected", "order_qty", "status", "note",
                                               "alpaca_order_id", "alpaca_client_order_id", "submitted_at_utc", "filled_avg_price"])
    tdf = read_worksheet_df(ws_trades)
    if tdf is None or tdf.empty:
        tdf = pd.DataFrame(columns=ws_trades.row_values(1) or [])

    # Make sure fields exist
    for c in ["status", "note", "alpaca_order_id", "alpaca_client_order_id", "filled_avg_price"]:
        if c not in tdf.columns:
            tdf[c] = ""

    # Build maps
    by_id = {o.get("id"): o for o in bot_orders if o.get("id")}
    by_cid = {o.get("client_order_id"): o for o in bot_orders if o.get("client_order_id")}

    # For exit detection, group child legs by parent_order_id
    child_by_parent = {}
    for o in bot_orders:
        pid = o.get("parent_order_id")
        if not pid:
            continue
        child_by_parent.setdefault(pid, []).append(o)

    updates = 0
    closed_trades = []
    entries_filled = 0
    exits_filled = 0
    orders_sent = 0

    for i in range(len(tdf)):
        status = str(tdf.at[i, "status"] or "").strip().upper()
        cid = str(tdf.at[i, "alpaca_client_order_id"] or "").strip()
        oid = str(tdf.at[i, "alpaca_order_id"] or "").strip()

        # Only manage trades that were sent / selected
        if not cid and not oid:
            continue

        parent = None
        if oid and oid in by_id:
            parent = by_id[oid]
        elif cid and cid in by_cid:
            parent = by_cid[cid]

        if not parent:
            continue

        p_status = str(parent.get("status", "")).strip().upper()
        orders_sent += 1

        # Update filled avg price on entry when available
        favg = parent.get("filled_avg_price")
        if favg:
            tdf.at[i, "filled_avg_price"] = favg

        # Map parent status to our status (entry lifecycle)
        if p_status in ("ACCEPTED", "NEW", "HELD", "PARTIALLY_FILLED"):
            if status in ("NEW", ""):
                tdf.at[i, "status"] = "SENT"
                updates += 1
        elif p_status == "FILLED":
            # entry filled => OPEN (unless already closed)
            if status not in ("CLOSED_TP", "CLOSED_SL", "CLOSED", "CANCELED", "REJECTED"):
                if status != "OPEN":
                    tdf.at[i, "status"] = "OPEN"
                    updates += 1
                entries_filled += 1
        elif p_status in ("CANCELED", "REJECTED", "EXPIRED"):
            if status not in ("CANCELED", "REJECTED"):
                tdf.at[i, "status"] = p_status
                tdf.at[i, "note"] = (str(tdf.at[i, "note"]) + " | " if str(tdf.at[i, "note"]).strip() else "") + f"ENTRY_{p_status}"
                updates += 1

        # If entry filled, check for exit legs filled
        if str(tdf.at[i, "status"]).strip().upper() == "OPEN":
            parent_id = parent.get("id")
            legs = child_by_parent.get(parent_id, []) if parent_id else []

            # Find a filled SELL leg
            filled_legs = [l for l in legs if str(l.get("status", "")).strip().upper() == "FILLED" and _order_is_exit_leg(l)]
            if filled_legs:
                # If multiple filled (rare), pick last by filled_at
                def _filled_at(x):
                    return str(x.get("filled_at") or "")
                filled_legs = sorted(filled_legs, key=_filled_at)
                leg = filled_legs[-1]

                reason = _exit_reason(leg)
                tdf.at[i, "status"] = "CLOSED_TP" if reason == "TP" else "CLOSED_SL"
                exit_price = _safe_float(leg.get("filled_avg_price"), 0.0)
                exit_qty = _safe_float(leg.get("filled_qty"), 0.0)
                entry_price = _safe_float(parent.get("filled_avg_price"), 0.0)

                # Compute rough PnL if possible (buy then sell)
                pnl = (exit_price - entry_price) * exit_qty if (exit_price and entry_price and exit_qty) else 0.0

                tdf.at[i, "note"] = (str(tdf.at[i, "note"]) + " | " if str(tdf.at[i, "note"]).strip() else "") + f"EXIT_{reason} pnl={round(pnl,2)}"
                updates += 1
                closed_trades.append(pnl)
                exits_filled += 1

    # Write Trades back if changed
    if updates > 0:
        # keep original column order; just write what we have (worksheet helper will align by headers)
        headers = list(tdf.columns)
        clear_and_write(ws_trades, headers, tdf)

    # Daily summary (NY date)
    regime = _read_latest_regime(ss)
    ny_date = _ny_now().date().isoformat()
    closed_count = len(closed_trades)
    wins = sum(1 for x in closed_trades if x > 0)
    losses = sum(1 for x in closed_trades if x < 0)
    winrate = (wins / closed_count) if closed_count else 0.0
    net_pnl = float(sum(closed_trades)) if closed_trades else 0.0

    # Upsert today's daily row (overwrite if exists)
    ddf = read_worksheet_df(ws_daily)
    if ddf is None or ddf.empty:
        ddf = pd.DataFrame(columns=DAILY_HEADERS)

    if "date_ny" not in ddf.columns:
        ddf["date_ny"] = ""

    # remove existing row for today, then append
    ddf = ddf[ddf["date_ny"].astype(str) != ny_date].copy()

    ddf = pd.concat([ddf, pd.DataFrame([{
        "date_ny": ny_date,
        "regime": regime,
        "orders_sent": int(orders_sent),
        "entries_filled": int(entries_filled),
        "exits_filled": int(exits_filled),
        "closed_trades": int(closed_count),
        "wins": int(wins),
        "losses": int(losses),
        "winrate": round(winrate, 3),
        "net_pnl_usd": round(net_pnl, 2),
    }])], ignore_index=True)

    clear_and_write(ws_daily, DAILY_HEADERS, ddf.reindex(columns=DAILY_HEADERS))

    msg = f"Sync done. orders={len(bot_orders)} trades_updates={updates} closed={closed_count} net_pnl={round(net_pnl,2)} regime={regime}"
    logger.info(msg)
    append_rows(ws_logs, [[utc_iso_z(), "alpaca_sync", "INFO", msg]])


if __name__ == "__main__":
    sync_orders_and_update_sheets()
