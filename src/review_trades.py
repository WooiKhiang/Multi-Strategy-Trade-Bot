from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .config import Config
from .logger import get_logger
from .sheets import open_sheet, ensure_worksheet, read_worksheet_df, append_rows
from .data_yf import download_batched


REVIEW_TAB = "Review"
REVIEW_HEADERS = [
    "symbol",
    "entry_time",
    "exit_time",
    "exit_reason",
    "r_multiple",
    "mfe_r",
    "mae_r",
    "regime",
    "notes",
]


def utc_iso_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_iso(dt_str: str) -> Optional[datetime]:
    if not dt_str:
        return None
    s = str(dt_str).strip()
    try:
        # Alpaca-like timestamps: "...Z"
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _to_ny_date(dt_utc: datetime) -> str:
    return dt_utc.astimezone(ZoneInfo("America/New_York")).date().isoformat()


def _safe_float(x, default=0.0) -> float:
    try:
        if x is None:
            return float(default)
        s = str(x).strip()
        if s == "":
            return float(default)
        return float(s)
    except Exception:
        return float(default)


def _as_upper(x) -> str:
    return str(x or "").strip().upper()


@dataclass
class BracketTrade:
    parent_id: str
    symbol: str
    entry_time: datetime
    entry_price: float
    qty: float
    stop_price: float
    tp_price: float
    exit_time: datetime
    exit_price: float
    exit_reason: str  # "TP" or "SL"


def _load_market_regime_map(ss) -> Dict[str, str]:
    """
    Market tab expected to contain at least: date, regime
    We'll map NY date string -> regime.
    """
    ws_market = ensure_worksheet(ss, "Market", ["date", "regime"])
    df = read_worksheet_df(ws_market)
    if df is None or df.empty:
        return {}

    # find columns robustly
    date_col = None
    regime_col = None
    for c in df.columns:
        lc = c.strip().lower()
        if lc in ("date", "date_ny", "day"):
            date_col = c
        if lc == "regime":
            regime_col = c

    if date_col is None:
        date_col = df.columns[0]
    if regime_col is None:
        # best guess: any column containing 'regime'
        for c in df.columns:
            if "regime" in c.strip().lower():
                regime_col = c
                break

    if regime_col is None:
        return {}

    out: Dict[str, str] = {}
    for _, r in df.iterrows():
        d = str(r.get(date_col, "")).strip()
        reg = str(r.get(regime_col, "")).strip().upper()
        if d:
            out[d] = reg if reg else "NEUTRAL"
    return out


def _index_to_utc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df index is tz-aware UTC for slicing.
    """
    if df is None or df.empty:
        return df
    idx = pd.to_datetime(df.index, errors="coerce")
    idx = idx[~idx.isna()]
    df = df.loc[idx].copy()

    # If tz-naive, assume UTC; if tz-aware, convert to UTC
    if getattr(idx, "tz", None) is None:
        df.index = pd.to_datetime(df.index).tz_localize("UTC")
    else:
        df.index = pd.to_datetime(df.index).tz_convert("UTC")
    return df


def _mfe_mae_r(df_5m: pd.DataFrame, entry: float, stop: float) -> Tuple[Optional[float], Optional[float]]:
    """
    mfe_r = max favorable excursion in R (using High)
    mae_r = max adverse excursion in R (using Low)
    R denominator = entry - stop (must be >0)
    """
    denom = entry - stop
    if denom <= 0:
        return None, None
    if df_5m is None or df_5m.empty:
        return None, None
    if "High" not in df_5m.columns or "Low" not in df_5m.columns:
        return None, None

    hi = float(df_5m["High"].max())
    lo = float(df_5m["Low"].min())
    mfe = (hi - entry) / denom
    mae = (lo - entry) / denom
    return float(mfe), float(mae)


def _extract_brackets_from_orders(orders_df: pd.DataFrame, logger) -> List[BracketTrade]:
    """
    Orders tab contains both parent + legs.
    We identify each bracket by parent BUY row with client_order_id "VG-*",
    then attach legs by parent_order_id.
    """
    if orders_df is None or orders_df.empty:
        return []

    df = orders_df.copy()

    # normalize key columns
    for col in ["id", "parent_order_id", "client_order_id", "symbol", "side", "type", "status"]:
        if col not in df.columns:
            df[col] = ""

    df["side"] = df["side"].astype(str).str.lower().str.strip()
    df["type"] = df["type"].astype(str).str.lower().str.strip()
    df["status"] = df["status"].astype(str).str.lower().str.strip()
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df["client_order_id"] = df["client_order_id"].astype(str).str.strip()

    # parent = buy filled with VG- prefix
    parents = df[
        (df["side"] == "buy")
        & (df["status"] == "filled")
        & (df["client_order_id"].str.startswith("VG-") | df["client_order_id"].str.startswith("VG_"))
    ].copy()

    if parents.empty:
        return []

    # build map parent_id -> legs
    legs = df[(df["parent_order_id"].astype(str).str.strip() != "")].copy()
    legs_map: Dict[str, pd.DataFrame] = {}
    for pid, g in legs.groupby(legs["parent_order_id"].astype(str).str.strip()):
        legs_map[str(pid)] = g

    out: List[BracketTrade] = []

    for _, p in parents.iterrows():
        parent_id = str(p.get("id", "")).strip()
        if not parent_id:
            continue

        sym = str(p.get("symbol", "")).strip().upper()
        entry_time = _parse_iso(str(p.get("filled_at", "") or p.get("submitted_at", "")))
        entry_price = _safe_float(p.get("filled_avg_price", 0.0), 0.0)
        qty = _safe_float(p.get("filled_qty", p.get("qty", 0.0)), 0.0)

        if not sym or not entry_time or entry_price <= 0 or qty <= 0:
            continue

        g = legs_map.get(parent_id)
        if g is None or g.empty:
            continue

        # find TP and SL legs (not necessarily filled)
        tp_leg = g[(g["side"] == "sell") & (g["type"] == "limit")].head(1)
        sl_leg = g[(g["side"] == "sell") & (g["type"] == "stop")].head(1)

        tp_price = _safe_float(tp_leg.iloc[0].get("limit_price", ""), 0.0) if not tp_leg.empty else 0.0
        stop_price = _safe_float(sl_leg.iloc[0].get("stop_price", ""), 0.0) if not sl_leg.empty else 0.0

        # find exit: sell filled (either TP limit filled OR SL stop filled)
        exits = g[(g["side"] == "sell") & (g["status"] == "filled")].copy()
        if exits.empty:
            continue

        # choose latest filled exit if somehow multiple
        exits["filled_at_dt"] = exits["filled_at"].apply(lambda x: _parse_iso(str(x)))
        exits = exits[~exits["filled_at_dt"].isna()]
        if exits.empty:
            continue
        exits = exits.sort_values("filled_at_dt")
        e = exits.iloc[-1]

        exit_time = _parse_iso(str(e.get("filled_at", "")))
        exit_price = _safe_float(e.get("filled_avg_price", 0.0), 0.0)
        exit_type = str(e.get("type", "")).lower().strip()

        if not exit_time or exit_price <= 0:
            continue

        exit_reason = "SL" if exit_type == "stop" else "TP"  # bracket exits are stop or limit
        if exit_type == "limit":
            exit_reason = "TP"
        elif exit_type == "stop":
            exit_reason = "SL"

        # if stop not found, approximate using parent stop_loss_price (some rows may carry it)
        if stop_price <= 0:
            stop_price = _safe_float(p.get("stop_loss_price", 0.0), 0.0)
        if tp_price <= 0:
            tp_price = _safe_float(p.get("take_profit_price", 0.0), 0.0)

        if stop_price <= 0:
            logger.warning(f"{sym} parent={parent_id}: missing stop_price; skip review")
            continue

        out.append(
            BracketTrade(
                parent_id=parent_id,
                symbol=sym,
                entry_time=entry_time,
                entry_price=entry_price,
                qty=qty,
                stop_price=stop_price,
                tp_price=tp_price,
                exit_time=exit_time,
                exit_price=exit_price,
                exit_reason=exit_reason,
            )
        )

    return out


def build_review_rows_for_today() -> None:
    cfg = Config()
    logger = get_logger("review_trades", cfg.log_level)

    if not cfg.gsheet_id:
        raise RuntimeError("GSHEET_ID missing in env/.env")

    ss = open_sheet(cfg.gsheet_id, cfg.google_service_account_json)

    ws_orders = ensure_worksheet(ss, "Orders", [])  # we won't enforce headers here
    ws_review = ensure_worksheet(ss, REVIEW_TAB, REVIEW_HEADERS)
    ws_logs = ensure_worksheet(ss, "Logs", ["timestamp_utc", "component", "level", "message"])

    orders_df = read_worksheet_df(ws_orders)
    if orders_df is None or orders_df.empty:
        msg = "Orders sheet empty; nothing to review."
        logger.info(msg)
        append_rows(ws_logs, [[utc_iso_z(), "review_trades", "INFO", msg]])
        return

    # Existing review keys (to avoid duplicates)
    review_df = read_worksheet_df(ws_review)
    existing_parent_ids = set()
    if review_df is not None and not review_df.empty:
        if "notes" in review_df.columns:
            for v in review_df["notes"].astype(str).tolist():
                v = v.strip()
                if v.startswith("parent_id="):
                    existing_parent_ids.add(v.replace("parent_id=", "").strip())

    market_map = _load_market_regime_map(ss)

    trades = _extract_brackets_from_orders(orders_df, logger)
    if not trades:
        msg = "No closed bracket trades detected in Orders."
        logger.info(msg)
        append_rows(ws_logs, [[utc_iso_z(), "review_trades", "INFO", msg]])
        return

    # Only review trades that exited today (NY date)
    ny_today = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    trades_today = [t for t in trades if _to_ny_date(t.exit_time) == ny_today]

    if not trades_today:
        msg = f"No closed trades today (NY={ny_today})."
        logger.info(msg)
        append_rows(ws_logs, [[utc_iso_z(), "review_trades", "INFO", msg]])
        return

    new_trades = [t for t in trades_today if t.parent_id not in existing_parent_ids]
    if not new_trades:
        msg = "All closed trades already present in Review."
        logger.info(msg)
        append_rows(ws_logs, [[utc_iso_z(), "review_trades", "INFO", msg]])
        return

    rows_to_append: List[List] = []

    # Download 5m data per symbol (small count, stable)
    for t in new_trades:
        # regime by entry NY date
        entry_ny_date = _to_ny_date(t.entry_time)
        regime = market_map.get(entry_ny_date, "NEUTRAL")

        risk_per_share = t.entry_price - t.stop_price
        if risk_per_share <= 0:
            r_mult = ""
            mfe_r = ""
            mae_r = ""
        else:
            r_mult = (t.exit_price - t.entry_price) / risk_per_share

            # yfinance 5m window: download last 5d and slice
            dmap = download_batched(
                [t.symbol],
                interval="5m",
                period="5d",
                batch_size=1,
                sleep_sec=float(os.getenv("YF_SLEEP_BETWEEN_BATCH_SEC", "1")),
                logger=logger,
                threads=False if hasattr(cfg, "threads") else False,
            )
            df5 = dmap.get(t.symbol)
            mfe_r = ""
            mae_r = ""

            if df5 is not None and not df5.empty:
                df5 = _index_to_utc(df5)

                start = t.entry_time
                end = t.exit_time
                # slice inclusive
                sliced = df5[(df5.index >= start) & (df5.index <= end)]
                mfe, mae = _mfe_mae_r(sliced, t.entry_price, t.stop_price)
                if mfe is not None:
                    mfe_r = round(float(mfe), 4)
                if mae is not None:
                    mae_r = round(float(mae), 4)

            r_mult = round(float(r_mult), 4)

        rows_to_append.append(
            [
                t.symbol,
                t.entry_time.isoformat().replace("+00:00", "Z"),
                t.exit_time.isoformat().replace("+00:00", "Z"),
                t.exit_reason,
                r_mult,
                mfe_r,
                mae_r,
                regime,
                f"parent_id={t.parent_id}",
            ]
        )

        # gentle pacing if many symbols
        time.sleep(0.2)

    append_rows(ws_review, rows_to_append)

    msg = f"Review appended {len(rows_to_append)} row(s) for NY={ny_today}."
    logger.info(msg)
    append_rows(ws_logs, [[utc_iso_z(), "review_trades", "INFO", msg]])


if __name__ == "__main__":
    build_review_rows_for_today()
