from __future__ import annotations

from datetime import datetime, timezone
import math
import pandas as pd

from .config import Config
from .logger import get_logger
from .sheets import open_sheet, ensure_worksheet, read_worksheet_df, clear_and_write


TRADE_HEADERS = [
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


def utc_iso_z() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _read_control_kv(ss) -> dict:
    ws_control = ensure_worksheet(ss, "Control", ["key", "value"])
    df = read_worksheet_df(ws_control)
    if df is None or df.empty:
        return {}

    key_col = None
    val_col = None
    for c in df.columns:
        if c.lower() in ("key", "name", "param"):
            key_col = c
        if c.lower() in ("value", "val"):
            val_col = c
    if key_col is None:
        key_col = df.columns[0]
    if val_col is None:
        val_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

    out = {}
    for _, r in df.iterrows():
        k = str(r.get(key_col, "")).strip()
        v = str(r.get(val_col, "")).strip()
        if k:
            out[k] = v
    return out


def _knobs(cfg: Config, control: dict) -> dict:
    def f(key: str, default: float) -> float:
        return _safe_float(control.get(key, default), default)

    def i(key: str, default: int) -> int:
        try:
            return int(float(control.get(key, default)))
        except Exception:
            return int(default)

    return {
        "total_capital_usd": f("total_capital_usd", 5000.0),
        "max_trade_budget_usd": f("max_trade_budget_usd", 2000.0),
        "max_concurrent_trades": i("max_concurrent_trades", 2),
        "risk_per_trade_usd": f("risk_per_trade_usd", getattr(cfg, "risk_per_trade_usd", 25.0)),
        "take_profit_r_mult": f("take_profit_r_mult", getattr(cfg, "take_profit_r_mult", 2.0)),
        "est_txn_cost_rate": f("est_txn_cost_rate", 0.005),
        "min_expected_net_r": f("min_expected_net_r", 1.2),
        "auto_select": str(control.get("auto_select", "TRUE")).strip().upper() in ("TRUE", "YES", "1", "Y"),
        "mode": str(control.get("mode", "PAPER")).strip().upper(),
    }


def select_trades():
    cfg = Config()
    logger = get_logger("select_trades", cfg.log_level)

    if not cfg.gsheet_id:
        raise RuntimeError("GSHEET_ID missing")

    ss = open_sheet(cfg.gsheet_id, cfg.google_service_account_json)
    ws_trades = ensure_worksheet(ss, "Trades", TRADE_HEADERS)
    ws_logs = ensure_worksheet(ss, "Logs", ["timestamp_utc", "component", "level", "message"])

    control = _read_control_kv(ss)
    k = _knobs(cfg, control)

    df = read_worksheet_df(ws_trades)
    if df is None or df.empty:
        msg = "Trades sheet is empty."
        logger.info(msg)
        ws_logs.append_rows([[utc_iso_z(), "select_trades", "INFO", msg]])
        return

    # Ensure all columns exist
    for c in TRADE_HEADERS:
        if c not in df.columns:
            df[c] = ""

    # Only consider NEW trades (not already executed)
    df["status"] = df["status"].astype(str).str.upper().str.strip()
    candidates = df[df["status"] == "NEW"].copy()

    if candidates.empty:
        msg = "No NEW trades to select."
        logger.info(msg)
        ws_logs.append_rows([[utc_iso_z(), "select_trades", "INFO", msg]])
        # Still write back normalized columns
        df = df[TRADE_HEADERS]
        clear_and_write(ws_trades, TRADE_HEADERS, df)
        return

    # Compute sizing fresh from Control (this is the control center)
    # and compute cost-aware net R and score.
    scored_rows = []

    for idx, r in candidates.iterrows():
        entry = _safe_float(r["entry"])
        stop = _safe_float(r["stop_loss"])
        tp = _safe_float(r["take_profit"])

        symbol = str(r["symbol"]).strip().upper()
        ref_time = str(r["ref_time"]).strip()

        if entry <= 0 or stop <= 0 or tp <= 0:
            continue

        risk_per_share = entry - stop
        if risk_per_share <= 0:
            continue

        # baseline shares by risk
        shares_by_risk = math.floor(k["risk_per_trade_usd"] / risk_per_share)
        if shares_by_risk <= 0:
            continue

        # budget cap (max notional per trade)
        shares_by_budget = math.floor(k["max_trade_budget_usd"] / entry)
        if shares_by_budget <= 0:
            continue

        order_qty = int(max(0, min(shares_by_risk, shares_by_budget)))
        if order_qty <= 0:
            continue

        entry_notional = entry * order_qty
        tp_notional = tp * order_qty

        # Cost estimate: round-trip cost rate applied to entry + exit
        cost_est = k["est_txn_cost_rate"] * (entry_notional + tp_notional)

        # Convert cost into R units
        denom = risk_per_share * order_qty
        cost_in_r = cost_est / denom if denom > 0 else 999.0

        # Net expected R after cost (using target multiple from Control)
        expected_net_r = k["take_profit_r_mult"] - cost_in_r

        # Risk percentage as a mild stabilizer (prefer more capital-efficient trades)
        risk_pct = risk_per_share / entry
        score = expected_net_r / max(risk_pct, 1e-9)

        scored_rows.append({
            "idx": idx,
            "symbol": symbol,
            "ref_time": ref_time,
            "risk_per_share": risk_per_share,
            "order_qty": order_qty,
            "notional": entry_notional,
            "cost_est": cost_est,
            "cost_in_r": cost_in_r,
            "expected_net_r": expected_net_r,
            "score": score,
        })

    if not scored_rows:
        msg = "No trades passed sizing/budget filters."
        logger.info(msg)
        ws_logs.append_rows([[utc_iso_z(), "select_trades", "INFO", msg]])
        df = df[TRADE_HEADERS]
        clear_and_write(ws_trades, TRADE_HEADERS, df)
        return

    scored = pd.DataFrame(scored_rows)

    # Filter by minimum net R
    scored = scored[scored["expected_net_r"] >= k["min_expected_net_r"]].copy()
    if scored.empty:
        msg = "All trades filtered out by min_expected_net_r."
        logger.info(msg)
        ws_logs.append_rows([[utc_iso_z(), "select_trades", "INFO", msg]])
        df = df[TRADE_HEADERS]
        clear_and_write(ws_trades, TRADE_HEADERS, df)
        return

    # Rank by score desc
    scored = scored.sort_values(["score", "expected_net_r"], ascending=[False, False]).reset_index(drop=True)
    scored["priority_rank"] = scored.index + 1

    # Greedy pick within total capital and max concurrent trades
    selected = []
    remaining = float(k["total_capital_usd"])
    for _, sr in scored.iterrows():
        if len(selected) >= int(k["max_concurrent_trades"]):
            break
        if sr["notional"] <= remaining:
            selected.append(int(sr["idx"]))
            remaining -= float(sr["notional"])

    # Apply results back to df
    df["selected"] = "FALSE"
    df["priority_rank"] = ""
    df["order_qty"] = ""
    df["cost_est"] = ""
    df["cost_in_r"] = ""
    df["expected_net_r"] = ""
    df["score"] = ""

    # Write scored values for NEW trades
    for _, sr in scored.iterrows():
        irow = int(sr["idx"])
        df.at[irow, "risk_per_share"] = round(float(sr["risk_per_share"]), 4)
        df.at[irow, "order_qty"] = int(sr["order_qty"])
        df.at[irow, "notional"] = round(float(sr["notional"]), 2)
        df.at[irow, "cost_est"] = round(float(sr["cost_est"]), 2)
        df.at[irow, "cost_in_r"] = round(float(sr["cost_in_r"]), 4)
        df.at[irow, "expected_net_r"] = round(float(sr["expected_net_r"]), 4)
        df.at[irow, "score"] = round(float(sr["score"]), 4)
        df.at[irow, "priority_rank"] = int(sr["priority_rank"])

    if k["auto_select"]:
        for idx in selected:
            df.at[idx, "selected"] = "TRUE"

    msg = f"Scored={len(scored)} Selected={len(selected)} RemainingCapital={round(remaining, 2)}"
    logger.info(msg)
    ws_logs.append_rows([[utc_iso_z(), "select_trades", "INFO", msg]])

    # Persist back to sheet (overwrite with updated data)
    df["timestamp_utc"] = df["timestamp_utc"].astype(str)
    df = df[TRADE_HEADERS]
    clear_and_write(ws_trades, TRADE_HEADERS, df)


if __name__ == "__main__":
    select_trades()
