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
        if x is None:
            return float(default)
        s = str(x).strip()
        if s == "" or s.lower() == "nan":
            return float(default)
        return float(s)
    except Exception:
        return float(default)


def _read_control_kv(ss) -> dict:
    ws_control = ensure_worksheet(ss, "Control", ["key", "value"])
    df = read_worksheet_df(ws_control)
    if df is None or df.empty:
        return {}

    # tolerate column naming
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
            return int(float(str(control.get(key, default)).strip()))
        except Exception:
            return int(default)

    return {
        "total_capital_usd": f("total_capital_usd", 5000.0),
        "max_trade_budget_usd": f("max_trade_budget_usd", 2000.0),
        "max_concurrent_trades": i("max_concurrent_trades", 2),
        "risk_per_trade_usd": f("risk_per_trade_usd", getattr(cfg, "risk_per_trade_usd", 25.0)),
        "take_profit_r_mult": f("take_profit_r_mult", getattr(cfg, "take_profit_r_mult", 2.0)),
        # IMPORTANT: realistic default (0.05% round-trip). You can override in Control.
        "est_txn_cost_rate": f("est_txn_cost_rate", 0.0005),
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

    # Ensure all columns exist in df
    for c in TRADE_HEADERS:
        if c not in df.columns:
            df[c] = ""

    # Normalize status column
    df["status"] = df["status"].astype(str).str.upper().str.strip()
    candidates = df[df["status"] == "NEW"].copy()

    if candidates.empty:
        msg = "No NEW trades to select."
        logger.info(msg)
        ws_logs.append_rows([[utc_iso_z(), "select_trades", "INFO", msg]])
        df = df[TRADE_HEADERS]
        clear_and_write(ws_trades, TRADE_HEADERS, df)
        return

    scored_rows = []

    for idx, r in candidates.iterrows():
        entry = _safe_float(r.get("entry", 0))
        stop = _safe_float(r.get("stop_loss", 0))
        tp = _safe_float(r.get("take_profit", 0))

        if entry <= 0 or stop <= 0 or tp <= 0:
            continue

        risk_per_share = entry - stop
        if risk_per_share <= 0:
            continue

        # sizing: risk-based
        shares_by_risk = math.floor(k["risk_per_trade_usd"] / risk_per_share)
        if shares_by_risk <= 0:
            continue

        # sizing: budget cap
        shares_by_budget = math.floor(k["max_trade_budget_usd"] / entry)
        if shares_by_budget <= 0:
            continue

        order_qty = int(min(shares_by_risk, shares_by_budget))
        if order_qty <= 0:
            continue

        entry_notional = entry * order_qty
        tp_notional = tp * order_qty

        # round-trip cost
        cost_est = k["est_txn_cost_rate"] * (entry_notional + tp_notional)

        denom = risk_per_share * order_qty
        cost_in_r = cost_est / denom if denom > 0 else 999.0

        expected_net_r = k["take_profit_r_mult"] - cost_in_r

        # stabilize by % risk (smaller % risk is better, all else equal)
        risk_pct = risk_per_share / entry
        score = expected_net_r / max(risk_pct, 1e-9)

        scored_rows.append({
            "idx": int(idx),
            "order_qty": int(order_qty),
            "notional": float(entry_notional),
            "risk_per_share": float(risk_per_share),
            "cost_est": float(cost_est),
            "cost_in_r": float(cost_in_r),
            "expected_net_r": float(expected_net_r),
            "score": float(score),
        })

    if not scored_rows:
        msg = "No trades passed sizing/budget filters."
        logger.info(msg)
        ws_logs.append_rows([[utc_iso_z(), "select_trades", "INFO", msg]])
        # Clear scoring columns anyway (so it’s deterministic)
        for col in ["cost_est", "cost_in_r", "expected_net_r", "score", "priority_rank", "selected", "order_qty"]:
            df[col] = "" if col != "selected" else "FALSE"
        df = df[TRADE_HEADERS]
        clear_and_write(ws_trades, TRADE_HEADERS, df)
        return

    scored_all = pd.DataFrame(scored_rows)

    # Rank ALL trades for visibility (even if filtered out later)
    scored_all = scored_all.sort_values(["score", "expected_net_r"], ascending=[False, False]).reset_index(drop=True)
    scored_all["priority_rank"] = scored_all.index + 1

    # Selection set: must pass min_expected_net_r
    scored_sel = scored_all[scored_all["expected_net_r"] >= k["min_expected_net_r"]].copy()

    # Greedy selection within total capital + max concurrent
    selected_idxs = []
    remaining = float(k["total_capital_usd"])

    if not scored_sel.empty and k["auto_select"]:
        for _, sr in scored_sel.iterrows():
            if len(selected_idxs) >= int(k["max_concurrent_trades"]):
                break
            if float(sr["notional"]) <= remaining:
                selected_idxs.append(int(sr["idx"]))
                remaining -= float(sr["notional"])

    # Reset columns for all rows first (deterministic)
    df["selected"] = "FALSE"
    df["priority_rank"] = ""
    df["order_qty"] = ""
    df["cost_est"] = ""
    df["cost_in_r"] = ""
    df["expected_net_r"] = ""
    df["score"] = ""

    # Write computed values back (for NEW trades that were scored)
    for _, sr in scored_all.iterrows():
        irow = int(sr["idx"])
        df.at[irow, "risk_per_share"] = round(float(sr["risk_per_share"]), 4)
        df.at[irow, "order_qty"] = int(sr["order_qty"])
        df.at[irow, "notional"] = round(float(sr["notional"]), 2)
        df.at[irow, "cost_est"] = round(float(sr["cost_est"]), 2)
        df.at[irow, "cost_in_r"] = round(float(sr["cost_in_r"]), 4)
        df.at[irow, "expected_net_r"] = round(float(sr["expected_net_r"]), 4)
        df.at[irow, "score"] = round(float(sr["score"]), 4)
        df.at[irow, "priority_rank"] = int(sr["priority_rank"])

    for irow in selected_idxs:
        df.at[irow, "selected"] = "TRUE"

    msg = f"Scored={len(scored_all)} Selected={len(selected_idxs)} RemainingCapital={round(remaining, 2)}"
    logger.info(msg)
    ws_logs.append_rows([[utc_iso_z(), "select_trades", "INFO", msg]])

    # Persist back
    df = df[TRADE_HEADERS]
    clear_and_write(ws_trades, TRADE_HEADERS, df)


if __name__ == "__main__":
    select_trades()
