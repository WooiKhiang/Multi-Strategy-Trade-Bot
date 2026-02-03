from __future__ import annotations

from datetime import datetime, timezone
import math
import pandas as pd

from .config import Config
from .logger import get_logger
from .sheets import open_sheet, ensure_worksheet, read_worksheet_df, clear_and_write, append_rows


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

CONTROL_HEADERS = ["key", "value"]
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
LOG_HEADERS = ["timestamp_utc", "component", "level", "message"]


# ----------------------------
# Time helpers
# ----------------------------
def utc_iso_z() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


# ----------------------------
# Safe conversion helpers
# ----------------------------
def _safe_float(x, default=0.0):
    """
    Returns float(default) on failure.
    If default is None, returns None on failure.
    """
    try:
        if x is None:
            return None if default is None else float(default)
        s = str(x).strip()
        if s == "" or s.lower() in ("nan", "none"):
            return None if default is None else float(default)
        return float(s)
    except Exception:
        return None if default is None else float(default)


def _fmt_float(x, nd=4) -> str:
    """Format as string for Google Sheets (prevents Arrow dtype crash)."""
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return ""


def _fmt_int(x) -> str:
    try:
        return str(int(x))
    except Exception:
        return ""


# ----------------------------
# Control / regime
# ----------------------------
def _read_control_kv(ss) -> dict:
    ws_control = ensure_worksheet(ss, "Control", CONTROL_HEADERS)
    df = read_worksheet_df(ws_control)
    if df is None or df.empty:
        return {}

    key_col = None
    val_col = None
    for c in df.columns:
        cl = c.strip().lower()
        if cl in ("key", "name", "param"):
            key_col = c
        if cl in ("value", "val"):
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


def _read_latest_market_regime(ss) -> str:
    ws = ensure_worksheet(
        ss,
        "Market",
        ["date", "spy_close", "spy_ema20", "spy_ema50", "vix", "regime", "confidence"],
    )
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

    val = str(df.iloc[-1][col]).strip().upper()
    return val if val in ("BULL", "NEUTRAL", "DEFENSIVE") else "NEUTRAL"


def _knobs(cfg: Config, control: dict) -> dict:
    def f(key: str, default: float) -> float:
        v = _safe_float(control.get(key, default), default)
        return float(v) if v is not None else float(default)

    def i(key: str, default: int) -> int:
        try:
            return int(float(str(control.get(key, default)).strip()))
        except Exception:
            return int(default)

    return {
        "strategy_capital_usd": f("strategy_capital_usd", 5000.0),
        "max_trade_budget_usd": f("max_trade_budget_usd", 2000.0),
        "max_concurrent_trades": i("max_concurrent_trades", 2),
        "risk_per_trade_usd": f("risk_per_trade_usd", getattr(cfg, "risk_per_trade_usd", 25.0)),
        "take_profit_r_mult": f("take_profit_r_mult", getattr(cfg, "take_profit_r_mult", 2.0)),
        "est_txn_cost_rate": f("est_txn_cost_rate", 0.0005),
        "min_expected_net_r": f("min_expected_net_r", 1.2),
        "auto_select": str(control.get("auto_select", "TRUE")).strip().upper() in ("TRUE", "YES", "1", "Y"),
    }


def _apply_regime_overlay(k: dict, control: dict, regime: str) -> dict:
    r = str(regime or "NEUTRAL").strip().lower()

    def f(key: str, default: float) -> float:
        v = _safe_float(control.get(key, default), default)
        return float(v) if v is not None else float(default)

    def i(key: str, default: int) -> int:
        try:
            return int(float(control.get(key, default)))
        except Exception:
            return int(default)

    # NOTE: fixed bug here (you had f(f"..."))
    k["risk_per_trade_usd"] = f(f"risk_per_trade_usd_{r}", k["risk_per_trade_usd"])
    k["max_concurrent_trades"] = i(f"max_concurrent_trades_{r}", k["max_concurrent_trades"])
    k["take_profit_r_mult"] = f(f"take_profit_r_mult_{r}", k["take_profit_r_mult"])
    return k


def _read_latest_alpaca_cash(ss) -> float | None:
    ws_capital = ensure_worksheet(ss, "Capital", CAPITAL_HEADERS)
    df = read_worksheet_df(ws_capital)
    if df is None or df.empty:
        return None
    if "cash" not in df.columns:
        return None

    for i in range(len(df) - 1, -1, -1):
        cash = _safe_float(df.iloc[i].get("cash", ""), default=None)
        if cash is not None and cash > 0:
            return float(cash)
    return None


# ----------------------------
# Main
# ----------------------------
def select_trades():
    cfg = Config()
    logger = get_logger("select_trades", cfg.log_level)

    if not cfg.gsheet_id:
        raise RuntimeError("GSHEET_ID missing")

    ss = open_sheet(cfg.gsheet_id, cfg.google_service_account_json)

    ws_trades = ensure_worksheet(ss, "Trades", TRADE_HEADERS)
    ws_logs = ensure_worksheet(ss, "Logs", LOG_HEADERS)

    control = _read_control_kv(ss)
    k = _knobs(cfg, control)
    regime = _read_latest_market_regime(ss)
    k = _apply_regime_overlay(k, control, regime)
    append_rows(ws_logs, [[utc_iso_z(), "select_trades", "INFO", f"Regime overlay applied: {regime}"]])

    alpaca_cash = _read_latest_alpaca_cash(ss)
    strategy_cap = float(k["strategy_capital_usd"])
    effective_capital = min(strategy_cap, float(alpaca_cash)) if alpaca_cash is not None else strategy_cap

    df = read_worksheet_df(ws_trades)
    if df is None or df.empty:
        msg = "Trades sheet is empty."
        logger.info(msg)
        append_rows(ws_logs, [[utc_iso_z(), "select_trades", "INFO", msg]])
        return

    # Ensure all columns exist
    for c in TRADE_HEADERS:
        if c not in df.columns:
            df[c] = ""

    df["status"] = df["status"].astype(str).str.upper().str.strip()
    candidates = df[df["status"] == "NEW"].copy()

    if candidates.empty:
        msg = "No NEW trades to select."
        logger.info(msg)
        append_rows(ws_logs, [[utc_iso_z(), "select_trades", "INFO", msg]])
        df = df[TRADE_HEADERS]
        clear_and_write(ws_trades, TRADE_HEADERS, df)
        return

    scored_rows = []

    for idx, r in candidates.iterrows():
        entry = _safe_float(r.get("entry", 0), 0.0)
        stop = _safe_float(r.get("stop_loss", 0), 0.0)
        tp = _safe_float(r.get("take_profit", 0), 0.0)

        if entry <= 0 or stop <= 0 or tp <= 0:
            continue

        risk_per_share = entry - stop
        if risk_per_share <= 0:
            continue

        shares_by_risk = math.floor(float(k["risk_per_trade_usd"]) / risk_per_share)
        if shares_by_risk <= 0:
            continue

        shares_by_budget = math.floor(float(k["max_trade_budget_usd"]) / entry)
        if shares_by_budget <= 0:
            continue

        order_qty = int(min(shares_by_risk, shares_by_budget))
        if order_qty <= 0:
            continue

        entry_notional = entry * order_qty
        tp_notional = tp * order_qty

        cost_est = float(k["est_txn_cost_rate"]) * (entry_notional + tp_notional)
        denom = risk_per_share * order_qty
        cost_in_r = cost_est / denom if denom > 0 else 999.0

        expected_net_r = float(k["take_profit_r_mult"]) - cost_in_r

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

    # Reset output columns deterministically (including risk_per_share + notional)
    for col in ["selected", "priority_rank", "order_qty", "risk_per_share", "notional", "cost_est", "cost_in_r", "expected_net_r", "score"]:
        if col == "selected":
            df[col] = "FALSE"
        else:
            df[col] = ""

    if not scored_rows:
        msg = "No trades passed sizing/budget filters."
        logger.info(msg)
        append_rows(ws_logs, [[utc_iso_z(), "select_trades", "INFO", msg]])
        df = df[TRADE_HEADERS]
        clear_and_write(ws_trades, TRADE_HEADERS, df)
        return

    scored_all = pd.DataFrame(scored_rows)
    scored_all = scored_all.sort_values(["score", "expected_net_r"], ascending=[False, False]).reset_index(drop=True)
    scored_all["priority_rank"] = scored_all.index + 1

    scored_sel = scored_all[scored_all["expected_net_r"] >= float(k["min_expected_net_r"])].copy()

    selected_idxs: list[int] = []
    remaining = float(effective_capital)

    if not scored_sel.empty and bool(k["auto_select"]):
        for _, sr in scored_sel.iterrows():
            if len(selected_idxs) >= int(k["max_concurrent_trades"]):
                break
            if float(sr["notional"]) <= remaining:
                selected_idxs.append(int(sr["idx"]))
                remaining -= float(sr["notional"])

    # Write scored values back as STRINGS (fixes Arrow dtype crash)
    for _, sr in scored_all.iterrows():
        irow = int(sr["idx"])
        df.at[irow, "risk_per_share"] = _fmt_float(sr["risk_per_share"], 4)
        df.at[irow, "order_qty"] = _fmt_int(sr["order_qty"])
        df.at[irow, "notional"] = _fmt_float(sr["notional"], 2)
        df.at[irow, "cost_est"] = _fmt_float(sr["cost_est"], 2)
        df.at[irow, "cost_in_r"] = _fmt_float(sr["cost_in_r"], 4)
        df.at[irow, "expected_net_r"] = _fmt_float(sr["expected_net_r"], 4)
        df.at[irow, "score"] = _fmt_float(sr["score"], 4)
        df.at[irow, "priority_rank"] = _fmt_int(sr["priority_rank"])

    for irow in selected_idxs:
        df.at[irow, "selected"] = "TRUE"

    msg = (
        f"strategy_cap={round(strategy_cap,2)} "
        f"alpaca_cash={(round(alpaca_cash,2) if alpaca_cash is not None else 'NA')} "
        f"effective_cap={round(effective_capital,2)} "
        f"Scored={len(scored_all)} Selected={len(selected_idxs)} Remaining={round(remaining,2)}"
    )
    logger.info(msg)
    append_rows(ws_logs, [[utc_iso_z(), "select_trades", "INFO", msg]])

    df = df[TRADE_HEADERS]
    clear_and_write(ws_trades, TRADE_HEADERS, df)


if __name__ == "__main__":
    select_trades()
