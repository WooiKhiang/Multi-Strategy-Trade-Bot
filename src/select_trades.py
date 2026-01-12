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


def utc_iso_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def _read_latest_market_regime(ss) -> str:
    ws = ensure_worksheet(ss, "Market", ["date", "spy_close", "spy_ema20", "spy_ema50", "vix", "regime", "confidence"])
    df = read_worksheet_df(ws)
    if df is None or df.empty:
        return "NEUTRAL"
    # take last non-empty regime
    col = None
    for c in df.columns:
        if c.strip().lower() == "regime":
            col = c
            break
    if not col:
        return "NEUTRAL"
    val = str(df.iloc[-1][col]).strip().upper()
    return val if val in ("BULL", "NEUTRAL", "DEFENSIVE") else "NEUTRAL"

def _apply_regime_overlay(k: dict, control: dict, regime: str) -> dict:
    r = regime.lower()

    def f(key: str, default: float) -> float:
        try:
            return float(control.get(key, default))
        except Exception:
            return float(default)

    def i(key: str, default: int) -> int:
        try:
            return int(float(control.get(key, default)))
        except Exception:
            return int(default)

    # override knobs using Control tab if present
    k["risk_per_trade_usd"] = f(f"risk_per_trade_usd_{r}", k["risk_per_trade_usd"])
    k["max_concurrent_trades"] = i(f"max_concurrent_trades_{r}", k["max_concurrent_trades"])
    k["take_profit_r_mult"] = f(f"take_profit_r_mult_{r}", k["take_profit_r_mult"])
    return k

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
    ws_control = ensure_worksheet(ss, "Control", CONTROL_HEADERS)
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


def _read_latest_alpaca_cash(ss, logger) -> float | None:
    """
    Reads latest row of Capital tab and returns cash as float.
    Returns None if not available.
    """
    ws_capital = ensure_worksheet(ss, "Capital", CAPITAL_HEADERS)
    df = read_worksheet_df(ws_capital)
    if df is None or df.empty:
        return None

    # best-effort: pick last non-empty cash value
    # (Capital tab is append-only)
    if "cash" not in df.columns:
        return None

    # Take last row where cash is parseable
    for i in range(len(df) - 1, -1, -1):
        cash = _safe_float(df.iloc[i].get("cash", ""), default=None)
        if cash is not None and cash > 0:
            return float(cash)

    return None


def _knobs(cfg: Config, control: dict) -> dict:
    def f(key: str, default: float) -> float:
        return _safe_float(control.get(key, default), default)

    def i(key: str, default: int) -> int:
        try:
            return int(float(str(control.get(key, default)).strip()))
        except Exception:
            return int(default)

    return {
        # Your “strategy constraint” capital
        "strategy_capital_usd": f("strategy_capital_usd", 5000.0),

        # Per-trade budget cap
        "max_trade_budget_usd": f("max_trade_budget_usd", 2000.0),

        # Concurrency cap
        "max_concurrent_trades": i("max_concurrent_trades", 2),

        # Risk model inputs
        "risk_per_trade_usd": f("risk_per_trade_usd", getattr(cfg, "risk_per_trade_usd", 25.0)),
        "take_profit_r_mult": f("take_profit_r_mult", getattr(cfg, "take_profit_r_mult", 2.0)),

        # Cost model (override in Control if you want)
        "est_txn_cost_rate": f("est_txn_cost_rate", 0.0005),

        # Selection filter
        "min_expected_net_r": f("min_expected_net_r", 1.2),

        # auto selection toggle
        "auto_select": str(control.get("auto_select", "TRUE")).strip().upper() in ("TRUE", "YES", "1", "Y"),
    }


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

    # Pull latest Alpaca cash from Capital tab (synced by sync_alpaca_state.py)
    alpaca_cash = _read_latest_alpaca_cash(ss, logger)

    # Compute effective capital: never exceed your strategy cap; also never exceed Alpaca cash if available.
    strategy_cap = float(k["strategy_capital_usd"])
    effective_capital = strategy_cap
    if alpaca_cash is not None:
        effective_capital = min(strategy_cap, float(alpaca_cash))

    df = read_worksheet_df(ws_trades)
    if df is None or df.empty:
        msg = "Trades sheet is empty."
        logger.info(msg)
        append_rows(ws_logs, [[utc_iso_z(), "select_trades", "INFO", msg]])
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
        append_rows(ws_logs, [[utc_iso_z(), "select_trades", "INFO", msg]])
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
        shares_by_risk = math.floor(float(k["risk_per_trade_usd"]) / risk_per_share)
        if shares_by_risk <= 0:
            continue

        # sizing: per-trade budget cap
        shares_by_budget = math.floor(float(k["max_trade_budget_usd"]) / entry)
        if shares_by_budget <= 0:
            continue

        order_qty = int(min(shares_by_risk, shares_by_budget))
        if order_qty <= 0:
            continue

        entry_notional = entry * order_qty
        tp_notional = tp * order_qty

        # round-trip cost estimate
        cost_est = float(k["est_txn_cost_rate"]) * (entry_notional + tp_notional)

        denom = risk_per_share * order_qty
        cost_in_r = cost_est / denom if denom > 0 else 999.0

        expected_net_r = float(k["take_profit_r_mult"]) - cost_in_r

        # mild stabilizer: prefer lower % risk
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
        append_rows(ws_logs, [[utc_iso_z(), "select_trades", "INFO", msg]])
        # Reset selection-related columns deterministically
        df["selected"] = "FALSE"
        for col in ["priority_rank", "order_qty", "cost_est", "cost_in_r", "expected_net_r", "score"]:
            df[col] = ""
        df = df[TRADE_HEADERS]
        clear_and_write(ws_trades, TRADE_HEADERS, df)
        return

    scored_all = pd.DataFrame(scored_rows)

    # Rank ALL scored trades (so you always see score + rank)
    scored_all = scored_all.sort_values(["score", "expected_net_r"], ascending=[False, False]).reset_index(drop=True)
    scored_all["priority_rank"] = scored_all.index + 1

    # Selection candidates: pass expected net R filter
    scored_sel = scored_all[scored_all["expected_net_r"] >= float(k["min_expected_net_r"])].copy()

    # Greedy pick within effective capital and concurrency cap
    selected_idxs: list[int] = []
    remaining = float(effective_capital)

    if not scored_sel.empty and bool(k["auto_select"]):
        for _, sr in scored_sel.iterrows():
            if len(selected_idxs) >= int(k["max_concurrent_trades"]):
                break
            if float(sr["notional"]) <= remaining:
                selected_idxs.append(int(sr["idx"]))
                remaining -= float(sr["notional"])

    # Reset output columns
    df["selected"] = "FALSE"
    df["priority_rank"] = ""
    df["order_qty"] = ""
    df["cost_est"] = ""
    df["cost_in_r"] = ""
    df["expected_net_r"] = ""
    df["score"] = ""

    # Write scored values back for NEW trades that were scored
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

    # Apply selection
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

    # Persist back
    df = df[TRADE_HEADERS]
    clear_and_write(ws_trades, TRADE_HEADERS, df)


if __name__ == "__main__":
    select_trades()
