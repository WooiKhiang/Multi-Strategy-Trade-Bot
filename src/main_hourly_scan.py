from __future__ import annotations

from datetime import datetime, timezone
import pandas as pd

from .config import Config
from .logger import get_logger
from .sheets import open_sheet, ensure_worksheet, clear_and_write, append_rows, read_worksheet_df
from .universe import load_static_universe
from .momentum_first_hour_scan import run_momentum_first_hour_scan


CAND_HEADERS = [
    "candidate_id",
    "symbol",
    "strategy",
    "timeframe",
    "trigger_reason",
    "source",
    "ref_price",
    "generated_at_ny",
    "expires_at_ny",
    "status",
    "params_json",
    "notes",
]


def utc_iso_z() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _read_universe_from_sheet(ss) -> list[str]:
    ws = ensure_worksheet(ss, "Universe", ["symbol"])
    df = read_worksheet_df(ws)
    if df is None or df.empty:
        return []
    # tolerate column naming
    col = None
    for c in df.columns:
        if c.strip().lower() == "symbol":
            col = c
            break
    if not col:
        col = df.columns[0]
    tickers = (
        df[col]
        .astype(str)
        .str.upper()
        .str.strip()
        .replace({"": None})
        .dropna()
        .tolist()
    )
    # de-dup preserve order
    seen = set()
    out = []
    for t in tickers:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def main():
    cfg = Config()
    logger = get_logger("main_hourly_scan", cfg.log_level)

    if not cfg.gsheet_id:
        raise RuntimeError("GSHEET_ID missing in env/.env")

    ss = open_sheet(cfg.gsheet_id, cfg.google_service_account_json)

    ws_logs = ensure_worksheet(ss, "Logs", ["timestamp_utc", "component", "level", "message"])
    ws_candidates_momentum = ensure_worksheet(ss, "Candidates_Momentum", CAND_HEADERS)

    # Prefer Universe from sheet (your current design),
    # fallback to static file if sheet empty.
    tickers = _read_universe_from_sheet(ss)
    if not tickers:
        tickers = load_static_universe(cfg.universe_static_file, cfg.max_universe_tickers)
        # also write snapshot back to sheet for visibility
        ws_universe = ensure_worksheet(ss, "Universe", ["symbol"])
        clear_and_write(ws_universe, ["symbol"], pd.DataFrame({"symbol": tickers}))

    # FIXED 1% adjustment for Momentum
    cand_df = run_momentum_first_hour_scan(
        tickers=tickers,
        logger=logger,
        adjustment_pct=0.01,
    )

    # Always enforce the headers on write
    for c in CAND_HEADERS:
        if c not in cand_df.columns:
            cand_df[c] = ""

    cand_df = cand_df[CAND_HEADERS]
    clear_and_write(ws_candidates_momentum, CAND_HEADERS, cand_df)

    append_rows(
        ws_logs,
        [[utc_iso_z(), "main_hourly_scan", "INFO", f"MomentumCandidates={len(cand_df)} Universe={len(tickers)}"]],
    )


if __name__ == "__main__":
    main()
