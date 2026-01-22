from __future__ import annotations

from datetime import datetime, timezone
import pandas as pd

from .config import Config
from .logger import get_logger
from .universe import load_static_universe
from .scan_1h import run_hourly_scan, CAND_HEADERS
from .sheets import open_sheet, ensure_worksheet, clear_and_write, append_rows


def utc_iso_z() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def main():
    cfg = Config()
    logger = get_logger("momentum_scan", cfg.log_level)

    if not cfg.gsheet_id:
        raise RuntimeError("GSHEET_ID missing in env/.env")

    # Open Google Sheet
    ss = open_sheet(cfg.gsheet_id, cfg.google_service_account_json)

    # Ensure required worksheets
    ws_universe = ensure_worksheet(ss, "Universe", ["symbol"])
    ws_candidates = ensure_worksheet(ss, "Candidates_Momentum", CAND_HEADERS)
    ws_logs = ensure_worksheet(
        ss,
        "Logs",
        ["timestamp_utc", "component", "level", "message"]
    )

    # Load universe FROM GOOGLE SHEETS (not hard-coded)
    tickers = load_static_universe(
        cfg.universe_static_file,
        cfg.max_universe_tickers
    )

    # Run FIRST-HOUR momentum scan only
    logger.info("Momentum scan start (first-hour candle only)")
    cand_df = run_hourly_scan(tickers, cfg, logger)

    # Overwrite Momentum candidates ONLY
    clear_and_write(ws_candidates, CAND_HEADERS, cand_df)

    # Log result
    append_rows(
        ws_logs,
        [[
            utc_iso_z(),
            "momentum_scan",
            "INFO",
            f"Momentum scan done. Candidates={len(cand_df)}",
        ]],
    )


if __name__ == "__main__":
    main()
