from __future__ import annotations

from datetime import datetime, timezone
import pandas as pd

from .config import Config
from .logger import get_logger
from .universe import load_static_universe
from .scan_1h import run_hourly_scan
from .sheets import open_sheet, ensure_worksheet, clear_and_write, append_rows

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


def main():
    cfg = Config()
    logger = get_logger("hourly_scan", cfg.log_level)

    if not cfg.gsheet_id:
        raise RuntimeError("GSHEET_ID missing in env/.env")

    ss = open_sheet(cfg.gsheet_id, cfg.google_service_account_json)

    # Universe snapshot
    ws_universe = ensure_worksheet(ss, "Universe", ["symbol"])

    # ✅ Momentum candidates go here now
    ws_candidates_momentum = ensure_worksheet(ss, "Candidates_Momentum", CAND_HEADERS)

    # Logs
    ws_logs = ensure_worksheet(ss, "Logs", ["timestamp_utc", "component", "level", "message"])

    tickers = load_static_universe(cfg.universe_static_file, cfg.max_universe_tickers)

    # Write universe snapshot (overwrite)
    u_df = pd.DataFrame({"symbol": tickers})
    clear_and_write(ws_universe, ["symbol"], u_df)

    # Run scan and overwrite momentum candidates
    cand_df = run_hourly_scan(tickers, cfg, logger)
    clear_and_write(ws_candidates_momentum, CAND_HEADERS, cand_df)

    # Log summary
    append_rows(
        ws_logs,
        [[
            utc_iso_z(),
            "hourly_scan",
            "INFO",
            f"Universe={len(tickers)} MomentumCandidates={len(cand_df)}",
        ]],
    )


if __name__ == "__main__":
    main()
