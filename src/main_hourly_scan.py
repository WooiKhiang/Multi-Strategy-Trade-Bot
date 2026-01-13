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
    logger = get_logger("hourly_scan", cfg.log_level)

    if not cfg.gsheet_id:
        raise RuntimeError("GSHEET_ID missing in env/.env")

    ss = open_sheet(cfg.gsheet_id, cfg.google_service_account_json)

    ws_universe = ensure_worksheet(ss, "Universe", ["symbol"])

    u_df = read_worksheet_df(ws_universe)
    if u_df is None or u_df.empty or "symbol" not in u_df.columns:
        raise RuntimeError("Universe tab is empty or missing 'symbol' column")

    tickers = (
        u_df["symbol"]
        .astype(str)
        .str.upper()
        .str.strip()
    )
    tickers = [t for t in tickers.tolist() if t and t != "NAN"]
    # de-dup keep order
    tickers = list(dict.fromkeys(tickers))

    # optional: cap if you still want
    if getattr(cfg, "max_universe_tickers", None):
        try:
            cap = int(cfg.max_universe_tickers)
            if cap > 0:
                tickers = tickers[:cap]
        except Exception:
            pass

    # Run scan and overwrite candidates (simple + avoids bloat)
    cand_df = run_hourly_scan(tickers, cfg, logger)
    clear_and_write(ws_candidates, CAND_HEADERS, cand_df)

    # Always write a clean standardized timestamp to Logs
    append_rows(
        ws_logs,
        [[
            utc_iso_z(),
            "hourly_scan",
            "INFO",
            f"Universe={len(tickers)} Candidates={len(cand_df)}",
        ]],
    )


if __name__ == "__main__":
    main()
