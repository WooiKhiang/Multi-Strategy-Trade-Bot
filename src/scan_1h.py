from __future__ import annotations

from datetime import datetime, timedelta
import pandas as pd

from .data_yf import download_batched
from .strategy import scan_candidate_1h

# NOTE:
# This module MUST export run_hourly_scan (main_hourly_scan imports it).


CAND_HEADERS = [
    "timestamp_utc",
    "symbol",
    "ref_time",
    "ref_open",
    "ref_high",
    "ref_low",
    "ref_close",
    "atr",
    "expires_utc",
    "status",
]


def run_hourly_scan(tickers, cfg, logger) -> pd.DataFrame:
    """
    1H scan used by main_hourly_scan.py.

    Downloads 1H bars via yfinance and builds candidate rows using
    strategy.scan_candidate_1h(). Output columns follow CAND_HEADERS.
    """
    tickers = list(tickers or [])
    logger.info(f"Hourly scan start. Universe size={len(tickers)} interval=1h")

    if not tickers:
        return pd.DataFrame(columns=CAND_HEADERS)

    data = download_batched(
        tickers=tickers,
        interval="60m",
        period="60d",
        batch_size=getattr(cfg, "yf_batch_size", 50),
        sleep_sec=float(getattr(cfg, "yf_sleep_between_batch_sec", 2)),
        logger=logger,
        cache_key_prefix="yf",
    )

    now_utc = datetime.utcnow()
    ttl_min = int(getattr(cfg, "candidate_ttl_minutes", 60))
    expires = now_utc + timedelta(minutes=ttl_min)

    rows = []
    for sym in tickers:
        df = data.get(sym)
        if df is None or df.empty:
            continue

        try:
            cand = scan_candidate_1h(df)
            if not cand:
                continue

            rows.append(
                {
                    "timestamp_utc": now_utc.isoformat(),
                    "symbol": sym,
                    **cand,
                    "expires_utc": expires.isoformat(),
                    "status": "ACTIVE",
                }
            )
        except Exception as e:
            logger.warning(f"scan candidate error {sym}: {e}")

    out = pd.DataFrame(rows, columns=CAND_HEADERS)
    logger.info(f"Hourly scan done. Candidates={len(out)}")
    return out
