from __future__ import annotations
import pandas as pd
from datetime import datetime, timedelta

from .data_yf import download_batched
from .strategy_valuegap import scan_candidate_1h

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
    logger.info(f"Hourly scan start. Universe size={len(tickers)} interval=1h")

    data = download_batched(
        tickers=tickers,
        interval="60m",
        period="60d",
        batch_size=cfg.yf_batch_size,
        sleep_sec=cfg.yf_sleep_between_batch_sec,
        logger=logger,
        cache_key_prefix="yf",
    )

    now_utc = datetime.utcnow()
    expires = now_utc + timedelta(minutes=cfg.candidate_ttl_minutes)

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
