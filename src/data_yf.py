from __future__ import annotations
import time
import pandas as pd
import yfinance as yf
from typing import List, Dict
from .cache import load_cache, save_cache

def _chunk(lst: List[str], n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def download_batched(
    tickers: List[str],
    interval: str,
    period: str,
    batch_size: int,
    sleep_sec: int,
    logger,
    cache_key_prefix: str,
) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}

    remaining = []
    for t in tickers:
        ck = f"{cache_key_prefix}_{t}_{interval}"
        cached = load_cache(ck)
        if cached is not None and not cached.empty:
            out[t] = cached
        else:
            remaining.append(t)

    if not remaining:
        return out

    for batch in _chunk(remaining, batch_size):
        tick_str = " ".join(batch)
        try:
            df = yf.download(
                tickers=tick_str,
                interval=interval,
                period=period,
                group_by="ticker",
                auto_adjust=False,
                threads=True,
                progress=False,
            )

            if isinstance(df.columns, pd.MultiIndex):
                for t in batch:
                    if t in df.columns.levels[0]:
                        tdf = df[t].dropna(how="all")
                        if not tdf.empty:
                            tdf.index = pd.to_datetime(tdf.index)
                            tdf = tdf.sort_index()
                            tdf = tdf[~tdf.index.duplicated(keep="last")]
                            out[t] = tdf
                            save_cache(f"{cache_key_prefix}_{t}_{interval}", tdf)
            else:
                t = batch[0]
                tdf = df.dropna(how="all")
                if not tdf.empty:
                    tdf.index = pd.to_datetime(tdf.index)
                    tdf = tdf.sort_index()
                    tdf = tdf[~tdf.index.duplicated(keep="last")]
                    out[t] = tdf
                    save_cache(f"{cache_key_prefix}_{t}_{interval}", tdf)

        except Exception as e:
            logger.warning(f"yfinance batch failed ({interval}/{period}) for {len(batch)} tickers: {e}")
            time.sleep(max(10, sleep_sec * 3))

        time.sleep(sleep_sec)

    return out
