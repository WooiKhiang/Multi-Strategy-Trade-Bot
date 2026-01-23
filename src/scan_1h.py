from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, time as dtime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from .data_yf import download_batched

NY = ZoneInfo("America/New_York")

# Unified headers (matches Candidates_Momentum / Candidates_Swing / Candidates_Trend)
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


@dataclass
class FirstHourMetrics:
    ref_time_ny: str
    o: float
    h: float
    l: float
    c: float
    v: float
    body_pct: float
    range_pct: float
    close_pos: float
    upper_wick_pct: float
    vol_ratio: float


def _to_ny_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df index is tz-aware and converted to NY."""
    if df is None or df.empty:
        return df
    df = df.copy()

    idx = df.index
    # If naive, assume UTC (common with yfinance)
    if getattr(idx, "tz", None) is None:
        df.index = pd.to_datetime(df.index, utc=True)
    else:
        df.index = pd.to_datetime(df.index)

    df.index = df.index.tz_convert(NY)
    return df


def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"], how="any")
    return df


def _get_first_hour_bar(df_1h_ny: pd.DataFrame, for_date_ny: datetime.date) -> pd.Series | None:
    """
    Find the 9:30-10:30 candle. In 1h bars, it's typically labeled at 10:30 NY.
    """
    if df_1h_ny is None or df_1h_ny.empty:
        return None

    day = df_1h_ny[df_1h_ny.index.date == for_date_ny]
    if day.empty:
        return None

    # Find bar whose timestamp is 10:30 NY
    target = day[(day.index.hour == 10) & (day.index.minute == 30)]
    if target.empty:
        return None

    return target.iloc[-1]


def _avg_first_hour_volume_last_n(df_1h_ny: pd.DataFrame, n: int = 10) -> float:
    """
    Baseline: avg volume of the 10:30 NY bar over last N trading days available.
    """
    if df_1h_ny is None or df_1h_ny.empty:
        return 0.0

    bars_1030 = df_1h_ny[(df_1h_ny.index.hour == 10) & (df_1h_ny.index.minute == 30)]
    if bars_1030.empty or "Volume" not in bars_1030.columns:
        return 0.0

    vol = bars_1030["Volume"].dropna().astype(float)
    if vol.empty:
        return 0.0

    return float(vol.tail(n).mean())


def _compute_first_hour_metrics(bar: pd.Series, vol_baseline: float) -> FirstHourMetrics | None:
    try:
        o = float(bar["Open"])
        h = float(bar["High"])
        l = float(bar["Low"])
        c = float(bar["Close"])
        v = float(bar.get("Volume", 0.0))
        if o <= 0 or h < l:
            return None

        body_pct = (c - o) / o
        rng = max(1e-9, h - l)
        range_pct = (h - l) / o
        close_pos = (c - l) / rng  # 0..1
        upper_wick_pct = (h - c) / rng  # 0..1

        vol_ratio = 0.0
        if vol_baseline > 0:
            vol_ratio = v / vol_baseline

        return FirstHourMetrics(
            ref_time_ny=str(bar.name),
            o=o, h=h, l=l, c=c, v=v,
            body_pct=float(body_pct),
            range_pct=float(range_pct),
            close_pos=float(close_pos),
            upper_wick_pct=float(upper_wick_pct),
            vol_ratio=float(vol_ratio),
        )
    except Exception:
        return None


def _passes_momentum_criteria(m: FirstHourMetrics) -> bool:
    # Green candle
    if m.c <= m.o:
        return False
    # Body >= 0.3%
    if m.body_pct < 0.003:
        return False
    # Close in top 50%
    if m.close_pos < 0.5:
        return False
    # Volume >= 1.2x baseline
    if m.vol_ratio < 1.2:
        return False
    # Range >= 0.5%
    if m.range_pct < 0.005:
        return False
    # Upper wick <= 40%
    if m.upper_wick_pct > 0.4:
        return False
    return True


def _avg_daily_range_pct_20(df_1d: pd.DataFrame) -> float:
    """
    avg((High-Low)/Close) over last 20 trading days.
    """
    df_1d = _clean_ohlcv(df_1d)
    if df_1d is None or df_1d.empty or len(df_1d) < 25:
        return 0.0

    x = df_1d.tail(20).copy()
    rng_pct = (x["High"].astype(float) - x["Low"].astype(float)) / x["Close"].astype(float)
    rng_pct = rng_pct.replace([pd.NA, pd.NaT], pd.NA).dropna()
    if rng_pct.empty:
        return 0.0
    return float(rng_pct.mean())


def _smart_adjustment_pct(avg_daily_range_pct: float) -> float:
    """
    LOW <1.5% => 0.8%
    MED 1.5%-3.5% => 1.0%
    HIGH >3.5% => 1.2%
    """
    if avg_daily_range_pct <= 0:
        return 0.01  # default
    if avg_daily_range_pct < 0.015:
        return 0.008
    if avg_daily_range_pct > 0.035:
        return 0.012
    return 0.01


def _expiry_1135_same_day(ny_now: datetime) -> datetime:
    exp = ny_now.replace(hour=11, minute=35, second=0, microsecond=0)
    return exp


def run_momentum_first_hour_scan(tickers, cfg, logger) -> pd.DataFrame:
    """
    Momentum strategy candidate generation:
    - Evaluate ONLY 9:30-10:30 candle (10:30 timestamp NY) once.
    - Compute smart % adjustment from daily volatility.
    - Output unified candidate schema for Candidates_Momentum.
    """
    ny_now = datetime.now(NY)

    # Fetch 1H data (enough history to get 10 first-hour candles baseline)
    data_1h = download_batched(
        tickers=tickers,
        interval="60m",
        period="60d",
        batch_size=cfg.yf_batch_size,
        sleep_sec=cfg.yf_sleep_between_batch_sec,
        logger=logger,
        cache_key_prefix="yf_1h",
    )

    # Fetch 1D data for volatility bucket
    data_1d = download_batched(
        tickers=tickers,
        interval="1d",
        period="6mo",
        batch_size=cfg.yf_batch_size,
        sleep_sec=cfg.yf_sleep_between_batch_sec,
        logger=logger,
        cache_key_prefix="yf_1d",
    )

    expires_ny = _expiry_1135_same_day(ny_now)

    rows = []
    for sym in tickers:
        try:
            df1h = data_1h.get(sym)
            df1d = data_1d.get(sym)

            if df1h is None or df1h.empty or df1d is None or df1d.empty:
                continue

            df1h = _to_ny_index(_clean_ohlcv(df1h))
            df1d = _clean_ohlcv(df1d)

            vol_baseline = _avg_first_hour_volume_last_n(df1h, n=10)
            bar = _get_first_hour_bar(df1h, ny_now.date())
            if bar is None:
                continue

            m = _compute_first_hour_metrics(bar, vol_baseline)
            if not m or not _passes_momentum_criteria(m):
                continue

            avg_rng20 = _avg_daily_range_pct_20(df1d)
            adj_pct = _smart_adjustment_pct(avg_rng20)

            # Reference for pullback: First-hour HIGH * (1 - adj_pct)
            touch_level = m.h * (1.0 - adj_pct)

            candidate_id = f"MOM_{sym}_{ny_now.strftime('%Y%m%d')}"
            params = {
                "ref_time_ny": m.ref_time_ny,
                "first_hour_open": m.o,
                "first_hour_high": m.h,
                "first_hour_low": m.l,
                "first_hour_close": m.c,
                "first_hour_volume": m.v,
                "body_pct": m.body_pct,
                "range_pct": m.range_pct,
                "close_pos": m.close_pos,
                "upper_wick_pct": m.upper_wick_pct,
                "vol_ratio": m.vol_ratio,
                "avg_daily_range_pct_20": avg_rng20,
                "adj_pct": adj_pct,
                "touch_level": touch_level,
            }

            rows.append({
                "candidate_id": candidate_id,
                "symbol": sym,
                "strategy": "MOMENTUM",
                "timeframe": "1H",
                "trigger_reason": "FIRST_HOUR_930_1030",
                "source": "1H_SCAN_1030",
                "ref_price": float(touch_level),
                "generated_at_ny": ny_now.replace(microsecond=0).isoformat(),
                "expires_at_ny": expires_ny.isoformat(),
                "status": "ACTIVE",
                "params_json": json.dumps(params, ensure_ascii=False),
                "notes": "First-hour momentum candidate; ref_price is smart % pullback from first-hour high",
            })

        except Exception as e:
            logger.warning(f"momentum scan error {sym}: {e}")

    out = pd.DataFrame(rows, columns=CAND_HEADERS)
    logger.info(f"Momentum first-hour scan done. Candidates={len(out)}")
    return out
