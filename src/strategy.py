from __future__ import annotations
import pandas as pd
import numpy as np


def _clean_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    # Ensure numeric
    for c in ["Open", "High", "Low", "Close"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Drop rows with missing OHLC
    df = df.dropna(subset=["Open", "High", "Low", "Close"], how="any")
    return df


def latest_closed_bar(df: pd.DataFrame, max_lookback: int = 12) -> pd.Series | None:
    """
    Get the most recent CLOSED bar that has valid OHLC.
    We skip the last row because it may still be forming/partial.
    Then we walk backwards to find a valid bar.
    """
    df = _clean_ohlc(df)
    if df is None or df.empty or len(df) < 3:
        return None

    # Start from -2 (most recent closed) and walk back
    for i in range(2, min(max_lookback + 2, len(df)) + 1):
        bar = df.iloc[-i]
        try:
            o = float(bar["Open"])
            h = float(bar["High"])
            l = float(bar["Low"])
            c = float(bar["Close"])
            if all(pd.notna([o, h, l, c])) and (h >= l):
                return bar
        except Exception:
            continue
    return None


def atr(df: pd.DataFrame, period: int = 14) -> float:
    df = _clean_ohlc(df)
    if df is None or df.empty or len(df) < period + 2:
        return 0.0

    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)

    tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
    atr_series = tr.rolling(period).mean()
    v = atr_series.iloc[-1]
    return float(v) if pd.notna(v) else float(tr.iloc[-1])


def is_green_bar(last_1h: pd.Series, min_body_ratio: float = 0.25) -> bool:
    o = float(last_1h["Open"])
    c = float(last_1h["Close"])
    h = float(last_1h["High"])
    l = float(last_1h["Low"])
    rng = max(1e-9, h - l)
    body = abs(c - o)
    return (c > o) and (body / rng >= min_body_ratio)


def scan_candidate_1h(df_1h: pd.DataFrame) -> dict | None:
    """
    Minimal candidate definition:
    - Use latest valid CLOSED 1H candle as reference.
    - Require a green bar with meaningful body.
    - Return ref OHLC + ATR for buffers.
    """
    df_1h = _clean_ohlc(df_1h)
    if df_1h is None or df_1h.empty or len(df_1h) < 50:
        return None

    last = latest_closed_bar(df_1h, max_lookback=12)
    if last is None:
        return None

    if not is_green_bar(last):
        return None

    a = atr(df_1h.iloc[-80:], period=14)
    if a <= 0:
        return None

    # Find the timestamp of this 'last' bar in df (best-effort)
    # (not strictly needed, but useful for audit)
    ref_time = None
    try:
        # last.name is the index label
        ref_time = str(last.name)
    except Exception:
        ref_time = ""

    return {
        "ref_time": ref_time,
        "ref_open": float(last["Open"]),
        "ref_high": float(last["High"]),
        "ref_low": float(last["Low"]),
        "ref_close": float(last["Close"]),
        "atr": float(a),
    }


def check_touch_and_confirm_5m(
    df_5m: pd.DataFrame,
    ref_low: float,
    atr_val: float,
    touch_buffer_atr_mult: float,
    confirm_break_buffer_atr_mult: float,
) -> dict | None:
    """
    5m event logic:
    - Touch: candle low <= (ref_low + buffer)
    - Rejection: candle closes back above touch level
    - Confirm: a later candle closes above rejection-high + buffer
    """
    df_5m = _clean_ohlc(df_5m)
    if df_5m is None or df_5m.empty or len(df_5m) < 30:
        return None

    touch_level = ref_low + atr_val * touch_buffer_atr_mult
    break_buf = atr_val * confirm_break_buffer_atr_mult

    # Use last ~9 closed candles (skip the newest forming candle by slicing -10:-1)
    recent = df_5m.iloc[-10:-1].copy()
    if recent.empty or len(recent) < 5:
        return None

    for i in range(len(recent) - 2):
        candle = recent.iloc[i]
        # Touch + Rejection in same candle: low touches, close back above zone
        if candle["Low"] <= touch_level and candle["Close"] > touch_level:
            rej_high = float(candle["High"])
            later = recent.iloc[i + 1 :]
            hit = later[later["Close"] > (rej_high + break_buf)]
            if not hit.empty:
                conf = hit.iloc[0]
                return {
                    "touch_level": float(touch_level),
                    "rejection_time": str(recent.index[i]),
                    "rejection_high": float(rej_high),
                    "confirm_time": str(hit.index[0]),
                    "confirm_close": float(conf["Close"]),
                }

    return None
