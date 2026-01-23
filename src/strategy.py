from __future__ import annotations

import pandas as pd


def _clean_ohlc(df: pd.DataFrame) -> pd.DataFrame:
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


def latest_closed_bar(df: pd.DataFrame, max_lookback: int = 12) -> pd.Series | None:
    df = _clean_ohlc(df)
    if df is None or df.empty or len(df) < 3:
        return None

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


def check_touch_and_confirm_5m(
    df_5m: pd.DataFrame,
    touch_level: float,
    touch_buffer_pct: float = 0.0005,   # 0.05% buffer
    break_buffer_pct: float = 0.0003,   # 0.03% buffer
    lookback_closed: int = 9,           # last N CLOSED bars
) -> dict | None:
    """
    Touch/Reject/Confirm using a FIXED touch level (percent-based level).
    - Touch: Low <= touch_level * (1 + touch_buffer_pct)
    - Rejection: same candle Close > touch_level
    - Confirm: later candle Close > rejection_high * (1 + break_buffer_pct)
    """
    df_5m = _clean_ohlc(df_5m)
    if df_5m is None or df_5m.empty or len(df_5m) < 30:
        return None

    if touch_level is None or float(touch_level) <= 0:
        return None

    # Use last ~lookback_closed CLOSED candles: skip newest forming candle
    recent = df_5m.iloc[-(lookback_closed + 1):-1].copy()
    if recent.empty or len(recent) < 5:
        return None

    touch_ceiling = float(touch_level) * (1.0 + float(touch_buffer_pct))

    for i in range(len(recent) - 2):
        candle = recent.iloc[i]

        low = float(candle["Low"])
        close = float(candle["Close"])
        high = float(candle["High"])

        # Touch + rejection in same candle
        if low <= touch_ceiling and close > float(touch_level):
            rej_high = high
            later = recent.iloc[i + 1:]

            # Confirm close above rejection high (+ buffer)
            confirm_level = float(rej_high) * (1.0 + float(break_buffer_pct))
            hit = later[later["Close"] > confirm_level]

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
