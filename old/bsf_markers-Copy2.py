import pandas as pd
import numpy as np
from datetime import datetime
# -------------------------------
# 1️⃣ Candlestick Pattern Engine
# -------------------------------
'''
Candlestick pattern markers:

Each boolean column flags the presence of a specific candlestick pattern on a given row of OHLC data.

Reversal Patterns:
- Doji: Small body, signals indecision
- Hammer / InvertedHammer: Long lower/upper shadow, potential reversal
- ShootingStar: Bearish reversal with long upper shadow
- BullishEngulfing / BearishEngulfing: Full-body engulfing of prior candle
- BullishHarami / BearishHarami: Small body inside prior candle range
- HaramiCross: Doji inside prior candle range
- PiercingLine / DarkCloudCover: Two-bar reversal with partial retracement
- MorningStar / EveningStar: Three-bar reversal with gap and confirmation
- ThreeWhiteSoldiers / ThreeBlackCrows: Strong trend continuation or reversal

Extended Patterns:
- BullishMarubozu / BearishMarubozu: Full-body candles with no shadows
- TweezerTop / TweezerBottom: Matching highs/lows across two candles
- InsideBar / OutsideBar: Volatility compression or expansion

Diagnostics:
- SuspiciousCandle: Zero range or negligible body
- NearHigh / NearLow: Candle near recent high/low (5-bar window)

Metadata:
- PatternCount: Total number of patterns detected on a row
- PatternType: Prioritized label for key bullish setups

'''
# -------------------------------
# Add Candle Patterns
# -------------------------------
import numpy as np
import pandas as pd

def get_candle_params(close_price):
    # normalize thresholds relative to magnitude
    if close_price > 50:
        return dict(doji_thresh=0.1, long_body=0.6, shadow_ratio=2.0)
    elif close_price > 1:
        return dict(doji_thresh=0.05, long_body=0.5, shadow_ratio=1.5)
    else:
        return dict(doji_thresh=0.01, long_body=0.3, shadow_ratio=1.2)

# -------------------------------
# Add Candlestick Patterns
# -------------------------------
def add_candle_patterns_old(df, o="Open", h="High", l="Low", c="Close",
                        doji_thresh=0.1, long_body=0.6, small_body=0.25,
                        shadow_ratio=2.0, near_edge=0.25, pattern_window=5):
    """
    Adds candlestick pattern columns to the DataFrame.
    
    Parameters:
        df : pd.DataFrame
            Stock OHLC data.
        o, h, l, c : str
            Column names for Open, High, Low, Close.
        doji_thresh : float
            Fraction of range to classify Doji.
        long_body : float
            Fraction of range to classify Marubozu.
        small_body : float
            Fraction for Morning/Evening Star small middle candle.
        shadow_ratio : float
            Min ratio for shadows (Hammer/Shooting Star).
        near_edge : float
            How close the candle is to high/low for certain patterns.
        pattern_window : int
            Rolling window for multi-period patterns like NearHigh/NearLow.
            
    Returns:
        pd.DataFrame with boolean pattern columns, PatternCount, PatternType
    """
    df = df.copy()
    O, H, L, C = df[o].astype(float), df[h].astype(float), df[l].astype(float), df[c].astype(float)
    rng   = (H - L).replace(0, np.nan)
    body  = (C - O).abs()
    upsh  = H - np.maximum(O, C)
    dnsh  = np.minimum(O, C) - L
    bull  = C > O
    bear  = O > C

    new_cols = {}

    # -------------------------------
    # Core single-bar patterns
    # -------------------------------
    new_cols["Doji"] = (body <= doji_thresh * rng)
    new_cols["Hammer"] = (dnsh >= shadow_ratio * body) & (upsh <= 0.2 * body) & (np.maximum(O, C) >= H - near_edge * rng)
    new_cols["InvertedHammer"] = (upsh >= shadow_ratio * body) & (dnsh <= 0.2 * body) & (np.minimum(O, C) <= L + near_edge * rng)
    new_cols["ShootingStar"] = (upsh >= shadow_ratio * body) & (dnsh <= 0.2 * body) & (np.minimum(O, C) <= L + near_edge * rng)

    # -------------------------------
    # Two-bar patterns
    # -------------------------------
    O1, C1 = O.shift(1), C.shift(1)
    new_cols["BullishEngulfing"] = (O1 > C1) & bull & (C >= O1) & (O <= C1)
    new_cols["BearishEngulfing"] = (C1 > O1) & bear & (O >= C1) & (C <= O1)

    new_cols["BullishHarami"] = (O1 > C1) & bull & (np.maximum(O, C) <= np.maximum(O1, C1)) & (np.minimum(O, C) >= np.minimum(O1, C1))
    new_cols["BearishHarami"] = (C1 > O1) & bear & (np.maximum(O, C) <= np.maximum(O1, C1)) & (np.minimum(O, C) >= np.minimum(O1, C1))
    new_cols["HaramiCross"] = new_cols["Doji"] & ((np.maximum(O, C) <= np.maximum(O1, C1)) & (np.minimum(O, C) >= np.minimum(O1, C1)))

    new_cols["PiercingLine"] = (O1 > C1) & bull & (O < C1) & (C > (O1 + C1) / 2) & (C < O1)
    new_cols["DarkCloudCover"] = (C1 > O1) & bear & (O > C1) & (C < (O1 + C1) / 2) & (C > O1)

    # -------------------------------
    # Three-bar patterns
    # -------------------------------
    O2, C2 = O.shift(2), C.shift(2)
    new_cols["MorningStar"] = (O2 > C2) & (abs(C1 - O1) < abs(C2 - O2) * small_body) & bull & (C >= (O2 + C2) / 2)
    new_cols["EveningStar"] = (C2 > O2) & (abs(C1 - O1) < abs(C2 - O2) * small_body) & bear & (C <= (O2 + C2) / 2)

    bull1, bull2 = bull.shift(1), bull.shift(2)
    bear1, bear2 = bear.shift(1), bear.shift(2)
    new_cols["ThreeWhiteSoldiers"] = bull & bull1 & bull2 & (C > C.shift(1)) & (C.shift(1) > C.shift(2))
    new_cols["ThreeBlackCrows"] = bear & bear1 & bear2 & (C < C.shift(1)) & (C.shift(1) < C.shift(2))

    # -------------------------------
    # Single-bar extreme candles
    # -------------------------------
    new_cols["BullishMarubozu"] = (body >= long_body * rng) & (upsh <= 0.05 * rng) & (dnsh <= 0.05 * rng) & bull
    new_cols["BearishMarubozu"] = (body >= long_body * rng) & (upsh <= 0.05 * rng) & (dnsh <= 0.05 * rng) & bear

    # -------------------------------
    # Tweezer & Inside/Outside
    # -------------------------------
    new_cols["TweezerTop"] = (H == H.shift(1)) & bear & bull.shift(1)
    new_cols["TweezerBottom"] = (L == L.shift(1)) & bull & bear.shift(1)
    new_cols["InsideBar"] = (H < H.shift(1)) & (L > L.shift(1))
    new_cols["OutsideBar"] = (H > H.shift(1)) & (L < L.shift(1))

    # -------------------------------
    # Multi-period diagnostics
    # -------------------------------
    new_cols["SuspiciousCandle"] = (rng <= 0.001) | (body <= 0.001)
    recent_high = H.rolling(pattern_window).max()
    recent_low = L.rolling(pattern_window).min()
    new_cols["NearHigh"] = H >= recent_high * (1 - near_edge)
    new_cols["NearLow"] = L <= recent_low * (1 + near_edge)

    # -------------------------------
    # Add to DataFrame
    # -------------------------------
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    pattern_cols = [col for col in new_cols if df[col].dtype == bool]
    df["PatternCount"] = df[pattern_cols].sum(axis=1)

    df["PatternType"] = np.select(
        [df.get("BullishEngulfing", False), df.get("MorningStar", False),
         df.get("ThreeWhiteSoldiers", False), df.get("BullishMarubozu", False)],
        ["BullishEngulfing", "MorningStar", "ThreeWhiteSoldiers", "BullishMarubozu"],
        default="None"
    )

    return df.fillna(False)

import numpy as np
import pandas as pd

def add_candle_patterns_old2(df, o="Open", h="High", l="Low", c="Close",
                        doji_thresh=0.1, long_body=0.6, small_body=0.25,
                        shadow_ratio=2.0, near_edge=0.25, pattern_window=5,
                        normalize=True):
    """
    Add candlestick patterns (scale-invariant).
    If normalize=True, all thresholds use % of price (not raw absolute values).
    """
    df = df.copy()
    O, H, L, C = df[o].astype(float), df[h].astype(float), df[l].astype(float), df[c].astype(float)

    # 🔹 Normalize ranges to % of Close price (scale-invariant)
    if normalize:
        ref_price = C.replace(0, np.nan)
        rng  = (H - L) / ref_price
        body = (C - O).abs() / ref_price
        upsh = (H - np.maximum(O, C)) / ref_price
        dnsh = (np.minimum(O, C) - L) / ref_price
    else:
        rng  = (H - L).replace(0, np.nan)
        body = (C - O).abs()
        upsh = H - np.maximum(O, C)
        dnsh = np.minimum(O, C) - L

    bull = C > O
    bear = O > C
    new_cols = {}

    # -------------------------------
    # Single-bar patterns
    # -------------------------------
    new_cols["Doji"] = (body <= doji_thresh * rng)
    new_cols["Hammer"] = (dnsh >= shadow_ratio * body) & (upsh <= 0.2 * body)
    new_cols["InvertedHammer"] = (upsh >= shadow_ratio * body) & (dnsh <= 0.2 * body)
    new_cols["ShootingStar"] = new_cols["InvertedHammer"]
    new_cols["BullishMarubozu"] = (body >= long_body * rng) & (upsh <= 0.05 * rng) & (dnsh <= 0.05 * rng) & bull
    new_cols["BearishMarubozu"] = (body >= long_body * rng) & (upsh <= 0.05 * rng) & (dnsh <= 0.05 * rng) & bear
    new_cols["SuspiciousCandle"] = (rng <= 0.0001) | (body <= 0.0001)  # in % terms if normalize=True

    # -------------------------------
    # Multi-bar patterns
    # -------------------------------
    O_shift1, C_shift1 = O.shift(1), C.shift(1)
    O_shift2, C_shift2 = O.shift(2), C.shift(2)
    H_shift1, L_shift1 = H.shift(1), L.shift(1)
    bull1, bull2 = bull.shift(1), bull.shift(2)
    bear1, bear2 = bear.shift(1), bear.shift(2)

    new_cols["BullishEngulfing"] = (O_shift1 > C_shift1) & bull & (C >= O_shift1) & (O <= C_shift1)
    new_cols["BearishEngulfing"] = (C_shift1 > O_shift1) & bear & (O >= C_shift1) & (C <= O_shift1)
    new_cols["BullishHarami"] = (O_shift1 > C_shift1) & bull & (np.maximum(O, C) <= np.maximum(O_shift1, C_shift1)) & (np.minimum(O, C) >= np.minimum(O_shift1, C_shift1))
    new_cols["BearishHarami"] = (C_shift1 > O_shift1) & bear & (np.maximum(O, C) <= np.maximum(O_shift1, C_shift1)) & (np.minimum(O, C) >= np.minimum(O_shift1, C_shift1))
    new_cols["HaramiCross"] = new_cols["Doji"] & (np.maximum(O, C) <= np.maximum(O_shift1, C_shift1)) & (np.minimum(O, C) >= np.minimum(O_shift1, C_shift1))
    new_cols["PiercingLine"] = (O_shift1 > C_shift1) & bull & (O < C_shift1) & (C > (O_shift1 + C_shift1)/2) & (C < O_shift1)
    new_cols["DarkCloudCover"] = (C_shift1 > O_shift1) & bear & (O > C_shift1) & (C < (O_shift1 + C_shift1)/2) & (C > O_shift1)
    new_cols["MorningStar"] = (O_shift2 > C_shift2) & (abs(C_shift1 - O_shift1) < abs(C_shift2 - O_shift2) * small_body) & bull & (C >= (O_shift2 + C_shift2)/2)
    new_cols["EveningStar"] = (C_shift2 > O_shift2) & (abs(C_shift1 - O_shift1) < abs(C_shift2 - O_shift2) * small_body) & bear & (C <= (O_shift2 + C_shift2)/2)
    new_cols["ThreeWhiteSoldiers"] = bull & bull1 & bull2 & (C > C_shift1) & (C_shift1 > C_shift2)
    new_cols["ThreeBlackCrows"] = bear & bear1 & bear2 & (C < C_shift1) & (C_shift1 < C_shift2)

    new_cols["TweezerTop"] = (H == H_shift1) & bear & bull1
    new_cols["TweezerBottom"] = (L == L_shift1) & bull & bear1

    # Inside/Outside Bar (window)
    new_cols["InsideBar"] = (H < H_shift1) & (L > L_shift1)
    new_cols["OutsideBar"] = (H > H_shift1) & (L < L_shift1)

    # -------------------------------
    # Rolling high/low diagnostics
    # -------------------------------
    recent_high = H.rolling(pattern_window).max()
    recent_low  = L.rolling(pattern_window).min()
    new_cols["NearHigh"] = H >= recent_high * (1 - near_edge)
    new_cols["NearLow"]  = L <= recent_low * (1 + near_edge)

    # -------------------------------
    # Add columns to DataFrame
    # -------------------------------
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    # Pattern metadata
    pattern_cols = [col for col in new_cols if df[col].dtype == bool]
    df["PatternCount"] = df[pattern_cols].sum(axis=1)
    df["PatternType"] = np.select(
        [df["BullishEngulfing"], df["MorningStar"], df["ThreeWhiteSoldiers"], df["BullishMarubozu"]],
        ["BullishEngulfing", "MorningStar", "ThreeWhiteSoldiers", "BullishMarubozu"],
        default="None"
    )

    return df.fillna(False)
def add_candle_patterns(df, o="Open", h="High", l="Low", c="Close",
                        doji_thresh=0.1, long_body=0.6, small_body=0.25,
                        shadow_ratio=2.0, near_edge=0.25, pattern_window=5,
                        normalize=True, rng_thresh=1e-4):
    """
    Detect candlestick patterns with scale-invariant normalization.
    Returns df with anatomy columns + single- and multi-bar patterns.
    """

    df = df.copy()
    O, H, L, C = df[o].astype(float), df[h].astype(float), df[l].astype(float), df[c].astype(float)

    # -------------------------------
    # Anatomy
    # -------------------------------
    if normalize:
        ref_price = C.replace(0, np.nan)
        body = (C - O).abs() / ref_price
        upsh = (H - np.maximum(O, C)) / ref_price
        dnsh = (np.minimum(O, C) - L) / ref_price
        rng = (H - L) / ref_price
    else:
        body = (C - O).abs()
        upsh = H - np.maximum(O, C)
        dnsh = np.minimum(O, C) - L
        rng = (H - L).replace(0, np.nan)

    bull, bear = C > O, O > C
    new_cols = {}

    # -------------------------------
    # Single-bar patterns
    # -------------------------------
    new_cols["Doji"] = body <= doji_thresh * rng
    new_cols["Hammer"] = (dnsh >= shadow_ratio * body) & (upsh <= 0.2 * body) & bull
    new_cols["InvertedHammer"] = (upsh >= shadow_ratio * body) & (dnsh <= 0.2 * body)
    new_cols["ShootingStar"] = new_cols["InvertedHammer"]
    new_cols["BullishMarubozu"] = (body >= long_body * rng) & (upsh <= 0.05 * rng) & (dnsh <= 0.05 * rng) & bull
    new_cols["BearishMarubozu"] = (body >= long_body * rng) & (upsh <= 0.05 * rng) & (dnsh <= 0.05 * rng) & bear
    new_cols["SuspiciousCandle"] = (rng <= rng_thresh) | (body <= rng_thresh)

    # -------------------------------
    # Multi-bar patterns
    # -------------------------------
    O_shift1, C_shift1 = O.shift(1), C.shift(1)
    O_shift2, C_shift2 = O.shift(2), C.shift(2)
    H_shift1, L_shift1 = H.shift(1), L.shift(1)
    bull1, bull2 = bull.shift(1), bull.shift(2)
    bear1, bear2 = bear.shift(1), bear.shift(2)

    new_cols["BullishEngulfing"] = (O_shift1 > C_shift1) & bull & (C >= O_shift1) & (O <= C_shift1)
    new_cols["BearishEngulfing"] = (C_shift1 > O_shift1) & bear & (O >= C_shift1) & (C <= O_shift1)
    new_cols["BullishHarami"] = (O_shift1 > C_shift1) & bull & (np.maximum(O, C) <= np.maximum(O_shift1, C_shift1)) & (np.minimum(O, C) >= np.minimum(O_shift1, C_shift1))
    new_cols["BearishHarami"] = (C_shift1 > O_shift1) & bear & (np.maximum(O, C) <= np.maximum(O_shift1, C_shift1)) & (np.minimum(O, C) >= np.minimum(O_shift1, C_shift1))
    new_cols["HaramiCross"] = new_cols["Doji"] & (np.maximum(O, C) <= np.maximum(O_shift1, C_shift1)) & (np.minimum(O, C) >= np.minimum(O_shift1, C_shift1))
    new_cols["PiercingLine"] = (O_shift1 > C_shift1) & bull & (O < C_shift1) & (C > (O_shift1 + C_shift1)/2) & (C < O_shift1)
    new_cols["DarkCloudCover"] = (C_shift1 > O_shift1) & bear & (O > C_shift1) & (C < (O_shift1 + C_shift1)/2) & (C > O_shift1)
    new_cols["MorningStar"] = (O_shift2 > C_shift2) & (abs(C_shift1 - O_shift1) < abs(C_shift2 - O_shift2) * small_body) & bull & (C >= (O_shift2 + C_shift2)/2)
    new_cols["EveningStar"] = (C_shift2 > O_shift2) & (abs(C_shift1 - O_shift1) < abs(C_shift2 - O_shift2) * small_body) & bear & (C <= (O_shift2 + C_shift2)/2)
    new_cols["ThreeWhiteSoldiers"] = bull & bull1 & bull2 & (C > C_shift1) & (C_shift1 > C_shift2)
    new_cols["ThreeBlackCrows"] = bear & bear1 & bear2 & (C < C_shift1) & (C_shift1 < C_shift2)

    new_cols["TweezerTop"] = (H == H_shift1) & bear & bull1
    new_cols["TweezerBottom"] = (L == L_shift1) & bull & bear1

    # Inside/Outside Bar
    new_cols["InsideBar"] = (H < H_shift1) & (L > L_shift1)
    new_cols["OutsideBar"] = (H > H_shift1) & (L < L_shift1)

    # -------------------------------
    # Rolling high/low diagnostics
    # -------------------------------
    recent_high = H.rolling(pattern_window).max()
    recent_low = L.rolling(pattern_window).min()
    new_cols["NearHigh"] = H >= recent_high * (1 - near_edge)
    new_cols["NearLow"] = L <= recent_low * (1 + near_edge)

    # -------------------------------
    # Add new columns
    # -------------------------------
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    # Pattern metadata
    pattern_cols = [col for col in new_cols if df[col].dtype == bool]
    df["PatternCount"] = df[pattern_cols].sum(axis=1)
    df["PatternType"] = np.select(
        [df.get("BullishEngulfing", False), df.get("MorningStar", False), df.get("ThreeWhiteSoldiers", False), df.get("BullishMarubozu", False)],
        ["BullishEngulfing", "MorningStar", "ThreeWhiteSoldiers", "BullishMarubozu"],
        default="None"
    )

    # -------------------------------
    # Anatomy columns (debugging / features)
    # -------------------------------
    df["BodyRel"] = body
    df["UpperShadowRel"] = upsh
    df["LowerShadowRel"] = dnsh
    df["RangeRel"] = rng

    return df.fillna(False)
    

def add_trend_filters_old(df, c="Close", timeframe="Daily"):
    """
    Adds trend indicators (MA, slope, momentum, volatility) for a single timeframe.

    Parameters:
        df : pd.DataFrame
            OHLC data.
        c : str
            Close column name.
        timeframe : str
            One of ["Short", "Swing", "Long", "Daily"].

    Returns:
        pd.DataFrame with new trend columns:
        MA, MA_slope, UpTrend_MA, DownTrend_MA,
        RecentReturn, UpTrend_Return, DownTrend_Return,
        Volatility, LowVolatility, HighVolatility,
        ROC, MomentumUp, MomentumDown,
        ConfirmedUpTrend, ConfirmedDownTrend
    """
    df = df.copy()

    # Define profile windows per timeframe
    profiles = {
        "Short":  {"ma": 2,  "ret": 1,  "vol": 3,  "roc_thresh": 0.02},
        "Swing":  {"ma": 5,  "ret": 5,  "vol": 5,  "roc_thresh": 0.02},
        "Long":   {"ma": 10, "ret": 10, "vol": 10, "roc_thresh": 0.02},
        "Daily":  {"ma": 7,  "ret": 1,  "vol": 5,  "roc_thresh": 0.02}
    }

    if timeframe not in profiles:
        raise ValueError(f"Invalid timeframe: {timeframe}. Must be one of {list(profiles.keys())}")

    params = profiles[timeframe]

    # Compute trend columns
    ma = df[c].rolling(params["ma"]).mean()
    ma_slope = ma.diff(params["ma"])
    recent_ret = df[c].pct_change(params["ret"])
    vol = df[c].rolling(params["vol"]).std()
    vol_med = vol.median()
    roc = df[c].pct_change(params["ma"])

    df["MA"] = ma
    df["MA_slope"] = ma_slope
    df["UpTrend_MA"] = ma_slope > 0
    df["DownTrend_MA"] = ma_slope < 0

    df["RecentReturn"] = recent_ret
    df["UpTrend_Return"] = recent_ret > 0
    df["DownTrend_Return"] = recent_ret < 0

    df["Volatility"] = vol
    df["LowVolatility"] = vol < vol_med
    df["HighVolatility"] = vol > vol_med

    df["ROC"] = roc
    df["MomentumUp"] = roc > params["roc_thresh"]
    df["MomentumDown"] = roc < -params["roc_thresh"]

    df["ConfirmedUpTrend"] = df["UpTrend_MA"] & df["UpTrend_Return"] & df["MomentumUp"]
    df["ConfirmedDownTrend"] = df["DownTrend_MA"] & df["DownTrend_Return"] & df["MomentumDown"]

    return df

def add_trend_filters(df, c="Close", timeframe="Daily"):
    """
    Adds trend indicators (scale-invariant, % returns not absolute deltas).
    Works consistently across penny stocks and high-priced stocks.

    Parameters:
        df : pd.DataFrame
            OHLC data.
        c : str
            Close column name.
        timeframe : str
            One of ["Short", "Swing", "Long", "Daily"].

    Returns:
        pd.DataFrame with new trend columns:
        MA, MA_slope, UpTrend_MA, DownTrend_MA,
        RecentReturn, UpTrend_Return, DownTrend_Return,
        Volatility, LowVolatility, HighVolatility,
        ROC, MomentumUp, MomentumDown,
        ConfirmedUpTrend, ConfirmedDownTrend
    """
    df = df.copy()

    # Define profile windows per timeframe
    profiles = {
        "Short":  {"ma": 2,  "ret": 1,  "vol": 3,  "roc_thresh": 0.02},
        "Swing":  {"ma": 5,  "ret": 5,  "vol": 5,  "roc_thresh": 0.02},
        "Long":   {"ma": 10, "ret": 10, "vol": 10, "roc_thresh": 0.02},
        "Daily":  {"ma": 7,  "ret": 1,  "vol": 5,  "roc_thresh": 0.02}
    }

    if timeframe not in profiles:
        raise ValueError(f"Invalid timeframe: {timeframe}. Must be one of {list(profiles.keys())}")

    params = profiles[timeframe]

    # -------------------------------
    # Moving Average (still in price units, but slope is normalized)
    # -------------------------------
    ma = df[c].rolling(params["ma"]).mean()

    # 🔹 Normalize slope by dividing by prior MA (i.e. % slope)
    ma_slope = ma.pct_change(params["ma"])

    # -------------------------------
    # Returns & Volatility (all %)
    # -------------------------------
    recent_ret = df[c].pct_change(params["ret"])
    vol = df[c].pct_change().rolling(params["vol"]).std()

    # Dynamic volatility bands (robust across scales)
    vol_med = vol.median()

    # -------------------------------
    # Rate of Change (ROC)
    # -------------------------------
    roc = df[c].pct_change(params["ma"])

    # -------------------------------
    # Add to DF
    # -------------------------------
    df["MA"] = ma
    df["MA_slope"] = ma_slope
    df["UpTrend_MA"] = ma_slope > 0
    df["DownTrend_MA"] = ma_slope < 0

    df["RecentReturn"] = recent_ret
    df["UpTrend_Return"] = recent_ret > 0
    df["DownTrend_Return"] = recent_ret < 0

    df["Volatility"] = vol
    df["LowVolatility"] = vol < vol_med
    df["HighVolatility"] = vol > vol_med

    df["ROC"] = roc
    df["MomentumUp"] = roc > params["roc_thresh"]
    df["MomentumDown"] = roc < -params["roc_thresh"]

    df["ConfirmedUpTrend"] = df["UpTrend_MA"] & df["UpTrend_Return"] & df["MomentumUp"]
    df["ConfirmedDownTrend"] = df["DownTrend_MA"] & df["DownTrend_Return"] & df["MomentumDown"]

    return df




# -------------------------------
# Confirmed Signals
# -------------------------------
def add_confirmed_signals(df):
    """
    Generate confirmed/validated candlestick signals, grouped by type.
    Assumes the dataframe contains only a single timeframe of trend columns.
    """
    df = df.copy()

    # Define groups with pattern -> required trend
    signal_groups = {
        "Bullish": {
            "ValidHammer": "DownTrend_MA",
            "ValidBullishEngulfing": "DownTrend_MA",
            "ValidPiercingLine": "DownTrend_Return",
            "ValidMorningStar": "DownTrend_MA",
            "ValidThreeWhiteSoldiers": "DownTrend_MA",
            "ValidBullishMarubozu": "DownTrend_MA",
            "ValidTweezerBottom": "DownTrend_MA"
        },
        "Bearish": {
            "ValidShootingStar": "UpTrend_MA",
            "ValidBearishEngulfing": "UpTrend_MA",
            "ValidDarkCloud": "UpTrend_Return",
            "ValidEveningStar": "UpTrend_MA",
            "ValidThreeBlackCrows": "UpTrend_MA",
            "ValidBearishMarubozu": "UpTrend_MA",
            "ValidTweezerTop": "UpTrend_MA"
        },
        "Reversal": {
            "ValidHaramiCross": "UpTrend_MA",  # example, could adjust by trend
            "ValidBullishHarami": "DownTrend_MA",
            "ValidBearishHarami": "UpTrend_MA"
        },
        "Continuation": {
            "ValidInsideBar": "UpTrend_MA",    # bullish continuation
            "ValidOutsideBar": "DownTrend_MA"  # bearish continuation
        },
        "Custom": {
            # This is where you can add your own custom rules
            "ValidCustomSignal1": "DownTrend_MA",
            "ValidCustomSignal2": "UpTrend_Return"
        }
    }

    # Generate validated signals per group
    for group_name, patterns in signal_groups.items():
        for valid_col, trend_col in patterns.items():
            # Original pattern name is valid_col without "Valid"
            pattern_name = valid_col.replace("Valid", "")
            df[valid_col] = df.get(pattern_name, False) & df.get(trend_col, False)

    return df


def add_signal_strength(df, group_prefixes=None):
    """
    Adds a SignalStrength column by summing all confirmed signals.
    Optionally, can sum only signals from certain groups by prefix.

    Parameters:
        df : pd.DataFrame
            DataFrame containing ValidXXX columns
        group_prefixes : list of str, optional
            Only include signals whose column names start with these prefixes
            Example: ["ValidBullish", "ValidBearish"]

    Returns:
        pd.DataFrame with "SignalStrength" column
    """
    df = df.copy()

    # Identify all "Valid" signal columns
    valid_cols = [col for col in df.columns if col.startswith("Valid")]

    # Filter by prefixes if provided
    if group_prefixes:
        valid_cols = [col for col in valid_cols if any(col.startswith(p) for p in group_prefixes)]

    # If no valid columns found, default to 0
    if not valid_cols:
        df["SignalStrength"] = 0
    else:
        df["SignalStrength"] = df[valid_cols].astype(int).sum(axis=1)

    return df



    '''
    Comprehensive forward-looking signal generator.

    Combines:
    - Rolling momentum & volatility filters (absolute + z-score);
    - Candlestick pattern counts and clusters;
    - Exhaustion logic (NearHigh/Low + OutsideBar);
    - Market structure shifts (higher highs / lower lows);
    - Robust forward-labeled TomorrowAction.

    Returns
    -------
    pd.DataFrame with new engineered features and TomorrowAction.

    flowchart TD
        A[Raw Candles + Indicators] --> B[Pattern Confluence]
        B --> C[Wick Rejection Filter]
        C --> D[Momentum + Threshold Filters]
        D --> E[Higher Timeframe Bias Check]
        E --> F[Candidate Action (Buy/Sell/Hold)]
        F --> G[Alternating Filter (Buy → Sell → Buy)]
        G --> H[Shift Signal to Prior Row]
        H --> I[Add Confidence Score]
        I --> J[Diagnostics + Flags]
        J --> K[TomorrowAction Column]

    📑 Signal Engine Output Fields
    Column	Purpose
    BullishPatternCount	Rolling count of bullish candlestick patterns in last N bars.
    BearishPatternCount	Rolling count of bearish candlestick patterns in last N bars.
    PatternScore	Net score = Bullish − Bearish (positive → bullish bias).
    UpperWick / LowerWick	Length of candle shadows (used for rejection detection).
    Body	Absolute size of candle body.
    WickRejection	Boolean flag: candle shows strong wick rejection (potential reversal).
    Return	% price change from previous close.
    AvgReturn	Rolling average return (trend filter).
    Volatility	Rolling standard deviation of returns.
    MomentumZ	Z-scored return (momentum relative to volatility).
    BuyThresh / SellThresh	Dynamic thresholds for bullish/bearish action based on volatility.
    CandidateAction	Raw signal before filtering (Buy / Sell / Hold).
    Action	Filtered signal, alternating Buy → Sell → Buy enforced.
    TomorrowAction	Shifted forward prediction (what action should occur next).
    SignalStrength	Composite score (pattern + momentum magnitude).
    ActionConfidence	Normalized confidence [0–1] relative to strongest signal.
    SignalDuration	Unique ID for each signal segment (group of same action).
    ValidAction	Boolean, whether Action is Buy/Sell (vs Hold).
    HasValidSignal	Boolean, requires Action + TomorrowAction + SignalStrength to exist.
    '''


def finalize_signals_old2(df,
                          pattern_window=5,
                          bullish_patterns=None,
                          bearish_patterns=None,
                          momentum_factor=0.5,
                          candle_columns=None):
    """
    Full signal finalizer for pandas DataFrame.
    
    Returns:
        df with:
        - MomentumAction
        - PatternAction
        - CandleAction
        - CandidateAction (consolidated)
        - Action, TomorrowAction, ActionConfidence
        - SignalStrength, SignalDuration, ValidAction, HasValidSignal
    """
    # Forward-looking return and label
    df["TomorrowClose"] = df["Close"].shift(-1)
    df["TomorrowReturn"] = (df["TomorrowClose"] - df["Close"]) / df["Close"]
    df["TomorrowReturn"] = df["TomorrowReturn"].fillna(0)

    # Backward-looking momentum
    df["YesterdayReturn"] = df["Close"].pct_change().fillna(0)
    
    # -------------------------------
    # 1) Pattern Confluence
    # -------------------------------
    if bullish_patterns and bearish_patterns:
        df["BullishPatternCount"] = (
            df[bullish_patterns].rolling(window=pattern_window, min_periods=1).sum(axis=1)
        ).astype(float)
        df["BearishPatternCount"] = (
            df[bearish_patterns].rolling(window=pattern_window, min_periods=1).sum(axis=1)
        ).astype(float)
        df["PatternScore"] = (df["BullishPatternCount"] - df["BearishPatternCount"]).astype(float)
    else:
        df["BullishPatternCount"] = df["BearishPatternCount"] = df["PatternScore"] = 0.0

    # -------------------------------
    # 2) Wick Rejection Filter
    # -------------------------------
    df["UpperWick"] = df["High"] - df[["Close", "Open"]].max(axis=1)
    df["LowerWick"] = df[["Close", "Open"]].min(axis=1) - df["Low"]
    df["Body"] = (df["Close"] - df["Open"]).abs()
    df["WickRejection"] = (df["LowerWick"] > df["Body"] * 1.5) | (df["UpperWick"] > df["Body"] * 1.5)

    # -------------------------------
    # 3) Momentum & Thresholds
    # -------------------------------
    df["Return"] = df["Close"].pct_change()
    df["AvgReturn"] = df["Return"].rolling(10, min_periods=1).mean()
    df["Volatility"] = df["Return"].rolling(10, min_periods=1).std().fillna(0)
    df["MomentumZ"] = (df["Return"] - df["AvgReturn"]) / df["Volatility"].replace(0, 1)

    mean_momentum = df["MomentumZ"].mean()
    std_momentum = df["MomentumZ"].std()

    df["BuyThresh"] = mean_momentum + momentum_factor * std_momentum
    df["SellThresh"] = mean_momentum - momentum_factor * std_momentum

    # -------------------------------
    # 4) Per-signal Actions
    # -------------------------------
    # Momentum-based
    df["MomentumAction"] = "Hold"
    df.loc[df["MomentumZ"] > df["BuyThresh"], "MomentumAction"] = "Buy"
    df.loc[df["MomentumZ"] < df["SellThresh"], "MomentumAction"] = "Sell"

    # Pattern-score based
    df["PatternAction"] = "Hold"
    df.loc[df["PatternScore"] > 0, "PatternAction"] = "Buy"
    df.loc[df["PatternScore"] < 0, "PatternAction"] = "Sell"

    # Candlestick-based
    if candle_columns:
        def classify_candle(row):
            buy_signals = any([row.get(c, False) for c in candle_columns.get("Buy", [])])
            sell_signals = any([row.get(c, False) for c in candle_columns.get("Sell", [])])
            if buy_signals:
                return "Buy"
            elif sell_signals:
                return "Sell"
            else:
                return "Hold"
        df["CandleAction"] = df.apply(classify_candle, axis=1)
    else:
        df["CandleAction"] = "Hold"

    # -------------------------------
    # 5) Consolidated CandidateAction (majority vote)
    # -------------------------------
    df["CandidateAction"] = df[["MomentumAction", "PatternAction", "CandleAction"]].mode(axis=1)[0]

    # -------------------------------
    # 6) Alternating Filter
    # -------------------------------
    filtered = []
    last = None
    for action in df["CandidateAction"]:
        if action in ["Buy", "Sell"]:
            if action == last:
                filtered.append("Hold")
            else:
                filtered.append(action)
                last = action
        else:
            filtered.append("Hold")
    df["Action"] = filtered

    # -------------------------------
    # 7) Predictive Shift
    # -------------------------------
    df["TomorrowAction"] = df["Action"].shift(-1).fillna("Hold")

    # -------------------------------
    # 8) Confidence, Duration, Flags
    # -------------------------------
    df["SignalStrength"] = (df["PatternScore"].abs() + df["MomentumZ"].abs())
    max_s = df["SignalStrength"].max()
    df["ActionConfidence"] = (df["SignalStrength"] / max_s if max_s > 0 else 0).fillna(0)

    df["SignalDuration"] = (df["Action"] != df["Action"].shift()).cumsum()
    df["ValidAction"] = df["Action"].isin(["Buy", "Sell"])
    df["HasValidSignal"] = df[["Action", "TomorrowAction", "SignalStrength"]].notna().all(axis=1)

    # -------------------------------
    # 9) Force numeric columns to float (PySpark-safe)
    # -------------------------------
    numeric_cols = [
        "BullishPatternCount", "BearishPatternCount", "PatternScore",
        "UpperWick", "LowerWick", "Body",
        "Return", "AvgReturn", "Volatility", "MomentumZ",
        "BuyThresh", "SellThresh",
        "SignalStrength", "ActionConfidence", "SignalDuration"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)

    return df
def finalize_signals(df, pattern_window=5, momentum_factor=0.5,
                     bullish_patterns=None, bearish_patterns=None,
                     candle_columns=None):
    """
    Consolidates momentum + pattern + candle into a unified Action.
    """

    df = df.copy()

    # Tomorrow’s return (look-ahead)
    df["TomorrowClose"] = df["Close"].shift(-1)
    df["TomorrowReturn"] = (df["TomorrowClose"] - df["Close"]) / df["Close"]

    # --- Momentum scoring ---
    df["Return"] = df["Close"].pct_change().fillna(0)
    df["AvgReturn"] = df["Return"].rolling(10, min_periods=1).mean()
    df["Volatility"] = df["Return"].rolling(10, min_periods=1).std().replace(0, 1e-8)
    df["MomentumZ"] = (df["Return"] - df["AvgReturn"]) / df["Volatility"]
    mean_momentum = df["MomentumZ"].mean()
    std_momentum = df["MomentumZ"].std()
    df["BuyThresh"] = mean_momentum + momentum_factor * std_momentum
    df["SellThresh"] = mean_momentum - momentum_factor * std_momentum

    df["MomentumAction"] = "Hold"
    df.loc[df["MomentumZ"] > df["BuyThresh"], "MomentumAction"] = "Buy"
    df.loc[df["MomentumZ"] < df["SellThresh"], "MomentumAction"] = "Sell"

    if bullish_patterns and bearish_patterns:
        bull = df[bullish_patterns].rolling(pattern_window, min_periods=1).sum()
        bear = df[bearish_patterns].rolling(pattern_window, min_periods=1).sum()
        score = bull - bear
    else:
        score = pd.Series(0, index=df.index)  # <- make it same length as df
    df["PatternScore"] = score

    df["PatternAction"] = "Hold"
    df.loc[score > 0, "PatternAction"] = "Buy"
    df.loc[score < 0, "PatternAction"] = "Sell"

    # --- Candle scoring ---
    if candle_columns:
        def classify(row):
            if any(row.get(c, False) for c in candle_columns.get("Buy", [])):
                return "Buy"
            if any(row.get(c, False) for c in candle_columns.get("Sell", [])):
                return "Sell"
            return "Hold"
        df["CandleAction"] = df.apply(classify, axis=1)
    else:
        df["CandleAction"] = "Hold"

    # --- Majority vote ---
    df["CandidateAction"] = df[["MomentumAction", "PatternAction", "CandleAction"]].mode(axis=1)[0]

    # --- Alternate filter (no consecutive same signals) ---
    filtered, last = [], None
    for action in df["CandidateAction"]:
        if action in ["Buy", "Sell"]:
            if action == last:
                filtered.append("Hold")
            else:
                filtered.append(action)
                last = action
        else:
            filtered.append("Hold")
    df["Action"] = filtered

    # Predictive shift
    df["TomorrowAction"] = df["Action"].shift(-1).fillna("Hold")
    # -------------------------------
    # 8) Confidence, Duration, Flags
    # -------------------------------
    df["SignalStrength"] = (df["PatternScore"].abs() + df["MomentumZ"].abs())
    max_s = df["SignalStrength"].max()
    df["ActionConfidence"] = (df["SignalStrength"] / max_s if max_s > 0 else 0).fillna(0)

    df["SignalDuration"] = (df["Action"] != df["Action"].shift()).cumsum()
    df["ValidAction"] = df["Action"].isin(["Buy", "Sell"])
    df["HasValidSignal"] = df[["Action", "TomorrowAction", "SignalStrength"]].notna().all(axis=1)
    '''
    # --- Pattern metadata ---
    pattern_cols = [col for col in df.columns if col not in
                    ["Open", "High", "Low", "Close", "Volume",
                     "MomentumAction", "PatternAction", "CandleAction",
                     "CandidateAction", "Action", "TomorrowAction"]]

    df["PatternCount"] = df[pattern_cols].astype(int).sum(axis=1)
    df["PatternType"] = df[pattern_cols].astype(int).dot(range(1, len(pattern_cols)+1))
    '''
    return df

# -------------------------------
# 4️⃣ Action Classification
# -------------------------------
def classify_action(row):
    buy_signals = any([
        row.get("ValidHammer", False),
        row.get("ValidBullishEngulfing", False),
        row.get("ValidPiercingLine", False)
    ])
    sell_signals = any([
        row.get("ValidShootingStar", False),
        row.get("ValidBearishEngulfing", False),
        row.get("ValidDarkCloud", False)
    ])
    if buy_signals:
        return "Buy"
    elif sell_signals:
        return "Sell"
    else:
        return "Hold"

def add_batch_metadata(df, company_id, timeframe, ingest_ts=None):
    df = df.copy()  # Ensure safe memory layout

    # Default timestamp if not provided
    if ingest_ts is None:
        ingest_ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # Add metadata columns in one step
    metadata = {
        "BatchId": f"{company_id}_{ingest_ts}",
        "IngestedAt": ingest_ts,
        "CompanyId": company_id,
        "TimeFrame": timeframe
    }

    for key, value in metadata.items():
        df[key] = value

    return df

def generate_signal_columns(df, timeframe="Short"):
    """
    Automatically generates candlestick and bullish/bearish trend/momentum columns
    for the given timeframe, optimized for that trading horizon.
    Also returns a recommended momentum_factor and pattern_window.

    Parameters:
        df : pd.DataFrame
            Stock/candlestick data including validated signals and trend indicators
        timeframe : str
            One of ["Short", "Swing", "Long", "Daily"]

    Returns:
        candle_columns : dict
            {"Buy": [...], "Sell": [...]}
        trend_columns : dict
            {"Bullish": [...], "Bearish": [...]}
        momentum_factor : float
            Recommended scaling factor for momentum thresholds
        pattern_window : int
            Suggested rolling window for pattern aggregation
    """
    import re

    # -------------------------------
    # 1) Determine keyword sets per timeframe
    # -------------------------------
    if timeframe == "Short":
        buy_keywords  = ["hammer", "bullish", "piercing", "morning", "white", "marubozu", "tweezerbottom"]
        sell_keywords = ["shooting", "bearish", "dark", "evening", "black", "marubozu", "tweezertop"]
        momentum_factor = 0.05  # sensitive for fast signals
        pattern_window = 3
    elif timeframe == "Swing":
        buy_keywords  = ["hammer", "bullish", "piercing", "morning", "white"]
        sell_keywords = ["shooting", "bearish", "dark", "evening", "black"]
        momentum_factor = 0.1
        pattern_window = 5
    elif timeframe == "Long":
        buy_keywords  = ["bullish", "morning", "white", "threewhitesoldiers"]
        sell_keywords = ["bearish", "evening", "black", "threeblackcrows"]
        momentum_factor = 0.2
        pattern_window = 10
    else:  # Daily
        buy_keywords  = ["bullish", "morning", "white"]
        sell_keywords = ["bearish", "evening", "black"]
        momentum_factor = 0.15
        pattern_window = 7

    # -------------------------------
    # 2) Candlestick Columns
    # -------------------------------
    candle_cols = [col for col in df.columns if re.match(r"Valid.*", col)]
    candle_columns = {
        "Buy":  [col for col in candle_cols if any(k in col.lower() for k in buy_keywords)],
        "Sell": [col for col in candle_cols if any(k in col.lower() for k in sell_keywords)]
    }

    # -------------------------------
    # 3) Trend / Momentum Columns
    # -------------------------------
    bullish_pattern = r"(MomentumUp|ConfirmedUpTrend|UpTrend_MA)$"
    bearish_pattern = r"(MomentumDown|ConfirmedDownTrend|DownTrend_MA)$"

    bullish_cols = [col for col in df.columns if re.match(bullish_pattern, col)]
    bearish_cols = [col for col in df.columns if re.match(bearish_pattern, col)]

    trend_columns = {"Bullish": bullish_cols, "Bearish": bearish_cols}

    return candle_columns, trend_columns, momentum_factor, pattern_window




