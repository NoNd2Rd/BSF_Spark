import pandas as pd
import numpy as np
from datetime import datetime
import re
from bsf_settings import load_settings
from operator import itemgetter
import unicodedata
# -------------------------------
# Candlestick Pattern Parameters
# -------------------------------
def normalize(series, invert=False):
    """
    Normalize a Pandas Series to 0-1.
    If invert=True, higher values become lower.
    Handles the case where all values are the same or NaN.
    """
    min_val = series.min()
    max_val = series.max()
    
    if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
        # all values are NaN or the same; return 0 (or 1 if inverted)
        normalized = pd.Series(0, index=series.index)
    else:
        normalized = (series - min_val) / (max_val - min_val)
        if invert:
            normalized = 1 - normalized
            
    # Replace any remaining NaNs with 0
    normalized = normalized.fillna(0)
    
    return normalized

def get_candle_params_old(close_price):
    """
    Returns thresholds for candlestick patterns scaled by price magnitude.
    This ensures that patterns work for cheap stocks ($0.001) and expensive stocks ($100+).
    
    Returns:
        dict: Thresholds for doji, hammer, marubozu, small/long bodies, shadows, near edge, suspicious candles.
    """
    price = max(close_price, 1e-6)
    logp = np.log10(price)

    doji_thresh = np.clip(0.01 + 0.02 * (logp + 6) / 8, 0.01, 0.1)
    long_body = np.clip(0.3 + 0.3 * (logp + 6) / 8, 0.3, 0.6)
    small_body = np.clip(0.15 + 0.1 * (logp + 6) / 8, 0.15, 0.25)
    shadow_ratio = np.clip(1.2 + 0.8 * (logp + 6) / 8, 1.2, 2.0)
    near_edge = 0.25
    hammer_thresh = np.clip(0.15 + 0.1 * (logp + 6) / 8, 0.15, 0.25)
    marubozu_thresh = np.clip(0.03 + 0.02 * (logp + 6) / 8, 0.03, 0.05)
    rng_thresh = np.clip(1e-5 + 1e-4 * (logp + 6) / 8, 1e-5, 1e-4)

    return dict(
        doji_thresh=doji_thresh,
        long_body=long_body,
        small_body=small_body,
        shadow_ratio=shadow_ratio,
        near_edge=near_edge,
        hammer_thresh=hammer_thresh,
        marubozu_thresh=marubozu_thresh,
        rng_thresh=rng_thresh,
    )



def get_candle_params(close_price: float, user: int = 1):
    """
    Returns candlestick pattern thresholds scaled by price magnitude.
    Uses formula parameters from settings, defaults if no user override.
    """
    price = max(close_price, 1e-6)
    logp = np.log10(price)

    # Convert int user to string key
    user_settings = load_settings(user)["candle_params"]

    thresholds = {
        "doji_thresh": np.clip(user_settings["doji_base"] + user_settings["doji_scale"] * (logp + 6) / 8,
                               user_settings["doji_min"], user_settings["doji_max"]),
        "long_body": np.clip(user_settings["long_body_base"] + user_settings["long_body_scale"] * (logp + 6) / 8,
                             user_settings["long_body_min"], user_settings["long_body_max"]),
        "small_body": np.clip(user_settings["small_body_base"] + user_settings["small_body_scale"] * (logp + 6) / 8,
                              user_settings["small_body_min"], user_settings["small_body_max"]),
        "shadow_ratio": np.clip(user_settings["shadow_ratio_base"] + user_settings["shadow_ratio_scale"] * (logp + 6) / 8,
                                user_settings["shadow_ratio_min"], user_settings["shadow_ratio_max"]),
        "near_edge": user_settings["near_edge"],
        "highvol_spike": user_settings["highvol_spike"],
        "lowvol_dip": user_settings["lowvol_dip"],
        "hammer_thresh": np.clip(user_settings["hammer_base"] + user_settings["hammer_scale"] * (logp + 6) / 8,
                                 user_settings["hammer_min"], user_settings["hammer_max"]),
        "marubozu_thresh": np.clip(user_settings["marubozu_base"] + user_settings["marubozu_scale"] * (logp + 6) / 8,
                                   user_settings["marubozu_min"], user_settings["marubozu_max"]),
        "rng_thresh": np.clip(user_settings["rng_base"] + user_settings["rng_scale"] * (logp + 6) / 8,
                              user_settings["rng_min"], user_settings["rng_max"])
    }
    
    # Print nicely
    #print(f"\nCandlestick thresholds for user {user} (close price={price}):")
    #for k, v in thresholds.items():
    #    print(f"  {k:15}: {v:.6f}")

    return thresholds


# -------------------------------
# Pattern Window by Timeframe
# -------------------------------
 
def get_tf_window(timeframe="Short"):
    """
    Returns rolling window size for candlestick pattern aggregation.
    Shorter windows detect small swings; longer windows capture trends.
    """
    windows = {"Short": 3, "Swing": 5, "Long": 10, "Daily": 1}
    return windows.get(timeframe, 1)
 
# -------------------------------
# Identify Relevant Columns by Signal Type
# -------------------------------
def generate_signal_columns_old(df, timeframe="Short"):
    """
    Determines bullish/bearish candle and trend columns, adjusting momentum for penny stocks.
    Returns: candle_columns, trend_columns, momentum_factor
    """
    BULLISH_PATTERNS = [
        "Hammer", "InvertedHammer", "BullishEngulfing", "BullishHarami", "PiercingLine",
        "MorningStar", "ThreeWhiteSoldiers", "BullishMarubozu", "TweezerBottom",
        "DragonflyDoji", "RisingThreeMethods", "GapUp"
    ]
    BEARISH_PATTERNS = [
        "HangingMan", "ShootingStar", "BearishEngulfing", "BearishHarami", "DarkCloudCover",
        "EveningStar", "ThreeBlackCrows", "BearishMarubozu", "TweezerTop",
        "GravestoneDoji", "FallingThreeMethods", "GapDown"
    ]

    # Timeframe-specific keywords & momentum factor
    tf_params = {
        "Short": {"buy": ["hammer","bullish","piercing","morning","white","marubozu","tweezerbottom"],
                  "sell": ["shooting","bearish","dark","evening","black","marubozu","tweezertop"],
                  "momentum": 0.05},
        "Swing": {"buy": ["hammer","bullish","piercing","morning","white"],
                  "sell": ["shooting","bearish","dark","evening","black"],
                  "momentum": 0.1},
        "Long":  {"buy": ["bullish","morning","white","threewhitesoldiers"],
                  "sell": ["bearish","evening","black","threeblackcrows"],
                  "momentum": 0.2},
        "Daily": {"buy": ["bullish","morning","white"],
                  "sell": ["bearish","evening","black"],
                  "momentum": 0.15}
    }
    params = tf_params.get(timeframe, tf_params["Daily"])
    momentum_factor = params["momentum"]

    # Penny stock adjustment
    last_close = df["Close"].iloc[-1] if not df.empty else 0
    if last_close < 1.0:
        momentum_factor = max(momentum_factor * 0.2, 0.005)

    # Candle columns
    candle_cols = [col for col in df.columns if col.startswith("Valid")]
    candle_columns = {
        "Buy": [col for col in candle_cols if any(k in col.lower() for k in params["buy"])],
        "Sell": [col for col in candle_cols if any(k in col.lower() for k in params["sell"])]
    }

    # Trend columns
    bullish_cols = [col for col in df.columns if col in ["MomentumUp","ConfirmedUpTrend","UpTrend_MA"]]
    bearish_cols = [col for col in df.columns if col in ["MomentumDown","ConfirmedDownTrend","DownTrend_MA"]]
    trend_columns = {"Bullish": bullish_cols, "Bearish": bearish_cols}

    return candle_columns, trend_columns, momentum_factor
 

def generate_signal_columns(df, timeframe="Short", user: int = None):
    """
    Determines bullish/bearish candle and trend columns, adjusting momentum for penny stocks.
    Pulls all patterns and momentum factors from config, supports user overrides.
    
    Returns:
        candle_columns: dict with "Buy" and "Sell" lists
        trend_columns: dict with "Bullish" and "Bearish" lists
        momentum_factor: float
    """
    # Load user settings (merge defaults + optional user override)
    settings = load_settings(str(user))["signals"]

    # Timeframe-specific keywords & momentum
    tf_settings = settings["timeframes"].get(timeframe, settings["timeframes"]["Daily"])
    momentum_factor = tf_settings["momentum"]

    # Penny stock adjustment
    last_close = df["Close"].iloc[-1] if not df.empty else 0
    ps = settings["penny_stock_adjustment"]
    if last_close < ps["threshold"]:
        momentum_factor = max(momentum_factor * ps["factor"], ps["min_momentum"])

    # Candle columns
    candle_cols = [col for col in df.columns if col.startswith("Valid")]
    candle_columns = {
        "Buy": [col for col in candle_cols if any(k in col.lower() for k in tf_settings["buy"])],
        "Sell": [col for col in candle_cols if any(k in col.lower() for k in tf_settings["sell"])]
    }

    # Trend columns (static, can also move to config if needed)
    bullish_cols = [col for col in df.columns if col in ["MomentumUp", "ConfirmedUpTrend", "UpTrend_MA"]]
    bearish_cols = [col for col in df.columns if col in ["MomentumDown", "ConfirmedDownTrend", "DownTrend_MA"]]
    trend_columns = {"Bullish": bullish_cols, "Bearish": bearish_cols}

    return candle_columns, trend_columns, momentum_factor


# -------------------------------
# Add Candlestick Patterns
# -------------------------------

def add_candle_patterns(df, tf_window=5, , user: int = 1):
    """
    Efficient version of candlestick pattern detection for large datasets.
    Uses vectorized NumPy + precomputed rolling features.
    """
    o="Open"
    h="High"
    l="Low"
    c="Close"
    v="Volume"
    
    # Get the last Close price
    last_close = df[c].iloc[-1]  # c = "Close"
    candle_params = get_candle_params(last_close)
    # unpack the keys you care about
    doji_thresh, hammer_thresh, marubozu_thresh, long_body, small_body, shadow_ratio, near_edge, highvol_spike, lowvol_dip, rng_thresh = \
        itemgetter("doji_thresh", "hammer_thresh", "marubozu_thresh", "long_body", "small_body", "shadow_ratio", "near_edge", "highvol_spike", "lowvol_dip", "rng_thresh")(candle_params)

    
    roll_first = df[o].rolling(tf_window, min_periods=tf_window).apply(lambda x: x[0], raw=True)
    roll_last  = df[c].rolling(tf_window, min_periods=tf_window).apply(lambda x: x[-1], raw=True)
    roll_max = df[h].rolling(tf_window, min_periods=tf_window).max()
    roll_min = df[l].rolling(tf_window, min_periods=tf_window).min()

    O = roll_first.to_numpy(dtype=float)
    C = roll_last.to_numpy(dtype=float)
    H = roll_max.to_numpy(dtype=float)
    L = roll_min.to_numpy(dtype=float)

    av = df[v].rolling(20, min_periods=1).mean()
    hv = (df[v] > highvol_spike * av).to_numpy()
    lv = (df[v] < lowvol_dip * av).to_numpy()

    # Core candle stats
    ref_price = np.where(C != 0, C, np.nan)
    body = np.abs(C - O) / ref_price
    upsh = (H - np.maximum(O, C)) / ref_price
    dnsh = (np.minimum(O, C) - L) / ref_price
    rng = (H - L) / ref_price

    bull = (C > O)
    bear = (O > C)

    # Trend detection
    downtrend = (pd.Series(C).shift(1).rolling(tf_window)
                 .apply(lambda x: x.iloc[-1] < x.iloc[0], raw=False)).to_numpy(dtype=bool)
    uptrend = (pd.Series(C).shift(1).rolling(tf_window)
               .apply(lambda x: x.iloc[-1] > x.iloc[0], raw=False)).to_numpy(dtype=bool)

    # Initialize dict of patterns
    new_cols = {}

    new_cols["HighVolume"] = hv
    new_cols["LowVolume"] = lv
    new_cols["Doji"] = body <= doji_thresh * rng
    new_cols["Hammer"] = (dnsh >= shadow_ratio * body) & (upsh <= hammer_thresh * body) & (body > 0) & (body <= hammer_thresh * 2 * rng) & downtrend
    new_cols["InvertedHammer"] = (upsh >= shadow_ratio * body) & (dnsh <= hammer_thresh * body) & (body > 0) & (body <= hammer_thresh * 2 * rng) & downtrend
    new_cols["BullishMarubozu"] = bull & (body >= long_body * rng) & (upsh <= marubozu_thresh * rng) & (dnsh <= marubozu_thresh * rng)
    new_cols["BearishMarubozu"] = bear & (body >= long_body * rng) & (upsh <= marubozu_thresh * rng) & (dnsh <= marubozu_thresh * rng)
    new_cols["SuspiciousCandle"] = (rng <= rng_thresh) | (body <= rng_thresh)

    # Hanging man / shooting star
    new_cols["HangingMan"] = new_cols["Hammer"] & uptrend
    new_cols["ShootingStar"] = new_cols["InvertedHammer"] & uptrend

    # Shifts for multi-bar patterns
    O1, O2 = np.roll(O, 1), np.roll(O, 2)
    C1, C2 = np.roll(C, 1), np.roll(C, 2)
    H1, L1 = np.roll(H, 1), np.roll(L, 1)
    bull1, bull2 = np.roll(bull, 1), np.roll(bull, 2)
    bear1, bear2 = np.roll(bear, 1), np.roll(bear, 2)

    # Multi-bar patterns
    new_cols.update({
        "BullishEngulfing": (O1 > C1) & bull & (C >= O1) & (O <= C1),
        "BearishEngulfing": (C1 > O1) & bear & (O >= C1) & (C <= O1),
        "BullishHarami": (O1 > C1) & bull & (np.maximum(O, C) <= np.maximum(O1, C1)) & (np.minimum(O, C) >= np.minimum(O1, C1)),
        "BearishHarami": (C1 > O1) & bear & (np.maximum(O, C) <= np.maximum(O1, C1)) & (np.minimum(O, C) >= np.minimum(O1, C1)),
        "HaramiCross": new_cols["Doji"] & (np.maximum(O, C) <= np.maximum(O1, C1)) & (np.minimum(O, C) >= np.minimum(O1, C1)),
        "PiercingLine": (O1 > C1) & bull & (O < C1) & (C > (O1 + C1)/2) & (C < O1),
        "DarkCloudCover": (C1 > O1) & bear & (O > C1) & (C < (O1 + C1)/2) & (C > O1),
        "MorningStar": (O2 > C2) & (np.abs(C1 - O1) < np.abs(C2 - O2) * small_body) & bull & (C >= (O2 + C2)/2),
        "EveningStar": (C2 > O2) & (np.abs(C1 - O1) < np.abs(C2 - O2) * small_body) & bear & (C <= (O2 + C2)/2),
        "ThreeWhiteSoldiers": bull & bull1 & bull2 & (C > C1) & (C1 > C2),
        "ThreeBlackCrows": bear & bear1 & bear2 & (C < C1) & (C1 < C2),
        "TweezerTop": (H == H1) & bear & bull1,
        "TweezerBottom": (L == L1) & bull & bear1,
        "InsideBar": (H < H1) & (L > L1),
        "OutsideBar": (H > H1) & (L < L1),
        "NearHigh": H >= pd.Series(H).rolling(tf_window).max().to_numpy() * (1 - near_edge),
        "NearLow": L <= pd.Series(L).rolling(tf_window).min().to_numpy() * (1 + near_edge),
        "DragonflyDoji": (np.abs(C - O) <= doji_thresh * rng) & (H == C) & (L < O),
        "GravestoneDoji": (np.abs(C - O) <= doji_thresh * rng) & (L == C) & (H > O),
        "LongLeggedDoji": (np.abs(C - O) <= doji_thresh * rng) & (upsh > shadow_ratio * body) & (dnsh > shadow_ratio * body),
        "RisingThreeMethods": bull2 & bull1 & bull & (C1 < O2) & (C > C1),
        "FallingThreeMethods": bear2 & bear1 & bear & (C1 > O2) & (C < C1),
        "GapUp": O > H1,
        "GapDown": O < L1,
        "SpinningTop": (body <= small_body * rng) & (upsh >= body) & (dnsh >= body),
        "ClimacticCandle": rng > pd.Series(rng).rolling(tf_window).mean().to_numpy() * 2,
    })

    # Attach to df
    for col, arr in new_cols.items():
        df[col] = arr

    # Count patterns
    pattern_cols = [col for col in new_cols if df[col].dtype == bool]
    df["PatternCount"] = df[pattern_cols].sum(axis=1)

    df["PatternType"] = np.select(
        [df.get("BullishEngulfing", False), df.get("MorningStar", False),
         df.get("ThreeWhiteSoldiers", False), df.get("BullishMarubozu", False)],
        ["BullishEngulfing", "MorningStar", "ThreeWhiteSoldiers", "BullishMarubozu"],
        default="None"
    )

    return df.fillna(False)


def add_trend_filters(df, timeframe="Daily", user: int = 1):
    '''
    **pd.DataFrame**  
    Original DataFrame with the following additional columns:
    
    - **MA** : Rolling moving average  
    - **MA_slope** : % change of MA (normalized slope)  
    - **UpTrend_MA, DownTrend_MA** : Trend direction based on MA slope  
    - **RecentReturn** : % change over return window  
    - **UpTrend_Return, DownTrend_Return** : Trend based on % return  
    - **Volatility** : Rolling std deviation of % returns  
    - **LowVolatility, HighVolatility** : Volatility relative to median  
    - **ROC** : Rate of change over MA window  
    - **MomentumUp, MomentumDown** : Trend direction based on ROC thresholds  
    - **ConfirmedUpTrend, ConfirmedDownTrend** : Combined trend confirmation  
    
    '''


    # -------------------------------
    # Timeframe-specific rolling windows
    # -------------------------------
    '''
    profiles = {
        "Short": {"ma": 2,  "ret": 1,  "vol": 3,  "roc_thresh": 0.02, "slope_horizon": 1},
        "Swing": {"ma": 5,  "ret": 5,  "vol": 5,  "roc_thresh": 0.02, "slope_horizon": 5},
        "Long":  {"ma": 10, "ret": 10, "vol": 10, "roc_thresh": 0.02, "slope_horizon": 10},
        "Daily": {"ma": 7,  "ret": 1,  "vol": 5,  "roc_thresh": 0.02, "slope_horizon": 1}
    }
    '''
    c="Close" 
    
    settings = load_settings(user)["profiles"]   # merged defaults + overrides
    profiles = settings

    if timeframe not in profiles:
        raise ValueError(f"Invalid timeframe: {timeframe}. Must be one of {list(profiles.keys())}")

    params = profiles[timeframe]

    # -------------------------------
    # Moving Average and normalized slope
    # -------------------------------
    ma = df[c].rolling(params["ma"]).mean()
    ma_slope = ma.pct_change(params["slope_horizon"])

    # -------------------------------
    # Returns and Volatility
    # -------------------------------
    recent_ret = df[c].pct_change(params["ret"])
    vol = df[c].pct_change().rolling(params["vol"]).std()
    vol_med = vol.median()  # Median for dynamic volatility thresholds

    # -------------------------------
    # Rate of Change (ROC)
    # -------------------------------
    roc = df[c].pct_change(params["ma"])

    # -------------------------------
    # Add columns to DataFrame
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
# 1. Confirmed Signals
# -------------------------------
def add_confirmed_signals(df):
    """
    Generate validated candlestick signals based on trend context.
    E.g. a bullish engulfing is only 'valid' if prior trend was bearish.
    """


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
            "ValidHaramiCross": "UpTrend_MA",
            "ValidBullishHarami": "DownTrend_MA",
            "ValidBearishHarami": "UpTrend_MA"
        },
        "Continuation": {
            "ValidInsideBar": "UpTrend_MA",
            "ValidOutsideBar": "DownTrend_MA",
            "ValidRisingThreeMethods": "UpTrend_MA",
            "ValidFallingThreeMethods": "DownTrend_MA",
            "ValidGapUp": "UpTrend_MA",
            "ValidGapDown": "DownTrend_MA",
        },
        "Exhaustion": {
            "ValidSpinningTop": "UpTrend_MA",        # neutral but often marks slowdown
            "ValidClimacticCandle": "UpTrend_MA",    # blow-off top / climax
        }
    }

    for group_name, patterns in signal_groups.items():
        for valid_col, trend_col in patterns.items():
            raw_col = valid_col.replace("Valid", "")
            df[valid_col] = df.get(raw_col, False) & df.get(trend_col, False)
 
    # Bullish reversal (works best after downtrend + strong buying pressure)
    df["ValidDragonflyDoji"]= df["DragonflyDoji"] & df["DownTrend_MA"] & df["HighVolume"]
    # Bearish reversal (works best after uptrend + strong selling pressure)
    df["ValidGravestoneDoji"]= df["GravestoneDoji"] & df["UpTrend_MA"] & df["HighVolume"]
 

    return df


# -------------------------------
# 2. Signal Strength (count-based)
# -------------------------------
import pandas as pd

def add_signal_strength(df, user: int = 1, directional_groups=None):
    """
    Adds SignalStrength counts and percentages for each row.
    Efficient version for large datasets (~0.5M rows).
    """
    

    # 1️⃣ All valid signal columns
    valid_cols = [c for c in df.columns if c.startswith("Valid")]
    if not valid_cols:
        df["SignalStrength"] = 0
        df["BullishPctRaw"] = 0
        df["BearishPctRaw"] = 0
        df["BullishPctDirectional"] = 0
        df["BearishPctDirectional"] = 0
        return df

    # Convert all valid signal columns to int once
    valid_data = df[valid_cols].astype(int)

    # Count of all signals
    df["SignalStrength"] = valid_data.sum(axis=1)

    # Raw percentages
    bullish_cols = [c for c in valid_cols if c.startswith("ValidBullish")]
    bearish_cols = [c for c in valid_cols if c.startswith("ValidBearish")]

    df["BullishPctRaw"] = valid_data[bullish_cols].sum(axis=1) / df["SignalStrength"].replace(0,1)
    df["BearishPctRaw"] = valid_data[bearish_cols].sum(axis=1) / df["SignalStrength"].replace(0,1)

    # Directional percentages
    if directional_groups is None:
        directional_groups = ["Bullish", "Bearish"]

    directional_cols = [c for c in valid_cols if any(c.startswith(f"Valid{g}") for g in directional_groups)]
    directional_data = valid_data[directional_cols]

    directional_sum = directional_data.sum(axis=1).replace(0,1)

    bullish_dir_cols = [c for c in directional_cols if c.startswith("ValidBullish")]
    bearish_dir_cols = [c for c in directional_cols if c.startswith("ValidBearish")]

    df["BullishPctDirectional"] = directional_data[bullish_dir_cols].sum(axis=1) / directional_sum
    df["BearishPctDirectional"] = directional_data[bearish_dir_cols].sum(axis=1) / directional_sum

    return df


# -------------------------------
# 3. Finalize Signals
# -------------------------------
def finalize_signals(df, tf, tf_window=5, use_fundamentals=True, user: int = 1):
    """
    Consolidates momentum + pattern + candle into a unified Action.
    """
    candle_columns, trend_cols, momentum_factor = generate_signal_columns(df, tf)
    bullish_patterns=trend_cols["Bullish"]
    bearish_patterns=trend_cols["Bearish"]

    # --- Tomorrow’s return (look-ahead) ---
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

    # --- Pattern scoring ---
    if bullish_patterns and bearish_patterns:
        bull = df[bullish_patterns].rolling(tf_window, min_periods=1).sum().sum(axis=1)
        bear = df[bearish_patterns].rolling(tf_window, min_periods=1).sum().sum(axis=1)
    else:
        bull = pd.Series(0, index=df.index)
        bear = pd.Series(0, index=df.index)

    score = bull - bear
    df["BullScore"] = bull
    df["BearScore"] = bear
    df["PatternScore"] = score
    df["PatternScoreNorm"] = (bull - bear) / float(tf_window)

    threshold = 0.2
    df["PatternAction"] = "Hold"
    df.loc[df["PatternScoreNorm"] > threshold, "PatternAction"] = "Buy"
    df.loc[df["PatternScoreNorm"] < -threshold, "PatternAction"] = "Sell"

    # --- Candle scoring (vectorized) ---
    if candle_columns:
        buy_mask = df[candle_columns.get("Buy", [])].any(axis=1) if candle_columns.get("Buy") else pd.Series(False, index=df.index)
        sell_mask = df[candle_columns.get("Sell", [])].any(axis=1) if candle_columns.get("Sell") else pd.Series(False, index=df.index)

        df["CandleAction"] = "Hold"
        df.loc[buy_mask, "CandleAction"] = "Buy"
        df.loc[sell_mask & ~buy_mask, "CandleAction"] = "Sell"
    else:
        df["CandleAction"] = "Hold"

    # --- Majority vote ---
    df["CandidateAction"] = df[["MomentumAction", "PatternAction", "CandleAction"]].mode(axis=1)[0]

    # --- Filter consecutive Buy/Sell ---
    df["Action"] = df["CandidateAction"]
    consec_mask = (df["Action"] == df["Action"].shift(1)) & df["Action"].isin(["Buy", "Sell"])
    df.loc[consec_mask, "Action"] = "Hold"

    # --- Predictive shift (TomorrowAction) ---
    df["TomorrowAction"] = df["Action"].shift(-1)
    df["TomorrowActionSource"] = np.where(
        df["TomorrowAction"].isin(["Buy", "Sell"]),
        "NextAction(filtered)",
        np.where(df["CandidateAction"].shift(-1).isin(["Buy", "Sell"]),
                 "NextCandidate(unfiltered)",
                 "Hold(no_signal)")
    )
    df.iloc[-1, df.columns.get_loc("TomorrowAction")] = "Hold"
    df.iloc[-1, df.columns.get_loc("TomorrowActionSource")] = "LastRowHold"

    # --- Hybrid Signal Strength (optimized) ---
    valid_cols = [col for col in df.columns if col.startswith("Valid")]
    if valid_cols:
        valid_data = df[valid_cols].astype(int)
        count_strength = valid_data.sum(axis=1)
        max_count = count_strength.max()
        count_norm = count_strength / max_count if max_count > 0 else 0
    else:
        count_strength = count_norm = 0

    magnitude_strength = (df["PatternScore"].abs() + df["MomentumZ"].abs())
    max_mag = magnitude_strength.max()
    mag_norm = magnitude_strength / max_mag if max_mag > 0 else 0

    count_weight = 1.5
    momentum_weight = 1.0
    df["SignalStrengthHybrid"] = (count_weight * count_norm) + (momentum_weight * mag_norm)
    df["ActionConfidence"] = df["SignalStrengthHybrid"].fillna(0)

    # --- Directional hybrid ---
    bullish_sig = [
        "ValidHammer", "ValidBullishEngulfing", "ValidPiercingLine", "ValidMorningStar",
        "ValidThreeWhiteSoldiers", "ValidBullishMarubozu", "ValidTweezerBottom",
        "ValidBullishHarami", "ValidDragonflyDoji"
    ]
    bearish_sig = [
        "ValidShootingStar", "ValidBearishEngulfing", "ValidDarkCloud", "ValidEveningStar",
        "ValidThreeBlackCrows", "ValidBearishMarubozu", "ValidTweezerTop",
        "ValidBearishHarami", "ValidGravestoneDoji"
    ]

    bullish_cols = [c for c in valid_cols if any(c.startswith(sig) for sig in bullish_sig)]
    bearish_cols = [c for c in valid_cols if any(c.startswith(sig) for sig in bearish_sig)]

    bull_count = df[bullish_cols].astype(int).sum(axis=1) if bullish_cols else 0
    bear_count = df[bearish_cols].astype(int).sum(axis=1) if bearish_cols else 0

    bull_count_norm = bull_count / bull_count.max() if not isinstance(bull_count, int) and bull_count.max() > 0 else 0
    bear_count_norm = bear_count / bear_count.max() if not isinstance(bear_count, int) and bear_count.max() > 0 else 0

    mag_norm = magnitude_strength / magnitude_strength.max() if magnitude_strength.max() > 0 else 0
    '''
    count_weight = 1.0
    momentum_weight = 1.0
    df["BullishStrengthHybrid"] = (count_weight * bull_count_norm) + (momentum_weight * mag_norm)
    df["BearishStrengthHybrid"] = (count_weight * bear_count_norm) + (momentum_weight * mag_norm)

    df["SignalStrengthHybrid"] = df[["BullishStrengthHybrid", "BearishStrengthHybrid"]].max(axis=1)

    df["ActionConfidence"] = 0.0
    df.loc[df["Action"] == "Buy", "ActionConfidence"] = df.loc[df["Action"] == "Buy", "BullishStrengthHybrid"]
    df.loc[df["Action"] == "Sell", "ActionConfidence"] = df.loc[df["Action"] == "Sell", "BearishStrengthHybrid"]
    
    # --- Blend FundamentalScore ---
    if use_fundamentals and "FundamentalScore" in df.columns:
        df["ActionConfidence"] = 0.6 * df["SignalStrengthHybrid"] + 0.4 * df["FundamentalScore"]
    else:
        df["ActionConfidence"] = df["SignalStrengthHybrid"]
    '''
    count_weight = 1.0
    momentum_weight = 1.0
    
    # --- Technical hybrid signals ---
    df["BullishStrengthHybrid"] = (count_weight * bull_count_norm) + (momentum_weight * mag_norm)
    df["BearishStrengthHybrid"] = (count_weight * bear_count_norm) + (momentum_weight * mag_norm)
    
    # Maximum technical signal
    df["SignalStrengthHybrid"] = df[["BullishStrengthHybrid", "BearishStrengthHybrid"]].max(axis=1)
    
    # --- Initialize confidence with technical signal ---
    df["ActionConfidence"] = df["SignalStrengthHybrid"].copy()
    
    # --- Blend in fundamental score if available ---
    if use_fundamentals and "FundamentalScore" in df.columns and "FundamentalBad" in df.columns:
        # Technical vs Fundamental weights
        technical_weight = 0.6
        fundamental_weight = 0.4
    
        # Blend for all rows, scaled by FundamentalBad
        df["ActionConfidence"] = (
            technical_weight * df["SignalStrengthHybrid"] +
            fundamental_weight * df["FundamentalScore"]
        ) * (~df["FundamentalBad"].astype(bool))  # reduce confidence if bad
    
        # Directional adjustment for Buy/Sell
        df.loc[df["Action"] == "Buy", "ActionConfidence"] = (
            technical_weight * df.loc[df["Action"] == "Buy", "BullishStrengthHybrid"] +
            fundamental_weight * df.loc[df["Action"] == "Buy", "FundamentalScore"]
        ) * (~df.loc[df["Action"] == "Buy", "FundamentalBad"].astype(bool))
    
        df.loc[df["Action"] == "Sell", "ActionConfidence"] = (
            technical_weight * df.loc[df["Action"] == "Sell", "BearishStrengthHybrid"] +
            fundamental_weight * df.loc[df["Action"] == "Sell", "FundamentalScore"]
        ) * (~df.loc[df["Action"] == "Sell", "FundamentalBad"].astype(bool))

    
    # --- Optional normalization to 0-1 ---
    df["ActionConfidenceNorm"] = df["ActionConfidence"] / df["ActionConfidence"].max()

    # --- Duration & flags ---
    df["SignalDuration"] = (df["Action"] != df["Action"].shift()).cumsum()
    df["ValidAction"] = df["Action"].isin(["Buy", "Sell"])
    df["HasValidSignal"] = df[["Action", "TomorrowAction", "SignalStrengthHybrid"]].notna().all(axis=1)

    # --- Force numeric columns ---
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

# -------------------------------
# 4. Batch Metadata
# -------------------------------
def add_batch_metadata(df, company_id, timeframe, user: int = 1, ingest_ts=None):
    """Add BatchId, IngestedAt, CompanyId, TimeFrame metadata."""

    if ingest_ts is None:
        ingest_ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    df["BatchId"] = f"{company_id}_{ingest_ts}"
    df["IngestedAt"] = ingest_ts
    df["CompanyId"] = company_id
    df["TimeFrame"] = timeframe
    df["UserId"] = user
    return df

def compute_fundamental_score(df):
    fundamentals = {}

    # --- Valuation (invert ratios, lower is better) ---
    fundamentals["PE"]  = normalize(df.get("PeRatio", pd.Series(0, index=df.index)), invert=True)
    fundamentals["PEG"] = normalize(df.get("PegRatio", pd.Series(0, index=df.index)), invert=True)
    fundamentals["PB"]  = normalize(df.get("PbRatio", pd.Series(0, index=df.index)), invert=True)

    # --- Profitability ---
    fundamentals["ROE"]        = normalize(df.get("ReturnOnEquity", pd.Series(0, index=df.index)))
    fundamentals["GrossMargin"] = normalize(df.get("GrossMarginTTM", pd.Series(0, index=df.index)))
    fundamentals["NetMargin"]   = normalize(df.get("NetProfitMarginTTM", pd.Series(0, index=df.index)))

    # --- Debt & Liquidity ---
    fundamentals["DebtEq"]     = normalize(df.get("TotalDebtToEquity", pd.Series(0, index=df.index)), invert=True)
    fundamentals["CurrentRat"] = normalize(df.get("CurrentRatio", pd.Series(0, index=df.index)))
    fundamentals["IntCover"]   = normalize(df.get("InterestCoverage", pd.Series(0, index=df.index)))

    # --- Growth ---
    fundamentals["EPSChange"] = normalize(df.get("EpsChangeYear", pd.Series(0, index=df.index)))
    fundamentals["RevChange"] = normalize(df.get("RevChangeYear", pd.Series(0, index=df.index)))

    # --- Sentiment / Risk ---
    fundamentals["Beta"]      = normalize(df.get("Beta", pd.Series(0, index=df.index)), invert=True)
    fundamentals["ShortInt"] = normalize(df.get("ShortIntToFloat", pd.Series(0, index=df.index)))

    # --- Combine weights ---
    df["FundamentalScore"] = (
        0.2 * (fundamentals["PE"] + fundamentals["PEG"] + fundamentals["PB"]) / 3 +
        0.3 * (fundamentals["ROE"] + fundamentals["GrossMargin"] + fundamentals["NetMargin"]) / 3 +
        0.2 * (fundamentals["DebtEq"] + fundamentals["CurrentRat"] + fundamentals["IntCover"]) / 3 +
        0.2 * (fundamentals["EPSChange"] + fundamentals["RevChange"]) / 2 +
        0.1 * (fundamentals["Beta"] + fundamentals["ShortInt"]) / 2
    )

    # --- Flag if any component is zero or NaN ---
    all_components = pd.concat(fundamentals.values(), axis=1)
    df["FundamentalBad"] = (all_components == 0).any(axis=1) | all_components.isna().any(axis=1)

    return df
