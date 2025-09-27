import pandas as pd
import numpy as np
from datetime import datetime
import re

# -------------------------------
# Add Candle Patterns
# -------------------------------
import numpy as np

import numpy as np

def get_candle_params(close_price):
    """
    Return candlestick pattern thresholds scaled smoothly by price magnitude.
    Works for stocks from $100 down to $0.001.
    All thresholds are relative ratios for normalized candlestick calculations.
    """
    # Avoid log(0)
    price = max(close_price, 1e-6)
    logp = np.log10(price)  # roughly -6 for $0.001, +2 for $100

    # Doji threshold: smaller for cheap stocks, larger for expensive stocks
    doji_thresh = np.clip(0.01 + 0.02 * (logp + 6) / 8, 0.01, 0.1)

    # Long body threshold
    long_body = np.clip(0.3 + 0.3 * (logp + 6) / 8, 0.3, 0.6)

    # Small body for multi-bar patterns
    small_body = np.clip(0.15 + 0.1 * (logp + 6) / 8, 0.15, 0.25)

    # Shadow ratio for Hammer / Inverted Hammer
    shadow_ratio = np.clip(1.2 + 0.8 * (logp + 6) / 8, 1.2, 2.0)

    # Near high/low detection
    near_edge = 0.25  # keep constant

    # Hammer upper shadow threshold
    hammer_thresh = np.clip(0.15 + 0.1 * (logp + 6) / 8, 0.15, 0.25)

    # Marubozu shadow threshold
    marubozu_thresh = np.clip(0.03 + 0.02 * (logp + 6) / 8, 0.03, 0.05)

    # Suspicious candle threshold (range/body too small)
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



def get_candle_params_old(close_price):
    """
    Return a dictionary of candlestick pattern thresholds tuned to price magnitude.
    These can be passed directly to add_candle_patterns(**params)
    """
    if close_price > 50:
        return dict(
            doji_thresh=0.1,
            long_body=0.6,
            small_body=0.25,
            shadow_ratio=2.0,
            near_edge=0.25,
            hammer_thresh=0.25,
            marubozu_thresh=0.05
        )
    elif close_price > 1:
        return dict(
            doji_thresh=0.05,
            long_body=0.5,
            small_body=0.2,
            shadow_ratio=1.5,
            near_edge=0.25,
            hammer_thresh=0.2,
            marubozu_thresh=0.05
        )
    else:
        return dict(
            doji_thresh=0.01,
            long_body=0.3,
            small_body=0.15,
            shadow_ratio=1.2,
            near_edge=0.25,
            hammer_thresh=0.15,
            marubozu_thresh=0.03
        )


def get_pattern_window(timeframe="Short"):
    if timeframe == "Short":
        pattern_window = 3
    
    elif timeframe == "Swing":
        pattern_window = 5
    
    elif timeframe == "Long":
        pattern_window = 10
    
    else:  # Daily
        pattern_window = 1
    
    return  pattern_window
    
def generate_signal_columns(df, timeframe="Short"):
    # -------------------------------
    # 1) Determine keyword sets per timeframe
    # -------------------------------
    if timeframe == "Short":
        buy_keywords  = ["hammer", "bullish", "piercing", "morning", "white", "marubozu", "tweezerbottom"]
        sell_keywords = ["shooting", "bearish", "dark", "evening", "black", "marubozu", "tweezertop"]
        momentum_factor = 0.05

    
    elif timeframe == "Swing":
        buy_keywords  = ["hammer", "bullish", "piercing", "morning", "white"]
        sell_keywords = ["shooting", "bearish", "dark", "evening", "black"]
        momentum_factor = 0.1

    
    elif timeframe == "Long":
        buy_keywords  = ["bullish", "morning", "white", "threewhitesoldiers"]
        sell_keywords = ["bearish", "evening", "black", "threeblackcrows"]
        momentum_factor = 0.2

    
    else:  # Daily
        buy_keywords  = ["bullish", "morning", "white"]
        sell_keywords = ["bearish", "evening", "black"]
        momentum_factor = 0.15

    
    # --- Penny stock sensitivity override ---
    last_close = df["Close"].iloc[-1]
    if last_close < 1.00:  # treat as penny/low-priced
        # Scale down momentum_factor to make thresholds easier to cross
        momentum_factor = max(momentum_factor * 0.2, 0.005)


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

    return candle_columns, trend_columns, momentum_factor
# -------------------------------
# Add Candlestick Patterns
# -------------------------------
def add_candle_patterns(df, o="Open", h="High", l="Low", c="Close", pattern_window=5,
                        doji_thresh=0.1, hammer_thresh = 0.25, marubozu_thresh = 0.05, long_body=0.6, small_body=0.25,
                        shadow_ratio=2.0, near_edge=0.25, rng_thresh=1e-4):
    df = df.copy()
    #O, H, L, C = df[o].astype(float), df[h].astype(float), df[l].astype(float), df[c].astype(float)
    # Aggregate into "pattern_window"-length candles
    O = df[o].rolling(pattern_window, min_periods=pattern_window).apply(lambda x: x.iloc[0], raw=False).astype(float)
    C = df[c].rolling(pattern_window, min_periods=pattern_window).apply(lambda x: x.iloc[-1], raw=False).astype(float)
    H = df[h].rolling(pattern_window, min_periods=pattern_window).max().astype(float)
    L = df[l].rolling(pattern_window, min_periods=pattern_window).min().astype(float)
    # -------------------------------
    # Anatomy
    # -------------------------------
    '''
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
    '''
    ref_price = C.replace(0, np.nan)
    body = (C - O).abs() / ref_price
    upsh = (H - np.maximum(O, C)) / ref_price
    dnsh = (np.minimum(O, C) - L) / ref_price
    rng = (H - L) / ref_price
    bull, bear = C > O, O > C
    new_cols = {}

    # -------------------------------
    # Single-bar patterns
    # -------------------------------
    #bull = df["Close"] > df["Open"]
    #bear = df["Open"] > df["Close"]
    # Optional trend filters (last 3 closes)

    # Shift previous closes, rolling over the pattern window
    downtrend = (
        C.shift(1)
          .rolling(pattern_window, min_periods=pattern_window)
          .apply(lambda x: x.iloc[-1] < x.iloc[0], raw=False)
          .astype(bool)
    )
    
    uptrend = (
        C.shift(1)
          .rolling(pattern_window, min_periods=pattern_window)
          .apply(lambda x: x.iloc[-1] > x.iloc[0], raw=False)
          .astype(bool)
    )
    #new_cols["Doji"] = body <= doji_thresh * rng

    #new_cols["Hammer"] = (dnsh >= shadow_ratio * body) & (upsh <= 0.2 * body) & bull
    #new_cols["InvertedHammer"] = (upsh >= shadow_ratio * body) & (dnsh <= 0.2 * body)
    # -------------------------------
    # Doji: very small body relative to range
    # -------------------------------
    new_cols["Doji"] = (body <= doji_thresh * rng)
    
    #threshold = 0.25  # adjust sensitivity
    # -------------------------------
    # Hammer: long lower shadow, small body, tiny upper shadow
    # -------------------------------
    new_cols["Hammer"] = (
        (dnsh >= shadow_ratio * body) &
        (upsh <= hammer_thresh * body) &
        (body > 0) &
        (body <= hammer_thresh * 2 * rng) &
        downtrend
    )
    
    # Hanging Man: same as Hammer but after uptrend
    new_cols["HangingMan"] = new_cols["Hammer"] & uptrend

    # -------------------------------
    # Inverted Hammer: long upper shadow, small body, tiny lower shadow
    # -------------------------------
    new_cols["InvertedHammer"] = (
        (upsh >= shadow_ratio * body) &
        (dnsh <= hammer_thresh * body) &
        (body > 0) &
        (body <= hammer_thresh * 2  * rng) &
        downtrend
    )

    # Shooting Star: same as Inverted Hammer but after uptrend
    new_cols["ShootingStar"] = new_cols["InvertedHammer"] & uptrend
    
    # -------------------------------
    # Marubozu: long body, very small shadows
    # -------------------------------
    #marubozu_threshold = 0.05  # adjust sensitivity
    new_cols["BullishMarubozu"] = (
        bull &
        (body >= long_body * rng) &
        (upsh <= marubozu_thresh * rng) &
        (dnsh <= marubozu_thresh * rng)
    )
    new_cols["BearishMarubozu"] = (
        bear &
        (body >= long_body * rng) &
        (upsh <= marubozu_thresh * rng) &
        (dnsh <= marubozu_thresh * rng)
    )

    #new_cols["ShootingStar"] = new_cols["InvertedHammer"]
    #new_cols["BullishMarubozu"] = (body >= long_body * rng) & (upsh <= 0.05 * rng) & (dnsh <= 0.05 * rng) & bull
    #new_cols["BearishMarubozu"] = (body >= long_body * rng) & (upsh <= 0.05 * rng) & (dnsh <= 0.05 * rng) & bear
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
    # Ensure all values are numeric, convert bool/None/NaN to float
    #df["BodyRel"] = pd.to_numeric(body, errors="coerce").astype(float)
    #df["UpperShadowRel"] = pd.to_numeric(upsh, errors="coerce").astype(float)
    #df["LowerShadowRel"] = pd.to_numeric(dnsh, errors="coerce").astype(float)
    #df["RangeRel"] = pd.to_numeric(rng, errors="coerce").astype(float)

    return df.fillna(False)
    


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


############################################################## not in CLEAN

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
        # Rolling sums across bullish patterns
        bull = (
            df[bullish_patterns]
            .rolling(pattern_window, min_periods=1)
            .sum()
            .sum(axis=1)  # collapse across pattern columns
        )
        # Rolling sums across bearish patterns
        bear = (
            df[bearish_patterns]
            .rolling(pattern_window, min_periods=1)
            .sum()
            .sum(axis=1)
        )
    else:
        bull = pd.Series(0, index=df.index)
        bear = pd.Series(0, index=df.index)

    score = bull - bear
    # Raw scores
    df["BullScore"] = bull
    df["BearScore"] = bear
    df["PatternScore"] = score
    # Normalized score: between -1 and 1
    df["PatternScoreNorm"] = (bull - bear) / float(pattern_window)
    # Action column based on normalized score
    threshold = 0.2  # adjust sensitivity
    df["PatternAction"] = "Hold"
    df.loc[df["PatternScoreNorm"] > threshold, "PatternAction"] = "Buy"
    df.loc[df["PatternScoreNorm"] < -threshold, "PatternAction"] = "Sell"

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
    '''
    # --- Majority vote ---
    df["CandidateAction"] = df[["MomentumAction", "PatternAction", "CandleAction"]].mode(axis=1)[0]

    # --- Alternate filter (no consecutive same signals) ---
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

    # --- Predictive shift with traceability ---
    tomorrow_actions = []
    tomorrow_sources = []
    for i in range(len(df)):
        if i < len(df) - 1:
            next_action = df.loc[df.index[i+1], "Action"]
            if next_action in ["Buy", "Sell"]:
                tomorrow_actions.append(next_action)
                tomorrow_sources.append("NextAction(filtered)")
            else:
                cand_next = df.loc[df.index[i+1], "CandidateAction"]
                if cand_next in ["Buy", "Sell"]:
                    tomorrow_actions.append(cand_next)
                    tomorrow_sources.append("NextCandidate(unfiltered)")
                else:
                    tomorrow_actions.append("Hold")
                    tomorrow_sources.append("Hold(no_signal)")
        else:
            tomorrow_actions.append(df.loc[df.index[i], "Action"])
            tomorrow_sources.append("LastRowSelf")

    df["TomorrowAction"] = tomorrow_actions
    df["TomorrowActionSource"] = tomorrow_sources
    '''
    import pandas as pd
    import numpy as np
    
    # -------------------------------
    # 1️⃣ Majority vote
    # -------------------------------
    df["CandidateAction"] = df[["MomentumAction", "PatternAction", "CandleAction"]].mode(axis=1)[0]
    
    # -------------------------------
    # 2️⃣ Filter consecutive Buy/Sell
    # -------------------------------
    # Start with CandidateAction
    df["Action"] = df["CandidateAction"]
    
    # Identify consecutive Buy/Sell
    consec_mask = (df["Action"] == df["Action"].shift(1)) & df["Action"].isin(["Buy", "Sell"])
    
    # Replace consecutive duplicates with Hold
    df.loc[consec_mask, "Action"] = "Hold"
    
    # -------------------------------
    # 3️⃣ Predictive shift (TomorrowAction)
    # -------------------------------
    # Default tomorrow action = next filtered action
    df["TomorrowAction"] = df["Action"].shift(-1)
    
    # Traceability: check if next action comes from filtered Action or CandidateAction
    df["TomorrowActionSource"] = np.where(
        df["TomorrowAction"].isin(["Buy", "Sell"]),
        "NextAction(filtered)",
        # fallback to CandidateAction if it has a signal
        np.where(
            df["CandidateAction"].shift(-1).isin(["Buy", "Sell"]),
            "NextCandidate(unfiltered)",
            "Hold(no_signal)"
        )
    )
    
    # For last row, there is no "tomorrow", so we set Hold
    df.iloc[-1, df.columns.get_loc("TomorrowAction")] = "Hold"
    df.iloc[-1, df.columns.get_loc("TomorrowActionSource")] = "LastRowHold"
    
    # -------------------------------
    # ✅ df now has:
    # CandidateAction: majority vote
    # Action: filtered action (no consecutive Buy/Sell)
    # TomorrowAction: next action traceable
    # TomorrowActionSource: origin of tomorrow's signal
    # -------------------------------

    # -------------------------------
    # Hybrid Signal Strength
    # -------------------------------
    # Count-based: how many confirmed patterns fire
    valid_cols = [col for col in df.columns if col.startswith("Valid")]
    count_strength = df[valid_cols].astype(int).sum(axis=1) if valid_cols else 0
    max_count = count_strength.max() if not isinstance(count_strength, int) else 0
    count_norm = count_strength / max_count if max_count > 0 else 0

    # Magnitude-based: your existing metric
    magnitude_strength = (df["PatternScore"].abs() + df["MomentumZ"].abs())
    max_mag = magnitude_strength.max()
    mag_norm = magnitude_strength / max_mag if max_mag > 0 else 0

    # Weighted combination (tweak weights as needed)
    count_weight = 1.0
    momentum_weight = 1.0
    df["SignalStrengthHybrid"] = (count_weight * count_norm) + (momentum_weight * mag_norm)

    # Use hybrid for ActionConfidence
    df["ActionConfidence"] = df["SignalStrengthHybrid"].fillna(0)

    # -------------------------------
    # Hybrid Signal Strength (Directional)
    # -------------------------------

    # 1. Identify confirmed signal columns by group
    bullish_cols = [c for c in df.columns if c.startswith("Valid") and any(
        c.startswith(sig) for sig in [
            "ValidHammer", "ValidBullishEngulfing", "ValidPiercingLine",
            "ValidMorningStar", "ValidThreeWhiteSoldiers", "ValidBullishMarubozu",
            "ValidTweezerBottom", "ValidBullishHarami"
        ]
    )]

    bearish_cols = [c for c in df.columns if c.startswith("Valid") and any(
        c.startswith(sig) for sig in [
            "ValidShootingStar", "ValidBearishEngulfing", "ValidDarkCloud",
            "ValidEveningStar", "ValidThreeBlackCrows", "ValidBearishMarubozu",
            "ValidTweezerTop", "ValidBearishHarami"
        ]
    )]

    # 2. Count-based strength
    bull_count = df[bullish_cols].astype(int).sum(axis=1) if bullish_cols else 0
    bear_count = df[bearish_cols].astype(int).sum(axis=1) if bearish_cols else 0

    bull_count_norm = bull_count / bull_count.max() if not isinstance(bull_count, int) and bull_count.max() > 0 else 0
    bear_count_norm = bear_count / bear_count.max() if not isinstance(bear_count, int) and bear_count.max() > 0 else 0

    # 3. Magnitude-based strength (same formula for both, but sign matters)
    magnitude_strength = (df["PatternScore"].abs() + df["MomentumZ"].abs())
    mag_norm = magnitude_strength / magnitude_strength.max() if magnitude_strength.max() > 0 else 0

    # 4. Weighted hybrid scores
    count_weight = 1.0
    momentum_weight = 1.0
    df["BullishStrengthHybrid"] = (count_weight * bull_count_norm) + (momentum_weight * mag_norm)
    df["BearishStrengthHybrid"] = (count_weight * bear_count_norm) + (momentum_weight * mag_norm)

    # 5. Overall hybrid score (max of the two)
    df["SignalStrengthHybrid"] = df[["BullishStrengthHybrid", "BearishStrengthHybrid"]].max(axis=1)

    # 6. ActionConfidence aligned with Action direction
    df["ActionConfidence"] = 0.0
    df.loc[df["Action"] == "Buy", "ActionConfidence"] = df.loc[df["Action"] == "Buy", "BullishStrengthHybrid"]
    df.loc[df["Action"] == "Sell", "ActionConfidence"] = df.loc[df["Action"] == "Sell", "BearishStrengthHybrid"]


    
    # Duration & flags
    df["SignalDuration"] = (df["Action"] != df["Action"].shift()).cumsum()
    df["ValidAction"] = df["Action"].isin(["Buy", "Sell"])
    df["HasValidSignal"] = df[["Action", "TomorrowAction", "SignalStrengthHybrid"]].notna().all(axis=1)

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






