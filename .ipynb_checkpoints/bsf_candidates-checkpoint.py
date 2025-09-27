import pandas as pd
import numpy as np
from datetime import datetime
import re

# -------------------------------
# Candlestick Pattern Parameters
# -------------------------------
def get_candle_params(close_price):
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

# -------------------------------
# Pattern Window by Timeframe
# -------------------------------
def get_pattern_window(timeframe="Short"):
    """
    Returns rolling window size for candlestick pattern aggregation.
    Shorter windows detect small swings; longer windows capture trends.
    """
    windows = {"Short": 3, "Swing": 5, "Long": 10, "Daily": 1}
    return windows.get(timeframe, 1)

# -------------------------------
# Identify Relevant Columns by Signal Type
# -------------------------------
def generate_signal_columns(df, timeframe="Short"):
    """
    Determines which columns correspond to bullish/bearish candlestick patterns and trends.
    Penny stock adjustments scale momentum to avoid over-sensitive triggers.
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

    if timeframe == "Short":
        buy_keywords = ["hammer", "bullish", "piercing", "morning", "white", "marubozu", "tweezerbottom"]
        sell_keywords = ["shooting", "bearish", "dark", "evening", "black", "marubozu", "tweezertop"]
        momentum_factor = 0.05
    elif timeframe == "Swing":
        buy_keywords = ["hammer", "bullish", "piercing", "morning", "white"]
        sell_keywords = ["shooting", "bearish", "dark", "evening", "black"]
        momentum_factor = 0.1
    elif timeframe == "Long":
        buy_keywords = ["bullish", "morning", "white", "threewhitesoldiers"]
        sell_keywords = ["bearish", "evening", "black", "threeblackcrows"]
        momentum_factor = 0.2
    else:
        buy_keywords = ["bullish", "morning", "white"]
        sell_keywords = ["bearish", "evening", "black"]
        momentum_factor = 0.15

    last_close = df["Close"].iloc[-1]
    if last_close < 1.00:
        momentum_factor = max(momentum_factor * 0.2, 0.005)

    candle_cols = [col for col in df.columns if re.match(r"Valid.*", col)]
    candle_columns = {
        "Buy": [col for col in candle_cols if any(k in col.lower() for k in buy_keywords)],
        "Sell": [col for col in candle_cols if any(k in col.lower() for k in sell_keywords)]
    }

    bullish_cols = [col for col in df.columns if re.match(r"(MomentumUp|ConfirmedUpTrend|UpTrend_MA)$", col)]
    bearish_cols = [col for col in df.columns if re.match(r"(MomentumDown|ConfirmedDownTrend|DownTrend_MA)$", col)]
    trend_columns = {"Bullish": bullish_cols, "Bearish": bearish_cols}

    return candle_columns, trend_columns, momentum_factor

# -------------------------------
# Add Candlestick Patterns
# -------------------------------
'''
✅ Explanation of Markers (Why they exist):
Doji – Identifies indecision candles with very small bodies relative to range. Useful to spot potential reversals.
Hammer / HangingMan – Long lower shadow, small body; bullish after downtrend (Hammer) or bearish after uptrend (HangingMan).
InvertedHammer / ShootingStar – Long upper shadow, small body; bullish after downtrend or bearish after uptrend.
Bullish/Bearish Marubozu – Candles with large bodies and minimal shadows, indicating strong continuation.
SuspiciousCandle – Detects tiny range or tiny body, potentially unreliable or “noise” candles.
Engulfing / Harami / HaramiCross / Piercing / DarkCloud / Morning / Evening Star / Three White / Three Black – Multi-bar reversal patterns signaling trend changes.
TweezerTop / TweezerBottom – Exact high/low matching prior candle; potential reversal signals.
InsideBar / OutsideBar – Measures price compression or expansion for continuation/reversal.
NearHigh / NearLow – Rolling window high/low detection, often used to validate momentum continuation or reversal points.
PatternCount / PatternType – Summary metrics: how many patterns fire and which key type is detected, useful for combined signal scoring.

| **Pattern**              | **Meaning / Shape**                                                 | **Why It’s Valuable**                               | **Typical Context / Use**                                                             |
| ------------------------ | ------------------------------------------------------------------- | --------------------------------------------------- | ------------------------------------------------------------------------------------- |
| **Doji**                 | Very small body relative to range                                   | Flags indecision; neither bulls nor bears dominate  | Precedes potential reversal or consolidation; watch following candle for confirmation |
| **Hammer**               | Long lower shadow, small body, after downtrend                      | Indicates potential bullish reversal                | Use to identify support; stronger when combined with low price volume                 |
| **Hanging Man**          | Long lower shadow, small body, after uptrend                        | Warns of potential bearish reversal                 | Acts as early warning; often confirmed by next candle direction                       |
| **Inverted Hammer**      | Long upper shadow, small body, after downtrend                      | Indicates potential bullish reversal                | Signals failed selling pressure; follow-up candle confirms trend                      |
| **Shooting Star**        | Long upper shadow, small body, after uptrend                        | Indicates potential bearish reversal                | Shows failed buying pressure; strong reversal signal when confirmed                   |
| **Bullish Marubozu**     | Large body, minimal shadows, bullish                                | Confirms strong buying pressure                     | Trend continuation signal; high confidence bullish candle                             |
| **Bearish Marubozu**     | Large body, minimal shadows, bearish                                | Confirms strong selling pressure                    | Trend continuation signal; high confidence bearish candle                             |
| **SuspiciousCandle**     | Tiny body or tiny range                                             | Flags unreliable / “noise” candles                  | Helps filter low-confidence patterns that may distort scoring                         |
| **Bullish Engulfing**    | Current candle fully engulfs prior bearish candle                   | Powerful bullish reversal indicator                 | Use after downtrend; confirms buyer dominance                                         |
| **Bearish Engulfing**    | Current candle fully engulfs prior bullish candle                   | Powerful bearish reversal indicator                 | Use after uptrend; confirms seller dominance                                          |
| **Bullish Harami**       | Small bullish candle within prior bearish body                      | Subtle bullish reversal signal                      | Early trend shift detection; needs confirmation from next candle                      |
| **Bearish Harami**       | Small bearish candle within prior bullish body                      | Subtle bearish reversal signal                      | Early trend shift detection; watch next candle                                        |
| **Harami Cross**         | Doji within prior candle’s body                                     | Strong indecision reversal pattern                  | Confirms potential reversal; often used with trend filters                            |
| **Piercing Line**        | Bullish second candle closes above midpoint of prior bearish candle | Early bullish reversal                              | Used for entry signals with defined risk levels                                       |
| **Dark Cloud Cover**     | Bearish second candle closes below midpoint of prior bullish candle | Early bearish reversal                              | Useful for early exits or short entries                                               |
| **Morning Star**         | Three-bar bullish reversal (small body between two larger ones)     | Confirms trend reversal with multi-bar confirmation | Reliable reversal pattern after downtrend                                             |
| **Evening Star**         | Three-bar bearish reversal (small body between two larger ones)     | Confirms trend reversal with multi-bar confirmation | Reliable reversal pattern after uptrend                                               |
| **Three White Soldiers** | Three consecutive bullish candles with rising closes                | Strong trend continuation / bullish confirmation    | Confirms strong upward momentum                                                       |
| **Three Black Crows**    | Three consecutive bearish candles with falling closes               | Strong trend continuation / bearish confirmation    | Confirms strong downward momentum                                                     |
| **Tweezer Top**          | Exact high matches previous candle’s high                           | Potential bearish reversal                          | Marks local resistance / reversal point                                               |
| **Tweezer Bottom**       | Exact low matches previous candle’s low                             | Potential bullish reversal                          | Marks local support / reversal point                                                  |
| **Inside Bar**           | Current high < prior high, current low > prior low                  | Indicates price compression / potential breakout    | Signals continuation or setup for breakout trade                                      |
| **Outside Bar**          | Current high > prior high, current low < prior low                  | Indicates expansion / potential reversal            | Shows strong directional move; can indicate continuation or exhaustion                |
| **Near High**            | Current high near rolling high                                      | Confirms bullish momentum                           | Helps validate continuation; combined with trend signals for stronger entries         |
| **Near Low**             | Current low near rolling low                                        | Confirms bearish momentum                           | Helps validate continuation; combined with trend signals for stronger exits           |
| **PatternCount**         | Total number of patterns firing for candle                          | Quantifies overall signal intensity                 | Higher counts → higher confidence in combined signal scoring                          |
| **PatternType**          | Dominant / key pattern detected                                     | Quickly identifies most relevant signal             | Guides weighted scoring in hybrid signal calculation                                  |


'''
def add_candle_patterns(df, o="Open", h="High", l="Low", c="Close", v="Volume" ,pattern_window=5,
                        doji_thresh=0.1, hammer_thresh=0.25, marubozu_thresh=0.05,
                        long_body=0.6, small_body=0.25, shadow_ratio=2.0, near_edge=0.25, rng_thresh=1e-4):
    """
    Adds a variety of single- and multi-bar candlestick pattern markers:
    - Doji, Hammer, InvertedHammer, HangingMan, ShootingStar
    - Bullish/Bearish Marubozu, Engulfing, Harami, Piercing/DarkCloud, Morning/Evening Star, Tweezers, Inside/Outside Bars
    Also adds rolling high/low detection (NearHigh/NearLow) and pattern count/metadata.
    """
    df = df.copy()
    O = df[o].rolling(pattern_window, min_periods=pattern_window).apply(lambda x: x.iloc[0], raw=False).astype(float)
    C = df[c].rolling(pattern_window, min_periods=pattern_window).apply(lambda x: x.iloc[-1], raw=False).astype(float)
    H = df[h].rolling(pattern_window, min_periods=pattern_window).max().astype(float)
    L = df[l].rolling(pattern_window, min_periods=pattern_window).min().astype(float)
    av = df[v].rolling(20, min_periods=1).mean()
    hv = df[v] > 1.5 * av
    lv  = df[v] < 0.7 * av




    ref_price = C.replace(0, np.nan)
    body = (C - O).abs() / ref_price
    upsh = (H - np.maximum(O, C)) / ref_price
    dnsh = (np.minimum(O, C) - L) / ref_price
    rng = (H - L) / ref_price

    bull, bear = C > O, O > C

    downtrend = (C.shift(1).rolling(pattern_window, min_periods=pattern_window).apply(lambda x: x.iloc[-1] < x.iloc[0], raw=False).astype(bool))
    uptrend = (C.shift(1).rolling(pattern_window, min_periods=pattern_window).apply(lambda x: x.iloc[-1] > x.iloc[0], raw=False).astype(bool))

    new_cols = {
        "HighVolume": hv,
        "LowVolume": lv,
        "Doji": body <= doji_thresh * rng,
        "Hammer": (dnsh >= shadow_ratio * body) & (upsh <= hammer_thresh * body) & (body > 0) & (body <= hammer_thresh * 2 * rng) & downtrend,
        "HangingMan": None,  # set below
        "InvertedHammer": (upsh >= shadow_ratio * body) & (dnsh <= hammer_thresh * body) & (body > 0) & (body <= hammer_thresh * 2 * rng) & downtrend,
        "ShootingStar": None,  # set below
        "BullishMarubozu": bull & (body >= long_body * rng) & (upsh <= marubozu_thresh * rng) & (dnsh <= marubozu_thresh * rng),
        "BearishMarubozu": bear & (body >= long_body * rng) & (upsh <= marubozu_thresh * rng) & (dnsh <= marubozu_thresh * rng),
        "SuspiciousCandle": (rng <= rng_thresh) | (body <= rng_thresh)
    }

    new_cols["HangingMan"] = new_cols["Hammer"] & uptrend
    new_cols["ShootingStar"] = new_cols["InvertedHammer"] & uptrend

    # Multi-bar patterns
    O_shift1, C_shift1 = O.shift(1), C.shift(1)
    O_shift2, C_shift2 = O.shift(2), C.shift(2)
    H_shift1, L_shift1 = H.shift(1), L.shift(1)
    bull1, bull2 = bull.shift(1), bull.shift(2)
    bear1, bear2 = bear.shift(1), bear.shift(2)

    new_cols.update({
        "BullishEngulfing": (O_shift1 > C_shift1) & bull & (C >= O_shift1) & (O <= C_shift1),
        "BearishEngulfing": (C_shift1 > O_shift1) & bear & (O >= C_shift1) & (C <= O_shift1),
        "BullishHarami": (O_shift1 > C_shift1) & bull & (np.maximum(O, C) <= np.maximum(O_shift1, C_shift1)) & (np.minimum(O, C) >= np.minimum(O_shift1, C_shift1)),
        "BearishHarami": (C_shift1 > O_shift1) & bear & (np.maximum(O, C) <= np.maximum(O_shift1, C_shift1)) & (np.minimum(O, C) >= np.minimum(O_shift1, C_shift1)),
        "HaramiCross": new_cols["Doji"] & (np.maximum(O, C) <= np.maximum(O_shift1, C_shift1)) & (np.minimum(O, C) >= np.minimum(O_shift1, C_shift1)),
        "PiercingLine": (O_shift1 > C_shift1) & bull & (O < C_shift1) & (C > (O_shift1 + C_shift1)/2) & (C < O_shift1),
        "DarkCloudCover": (C_shift1 > O_shift1) & bear & (O > C_shift1) & (C < (O_shift1 + C_shift1)/2) & (C > O_shift1),
        "MorningStar": (O_shift2 > C_shift2) & (abs(C_shift1 - O_shift1) < abs(C_shift2 - O_shift2) * small_body) & bull & (C >= (O_shift2 + C_shift2)/2),
        "EveningStar": (C_shift2 > O_shift2) & (abs(C_shift1 - O_shift1) < abs(C_shift2 - O_shift2) * small_body) & bear & (C <= (O_shift2 + C_shift2)/2),
        "ThreeWhiteSoldiers": bull & bull1 & bull2 & (C > C_shift1) & (C_shift1 > C_shift2),
        "ThreeBlackCrows": bear & bear1 & bear2 & (C < C_shift1) & (C_shift1 < C_shift2),
        "TweezerTop": (H == H_shift1) & bear & bull1,
        "TweezerBottom": (L == L_shift1) & bull & bear1,
        "InsideBar": (H < H_shift1) & (L > L_shift1),
        "OutsideBar": (H > H_shift1) & (L < L_shift1),
        "NearHigh": H >= H.rolling(pattern_window).max() * (1 - near_edge),
        "NearLow": L <= L.rolling(pattern_window).min() * (1 + near_edge),
        
        "DragonflyDoji": (abs(C - O) <= doji_thresh * rng) & (H == C) & (L < O),
        "GravestoneDoji": (abs(C - O) <= doji_thresh * rng) & (L == C) & (H > O),
        "LongLeggedDoji": (abs(C - O) <= doji_thresh * rng) & (upsh > shadow_ratio * body) & (dnsh > shadow_ratio * body),
        # Continuation patterns
        "RisingThreeMethods": bull2 & bull1 & bull &
            (C_shift1 < O_shift2) & (C > C_shift1),
        "FallingThreeMethods": bear2 & bear1 & bear &
            (C_shift1 > O_shift2) & (C < C_shift1),
                "GapUp": O > H_shift1,
        # Gap signals
        "GapDown": O < L_shift1,
                # Spinning top (small body, long shadows both sides)
        "SpinningTop": (body <= small_body * rng) &
                       (upsh >= body) & (dnsh >= body),

        # Climax candle (range much larger than rolling avg)
        "ClimacticCandle": rng > rng.rolling(pattern_window).mean() * 2,

    })

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    pattern_cols = [col for col in new_cols if df[col].dtype == bool]
    df["PatternCount"] = df[pattern_cols].sum(axis=1)
    df["PatternType"] = np.select(
        [df.get("BullishEngulfing", False), df.get("MorningStar", False), df.get("ThreeWhiteSoldiers", False), df.get("BullishMarubozu", False)],
        ["BullishEngulfing", "MorningStar", "ThreeWhiteSoldiers", "BullishMarubozu"],
        default="None"
    )

    return df.fillna(False)

def add_trend_filters(df, c="Close", timeframe="Daily"):
    """
    Adds trend, momentum, volatility, and confirmation indicators to OHLC price data.

    Works consistently across low-priced (penny) and high-priced stocks by using 
    percentage changes rather than absolute deltas. Includes scale-invariant slope 
    and dynamic volatility thresholds.

    Parameters
    ----------
    df : pd.DataFrame
        OHLC data containing at least the column specified by `c`.
    c : str, default "Close"
        Name of the column containing closing prices.
    timeframe : str, default "Daily"
        One of ["Short", "Swing", "Long", "Daily"], which sets rolling windows 
        for moving averages, returns, volatility, and rate-of-change.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with the following additional columns:
        - MA : Rolling moving average
        - MA_slope : % change of MA (normalized slope)
        - UpTrend_MA, DownTrend_MA : Trend direction based on MA slope
        - RecentReturn : % change over return window
        - UpTrend_Return, DownTrend_Return : Trend based on % return
        - Volatility : Rolling std deviation of % returns
        - LowVolatility, HighVolatility : Volatility relative to median
        - ROC : Rate of change over MA window
        - MomentumUp, MomentumDown : Trend direction based on ROC thresholds
        - ConfirmedUpTrend, ConfirmedDownTrend : Combined trend confirmation

    | Feature                                   | When It’s Useful                                                     | Why It Matters                                                                                                                     |
    | ----------------------------------------- | -------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
    | **MA (Moving Average)**                   | Always, baseline trend detection                                     | Smooths out price noise; shows average price over a rolling window.                                                                |
    | **MA\_slope (normalized % slope)**        | When you want to compare momentum across penny stocks vs. large caps | Converts slope into a % change → scale-invariant. Prevents false signals from absolute price levels.                               |
    | **UpTrend\_MA / DownTrend\_MA**           | Short- to long-term trend following                                  | Simple binary indicator: is the average slope pointing up or down? Good for building rules.                                        |
    | **RecentReturn (% change over window)**   | Detecting short-term momentum                                        | Captures raw % gain/loss over the return window. Helps spot sudden moves.                                                          |
    | **UpTrend\_Return / DownTrend\_Return**   | For classifying recent moves into binary up/down trends              | Makes recent return usable in filters, strategies, or ML features.                                                                 |
    | **Volatility (rolling std of % returns)** | Risk assessment, breakout strategies, stop-loss sizing               | Captures recent price variability. Useful for distinguishing calm vs. choppy periods.                                              |
    | **LowVolatility / HighVolatility**        | Position sizing or strategy switching                                | Flags if volatility is above/below median → helps adjust trading style (mean-reversion vs breakout).                               |
    | **ROC (Rate of Change)**                  | Medium-term momentum                                                 | Measures speed of price change, similar to momentum indicators in TA.                                                              |
    | **MomentumUp / MomentumDown**             | Breakout or reversal confirmation                                    | Binary flag: is ROC strong enough beyond a threshold (e.g., 2%)? Filters out weak/noisy moves.                                     |
    | **ConfirmedUpTrend / ConfirmedDownTrend** | Signal validation for entries/exits                                  | Strongest signal → requires alignment of MA slope, returns, and momentum. Helps avoid false positives from single-indicator moves. |

    """
    df = df.copy()

    # -------------------------------
    # Timeframe-specific rolling windows
    # -------------------------------
    profiles = {
        "Short": {"ma": 2,  "ret": 1,  "vol": 3,  "roc_thresh": 0.02},
        "Swing": {"ma": 5,  "ret": 5,  "vol": 5,  "roc_thresh": 0.02},
        "Long":  {"ma": 10, "ret": 10, "vol": 10, "roc_thresh": 0.02},
        "Daily": {"ma": 7,  "ret": 1,  "vol": 5,  "roc_thresh": 0.02}
    }

    if timeframe not in profiles:
        raise ValueError(f"Invalid timeframe: {timeframe}. Must be one of {list(profiles.keys())}")

    params = profiles[timeframe]

    # -------------------------------
    # Moving Average and normalized slope
    # -------------------------------
    ma = df[c].rolling(params["ma"]).mean()
    ma_slope = ma.pct_change(params["ma"])

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
    df = df.copy()

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

def add_signal_strength(df, directional_groups=None):
    """
    Adds SignalStrength counts and percentages for each row.
    
    Parameters:
        df (pd.DataFrame): DataFrame with confirmed signals (columns starting with 'Valid').
        directional_groups (list, optional): List of prefixes to restrict for directional percentages
                                             e.g. ['Bullish', 'Bearish']. Default considers only these two.
    
    Returns:
        df (pd.DataFrame): Original DataFrame with new columns:
            - SignalStrength: count of all confirmed signals
            - BullishPctRaw / BearishPctRaw: percentage including all signals
            - BullishPctDirectional / BearishPctDirectional: percentage considering only Bullish vs Bearish
    """
    df = df.copy()
    
    # 1️⃣ All valid signal columns
    valid_cols = [c for c in df.columns if c.startswith("Valid")]
    
    # Count of all signals
    df["SignalStrength"] = df[valid_cols].astype(int).sum(axis=1) if valid_cols else 0
    
    # Optional: percentages of bullish/bearish including all signals (raw)
    bullish_cols = [c for c in valid_cols if c.startswith("ValidBullish")]
    bearish_cols = [c for c in valid_cols if c.startswith("ValidBearish")]
    
    df["BullishPctRaw"] = df[bullish_cols].astype(int).sum(axis=1) / df["SignalStrength"].replace(0,1)
    df["BearishPctRaw"] = df[bearish_cols].astype(int).sum(axis=1) / df["SignalStrength"].replace(0,1)
    
    # 2️⃣ Directional only percentages (ignoring neutral/exhaustion)
    if directional_groups is None:
        directional_groups = ["Bullish", "Bearish"]
    
    directional_cols = [c for c in valid_cols if any(c.startswith(f"Valid{g}") for g in directional_groups)]
    directional_sum = df[directional_cols].astype(int).sum(axis=1).replace(0,1)  # avoid division by 0
    
    bullish_dir_cols = [c for c in directional_cols if c.startswith("ValidBullish")]
    bearish_dir_cols = [c for c in directional_cols if c.startswith("ValidBearish")]
    
    df["BullishPctDirectional"] = df[bullish_dir_cols].astype(int).sum(axis=1) / directional_sum
    df["BearishPctDirectional"] = df[bearish_dir_cols].astype(int).sum(axis=1) / directional_sum
    
    return df


# -------------------------------
# 3. Finalize Signals
# -------------------------------
'''
Raw Price Data (Open, High, Low, Close)
           |
           v
---------------------------
1️⃣ Compute Momentum Metrics
   - Return = pct_change(Close)
   - AvgReturn = rolling mean
   - Volatility = rolling std
   - MomentumZ = (Return - AvgReturn) / Volatility
           |
           v
2️⃣ Determine MomentumAction
   - Buy if MomentumZ > BuyThresh
   - Sell if MomentumZ < SellThresh
   - Hold otherwise
           |
           v
---------------------------
3️⃣ Pattern Scoring (Bullish/Bearish Patterns)
   - Rolling sum of confirmed patterns over window
   - Score = Bull - Bear
   - Normalize → PatternScoreNorm
           |
           v
4️⃣ Determine PatternAction
   - Buy if PatternScoreNorm > threshold
   - Sell if PatternScoreNorm < -threshold
   - Hold otherwise
           |
           v
---------------------------
5️⃣ Candlestick Pattern Signals
   - Check candle_columns["Buy"] / ["Sell"]
   - Classify → CandleAction (Buy/Sell/Hold)
           |
           v
---------------------------
6️⃣ CandidateAction
   - Majority vote of MomentumAction, PatternAction, CandleAction
           |
           v
7️⃣ Filter Consecutive Signals
   - If same as previous Buy/Sell → convert to Hold
           |
           v
8️⃣ Determine TomorrowAction
   - TomorrowAction = Action.shift(-1)
   - Trace source: filtered vs unfiltered CandidateAction
           |
           v
---------------------------
9️⃣ Compute Signal Strength / Confidence
   - Count-based: # of valid patterns firing
   - Magnitude-based: PatternScore + MomentumZ
   - Weighted combination → SignalStrengthHybrid
   - ActionConfidence aligned with Action direction
           |
           v
---------------------------
Final Output Columns:
   - Action: today’s filtered action (Buy/Sell/Hold)
   - TomorrowAction: predicted next day action
   - CandidateAction: majority vote before filtering
   - ActionConfidence: hybrid strength score

'''
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
    count_weight = 1.5   # emphasize actual pattern confirmation
    momentum_weight = 1.0  # still use momentum

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
            "ValidTweezerBottom", "ValidBullishHarami","ValidDragonflyDoji"
        ]
    )]

    bearish_cols = [c for c in df.columns if c.startswith("Valid") and any(
        c.startswith(sig) for sig in [
            "ValidShootingStar", "ValidBearishEngulfing", "ValidDarkCloud",
            "ValidEveningStar", "ValidThreeBlackCrows", "ValidBearishMarubozu",
            "ValidTweezerTop", "ValidBearishHarami","ValidGravestoneDoji"
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


# -------------------------------
# 4. Batch Metadata
# -------------------------------
def add_batch_metadata(df, company_id, timeframe, ingest_ts=None):
    """Add BatchId, IngestedAt, CompanyId, TimeFrame metadata."""
    df = df.copy()
    if ingest_ts is None:
        ingest_ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    df["BatchId"] = f"{company_id}_{ingest_ts}"
    df["IngestedAt"] = ingest_ts
    df["CompanyId"] = company_id
    df["TimeFrame"] = timeframe
    return df
