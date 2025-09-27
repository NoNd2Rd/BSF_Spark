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
def add_candle_patterns(df, o="Open", h="High", l="Low", c="Close",
                        doji_thresh=0.1, long_body=0.6, small_body=0.25,
                        shadow_ratio=2.0, near_edge=0.25):
    O, H, L, C = df[o].astype(float), df[h].astype(float), df[l].astype(float), df[c].astype(float)
    rng   = (H - L).replace(0, np.nan)
    body  = (C - O).abs()
    upsh  = H - np.maximum(O, C)
    dnsh  = np.minimum(O, C) - L
    bull  = C > O
    bear  = O > C

    new_cols = {}

    # Core patterns
    new_cols["Doji"] = (body <= doji_thresh * rng)
    new_cols["Hammer"] = (dnsh >= shadow_ratio * body) & (upsh <= 0.2 * body) & (np.maximum(O, C) >= H - near_edge * rng)
    new_cols["InvertedHammer"] = (upsh >= shadow_ratio * body) & (dnsh <= 0.2 * body) & (np.minimum(O, C) <= L + near_edge * rng)
    new_cols["ShootingStar"] = (upsh >= shadow_ratio * body) & (dnsh <= 0.2 * body) & (np.minimum(O, C) <= L + near_edge * rng)

    O1, C1 = O.shift(1), C.shift(1)
    new_cols["BullishEngulfing"] = (O1 > C1) & bull & (C >= O1) & (O <= C1)
    new_cols["BearishEngulfing"] = (C1 > O1) & bear & (O >= C1) & (C <= O1)

    new_cols["BullishHarami"] = (O1 > C1) & bull & (np.maximum(O, C) <= np.maximum(O1, C1)) & (np.minimum(O, C) >= np.minimum(O1, C1))
    new_cols["BearishHarami"] = (C1 > O1) & bear & (np.maximum(O, C) <= np.maximum(O1, C1)) & (np.minimum(O, C) >= np.minimum(O1, C1))
    new_cols["HaramiCross"] = new_cols["Doji"] & ((np.maximum(O, C) <= np.maximum(O1, C1)) & (np.minimum(O, C) >= np.minimum(O1, C1)))

    new_cols["PiercingLine"] = (O1 > C1) & bull & (O < C1) & (C > (O1 + C1) / 2) & (C < O1)
    new_cols["DarkCloudCover"] = (C1 > O1) & bear & (O > C1) & (C < (O1 + C1) / 2) & (C > O1)

    O2, C2 = O.shift(2), C.shift(2)
    new_cols["MorningStar"] = (O2 > C2) & (abs(C1 - O1) < abs(C2 - O2) * small_body) & bull & (C >= (O2 + C2) / 2)
    new_cols["EveningStar"] = (C2 > O2) & (abs(C1 - O1) < abs(C2 - O2) * small_body) & bear & (C <= (O2 + C2) / 2)

    bull1, bull2 = bull.shift(1), bull.shift(2)
    bear1, bear2 = bear.shift(1), bear.shift(2)
    new_cols["ThreeWhiteSoldiers"] = bull & bull1 & bull2 & (C > C.shift(1)) & (C.shift(1) > C.shift(2))
    new_cols["ThreeBlackCrows"] = bear & bear1 & bear2 & (C < C.shift(1)) & (C.shift(1) < C.shift(2))

    # Additional patterns
    new_cols["BullishMarubozu"] = (body >= long_body * rng) & (upsh <= 0.05 * rng) & (dnsh <= 0.05 * rng) & bull
    new_cols["BearishMarubozu"] = (body >= long_body * rng) & (upsh <= 0.05 * rng) & (dnsh <= 0.05 * rng) & bear

    new_cols["TweezerTop"] = (H == H.shift(1)) & bear & bull.shift(1)
    new_cols["TweezerBottom"] = (L == L.shift(1)) & bull & bear.shift(1)

    new_cols["InsideBar"] = (H < H.shift(1)) & (L > L.shift(1))
    new_cols["OutsideBar"] = (H > H.shift(1)) & (L < L.shift(1))

    # Diagnostics
    new_cols["SuspiciousCandle"] = (rng <= 0.001) | (body <= 0.001)
    recent_high = H.rolling(5).max()
    recent_low = L.rolling(5).min()
    new_cols["NearHigh"] = H >= recent_high * (1 - near_edge)
    new_cols["NearLow"] = L <= recent_low * (1 + near_edge)

    # Add all pattern columns at once
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    # Metadata
    pattern_cols = [col for col in new_cols if df[col].dtype == bool]
    df["PatternCount"] = df[pattern_cols].sum(axis=1)

    df["PatternType"] = np.select(
        [df["BullishEngulfing"], df["MorningStar"], df["ThreeWhiteSoldiers"], df["BullishMarubozu"]],
        ["BullishEngulfing", "MorningStar", "ThreeWhiteSoldiers", "BullishMarubozu"],
        default="None"
    )

    return df.copy().fillna(False)


# -------------------------------
# Add Trends
# -------------------------------
def add_trend_filters(df, c="Close"):
    df = df.copy()  # Ensure safe assignment

    profiles = {
        "Short":  {"ma": 2,  "ret": 1,  "vol": 3},
        "Long":   {"ma": 50, "ret": 20, "vol": 30},
        "Swing":  {"ma": 20, "ret": 5,  "vol": 10},
        "Daily":  {"ma": 5,  "ret": 1,  "vol": 5}
    }

    new_cols = {}

    for label, p in profiles.items():
        # Moving Average & Slope
        ma = df[c].rolling(p["ma"]).mean()
        ma_slope = ma.diff(p["ma"])
        new_cols[f"MA_{label}"] = ma
        new_cols[f"MA_slope_{label}"] = ma_slope
        new_cols[f"UpTrend_MA_{label}"] = ma_slope > 0
        new_cols[f"DownTrend_MA_{label}"] = ma_slope < 0

        # Recent Return
        recent_ret = df[c].pct_change(p["ret"])
        new_cols[f"RecentReturn_{label}"] = recent_ret
        new_cols[f"UpTrend_Return_{label}"] = recent_ret > 0
        new_cols[f"DownTrend_Return_{label}"] = recent_ret < 0

        # Volatility
        vol = df[c].rolling(p["vol"]).std()
        vol_median = vol.median()
        new_cols[f"Volatility_{label}"] = vol
        new_cols[f"LowVolatility_{label}"] = vol < vol_median
        new_cols[f"HighVolatility_{label}"] = vol > vol_median

        # Momentum
        roc = df[c].pct_change(p["ma"])
        new_cols[f"ROC_{label}"] = roc
        new_cols[f"MomentumUp_{label}"] = roc > 0.02
        new_cols[f"MomentumDown_{label}"] = roc < -0.02

        # Confirmed Trend
        new_cols[f"ConfirmedUpTrend_{label}"] = (
            new_cols[f"UpTrend_MA_{label}"] &
            new_cols[f"UpTrend_Return_{label}"] &
            new_cols[f"MomentumUp_{label}"]
        )
        new_cols[f"ConfirmedDownTrend_{label}"] = (
            new_cols[f"DownTrend_MA_{label}"] &
            new_cols[f"DownTrend_Return_{label}"] &
            new_cols[f"MomentumDown_{label}"]
        )

    # Add all columns at once
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    return df.copy()


# -------------------------------
# Confirmed Signals
# -------------------------------
def add_confirmed_signals(df):
    df = df.copy()  # Ensure safe assignment
    timeframes = ['Short', 'Long', 'Swing', 'Daily']
    new_cols = {}

    for tf in timeframes:
        new_cols[f"ValidHammer_{tf}"] = df.get("Hammer", False) & df.get(f"DownTrend_MA_{tf}", False)
        new_cols[f"ValidShootingStar_{tf}"] = df.get("ShootingStar", False) & df.get(f"UpTrend_MA_{tf}", False)
        new_cols[f"ValidBullishEngulfing_{tf}"] = df.get("BullishEngulfing", False) & df.get(f"DownTrend_MA_{tf}", False)
        new_cols[f"ValidBearishEngulfing_{tf}"] = df.get("BearishEngulfing", False) & df.get(f"UpTrend_MA_{tf}", False)
        new_cols[f"ValidPiercingLine_{tf}"] = df.get("PiercingLine", False) & df.get(f"DownTrend_Return_{tf}", False)
        new_cols[f"ValidDarkCloud_{tf}"] = df.get("DarkCloudCover", False) & df.get(f"UpTrend_Return_{tf}", False)
        new_cols[f"ValidMorningStar_{tf}"] = df.get("MorningStar", False) & df.get(f"DownTrend_MA_{tf}", False)
        new_cols[f"ValidEveningStar_{tf}"] = df.get("EveningStar", False) & df.get(f"UpTrend_MA_{tf}", False)
        new_cols[f"ValidThreeWhiteSoldiers_{tf}"] = df.get("ThreeWhiteSoldiers", False) & df.get(f"DownTrend_MA_{tf}", False)
        new_cols[f"ValidThreeBlackCrows_{tf}"] = df.get("ThreeBlackCrows", False) & df.get(f"UpTrend_MA_{tf}", False)

    # Add all columns at once
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    return df.copy()

def add_signal_strength(df):
    df = df.copy()  # Ensure safe assignment

    timeframes = ["Short", "Long", "Swing", "Daily"]
    base_signals = [
        "ValidHammer", "ValidBullishEngulfing", "ValidPiercingLine",
        "ValidMorningStar", "ValidThreeWhiteSoldiers",
        "ValidShootingStar", "ValidBearishEngulfing", "ValidDarkCloud",
        "ValidEveningStar", "ValidThreeBlackCrows"
    ]

    signal_cols = [f"{sig}_{tf}" for sig in base_signals for tf in timeframes]
    available_cols = [col for col in signal_cols if col in df.columns]

    if not available_cols:
        df["SignalStrength"] = 0
    else:
        df["SignalStrength"] = df[available_cols].astype(int).sum(axis=1)

    return df

def finalize_signals_old(df, buy_thresh=0.01, sell_thresh=-0.01):
    df = df.copy()  # Ensure safe memory layout

    # Classify action based on multi-timeframe signals
    if "SignalStrength" not in df.columns:
        df["SignalStrength"] = 0

    df["Action"] = df.apply(classify_action, axis=1)
    

    # Suppress repeated Buy/Sell signals
    filtered_actions = []
    prev_trade_action = "Hold"
    for action in df["Action"]:
        if action in ["Buy", "Sell"]:
            if action == prev_trade_action:
                filtered_actions.append("Hold")
            else:
                filtered_actions.append(action)
                prev_trade_action = action
        else:
            filtered_actions.append("Hold")

    df["PrevAction"] = ["Hold"] + filtered_actions[:-1]
    df["Action"] = filtered_actions

    # Forward-looking return and label
    df["TomorrowClose"] = df["Close"].shift(-1)
    df["TomorrowReturn"] = (df["TomorrowClose"] - df["Close"]) / df["Close"]
    df["TomorrowReturn"] = df["TomorrowReturn"].fillna(0)

    # Backward-looking momentum
    df["YesterdayReturn"] = df["Close"].pct_change().fillna(0)

    # Momentum-based guess for tomorrow's action
    df["TomorrowAction"] = np.where(
        df["YesterdayReturn"] > buy_thresh, "Buy",
        np.where(df["YesterdayReturn"] < sell_thresh, "Sell", "Hold")
    )

    # Enforce alternating Buy/Sell in TomorrowAction
    filtered_tomorrow = []
    prev_trade = "Hold"
    for action in df["TomorrowAction"]:
        if action in ["Buy", "Sell"]:
            if action == prev_trade:
                filtered_tomorrow.append("Hold")
            else:
                filtered_tomorrow.append(action)
                prev_trade = action
        else:
            filtered_tomorrow.append("Hold")

    df["TomorrowAction"] = pd.Series(filtered_tomorrow, index=df.index).fillna("Hold")

    # Confidence score from signal strength
    max_strength = df["SignalStrength"].max()
    df["ActionConfidence"] = (
        df["SignalStrength"] / max_strength if max_strength > 0 else 0
    )
    df["ActionConfidence"] = df["ActionConfidence"].fillna(0)

    # Signal duration tracker
    df["SignalDuration"] = (df["Action"] != df["Action"].shift()).cumsum()

    # Valid action flag
    df["ValidAction"] = df["Action"].isin(["Buy", "Sell"])

    # Flag for rows with complete signal data
    required_cols = ["Action", "TomorrowAction", "SignalStrength"]
    df["HasValidSignal"] = df[required_cols].notnull().all(axis=1)

    return df

def finalize_signals_old1(df, buy_thresh=0.01, sell_thresh=-0.01):
    df = df.copy()  # safe memory layout

    # 1) Core action classification
    if "SignalStrength" not in df:
        df["SignalStrength"] = 0
    df["Action"] = df.apply(classify_action, axis=1)

    # 2) Suppress repeated trades
    filtered = []
    prev = "Hold"
    for a in df["Action"]:
        if a in ("Buy", "Sell"):
            if a == prev:
                filtered.append("Hold")
            else:
                filtered.append(a)
                prev = a
        else:
            filtered.append("Hold")
    df["PrevAction"] = ["Hold"] + filtered[:-1]
    df["Action"]     = filtered

    # 3) Shift Action backward so it appears on the prior row
    #    (last row becomes NaN → fill with "Hold")
    df["Action"] = df["Action"].shift(-1).fillna("Hold")
    df["PrevAction"] = df["Action"].shift()  # adjust PrevAction to match shifted Action

    # 4) Compute forward/back returns and tomorrow’s guess
    df["TomorrowClose"]  = df["Close"].shift(-1)
    df["TomorrowReturn"] = (df["TomorrowClose"] - df["Close"]) / df["Close"]
    df["YesterdayReturn"] = df["Close"].pct_change().fillna(0)
    df["TomorrowAction"] = np.where(
        df["YesterdayReturn"] > buy_thresh, "Buy",
        np.where(df["YesterdayReturn"] < sell_thresh, "Sell", "Hold")
    )

    # 5) Enforce alternating tomorrow signals
    filt2, prev2 = [], "Hold"
    for a in df["TomorrowAction"]:
        if a in ("Buy", "Sell"):
            if a == prev2:
                filt2.append("Hold")
            else:
                filt2.append(a)
                prev2 = a
        else:
            filt2.append("Hold")
    df["TomorrowAction"] = pd.Series(filt2, index=df.index).fillna("Hold")

    # 6) Confidence, duration, flags
    max_s = df["SignalStrength"].max()
    df["ActionConfidence"] = (df["SignalStrength"] / max_s if max_s>0 else 0).fillna(0)
    df["SignalDuration"]   = (df["Action"] != df["Action"].shift()).cumsum()
    df["ValidAction"]      = df["Action"].isin(["Buy","Sell"])
    df["HasValidSignal"]   = df[["Action","TomorrowAction","SignalStrength"]].notna().all(axis=1)

    return df
    
def finalize_signals_old3(df, buy_thresh=0.01, sell_thresh=-0.01):
    df = df.copy()

    # 1) Compute tomorrow's return and yesterday's return
    df["TomorrowReturn"] = (df["Close"].shift(-1) - df["Close"]) / df["Close"]
    df["YesterdayReturn"] = df["Close"].pct_change().fillna(0)

    # 2) Momentum‐based TomorrowAction (signal for next bar)
    df["TomorrowAction"] = np.where(
        df["YesterdayReturn"] > buy_thresh, "Buy",
        np.where(df["YesterdayReturn"] < sell_thresh, "Sell", "Hold")
    )

    # 3) Enforce alternating Buy/Sell on TomorrowAction
    filt, prev = [], "Hold"
    for a in df["TomorrowAction"]:
        if a in ("Buy", "Sell"):
            if a == prev:
                filt.append("Hold")
            else:
                filt.append(a)
                prev = a
        else:
            filt.append("Hold")
    df["TomorrowAction"] = pd.Series(filt, index=df.index)

    # 4) Shift it back one day so the action appears *before* the move
    df["Action"] = df["TomorrowAction"].shift(-1).fillna("Hold")

    # 5) (Optional) Clean up intermediate cols
    df = df.drop(columns=["TomorrowAction", "YesterdayReturn", "TomorrowReturn"])

    return df
import pandas as pd
import numpy as np

def finalize_signals(df):
    """
    Refines forward-looking trade signal generation for 'TomorrowAction' using confluence
    of candlestick patterns and momentum thresholds. Designed to fire signals *before* price moves,
    ensuring labels are usable for live prediction, not lagging. No external packages required.
    
    Inputs:
      df: pandas DataFrame with precomputed candlestick columns (see list below)
    Outputs:
      df: with updated 'TomorrowAction' column and new feature columns for enhanced prediction.
    """

    # 1. --- Feature Engineering: Candlestick Pattern Strength and Confluence ---

    # Define strong bullish/bearish candlestick columns (based on pattern reliability)
    bullish_patterns = [
        'Hammer', 'BullishEngulfing', 'BullishHarami', 'PiercingLine', 'MorningStar',
        'ThreeWhiteSoldiers', 'BullishMarubozu', 'TweezerBottom', 'HaramiCross',
        'InsideBar', 'OutsideBar', 'NearLow'
    ]
    bearish_patterns = [
        'ShootingStar', 'BearishEngulfing', 'BearishHarami', 'DarkCloudCover', 'EveningStar',
        'ThreeBlackCrows', 'BearishMarubozu', 'TweezerTop', 'HaramiCross',
        'InsideBar', 'OutsideBar', 'NearHigh'
    ]
    # Additional context: Doji, SuspiciousCandle can indicate indecision/volatility.

    # Compute strength metrics: sum of present bullish/bearish patterns per row
    df['BullishPatternCount'] = df[bullish_patterns].fillna(0).sum(axis=1)
    df['BearishPatternCount'] = df[bearish_patterns].fillna(0).sum(axis=1)

    # PatternStrength: +N if more bullish, -N if more bearish, 0 otherwise
    df['PatternStrength'] = df['BullishPatternCount'] - df['BearishPatternCount']

    # PatternConfluence: Only strong if two or more strong signals in same direction
    df['PatternConfluence'] = np.where(np.abs(df['PatternStrength']) >= 2,
                                       np.sign(df['PatternStrength']), 0)
    # +1: Bullish confluence; -1: Bearish confluence; 0: Neutral

    # Synthesize a PatternType summary field for further logic
    def synth_pattern_type(row):
        if row['PatternConfluence'] > 0:
            return 'BullishConfluence'
        elif row['PatternConfluence'] < 0:
            return 'BearishConfluence'
        elif (row['Doji'] or row['SuspiciousCandle']):
            return 'Uncertain'
        else:
            return 'Neutral'
    df['SynthPatternType'] = df.apply(synth_pattern_type, axis=1)


    # 2. --- Wick-Body Ratio Calculation: Rejection and Pattern Strength ---

    # Calculate wick/body ratios, often strong reversal candles show long wicks/small bodies
    # Requires open, high, low, close columns
    def compute_wick_body(row):
        body = abs(row['Close'] - row['Open'])
        upper_wick = row['High'] - max(row['Close'], row['Open'])
        lower_wick = min(row['Close'], row['Open']) - row['Low']
        candle_range = row['High'] - row['Low'] if (row['High'] - row['Low']) != 0 else 1e-8
        # Avoid division by zero
        body_pct = body / candle_range
        upper_pct = upper_wick / candle_range
        lower_pct = lower_wick / candle_range
        return pd.Series({'BodyPct': body_pct, 'UpperWickPct': upper_pct, 'LowerWickPct': lower_pct})
    wick_body_df = df.apply(compute_wick_body, axis=1)
    df = pd.concat([df, wick_body_df], axis=1)

    # Construct CandleStrength: Firm/Weak/Neutral (e.g., firm body or strong wick rejection)
    def candle_strength(row):
        if row['BodyPct'] > 0.65:
            return 'Firm'
        elif (row['LowerWickPct'] > 0.5 and row['PatternConfluence'] > 0):
            return 'BullishRejection'
        elif (row['UpperWickPct'] > 0.5 and row['PatternConfluence'] < 0):
            return 'BearishRejection'
        else:
            return 'Neutral'
    df['CandleStrength'] = df.apply(candle_strength, axis=1)

    # 3. --- Multi-Timeframe (MTF) Pattern Bias: Trend Context from Higher Timeframe ---

    # Simple weekly trend for context: rolling 5-day close mean as higher timeframe proxy
    # (If you already have pattern features from actual higher TF, swap in here)
    df['HTF_Trend'] = df['Close'].rolling(window=5, min_periods=1).mean().shift(1)
    df['MTF_Bias'] = np.where(df['Close'] > df['HTF_Trend'], 1,
                        np.where(df['Close'] < df['HTF_Trend'], -1, 0))
    # +1 = Above HTF mean; -1 = Below; 0 = at mean

    # 4. --- Dynamic Momentum Threshold Feature Engineering ---

    # Use rolling volatility (e.g., stdev of returns) to create dynamic momentum thresholds
    df['Return_1d'] = df['Close'].pct_change()
    # Rolling 10-day volatility
    df['RollingVol'] = df['Return_1d'].rolling(window=10, min_periods=5).std()
    # Dynamic thresholds: mean ± k*volatility (e.g., k=1, can be parameterized)
    k = 1.0
    df['Up_Thresh'] = df['Return_1d'].rolling(window=10, min_periods=5).mean() + k * df['RollingVol']
    df['Down_Thresh'] = df['Return_1d'].rolling(window=10, min_periods=5).mean() - k * df['RollingVol']

    # Current momentum signal: fires only if return exceeds dynamic threshold
    df['PositiveMomentum'] = (df['Return_1d'] > df['Up_Thresh']).astype(int)
    df['NegativeMomentum'] = (df['Return_1d'] < df['Down_Thresh']).astype(int)

    # 5. --- Final Action Synthesis (Forward-Looking Label Creation) ---

    # The synthesized action label considers:
    # - Candlestick confluence
    # - Candle strength/rejection feature
    # - Dynamic momentum signals
    # - Higher timeframe bias ("trade with trend")
    # - Optionally, require at least 2 elements of confluence for 'strong' signal

    def classify_action(row):
        # Buy scenario
        if (row['PatternConfluence'] > 0 and row['CandleStrength'] in ['Firm', 'BullishRejection']
            and row['PositiveMomentum'] and row['MTF_Bias'] > 0):
            return 'Buy'
        # Sell scenario
        elif (row['PatternConfluence'] < 0 and row['CandleStrength'] in ['Firm', 'BearishRejection']
              and row['NegativeMomentum'] and row['MTF_Bias'] < 0):
            return 'Sell'
        else:
            return 'Hold'
    df['RawAction'] = df.apply(classify_action, axis=1)

    # Shift the label forward by 1 to make it truly predictive: signal as of T applies to T+1
    df['TomorrowAction'] = df['RawAction'].shift(1)

    # Remove any lookahead bias: the action for each row is derived strictly from up-to-that-row features.

    # 6. --- Add Signal Timing Diagnostic Columns for Analysis Purposes ---

    # Label if action is 'anticipatory' (fires before a move), 'on-move' (within 1 bar), or 'lagging'
    # Compare future returns to action for evaluation (not for training!)
    look_forward = 2  # e.g., look 2 days ahead
    df['FutureReturn'] = df['Close'].shift(-look_forward) / df['Close'] - 1
    def timing_flag(row):
        if row['TomorrowAction'] == 'Buy' and row['FutureReturn'] > row['RollingVol']:
            return 'Early'
        elif row['TomorrowAction'] == 'Sell' and row['FutureReturn'] < -row['RollingVol']:
            return 'Early'
        elif row['TomorrowAction'] in ['Buy', 'Sell']:
            return 'Late'
        else:
            return 'Neutral'
    df['ActionWindow'] = df.apply(timing_flag, axis=1)

    # 7. --- Preserve Existing Fields and Return ---

    # All original columns are kept; new columns supplement for improved prediction and diagnostics.
    return df

# Example of usage:
# df = pd.read_csv('your_stock_data_with_patterns.csv')
# df = finalize_signals(df)
# print(df[['Date','TomorrowAction','PatternConfluence','CandleStrength','MTF_Bias','PositiveMomentum','NegativeMomentum','ActionWindow']].tail(10))
import numpy as np
import pandas as pd

def finalize_signals_refined(df, 
                             pattern_columns=None,
                             bullish_patterns=None,
                             bearish_patterns=None,
                             pattern_count_threshold=1,
                             momentum_lookback=5, 
                             momentum_zscore=1.0,
                             volatility_lookback=10,
                             pattern_reliability=None):
    """
    Enhanced signal finalization to produce a genuinely forward-looking TomorrowAction column.

    Parameters:
    -----------
    df: pd.DataFrame
        DataFrame with all candlestick patterns, OHLC, 'Return', PatternCount, PatternType, etc.
    pattern_columns: list of str, optional
        List of candlestick pattern column names (one-hot or binary).
    bullish_patterns: list of str, optional
        List of column names indicating bullish reversal/continuation signals.
    bearish_patterns: list of str, optional
        List of column names for bearish setups.
    pattern_count_threshold: int
        Minimum number of concurrent pattern signals for confirmation.
    momentum_lookback: int
        Rolling window (days) for mean and std computation.
    momentum_zscore: float
        Number of std deviations above mean return to trigger momentum entry (dynamic threshold).
    volatility_lookback: int
        Days for rolling volatility calculation.
    pattern_reliability: dict, optional
        Mapping pattern name -> historical reliability (dict for customizing weights).

    Returns:
    --------
    pd.DataFrame
        The original df with an improved, forward-looking 'TomorrowAction' column added.
    """

    df = df.copy()

    #--- 1. Identify all pattern columns if not supplied ---
    if pattern_columns is None:
        # Default: all columns matching known pattern suffix
        known_patterns = [
            'Doji', 'Hammer', 'InvertedHammer', 'ShootingStar',
            'BullishEngulfing', 'BearishEngulfing', 'BullishHarami', 'BearishHarami',
            'HaramiCross', 'PiercingLine', 'DarkCloudCover', 'MorningStar', 'EveningStar',
            'ThreeWhiteSoldiers', 'ThreeBlackCrows',
            'BullishMarubozu', 'BearishMarubozu',
            'TweezerTop', 'TweezerBottom',
            'InsideBar', 'OutsideBar', 'SuspiciousCandle'
        ]
        pattern_columns = [col for col in df.columns if col in known_patterns]
    
    #--- 2. Set default bullish and bearish patterns if not provided ---
    if bullish_patterns is None:
        bullish_patterns = [
            'Hammer', 'BullishEngulfing', 'BullishHarami', 'PiercingLine',
            'MorningStar', 'ThreeWhiteSoldiers', 'BullishMarubozu', 'TweezerBottom',
        ]
    if bearish_patterns is None:
        bearish_patterns = [
            'ShootingStar', 'BearishEngulfing', 'BearishHarami', 'DarkCloudCover',
            'EveningStar', 'ThreeBlackCrows', 'BearishMarubozu', 'TweezerTop',
        ]

    #--- 3. Compute rolling volatility and momentum filters ---
    # Rolling volatility (standard deviation of returns)
    df['RollingVolatility'] = df['Return'].rolling(volatility_lookback, min_periods=1).std()

    # Standardized Z-score momentum based on past average and volatility
    df['MomentumZ'] = (
        (df['Return'] - df['Return'].rolling(momentum_lookback, min_periods=1).mean()) /
        (df['Return'].rolling(momentum_lookback, min_periods=1).std(ddof=0) + 1e-8)
    )

    #--- 4. Candlestick pattern-based signals ---
    df['BullishSignal'] = df[bullish_patterns].sum(axis=1)
    df['BearishSignal'] = df[bearish_patterns].sum(axis=1)

    # Optionally combine with PatternCount if available
    if 'PatternCount' in df.columns:
        df['BullishSignal'] += (df['PatternCount'] >= pattern_count_threshold).astype(int)
        df['BearishSignal'] += (df['PatternCount'] >= pattern_count_threshold).astype(int)

    #--- 5. Optional: Weight patterns by historical reliability ---
    if pattern_reliability is not None:
        for pattern in bullish_patterns:
            if pattern in df and pattern in pattern_reliability:
                df['BullishSignal'] += df[pattern] * pattern_reliability.get(pattern, 1.0)
        for pattern in bearish_patterns:
            if pattern in df and pattern in pattern_reliability:
                df['BearishSignal'] += df[pattern] * pattern_reliability.get(pattern, 1.0)
    
    #--- 6. Risk filter: Remove signals on high-risk or fake candles if flagged ---
    if 'SuspiciousCandle' in df.columns:
        df['BullishSignal'] = df['BullishSignal'].where(~df['SuspiciousCandle'])
        df['BearishSignal'] = df['BearishSignal'].where(~df['SuspiciousCandle'])

    #--- 7. Multi-timeframe confirmation (e.g., confirm with trend from higher timeframe) ---
    # For illustration: align with rolling N-day trend
    if 'Close' in df.columns:
        high_tf_trend = df['Close'].rolling(window=5, min_periods=1).mean().diff()
        df['BullishMTF'] = (high_tf_trend > 0).astype(int)
        df['BearishMTF'] = (high_tf_trend < 0).astype(int)
        df['BullishSignal'] = df['BullishSignal'] * df['BullishMTF']
        df['BearishSignal'] = df['BearishSignal'] * df['BearishMTF']

    #--- 8. Final signal logic (must satisfy both candlestick and momentum filter) ---
    def resolve_action(row):
        bullish = row['BullishSignal'] >= 1
        bearish = row['BearishSignal'] >= 1
        # Dynamic threshold example: Only trigger when standardized momentum exceeds threshold
        momentum_good = row['MomentumZ'] > momentum_zscore
        momentum_bad = row['MomentumZ'] < -momentum_zscore
        # Require both pattern and momentum filter
        if bullish and momentum_good:
            return 'Buy'
        elif bearish and momentum_bad:
            return 'Sell'
        else:
            return 'Hold'

    df['RawTomorrowAction'] = df.apply(resolve_action, axis=1)
    #--- 9. Ensure signal is forward-looking ("TomorrowAction" for day t+1 is computed as of the close of day t) ---
    df['TomorrowAction'] = df['RawTomorrowAction'].shift(1)
    # For initial warm-up, fill missing with neutral action
    df['TomorrowAction'] = df['TomorrowAction'].fillna('Hold')

    #--- 10. Keep all original and new fields for downstream analysis ---
    # Optionally, drop intermediate columns: 'BullishSignal', 'BearishSignal', 'RawTomorrowAction', etc.
    return df

# Usage example:
# df = finalize_signals_refined(df)
# Now df has a reliable, forward-predictive 'TomorrowAction' column ready for real-time or out-of-sample testing.

import pandas as pd
import numpy as np

def finalize_signals_new(df,
                     momentum_window=3,
                     momentum_threshold=0.006,  # ~0.6%: adjust to fit average volatility
                     pattern_window=3,
                     cluster_count=2):
    """
    Refined forward-looking stock movement signal generator.

    Enhances timing and predictive value of TomorrowAction by:
    - Only using data up to each current bar (no lookahead);
    - Creating momentum and pattern confluence features;
    - Tagging exhaustion zones (NearHigh/NearLow) with patterns;
    - Firing action at close of bar T for possible execution at bar T+1.

    Assumes candlestick pattern columns and 'Close' exist in df.
    Retains all original fields and logic.
    """
    df = df.copy()
    '''
    for col in df.columns:
        print(col)
    '''
    # --- 1. Calculate Rolling Momentum (Rate of Change) ---
    # Use past momentum_window bars to calc smoothed return for momentum gating
    df['RollingReturn'] = df['Close'].pct_change(periods=momentum_window).shift(1)  # Only use completed readings

    # --- 2. Composite Candlestick Pattern Score ---
    # Assign weights (customizable) to bullish, bearish, indecision patterns
    bullish_patterns = [
        'Hammer', 'BullishEngulfing', 'BullishHarami', 'PiercingLine',
        'MorningStar', 'ThreeWhiteSoldiers', 'BullishMarubozu', 'TweezerBottom'
    ]
    bearish_patterns = [
        'ShootingStar', 'BearishEngulfing', 'BearishHarami', 'DarkCloudCover',
        'EveningStar', 'ThreeBlackCrows', 'BearishMarubozu', 'TweezerTop'
    ]
    indecision_patterns = ['Doji', 'HaramiCross', 'SuspiciousCandle']

    # Compute rolling counts for bullish/bearish within window up to T (no future lookahead)
    df['BullishPatternCount'] = (
        df[bullish_patterns].sum(axis=1)
        .rolling(window=pattern_window, min_periods=1).sum()
    )
    
    df['BearishPatternCount'] = (
        df[bearish_patterns].sum(axis=1)
        .rolling(window=pattern_window, min_periods=1).sum()
    )
    
    df['IndecisionPatternCount'] = (
        df[indecision_patterns].sum(axis=1)
        .rolling(window=pattern_window, min_periods=1).sum()
    )

    df['NetPatternScore'] = df['BullishPatternCount'] - df['BearishPatternCount']

    # --- 3. Pattern Cluster Flags ---
    # Fires if >= cluster_count bullish or bearish patterns in recent window
    df['BullishCluster'] = (df['BullishPatternCount'] >= cluster_count).astype(int)
    df['BearishCluster'] = (df['BearishPatternCount'] >= cluster_count).astype(int)

    # --- 4. Exhaustion Signals (Near High/Low with Patterns) ---
    df['BullishExhaustion'] = (
        ((df['NearLow'] == 1) | (df['OutsideBar'] == 1)) &
        (df['BullishCluster'] == 1)
    ).astype(int)
    df['BearishExhaustion'] = (
        ((df['NearHigh'] == 1) | (df['OutsideBar'] == 1)) &
        (df['BearishCluster'] == 1)
    ).astype(int)

    # --- 5. Multi-Timeframe "Market Structure" -- Identify higher highs/lows ---
    df['HigherHigh'] = (df['High'] > df['High'].shift(1)) & (df['High'].shift(1) > df['High'].shift(2))
    df['LowerLow'] = (df['Low'] < df['Low'].shift(1)) & (df['Low'].shift(1) < df['Low'].shift(2))
    df['StructureShift'] = np.where(df['HigherHigh'], 1, np.where(df['LowerLow'], -1, 0))

    # --- 6. Classify Tomorrow's Action ---
    # Core rule: predict up if bullish exhaustion & momentum up, down if bearish exhaustion & momentum down

    action = []
    for idx, row in df.iterrows():
        # Forward-looking logic: action is based only on info available at this bar,
        # but labeled for the *next* bar (shift back one at end)
        # Conservative default: only fire if both pattern and momentum agree
        if row['BullishExhaustion'] and row['RollingReturn'] > momentum_threshold:
            action.append('Buy')
        elif row['BearishExhaustion'] and row['RollingReturn'] < -momentum_threshold:
            action.append('Sell')
        elif row['NetPatternScore'] > 0 and row['RollingReturn'] > momentum_threshold:
            action.append('Buy')
        elif row['NetPatternScore'] < 0 and row['RollingReturn'] < -momentum_threshold:
            action.append('Sell')
        else:
            action.append('Hold')
    df['TomorrowAction'] = action

    # --- 7. Shift TomorrowAction So It Can Be Used for Next Bar Prediction (no lookahead) ---
    df['TomorrowAction'] = df['TomorrowAction'].shift(1)

    # --- 8. Keep All Original/Existing Fields ---
    # If original logic for classify_action or previous labels needed, assign/copy here as backup
    # For example:
    # df['PreviousTomorrowAction'] = ... # Copy if required

    # --- 9. Additional Diagnostics (optional but recommended for evaluation) ---
    # ForwardReturn: realized forward return; used for backtest validation, NOT in live trading signal
    df['ForwardReturn'] = df['Close'].shift(-1) / df['Close'] - 1

    # --- 10. Clean-up (Optional): Fill NAs at extreme ends, e.g., with 'Hold'
    df['TomorrowAction'] = df['TomorrowAction'].fillna('Hold')

    return df


import pandas as pd
import numpy as np

def finalize_signals_merged(df,
                            momentum_window=3,
                            momentum_threshold=0.006,   # ~0.6% return filter
                            momentum_lookback=5,
                            momentum_zscore=1.0,        # dynamic z-threshold
                            volatility_lookback=10,
                            pattern_window=3,
                            cluster_count=2):
    """
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
    """
    df = df.copy()

    # --- 1. Momentum & Volatility ---
    df['RollingReturn'] = df['Close'].pct_change(periods=momentum_window).shift(1)
    df['RollingVolatility'] = df['Return'].rolling(volatility_lookback, min_periods=1).std()
    df['MomentumZ'] = (
        (df['Return'] - df['Return'].rolling(momentum_lookback, min_periods=1).mean()) /
        (df['Return'].rolling(momentum_lookback, min_periods=1).std(ddof=0) + 1e-8)
    )

    # --- 2. Pattern Counts ---
    bullish_patterns = [
        'Hammer', 'BullishEngulfing', 'BullishHarami', 'PiercingLine',
        'MorningStar', 'ThreeWhiteSoldiers', 'BullishMarubozu', 'TweezerBottom'
    ]
    bearish_patterns = [
        'ShootingStar', 'BearishEngulfing', 'BearishHarami', 'DarkCloudCover',
        'EveningStar', 'ThreeBlackCrows', 'BearishMarubozu', 'TweezerTop'
    ]
    indecision_patterns = ['Doji', 'HaramiCross', 'SuspiciousCandle']

    df['BullishPatternCount'] = (
        df[bullish_patterns].sum(axis=1)
          .rolling(window=pattern_window, min_periods=1).sum()
    )
    df['BearishPatternCount'] = (
        df[bearish_patterns].sum(axis=1)
          .rolling(window=pattern_window, min_periods=1).sum()
    )
    df['IndecisionPatternCount'] = (
        df[indecision_patterns].sum(axis=1)
          .rolling(window=pattern_window, min_periods=1).sum()
    )
    df['NetPatternScore'] = df['BullishPatternCount'] - df['BearishPatternCount']

    # --- 3. Pattern Clusters ---
    df['BullishCluster'] = (df['BullishPatternCount'] >= cluster_count).astype(int)
    df['BearishCluster'] = (df['BearishPatternCount'] >= cluster_count).astype(int)

    # --- 4. Exhaustion Signals ---
    df['BullishExhaustion'] = (
        ((df.get('NearLow', 0) == 1) | (df.get('OutsideBar', 0) == 1)) &
        (df['BullishCluster'] == 1)
    ).astype(int)
    df['BearishExhaustion'] = (
        ((df.get('NearHigh', 0) == 1) | (df.get('OutsideBar', 0) == 1)) &
        (df['BearishCluster'] == 1)
    ).astype(int)

    # --- 5. Market Structure ---
    df['HigherHigh'] = (df['High'] > df['High'].shift(1)) & (df['High'].shift(1) > df['High'].shift(2))
    df['LowerLow'] = (df['Low'] < df['Low'].shift(1)) & (df['Low'].shift(1) < df['Low'].shift(2))
    df['StructureShift'] = np.where(df['HigherHigh'], 1,
                             np.where(df['LowerLow'], -1, 0))

    # --- 6. Classification Logic ---
    def resolve_action(row):
        bullish = (row['BullishExhaustion'] or row['NetPatternScore'] > 0)
        bearish = (row['BearishExhaustion'] or row['NetPatternScore'] < 0)

        momentum_good = (row['RollingReturn'] > momentum_threshold) or (row['MomentumZ'] > momentum_zscore)
        momentum_bad  = (row['RollingReturn'] < -momentum_threshold) or (row['MomentumZ'] < -momentum_zscore)

        if bullish and momentum_good:
            return 'Buy'
        elif bearish and momentum_bad:
            return 'Sell'
        else:
            return 'Hold'

    df['RawTomorrowAction'] = df.apply(resolve_action, axis=1)

    # --- 7. Forward Labeling ---
    df['TomorrowAction'] = df['RawTomorrowAction'].shift(1).fillna('Hold')

    # --- 8. Diagnostics ---
    df['ForwardReturn'] = df['Close'].shift(-1) / df['Close'] - 1

    return df


'''
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
'''
'''
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
def finalize_signals_best(df, pattern_window=5, bullish_patterns=None, bearish_patterns=None):
    """
    Super Finalizer: Combines confluence, wick rejection, momentum filters,
    alternating logic, predictive shifting, and confidence scoring.
    Works on a pandas DataFrame with per-company dynamic thresholds.
    """

    import pandas as pd
    import numpy as np

    # -------------------------------
    # 1) Pattern Confluence
    # -------------------------------
    if bullish_patterns and bearish_patterns:
        df["BullishPatternCount"] = (
            df[bullish_patterns]
            .rolling(window=pattern_window, min_periods=1)
            .sum(axis=1)
        ).astype(float)
        df["BearishPatternCount"] = (
            df[bearish_patterns]
            .rolling(window=pattern_window, min_periods=1)
            .sum(axis=1)
        ).astype(float)
        df["PatternScore"] = (df["BullishPatternCount"] - df["BearishPatternCount"]).astype(float)
    else:
        df["BullishPatternCount"] = df["BearishPatternCount"] = df["PatternScore"] = 0.0

    # -------------------------------
    # 2) Wick Rejection Filter
    # -------------------------------
    df["UpperWick"] = df["High"] - df[["Close", "Open"]].max(axis=1)
    df["LowerWick"] = df[["Close", "Open"]].min(axis=1) - df["Low"]
    df["Body"]      = (df["Close"] - df["Open"]).abs()
    df["WickRejection"] = (df["LowerWick"] > df["Body"] * 1.5) | (df["UpperWick"] > df["Body"] * 1.5)

    # -------------------------------
    # 3) Momentum & Thresholds
    # -------------------------------
    df["Return"]     = df["Close"].pct_change()
    df["AvgReturn"]  = df["Return"].rolling(10, min_periods=1).mean()
    df["Volatility"] = df["Return"].rolling(10, min_periods=1).std().fillna(0)
    df["MomentumZ"]  = (df["Return"] - df["AvgReturn"]) / df["Volatility"].replace(0, 1)

    # -------------------------------
    # 3b) Dynamic Thresholds
    # -------------------------------
    factor = 0.5  # smaller factor to catch more signals .01 doesn't give enough of a change
    # Compute global thresholds for the single company
    mean_momentum = df["MomentumZ"].mean()
    std_momentum  = df["MomentumZ"].std()
    
    df["BuyThresh"]  = mean_momentum + factor * std_momentum
    df["SellThresh"] = mean_momentum - factor * std_momentum
    
    # -------------------------------
    # 4) Candidate Action
    # -------------------------------
    df["CandidateAction"] = "Hold"
    df.loc[(df["PatternScore"] > 0) & (df["MomentumZ"] > df["BuyThresh"]), "CandidateAction"] = "Buy"
    df.loc[(df["PatternScore"] < 0) & (df["MomentumZ"] < df["SellThresh"]), "CandidateAction"] = "Sell"

    # -------------------------------
    # 5) Alternating Filter
    # -------------------------------
    filtered = []
    last = None
    for action in df["CandidateAction"]:
        if action in ["Buy", "Sell"]:
            if action == last:  # prevent duplicate Buy-Buy or Sell-Sell
                filtered.append("Hold")
            else:
                filtered.append(action)
                last = action
        else:
            filtered.append("Hold")
    df["Action"] = filtered

    # -------------------------------
    # 6) Predictive Shift
    # -------------------------------
    df["TomorrowAction"] = df["Action"].shift(-1).fillna("Hold")

    # -------------------------------
    # 7) Confidence, Duration, Flags
    # -------------------------------
    df["SignalStrength"] = (df["PatternScore"].astype(float).abs() + df["MomentumZ"].astype(float).abs())
    max_s = df["SignalStrength"].max()
    df["ActionConfidence"] = (df["SignalStrength"] / max_s if max_s > 0 else 0).fillna(0)

    df["SignalDuration"] = (df["Action"] != df["Action"].shift()).cumsum()
    df["ValidAction"]    = df["Action"].isin(["Buy", "Sell"])
    df["HasValidSignal"] = df[["Action", "TomorrowAction", "SignalStrength"]].notna().all(axis=1)

    # -------------------------------
    # 8) Force all numeric columns to float (PySpark-safe)
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
# 4️⃣ Action Classification
# -------------------------------
def classify_action(row):
    buy_signals = any([
        row.get("ValidHammer_long", False),
        row.get("ValidBullishEngulfing_swing", False),
        row.get("ValidPiercingLine_daily", False)
    ])
    sell_signals = any([
        row.get("ValidShootingStar_long", False),
        row.get("ValidBearishEngulfing_swing", False),
        row.get("ValidDarkCloud_daily", False)
    ])
    if buy_signals:
        return "Buy"
    elif sell_signals:
        return "Sell"
    else:
        return "Hold"

def add_batch_metadata(df, company_id, ingest_ts=None):
    df = df.copy()  # Ensure safe memory layout

    # Default timestamp if not provided
    if ingest_ts is None:
        ingest_ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # Add metadata columns in one step
    metadata = {
        "BatchId": f"{company_id}_{ingest_ts}",
        "IngestedAt": ingest_ts,
        "CompanyId": company_id
    }

    for key, value in metadata.items():
        df[key] = value

    return df