import pandas as pd
import numpy as np
from datetime import datetime
import re
from bsf_settings import load_settings
import unicodedata
from functools import reduce
from operator import itemgetter


def get_candle_params(df: pd.DataFrame, profile: str = "default", close_col: str = "Close") -> pd.DataFrame:
    """
    Optimized version: Collect all new columns in a dict and concat once at the end.
    """
    required_cols = ["CompanyId", close_col, "StockDate"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Input DataFrame missing columns: {missing_cols}")
    if profile is None:
        raise ValueError("profile ID cannot be None")

    user_settings = load_settings(profile).get("candle_params", {
        "doji_base": 0.1, "doji_scale": 0.05, "doji_min": 0.05, "doji_max": 0.2,
        "long_body_base": 0.7, "long_body_scale": 0.1, "long_body_min": 0.6, "long_body_max": 0.9,
        "small_body_base": 0.3, "small_body_scale": 0.05, "small_body_min": 0.2, "small_body_max": 0.4,
        "shadow_ratio_base": 2.0, "shadow_ratio_scale": 0.5, "shadow_ratio_min": 1.5, "shadow_ratio_max": 3.0,
        "near_edge": 0.05, "highvol_spike": 2.0, "lowvol_dip": 0.5,
        "hammer_base": 0.2, "hammer_scale": 0.05, "hammer_min": 0.1, "hammer_max": 0.3,
        "marubozu_base": 0.1, "marubozu_scale": 0.05, "marubozu_min": 0.05, "marubozu_max": 0.2,
        "rng_base": 0.05, "rng_scale": 0.02, "rng_min": 0.03, "rng_max": 0.1
    })

    df[close_col] = pd.to_numeric(df[close_col], errors="coerce").fillna(1e-6)

    logp = np.log10(np.maximum(df[close_col].values, 1e-6))
    scale_factor = (logp + 6) / 8

    def threshold(base, scale, min_val, max_val):
        return np.clip(base + scale * scale_factor, min_val, max_val)

    # Collect new columns
    new_cols = {}
    new_cols["doji_thresh"] = threshold(user_settings["doji_base"], user_settings["doji_scale"],
                                      user_settings["doji_min"], user_settings["doji_max"])
    new_cols["long_body"] = threshold(user_settings["long_body_base"], user_settings["long_body_scale"],
                                      user_settings["long_body_min"], user_settings["long_body_max"])
    new_cols["small_body"] = threshold(user_settings["small_body_base"], user_settings["small_body_scale"],
                                      user_settings["small_body_min"], user_settings["small_body_max"])
    new_cols["shadow_ratio"] = threshold(user_settings["shadow_ratio_base"], user_settings["shadow_ratio_scale"],
                                      user_settings["shadow_ratio_min"], user_settings["shadow_ratio_max"])
    new_cols["near_edge"] = np.full(len(df), user_settings["near_edge"])
    new_cols["highvol_spike"] = np.full(len(df), user_settings["highvol_spike"])
    new_cols["lowvol_dip"] = np.full(len(df), user_settings["lowvol_dip"])
    new_cols["hammer_thresh"] = threshold(user_settings["hammer_base"], user_settings["hammer_scale"],
                                      user_settings["hammer_min"], user_settings["hammer_max"])
    new_cols["marubozu_thresh"] = threshold(user_settings["marubozu_base"], user_settings["marubozu_scale"],
                                      user_settings["marubozu_min"], user_settings["marubozu_max"])
    new_cols["rng_thresh"] = threshold(user_settings["rng_base"], user_settings["rng_scale"],
                                      user_settings["rng_min"], user_settings["rng_max"])

    # Concat once
    new_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, new_df], axis=1)

def get_required_columns(df, show_columns):
    # Base requirements
    required_trend = [
        "UpTrend_MA",
        "DownTrend_MA",
        "DownTrend_Return"
    ]

    required_patterns = [
        "Hammer", "BullishEngulfing", "PiercingLine", "MorningStar", "ThreeWhiteSoldiers",
        "BullishMarubozu", "TweezerBottom", "ShootingStar", "BearishEngulfing", "DarkCloudCover",
        "EveningStar", "ThreeBlackCrows", "BearishMarubozu", "TweezerTop",
        "HaramiCross", "BullishHarami", "BearishHarami",
        "InsideBar", "OutsideBar", "RisingThreeMethods", "FallingThreeMethods",
        "GapUp", "GapDown", "SpinningTop", "ClimacticCandle",
        "DragonflyDoji", "GravestoneDoji"
    ]

    # Keep only the ones that exist in df
    required_trend = [col for col in required_trend if col in df.columns]
    required_patterns = [col for col in required_patterns if col in df.columns]

    if show_columns:
        print("Trends used:", required_trend)
        print("Patterns used:", required_patterns)
    return required_trend, required_patterns
    
def generate_signal_columns_optimized(df, timeframe="Short", profile: str = "default"):
    """
    Optimized version: Use aggregate-then-merge for last_close and momentum.
    """
    if profile is None:
        raise ValueError("profile ID cannot be None")
    if df.empty or not all(col in df.columns for col in ["CompanyId", "StockDate", "Close"]):
        raise ValueError("Input DataFrame must contain CompanyId, StockDate, and Close columns")

    settings = load_settings(str(profile))["signals"]
    tf_settings = settings["timeframes"].get(timeframe, settings["timeframes"]["Daily"])
    
    if not all(key in tf_settings for key in ["momentum", "buy", "sell"]):
        raise ValueError("Timeframe settings must include momentum, buy, and sell keys")
    if not all(key in settings["penny_stock_adjustment"] for key in ["threshold", "factor", "min_momentum"]):
        raise ValueError("Penny stock settings must include threshold, factor, and min_momentum")

    momentum_factor_base = tf_settings["momentum"]
    ps = settings["penny_stock_adjustment"]

    # Aggregate last_close using groupby.last (faster than sort + first)
    df_last = df.groupby("CompanyId", as_index=False)["Close"].last().rename(columns={"Close": "last_close"})
    
    # Apply penny stock adjustment
    df_momentum = df_last.copy()
    df_momentum["momentum_factor"] = np.where(
        df_momentum["last_close"] < ps["threshold"],
        np.maximum(momentum_factor_base * ps["factor"], ps["min_momentum"]),
        momentum_factor_base
    )
    
    # Create dict
    momentum_dict = dict(zip(df_momentum["CompanyId"], df_momentum["momentum_factor"]))
    
    # Candle and trend columns (no change, as it's column filtering)
    all_columns = df.columns.tolist()
    candle_cols = [
        "Doji", "Hammer", "InvertedHammer", "BullishMarubozu", "BearishMarubozu", 
        "HangingMan", "ShootingStar", "SpinningTop", "BullishEngulfing", "BearishEngulfing",
        "BullishHarami", "BearishHarami", "HaramiCross", "PiercingLine", "DarkCloudCover",
        "MorningStar", "EveningStar", "ThreeWhiteSoldiers", "ThreeBlackCrows", "TweezerTop",
        "TweezerBottom", "InsideBar", "OutsideBar", "NearHigh", "NearLow", "DragonflyDoji",
        "GravestoneDoji", "LongLeggedDoji", "RisingThreeMethods", "FallingThreeMethods",
        "GapUp", "GapDown", "ClimacticCandle"
    ]
    candle_cols = [col for col in candle_cols if col in all_columns]
    
    candle_columns = {
        "Buy": [col for col in candle_cols if any(k.lower() in col.lower() for k in tf_settings["buy"])],
        "Sell": [col for col in candle_cols if any(k.lower() in col.lower() for k in tf_settings["sell"])]
    }

    trend_columns = {
        "Bullish": [col for col in all_columns if col in ["MomentumUp", "ConfirmedUpTrend", "UpTrend_MA"]],
        "Bearish": [col for col in all_columns if col in ["MomentumDown", "ConfirmedDownTrend", "DownTrend_MA"]]
    }

    if not candle_columns["Buy"] and not candle_columns["Sell"]:
        print(f"Warning: No valid candle columns found for timeframe {timeframe} and profile {profile}")
    if not trend_columns["Bullish"] and not trend_columns["Bearish"]:
        print(f"Warning: No valid trend columns found for timeframe {timeframe} and profile {profile}")

    return candle_columns, trend_columns, momentum_factor_base, momentum_dict 



def step1_add_candle_patterns_dynamic(df, tf_window=5, profile: str = "default",
                                      open_col="Open", high_col="High", low_col="Low",
                                      close_col="Close", volume_col="Volume", max_lag=4):
    """
    Optimized and fully dynamic:
    - All companies processed at once
    - Lag columns generated dynamically
    - Multi-bar patterns use dynamically generated lags
    """
    import numpy as np

    # --- Check required columns ---
    required_cols = ["CompanyId", "StockDate", open_col, high_col, low_col, close_col, volume_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Input DataFrame missing columns: {missing_cols}")
    if tf_window < 1:
        raise ValueError("tf_window must be positive")
    if profile is None:
        raise ValueError("profile ID cannot be None")

    df = df.sort_values(["CompanyId", "StockDate"])
    o, h, l, c, v = open_col, high_col, low_col, close_col, volume_col

    # --- Step 2: get candle parameters ---
    df = get_candle_params(df, profile=profile, close_col=close_col)

    group = df.groupby("CompanyId")
    new_cols = {}

    # --- Rolling calculations ---
    new_cols["O_roll"] = group[o].transform(lambda x: x.shift(tf_window - 1).fillna(x.iloc[0]) if len(x) > 0 else x)
    new_cols["C_roll"] = df[c].copy()
    new_cols["H_roll"] = group[h].rolling(tf_window, min_periods=1).max().reset_index(0, drop=True)
    new_cols["L_roll"] = group[l].rolling(tf_window, min_periods=1).min().reset_index(0, drop=True)
    new_cols["V_avg20"] = group[v].rolling(20, min_periods=1).mean().reset_index(0, drop=True)

    # --- Volume spikes ---
    new_cols["HighVolume"] = df[v] > df["highvol_spike"] * new_cols["V_avg20"]
    new_cols["LowVolume"] = df[v] < df["lowvol_dip"] * new_cols["V_avg20"]

    # --- Body, shadows, range, Bull/Bear ---
    C, O, H, L = new_cols["C_roll"], new_cols["O_roll"], new_cols["H_roll"], new_cols["L_roll"]
    new_cols["Body"] = np.where(C != 0, np.abs(C - O) / C, 0.0)
    new_cols["UpShadow"] = np.where(C != 0, (H - np.maximum(O, C)) / C, 0.0)
    new_cols["DownShadow"] = np.where(C != 0, (np.minimum(O, C) - L) / C, 0.0)
    new_cols["Range"] = np.where(C != 0, (H - L) / C, 0.0)
    new_cols["Bull"] = C > O
    new_cols["Bear"] = O > C

    # --- Trend detection ---
    new_cols["UpTrend"] = group[c].transform(lambda x: x > x.shift(tf_window - 1).fillna(x.iloc[0]) if len(x) >= tf_window else False)
    new_cols["DownTrend"] = group[c].transform(lambda x: x < x.shift(tf_window - 1).fillna(x.iloc[0]) if len(x) >= tf_window else False)

    # --- Single-bar patterns ---
    B, R, US, DS = new_cols["Body"], new_cols["Range"], new_cols["UpShadow"], new_cols["DownShadow"]
    new_cols["Doji"] = B <= df["doji_thresh"] * R
    new_cols["Hammer"] = (DS >= df["shadow_ratio"] * B) & (US <= df["hammer_thresh"] * B) & (B > 0) & new_cols["DownTrend"]
    new_cols["InvertedHammer"] = (US >= df["shadow_ratio"] * B) & (DS <= df["hammer_thresh"] * B) & (B > 0) & new_cols["DownTrend"]
    new_cols["BullishMarubozu"] = new_cols["Bull"] & (B >= df["long_body"] * R) & (US <= df["marubozu_thresh"] * R) & (DS <= df["marubozu_thresh"] * R)
    new_cols["BearishMarubozu"] = new_cols["Bear"] & (B >= df["long_body"] * R) & (US <= df["marubozu_thresh"] * R) & (DS <= df["marubozu_thresh"] * R)
    new_cols["SpinningTop"] = (B <= df["small_body"] * R) & (US >= B) & (DS >= B)
    '''
    # --- Dynamic lags ---
    lag_cols = ["O_roll", "C_roll", "H_roll", "L_roll", "Bull", "Bear", "Body"]
    for col in lag_cols:
        is_bool = col in ["Bull", "Bear"]
        for lag in range(1, max_lag + 1):
            new_cols[f"{col}{lag}"] = group[col].shift(lag).fillna(False if is_bool else 0.0)
    '''
    df = df.assign(**new_cols)
    group = df.groupby("CompanyId")  # now group sees them


    # Now lag generation works
    lag_cols = ["O_roll", "C_roll", "H_roll", "L_roll", "Bull", "Bear", "Body"]
    for col in lag_cols:
        is_bool = col in ["Bull", "Bear"]
        for lag in range(1, max_lag + 1):
            new_cols[f"{col}{lag}"] = group[col].shift(lag).fillna(False if is_bool else 0.0)
 

    
    # --- Dynamic multi-bar patterns ---
    for lag in range(1, max_lag):
        Olag, Clag = new_cols[f"O_roll{lag}"], new_cols[f"C_roll{lag}"]
        if lag == 1:
            new_cols.update({
                "BullishEngulfing": (Olag > Clag) & new_cols["Bull"] & (C >= Olag) & (O <= Clag),
                "BearishEngulfing": (Clag > Olag) & new_cols["Bear"] & (O >= Clag) & (C <= Olag),
                "BullishHarami": (Olag > Clag) & new_cols["Bull"] & (np.maximum(O, C) <= np.maximum(Olag, Clag)) & (np.minimum(O, C) >= np.minimum(Olag, Clag)),
                "BearishHarami": (Clag > Olag) & new_cols["Bear"] & (np.maximum(O, C) <= np.maximum(Olag, Clag)) & (np.minimum(O, C) >= np.minimum(Olag, Clag))
            })

    # Three White/Black Soldiers
    if max_lag >= 2:
        new_cols["ThreeWhiteSoldiers"] = new_cols["Bull"] & new_cols["Bull1"] & new_cols["Bull2"] & (C > new_cols["C_roll1"]) & (new_cols["C_roll1"] > new_cols["C_roll2"])
        new_cols["ThreeBlackCrows"] = new_cols["Bear"] & new_cols["Bear1"] & new_cols["Bear2"] & (C < new_cols["C_roll1"]) & (new_cols["C_roll1"] < new_cols["C_roll2"])

    # --- Gap & Climactic ---
    new_cols["GapUp"] = O > new_cols["H_roll1"]
    new_cols["GapDown"] = O < new_cols["L_roll1"]
    new_cols["RangeMean"] = group["Range"].rolling(tf_window, min_periods=1).mean().reset_index(0, drop=True)
    new_cols["ClimacticCandle"] = new_cols["Range"] > 2 * new_cols["RangeMean"]

    # --- Near edge ---
    new_cols["NearHigh"] = group["H_roll"].transform(lambda x: x >= x.rolling(tf_window, min_periods=1).max() * (1 - df["near_edge"].loc[x.index]))
    new_cols["NearLow"] = group["L_roll"].transform(lambda x: x <= x.rolling(tf_window, min_periods=1).min() * (1 + df["near_edge"].loc[x.index]))

    # --- Assign all new columns at once ---
    df = df.assign(**new_cols)

    # --- Pattern count/type ---
    pattern_cols = [
        "Doji", "Hammer", "InvertedHammer", "BullishMarubozu", "BearishMarubozu",
        "SpinningTop", "BullishEngulfing", "BearishEngulfing", "BullishHarami",
        "BearishHarami", "ThreeWhiteSoldiers", "ThreeBlackCrows", "GapUp", "GapDown",
        "ClimacticCandle", "NearHigh", "NearLow"
    ]
    df["PatternCount"] = df[pattern_cols].sum(axis=1)
    df[pattern_cols] = df[pattern_cols].astype(int)
    df["PatternType"] = df[pattern_cols].idxmax(axis=1, skipna=True).where(df["PatternCount"] > 0, "None")
    
    
    print("\nStep1 columns:")
    for c in df.columns:
        print(c)
        
    return df


import numpy as np

def step2_add_trend_filters_optimized(df, timeframe="Daily", profile: str = "default"):
    """
    Optimized version: vectorized groupby.rolling/shift calculations.
    Collect all new columns first and append them to df at once.
    """
    if profile is None:
        raise ValueError("profile ID cannot be None")
    required_cols = ["CompanyId", "StockDate", "Close"]
    if df.empty or not all(col in df.columns for col in required_cols):
        raise ValueError(f"Input DataFrame must contain columns: {required_cols}")

    c = "Close"

    # Load settings
    settings = load_settings(profile)["profiles"]
    if timeframe not in settings:
        raise ValueError(f"Invalid timeframe: {timeframe}. Must be one of {list(settings.keys())}")
    params = settings[timeframe]
    required_keys = ["ma", "ret", "vol", "roc_thresh", "slope_horizon"]
    if not all(key in params for key in required_keys):
        raise ValueError(f"Settings for timeframe {timeframe} must include {required_keys}")

    ma_window = params["ma"]
    ret_window = params["ret"]
    vol_window = params["vol"]
    roc_thresh = params["roc_thresh"]
    slope_horizon = params["slope_horizon"]
    if any(w <= 0 for w in [ma_window, ret_window, vol_window, slope_horizon]):
        raise ValueError("Window sizes must be positive")

    df = df.sort_values(["CompanyId", "StockDate"])
    group = df.groupby("CompanyId")

    # Dictionary to collect new columns
    new_cols = {}

    # --- Moving Average & slope ---
    new_cols["MA"] = group[c].rolling(ma_window, min_periods=1).mean().reset_index(0, drop=True)
    ma_lag = new_cols["MA"].groupby(df["CompanyId"]).shift(slope_horizon)
    new_cols["MA_slope"] = np.where(ma_lag.notnull(), (new_cols["MA"] - ma_lag) / ma_lag, 0.0)
    new_cols["UpTrend_MA"] = new_cols["MA_slope"] > 0
    new_cols["DownTrend_MA"] = new_cols["MA_slope"] < 0

    # --- Returns ---
    recent_shift = group[c].shift(ret_window)
    new_cols["RecentReturn"] = np.where(recent_shift.notnull(), (df[c] - recent_shift) / recent_shift, 0.0)
    new_cols["UpTrend_Return"] = new_cols["RecentReturn"] > 0
    new_cols["DownTrend_Return"] = new_cols["RecentReturn"] < 0

    # --- ReturnPct & Volatility ---
    ret_shift1 = group[c].shift(1)
    new_cols["ReturnPct"] = np.where(ret_shift1.notnull(), (df[c] - ret_shift1) / ret_shift1, 0.0)

    vol = group["Close"].apply(
        lambda x: x.pct_change().rolling(vol_window, min_periods=1).std()
    ).reset_index(level=0, drop=True)
    new_cols["Volatility"] = vol

    median_vol = vol.groupby(df["CompanyId"]).transform("median")
    new_cols["LowVolatility"] = vol < median_vol
    new_cols["HighVolatility"] = vol > median_vol

    # --- Rate of Change & Momentum ---
    roc_shift = group[c].shift(ma_window)
    new_cols["ROC"] = np.where(roc_shift.notnull(), (df[c] - roc_shift) / roc_shift, 0.0)
    new_cols["MomentumUp"] = new_cols["ROC"] > roc_thresh
    new_cols["MomentumDown"] = new_cols["ROC"] < -roc_thresh

    # --- Confirmed trends ---
    up_sum = (new_cols["UpTrend_MA"].astype(int) +
              new_cols["UpTrend_Return"].astype(int) +
              new_cols["MomentumUp"].astype(int))
    down_sum = (new_cols["DownTrend_MA"].astype(int) +
                new_cols["DownTrend_Return"].astype(int) +
                new_cols["MomentumDown"].astype(int))
    new_cols["ConfirmedUpTrend"] = up_sum >= 2
    new_cols["ConfirmedDownTrend"] = down_sum >= 2

    # --- Append all new columns at once ---
    df = df.assign(**new_cols)
    print("\nStep2 columns:")
    for c in df.columns:
        print(c)
    return df



    



def step3_add_confirmed_signals_optimized(df, verbose=True):
    """
    Optimized version: Collect new columns in dict and assign at end.
    Vectorized, automatically skips missing columns.
    """

    print("Columns in df:", df.columns.tolist())

    # Get only the required columns that exist in df
    required_trend, required_patterns = get_required_columns(df, show_columns=verbose)

    # Ensure all required trend columns exist
    for col in required_trend + required_patterns + ["HighVolume"]:
        if col not in df.columns:
            df[col] = False
            if verbose:
                print(f"⚠️ Adding missing column {col} as False")

    new_cols = {}

    # Signal groups mapping
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
            "ValidDarkCloud": "UpTrend_MA",
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
            "ValidSpinningTop": "UpTrend_MA",
            "ValidClimacticCandle": "UpTrend_MA",
        }
    }

    # Generate signals safely
    for patterns in signal_groups.values():
        for valid_col, trend_col in patterns.items():
            raw_col = valid_col.replace("Valid", "")
            if raw_col in df.columns and trend_col in df.columns:
                new_cols[valid_col] = df[raw_col] & df[trend_col]
            elif verbose:
                print(f"⚠️ Skipping {valid_col}: {raw_col} or {trend_col} missing")

    # Dragonfly & Gravestone Doji with volume filter
    for doji, trend, col_name in [
        ("DragonflyDoji", "DownTrend_MA", "ValidDragonflyDoji"),
        ("GravestoneDoji", "UpTrend_MA", "ValidGravestoneDoji")
    ]:
        if doji in df.columns and trend in df.columns and "HighVolume" in df.columns:
            new_cols[col_name] = df[doji] & df[trend] & df["HighVolume"]
        elif verbose:
            print(f"⚠️ Skipping {col_name}: required columns missing")

    # Assign all new columns at once
    if new_cols:
        df = df.assign(**new_cols)

        
    print("\nStep3 columns:")
    for c in df.columns:
        print(c)
    return df

    
def step4_compute_fundamental_score_optimized(df, profile: str = "default"):
    """
    Fully vectorized fundamental score computation for multiple companies.
    Creates normalized metrics and weighted fundamental score.
    """
    import numpy as np
    import pandas as pd

    if profile is None:
        raise ValueError("profile ID cannot be None")

    required_cols = [
        "CompanyId", "PeRatio", "PbRatio", "PegRatio", "ReturnOnEquity",
        "GrossMarginTTM", "NetProfitMarginTTM", "TotalDebtToEquity",
        "CurrentRatio", "InterestCoverage", "EpsChangeYear", "RevChangeYear",
        "Beta", "ShortIntToFloat"
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Input DataFrame missing columns: {missing_cols}")

    # Load weights
    user_settings = load_settings(profile).get("fundamental_weights", {})
    weights = {
        "valuation": user_settings.get("valuation", 0.2),
        "profitability": user_settings.get("profitability", 0.3),
        "DebtLiquidity": user_settings.get("DebtLiquidity", 0.2),
        "Growth": user_settings.get("Growth", 0.2),
        "Sentiment": user_settings.get("Sentiment", 0.1),
    }
    total_w = sum(weights.values())
    if abs(total_w - 1.0) > 1e-6:
        weights = {k: v / total_w for k, v in weights.items()}

    numeric_cols = required_cols[1:]  # skip CompanyId

    # Compute per-company min/max
    #agg_min = df.groupby("CompanyId")[numeric_cols].transform("min")
    #agg_max = df.groupby("CompanyId")[numeric_cols].transform("max")

    # Normalization map: True = invert (smaller is better)
    norm_map = {
        "PeRatio": True, "PbRatio": True, "PegRatio": True,
        "ReturnOnEquity": False, "GrossMarginTTM": False, "NetProfitMarginTTM": False,
        "TotalDebtToEquity": True, "CurrentRatio": False, "InterestCoverage": False,
        "EpsChangeYear": False, "RevChangeYear": False,
        "Beta": True, "ShortIntToFloat": False
    }

    # Per-company min/max
    agg_min = df.groupby("CompanyId")[numeric_cols].transform("min")
    agg_max = df.groupby("CompanyId")[numeric_cols].transform("max")
        
    # Normalize
    for col, invert in norm_map.items():
        normalized = (df[col] - agg_min[col]) / (agg_max[col] - agg_min[col])
        normalized = normalized.fillna(0.0)
        normalized[agg_max[col] == agg_min[col]] = 0.0
        if invert:
            normalized = 1.0 - normalized
        df[f"{col.lower()}_norm"] = normalized


    # Weighted score
    df['FundamentalScore'] = (
        weights["valuation"] * (df["peratio_norm"] + df["pbratio_norm"] + df["pegratio_norm"]) / 3 +
        weights["profitability"] * (df["returnonequity_norm"] + df["grossmarginttm_norm"] + df["netprofitmarginttm_norm"]) / 3 +
        weights["DebtLiquidity"] * (df["totaldebttoequity_norm"] + df["currentratio_norm"] + df["interestcoverage_norm"]) / 3 +
        weights["Growth"] * (df["epschangeyear_norm"] + df["revchangeyear_norm"]) / 2 +
        weights["Sentiment"] * (df["beta_norm"] + df["shortinttofloat_norm"]) / 2
    ).fillna(0.0)

    # Flag rows with any missing normalized values
    norm_cols_lower = [c.lower() + "_norm" for c in numeric_cols]
    df['FundamentalBad'] = df[norm_cols_lower].isna().any(axis=1)
    print("\nStep4 columns:")
    for c in df.columns:
        print(c)
    return df



    
def step5_add_signal_strength_vectorized(df, tf, tf_window=5, use_fundamentals=True, profile="default"):
    """
    Vectorized feature creation for XGBoost using a new_cols dict:
    - Computes momentum, pattern counts, signal strength features
    - Optionally includes fundamental score
    - Keeps all original columns untouched
    """
    import numpy as np
    import pandas as pd

    if profile is None:
        raise ValueError("profile ID cannot be None")

    required_cols = ["CompanyId", "StockDate", "Close"]
    if df.empty or not all(c in df.columns for c in required_cols):
        raise ValueError(f"Input DataFrame must contain {required_cols}")

    df = df.sort_values(['CompanyId', 'StockDate']).reset_index(drop=True)
    new_cols = {}

    # ----------------------
    # 1️⃣ Momentum features
    # ----------------------
    new_cols['Return'] = df.groupby('CompanyId')['Close'].pct_change().fillna(0.0)
    new_cols['MomentumZ'] = (new_cols['Return'] - new_cols['Return'].groupby(df['CompanyId']).transform('mean')) / \
                            new_cols['Return'].groupby(df['CompanyId']).transform('std').replace(0, 1e-8)

    # ----------------------
    # 2️⃣ Pattern counts
    # ----------------------
    valid_cols = [c for c in df.columns if c.startswith("Valid")]
    bull_keywords = ["Bull","Hammer","MorningStar","ThreeWhiteSoldiers","TweezerBottom","DragonflyDoji"]
    bear_keywords = ["Bear","ShootingStar","EveningStar","ThreeBlackCrows","TweezerTop","GravestoneDoji","DarkCloudCover"]

    bull_cols = [c for c in valid_cols if any(k.lower() in c.lower() for k in bull_keywords)]
    bear_cols = [c for c in valid_cols if any(k.lower() in c.lower() for k in bear_keywords)]

    new_cols['BullCount'] = df[bull_cols].sum(axis=1) if bull_cols else 0
    new_cols['BearCount'] = df[bear_cols].sum(axis=1) if bear_cols else 0
    new_cols['PatternScoreNorm'] = (new_cols['BullCount'] - new_cols['BearCount']) / tf_window

    # ----------------------
    # 3️⃣ Fundamental score
    # ----------------------
    if use_fundamentals and 'FundamentalScore' in df.columns:
        new_cols['FundamentalScore'] = df['FundamentalScore']

    # ----------------------
    # 4️⃣ Signal strength
    # ----------------------
    for col in ["ConfirmedUpTrend", "ConfirmedDownTrend"]:
        if col not in df.columns:
            df[col] = False

    signal_strength = df[valid_cols].sum(axis=1)
    signal_strength_nonzero = signal_strength.replace(0, 1)
    new_cols['SignalStrength'] = signal_strength * (
        1.0 + df['ConfirmedUpTrend'].astype(float) * 0.2 + df['ConfirmedDownTrend'].astype(float) * 0.2
    )

    new_cols['BullishPctRaw'] = df[bull_cols].sum(axis=1) / signal_strength_nonzero if bull_cols else 0
    new_cols['BearishPctRaw'] = df[bear_cols].sum(axis=1) / signal_strength_nonzero if bear_cols else 0

    directional_groups = load_settings(profile)["signals"]["timeframes"][tf].get(
        "directional_groups", ["Bullish", "Bearish", "Reversal", "Continuation"]
    )
    directional_cols = [c for c in valid_cols if any(c.startswith(f"Valid{g}") for g in directional_groups)]
    dir_sum_nonzero = df[directional_cols].sum(axis=1).replace(0, 1) if directional_cols else pd.Series(1, index=df.index)

    bullish_dir_cols = [c for c in directional_cols if any(k.lower() in c.lower() for k in bull_keywords)]
    bearish_dir_cols = [c for c in directional_cols if any(k.lower() in c.lower() for k in bear_keywords)]

    new_cols['BullishPctDirectional'] = df[bullish_dir_cols].sum(axis=1) / dir_sum_nonzero if bullish_dir_cols else 0
    new_cols['BearishPctDirectional'] = df[bearish_dir_cols].sum(axis=1) / dir_sum_nonzero if bearish_dir_cols else 0

    # ----------------------
    # 5️⃣ Assign all new columns back
    # ----------------------
    for k, v in new_cols.items():
        df[k] = v
    print("\nStep5 columns:")
    for c in df.columns:
        print(c)
    return df

def select_ml_columns(df, tf, tf_window=5, profile="default"):
    # ----------------------------
    # Columns to keep for XGBoost
    # ----------------------------
    xgb_columns = [
        # Identity / OHLCV
        "CompanyId", "StockDate", "Open", "High", "Low", "Close", "Volume",
        
        # Lagged OHLC + Bull/Bear/Body
        *[f"O_roll{i}" for i in range(1, 5)],
        *[f"C_roll{i}" for i in range(1, 5)],
        *[f"H_roll{i}" for i in range(1, 5)],
        *[f"L_roll{i}" for i in range(1, 5)],
        *[f"Bull{i}" for i in range(1, 5)],
        *[f"Bear{i}" for i in range(1, 5)],
        *[f"Body{i}" for i in range(1, 5)],
        
        # Immediate candlestick features
        "Body", "UpShadow", "DownShadow", "Range",
        
        # Pattern flags
        "Doji", "Hammer", "InvertedHammer", "BullishMarubozu", "BearishMarubozu", "SpinningTop",
        "BullishEngulfing", "BearishEngulfing", "BullishHarami", "BearishHarami",
        "ThreeWhiteSoldiers", "ThreeBlackCrows", "GapUp", "GapDown", "ClimacticCandle", "NearHigh", "NearLow",
        
        # Pattern summary stats
        "PatternType", "PatternCount", "PatternScoreNorm", "SignalStrength",
        "BullCount","BearCount",
        "BullishPctRaw", "BearishPctRaw", "BullishPctDirectional", "BearishPctDirectional",
        
        # Trend / momentum
        "MA", "MA_slope", "UpTrend_MA", "DownTrend_MA", "RecentReturn", "UpTrend_Return",
        "DownTrend_Return", "ReturnPct", "Volatility", "LowVolatility", "HighVolatility",
        "ROC", "MomentumUp", "MomentumDown", "ConfirmedUpTrend", "ConfirmedDownTrend",
        "Return", "MomentumZ",
        
        # Fundamentals
        "FundamentalScore", "FundamentalBad",
        "peratio_norm", "pbratio_norm", "pegratio_norm", "returnonequity_norm",
        "grossmarginttm_norm", "netprofitmarginttm_norm", "totaldebttoequity_norm",
        "currentratio_norm", "interestcoverage_norm", "epschangeyear_norm",
        "revchangeyear_norm", "beta_norm", "shortinttofloat_norm"
    ]
    
    # ----------------------------
    # Select columns present in the DataFrame
    # ----------------------------
    xgb_columns_present = [c for c in xgb_columns if c in df.columns]
    
    # ----------------------------
    # Create XGBoost-ready DataFrame
    # ----------------------------
    df_xgb = df[xgb_columns_present].copy()
    
    return df_xgb



def add_batch_metadata_optimized(df, timeframe, user: int = 1, profile: str = "default", ingest_ts=None):
    """
    Optimized version: Assign multiple columns at once.
    """
    if ingest_ts is None:
        ingest_ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    df = df.assign(
        BatchId=f"{user}_{ingest_ts}",
        IngestedAt=ingest_ts,
        TimeFrame=timeframe,
        Profile=profile,
        UserId=user
    )
    return df