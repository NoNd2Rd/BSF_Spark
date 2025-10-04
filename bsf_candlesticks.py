import pandas as pd
import numpy as np
from datetime import datetime
import re
from bsf_settings import load_settings
import unicodedata
from functools import reduce
from operator import itemgetter

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

def get_candle_params_optimized(df: pd.DataFrame, profile: str = "default", close_col: str = "Close") -> pd.DataFrame:
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

def add_candle_patterns_optimized(df, tf_window=5, profile: str = "default", 
                            open_col="Open", high_col="High", low_col="Low", 
                            close_col="Close", volume_col="Volume"):
    """
    Optimized version: Use groupby.rolling and transform; collect lags and new columns where possible; use idxmax for PatternType.
    """
    required_cols = ["CompanyId", open_col, high_col, low_col, close_col, volume_col, "StockDate"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Input DataFrame missing columns: {missing_cols}")
    if tf_window < 1:
        raise ValueError("tf_window must be positive")
    if profile is None:
        raise ValueError("profile ID cannot be None")

    df = df.sort_values(["CompanyId", "StockDate"])

    o, h, l, c, v = open_col, high_col, low_col, close_col, volume_col

    df = get_candle_params_optimized(df, profile=profile, close_col=close_col)

    group = df.groupby("CompanyId")

    # Rolling calculations using groupby.rolling
    df["O_roll"] = group[o].transform(lambda x: x.shift(tf_window - 1).fillna(x.iloc[0]) if len(x) > 0 else x)
    df["C_roll"] = df[c].copy()  # Current close
    df["H_roll"] = group[h].rolling(tf_window, min_periods=1).max().reset_index(0, drop=True)
    df["L_roll"] = group[l].rolling(tf_window, min_periods=1).min().reset_index(0, drop=True)
    df["V_avg20"] = group[v].rolling(20, min_periods=1).mean().reset_index(0, drop=True)

    # Volume spikes
    df["HighVolume"] = df[v] > df["highvol_spike"] * df["V_avg20"]
    df["LowVolume"] = df[v] < df["lowvol_dip"] * df["V_avg20"]

    # Body, shadows, range
    df["Body"] = np.where(df["C_roll"] != 0, np.abs(df["C_roll"] - df["O_roll"]) / df["C_roll"], 0.0)
    df["UpShadow"] = np.where(df["C_roll"] != 0, 
                             (df["H_roll"] - np.maximum(df["O_roll"], df["C_roll"])) / df["C_roll"], 0.0)
    df["DownShadow"] = np.where(df["C_roll"] != 0, 
                               (np.minimum(df["O_roll"], df["C_roll"]) - df["L_roll"]) / df["C_roll"], 0.0)
    df["Range"] = np.where(df["C_roll"] != 0, (df["H_roll"] - df["L_roll"]) / df["C_roll"], 0.0)
    df["Bull"] = df["C_roll"] > df["O_roll"]
    df["Bear"] = df["O_roll"] > df["C_roll"]

    # Trend detection
    df["UpTrend"] = group[c].transform(lambda x: x > x.shift(tf_window - 1).fillna(x.iloc[0]) if len(x) >= tf_window else False)
    df["DownTrend"] = group[c].transform(lambda x: x < x.shift(tf_window - 1).fillna(x.iloc[0]) if len(x) >= tf_window else False)

    # Single-bar patterns
    df["Doji"] = df["Body"] <= df["doji_thresh"] * df["Range"]
    df["Hammer"] = (df["DownShadow"] >= df["shadow_ratio"] * df["Body"]) & \
                   (df["UpShadow"] <= df["hammer_thresh"] * df["Body"]) & \
                   (df["Body"] > 0) & \
                   (df["Body"] <= 2 * df["hammer_thresh"] * df["Range"]) & \
                   df["DownTrend"]
    df["InvertedHammer"] = (df["UpShadow"] >= df["shadow_ratio"] * df["Body"]) & \
                          (df["DownShadow"] <= df["hammer_thresh"] * df["Body"]) & \
                          (df["Body"] > 0) & \
                          (df["Body"] <= 2 * df["hammer_thresh"] * df["Range"]) & \
                          df["DownTrend"]
    df["BullishMarubozu"] = df["Bull"] & (df["Body"] >= df["long_body"] * df["Range"]) & \
                           (df["UpShadow"] <= df["marubozu_thresh"] * df["Range"]) & \
                           (df["DownShadow"] <= df["marubozu_thresh"] * df["Range"])
    df["BearishMarubozu"] = df["Bear"] & (df["Body"] >= df["long_body"] * df["Range"]) & \
                           (df["UpShadow"] <= df["marubozu_thresh"] * df["Range"]) & \
                           (df["DownShadow"] <= df["marubozu_thresh"] * df["Range"])
    df["HangingMan"] = df["Hammer"] & df["UpTrend"]
    df["ShootingStar"] = df["InvertedHammer"] & df["UpTrend"]
    df["SpinningTop"] = (df["Body"] <= df["small_body"] * df["Range"]) & \
                       (df["UpShadow"] >= df["Body"]) & \
                       (df["DownShadow"] >= df["Body"])

    # Multi-bar lags
    lag_cols = ["O_roll", "C_roll", "H_roll", "L_roll", "Bull", "Bear", "Body"]
    bool_lag_cols = ["Bull", "Bear"]
    for col_name in lag_cols:
        fill_val = False if col_name in bool_lag_cols else 0.0
        for lag in [1, 2, 3, 4]:
            df[f"{col_name}{lag}"] = group[col_name].shift(lag).fillna(fill_val)

    # Multi-bar patterns
    df["BullishEngulfing"] = (df["O_roll1"] > df["C_roll1"]) & df["Bull"] & \
                            (df["C_roll"] >= df["O_roll1"]) & (df["O_roll"] <= df["C_roll1"])
    df["BearishEngulfing"] = (df["C_roll1"] > df["O_roll1"]) & df["Bear"] & \
                            (df["O_roll"] >= df["C_roll1"]) & (df["C_roll"] <= df["O_roll1"])
    df["BullishHarami"] = (df["O_roll1"] > df["C_roll1"]) & df["Bull"] & \
                         (np.maximum(df["O_roll"], df["C_roll"]) <= np.maximum(df["O_roll1"], df["C_roll1"])) & \
                         (np.minimum(df["O_roll"], df["C_roll"]) >= np.minimum(df["O_roll1"], df["C_roll1"]))
    df["BearishHarami"] = (df["C_roll1"] > df["O_roll1"]) & df["Bear"] & \
                         (np.maximum(df["O_roll"], df["C_roll"]) <= np.maximum(df["O_roll1"], df["C_roll1"])) & \
                         (np.minimum(df["O_roll"], df["C_roll"]) >= np.minimum(df["O_roll1"], df["C_roll1"]))
    df["HaramiCross"] = df["Doji"] & \
                       (np.maximum(df["O_roll"], df["C_roll"]) <= np.maximum(df["O_roll1"], df["C_roll1"])) & \
                       (np.minimum(df["O_roll"], df["C_roll"]) >= np.minimum(df["O_roll1"], df["C_roll1"]))
    df["PiercingLine"] = (df["O_roll1"] > df["C_roll1"]) & df["Bull"] & \
                        (df["O_roll"] < df["C_roll1"]) & (df["C_roll"] > (df["O_roll1"] + df["C_roll1"]) / 2) & \
                        (df["C_roll"] < df["O_roll1"])
    df["DarkCloudCover"] = (df["C_roll1"] > df["O_roll1"]) & df["Bear"] & \
                          (df["O_roll"] > df["C_roll1"]) & (df["C_roll"] < (df["O_roll1"] + df["C_roll1"]) / 2) & \
                          (df["C_roll"] > df["O_roll1"])
    df["MorningStar"] = (df["O_roll2"] > df["C_roll2"]) & \
                       (np.abs(df["C_roll1"] - df["O_roll1"]) < np.abs(df["C_roll2"] - df["O_roll2"]) * df["small_body"]) & \
                       df["Bull"]
    df["EveningStar"] = (df["C_roll2"] > df["O_roll2"]) & \
                       (np.abs(df["C_roll1"] - df["O_roll1"]) < np.abs(df["C_roll2"] - df["O_roll2"]) * df["small_body"]) & \
                       df["Bear"]
    df["ThreeWhiteSoldiers"] = df["Bull"] & df["Bull1"] & df["Bull2"] & \
                              (df["C_roll"] > df["C_roll1"]) & (df["C_roll1"] > df["C_roll2"])
    df["ThreeBlackCrows"] = df["Bear"] & df["Bear1"] & df["Bear2"] & \
                           (df["C_roll"] < df["C_roll1"]) & (df["C_roll1"] < df["C_roll2"])
    df["TweezerTop"] = (df["H_roll"] == df["H_roll1"]) & df["Bear"] & df["Bull1"]
    df["TweezerBottom"] = (df["L_roll"] == df["L_roll1"]) & df["Bull"] & df["Bear1"]
    df["InsideBar"] = (df["H_roll"] < df["H_roll1"]) & (df["L_roll"] > df["L_roll1"])
    df["OutsideBar"] = (df["H_roll"] > df["H_roll1"]) & (df["L_roll"] < df["L_roll1"])

    # Near edge
    df["NearHigh"] = group["H_roll"].transform(lambda x: x >= x.rolling(tf_window, min_periods=1).max() * (1 - df["near_edge"].loc[x.index]))
    df["NearLow"] = group["L_roll"].transform(lambda x: x <= x.rolling(tf_window, min_periods=1).min() * (1 + df["near_edge"].loc[x.index]))

    df["DragonflyDoji"] = (np.abs(df["C_roll"] - df["O_roll"]) <= df["doji_thresh"] * df["Range"]) & \
                         (df["H_roll"] == df["C_roll"]) & (df["L_roll"] < df["O_roll"])
    df["GravestoneDoji"] = (np.abs(df["C_roll"] - df["O_roll"]) <= df["doji_thresh"] * df["Range"]) & \
                          (df["L_roll"] == df["C_roll"]) & (df["H_roll"] > df["O_roll"])
    df["LongLeggedDoji"] = (np.abs(df["C_roll"] - df["O_roll"]) <= df["doji_thresh"] * df["Range"]) & \
                          (df["UpShadow"] > df["shadow_ratio"] * df["Body"]) & \
                          (df["DownShadow"] > df["shadow_ratio"] * df["Body"])
    df["RisingThreeMethods"] = df["Bull4"] & df["Bear3"] & df["Bear2"] & df["Bear1"] & df["Bull"] & \
                              (df["Body3"].fillna(0.0) < df["small_body"] * df["Body4"].fillna(1.0)) & \
                              (df["Body2"].fillna(0.0) < df["small_body"] * df["Body4"].fillna(1.0)) & \
                              (df["Body1"].fillna(0.0) < df["small_body"] * df["Body4"].fillna(1.0)) & \
                              (df["H_roll3"].fillna(0.0) <= df["H_roll4"].fillna(0.0)) & \
                              (df["L_roll3"].fillna(0.0) >= df["L_roll4"].fillna(0.0)) & \
                              (df["H_roll2"].fillna(0.0) <= df["H_roll4"].fillna(0.0)) & \
                              (df["L_roll2"].fillna(0.0) >= df["L_roll4"].fillna(0.0)) & \
                              (df["H_roll1"].fillna(0.0) <= df["H_roll4"].fillna(0.0)) & \
                              (df["L_roll1"].fillna(0.0) >= df["L_roll4"].fillna(0.0)) & \
                              (df["C_roll"] > df["C_roll4"].fillna(0.0))
    df["FallingThreeMethods"] = df["Bear4"] & df["Bull3"] & df["Bull2"] & df["Bull1"] & df["Bear"] & \
                              (df["Body3"].fillna(0.0) < df["small_body"] * df["Body4"].fillna(1.0)) & \
                              (df["Body2"].fillna(0.0) < df["small_body"] * df["Body4"].fillna(1.0)) & \
                              (df["Body1"].fillna(0.0) < df["small_body"] * df["Body4"].fillna(1.0)) & \
                              (df["H_roll3"].fillna(0.0) <= df["H_roll4"].fillna(0.0)) & \
                              (df["L_roll3"].fillna(0.0) >= df["L_roll4"].fillna(0.0)) & \
                              (df["H_roll2"].fillna(0.0) <= df["H_roll4"].fillna(0.0)) & \
                              (df["L_roll2"].fillna(0.0) >= df["L_roll4"].fillna(0.0)) & \
                              (df["H_roll1"].fillna(0.0) <= df["H_roll4"].fillna(0.0)) & \
                              (df["L_roll1"].fillna(0.0) >= df["L_roll4"].fillna(0.0)) & \
                              (df["C_roll"] < df["C_roll4"].fillna(0.0))
    df["GapUp"] = df["O_roll"] > df["H_roll1"]
    df["GapDown"] = df["O_roll"] < df["L_roll1"]
    
    # ClimacticCandle
    df["RangeMean"] = group["Range"].rolling(tf_window, min_periods=1).mean().reset_index(0, drop=True)
    df["ClimacticCandle"] = df["Range"] > 2 * df["RangeMean"]

    # PatternCount
    pattern_cols = [
        "Doji", "Hammer", "InvertedHammer", "BullishMarubozu", "BearishMarubozu", 
        "HangingMan", "ShootingStar", "SpinningTop", "BullishEngulfing", "BearishEngulfing",
        "BullishHarami", "BearishHarami", "HaramiCross", "PiercingLine", "DarkCloudCover",
        "MorningStar", "EveningStar", "ThreeWhiteSoldiers", "ThreeBlackCrows", "TweezerTop",
        "TweezerBottom", "InsideBar", "OutsideBar", "NearHigh", "NearLow", "DragonflyDoji",
        "GravestoneDoji", "LongLeggedDoji", "RisingThreeMethods", "FallingThreeMethods",
        "GapUp", "GapDown", "ClimacticCandle"
    ]
    df["PatternCount"] = df[pattern_cols].sum(axis=1)

    # PatternType
    df[pattern_cols] = df[pattern_cols].astype(int)
    df["PatternType"] = df[pattern_cols].idxmax(axis=1, skipna=True).where(df["PatternCount"] > 0, "None")

    # Validate output
    expected_cols = pattern_cols + ["PatternCount", "PatternType"]
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Failed to create columns: {missing_cols}")

    return df

def add_confirmed_signals_optimized(df):
    """
    Optimized version: Collect new columns in dict and concat at end.
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
            "ValidSpinningTop": "UpTrend_MA",        
            "ValidClimacticCandle": "UpTrend_MA",    
        }
    }

    new_cols = {}
    for group_name, patterns in signal_groups.items():
        for valid_col, trend_col in patterns.items():
            raw_col = valid_col.replace("Valid", "")
            new_cols[valid_col] = df.get(raw_col, False) & df.get(trend_col, False)
 
    new_cols["ValidDragonflyDoji"]= df["DragonflyDoji"] & df["DownTrend_MA"] & df["HighVolume"]
    new_cols["ValidGravestoneDoji"]= df["GravestoneDoji"] & df["UpTrend_MA"] & df["HighVolume"]

    new_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, new_df], axis=1)

def add_trend_filters_optimized(df, timeframe="Daily", profile: str = "default"):
    """
    Optimized version: Use groupby.rolling and transform; avoid apply.
    """
    if profile is None:
        raise ValueError("profile ID cannot be None")
    if df.empty or not all(col in df.columns for col in ["CompanyId", "StockDate", "Close"]):
        raise ValueError("Input DataFrame must contain CompanyId, StockDate, and Close columns")

    c = "Close"

    # Load settings
    settings = load_settings(profile)["profiles"]
    if timeframe not in settings:
        raise ValueError(f"Invalid timeframe: {timeframe}. Must be one of {list(settings.keys())}")
    params = settings[timeframe]

    # Validate settings
    required_keys = ["ma", "ret", "vol", "roc_thresh", "slope_horizon"]
    if not all(key in params for key in required_keys):
        raise ValueError(f"Settings for timeframe {timeframe} must include {required_keys}")

    ma_window = params["ma"]
    ret_window = params["ret"]
    vol_window = params["vol"]
    roc_thresh = params["roc_thresh"]
    slope_horizon = params["slope_horizon"]

    # Validate window sizes
    if any(w <= 0 for w in [ma_window, ret_window, vol_window, slope_horizon]):
        raise ValueError("Window sizes (ma, ret, vol, slope_horizon) must be positive")

    df = df.sort_values(["CompanyId", "StockDate"])

    group = df.groupby("CompanyId")

    # Moving Average & slope
    df["MA"] = group[c].rolling(ma_window, min_periods=1).mean().reset_index(0, drop=True)
    df["MA_lag"] = df.groupby("CompanyId")["MA"].shift(slope_horizon)
    df["MA_slope"] = np.where(df["MA_lag"].notnull(), 
                             (df["MA"] - df["MA_lag"]) / df["MA_lag"], 0.0)
    df["UpTrend_MA"] = df["MA_slope"] > 0
    df["DownTrend_MA"] = df["MA_slope"] < 0
    df = df.drop(columns=["MA_lag"])

    # Returns & trend flags
    df["RecentReturn"] = np.where(
        group[c].shift(ret_window).notnull(),
        (df[c] - group[c].shift(ret_window)) / group[c].shift(ret_window),
        0.0
    )
    df["UpTrend_Return"] = df["RecentReturn"] > 0
    df["DownTrend_Return"] = df["RecentReturn"] < 0

    # Volatility (per-company median)
    df["ReturnPct"] = np.where(
        group[c].shift(1).notnull(),
        (df[c] - group[c].shift(1)) / group[c].shift(1),
        0.0
    )
    df["Volatility"] = df.groupby("CompanyId")["ReturnPct"].rolling(vol_window, min_periods=1).std().reset_index(0, drop=True)

    # Compute per-company median volatility
    df["Volatility_Median"] = df.groupby("CompanyId")["Volatility"].transform("median")
    df["LowVolatility"] = np.where(df["Volatility"].notnull(), df["Volatility"] < df["Volatility_Median"], False)
    df["HighVolatility"] = np.where(df["Volatility"].notnull(), df["Volatility"] > df["Volatility_Median"], False)

    # Rate of Change (ROC) & momentum
    df["ROC"] = np.where(
        group[c].shift(ma_window).notnull(),
        (df[c] - group[c].shift(ma_window)) / group[c].shift(ma_window),
        0.0
    )
    df["MomentumUp"] = df["ROC"] > roc_thresh
    df["MomentumDown"] = df["ROC"] < -roc_thresh

    # Confirmed trends (relaxed to 2/3 conditions)
    df["ConfirmedUpTrend"] = (df["UpTrend_MA"].astype(int) + 
                            df["UpTrend_Return"].astype(int) + 
                            df["MomentumUp"].astype(int)) >= 2
    df["ConfirmedDownTrend"] = (df["DownTrend_MA"].astype(int) + 
                              df["DownTrend_Return"].astype(int) + 
                              df["MomentumDown"].astype(int)) >= 2

    # Drop helper columns
    df = df.drop(columns=["ReturnPct", "Volatility_Median"])

    return df

def add_signal_strength_optimized(df, timeframe="Daily", profile: str = "default"):
    """
    Optimized version: Use pd.Series for sums to avoid if-else overhead; concat if needed.
    """
    if profile is None:
        raise ValueError("profile ID cannot be None")
    if df.empty or not all(col in df.columns for col in ["CompanyId", "StockDate"]):
        raise ValueError("Input DataFrame must contain CompanyId and StockDate columns")

    # Load settings
    settings = load_settings(profile)["signals"]
    if timeframe not in settings["timeframes"]:
        raise ValueError(f"Invalid timeframe: {timeframe}. Must be one of {list(settings['timeframes'].keys())}")
    tf_settings = settings["timeframes"][timeframe]

    # Get directional groups from settings or default
    directional_groups = tf_settings.get("directional_groups", ["Bullish", "Bearish", "Reversal", "Continuation"])

    # Identify valid signal columns
    valid_cols = [c for c in df.columns if c.startswith("Valid")]
    if not valid_cols:
        print("Warning: No Valid* columns found; returning default signal strength columns")
        df["SignalStrength"] = 0
        df["BullishPctRaw"] = 0.0
        df["BearishPctRaw"] = 0.0
        df["BullishPctDirectional"] = 0.0
        df["BearishPctDirectional"] = 0.0
        return df

    # Ensure trend columns exist
    for col in ["ConfirmedUpTrend", "ConfirmedDownTrend"]:
        if col not in df.columns:
            df[col] = False

    # SignalStrength with trend weighting
    signal_strength = df[valid_cols].sum(axis=1)
    df["SignalStrengthNonZero"] = np.where(signal_strength == 0, 1, signal_strength)
    df["SignalStrength"] = signal_strength * (
        1.0 + 
        np.where(df["ConfirmedUpTrend"], 0.2, 0.0) + 
        np.where(df["ConfirmedDownTrend"], 0.2, 0.0)
    )

    # Bullish/Bearish raw percentages
    bullish_cols = [c for c in valid_cols if any(k.lower() in c.lower() for k in ["Bullish", "Hammer", "MorningStar", "ThreeWhiteSoldiers", "TweezerBottom", "DragonflyDoji"])]
    bearish_cols = [c for c in valid_cols if any(k.lower() in c.lower() for k in ["Bearish", "ShootingStar", "EveningStar", "ThreeBlackCrows", "TweezerTop", "GravestoneDoji", "DarkCloudCover"])]

    bullish_sum = df[bullish_cols].sum(axis=1) if bullish_cols else pd.Series(0, index=df.index)
    bearish_sum = df[bearish_cols].sum(axis=1) if bearish_cols else pd.Series(0, index=df.index)

    df["BullishPctRaw"] = bullish_sum / df["SignalStrengthNonZero"]
    df["BearishPctRaw"] = bearish_sum / df["SignalStrengthNonZero"]

    # Directional percentages
    directional_cols = [c for c in valid_cols if any(c.startswith(f"Valid{g}") for g in directional_groups)]
    directional_sum = df[directional_cols].sum(axis=1) if directional_cols else pd.Series(0, index=df.index)
    df["DirectionalSumNonZero"] = np.where(directional_sum == 0, 1, directional_sum)

    bullish_dir_cols = [c for c in directional_cols if any(k.lower() in c.lower() for k in ["Bullish", "Hammer", "MorningStar", "ThreeWhiteSoldiers", "TweezerBottom", "DragonflyDoji"])]
    bearish_dir_cols = [c for c in directional_cols if any(k.lower() in c.lower() for k in ["Bearish", "ShootingStar", "EveningStar", "ThreeBlackCrows", "TweezerTop", "GravestoneDoji", "DarkCloudCover"])]

    bullish_dir_sum = df[bullish_dir_cols].sum(axis=1) if bullish_dir_cols else pd.Series(0, index=df.index)
    bearish_dir_sum = df[bearish_dir_cols].sum(axis=1) if bearish_dir_cols else pd.Series(0, index=df.index)

    df["BullishPctDirectional"] = bullish_dir_sum / df["DirectionalSumNonZero"]
    df["BearishPctDirectional"] = bearish_dir_sum / df["DirectionalSumNonZero"]

    # Drop helper columns
    df = df.drop(columns=["SignalStrengthNonZero", "DirectionalSumNonZero"])

    # Validate output
    expected_cols = ["SignalStrength", "BullishPctRaw", "BearishPctRaw", "BullishPctDirectional", "BearishPctDirectional"]
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Failed to create columns: {missing_cols}")

    return df

def finalize_signals_optimized(df, tf, tf_window=5, use_fundamentals=True, profile: str = "default"):
    """
    Fully Pandas-native signal consolidation with per-company aggregates.
    Uses rolling windows, true majority voting, and robust error handling.
    Optimized for efficiency with ~912,500 rows.
    
    Args:
        df: Pandas DataFrame with CompanyId, StockDate, Close, Valid*, FundamentalScore (optional)
        tf: Timeframe string (e.g., "Daily")
        tf_window: Integer window for pattern normalization
        use_fundamentals: Boolean to include FundamentalScore
        profile: String profile for settings
    
    Returns:
        Pandas DataFrame with Action, TomorrowAction, ActionConfidenceNorm, SignalStrengthHybrid, etc.
    """
    # Validate inputs
    if profile is None:
        raise ValueError("profile ID cannot be None")
    required_cols = ["CompanyId", "StockDate", "Close"]
    if df.empty or not all(col in df.columns for col in required_cols):
        raise ValueError(f"Input DataFrame must contain {required_cols}")
    if use_fundamentals and "FundamentalScore" not in df.columns:
        print("Warning: FundamentalScore missing; disabling fundamentals")
        use_fundamentals = False

    # Assume df is sorted; sort if necessary
    df = df.sort_values(['CompanyId', 'StockDate']).reset_index(drop=True)

    # Generate columns and momentum factor
    candle_columns, trend_cols, momentum_factor, momentum_factor_dict = generate_signal_columns_optimized(df, tf)
    bullish_patterns = trend_cols.get("Bullish", [])
    bearish_patterns = trend_cols.get("Bearish", [])
    if not (bullish_patterns or bearish_patterns or candle_columns.get("Buy") or candle_columns.get("Sell")):
        print("Warning: No valid patterns or candle columns from generate_signal_columns")

    # Tomorrow returns / momentum
    df['TomorrowClose'] = df.groupby('CompanyId')['Close'].shift(-1)
    df['TomorrowReturn'] = np.where(df['Close'] != 0, (df['TomorrowClose'] - df['Close']) / df['Close'], 0.0)
    prev_close = df.groupby('CompanyId')['Close'].shift(1)
    df['Return'] = np.where(prev_close.notna() & (prev_close != 0), (df['Close'] / prev_close - 1), 0.0)

    # Rolling stats (last 10 rows)
    df['AvgReturn'] = df.groupby('CompanyId')['Return'].rolling(10, min_periods=1).mean().reset_index(0, drop=True)
    df['Volatility'] = df.groupby('CompanyId')['Return'].rolling(10, min_periods=1).std().fillna(1e-8).reset_index(0, drop=True)
    df['MomentumZ'] = np.where(df['Volatility'] != 0, (df['Return'] - df['AvgReturn']) / df['Volatility'], 0.0)

    # Per-company momentum thresholds
    agg_df = df.groupby('CompanyId')['MomentumZ'].agg(mean_mom='mean', std_mom='std').reset_index()
    df = df.merge(agg_df, on='CompanyId', how='left')
    df['buy_thresh'] = df['mean_mom'] + momentum_factor * df['std_mom'].fillna(1.0)
    df['sell_thresh'] = df['mean_mom'] - momentum_factor * df['std_mom'].fillna(1.0)
    df['MomentumAction'] = np.select(
        [df['MomentumZ'] > df['buy_thresh'], df['MomentumZ'] < df['sell_thresh']],
        ['Buy', 'Sell'],
        default='Hold'
    )
    df = df.drop(['mean_mom', 'std_mom', 'buy_thresh', 'sell_thresh'], axis=1)

    # Pattern scores
    def pattern_sum(df, cols):
        return df[cols].sum(axis=1) if cols else pd.Series(0.0, index=df.index)

    df['BullScore'] = pattern_sum(df, bullish_patterns)
    df['BearScore'] = pattern_sum(df, bearish_patterns)
    df['PatternScore'] = df['BullScore'] - df['BearScore']
    df['PatternScoreNorm'] = df['PatternScore'] / tf_window
    df['PatternAction'] = np.select(
        [df['PatternScoreNorm'] > 0.3, df['PatternScoreNorm'] < -0.3],
        ['Buy', 'Sell'],
        default='Hold'
    )

    # Candle action
    buy_cols = candle_columns.get("Buy", [])
    sell_cols = candle_columns.get("Sell", [])
    buy_mask = pattern_sum(df, buy_cols) > 0 if buy_cols else pd.Series(False, index=df.index)
    sell_mask = pattern_sum(df, sell_cols) > 0 if sell_cols else pd.Series(False, index=df.index)
    df['CandleAction'] = np.select([buy_mask, sell_mask & ~buy_mask], ['Buy', 'Sell'], default='Hold')

    # CandidateAction (true majority vote)
    df['BuyCount'] = ((df['MomentumAction'] == 'Buy') + (df['PatternAction'] == 'Buy') + (df['CandleAction'] == 'Buy')).astype(int)
    df['SellCount'] = ((df['MomentumAction'] == 'Sell') + (df['PatternAction'] == 'Sell') + (df['CandleAction'] == 'Sell')).astype(int)
    df['CandidateAction'] = np.select(
        [df['BuyCount'] > df['SellCount'], df['SellCount'] > df['BuyCount']],
        ['Buy', 'Sell'],
        default='Hold'
    )
    df = df.drop(['BuyCount', 'SellCount'], axis=1)
    
    # Filter consecutive Buy/Sell
    df['PrevAction'] = df.groupby('CompanyId')['CandidateAction'].shift(1)
    df['Action'] = np.where(
        (df['CandidateAction'] == df['PrevAction']) & df['CandidateAction'].isin(['Buy', 'Sell']),
        'Hold',
        df['CandidateAction']
    )
    df = df.drop('PrevAction', axis=1)

    # TomorrowAction
    df['TomorrowAction'] = df.groupby('CompanyId')['Action'].shift(-1)
    df['TomorrowActionSource'] = np.where(
        df['TomorrowAction'].isin(['Buy', 'Sell']),
        'NextAction(filtered)',
        'Hold(no_signal)'
    )

    # Hybrid signal strength
    valid_cols = [c for c in df.columns if c.startswith("Valid")]
    if valid_cols:
        bull_keywords = ["Bull", "Hammer", "MorningStar", "ThreeWhiteSoldiers", "TweezerBottom", "DragonflyDoji"]
        bear_keywords = ["Bear", "ShootingStar", "EveningStar", "ThreeBlackCrows", "TweezerTop", "GravestoneDoji", "DarkCloudCover"]
        bull_cols = [c for c in valid_cols if any(k.lower() in c.lower() for k in bull_keywords)]
        bear_cols = [c for c in valid_cols if any(k.lower() in c.lower() for k in bear_keywords)]
        
        df['BullishCount'] = pattern_sum(df, bull_cols)
        df['BearishCount'] = pattern_sum(df, bear_cols)
        df['MagnitudeStrength'] = np.abs(df['PatternScore']) + np.abs(df['MomentumZ'])

        # First per-company normalization for bull/bear/mag
        agg_df = df.groupby('CompanyId').agg(
            max_bull=('BullishCount', 'max'),
            max_bear=('BearishCount', 'max'),
            max_mag=('MagnitudeStrength', 'max')
        ).reset_index()
        df = df.merge(agg_df, on='CompanyId', how='left')
        
        df['BullishStrengthHybrid'] = (
            df['BullishCount'] / df['max_bull'].replace(0, 1.0) +
            df['MagnitudeStrength'] / df['max_mag'].replace(0, 1.0)
        )
        df['BearishStrengthHybrid'] = (
            df['BearishCount'] / df['max_bear'].replace(0, 1.0) +
            df['MagnitudeStrength'] / df['max_mag'].replace(0, 1.0)
        )
        df['SignalStrengthHybrid'] = np.maximum(df['BullishStrengthHybrid'], df['BearishStrengthHybrid'])

        # ActionConfidence
        if use_fundamentals:
            df['ActionConfidence'] = np.where(
                df['FundamentalScore'].notna(),
                0.6 * df['SignalStrengthHybrid'] + 0.4 * df['FundamentalScore'],
                df['SignalStrengthHybrid']
            )
        else:
            df['ActionConfidence'] = df['SignalStrengthHybrid']

        # Second agg for max_conf
        agg_df_conf = df.groupby('CompanyId')['ActionConfidence'].max().reset_index(name='max_conf')
        df = df.merge(agg_df_conf, on='CompanyId', how='left')
        df['ActionConfidenceNorm'] = df['ActionConfidence'] / df['max_conf'].replace(0, 1.0)

        df = df.drop(['max_bull', 'max_bear', 'max_mag', 'max_conf'], axis=1)

        # Signal duration (bounded window)
        max_lag = max(10, tf_window)
        df['change'] = (df['Action'] != df.groupby('CompanyId')['Action'].shift(1)).astype(int)
        df['SignalDuration'] = df.groupby('CompanyId')['change'].rolling(2 * max_lag + 1, min_periods=1, center=True).sum().reset_index(0, drop=True)
        df = df.drop('change', axis=1)

        df['ValidAction'] = df['Action'].isin(['Buy', 'Sell'])
        df['HasValidSignal'] = df['Action'].notna() & df['TomorrowAction'].notna() & df['SignalStrengthHybrid'].notna()
    else:
        df['SignalStrengthHybrid'] = 0.0
        df['ActionConfidence'] = 0.0
        df['ActionConfidenceNorm'] = 0.0
        df['SignalDuration'] = 0
        df['ValidAction'] = False
        df['HasValidSignal'] = False

    # Validate output
    expected_cols = ["Action", "TomorrowAction", "ActionConfidenceNorm", "SignalStrengthHybrid", "SignalDuration", "ValidAction", "HasValidSignal"]
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Failed to create columns: {missing_cols}")

    return df

def compute_fundamental_score_optimized(df, profile: str = "default"):
    """
    Compute normalized fundamental score using per-company aggregates.
    Fully vectorized, minimizes shuffles, handles nulls/outliers.
    Optimized for efficiency with ~912,500 rows.
    
    Args:
        df: Pandas DataFrame with CompanyId, PeRatio, PbRatio, ..., ShortIntToFloat
        profile: String profile ID for settings
    
    Returns:
        Pandas DataFrame with FundamentalScore, FundamentalBad, and normalized columns
    """
    # Validate inputs
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

    # Load profile settings
    user_settings = load_settings(profile).get("fundamental_weights", {})
    weights = {
        "valuation": user_settings.get("valuation", 0.2),
        "profitability": user_settings.get("profitability", 0.3),
        "DebtLiquidity": user_settings.get("DebtLiquidity", 0.2),
        "Growth": user_settings.get("Growth", 0.2),
        "Sentiment": user_settings.get("Sentiment", 0.1),
    }
    if abs(sum(weights.values()) - 1.0) > 1e-6:
        print("Warning: Weights do not sum to 1.0; normalizing")
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}

    # Per-company min/max aggregates
    agg_dict = {}
    group = df.groupby('CompanyId')
    for col in required_cols[1:]:  # Skip CompanyId
        agg_dict[f"min_{col.lower()}"] = group[col].transform('min').fillna(0)
        agg_dict[f"max_{col.lower()}"] = group[col].transform('max').fillna(0)

    agg_df = pd.DataFrame(agg_dict, index=df.index)
    df = pd.concat([df, agg_df], axis=1)

    def normalize(df, col_name, min_col, max_col, alias, invert=False):
        """Add a normalized version of a column."""
        normalized = (df[col_name] - df[min_col]) / (df[max_col] - df[min_col])
        normalized = np.where(df[max_col] == df[min_col], 0.0, normalized.fillna(0.0))
        if invert:
            normalized = 1.0 - normalized
        df[alias] = normalized
        return df

    # Normalize with outlier capping
    df = normalize(df, "PeRatio", "min_peratio", "max_peratio", "pe_norm", invert=True)
    df = normalize(df, "PbRatio", "min_pbratio", "max_pbratio", "pb_norm", invert=True)
    df = normalize(df, "PegRatio", "min_pegratio", "max_pegratio", "peg_norm", invert=True)
    df = normalize(df, "ReturnOnEquity", "min_returnonequity", "max_returnonequity", "roe_norm")
    df = normalize(df, "GrossMarginTTM", "min_grossmarginttm", "max_grossmarginttm", "gross_margin_norm")
    df = normalize(df, "NetProfitMarginTTM", "min_netprofitmarginttm", "max_netprofitmarginttm", "net_margin_norm")
    df = normalize(df, "TotalDebtToEquity", "min_totaldebttoequity", "max_totaldebttoequity", "de_norm", invert=True)
    df = normalize(df, "CurrentRatio", "min_currentratio", "max_currentratio", "current_ratio_norm")
    df = normalize(df, "InterestCoverage", "min_interestcoverage", "max_interestcoverage", "int_cov_norm")
    df = normalize(df, "EpsChangeYear", "min_epschangeyear", "max_epschangeyear", "eps_change_norm")
    df = normalize(df, "RevChangeYear", "min_revchangeyear", "max_revchangeyear", "rev_change_norm")
    df = normalize(df, "Beta", "min_beta", "max_beta", "beta_norm", invert=True)
    df = normalize(df, "ShortIntToFloat", "min_shortinttofloat", "max_shortinttofloat", "short_int_norm")  # Not inverted

    # Drop intermediate columns
    df = df.drop(columns=[c for c in df.columns if c.startswith("min_") or c.startswith("max_")])

    # Combine weighted score
    df['FundamentalScore'] = (
        weights["valuation"] * (df["pe_norm"] + df["peg_norm"] + df["pb_norm"]) / 3 +
        weights["profitability"] * (df["roe_norm"] + df["gross_margin_norm"] + df["net_margin_norm"]) / 3 +
        weights["DebtLiquidity"] * (df["de_norm"] + df["current_ratio_norm"] + df["int_cov_norm"]) / 3 +
        weights["Growth"] * (df["eps_change_norm"] + df["rev_change_norm"]) / 2 +
        weights["Sentiment"] * (df["beta_norm"] + df["short_int_norm"]) / 2
    ).fillna(0.0)

    # Flag bad rows (nulls only, not zeros)
    norm_cols = [
        "pe_norm", "peg_norm", "pb_norm",
        "roe_norm", "gross_margin_norm", "net_margin_norm",
        "de_norm", "current_ratio_norm", "int_cov_norm",
        "eps_change_norm", "rev_change_norm",
        "beta_norm", "short_int_norm"
    ]
    df['FundamentalBad'] = df[norm_cols].isna().any(axis=1)

    # Validate output
    expected_cols = ["FundamentalScore", "FundamentalBad"] + norm_cols
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Failed to create columns: {missing_cols}")

    return df

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