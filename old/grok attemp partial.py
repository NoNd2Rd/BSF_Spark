import pandas as pd
import numpy as np
from datetime import datetime
import re
from bsf_settings import load_settings  # Assume this is defined elsewhere
from functools import reduce

def generate_signal_columns_optimized(df, timeframe="Short", user: int = 1):
    """
    Optimized version: Use aggregate-then-merge for last_close and momentum.
    """
    if user is None:
        raise ValueError("User ID cannot be None")
    if df.empty or not all(col in df.columns for col in ["CompanyId", "StockDate", "Close"]):
        raise ValueError("Input DataFrame must contain CompanyId, StockDate, and Close columns")

    settings = load_settings(str(user))["signals"]
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
        print(f"Warning: No valid candle columns found for timeframe {timeframe} and user {user}")
    if not trend_columns["Bullish"] and not trend_columns["Bearish"]:
        print(f"Warning: No valid trend columns found for timeframe {timeframe} and user {user}")

    return candle_columns, trend_columns, momentum_factor_base, momentum_dict 

def get_candle_params_optimized(df: pd.DataFrame, user: int = 1, close_col: str = "Close") -> pd.DataFrame:
    """
    Optimized version: Collect all new columns in a dict and concat once at the end.
    """
    required_cols = ["CompanyId", close_col, "StockDate"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Input DataFrame missing columns: {missing_cols}")
    if user is None:
        raise ValueError("User ID cannot be None")

    user_settings = load_settings(user).get("candle_params", {
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
    new_cols["near_edge"] = user_settings["near_edge"]
    new_cols["highvol_spike"] = user_settings["highvol_spike"]
    new_cols["lowvol_dip"] = user_settings["lowvol_dip"]
    new_cols["hammer_thresh"] = threshold(user_settings["hammer_base"], user_settings["hammer_scale"],
                                      user_settings["hammer_min"], user_settings["hammer_max"])
    new_cols["marubozu_thresh"] = threshold(user_settings["marubozu_base"], user_settings["marubozu_scale"],
                                      user_settings["marubozu_min"], user_settings["marubozu_max"])
    new_cols["rng_thresh"] = threshold(user_settings["rng_base"], user_settings["rng_scale"],
                                      user_settings["rng_min"], user_settings["rng_max"])

    # Concat once
    new_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, new_df], axis=1)

def add_candle_patterns_optimized(df, tf_window=5, user: int = 1, 
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
    if user is None:
        raise ValueError("User ID cannot be None")

    df = df.sort_values(["CompanyId", "StockDate"])

    o, h, l, c, v = open_col, high_col, low_col, close_col, volume_col

    df = get_candle_params_optimized(df, user=user, close_col=close_col)

    group = df.groupby("CompanyId")

    # Rolling calculations using groupby.rolling
    df["O_roll"] = group[o].transform(lambda x: x.shift(tf_window - 1).fillna(x.iloc[0]))
    df["C_roll"] = df[c].copy()  # Current close
    df["H_roll"] = group[h].rolling(tf_window, min_periods=1).max().reset_index(level=0, drop=True)
    df["L_roll"] = group[l].rolling(tf_window, min_periods=1).min().reset_index(level=0, drop=True)
    df["V_avg20"] = group[v].rolling(20, min_periods=1).mean().reset_index(level=0, drop=True)

    # Volume spikes
    df["HighVolume"] = df[v] > df["highvol_spike"] * df["V_avg20"]
    df["LowVolume"] = df[v] < df["lowvol_dip"] * df["V_avg20"]

    # Body, shadows, range (vectorized)
    df["Body"] = np.where(df["C_roll"] != 0, np.abs(df["C_roll"] - df["O_roll"]) / df["C_roll"], 0.0)
    df["UpShadow"] = np.where(df["C_roll"] != 0, 
                             (df["H_roll"] - np.maximum(df["O_roll"], df["C_roll"])) / df["C_roll"], 0.0)
    df["DownShadow"] = np.where(df["C_roll"] != 0, 
                               (np.minimum(df["O_roll"], df["C_roll"]) - df["L_roll"]) / df["C_roll"], 0.0)
    df["Range"] = np.where(df["C_roll"] != 0, (df["H_roll"] - df["L_roll"]) / df["C_roll"], 0.0)
    df["Bull"] = df["C_roll"] > df["O_roll"]
    df["Bear"] = df["O_roll"] > df["C_roll"]

    # Trend detection using transform
    df["UpTrend"] = group[c].transform(lambda x: (x > x.shift(tf_window - 1)).fillna(False))
    df["DownTrend"] = group[c].transform(lambda x: (x < x.shift(tf_window - 1)).fillna(False))

    # Single-bar patterns (vectorized)
    df["Doji"] = df["Body"] <= df["doji_thresh"] * df["Range"]
    df["Hammer"] = (df["DownShadow"] >= df["shadow_ratio"] * df["Body"]) & \
                   (df["UpShadow"] <= df["hammer_thresh"] * df["Body"]) & \
                   (df["Body"] > 0) & \
                   (df["Body"] <= 2 * df["hammer_thresh"] * df["Range"]) & \
                   df["DownTrend"]
    # ... (similar for other single-bar patterns, no change as they are vectorized)

    # Multi-bar lags using shift on groups
    lag_cols = ["O_roll", "C_roll", "H_roll", "L_roll", "Bull", "Bear", "Body"]
    for col_name in lag_cols:
        for lag in [1, 2, 3, 4]:
            df[f"{col_name}{lag}"] = group[col_name].shift(lag).fillna(0.0)  # Fill to avoid NaN propagation

    # Multi-bar patterns (vectorized, no change)

    # Near edge using transform with loc for alignment
    df["NearHigh"] = group["H_roll"].transform(lambda x: x >= x.rolling(tf_window, min_periods=1).max() * (1 - df["near_edge"].loc[x.index]))
    df["NearLow"] = group["L_roll"].transform(lambda x: x <= x.rolling(tf_window, min_periods=1).min() * (1 + df["near_edge"].loc[x.index]))

    # ... (similar for other patterns)

    # PatternCount (vectorized sum)
    pattern_cols = [...]  # Your list
    df["PatternCount"] = df[pattern_cols].sum(axis=1)

    # PatternType using idxmax (fast vectorized alternative to apply)
    df[pattern_cols] = df[pattern_cols].astype(int)  # True->1, False->0
    df["PatternType"] = df[pattern_cols].idxmax(axis=1, skipna=True).where(df["PatternCount"] > 0, "None")

    return df

def add_confirmed_signals_optimized(df):
    """
    Optimized version: Collect new columns in dict and concat at end.
    """
    signal_groups = { ... }  # Your dict

    new_cols = {}
    for group_name, patterns in signal_groups.items():
        for valid_col, trend_col in patterns.items():
            raw_col = valid_col.replace("Valid", "")
            new_cols[valid_col] = df.get(raw_col, False) & df.get(trend_col, False)
 
    new_cols["ValidDragonflyDoji"] = df["DragonflyDoji"] & df["DownTrend_MA"] & df["HighVolume"]
    new_cols["ValidGravestoneDoji"] = df["GravestoneDoji"] & df["UpTrend_MA"] & df["HighVolume"]

    new_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, new_df], axis=1)

def add_trend_filters_optimized(df, timeframe="Daily", user: int = 1):
    """
    Optimized version: Use groupby.rolling and transform; avoid apply.
    """
    # ... (your code, already mostly optimized with rolling and transform; no major changes needed besides ensuring fills)

def add_signal_strength_optimized(df, timeframe="Daily", user: int = 1):
    """
    Optimized version: Use pd.Series for sums to avoid if-else overhead; concat if needed.
    """
    # ... (your code, already good; use pd.Series(0, index=df.index) for empty sums)

def finalize_signals_optimized(df, tf, tf_window=5, use_fundamentals=True, user: int = 1):
    """
    Your existing optimized version is good (uses merge for aggs). No major changes.
    """

def compute_fundamental_score_optimized(df, user: int = 1):
    """
    Your existing version is good (uses transform for min/max). No major changes.
    """

def add_batch_metadata_optimized(df, timeframe, user: int = 1, ingest_ts=None):
    """
    Optimized version: Assign multiple columns at once.
    """
    if ingest_ts is None:
        ingest_ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    df = df.assign(
        BatchId=f"{user}_{ingest_ts}",
        IngestedAt=ingest_ts,
        TimeFrame=timeframe,
        UserId=user
    )
    return df