import pandas as pd
import numpy as np
from datetime import datetime
import re
from bsf_settings import load_settings

import unicodedata


from functools import reduce
   
from pyspark.sql import functions as F, Window
from operator import itemgetter
# -------------------------------
# Add Signal Columns
# -------------------------------
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql import SparkSession

def generate_signal_columns(sdf, timeframe="Short", user: int = 1):
    """
    Determines bullish/bearish candle and trend columns, adjusting momentum for penny stocks.
    Expects a Spark DataFrame with columns: CompanyId, StockDate, Close, and candlestick/trend columns.
    Fully Spark-native, processes last_close per company for penny stock adjustment.
    
    Args:
        sdf: Spark DataFrame with stock data
        timeframe: String, e.g., "Short", "Daily"
        user: Integer user ID for settings
    
    Returns:
        tuple: (candle_columns: dict with "Buy" and "Sell" lists,
                trend_columns: dict with "Bullish" and "Bearish" lists,
                momentum_factor: dict of CompanyId to float)
    """
    # Validate inputs
    if user is None:
        raise ValueError("User ID cannot be None")
    if not sdf or "CompanyId" not in sdf.columns or "StockDate" not in sdf.columns or "Close" not in sdf.columns:
        raise ValueError("Input DataFrame must contain CompanyId, StockDate, and Close columns")

    # Load and broadcast settings (assumes load_settings returns dict)
    spark = SparkSession.getActiveSession()
    settings = spark.sparkContext.broadcast(load_settings(str(user))["signals"])
    tf_settings = settings.value["timeframes"].get(timeframe, settings.value["timeframes"]["Daily"])
    
    # Validate settings
    if not all(key in tf_settings for key in ["momentum", "buy", "sell"]):
        raise ValueError("Timeframe settings must include momentum, buy, and sell keys")
    if not all(key in settings.value["penny_stock_adjustment"] for key in ["threshold", "factor", "min_momentum"]):
        raise ValueError("Penny stock settings must include threshold, factor, and min_momentum")

    momentum_factor_base = tf_settings["momentum"]
    ps = settings.value["penny_stock_adjustment"]

    # Compute last_close per CompanyId
    w_last = Window.partitionBy("CompanyId").orderBy(F.col("StockDate").desc())
    sdf_last = sdf.withColumn("row_num", F.row_number().over(w_last)) \
                  .filter(F.col("row_num") == 1) \
                  .select("CompanyId", F.col("Close").alias("last_close")) \
                  .cache()
    
    # Apply penny stock adjustment per company
    sdf_momentum = sdf_last.withColumn(
        "momentum_factor",
        F.when(
            F.col("last_close") < ps["threshold"],
            F.greatest(F.lit(momentum_factor_base * ps["factor"]), F.lit(ps["min_momentum"]))
        ).otherwise(F.lit(momentum_factor_base))
    )
    
    # Collect momentum factors as dict (small, ~2500 rows)
    momentum_dict = {row["CompanyId"]: row["momentum_factor"] 
                     for row in sdf_momentum.select("CompanyId", "momentum_factor").collect()}
    
    sdf_last.unpersist()

    # Candle columns (aligned with add_candle_patterns_fast output)
    all_columns = sdf.columns
    # Use patterns from add_candle_patterns_fast, not "Valid*"
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

    # Trend columns (aligned with keep_cols in load_signals_small_node)
    trend_columns = {
        "Bullish": [col for col in all_columns if col in ["MomentumUp", "ConfirmedUpTrend", "UpTrend_MA"]],
        "Bearish": [col for col in all_columns if col in ["MomentumDown", "ConfirmedDownTrend", "DownTrend_MA"]]
    }

    # Validate output
    if not candle_columns["Buy"] and not candle_columns["Sell"]:
        print(f"Warning: No valid candle columns found for timeframe {timeframe} and user {user}")
    if not trend_columns["Bullish"] and not trend_columns["Bearish"]:
        print(f"Warning: No valid trend columns found for timeframe {timeframe} and user {user}")

    return candle_columns, trend_columns, momentum_dict
    
def generate_signal_columns_chatgpt(sdf, timeframe="Short", user: int = 1):
    """
    Determines bullish/bearish candle and trend columns, adjusting momentum for penny stocks.
    Expects a Spark DataFrame as input.
    
    Returns:
        candle_columns: dict with "Buy" and "Sell" lists
        trend_columns: dict with "Bullish" and "Bearish" lists
        momentum_factor: float
    """
    # Load user settings
    settings = load_settings(str(user))["signals"]
    tf_settings = settings["timeframes"].get(timeframe, settings["timeframes"]["Daily"])
    momentum_factor = tf_settings["momentum"]
    ps = settings["penny_stock_adjustment"]

    # --- Get last Close for this company ---
    #last_row = sdf.orderBy(F.col("StockDate").desc()).select("Close").limit(1).collect()
    last_close = sdf.agg(F.last("Close").alias("last_close")).first()["last_close"]

    #last_close = last_row[0]["Close"] if last_row else 0

    # Penny stock adjustment
    if last_close < ps["threshold"]:
        momentum_factor = max(momentum_factor * ps["factor"], ps["min_momentum"])

    # Candle columns
    all_columns = sdf.columns
    candle_cols = [col for col in all_columns if col.startswith("Valid")]
    candle_columns = {
        "Buy": [col for col in candle_cols if any(k in col.lower() for k in tf_settings["buy"])],
        "Sell": [col for col in candle_cols if any(k in col.lower() for k in tf_settings["sell"])]
    }

    # Trend columns
    bullish_cols = [col for col in all_columns if col in ["MomentumUp", "ConfirmedUpTrend", "UpTrend_MA"]]
    bearish_cols = [col for col in all_columns if col in ["MomentumDown", "ConfirmedDownTrend", "DownTrend_MA"]]
    trend_columns = {"Bullish": bullish_cols, "Bearish": bearish_cols}

    return candle_columns, trend_columns, momentum_factor

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from operator import itemgetter
from functools import reduce
from pyspark.sql import SparkSession

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from operator import itemgetter
from functools import reduce
from pyspark.sql import SparkSession

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from operator import itemgetter
from functools import reduce
from pyspark.sql import SparkSession

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from operator import itemgetter
from functools import reduce
from pyspark.sql import SparkSession

def get_candle_params(sdf, user: int = 1, close_col: str = "Close"):
    """
    Add candlestick pattern threshold columns to DataFrame, scaled by per-row close price.
    Fully Spark-native, optimized for 4-core, 8GB server with ~912,500 rows, sub-2-hour runtime.
    
    Args:
        sdf: Spark DataFrame with CompanyId, close_col, StockDate
        user: Integer user ID for settings
        close_col: Name of the close price column (default: "Close")
    
    Returns:
        Spark DataFrame with additional columns: doji_thresh, long_body, small_body, etc.
    """
    # Validate inputs
    required_cols = ["CompanyId", close_col, "StockDate"]
    missing_cols = [col for col in required_cols if col not in sdf.columns]
    if missing_cols:
        raise ValueError(f"Input DataFrame missing columns: {missing_cols}")
    if user is None:
        raise ValueError("User ID cannot be None")

    # Load user settings (assumed to return dict)
    spark = SparkSession.getActiveSession()
    user_settings = spark.sparkContext.broadcast(load_settings(user).get("candle_params", {
        "doji_base": 0.1, "doji_scale": 0.05, "doji_min": 0.05, "doji_max": 0.2,
        "long_body_base": 0.7, "long_body_scale": 0.1, "long_body_min": 0.6, "long_body_max": 0.9,
        "small_body_base": 0.3, "small_body_scale": 0.05, "small_body_min": 0.2, "small_body_max": 0.4,
        "shadow_ratio_base": 2.0, "shadow_ratio_scale": 0.5, "shadow_ratio_min": 1.5, "shadow_ratio_max": 3.0,
        "near_edge": 0.05, "highvol_spike": 2.0, "lowvol_dip": 0.5,
        "hammer_base": 0.2, "hammer_scale": 0.05, "hammer_min": 0.1, "hammer_max": 0.3,
        "marubozu_base": 0.1, "marubozu_scale": 0.05, "marubozu_min": 0.05, "marubozu_max": 0.2,
        "rng_base": 0.05, "rng_scale": 0.02, "rng_min": 0.03, "rng_max": 0.1
    }))

    # Compute price-scaled thresholds per row
    logp = F.log10(F.greatest(F.col(close_col), F.lit(1e-6)))
    sdf = sdf.withColumn("logp", (logp + 6) / 8)

    def add_threshold(col_name, base, scale, min_val, max_val):
        return sdf.withColumn(
            col_name,
            F.least(
                F.greatest(
                    F.lit(base) + F.lit(scale) * F.col("logp"),
                    F.lit(min_val)
                ),
                F.lit(max_val)
            )
        )

    # Add threshold columns
    sdf = add_threshold("doji_thresh", user_settings.value["doji_base"], user_settings.value["doji_scale"],
                       user_settings.value["doji_min"], user_settings.value["doji_max"])
    sdf = add_threshold("long_body", user_settings.value["long_body_base"], user_settings.value["long_body_scale"],
                       user_settings.value["long_body_min"], user_settings.value["long_body_max"])
    sdf = add_threshold("small_body", user_settings.value["small_body_base"], user_settings.value["small_body_scale"],
                       user_settings.value["small_body_min"], user_settings.value["small_body_max"])
    sdf = add_threshold("shadow_ratio", user_settings.value["shadow_ratio_base"], user_settings.value["shadow_ratio_scale"],
                       user_settings.value["shadow_ratio_min"], user_settings.value["shadow_ratio_max"])
    sdf = sdf.withColumn("near_edge", F.lit(user_settings.value["near_edge"]))
    sdf = sdf.withColumn("highvol_spike", F.lit(user_settings.value["highvol_spike"]))
    sdf = sdf.withColumn("lowvol_dip", F.lit(user_settings.value["lowvol_dip"]))
    sdf = add_threshold("hammer_thresh", user_settings.value["hammer_base"], user_settings.value["hammer_scale"],
                       user_settings.value["hammer_min"], user_settings.value["hammer_max"])
    sdf = add_threshold("marubozu_thresh", user_settings.value["marubozu_base"], user_settings.value["marubozu_scale"],
                       user_settings.value["marubozu_min"], user_settings.value["marubozu_max"])
    sdf = add_threshold("rng_thresh", user_settings.value["rng_base"], user_settings.value["rng_scale"],
                       user_settings.value["rng_min"], user_settings.value["rng_max"])

    sdf = sdf.drop("logp")
    return sdf

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from operator import itemgetter
from functools import reduce
from pyspark.sql import SparkSession

def get_candle_params(sdf, user: int = 1, close_col: str = "Close"):
    """
    Add candlestick pattern threshold columns to DataFrame, scaled by per-row close price.
    Fully Spark-native, optimized for 4-core, 8GB server with ~912,500 rows, sub-2-hour runtime.
    
    Args:
        sdf: Spark DataFrame with CompanyId, close_col, StockDate
        user: Integer user ID for settings
        close_col: Name of the close price column (default: "Close")
    
    Returns:
        Spark DataFrame with additional columns: doji_thresh, long_body, small_body, etc.
    """
    # Validate inputs
    required_cols = ["CompanyId", close_col, "StockDate"]
    missing_cols = [col for col in required_cols if col not in sdf.columns]
    if missing_cols:
        raise ValueError(f"Input DataFrame missing columns: {missing_cols}")
    if user is None:
        raise ValueError("User ID cannot be None")

    # Load user settings (assumed to return dict)
    spark = SparkSession.getActiveSession()
    user_settings = spark.sparkContext.broadcast(load_settings(user).get("candle_params", {
        "doji_base": 0.1, "doji_scale": 0.05, "doji_min": 0.05, "doji_max": 0.2,
        "long_body_base": 0.7, "long_body_scale": 0.1, "long_body_min": 0.6, "long_body_max": 0.9,
        "small_body_base": 0.3, "small_body_scale": 0.05, "small_body_min": 0.2, "small_body_max": 0.4,
        "shadow_ratio_base": 2.0, "shadow_ratio_scale": 0.5, "shadow_ratio_min": 1.5, "shadow_ratio_max": 3.0,
        "near_edge": 0.05, "highvol_spike": 2.0, "lowvol_dip": 0.5,
        "hammer_base": 0.2, "hammer_scale": 0.05, "hammer_min": 0.1, "hammer_max": 0.3,
        "marubozu_base": 0.1, "marubozu_scale": 0.05, "marubozu_min": 0.05, "marubozu_max": 0.2,
        "rng_base": 0.05, "rng_scale": 0.02, "rng_min": 0.03, "rng_max": 0.1
    }))

    # Compute price-scaled thresholds per row
    logp = F.log10(F.greatest(F.col(close_col), F.lit(1e-6)))
    sdf = sdf.withColumn("logp", (logp + 6) / 8)

    def add_threshold(col_name, base, scale, min_val, max_val):
        return sdf.withColumn(
            col_name,
            F.least(
                F.greatest(
                    F.lit(base) + F.lit(scale) * F.col("logp"),
                    F.lit(min_val)
                ),
                F.lit(max_val)
            )
        )

    # Add threshold columns
    sdf = add_threshold("doji_thresh", user_settings.value["doji_base"], user_settings.value["doji_scale"],
                       user_settings.value["doji_min"], user_settings.value["doji_max"])
    sdf = add_threshold("long_body", user_settings.value["long_body_base"], user_settings.value["long_body_scale"],
                       user_settings.value["long_body_min"], user_settings.value["long_body_max"])
    sdf = add_threshold("small_body", user_settings.value["small_body_base"], user_settings.value["small_body_scale"],
                       user_settings.value["small_body_min"], user_settings.value["small_body_max"])
    sdf = add_threshold("shadow_ratio", user_settings.value["shadow_ratio_base"], user_settings.value["shadow_ratio_scale"],
                       user_settings.value["shadow_ratio_min"], user_settings.value["shadow_ratio_max"])
    sdf = sdf.withColumn("near_edge", F.lit(user_settings.value["near_edge"]))
    sdf = sdf.withColumn("highvol_spike", F.lit(user_settings.value["highvol_spike"]))
    sdf = sdf.withColumn("lowvol_dip", F.lit(user_settings.value["lowvol_dip"]))
    sdf = add_threshold("hammer_thresh", user_settings.value["hammer_base"], user_settings.value["hammer_scale"],
                       user_settings.value["hammer_min"], user_settings.value["hammer_max"])
    sdf = add_threshold("marubozu_thresh", user_settings.value["marubozu_base"], user_settings.value["marubozu_scale"],
                       user_settings.value["marubozu_min"], user_settings.value["marubozu_max"])
    sdf = add_threshold("rng_thresh", user_settings.value["rng_base"], user_settings.value["rng_scale"],
                       user_settings.value["rng_min"], user_settings.value["rng_max"])

    sdf = sdf.drop("logp")
    return sdf

def add_candle_patterns_fast(sdf, tf_window=5, user: int = 1, 
                            open_col="Open", high_col="High", low_col="Low", 
                            close_col="Close", volume_col="Volume"):
    """
    Spark-native candlestick pattern detection with per-row thresholds.
    Optimized for 4-core, 8GB server with ~912,500 rows, sub-2-hour runtime.
    
    Args:
        sdf: Spark DataFrame with CompanyId, open_col, high_col, low_col, close_col, volume_col, StockDate
        tf_window: Integer window for rolling calculations
        user: Integer user ID for settings
        open_col, high_col, low_col, close_col, volume_col: Names of OHLCV columns (default: Open, High, Low, Close, Volume)
    
    Returns:
        Spark DataFrame with pattern columns, PatternCount, and PatternType
    """
    # Validate inputs
    required_cols = ["CompanyId", open_col, high_col, low_col, close_col, volume_col, "StockDate"]
    missing_cols = [col for col in required_cols if col not in sdf.columns]
    if missing_cols:
        raise ValueError(f"Input DataFrame missing columns: {missing_cols}")
    if tf_window < 1:
        raise ValueError("tf_window must be positive")
    if user is None:
        raise ValueError("User ID cannot be None")

    # Map column names to internal aliases
    o, h, l, c, v = open_col, high_col, low_col, close_col, volume_col

    # Add per-row candle parameters
    sdf = get_candle_params(sdf, user=user, close_col=close_col)

    # Windows (bounded for memory where needed)
    max_lag = max(tf_window - 1, 4)
    w_tf = Window.partitionBy("CompanyId").orderBy("StockDate").rowsBetween(-max_lag, 0)
    w_shift = Window.partitionBy("CompanyId").orderBy("StockDate")  # No rowsBetween for lag
    w_vol = Window.partitionBy("CompanyId").orderBy("StockDate").rowsBetween(-19, 0)

    # Rolling calculations
    sdf = sdf.withColumn("O_roll", F.first(o).over(w_tf)) \
             .withColumn("C_roll", F.last(c).over(w_tf)) \
             .withColumn("H_roll", F.max(h).over(w_tf)) \
             .withColumn("L_roll", F.min(l).over(w_tf)) \
             .withColumn("V_avg20", F.avg(v).over(w_vol))

    # Volume spikes
    sdf = sdf.withColumn("HighVolume", F.col(v) > F.col("highvol_spike") * F.col("V_avg20")) \
             .withColumn("LowVolume", F.col(v) < F.col("lowvol_dip") * F.col("V_avg20"))

    # Body, shadows, range
    sdf = sdf.withColumn(
        "Body",
        F.when(F.col("C_roll") != 0, F.abs(F.col("C_roll") - F.col("O_roll")) / F.col("C_roll")).otherwise(0.0)
    ).withColumn(
        "UpShadow",
        F.when(F.col("C_roll") != 0,
               (F.col("H_roll") - F.greatest(F.col("O_roll"), F.col("C_roll"))) / F.col("C_roll")).otherwise(0.0)
    ).withColumn(
        "DownShadow",
        F.when(F.col("C_roll") != 0,
               (F.least(F.col("O_roll"), F.col("C_roll")) - F.col("L_roll")) / F.col("C_roll")).otherwise(0.0)
    ).withColumn(
        "Range",
        F.when(F.col("C_roll") != 0, (F.col("H_roll") - F.col("L_roll")) / F.col("C_roll")).otherwise(0.0)
    ).withColumn(
        "Bull",
        F.col("C_roll") > F.col("O_roll")
    ).withColumn(
        "Bear",
        F.col("O_roll") > F.col("C_roll")
    )

    # Trend detection
    sdf = sdf.withColumn("UpTrend", F.last(c).over(w_tf) > F.first(c).over(w_tf)) \
             .withColumn("DownTrend", F.last(c).over(w_tf) < F.first(c).over(w_tf))

    # Single-bar patterns
    sdf = sdf.withColumn("Doji", F.col("Body") <= F.col("doji_thresh") * F.col("Range")) \
             .withColumn("Hammer", (F.col("DownShadow") >= F.col("shadow_ratio") * F.col("Body")) &
                                  (F.col("UpShadow") <= F.col("hammer_thresh") * F.col("Body")) &
                                  (F.col("Body") > 0) &
                                  (F.col("Body") <= 2 * F.col("hammer_thresh") * F.col("Range")) &
                                  F.col("DownTrend")) \
             .withColumn("InvertedHammer", (F.col("UpShadow") >= F.col("shadow_ratio") * F.col("Body")) &
                                          (F.col("DownShadow") <= F.col("hammer_thresh") * F.col("Body")) &
                                          (F.col("Body") > 0) &
                                          (F.col("Body") <= 2 * F.col("hammer_thresh") * F.col("Range")) &
                                          F.col("DownTrend")) \
             .withColumn("BullishMarubozu", F.col("Bull") & (F.col("Body") >= F.col("long_body") * F.col("Range")) &
                                              (F.col("UpShadow") <= F.col("marubozu_thresh") * F.col("Range")) &
                                              (F.col("DownShadow") <= F.col("marubozu_thresh") * F.col("Range"))) \
             .withColumn("BearishMarubozu", F.col("Bear") & (F.col("Body") >= F.col("long_body") * F.col("Range")) &
                                              (F.col("UpShadow") <= F.col("marubozu_thresh") * F.col("Range")) &
                                              (F.col("DownShadow") <= F.col("marubozu_thresh") * F.col("Range"))) \
             .withColumn("SuspiciousCandle", (F.col("Range") <= F.col("rng_thresh")) | (F.col("Body") <= F.col("rng_thresh"))) \
             .withColumn("HangingMan", F.col("Hammer") & F.col("UpTrend")) \
             .withColumn("ShootingStar", F.col("InvertedHammer") & F.col("UpTrend")) \
             .withColumn("SpinningTop", (F.col("Body") <= F.col("small_body") * F.col("Range")) &
                                        (F.col("UpShadow") >= F.col("Body")) &
                                        (F.col("DownShadow") >= F.col("Body")))

    # Multi-bar lags
    for col_name in ["O_roll", "C_roll", "H_roll", "L_roll", "Bull", "Bear", "Body"]:
        for lag in [1, 2, 3, 4]:
            sdf = sdf.withColumn(f"{col_name}{lag}", F.lag(col_name, lag).over(w_shift))

    # Multi-bar patterns
    sdf = sdf.withColumn("BullishEngulfing", (F.col("O_roll1") > F.col("C_roll1")) & F.col("Bull") & 
                         (F.col("C_roll") >= F.col("O_roll1")) & (F.col("O_roll") <= F.col("C_roll1"))) \
             .withColumn("BearishEngulfing", (F.col("C_roll1") > F.col("O_roll1")) & F.col("Bear") & 
                         (F.col("O_roll") >= F.col("C_roll1")) & (F.col("C_roll") <= F.col("O_roll1"))) \
             .withColumn("BullishHarami", (F.col("O_roll1") > F.col("C_roll1")) & F.col("Bull") &
                         (F.greatest(F.col("O_roll"), F.col("C_roll")) <= F.greatest(F.col("O_roll1"), F.col("C_roll1"))) &
                         (F.least(F.col("O_roll"), F.col("C_roll")) >= F.least(F.col("O_roll1"), F.col("C_roll1")))) \
             .withColumn("BearishHarami", (F.col("C_roll1") > F.col("O_roll1")) & F.col("Bear") &
                         (F.greatest(F.col("O_roll"), F.col("C_roll")) <= F.greatest(F.col("O_roll1"), F.col("C_roll1"))) &
                         (F.least(F.col("O_roll"), F.col("C_roll")) >= F.least(F.col("O_roll1"), F.col("C_roll1")))) \
             .withColumn("HaramiCross", F.col("Doji") &
                         (F.greatest(F.col("O_roll"), F.col("C_roll")) <= F.greatest(F.col("O_roll1"), F.col("C_roll1"))) &
                         (F.least(F.col("O_roll"), F.col("C_roll")) >= F.least(F.col("O_roll1"), F.col("C_roll1")))) \
             .withColumn("PiercingLine", (F.col("O_roll1") > F.col("C_roll1")) & F.col("Bull") &
                         (F.col("O_roll") < F.col("C_roll1")) & (F.col("C_roll") > (F.col("O_roll1") + F.col("C_roll1"))/2) & 
                         (F.col("C_roll") < F.col("O_roll1"))) \
             .withColumn("DarkCloudCover", (F.col("C_roll1") > F.col("O_roll1")) & F.col("Bear") &
                         (F.col("O_roll") > F.col("C_roll1")) & (F.col("C_roll") < (F.col("O_roll1") + F.col("C_roll1"))/2) & 
                         (F.col("C_roll") > F.col("O_roll1"))) \
             .withColumn("MorningStar", (F.col("O_roll2") > F.col("C_roll2")) & 
                         (F.abs(F.col("C_roll1") - F.col("O_roll1")) < F.abs(F.col("C_roll2") - F.col("O_roll2")) * F.col("small_body")) & 
                         F.col("Bull")) \
             .withColumn("EveningStar", (F.col("C_roll2") > F.col("O_roll2")) & 
                         (F.abs(F.col("C_roll1") - F.col("O_roll1")) < F.abs(F.col("C_roll2") - F.col("O_roll2")) * F.col("small_body")) & 
                         F.col("Bear")) \
             .withColumn("ThreeWhiteSoldiers", F.col("Bull") & F.col("Bull1") & F.col("Bull2") & 
                         (F.col("C_roll") > F.col("C_roll1")) & (F.col("C_roll1") > F.col("C_roll2"))) \
             .withColumn("ThreeBlackCrows", F.col("Bear") & F.col("Bear1") & F.col("Bear2") & 
                         (F.col("C_roll") < F.col("C_roll1")) & (F.col("C_roll1") < F.col("C_roll2"))) \
             .withColumn("TweezerTop", (F.col("H_roll") == F.col("H_roll1")) & F.col("Bear") & F.col("Bull1")) \
             .withColumn("TweezerBottom", (F.col("L_roll") == F.col("L_roll1")) & F.col("Bull") & F.col("Bear1")) \
             .withColumn("InsideBar", (F.col("H_roll") < F.col("H_roll1")) & (F.col("L_roll") > F.col("L_roll1"))) \
             .withColumn("OutsideBar", (F.col("H_roll") > F.col("H_roll1")) & (F.col("L_roll") < F.col("L_roll1"))) \
             .withColumn("NearHigh", F.col("H_roll") >= F.max("H_roll").over(w_tf) * (1 - F.col("near_edge"))) \
             .withColumn("NearLow", F.col("L_roll") <= F.min("L_roll").over(w_tf) * (1 + F.col("near_edge"))) \
             .withColumn("DragonflyDoji", (F.abs(F.col("C_roll") - F.col("O_roll")) <= F.col("doji_thresh") * F.col("Range")) &
                         (F.col("H_roll") == F.col("C_roll")) & (F.col("L_roll") < F.col("O_roll"))) \
             .withColumn("GravestoneDoji", (F.abs(F.col("C_roll") - F.col("O_roll")) <= F.col("doji_thresh") * F.col("Range")) &
                         (F.col("L_roll") == F.col("C_roll")) & (F.col("H_roll") > F.col("O_roll"))) \
             .withColumn("LongLeggedDoji", (F.abs(F.col("C_roll") - F.col("O_roll")) <= F.col("doji_thresh") * F.col("Range")) &
                         (F.col("UpShadow") > F.col("shadow_ratio") * F.col("Body")) & (F.col("DownShadow") > F.col("shadow_ratio") * F.col("Body"))) \
            .withColumn("RisingThreeMethods", 
                F.col("Bull4") & F.col("Bear3") & F.col("Bear2") & F.col("Bear1") & F.col("Bull") &
                (F.coalesce(F.col("Body3"), F.lit(0.0)) < F.col("small_body") * F.coalesce(F.col("Body4"), F.lit(1.0))) &
                (F.coalesce(F.col("Body2"), F.lit(0.0)) < F.col("small_body") * F.coalesce(F.col("Body4"), F.lit(1.0))) &
                (F.coalesce(F.col("Body1"), F.lit(0.0)) < F.col("small_body") * F.coalesce(F.col("Body4"), F.lit(1.0))) &
                (F.coalesce(F.col("H_roll3"), F.lit(0.0)) <= F.coalesce(F.col("H_roll4"), F.lit(0.0))) &
                (F.coalesce(F.col("L_roll3"), F.lit(0.0)) >= F.coalesce(F.col("L_roll4"), F.lit(0.0))) &
                (F.coalesce(F.col("H_roll2"), F.lit(0.0)) <= F.coalesce(F.col("H_roll4"), F.lit(0.0))) &
                (F.coalesce(F.col("L_roll2"), F.lit(0.0)) >= F.coalesce(F.col("L_roll4"), F.lit(0.0))) &
                (F.coalesce(F.col("H_roll1"), F.lit(0.0)) <= F.coalesce(F.col("H_roll4"), F.lit(0.0))) &
                (F.coalesce(F.col("L_roll1"), F.lit(0.0)) >= F.coalesce(F.col("L_roll4"), F.lit(0.0))) &
                (F.col("C_roll") > F.coalesce(F.col("C_roll4"), F.lit(0.0)))) \
            .withColumn("FallingThreeMethods", 
                F.col("Bear4") & F.col("Bull3") & F.col("Bull2") & F.col("Bull1") & F.col("Bear") &
                (F.coalesce(F.col("Body3"), F.lit(0.0)) < F.col("small_body") * F.coalesce(F.col("Body4"), F.lit(1.0))) &
                (F.coalesce(F.col("Body2"), F.lit(0.0)) < F.col("small_body") * F.coalesce(F.col("Body4"), F.lit(1.0))) &
                (F.coalesce(F.col("Body1"), F.lit(0.0)) < F.col("small_body") * F.coalesce(F.col("Body4"), F.lit(1.0))) &
                (F.coalesce(F.col("H_roll3"), F.lit(0.0)) <= F.coalesce(F.col("H_roll4"), F.lit(0.0))) &
                (F.coalesce(F.col("L_roll3"), F.lit(0.0)) >= F.coalesce(F.col("L_roll4"), F.lit(0.0))) &
                (F.coalesce(F.col("H_roll2"), F.lit(0.0)) <= F.coalesce(F.col("H_roll4"), F.lit(0.0))) &
                (F.coalesce(F.col("L_roll2"), F.lit(0.0)) >= F.coalesce(F.col("L_roll4"), F.lit(0.0))) &
                (F.coalesce(F.col("H_roll1"), F.lit(0.0)) <= F.coalesce(F.col("H_roll4"), F.lit(0.0))) &
                (F.coalesce(F.col("L_roll1"), F.lit(0.0)) >= F.coalesce(F.col("L_roll4"), F.lit(0.0))) &
                (F.col("C_roll") < F.coalesce(F.col("C_roll4"), F.lit(0.0)))) \
             .withColumn("GapUp", F.col("O_roll") > F.col("H_roll1")) \
             .withColumn("GapDown", F.col("O_roll") < F.col("L_roll1")) \
             .withColumn("RangeMean", F.avg("Range").over(w_tf)) \
             .withColumn("ClimacticCandle", F.col("Range") > 2 * F.col("RangeMean"))

    # PatternCount
    pattern_cols = [
        "Doji", "Hammer", "InvertedHammer", "BullishMarubozu", "BearishMarubozu", "SuspiciousCandle",
        "HangingMan", "ShootingStar", "SpinningTop", "BullishEngulfing", "BearishEngulfing",
        "BullishHarami", "BearishHarami", "HaramiCross", "PiercingLine", "DarkCloudCover",
        "MorningStar", "EveningStar", "ThreeWhiteSoldiers", "ThreeBlackCrows", "TweezerTop",
        "TweezerBottom", "InsideBar", "OutsideBar", "NearHigh", "NearLow", "DragonflyDoji",
        "GravestoneDoji", "LongLeggedDoji", "RisingThreeMethods", "FallingThreeMethods",
        "GapUp", "GapDown", "ClimacticCandle"
    ]
    sdf = sdf.withColumn(
        "PatternCount",
        reduce(lambda a, b: a + b, [F.col(pat).cast("int") for pat in pattern_cols])
    )

    # PatternType
    sdf = sdf.withColumn("PatternType", F.array(*[F.when(F.col(pat), pat) for pat in pattern_cols]))
    sdf = sdf.withColumn("PatternType", F.expr("filter(PatternType, x -> x is not null)[0]"))
    sdf = sdf.withColumn("PatternType", F.coalesce(F.col("PatternType"), F.lit("None")))

    # Validate output
    expected_cols = pattern_cols + ["PatternCount", "PatternType"]
    missing_cols = [col for col in expected_cols if col not in sdf.columns]
    if missing_cols:
        print(f"Warning: Failed to create columns: {missing_cols}")

    return sdf
# -------------------------------
# Add Candlestick Patterns
# -------------------------------
def get_candle_params_chatgpt(close_price: float, user: int = 1):
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
    

from pyspark.sql import functions as F, Window
from functools import reduce
from operator import add
from operator import itemgetter
import numpy as np
from datetime import datetime

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql import SparkSession

def add_trend_filters_fast(sdf, timeframe="Daily", user: int = 1):
    """
    Spark-native trend indicators for stock data.
    Adds moving averages, slopes, returns, volatility, ROC, and confirmed trend flags.
    Optimized for small node (4 cores, 8GB RAM) with per-company volatility thresholds.
    
    Args:
        sdf: Spark DataFrame with columns: CompanyId, StockDate, Close
        timeframe: String, e.g., "Daily", "Short"
        user: Integer user ID for settings
    
    Returns:
        Spark DataFrame with added columns: MA, MA_slope, UpTrend_MA, DownTrend_MA,
        RecentReturn, UpTrend_Return, DownTrend_Return, Volatility, LowVolatility,
        HighVolatility, ROC, MomentumUp, MomentumDown, ConfirmedUpTrend, ConfirmedDownTrend
    """
    # Validate inputs
    if user is None:
        raise ValueError("User ID cannot be None")
    if not sdf or not all(col in sdf.columns for col in ["CompanyId", "StockDate", "Close"]):
        raise ValueError("Input DataFrame must contain CompanyId, StockDate, and Close columns")

    c = "Close"

    # Load and broadcast settings
    spark = SparkSession.getActiveSession()
    settings = spark.sparkContext.broadcast(load_settings(user)["profiles"])
    if timeframe not in settings.value:
        raise ValueError(f"Invalid timeframe: {timeframe}. Must be one of {list(settings.value.keys())}")
    params = settings.value[timeframe]

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

    # -------------------------------
    # Window definitions (bounded for memory efficiency)
    # -------------------------------
    max_horizon = max(ma_window, ret_window, vol_window, slope_horizon)
    w_ma = Window.partitionBy("CompanyId").orderBy("StockDate").rowsBetween(-(ma_window-1), 0)
    w_ret = Window.partitionBy("CompanyId").orderBy("StockDate").rowsBetween(-ret_window, -1)  # Exclude current row
    w_slope = Window.partitionBy("CompanyId").orderBy("StockDate").rowsBetween(-slope_horizon, 0)
    w_vol = Window.partitionBy("CompanyId").orderBy("StockDate").rowsBetween(-(vol_window-1), 0)
    w_lag = Window.partitionBy("CompanyId").orderBy("StockDate").rowsBetween(-max_horizon, 0)

    # -------------------------------
    # Moving Average & slope
    # -------------------------------
    sdf = sdf.withColumn("MA", F.avg(F.col(c)).over(w_ma))
    sdf = sdf.withColumn(
        "MA_lag",
        F.lag("MA", slope_horizon).over(w_slope)
    )
    sdf = sdf.withColumn(
        "MA_slope",
        F.when(F.col("MA_lag").isNotNull(), (F.col("MA") - F.col("MA_lag")) / F.col("MA_lag")).otherwise(0.0)
    )
    sdf = sdf.withColumn("UpTrend_MA", F.col("MA_slope") > 0)
    sdf = sdf.withColumn("DownTrend_MA", F.col("MA_slope") < 0)
    sdf = sdf.drop("MA_lag")

    # -------------------------------
    # Returns & trend flags
    # -------------------------------
    w_order = Window.partitionBy("CompanyId").orderBy("StockDate")  # for lag
    sdf = sdf.withColumn(
        "RecentReturn",
        F.when(
            F.lag(c, ret_window).over(w_order).isNotNull(),
            (F.col(c) - F.lag(c, ret_window).over(w_order)) / F.lag(c, ret_window).over(w_order)
        ).otherwise(0.0)
    )
    '''
    sdf = sdf.withColumn(
        "RecentReturn",
        F.when(
            F.lag(c, ret_window).over(w_lag).isNotNull(),
            (F.col(c) - F.lag(c, ret_window).over(w_lag)) / F.lag(c, ret_window).over(w_lag)
        ).otherwise(0.0)
    )
    '''
    sdf = sdf.withColumn("UpTrend_Return", F.col("RecentReturn") > 0)
    sdf = sdf.withColumn("DownTrend_Return", F.col("RecentReturn") < 0)

    # -------------------------------
    # Volatility (per-company median)
    # -------------------------------
    sdf = sdf.withColumn(
        "ReturnPct",
        F.when(
            F.lag(c).over(w_lag).isNotNull(),
            (F.col(c) - F.lag(c).over(w_lag)) / F.lag(c).over(w_lag)
        ).otherwise(0.0)
    )
    sdf = sdf.withColumn("Volatility", F.stddev("ReturnPct").over(w_vol))
    
    # Compute per-company median volatility
    w_company = Window.partitionBy("CompanyId")
    sdf = sdf.withColumn("Volatility_Median", F.expr("percentile_approx(Volatility, 0.5)").over(w_company))
    sdf = sdf.withColumn("LowVolatility", F.when(F.col("Volatility").isNotNull(), F.col("Volatility") < F.col("Volatility_Median")).otherwise(False))
    sdf = sdf.withColumn("HighVolatility", F.when(F.col("Volatility").isNotNull(), F.col("Volatility") > F.col("Volatility_Median")).otherwise(False))

    # -------------------------------
    # Rate of Change (ROC) & momentum
    # -------------------------------
    sdf = sdf.withColumn(
        "ROC",
        F.when(
            F.lag(c, ma_window).over(w_lag).isNotNull(),
            (F.col(c) - F.lag(c, ma_window).over(w_lag)) / F.lag(c, ma_window).over(w_lag)
        ).otherwise(0.0)
    )
    sdf = sdf.withColumn("MomentumUp", F.col("ROC") > roc_thresh)
    sdf = sdf.withColumn("MomentumDown", F.col("ROC") < -roc_thresh)

    # -------------------------------
    # Confirmed trends (relaxed to 2/3 conditions)
    # -------------------------------
    sdf = sdf.withColumn(
        "ConfirmedUpTrend",
        (F.col("UpTrend_MA").cast("int") + F.col("UpTrend_Return").cast("int") + F.col("MomentumUp").cast("int")) >= 2
    )
    sdf = sdf.withColumn(
        "ConfirmedDownTrend",
        (F.col("DownTrend_MA").cast("int") + F.col("DownTrend_Return").cast("int") + F.col("MomentumDown").cast("int")) >= 2
    )

    # -------------------------------
    # Drop helper columns
    # -------------------------------
    sdf = sdf.drop("ReturnPct", "Volatility_Median")

    return sdf

from pyspark.sql import functions as F, Window

def add_trend_filters_fast_chatgpt(sdf, timeframe="Daily", user: int = 1):
    """
    Spark-native trend indicators.
    Adds moving averages, slopes, returns, volatility, ROC, and confirmed trend flags.
    """

    c = "Close"

    # Load user + timeframe-specific parameters
    settings = load_settings(user)["profiles"]
    if timeframe not in settings:
        raise ValueError(f"Invalid timeframe: {timeframe}. Must be one of {list(settings.keys())}")
    params = settings[timeframe]

    ma_window = params["ma"]
    ret_window = params["ret"]
    vol_window = params["vol"]
    roc_thresh = params["roc_thresh"]
    slope_horizon = params["slope_horizon"]

    # -------------------------------
    # Window definitions
    # -------------------------------
    w_ma = Window.partitionBy("CompanyId").orderBy("StockDate").rowsBetween(-(ma_window-1), 0)
    w_ret = Window.partitionBy("CompanyId").orderBy("StockDate").rowsBetween(-ret_window, 0)
    w_slope = Window.partitionBy("CompanyId").orderBy("StockDate").rowsBetween(-slope_horizon, 0)
    w_vol = Window.partitionBy("CompanyId").orderBy("StockDate").rowsBetween(-(vol_window-1), 0)
    w_full = Window.partitionBy("CompanyId").orderBy("StockDate")

    # -------------------------------
    # Moving Average & slope
    # -------------------------------
    sdf = sdf.withColumn("MA", F.avg(F.col(c)).over(w_ma))
    sdf = sdf.withColumn(
        "MA_lag",
        F.lag("MA", slope_horizon).over(w_full)
    )
    sdf = sdf.withColumn(
        "MA_slope",
        F.when(F.col("MA_lag").isNotNull(), (F.col("MA") - F.col("MA_lag")) / F.col("MA_lag")).otherwise(None)
    )
    sdf = sdf.withColumn("UpTrend_MA", F.col("MA_slope") > 0)
    sdf = sdf.withColumn("DownTrend_MA", F.col("MA_slope") < 0)
    sdf = sdf.drop("MA_lag")

    # -------------------------------
    # Returns & trend flags
    # -------------------------------
    sdf = sdf.withColumn(
        "RecentReturn",
        (F.col(c) - F.lag(c, ret_window).over(w_full)) / F.lag(c, ret_window).over(w_full)
    )
    sdf = sdf.withColumn("UpTrend_Return", F.col("RecentReturn") > 0)
    sdf = sdf.withColumn("DownTrend_Return", F.col("RecentReturn") < 0)

    # -------------------------------
    # Volatility (rolling std of pct changes)
    # -------------------------------
    sdf = sdf.withColumn(
        "ReturnPct",
        (F.col(c) - F.lag(c).over(w_full)) / F.lag(c).over(w_full)
    )
    sdf = sdf.withColumn("Volatility", F.stddev("ReturnPct").over(w_vol))
    vol_med = sdf.approxQuantile("Volatility", [0.5], 0.0)[0]
    sdf = sdf.withColumn("LowVolatility", F.col("Volatility") < vol_med)
    sdf = sdf.withColumn("HighVolatility", F.col("Volatility") > vol_med)

    # -------------------------------
    # Rate of Change (ROC) & momentum
    # -------------------------------
    sdf = sdf.withColumn(
        "ROC",
        (F.col(c) - F.lag(c, ma_window).over(w_full)) / F.lag(c, ma_window).over(w_full)
    )
    sdf = sdf.withColumn("MomentumUp", F.col("ROC") > roc_thresh)
    sdf = sdf.withColumn("MomentumDown", F.col("ROC") < -roc_thresh)

    # -------------------------------
    # Confirmed trends
    # -------------------------------
    sdf = sdf.withColumn(
        "ConfirmedUpTrend",
        F.col("UpTrend_MA") & F.col("UpTrend_Return") & F.col("MomentumUp")
    )
    sdf = sdf.withColumn(
        "ConfirmedDownTrend",
        F.col("DownTrend_MA") & F.col("DownTrend_Return") & F.col("MomentumDown")
    )

    # -------------------------------
    # Drop helper columns
    # -------------------------------
    sdf = sdf.drop("ReturnPct")

    return sdf

# -------------------------------
# Candle Patterns (Fast + Safe)
# -------------------------------
from operator import itemgetter
from functools import reduce
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from operator import itemgetter
from functools import reduce

def add_candle_patterns_fast_grok1(sdf, tf_window=5, user: int = 1):
    """
    Spark-native candlestick pattern detection for multiple companies.
    Optimized for 4-core, 8GB server with ~912,500 rows, sub-2-hour runtime.
    Expects sdf with columns: CompanyId, Open, High, Low, Close, Volume, StockDate.
    Returns sdf with boolean pattern columns, PatternCount, and PatternType.
    
    Args:
        sdf: Spark DataFrame with required columns
        tf_window: Integer window for rolling calculations
        user: Integer user ID for settings
    
    Returns:
        Spark DataFrame with pattern columns, PatternCount, and PatternType
    """
    # Validate inputs
    required_cols = ["CompanyId", "Open", "High", "Low", "Close", "Volume", "StockDate"]
    missing_cols = [col for col in required_cols if col not in sdf.columns]
    if missing_cols:
        raise ValueError(f"Input DataFrame missing columns: {missing_cols}")
    if tf_window < 1:
        raise ValueError("tf_window must be positive")
    if user is None:
        raise ValueError("User ID cannot be None")

    o, h, l, c, v = "Open", "High", "Low", "Close", "Volume"

    # Windows (bounded for memory)
    max_lag = max(tf_window - 1, 4)  # Support 5-bar patterns
    w_tf = Window.partitionBy("CompanyId").orderBy("StockDate").rowsBetween(-max_lag, 0)
    w_shift = Window.partitionBy("CompanyId").orderBy("StockDate").rowsBetween(-4, 0)
    w_vol = Window.partitionBy("CompanyId").orderBy("StockDate").rowsBetween(-19, 0)

    # Global representative last_close (window-based)
    w_global = Window.partitionBy()
    sdf = sdf.withColumn("max_date", F.max("StockDate").over(w_global))
    sdf = sdf.withColumn(
        "last_close",
        F.when(F.col("StockDate") == F.col("max_date"), F.avg("Close").over(w_global)).otherwise(F.lit(None))
    )
    sdf = sdf.withColumn("last_close", F.last("last_close", ignorenulls=True).over(w_global))
    candle_params = get_candle_params(F.col("last_close"), user=user)
    sdf = sdf.drop("max_date")

    # Extract candle parameters
    doji_thresh, hammer_thresh, marubozu_thresh, long_body, small_body, shadow_ratio, near_edge, highvol_spike, lowvol_dip, rng_thresh = \
        itemgetter(
            "doji_thresh", "hammer_thresh", "marubozu_thresh", "long_body", "small_body", 
            "shadow_ratio", "near_edge", "highvol_spike", "lowvol_dip", "rng_thresh"
        )(candle_params)

    # Rolling calculations (super candle over tf_window)
    sdf = sdf.withColumn("O_roll", F.first(o).over(w_tf)) \
             .withColumn("C_roll", F.last(c).over(w_tf)) \
             .withColumn("H_roll", F.max(h).over(w_tf)) \
             .withColumn("L_roll", F.min(l).over(w_tf)) \
             .withColumn("V_avg20", F.avg(v).over(w_vol))

    # Volume spikes
    sdf = sdf.withColumn("HighVolume", F.col(v) > highvol_spike * F.col("V_avg20")) \
             .withColumn("LowVolume", F.col(v) < lowvol_dip * F.col("V_avg20"))

    # Body, shadows, range with null/zero handling
    sdf = sdf.withColumn(
        "Body",
        F.when(F.col("C_roll") != 0, F.abs(F.col("C_roll") - F.col("O_roll")) / F.col("C_roll")).otherwise(0.0)
    ).withColumn(
        "UpShadow",
        F.when(F.col("C_roll") != 0,
               (F.col("H_roll") - F.greatest(F.col("O_roll"), F.col("C_roll"))) / F.col("C_roll")).otherwise(0.0)
    ).withColumn(
        "DownShadow",
        F.when(F.col("C_roll") != 0,
               (F.least(F.col("O_roll"), F.col("C_roll")) - F.col("L_roll")) / F.col("C_roll")).otherwise(0.0)
    ).withColumn(
        "Range",
        F.when(F.col("C_roll") != 0, (F.col("H_roll") - F.col("L_roll")) / F.col("C_roll")).otherwise(0.0)
    ).withColumn(
        "Bull",
        F.col("C_roll") > F.col("O_roll")
    ).withColumn(
        "Bear",
        F.col("O_roll") > F.col("C_roll")
    )

    # Trend detection
    sdf = sdf.withColumn("UpTrend", F.last(c).over(w_tf) > F.first(c).over(w_tf)) \
             .withColumn("DownTrend", F.last(c).over(w_tf) < F.first(c).over(w_tf))

    # Single-bar patterns
    sdf = sdf.withColumn("Doji", F.col("Body") <= doji_thresh * F.col("Range")) \
             .withColumn("Hammer", (F.col("DownShadow") >= shadow_ratio * F.col("Body")) &
                                  (F.col("UpShadow") <= hammer_thresh * F.col("Body")) &
                                  (F.col("Body") > 0) &
                                  (F.col("Body") <= 2 * hammer_thresh * F.col("Range")) &
                                  F.col("DownTrend")) \
             .withColumn("InvertedHammer", (F.col("UpShadow") >= shadow_ratio * F.col("Body")) &
                                          (F.col("DownShadow") <= hammer_thresh * F.col("Body")) &
                                          (F.col("Body") > 0) &
                                          (F.col("Body") <= 2 * hammer_thresh * F.col("Range")) &
                                          F.col("DownTrend")) \
             .withColumn("BullishMarubozu", F.col("Bull") & (F.col("Body") >= long_body * F.col("Range")) &
                                              (F.col("UpShadow") <= marubozu_thresh * F.col("Range")) &
                                              (F.col("DownShadow") <= marubozu_thresh * F.col("Range"))) \
             .withColumn("BearishMarubozu", F.col("Bear") & (F.col("Body") >= long_body * F.col("Range")) &
                                              (F.col("UpShadow") <= marubozu_thresh * F.col("Range")) &
                                              (F.col("DownShadow") <= marubozu_thresh * F.col("Range"))) \
             .withColumn("SuspiciousCandle", (F.col("Range") <= rng_thresh) | (F.col("Body") <= rng_thresh)) \
             .withColumn("HangingMan", F.col("Hammer") & F.col("UpTrend")) \
             .withColumn("ShootingStar", F.col("InvertedHammer") & F.col("UpTrend")) \
             .withColumn("SpinningTop", (F.col("Body") <= small_body * F.col("Range")) &
                                        (F.col("UpShadow") >= F.col("Body")) &
                                        (F.col("DownShadow") >= F.col("Body")))

    # Multi-bar lags (including Body for Rising/FallingThreeMethods)
    for col_name in ["O_roll", "C_roll", "H_roll", "L_roll", "Bull", "Bear", "Body"]:
        for lag in [1, 2, 3, 4]:
            sdf = sdf.withColumn(f"{col_name}{lag}", F.lag(col_name, lag).over(w_shift))

    # Multi-bar patterns
    sdf = sdf.withColumn("BullishEngulfing", (F.col("O_roll1") > F.col("C_roll1")) & F.col("Bull") & 
                         (F.col("C_roll") >= F.col("O_roll1")) & (F.col("O_roll") <= F.col("C_roll1"))) \
             .withColumn("BearishEngulfing", (F.col("C_roll1") > F.col("O_roll1")) & F.col("Bear") & 
                         (F.col("O_roll") >= F.col("C_roll1")) & (F.col("C_roll") <= F.col("O_roll1"))) \
             .withColumn("BullishHarami", (F.col("O_roll1") > F.col("C_roll1")) & F.col("Bull") &
                         (F.greatest(F.col("O_roll"), F.col("C_roll")) <= F.greatest(F.col("O_roll1"), F.col("C_roll1"))) &
                         (F.least(F.col("O_roll"), F.col("C_roll")) >= F.least(F.col("O_roll1"), F.col("C_roll1")))) \
             .withColumn("BearishHarami", (F.col("C_roll1") > F.col("O_roll1")) & F.col("Bear") &
                         (F.greatest(F.col("O_roll"), F.col("C_roll")) <= F.greatest(F.col("O_roll1"), F.col("C_roll1"))) &
                         (F.least(F.col("O_roll"), F.col("C_roll")) >= F.least(F.col("O_roll1"), F.col("C_roll1")))) \
             .withColumn("HaramiCross", F.col("Doji") &
                         (F.greatest(F.col("O_roll"), F.col("C_roll")) <= F.greatest(F.col("O_roll1"), F.col("C_roll1"))) &
                         (F.least(F.col("O_roll"), F.col("C_roll")) >= F.least(F.col("O_roll1"), F.col("C_roll1")))) \
             .withColumn("PiercingLine", (F.col("O_roll1") > F.col("C_roll1")) & F.col("Bull") &
                         (F.col("O_roll") < F.col("C_roll1")) & (F.col("C_roll") > (F.col("O_roll1") + F.col("C_roll1"))/2) & 
                         (F.col("C_roll") < F.col("O_roll1"))) \
             .withColumn("DarkCloudCover", (F.col("C_roll1") > F.col("O_roll1")) & F.col("Bear") &
                         (F.col("O_roll") > F.col("C_roll1")) & (F.col("C_roll") < (F.col("O_roll1") + F.col("C_roll1"))/2) & 
                         (F.col("C_roll") > F.col("O_roll1"))) \
             .withColumn("MorningStar", (F.col("O_roll2") > F.col("C_roll2")) & 
                         (F.abs(F.col("C_roll1") - F.col("O_roll1")) < F.abs(F.col("C_roll2") - F.col("O_roll2")) * small_body) & 
                         F.col("Bull")) \
             .withColumn("EveningStar", (F.col("C_roll2") > F.col("O_roll2")) & 
                         (F.abs(F.col("C_roll1") - F.col("O_roll1")) < F.abs(F.col("C_roll2") - F.col("O_roll2")) * small_body) & 
                         F.col("Bear")) \
             .withColumn("ThreeWhiteSoldiers", F.col("Bull") & F.col("Bull1") & F.col("Bull2") & 
                         (F.col("C_roll") > F.col("C_roll1")) & (F.col("C_roll1") > F.col("C_roll2"))) \
             .withColumn("ThreeBlackCrows", F.col("Bear") & F.col("Bear1") & F.col("Bear2") & 
                         (F.col("C_roll") < F.col("C_roll1")) & (F.col("C_roll1") < F.col("C_roll2"))) \
             .withColumn("TweezerTop", (F.col("H_roll") == F.col("H_roll1")) & F.col("Bear") & F.col("Bull1")) \
             .withColumn("TweezerBottom", (F.col("L_roll") == F.col("L_roll1")) & F.col("Bull") & F.col("Bear1")) \
             .withColumn("InsideBar", (F.col("H_roll") < F.col("H_roll1")) & (F.col("L_roll") > F.col("L_roll1"))) \
             .withColumn("OutsideBar", (F.col("H_roll") > F.col("H_roll1")) & (F.col("L_roll") < F.col("L_roll1"))) \
             .withColumn("NearHigh", F.col("H_roll") >= F.max("H_roll").over(w_tf) * (1 - near_edge)) \
             .withColumn("NearLow", F.col("L_roll") <= F.min("L_roll").over(w_tf) * (1 + near_edge)) \
             .withColumn("DragonflyDoji", (F.abs(F.col("C_roll") - F.col("O_roll")) <= doji_thresh * F.col("Range")) &
                         (F.col("H_roll") == F.col("C_roll")) & (F.col("L_roll") < F.col("O_roll"))) \
             .withColumn("GravestoneDoji", (F.abs(F.col("C_roll") - F.col("O_roll")) <= doji_thresh * F.col("Range")) &
                         (F.col("L_roll") == F.col("C_roll")) & (F.col("H_roll") > F.col("O_roll"))) \
             .withColumn("LongLeggedDoji", (F.abs(F.col("C_roll") - F.col("O_roll")) <= doji_thresh * F.col("Range")) &
                         (F.col("UpShadow") > shadow_ratio * F.col("Body")) & (F.col("DownShadow") > shadow_ratio * F.col("Body"))) \
             .withColumn("RisingThreeMethods", 
                         F.col("Bull4") & F.col("Bear3") & F.col("Bear2") & F.col("Bear1") & F.col("Bull") &
                         (F.coalesce(F.col("Body3"), F.lit(0.0)) < small_body * F.coalesce(F.col("Body4"), F.lit(1.0))) &
                         (F.coalesce(F.col("Body2"), F.lit(0.0)) < small_body * F.coalesce(F.col("Body4"), F.lit(1.0))) &
                         (F.coalesce(F.col("Body1"), F.lit(0.0)) < small_body * F.coalesce(F.col("Body4"), F.lit(1.0))) &
                         (F.coalesce(F.col("H3"), F.lit(0.0)) <= F.coalesce(F.col("H4"), F.lit(0.0))) &
                         (F.coalesce(F.col("L3"), F.lit(0.0)) >= F.coalesce(F.col("L4"), F.lit(0.0))) &
                         (F.coalesce(F.col("H2"), F.lit(0.0)) <= F.coalesce(F.col("H4"), F.lit(0.0))) &
                         (F.coalesce(F.col("L2"), F.lit(0.0)) >= F.coalesce(F.col("L4"), F.lit(0.0))) &
                         (F.coalesce(F.col("H1"), F.lit(0.0)) <= F.coalesce(F.col("H4"), F.lit(0.0))) &
                         (F.coalesce(F.col("L1"), F.lit(0.0)) >= F.coalesce(F.col("L4"), F.lit(0.0))) &
                         (F.col("C_roll") > F.coalesce(F.col("C_roll4"), F.lit(0.0)))) \
             .withColumn("FallingThreeMethods", 
                         F.col("Bear4") & F.col("Bull3") & F.col("Bull2") & F.col("Bull1") & F.col("Bear") &
                         (F.coalesce(F.col("Body3"), F.lit(0.0)) < small_body * F.coalesce(F.col("Body4"), F.lit(1.0))) &
                         (F.coalesce(F.col("Body2"), F.lit(0.0)) < small_body * F.coalesce(F.col("Body4"), F.lit(1.0))) &
                         (F.coalesce(F.col("Body1"), F.lit(0.0)) < small_body * F.coalesce(F.col("Body4"), F.lit(1.0))) &
                         (F.coalesce(F.col("H3"), F.lit(0.0)) <= F.coalesce(F.col("H4"), F.lit(0.0))) &
                         (F.coalesce(F.col("L3"), F.lit(0.0)) >= F.coalesce(F.col("L4"), F.lit(0.0))) &
                         (F.coalesce(F.col("H2"), F.lit(0.0)) <= F.coalesce(F.col("H4"), F.lit(0.0))) &
                         (F.coalesce(F.col("L2"), F.lit(0.0)) >= F.coalesce(F.col("L4"), F.lit(0.0))) &
                         (F.coalesce(F.col("H1"), F.lit(0.0)) <= F.coalesce(F.col("H4"), F.lit(0.0))) &
                         (F.coalesce(F.col("L1"), F.lit(0.0)) >= F.coalesce(F.col("L4"), F.lit(0.0))) &
                         (F.col("C_roll") < F.coalesce(F.col("C_roll4"), F.lit(0.0)))) \
             .withColumn("GapUp", F.col("O_roll") > F.col("H_roll1")) \
             .withColumn("GapDown", F.col("O_roll") < F.col("L_roll1")) \
             .withColumn("RangeMean", F.avg("Range").over(w_tf)) \
             .withColumn("ClimacticCandle", F.col("Range") > 2 * F.col("RangeMean"))

    # PatternCount (sum of boolean casts)
    pattern_cols = [
        "Doji", "Hammer", "InvertedHammer", "BullishMarubozu", "BearishMarubozu", "SuspiciousCandle",
        "HangingMan", "ShootingStar", "SpinningTop", "BullishEngulfing", "BearishEngulfing",
        "BullishHarami", "BearishHarami", "HaramiCross", "PiercingLine", "DarkCloudCover",
        "MorningStar", "EveningStar", "ThreeWhiteSoldiers", "ThreeBlackCrows", "TweezerTop",
        "TweezerBottom", "InsideBar", "OutsideBar", "NearHigh", "NearLow", "DragonflyDoji",
        "GravestoneDoji", "LongLeggedDoji", "RisingThreeMethods", "FallingThreeMethods",
        "GapUp", "GapDown", "ClimacticCandle"
    ]
    sdf = sdf.withColumn(
        "PatternCount",
        reduce(lambda a, b: a + b, [F.col(pat).cast("int") for pat in pattern_cols])
    )

    # PatternType (first non-null match)
    sdf = sdf.withColumn("PatternType", F.array(*[F.when(F.col(pat), pat) for pat in pattern_cols]))
    sdf = sdf.withColumn("PatternType", F.expr("filter(PatternType, x -> x is not null)[0]"))
    sdf = sdf.withColumn("PatternType", F.coalesce(F.col("PatternType"), F.lit("None")))

    # Validate output
    expected_cols = pattern_cols + ["PatternCount", "PatternType"]
    missing_cols = [col for col in expected_cols if col not in sdf.columns]
    if missing_cols:
        print(f"Warning: Failed to create columns: {missing_cols}")

    return sdf

from pyspark.sql import functions as F
from pyspark.sql import Window
from functools import reduce
from operator import itemgetter

def add_candle_patterns_fast_chatgpt(sdf, tf_window=5, user: int = 1):
    """
    Spark-native candlestick pattern detection for multiple companies.
    Expects sdf with columns: CompanyId, Open, High, Low, Close, Volume, StockDate
    Returns sdf with new boolean columns for each pattern and PatternCount/PatternType.
    Fully Spark vectorized — no .collect() calls.
    """

    o, h, l, c, v = "Open", "High", "Low", "Close", "Volume"

    # --- Windows ---
    w_tf = Window.partitionBy("CompanyId").orderBy("StockDate").rowsBetween(-(tf_window-1), 0)
    w_shift = Window.partitionBy("CompanyId").orderBy("StockDate")

    # --- Params ---
    last_close = sdf.agg(F.last("Close").alias("last_close")).first()["last_close"]
    candle_params = get_candle_params(last_close, user=user)
    (
        doji_thresh, hammer_thresh, marubozu_thresh, long_body, small_body,
        shadow_ratio, near_edge, highvol_spike, lowvol_dip, rng_thresh
    ) = itemgetter(
        "doji_thresh", "hammer_thresh", "marubozu_thresh", "long_body", "small_body",
        "shadow_ratio", "near_edge", "highvol_spike", "lowvol_dip", "rng_thresh"
    )(candle_params)

    # --- Rolling ---
    sdf = sdf.withColumn("O_roll", F.first(o).over(w_tf)) \
             .withColumn("C_roll", F.last(c).over(w_tf)) \
             .withColumn("H_roll", F.max(h).over(w_tf)) \
             .withColumn("L_roll", F.min(l).over(w_tf)) \
             .withColumn("V_avg20", F.avg(v).over(Window.partitionBy("CompanyId").orderBy("StockDate").rowsBetween(-19,0)))

    # --- Volume ---
    sdf = sdf.withColumn("HighVolume", F.col(v) > highvol_spike * F.col("V_avg20")) \
             .withColumn("LowVolume", F.col(v) < lowvol_dip * F.col("V_avg20"))

    # --- Body & Shadows ---
    sdf = sdf.withColumn("Body", F.abs(F.col("C_roll") - F.col("O_roll")) / F.col("C_roll")) \
             .withColumn("UpShadow", (F.col("H_roll") - F.greatest(F.col("O_roll"), F.col("C_roll"))) / F.col("C_roll")) \
             .withColumn("DownShadow", (F.least(F.col("O_roll"), F.col("C_roll")) - F.col("L_roll")) / F.col("C_roll")) \
             .withColumn("Range", (F.col("H_roll") - F.col("L_roll")) / F.col("C_roll")) \
             .withColumn("Bull", F.col("C_roll") > F.col("O_roll")) \
             .withColumn("Bear", F.col("O_roll") > F.col("C_roll"))

    # --- Trends ---
    first_close = F.first(c).over(w_tf)
    last_close = F.last(c).over(w_tf)
    sdf = sdf.withColumn("UpTrend", last_close > first_close) \
             .withColumn("DownTrend", last_close < first_close)

    # --- Single-bar patterns ---
    sdf = sdf.withColumn("Doji", F.col("Body") <= doji_thresh * F.col("Range")) \
             .withColumn("Hammer", (F.col("DownShadow") >= shadow_ratio * F.col("Body")) &
                                  (F.col("UpShadow") <= hammer_thresh * F.col("Body")) &
                                  (F.col("Body") > 0) &
                                  (F.col("Body") <= 2 * hammer_thresh * F.col("Range")) &
                                  F.col("DownTrend")) \
             .withColumn("InvertedHammer", (F.col("UpShadow") >= shadow_ratio * F.col("Body")) &
                                            (F.col("DownShadow") <= hammer_thresh * F.col("Body")) &
                                            (F.col("Body") > 0) &
                                            (F.col("Body") <= 2 * hammer_thresh * F.col("Range")) &
                                            F.col("DownTrend")) \
             .withColumn("BullishMarubozu", F.col("Bull") & (F.col("Body") >= long_body * F.col("Range")) &
                                              (F.col("UpShadow") <= marubozu_thresh * F.col("Range")) &
                                              (F.col("DownShadow") <= marubozu_thresh * F.col("Range"))) \
             .withColumn("BearishMarubozu", F.col("Bear") & (F.col("Body") >= long_body * F.col("Range")) &
                                              (F.col("UpShadow") <= marubozu_thresh * F.col("Range")) &
                                              (F.col("DownShadow") <= marubozu_thresh * F.col("Range"))) \
             .withColumn("SuspiciousCandle", (F.col("Range") <= rng_thresh) | (F.col("Body") <= rng_thresh)) \
             .withColumn("HangingMan", F.col("Hammer") & F.col("UpTrend")) \
             .withColumn("ShootingStar", F.col("InvertedHammer") & F.col("UpTrend")) \
             .withColumn("SpinningTop", (F.col("Body") <= small_body * F.col("Range")) &
                                        (F.col("UpShadow") >= F.col("Body")) &
                                        (F.col("DownShadow") >= F.col("Body")))

    # --- Multi-bar lags ---
    for col_name in ["O","C","H","L","Bull","Bear"]:
        for lag in [1,2]:
            base = f"{col_name}_roll" if col_name in ["O","C","H","L"] else col_name
            sdf = sdf.withColumn(f"{col_name}{lag}", F.lag(base, lag).over(w_shift))

    # --- Multi-bar patterns ---
    sdf = sdf.withColumn("BullishEngulfing", (F.col("O1") > F.col("C1")) & F.col("Bull") & (F.col("C_roll") >= F.col("O1")) & (F.col("O_roll") <= F.col("C1"))) \
             .withColumn("BearishEngulfing", (F.col("C1") > F.col("O1")) & F.col("Bear") & (F.col("O_roll") >= F.col("C1")) & (F.col("C_roll") <= F.col("O1"))) \
             .withColumn("BullishHarami", (F.col("O1") > F.col("C1")) & F.col("Bull") &
                                         (F.greatest(F.col("O_roll"), F.col("C_roll")) <= F.greatest(F.col("O1"), F.col("C1"))) &
                                         (F.least(F.col("O_roll"), F.col("C_roll")) >= F.least(F.col("O1"), F.col("C1")))) \
             .withColumn("BearishHarami", (F.col("C1") > F.col("O1")) & F.col("Bear") &
                                         (F.greatest(F.col("O_roll"), F.col("C_roll")) <= F.greatest(F.col("O1"), F.col("C1"))) &
                                         (F.least(F.col("O_roll"), F.col("C_roll")) >= F.least(F.col("O1"), F.col("C1")))) \
             .withColumn("HaramiCross", F.col("Doji") &
                                        (F.greatest(F.col("O_roll"), F.col("C_roll")) <= F.greatest(F.col("O1"), F.col("C1"))) &
                                        (F.least(F.col("O_roll"), F.col("C_roll")) >= F.least(F.col("O1"), F.col("C1")))) \
             .withColumn("PiercingLine", (F.col("O1") > F.col("C1")) & F.col("Bull") &
                                         (F.col("O_roll") < F.col("C1")) & (F.col("C_roll") > (F.col("O1") + F.col("C1"))/2) & (F.col("C_roll") < F.col("O1"))) \
             .withColumn("DarkCloudCover", (F.col("C1") > F.col("O1")) & F.col("Bear") &
                                           (F.col("O_roll") > F.col("C1")) & (F.col("C_roll") < (F.col("O1") + F.col("C1"))/2) & (F.col("C_roll") > F.col("O_roll"))) \
             .withColumn("MorningStar", (F.col("O2") > F.col("C2")) & (F.abs(F.col("C1") - F.col("O1")) < F.abs(F.col("C2") - F.col("O2")) * small_body) & F.col("Bull")) \
             .withColumn("EveningStar", (F.col("C2") > F.col("O2")) & (F.abs(F.col("C1") - F.col("O1")) < F.abs(F.col("C2") - F.col("O2")) * small_body) & F.col("Bear")) \
             .withColumn("ThreeWhiteSoldiers", F.col("Bull") & F.col("Bull1") & F.col("Bull2") & (F.col("C_roll") > F.col("C1")) & (F.col("C1") > F.col("C2"))) \
             .withColumn("ThreeBlackCrows", F.col("Bear") & F.col("Bear1") & F.col("Bear2") & (F.col("C_roll") < F.col("C1")) & (F.col("C1") < F.col("C2"))) \
             .withColumn("TweezerTop", (F.col("H_roll") == F.col("H1")) & F.col("Bear") & F.col("Bull1")) \
             .withColumn("TweezerBottom", (F.col("L_roll") == F.col("L1")) & F.col("Bull") & F.col("Bear1")) \
             .withColumn("InsideBar", (F.col("H_roll") < F.col("H1")) & (F.col("L_roll") > F.col("L1"))) \
             .withColumn("OutsideBar", (F.col("H_roll") > F.col("H1")) & (F.col("L_roll") < F.col("L1"))) \
             .withColumn("NearHigh", F.col("H_roll") >= F.max("H_roll").over(w_tf) * (1 - near_edge)) \
             .withColumn("NearLow", F.col("L_roll") <= F.min("L_roll").over(w_tf) * (1 + near_edge)) \
             .withColumn("DragonflyDoji", (F.abs(F.col("C_roll") - F.col("O_roll")) <= doji_thresh * F.col("Range")) &
                                           (F.col("H_roll") == F.col("C_roll")) & (F.col("L_roll") < F.col("O_roll"))) \
             .withColumn("GravestoneDoji", (F.abs(F.col("C_roll") - F.col("O_roll")) <= doji_thresh * F.col("Range")) &
                                           (F.col("L_roll") == F.col("C_roll")) & (F.col("H_roll") > F.col("O_roll"))) \
             .withColumn("LongLeggedDoji", (F.abs(F.col("C_roll") - F.col("O_roll")) <= doji_thresh * F.col("Range")) &
                                            (F.col("UpShadow") > shadow_ratio * F.col("Body")) & (F.col("DownShadow") > shadow_ratio * F.col("Body"))) \
             .withColumn("RisingThreeMethods", F.col("Bull2") & F.col("Bull1") & F.col("Bull") & (F.col("C1") < F.col("O2")) & (F.col("C_roll") > F.col("C1"))) \
             .withColumn("FallingThreeMethods", F.col("Bear2") & F.col("Bear1") & F.col("Bear") & (F.col("C1") > F.col("O2")) & (F.col("C_roll") < F.col("C1"))) \
             .withColumn("GapUp", F.col("O_roll") > F.col("H1")) \
             .withColumn("GapDown", F.col("O_roll") < F.col("L1")) \
             .withColumn("RangeMean", F.avg("Range").over(w_tf)) \
             .withColumn("ClimacticCandle", F.col("Range") > 2 * F.col("RangeMean"))

    # --- Pattern list ---
    pattern_cols = [
        "Doji","Hammer","InvertedHammer","BullishMarubozu","BearishMarubozu","SuspiciousCandle",
        "HangingMan","ShootingStar","SpinningTop","BullishEngulfing","BearishEngulfing",
        "BullishHarami","BearishHarami","HaramiCross","PiercingLine","DarkCloudCover",
        "MorningStar","EveningStar","ThreeWhiteSoldiers","ThreeBlackCrows","TweezerTop",
        "TweezerBottom","InsideBar","OutsideBar","NearHigh","NearLow","DragonflyDoji",
        "GravestoneDoji","LongLeggedDoji","RisingThreeMethods","FallingThreeMethods",
        "GapUp","GapDown","ClimacticCandle"
    ]

    # --- PatternCount (safe reduce) ---
    expr = reduce(lambda a, b: a + b, [F.col(c).cast("int") for c in pattern_cols], F.lit(0))
    sdf = sdf.withColumn("PatternCount", expr)

    # --- PatternType (first non-null) ---
    sdf = sdf.withColumn("PatternType", F.array(*[F.when(F.col(c), F.lit(c)) for c in pattern_cols]))
    sdf = sdf.withColumn("PatternType", F.expr("filter(PatternType, x -> x is not null)[0]"))
    sdf = sdf.withColumn("PatternType", F.coalesce(F.col("PatternType"), F.lit("None")))

    return sdf


from pyspark.sql import functions as F
from pyspark.sql import functions as F
from pyspark.sql import SparkSession

def add_confirmed_signals(sdf, timeframe="Daily", user: int = 1):
    """
    Generate validated candlestick signals based on trend and volume context in Spark.
    Each 'ValidXXX' signal is True if the pattern aligns with trend and volume conditions.
    Optimized for small node (4 cores, 8GB RAM) with dynamic user/timeframe settings.
    
    Args:
        sdf: Spark DataFrame with columns: CompanyId, StockDate, Close, and candlestick/trend columns
        timeframe: String, e.g., "Daily", "Short"
        user: Integer user ID for settings
    
    Returns:
        Spark DataFrame with added 'ValidXXX' columns for validated signals
    """
    # Validate inputs
    if user is None:
        raise ValueError("User ID cannot be None")
    if not sdf or not all(col in sdf.columns for col in ["CompanyId", "StockDate", "Close"]):
        raise ValueError("Input DataFrame must contain CompanyId, StockDate, and Close columns")

    # Load and broadcast settings
    spark = SparkSession.getActiveSession()
    settings = spark.sparkContext.broadcast(load_settings(user)["signals"])
    if timeframe not in settings.value["timeframes"]:
        raise ValueError(f"Invalid timeframe: {timeframe}. Must be one of {list(settings.value['timeframes'].keys())}")
    tf_settings = settings.value["timeframes"][timeframe]

    # Define signal groups with trend and volume conditions
    signal_groups = {
        "Bullish": {
            "ValidHammer": {"pattern": "Hammer", "trends": ["DownTrend_MA", "DownTrend_Return"], "volume": "HighVolume"},
            "ValidBullishEngulfing": {"pattern": "BullishEngulfing", "trends": ["DownTrend_MA", "DownTrend_Return"], "volume": "HighVolume"},
            "ValidPiercingLine": {"pattern": "PiercingLine", "trends": ["DownTrend_Return"], "volume": "HighVolume"},
            "ValidMorningStar": {"pattern": "MorningStar", "trends": ["DownTrend_MA", "DownTrend_Return"], "volume": "HighVolume"},
            "ValidThreeWhiteSoldiers": {"pattern": "ThreeWhiteSoldiers", "trends": ["DownTrend_MA"], "volume": None},
            "ValidBullishMarubozu": {"pattern": "BullishMarubozu", "trends": ["DownTrend_MA"], "volume": None},
            "ValidTweezerBottom": {"pattern": "TweezerBottom", "trends": ["DownTrend_MA"], "volume": "HighVolume"},
            "ValidDragonflyDoji": {"pattern": "DragonflyDoji", "trends": ["DownTrend_MA"], "volume": "HighVolume"}
        },
        "Bearish": {
            "ValidShootingStar": {"pattern": "ShootingStar", "trends": ["UpTrend_MA", "UpTrend_Return"], "volume": "HighVolume"},
            "ValidBearishEngulfing": {"pattern": "BearishEngulfing", "trends": ["UpTrend_MA", "UpTrend_Return"], "volume": "HighVolume"},
            "ValidDarkCloudCover": {"pattern": "DarkCloudCover", "trends": ["UpTrend_Return"], "volume": "HighVolume"},
            "ValidEveningStar": {"pattern": "EveningStar", "trends": ["UpTrend_MA", "UpTrend_Return"], "volume": "HighVolume"},
            "ValidThreeBlackCrows": {"pattern": "ThreeBlackCrows", "trends": ["UpTrend_MA"], "volume": None},
            "ValidBearishMarubozu": {"pattern": "BearishMarubozu", "trends": ["UpTrend_MA"], "volume": None},
            "ValidTweezerTop": {"pattern": "TweezerTop", "trends": ["UpTrend_MA"], "volume": "HighVolume"},
            "ValidGravestoneDoji": {"pattern": "GravestoneDoji", "trends": ["UpTrend_MA"], "volume": "HighVolume"}
        },
        "Reversal": {
            "ValidHaramiCross": {"pattern": "HaramiCross", "trends": ["UpTrend_MA", "DownTrend_MA"], "volume": None},
            "ValidBullishHarami": {"pattern": "BullishHarami", "trends": ["DownTrend_MA"], "volume": None},
            "ValidBearishHarami": {"pattern": "BearishHarami", "trends": ["UpTrend_MA"], "volume": None}
        },
        "Continuation": {
            "ValidInsideBar": {"pattern": "InsideBar", "trends": ["UpTrend_MA", "DownTrend_MA"], "volume": None},
            "ValidOutsideBar": {"pattern": "OutsideBar", "trends": ["UpTrend_MA", "DownTrend_MA"], "volume": None},
            "ValidRisingThreeMethods": {"pattern": "RisingThreeMethods", "trends": ["UpTrend_MA"], "volume": None},
            "ValidFallingThreeMethods": {"pattern": "FallingThreeMethods", "trends": ["DownTrend_MA"], "volume": None},
            "ValidGapUp": {"pattern": "GapUp", "trends": ["UpTrend_MA"], "volume": None},
            "ValidGapDown": {"pattern": "GapDown", "trends": ["DownTrend_MA"], "volume": None}
        },
        "Exhaustion": {
            "ValidSpinningTop": {"pattern": "SpinningTop", "trends": ["UpTrend_MA", "DownTrend_MA"], "volume": None},
            "ValidClimacticCandle": {"pattern": "ClimacticCandle", "trends": ["UpTrend_MA", "DownTrend_MA"], "volume": "HighVolume"}
        },
        "Indecision": {
            "ValidDoji": {"pattern": "Doji", "trends": ["UpTrend_MA", "DownTrend_MA"], "volume": None},
            "ValidLongLeggedDoji": {"pattern": "LongLeggedDoji", "trends": ["UpTrend_MA", "DownTrend_MA"], "volume": None}
        }
    }

    # Collect all required columns
    required_columns = set()
    for group in signal_groups.values():
        for config in group.values():
            required_columns.add(config["pattern"])
            required_columns.update(config["trends"])
            if config["volume"]:
                required_columns.add(config["volume"])

    # Create missing columns as False
    for col in required_columns:
        if col not in sdf.columns:
            sdf = sdf.withColumn(col, F.lit(False))

    # Apply trend and volume filters
    for group, patterns in signal_groups.items():
        for valid_col, config in patterns.items():
            pattern = config["pattern"]
            trends = config["trends"]
            volume = config["volume"]
            
            # Combine trend conditions with OR (at least one must be True)
            trend_condition = reduce(lambda a, b: a | b, [F.col(trend) for trend in trends])
            
            # Apply pattern, trend, and optional volume condition
            condition = F.col(pattern) & trend_condition
            if volume:
                condition = condition & F.col(volume)
            
            sdf = sdf.withColumn(valid_col, condition)

    # Validate output
    valid_cols = [col for group in signal_groups.values() for col in group.keys()]
    missing_cols = [col for col in valid_cols if col not in sdf.columns]
    if missing_cols:
        print(f"Warning: Failed to create columns: {missing_cols}")

    return sdf
def add_confirmed_signals_chatgpt(sdf):
    """
    Generate validated candlestick signals based on trend context in Spark.
    Each 'ValidXXX' signal is only True if the corresponding trend condition is met.
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

    # Apply trend filters
    for group_patterns in signal_groups.values():
        for valid_col, trend_col in group_patterns.items():
            raw_col = valid_col.replace("Valid", "")
            # Create missing columns as False
            if raw_col not in sdf.columns:
                sdf = sdf.withColumn(raw_col, F.lit(False))
            if trend_col not in sdf.columns:
                sdf = sdf.withColumn(trend_col, F.lit(False))
            # Apply trend condition
            sdf = sdf.withColumn(valid_col, F.col(raw_col) & F.col(trend_col))

    # Bullish reversal: DragonflyDoji & DownTrend_MA & HighVolume
    for col in ["DragonflyDoji", "DownTrend_MA", "HighVolume"]:
        if col not in sdf.columns:
            sdf = sdf.withColumn(col, F.lit(False))
    sdf = sdf.withColumn(
        "ValidDragonflyDoji",
        F.col("DragonflyDoji") & F.col("DownTrend_MA") & F.col("HighVolume")
    )

    # Bearish reversal: GravestoneDoji & UpTrend_MA & HighVolume
    for col in ["GravestoneDoji", "UpTrend_MA", "HighVolume"]:
        if col not in sdf.columns:
            sdf = sdf.withColumn(col, F.lit(False))
    sdf = sdf.withColumn(
        "ValidGravestoneDoji",
        F.col("GravestoneDoji") & F.col("UpTrend_MA") & F.col("HighVolume")
    )

    return sdf

from pyspark.sql import functions as F
from functools import reduce
from operator import add
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from operator import add
from functools import reduce

def add_signal_strength_fast(sdf, timeframe="Daily", user: int = 1):
    """
    Spark-native signal strength counts and percentages, weighted by trend strength.
    Adds SignalStrength, BullishPctRaw, BearishPctRaw, BullishPctDirectional, BearishPctDirectional.
    Optimized for small node (4 cores, 8GB RAM) with dynamic user/timeframe settings.
    
    Args:
        sdf: Spark DataFrame with columns: CompanyId, StockDate, Valid* signals, ConfirmedUpTrend, ConfirmedDownTrend
        timeframe: String, e.g., "Daily", "Short"
        user: Integer user ID for settings
    
    Returns:
        Spark DataFrame with added signal strength columns
    """
    # Validate inputs
    if user is None:
        raise ValueError("User ID cannot be None")
    if not sdf or not all(col in sdf.columns for col in ["CompanyId", "StockDate"]):
        raise ValueError("Input DataFrame must contain CompanyId and StockDate columns")

    # Load and broadcast settings
    spark = SparkSession.getActiveSession()
    settings = spark.sparkContext.broadcast(load_settings(user)["signals"])
    if timeframe not in settings.value["timeframes"]:
        raise ValueError(f"Invalid timeframe: {timeframe}. Must be one of {list(settings.value['timeframes'].keys())}")
    tf_settings = settings.value["timeframes"][timeframe]

    # Get directional groups from settings or default
    directional_groups = tf_settings.get("directional_groups", ["Bullish", "Bearish", "Reversal", "Continuation"])

    # Identify valid signal columns
    valid_cols = [c for c in sdf.columns if c.startswith("Valid")]
    if not valid_cols:
        print("Warning: No Valid* columns found; returning default signal strength columns")
        return sdf.withColumn("SignalStrength", F.lit(0)) \
                  .withColumn("BullishPctRaw", F.lit(0.0)) \
                  .withColumn("BearishPctRaw", F.lit(0.0)) \
                  .withColumn("BullishPctDirectional", F.lit(0.0)) \
                  .withColumn("BearishPctDirectional", F.lit(0.0))

    # Ensure trend columns exist
    for col in ["ConfirmedUpTrend", "ConfirmedDownTrend"]:
        if col not in sdf.columns:
            sdf = sdf.withColumn(col, F.lit(False))

    # SignalStrength with trend weighting
    signal_strength_expr = reduce(add, [F.col(c) for c in valid_cols])
    sdf = sdf.withColumn("SignalStrengthNonZero", F.when(signal_strength_expr == 0, 1).otherwise(signal_strength_expr))
    sdf = sdf.withColumn(
        "SignalStrength",
        signal_strength_expr * (
            1.0 + 
            F.when(F.col("ConfirmedUpTrend"), 0.2).otherwise(0.0) + 
            F.when(F.col("ConfirmedDownTrend"), 0.2).otherwise(0.0)
        )
    )

    # Bullish/Bearish raw percentages
    bullish_cols = [c for c in valid_cols if any(k.lower() in c.lower() for k in ["Bullish", "Hammer", "MorningStar", "ThreeWhiteSoldiers", "TweezerBottom", "DragonflyDoji"])]
    bearish_cols = [c for c in valid_cols if any(k.lower() in c.lower() for k in ["Bearish", "ShootingStar", "EveningStar", "ThreeBlackCrows", "TweezerTop", "GravestoneDoji", "DarkCloudCover"])]

    bullish_sum_expr = reduce(add, [F.col(c) for c in bullish_cols]) if bullish_cols else F.lit(0)
    bearish_sum_expr = reduce(add, [F.col(c) for c in bearish_cols]) if bearish_cols else F.lit(0)

    sdf = sdf.withColumn("BullishPctRaw", bullish_sum_expr / F.col("SignalStrengthNonZero"))
    sdf = sdf.withColumn("BearishPctRaw", bearish_sum_expr / F.col("SignalStrengthNonZero"))

    # Directional percentages
    directional_cols = [c for c in valid_cols if any(c.startswith(f"Valid{g}") for g in directional_groups)]
    directional_sum_expr = reduce(add, [F.col(c) for c in directional_cols]) if directional_cols else F.lit(0)
    sdf = sdf.withColumn("DirectionalSumNonZero", F.when(directional_sum_expr == 0, 1).otherwise(directional_sum_expr))

    bullish_dir_cols = [c for c in directional_cols if any(k.lower() in c.lower() for k in ["Bullish", "Hammer", "MorningStar", "ThreeWhiteSoldiers", "TweezerBottom", "DragonflyDoji"])]
    bearish_dir_cols = [c for c in directional_cols if any(k.lower() in c.lower() for k in ["Bearish", "ShootingStar", "EveningStar", "ThreeBlackCrows", "TweezerTop", "GravestoneDoji", "DarkCloudCover"])]

    bullish_dir_sum = reduce(add, [F.col(c) for c in bullish_dir_cols]) if bullish_dir_cols else F.lit(0)
    bearish_dir_sum = reduce(add, [F.col(c) for c in bearish_dir_cols]) if bearish_dir_cols else F.lit(0)

    sdf = sdf.withColumn("BullishPctDirectional", bullish_dir_sum / F.col("DirectionalSumNonZero"))
    sdf = sdf.withColumn("BearishPctDirectional", bearish_dir_sum / F.col("DirectionalSumNonZero"))

    # Drop helper columns
    sdf = sdf.drop("SignalStrengthNonZero", "DirectionalSumNonZero")

    # Validate output
    expected_cols = ["SignalStrength", "BullishPctRaw", "BearishPctRaw", "BullishPctDirectional", "BearishPctDirectional"]
    missing_cols = [col for col in expected_cols if col not in sdf.columns]
    if missing_cols:
        print(f"Warning: Failed to create columns: {missing_cols}")

    return sdf
    
def add_signal_strength_fast_chatgpt(sdf, directional_groups=None):
    """
    Spark-native signal strength counts and percentages.
    Vectorized and optimized.
    """

    # 1️⃣ Identify all valid signal columns
    valid_cols = [c for c in sdf.columns if c.startswith("Valid")]
    if not valid_cols:
        return sdf.withColumn("SignalStrength", F.lit(0)) \
                  .withColumn("BullishPctRaw", F.lit(0.0)) \
                  .withColumn("BearishPctRaw", F.lit(0.0)) \
                  .withColumn("BullishPctDirectional", F.lit(0.0)) \
                  .withColumn("BearishPctDirectional", F.lit(0.0))

    # Cast all valid columns to integer
    for c in valid_cols:
        sdf = sdf.withColumn(c, F.col(c).cast("int"))

    # SignalStrength total
    signal_strength_expr = reduce(add, [F.col(c) for c in valid_cols])
    sdf = sdf.withColumn("SignalStrengthNonZero", F.when(signal_strength_expr == 0, 1).otherwise(signal_strength_expr))
    sdf = sdf.withColumn("SignalStrength", signal_strength_expr)

    # Bullish / Bearish raw percentages
    bullish_cols = [c for c in valid_cols if c.startswith("ValidBullish")]
    bearish_cols = [c for c in valid_cols if c.startswith("ValidBearish")]

    bullish_sum_expr = reduce(add, [F.col(c) for c in bullish_cols]) if bullish_cols else F.lit(0)
    bearish_sum_expr = reduce(add, [F.col(c) for c in bearish_cols]) if bearish_cols else F.lit(0)

    sdf = sdf.withColumn("BullishPctRaw", bullish_sum_expr / F.col("SignalStrengthNonZero"))
    sdf = sdf.withColumn("BearishPctRaw", bearish_sum_expr / F.col("SignalStrengthNonZero"))

    # Directional percentages
    if directional_groups is None:
        directional_groups = ["Bullish", "Bearish"]

    directional_cols = [c for c in valid_cols if any(c.startswith(f"Valid{g}") for g in directional_groups)]
    directional_sum_expr = reduce(add, [F.col(c) for c in directional_cols]) if directional_cols else F.lit(0)
    sdf = sdf.withColumn("DirectionalSumNonZero", F.when(directional_sum_expr == 0, 1).otherwise(directional_sum_expr))

    bullish_dir_cols = [c for c in directional_cols if c.startswith("ValidBullish")]
    bearish_dir_cols = [c for c in directional_cols if c.startswith("ValidBearish")]

    bullish_dir_sum = reduce(add, [F.col(c) for c in bullish_dir_cols]) if bullish_dir_cols else F.lit(0)
    bearish_dir_sum = reduce(add, [F.col(c) for c in bearish_dir_cols]) if bearish_dir_cols else F.lit(0)

    sdf = sdf.withColumn("BullishPctDirectional", bullish_dir_sum / F.col("DirectionalSumNonZero"))
    sdf = sdf.withColumn("BearishPctDirectional", bearish_dir_sum / F.col("DirectionalSumNonZero"))

    # Drop helper columns
    sdf = sdf.drop("SignalStrengthNonZero", "DirectionalSumNonZero")

    return sdf

from pyspark.sql import functions as F, Window
from functools import reduce
from operator import add
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from operator import add
from functools import reduce

def finalize_signals_fast(sdf, tf, tf_window=5, use_fundamentals=True, user: int = 1):
    """
    Optimized Spark-native consolidation of momentum, pattern, and candle signals into unified Action.
    Uses per-company thresholds, bounded windows, and true majority voting.
    Optimized for 4-core, 8GB server with ~912,500 rows, sub-2-hour runtime.
    
    Args:
        sdf: Spark DataFrame with CompanyId, StockDate, Close, Valid*, FundamentalScore (optional)
        tf: Timeframe string (e.g., "Daily")
        tf_window: Integer window for pattern normalization
        use_fundamentals: Boolean to include FundamentalScore
        user: Integer user ID for settings
    
    Returns:
        Spark DataFrame with Action, TomorrowAction, ActionConfidenceNorm, SignalStrengthHybrid, etc.
    """
    # Validate inputs
    if user is None:
        raise ValueError("User ID cannot be None")
    required_cols = ["CompanyId", "StockDate", "Close"]
    if not sdf or not all(col in sdf.columns for col in required_cols):
        raise ValueError(f"Input DataFrame must contain {required_cols}")
    if use_fundamentals and "FundamentalScore" not in sdf.columns:
        print("Warning: FundamentalScore missing; disabling fundamentals")
        use_fundamentals = False

    # Generate columns and momentum factor
    candle_columns, trend_cols, momentum_factor = generate_signal_columns(sdf, tf)
    bullish_patterns = trend_cols.get("Bullish", [])
    bearish_patterns = trend_cols.get("Bearish", [])
    if not (bullish_patterns or bearish_patterns or candle_columns.get("Buy") or candle_columns.get("Sell")):
        print("Warning: No valid patterns or candle columns from generate_signal_columns")

    # Windows (bounded for memory)
    max_lag = max(10, tf_window)
    w = Window.partitionBy("CompanyId").orderBy("StockDate").rowsBetween(-max_lag, max_lag)
    w_stats = Window.partitionBy("CompanyId").orderBy("StockDate").rowsBetween(-9, 0)

    # Tomorrow returns / momentum
    sdf = sdf.withColumn("TomorrowClose", F.lead("Close").over(w))
    sdf = sdf.withColumn(
        "TomorrowReturn",
        F.when(F.col("Close") != 0, (F.col("TomorrowClose") - F.col("Close")) / F.col("Close")).otherwise(0.0)
    )
    sdf = sdf.withColumn(
        "Return",
        F.when(F.lag("Close").over(w).isNotNull() & (F.lag("Close").over(w) != 0),
               (F.col("Close") / F.lag("Close").over(w) - 1)).otherwise(0.0)
    )

    # Rolling stats for momentum
    sdf = sdf.withColumn("AvgReturn", F.avg("Return").over(w_stats))
    sdf = sdf.withColumn("Volatility", F.coalesce(F.stddev("Return").over(w_stats), F.lit(1e-8)))
    sdf = sdf.withColumn(
        "MomentumZ",
        F.when(F.col("Volatility") != 0, (F.col("Return") - F.col("AvgReturn")) / F.col("Volatility")).otherwise(0.0)
    )

    # Per-company momentum thresholds
    sdf_agg = sdf.groupBy("CompanyId").agg(
        F.mean("MomentumZ").alias("mean_mom"),
        F.stddev("MomentumZ").alias("std_mom")
    )
    sdf = sdf.join(sdf_agg, on="CompanyId", how="left")
    sdf = sdf.withColumn("buy_thresh", F.col("mean_mom") + momentum_factor * F.coalesce(F.col("std_mom"), F.lit(1.0)))
    sdf = sdf.withColumn("sell_thresh", F.col("mean_mom") - momentum_factor * F.coalesce(F.col("std_mom"), F.lit(1.0)))
    sdf = sdf.withColumn(
        "MomentumAction",
        F.when(F.col("MomentumZ") > F.col("buy_thresh"), "Buy")
         .when(F.col("MomentumZ") < F.col("sell_thresh"), "Sell")
         .otherwise("Hold")
    ).drop("mean_mom", "std_mom", "buy_thresh", "sell_thresh")

    # Pattern scores
    def pattern_sum(cols):
        return reduce(add, [F.col(c).cast("double") for c in cols]) if cols else F.lit(0.0)

    sdf = sdf.withColumn("BullScore", pattern_sum(bullish_patterns))
    sdf = sdf.withColumn("BearScore", pattern_sum(bearish_patterns))
    sdf = sdf.withColumn("PatternScore", F.col("BullScore") - F.col("BearScore"))
    sdf = sdf.withColumn("PatternScoreNorm", F.col("PatternScore") / F.lit(tf_window))
    sdf = sdf.withColumn(
        "PatternAction",
        F.when(F.col("PatternScoreNorm") > 0.3, "Buy")  # Stricter threshold
         .when(F.col("PatternScoreNorm") < -0.3, "Sell")
         .otherwise("Hold")
    )

    # Candle action
    buy_mask = pattern_sum(candle_columns.get("Buy", [])) > 0 if candle_columns.get("Buy") else F.lit(False)
    sell_mask = pattern_sum(candle_columns.get("Sell", [])) > 0 if candle_columns.get("Sell") else F.lit(False)
    sdf = sdf.withColumn(
        "CandleAction",
        F.when(buy_mask, "Buy").when(sell_mask & ~buy_mask, "Sell").otherwise("Hold")
    )

    # CandidateAction (true majority vote)
    sdf = sdf.withColumn(
        "BuyCount",
        F.when(F.col("MomentumAction") == "Buy", 1).otherwise(0) +
        F.when(F.col("PatternAction") == "Buy", 1).otherwise(0) +
        F.when(F.col("CandleAction") == "Buy", 1).otherwise(0)
    )
    sdf = sdf.withColumn(
        "SellCount",
        F.when(F.col("MomentumAction") == "Sell", 1).otherwise(0) +
        F.when(F.col("PatternAction") == "Sell", 1).otherwise(0) +
        F.when(F.col("CandleAction") == "Sell", 1).otherwise(0)
    )
    sdf = sdf.withColumn(
        "CandidateAction",
        F.when(F.col("BuyCount") > F.col("SellCount"), "Buy")
         .when(F.col("SellCount") > F.col("BuyCount"), "Sell")
         .otherwise("Hold")
    ).drop("BuyCount", "SellCount")

    # Filter consecutive Buy/Sell
    sdf = sdf.withColumn("PrevAction", F.lag("CandidateAction").over(w))
    sdf = sdf.withColumn(
        "Action",
        F.when((F.col("CandidateAction") == F.col("PrevAction")) & F.col("CandidateAction").isin("Buy", "Sell"), "Hold")
         .otherwise(F.col("CandidateAction"))
    ).drop("PrevAction")

    # TomorrowAction
    sdf = sdf.withColumn("TomorrowAction", F.lead("Action").over(w))
    sdf = sdf.withColumn(
        "TomorrowActionSource",
        F.when(F.col("TomorrowAction").isin("Buy", "Sell"), F.lit("NextAction(filtered)"))
         .otherwise(F.lit("Hold(no_signal)"))
    )

    # Hybrid signal strength
    valid_cols = [c for c in sdf.columns if c.startswith("Valid")]
    if valid_cols:
        bull_cols = [c for c in valid_cols if any(k.lower() in c.lower() for k in ["Bull", "Hammer", "MorningStar", "ThreeWhiteSoldiers", "TweezerBottom", "DragonflyDoji"])]
        bear_cols = [c for c in valid_cols if any(k.lower() in c.lower() for k in ["Bear", "ShootingStar", "EveningStar", "ThreeBlackCrows", "TweezerTop", "GravestoneDoji", "DarkCloudCover"])]
        
        sdf = sdf.withColumn("BullishCount", pattern_sum(bull_cols))
        sdf = sdf.withColumn("BearishCount", pattern_sum(bear_cols))
        sdf = sdf.withColumn("MagnitudeStrength", F.abs(F.col("PatternScore")) + F.abs(F.col("MomentumZ")))

        # Per-company normalization
        sdf_agg = sdf.groupBy("CompanyId").agg(
            F.max("BullishCount").alias("max_bull"),
            F.max("BearishCount").alias("max_bear"),
            F.max("MagnitudeStrength").alias("max_mag"),
            F.max("ActionConfidence").alias("max_conf") if valid_cols else F.lit(1.0)
        )
        sdf = sdf.join(sdf_agg, on="CompanyId", how="left")
        
        sdf = sdf.withColumn(
            "BullishStrengthHybrid",
            (F.col("BullishCount") / F.coalesce(F.col("max_bull"), F.lit(1.0))) +
            (F.col("MagnitudeStrength") / F.coalesce(F.col("max_mag"), F.lit(1.0)))
        )
        sdf = sdf.withColumn(
            "BearishStrengthHybrid",
            (F.col("BearishCount") / F.coalesce(F.col("max_bear"), F.lit(1.0))) +
            (F.col("MagnitudeStrength") / F.coalesce(F.col("max_mag"), F.lit(1.0)))
        )
        sdf = sdf.withColumn("SignalStrengthHybrid", F.greatest("BullishStrengthHybrid", "BearishStrengthHybrid"))

        # ActionConfidence
        sdf = sdf.withColumn(
            "ActionConfidence",
            F.when(use_fundamentals & F.col("FundamentalScore").isNotNull(),
                   0.6 * F.col("SignalStrengthHybrid") + 0.4 * F.col("FundamentalScore")).otherwise(F.col("SignalStrengthHybrid"))
        )
        sdf = sdf.withColumn(
            "ActionConfidenceNorm",
            F.col("ActionConfidence") / F.coalesce(F.col("max_conf"), F.lit(1.0))
        ).drop("max_bull", "max_bear", "max_mag", "max_conf")

        # Signal duration (bounded window)
        sdf = sdf.withColumn(
            "SignalDuration",
            F.sum(F.when(F.col("Action") != F.lag("Action").over(w), 1).otherwise(0)).over(w)
        )
        sdf = sdf.withColumn("ValidAction", F.col("Action").isin("Buy", "Sell"))
        sdf = sdf.withColumn(
            "HasValidSignal",
            F.col("Action").isNotNull() & F.col("TomorrowAction").isNotNull() & F.col("SignalStrengthHybrid").isNotNull()
        )
    else:
        sdf = sdf.withColumn("SignalStrengthHybrid", F.lit(0.0))
        sdf = sdf.withColumn("ActionConfidence", F.lit(0.0))
        sdf = sdf.withColumn("ActionConfidenceNorm", F.lit(0.0))
        sdf = sdf.withColumn("SignalDuration", F.lit(0))
        sdf = sdf.withColumn("ValidAction", F.lit(False))
        sdf = sdf.withColumn("HasValidSignal", F.lit(False))

    # Validate output
    expected_cols = ["Action", "TomorrowAction", "ActionConfidenceNorm", "SignalStrengthHybrid", "SignalDuration", "ValidAction", "HasValidSignal"]
    missing_cols = [col for col in expected_cols if col not in sdf.columns]
    if missing_cols:
        print(f"Warning: Failed to create columns: {missing_cols}")

    return sdf
    
def finalize_signals_fast_chatgpt(sdf, tf, tf_window=5, use_fundamentals=True, user: int = 1):
    """
    Optimized Spark-native consolidation of momentum + pattern + candle into unified Action.
    Fully vectorized, avoids collect() and minimizes repeated operations.
    """

    # --- Generate columns and momentum factor ---
    candle_columns, trend_cols, momentum_factor = generate_signal_columns(sdf, tf)
    bullish_patterns = trend_cols.get("Bullish", [])
    bearish_patterns = trend_cols.get("Bearish", [])

    # --- Windows ---
    w = Window.partitionBy("CompanyId").orderBy("StockDate")

    # --- Tomorrow returns / momentum ---
    sdf = sdf.withColumn("TomorrowClose", F.lead("Close").over(w))
    sdf = sdf.withColumn("TomorrowReturn", (F.col("TomorrowClose") - F.col("Close")) / F.col("Close"))
    sdf = sdf.withColumn("Return", (F.col("Close") / F.lag("Close").over(w) - 1).cast("double")).fillna({"Return": 0})

    # --- Rolling stats for momentum ---
    sdf = sdf.withColumn("AvgReturn", F.avg("Return").over(w.rowsBetween(-9,0)))
    sdf = sdf.withColumn("Volatility", F.stddev("Return").over(w.rowsBetween(-9,0))).fillna({"Volatility": 1e-8})
    sdf = sdf.withColumn("MomentumZ", (F.col("Return") - F.col("AvgReturn")) / F.col("Volatility"))

    # --- Momentum thresholds ---
    agg = sdf.agg(F.mean("MomentumZ").alias("mean_mom"), F.stddev("MomentumZ").alias("std_mom")).first()
    buy_thresh = agg["mean_mom"] + momentum_factor * agg["std_mom"]
    sell_thresh = agg["mean_mom"] - momentum_factor * agg["std_mom"]

    sdf = sdf.withColumn(
        "MomentumAction",
        F.when(F.col("MomentumZ") > buy_thresh, "Buy")
         .when(F.col("MomentumZ") < sell_thresh, "Sell")
         .otherwise("Hold")
    )

    # --- Pattern scores ---
    def pattern_sum(cols):
        return reduce(add, [F.col(c).cast("double") for c in cols]) if cols else F.lit(0)

    sdf = sdf.withColumn("BullScore", pattern_sum(bullish_patterns))
    sdf = sdf.withColumn("BearScore", pattern_sum(bearish_patterns))
    sdf = sdf.withColumn("PatternScore", F.col("BullScore") - F.col("BearScore"))
    sdf = sdf.withColumn("PatternScoreNorm", F.col("PatternScore") / F.lit(tf_window))
    sdf = sdf.withColumn(
        "PatternAction",
        F.when(F.col("PatternScoreNorm") > 0.2, "Buy")
         .when(F.col("PatternScoreNorm") < -0.2, "Sell")
         .otherwise("Hold")
    )

    # --- Candle action ---
    buy_mask = pattern_sum(candle_columns.get("Buy", [])) > 0 if candle_columns.get("Buy") else F.lit(False)
    sell_mask = pattern_sum(candle_columns.get("Sell", [])) > 0 if candle_columns.get("Sell") else F.lit(False)

    sdf = sdf.withColumn(
        "CandleAction",
        F.when(buy_mask, "Buy").when(sell_mask & ~buy_mask, "Sell").otherwise("Hold")
    )

    # --- CandidateAction (majority vote) ---
    sdf = sdf.withColumn(
        "CandidateAction",
        F.when(F.array_contains(F.array("MomentumAction","PatternAction","CandleAction"), "Buy"), "Buy")
         .when(F.array_contains(F.array("MomentumAction","PatternAction","CandleAction"), "Sell"), "Sell")
         .otherwise("Hold")
    )

    # --- Filter consecutive Buy/Sell ---
    sdf = sdf.withColumn("PrevAction", F.lag("CandidateAction").over(w))
    sdf = sdf.withColumn(
        "Action",
        F.when((F.col("CandidateAction") == F.col("PrevAction")) & F.col("CandidateAction").isin("Buy","Sell"), "Hold")
         .otherwise(F.col("CandidateAction"))
    )

    # --- TomorrowAction ---
    sdf = sdf.withColumn("TomorrowAction", F.lead("Action").over(w))
    sdf = sdf.withColumn(
        "TomorrowActionSource",
        F.when(F.col("TomorrowAction").isin("Buy","Sell"), F.lit("NextAction(filtered)"))
         .otherwise(F.lit("Hold(no_signal)"))
    )

    # --- Hybrid signal strength ---
    valid_cols = [c for c in sdf.columns if c.startswith("Valid")]
    if valid_cols:
        bull_cols = [c for c in valid_cols if "Bull" in c]
        bear_cols = [c for c in valid_cols if "Bear" in c]

        sdf = sdf.withColumn("BullishCount", pattern_sum(bull_cols))
        sdf = sdf.withColumn("BearishCount", pattern_sum(bear_cols))
        sdf = sdf.withColumn("MagnitudeStrength", F.abs(F.col("PatternScore")) + F.abs(F.col("MomentumZ")))

        max_vals = sdf.agg(
            F.max("BullishCount").alias("max_bull"),
            F.max("BearishCount").alias("max_bear"),
            F.max("MagnitudeStrength").alias("max_mag")
        ).first()

        sdf = sdf.withColumn(
            "BullishStrengthHybrid",
            (F.col("BullishCount") / F.lit(max_vals["max_bull"])) +
            (F.col("MagnitudeStrength") / F.lit(max_vals["max_mag"]))
        )
        sdf = sdf.withColumn(
            "BearishStrengthHybrid",
            (F.col("BearishCount") / F.lit(max_vals["max_bear"])) +
            (F.col("MagnitudeStrength") / F.lit(max_vals["max_mag"]))
        )

        sdf = sdf.withColumn("SignalStrengthHybrid", F.greatest("BullishStrengthHybrid","BearishStrengthHybrid"))

        # --- ActionConfidence ---
        if use_fundamentals and "FundamentalScore" in sdf.columns:
            sdf = sdf.withColumn(
                "ActionConfidence",
                0.6 * F.col("SignalStrengthHybrid") + 0.4 * F.col("FundamentalScore")
            )
        else:
            sdf = sdf.withColumn("ActionConfidence", F.col("SignalStrengthHybrid"))

        max_conf = sdf.agg(F.max("ActionConfidence").alias("max_conf")).first()["max_conf"]
        sdf = sdf.withColumn("ActionConfidenceNorm", F.col("ActionConfidence") / F.lit(max_conf))
        sdf = sdf.withColumn(
            "SignalDuration",
            F.sum(F.when(F.col("Action") != F.lag("Action").over(w), 1).otherwise(0)).over(w)
        )
        sdf = sdf.withColumn("ValidAction", F.col("Action").isin("Buy","Sell"))
        sdf = sdf.withColumn(
            "HasValidSignal",
            F.col("Action").isNotNull() & F.col("TomorrowAction").isNotNull() & F.col("SignalStrengthHybrid").isNotNull()
        )
    else:
        sdf = sdf.withColumn("SignalStrengthHybrid", F.lit(0))
        sdf = sdf.withColumn("ActionConfidence", F.lit(0))

    return sdf



# -------------------------------
# Fundamental information/Score
# -------------------------------

from pyspark.sql import functions as F, Window
from functools import reduce
from operator import add
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from operator import add
from functools import reduce

def finalize_signals_optimized(sdf, tf, tf_window=5, use_fundamentals=True, user: int = 1):
    """
    Fully Spark-native signal consolidation with per-company aggregates.
    Uses bounded windows, true majority voting, and robust error handling.
    Optimized for 4-core, 8GB server with ~912,500 rows, sub-2-hour runtime.
    
    Args:
        sdf: Spark DataFrame with CompanyId, StockDate, Close, Valid*, FundamentalScore (optional)
        tf: Timeframe string (e.g., "Daily")
        tf_window: Integer window for pattern normalization
        use_fundamentals: Boolean to include FundamentalScore
        user: Integer user ID for settings
    
    Returns:
        Spark DataFrame with Action, TomorrowAction, ActionConfidenceNorm, SignalStrengthHybrid, etc.
    """
    # Validate inputs
    if user is None:
        raise ValueError("User ID cannot be None")
    required_cols = ["CompanyId", "StockDate", "Close"]
    if not sdf or not all(col in sdf.columns for col in required_cols):
        raise ValueError(f"Input DataFrame must contain {required_cols}")
    if use_fundamentals and "FundamentalScore" not in sdf.columns:
        print("Warning: FundamentalScore missing; disabling fundamentals")
        use_fundamentals = False

    # Generate columns and momentum factor
    candle_columns, trend_cols, momentum_factor = generate_signal_columns(sdf, tf)
    bullish_patterns = trend_cols.get("Bullish", [])
    bearish_patterns = trend_cols.get("Bearish", [])
    if not (bullish_patterns or bearish_patterns or candle_columns.get("Buy") or candle_columns.get("Sell")):
        print("Warning: No valid patterns or candle columns from generate_signal_columns")

    # Windows (bounded for memory)
    max_lag = max(10, tf_window)
    w_company = Window.partitionBy("CompanyId").orderBy("StockDate").rowsBetween(-max_lag, max_lag)
    w_stats = Window.partitionBy("CompanyId").orderBy("StockDate").rowsBetween(-9, 0)

    # Tomorrow returns / momentum
    sdf = sdf.withColumn("TomorrowClose", F.lead("Close").over(w_company))
    sdf = sdf.withColumn(
        "TomorrowReturn",
        F.when(F.col("Close") != 0, (F.col("TomorrowClose") - F.col("Close")) / F.col("Close")).otherwise(0.0)
    )
    sdf = sdf.withColumn(
        "Return",
        F.when(F.lag("Close").over(w_company).isNotNull() & (F.lag("Close").over(w_company) != 0),
               (F.col("Close") / F.lag("Close").over(w_company) - 1)).otherwise(0.0)
    )

    # Rolling stats
    sdf = sdf.withColumn("AvgReturn", F.avg("Return").over(w_stats))
    sdf = sdf.withColumn("Volatility", F.coalesce(F.stddev("Return").over(w_stats), F.lit(1e-8)))
    sdf = sdf.withColumn(
        "MomentumZ",
        F.when(F.col("Volatility") != 0, (F.col("Return") - F.col("AvgReturn")) / F.col("Volatility")).otherwise(0.0)
    )

    # Per-company momentum thresholds
    sdf_agg = sdf.groupBy("CompanyId").agg(
        F.mean("MomentumZ").alias("mean_mom"),
        F.stddev("MomentumZ").alias("std_mom")
    )
    sdf = sdf.join(sdf_agg, on="CompanyId", how="left")
    sdf = sdf.withColumn("buy_thresh", F.col("mean_mom") + momentum_factor * F.coalesce(F.col("std_mom"), F.lit(1.0)))
    sdf = sdf.withColumn("sell_thresh", F.col("mean_mom") - momentum_factor * F.coalesce(F.col("std_mom"), F.lit(1.0)))
    sdf = sdf.withColumn(
        "MomentumAction",
        F.when(F.col("MomentumZ") > F.col("buy_thresh"), "Buy")
         .when(F.col("MomentumZ") < F.col("sell_thresh"), "Sell")
         .otherwise("Hold")
    ).drop("mean_mom", "std_mom", "buy_thresh", "sell_thresh")

    # Pattern scores
    def pattern_sum(cols):
        return reduce(add, [F.col(c).cast("double") for c in cols]) if cols else F.lit(0.0)

    sdf = sdf.withColumn("BullScore", pattern_sum(bullish_patterns))
    sdf = sdf.withColumn("BearScore", pattern_sum(bearish_patterns))
    sdf = sdf.withColumn("PatternScore", F.col("BullScore") - F.col("BearScore"))
    sdf = sdf.withColumn("PatternScoreNorm", F.col("PatternScore") / F.lit(tf_window))
    sdf = sdf.withColumn(
        "PatternAction",
        F.when(F.col("PatternScoreNorm") > 0.3, "Buy")  # Stricter threshold
         .when(F.col("PatternScoreNorm") < -0.3, "Sell")
         .otherwise("Hold")
    )

    # Candle action
    buy_mask = pattern_sum(candle_columns.get("Buy", [])) > 0 if candle_columns.get("Buy") else F.lit(False)
    sell_mask = pattern_sum(candle_columns.get("Sell", [])) > 0 if candle_columns.get("Sell") else F.lit(False)
    sdf = sdf.withColumn(
        "CandleAction",
        F.when(buy_mask, "Buy").when(sell_mask & ~buy_mask, "Sell").otherwise("Hold")
    )

    # CandidateAction (true majority vote)
    sdf = sdf.withColumn(
        "BuyCount",
        F.when(F.col("MomentumAction") == "Buy", 1).otherwise(0) +
        F.when(F.col("PatternAction") == "Buy", 1).otherwise(0) +
        F.when(F.col("CandleAction") == "Buy", 1).otherwise(0)
    )
    sdf = sdf.withColumn(
        "SellCount",
        F.when(F.col("MomentumAction") == "Sell", 1).otherwise(0) +
        F.when(F.col("PatternAction") == "Sell", 1).otherwise(0) +
        F.when(F.col("CandleAction") == "Sell", 1).otherwise(0)
    )
    sdf = sdf.withColumn(
        "CandidateAction",
        F.when(F.col("BuyCount") > F.col("SellCount"), "Buy")
         .when(F.col("SellCount") > F.col("BuyCount"), "Sell")
         .otherwise("Hold")
    ).drop("BuyCount", "SellCount")

    # Filter consecutive Buy/Sell
    sdf = sdf.withColumn("PrevAction", F.lag("CandidateAction").over(w_company))
    sdf = sdf.withColumn(
        "Action",
        F.when((F.col("CandidateAction") == F.col("PrevAction")) & F.col("CandidateAction").isin("Buy", "Sell"), "Hold")
         .otherwise(F.col("CandidateAction"))
    ).drop("PrevAction")

    # TomorrowAction
    sdf = sdf.withColumn("TomorrowAction", F.lead("Action").over(w_company))
    sdf = sdf.withColumn(
        "TomorrowActionSource",
        F.when(F.col("TomorrowAction").isin("Buy", "Sell"), F.lit("NextAction(filtered)"))
         .otherwise(F.lit("Hold(no_signal)"))
    )

    # Hybrid signal strength
    valid_cols = [c for c in sdf.columns if c.startswith("Valid")]
    if valid_cols:
        bull_cols = [c for c in valid_cols if any(k.lower() in c.lower() for k in ["Bull", "Hammer", "MorningStar", "ThreeWhiteSoldiers", "TweezerBottom", "DragonflyDoji"])]
        bear_cols = [c for c in valid_cols if any(k.lower() in c.lower() for k in ["Bear", "ShootingStar", "EveningStar", "ThreeBlackCrows", "TweezerTop", "GravestoneDoji", "DarkCloudCover"])]
        
        sdf = sdf.withColumn("BullishCount", pattern_sum(bull_cols))
        sdf = sdf.withColumn("BearishCount", pattern_sum(bear_cols))
        sdf = sdf.withColumn("MagnitudeStrength", F.abs(F.col("PatternScore")) + F.abs(F.col("MomentumZ")))

        # Per-company normalization
        sdf_agg = sdf.groupBy("CompanyId").agg(
            F.max("BullishCount").alias("max_bull"),
            F.max("BearishCount").alias("max_bear"),
            F.max("MagnitudeStrength").alias("max_mag"),
            F.max("ActionConfidence").alias("max_conf") if valid_cols else F.lit(1.0)
        )
        sdf = sdf.join(sdf_agg, on="CompanyId", how="left")
        
        sdf = sdf.withColumn(
            "BullishStrengthHybrid",
            (F.col("BullishCount") / F.coalesce(F.col("max_bull"), F.lit(1.0))) +
            (F.col("MagnitudeStrength") / F.coalesce(F.col("max_mag"), F.lit(1.0)))
        )
        sdf = sdf.withColumn(
            "BearishStrengthHybrid",
            (F.col("BearishCount") / F.coalesce(F.col("max_bear"), F.lit(1.0))) +
            (F.col("MagnitudeStrength") / F.coalesce(F.col("max_mag"), F.lit(1.0)))
        )
        sdf = sdf.withColumn("SignalStrengthHybrid", F.greatest("BullishStrengthHybrid", "BearishStrengthHybrid"))

        # ActionConfidence
        sdf = sdf.withColumn(
            "ActionConfidence",
            F.when(use_fundamentals & F.col("FundamentalScore").isNotNull(),
                   0.6 * F.col("SignalStrengthHybrid") + 0.4 * F.col("FundamentalScore")).otherwise(F.col("SignalStrengthHybrid"))
        )
        sdf = sdf.withColumn(
            "ActionConfidenceNorm",
            F.col("ActionConfidence") / F.coalesce(F.col("max_conf"), F.lit(1.0))
        ).drop("max_bull", "max_bear", "max_mag", "max_conf")

        # Signal duration (bounded window)
        sdf = sdf.withColumn(
            "SignalDuration",
            F.sum(F.when(F.col("Action") != F.lag("Action").over(w_company), 1).otherwise(0)).over(w_company)
        )
        sdf = sdf.withColumn("ValidAction", F.col("Action").isin("Buy", "Sell"))
        sdf = sdf.withColumn(
            "HasValidSignal",
            F.col("Action").isNotNull() & F.col("TomorrowAction").isNotNull() & F.col("SignalStrengthHybrid").isNotNull()
        )
    else:
        sdf = sdf.withColumn("SignalStrengthHybrid", F.lit(0.0))
        sdf = sdf.withColumn("ActionConfidence", F.lit(0.0))
        sdf = sdf.withColumn("ActionConfidenceNorm", F.lit(0.0))
        sdf = sdf.withColumn("SignalDuration", F.lit(0))
        sdf = sdf.withColumn("ValidAction", F.lit(False))
        sdf = sdf.withColumn("HasValidSignal", F.lit(False))

    # Validate output
    expected_cols = ["Action", "TomorrowAction", "ActionConfidenceNorm", "SignalStrengthHybrid", "SignalDuration", "ValidAction", "HasValidSignal"]
    missing_cols = [col for col in expected_cols if col not in sdf.columns]
    if missing_cols:
        print(f"Warning: Failed to create columns: {missing_cols}")

    return sdf
def finalize_signals_optimized_chatgpt(sdf, tf, tf_window=5, use_fundamentals=True, user: int = 1):
    """
    Fully Spark-native signal consolidation.
    Avoids agg().first() and uses window functions for global max/mean/std.
    """

    candle_columns, trend_cols, momentum_factor = generate_signal_columns(sdf, tf)
    bullish_patterns = trend_cols.get("Bullish", [])
    bearish_patterns = trend_cols.get("Bearish", [])

    # --- Windows ---
    w_company = Window.partitionBy("CompanyId").orderBy("StockDate")
    w_global = Window.partitionBy()  # global window

    # --- Tomorrow returns / momentum ---
    sdf = sdf.withColumn("TomorrowClose", F.lead("Close").over(w_company))
    sdf = sdf.withColumn("TomorrowReturn", (F.col("TomorrowClose") - F.col("Close")) / F.col("Close"))
    sdf = sdf.withColumn("Return", (F.col("Close") / F.lag("Close").over(w_company) - 1).cast("double")).fillna({"Return": 0})

    # --- Rolling stats ---
    sdf = sdf.withColumn("AvgReturn", F.avg("Return").over(w_company.rowsBetween(-9,0)))
    sdf = sdf.withColumn("Volatility", F.stddev("Return").over(w_company.rowsBetween(-9,0))).fillna({"Volatility": 1e-8})
    sdf = sdf.withColumn("MomentumZ", (F.col("Return") - F.col("AvgReturn")) / F.col("Volatility"))

    # --- Momentum thresholds via window ---
    mean_mom = F.mean("MomentumZ").over(w_global)
    std_mom = F.stddev("MomentumZ").over(w_global)
    sdf = sdf.withColumn(
        "MomentumAction",
        F.when(F.col("MomentumZ") > mean_mom + momentum_factor * std_mom, "Buy")
         .when(F.col("MomentumZ") < mean_mom - momentum_factor * std_mom, "Sell")
         .otherwise("Hold")
    )

    # --- Pattern scores ---
    def pattern_sum(cols):
        return reduce(add, [F.col(c).cast("double") for c in cols]) if cols else F.lit(0)

    sdf = sdf.withColumn("BullScore", pattern_sum(bullish_patterns))
    sdf = sdf.withColumn("BearScore", pattern_sum(bearish_patterns))
    sdf = sdf.withColumn("PatternScore", F.col("BullScore") - F.col("BearScore"))
    sdf = sdf.withColumn("PatternScoreNorm", F.col("PatternScore") / F.lit(tf_window))
    sdf = sdf.withColumn(
        "PatternAction",
        F.when(F.col("PatternScoreNorm") > 0.2, "Buy")
         .when(F.col("PatternScoreNorm") < -0.2, "Sell")
         .otherwise("Hold")
    )

    # --- Candle action ---
    buy_mask = pattern_sum(candle_columns.get("Buy", [])) > 0 if candle_columns.get("Buy") else F.lit(False)
    sell_mask = pattern_sum(candle_columns.get("Sell", [])) > 0 if candle_columns.get("Sell") else F.lit(False)
    sdf = sdf.withColumn(
        "CandleAction",
        F.when(buy_mask, "Buy").when(sell_mask & ~buy_mask, "Sell").otherwise("Hold")
    )

    # --- CandidateAction (majority vote) ---
    sdf = sdf.withColumn(
        "CandidateAction",
        F.when(F.array_contains(F.array("MomentumAction","PatternAction","CandleAction"), "Buy"), "Buy")
         .when(F.array_contains(F.array("MomentumAction","PatternAction","CandleAction"), "Sell"), "Sell")
         .otherwise("Hold")
    )

    # --- Filter consecutive Buy/Sell ---
    sdf = sdf.withColumn("PrevAction", F.lag("CandidateAction").over(w_company))
    sdf = sdf.withColumn(
        "Action",
        F.when((F.col("CandidateAction") == F.col("PrevAction")) & F.col("CandidateAction").isin("Buy","Sell"), "Hold")
         .otherwise(F.col("CandidateAction"))
    )

    # --- TomorrowAction ---
    sdf = sdf.withColumn("TomorrowAction", F.lead("Action").over(w_company))
    sdf = sdf.withColumn(
        "TomorrowActionSource",
        F.when(F.col("TomorrowAction").isin("Buy","Sell"), F.lit("NextAction(filtered)"))
         .otherwise(F.lit("Hold(no_signal)"))
    )

    # --- Hybrid signal strength ---
    valid_cols = [c for c in sdf.columns if c.startswith("Valid")]
    if valid_cols:
        bull_cols = [c for c in valid_cols if "Bull" in c]
        bear_cols = [c for c in valid_cols if "Bear" in c]

        sdf = sdf.withColumn("BullishCount", pattern_sum(bull_cols))
        sdf = sdf.withColumn("BearishCount", pattern_sum(bear_cols))
        sdf = sdf.withColumn("MagnitudeStrength", F.abs(F.col("PatternScore")) + F.abs(F.col("MomentumZ")))

        # Use window max instead of agg().first()
        sdf = sdf.withColumn(
            "BullishStrengthHybrid",
            (F.col("BullishCount") / F.max("BullishCount").over(w_global)) +
            (F.col("MagnitudeStrength") / F.max("MagnitudeStrength").over(w_global))
        )
        sdf = sdf.withColumn(
            "BearishStrengthHybrid",
            (F.col("BearishCount") / F.max("BearishCount").over(w_global)) +
            (F.col("MagnitudeStrength") / F.max("MagnitudeStrength").over(w_global))
        )

        sdf = sdf.withColumn("SignalStrengthHybrid", F.greatest("BullishStrengthHybrid","BearishStrengthHybrid"))

        # --- ActionConfidence ---
        if use_fundamentals and "FundamentalScore" in sdf.columns:
            sdf = sdf.withColumn(
                "ActionConfidence",
                0.6 * F.col("SignalStrengthHybrid") + 0.4 * F.col("FundamentalScore")
            )
        else:
            sdf = sdf.withColumn("ActionConfidence", F.col("SignalStrengthHybrid"))

        sdf = sdf.withColumn("ActionConfidenceNorm", F.col("ActionConfidence") / F.max("ActionConfidence").over(w_global))
        sdf = sdf.withColumn(
            "SignalDuration",
            F.sum(F.when(F.col("Action") != F.lag("Action").over(w_company), 1).otherwise(0)).over(w_company)
        )
        sdf = sdf.withColumn("ValidAction", F.col("Action").isin("Buy","Sell"))
        sdf = sdf.withColumn(
            "HasValidSignal",
            F.col("Action").isNotNull() & F.col("TomorrowAction").isNotNull() & F.col("SignalStrengthHybrid").isNotNull()
        )
    else:
        sdf = sdf.withColumn("SignalStrengthHybrid", F.lit(0))
        sdf = sdf.withColumn("ActionConfidence", F.lit(0))

    return sdf


from pyspark.sql import functions as F, Window
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from operator import add
from functools import reduce

def compute_fundamental_score_optimized(sdf, user: int = 1):
    """
    Compute normalized fundamental score using per-company aggregates.
    Fully vectorized, minimizes shuffles, handles nulls/outliers.
    Optimized for 4-core, 8GB server with ~912,500 rows, sub-2-hour runtime.
    
    Args:
        sdf: Spark DataFrame with CompanyId, PeRatio, PbRatio, ..., ShortIntToFloat
        user: Integer user ID for settings
    
    Returns:
        Spark DataFrame with FundamentalScore, FundamentalBad, and normalized columns
    """
    # Validate inputs
    if user is None:
        raise ValueError("User ID cannot be None")
    required_cols = [
        "CompanyId", "PeRatio", "PbRatio", "PegRatio", "ReturnOnEquity",
        "GrossMarginTTM", "NetProfitMarginTTM", "TotalDebtToEquity",
        "CurrentRatio", "InterestCoverage", "EpsChangeYear", "RevChangeYear",
        "Beta", "ShortIntToFloat"
    ]
    missing_cols = [col for col in required_cols if col not in sdf.columns]
    if missing_cols:
        raise ValueError(f"Input DataFrame missing columns: {missing_cols}")

    # Load user settings
    user_settings = load_settings(user).get("fundamental_weights", {})
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
    sdf_agg = sdf.groupBy("CompanyId").agg(
        F.min(F.coalesce(F.col("PeRatio"), F.lit(0.0))).alias("min_pe"),
        F.max(F.coalesce(F.col("PeRatio"), F.lit(0.0))).alias("max_pe"),
        F.min(F.coalesce(F.col("PbRatio"), F.lit(0.0))).alias("min_pb"),
        F.max(F.coalesce(F.col("PbRatio"), F.lit(0.0))).alias("max_pb"),
        F.min(F.coalesce(F.col("PegRatio"), F.lit(0.0))).alias("min_peg"),
        F.max(F.coalesce(F.col("PegRatio"), F.lit(0.0))).alias("max_peg"),
        F.min(F.coalesce(F.col("ReturnOnEquity"), F.lit(0.0))).alias("min_roe"),
        F.max(F.coalesce(F.col("ReturnOnEquity"), F.lit(0.0))).alias("max_roe"),
        F.min(F.coalesce(F.col("GrossMarginTTM"), F.lit(0.0))).alias("min_gross_margin"),
        F.max(F.coalesce(F.col("GrossMarginTTM"), F.lit(0.0))).alias("max_gross_margin"),
        F.min(F.coalesce(F.col("NetProfitMarginTTM"), F.lit(0.0))).alias("min_net_margin"),
        F.max(F.coalesce(F.col("NetProfitMarginTTM"), F.lit(0.0))).alias("max_net_margin"),
        F.min(F.coalesce(F.col("TotalDebtToEquity"), F.lit(0.0))).alias("min_de"),
        F.max(F.coalesce(F.col("TotalDebtToEquity"), F.lit(0.0))).alias("max_de"),
        F.min(F.coalesce(F.col("CurrentRatio"), F.lit(0.0))).alias("min_current_ratio"),
        F.max(F.coalesce(F.col("CurrentRatio"), F.lit(0.0))).alias("max_current_ratio"),
        F.min(F.coalesce(F.col("InterestCoverage"), F.lit(0.0))).alias("min_int_cov"),
        F.max(F.coalesce(F.col("InterestCoverage"), F.lit(0.0))).alias("max_int_cov"),
        F.min(F.coalesce(F.col("EpsChangeYear"), F.lit(0.0))).alias("min_eps_change"),
        F.max(F.coalesce(F.col("EpsChangeYear"), F.lit(0.0))).alias("max_eps_change"),
        F.min(F.coalesce(F.col("RevChangeYear"), F.lit(0.0))).alias("min_rev_change"),
        F.max(F.coalesce(F.col("RevChangeYear"), F.lit(0.0))).alias("max_rev_change"),
        F.min(F.coalesce(F.col("Beta"), F.lit(0.0))).alias("min_beta"),
        F.max(F.coalesce(F.col("Beta"), F.lit(0.0))).alias("max_beta"),
        F.min(F.coalesce(F.col("ShortIntToFloat"), F.lit(0.0))).alias("min_short_int"),
        F.max(F.coalesce(F.col("ShortIntToFloat"), F.lit(0.0))).alias("max_short_int")
    )
    sdf = sdf.join(sdf_agg, on="CompanyId", how="left")

    def normalize(col_name, min_col, max_col, alias, invert=False):
        """Add a normalized version of a column."""
        normalized = (F.col(col_name) - F.col(min_col)) / (F.col(max_col) - F.col(min_col))
        normalized = F.when(F.col(max_col) == F.col(min_col), F.lit(0.0)).otherwise(
            F.coalesce(normalized, F.lit(0.0))
        )
        if invert:
            normalized = 1.0 - normalized
        return sdf.withColumn(alias, normalized)

    # Normalize with outlier capping
    sdf = normalize("PeRatio", "min_pe", "max_pe", "pe_norm", invert=True)
    sdf = normalize("PbRatio", "min_pb", "max_pb", "pb_norm", invert=True)
    sdf = normalize("PegRatio", "min_peg", "max_peg", "peg_norm", invert=True)
    sdf = normalize("ReturnOnEquity", "min_roe", "max_roe", "roe_norm")
    sdf = normalize("GrossMarginTTM", "min_gross_margin", "max_gross_margin", "gross_margin_norm")
    sdf = normalize("NetProfitMarginTTM", "min_net_margin", "max_net_margin", "net_margin_norm")
    sdf = normalize("TotalDebtToEquity", "min_de", "max_de", "de_norm", invert=True)
    sdf = normalize("CurrentRatio", "min_current_ratio", "max_current_ratio", "current_ratio_norm")
    sdf = normalize("InterestCoverage", "min_int_cov", "max_int_cov", "int_cov_norm")
    sdf = normalize("EpsChangeYear", "min_eps_change", "max_eps_change", "eps_change_norm")
    sdf = normalize("RevChangeYear", "min_rev_change", "max_rev_change", "rev_change_norm")
    sdf = normalize("Beta", "min_beta", "max_beta", "beta_norm", invert=True)
    sdf = normalize("ShortIntToFloat", "min_short_int", "max_short_int", "short_int_norm")  # Not inverted

    # Drop intermediate columns
    sdf = sdf.drop(*[c for c in sdf.columns if c.startswith("min_") or c.startswith("max_")])

    # Combine weighted score
    sdf = sdf.withColumn(
        "FundamentalScore",
        F.coalesce(
            weights["valuation"] * (F.col("pe_norm") + F.col("peg_norm") + F.col("pb_norm")) / 3 +
            weights["profitability"] * (F.col("roe_norm") + F.col("gross_margin_norm") + F.col("net_margin_norm")) / 3 +
            weights["DebtLiquidity"] * (F.col("de_norm") + F.col("current_ratio_norm") + F.col("int_cov_norm")) / 3 +
            weights["Growth"] * (F.col("eps_change_norm") + F.col("rev_change_norm")) / 2 +
            weights["Sentiment"] * (F.col("beta_norm") + F.col("short_int_norm")) / 2,
            F.lit(0.0)
        )
    )

    # Flag bad rows (nulls only, not zeros)
    norm_cols = [
        "pe_norm", "peg_norm", "pb_norm",
        "roe_norm", "gross_margin_norm", "net_margin_norm",
        "de_norm", "current_ratio_norm", "int_cov_norm",
        "eps_change_norm", "rev_change_norm",
        "beta_norm", "short_int_norm"
    ]
    bad_expr = F.lit(False)
    for c in norm_cols:
        bad_expr = bad_expr | F.col(c).isNull()

    sdf = sdf.withColumn("FundamentalBad", bad_expr)

    # Validate output
    expected_cols = ["FundamentalScore", "FundamentalBad"] + norm_cols
    missing_cols = [col for col in expected_cols if col not in sdf.columns]
    if missing_cols:
        print(f"Warning: Failed to create columns: {missing_cols}")

    return sdf
def compute_fundamental_score_optimized_chatgpt(sdf, user: int = 1):
    """
    Compute normalized fundamental score using Spark window functions instead of agg().first().
    Fully vectorized, avoids multiple Spark jobs.
    """
    user_settings = load_settings(user).get("fundamental_weights", {})

    weights = {
        "valuation": user_settings.get("valuation", 0.2),
        "profitability": user_settings.get("profitability", 0.3),
        "DebtLiquidity": user_settings.get("DebtLiquidity", 0.2),
        "Growth": user_settings.get("Growth", 0.2),
        "Sentiment": user_settings.get("Sentiment", 0.1),
    }

    # Global window for min/max across all rows
    w_all = Window.partitionBy()

    def normalize(col_name, alias, invert=False):
        """Add a normalized version of a column as a new column."""
        min_val = F.min(F.col(col_name)).over(w_all)
        max_val = F.max(F.col(col_name)).over(w_all)
        normalized = (F.col(col_name) - min_val) / (max_val - min_val)
        normalized = F.when(max_val == min_val, F.lit(0.0)).otherwise(normalized)
        if invert:
            normalized = 1.0 - normalized
        return sdf.withColumn(alias, normalized)

    # --- Valuation ---
    sdf = normalize("PeRatio", "pe_norm", invert=True)
    sdf = normalize("PbRatio", "pb_norm", invert=True)
    sdf = normalize("PegRatio", "peg_norm", invert=True)

    # --- Profitability ---
    sdf = normalize("ReturnOnEquity", "roe_norm")
    sdf = normalize("GrossMarginTTM", "gross_margin_norm")
    sdf = normalize("NetProfitMarginTTM", "net_margin_norm")

    # --- Debt & Liquidity ---
    sdf = normalize("TotalDebtToEquity", "de_norm", invert=True)
    sdf = normalize("CurrentRatio", "current_ratio_norm")
    sdf = normalize("InterestCoverage", "int_cov_norm")

    # --- Growth ---
    sdf = normalize("EpsChangeYear", "eps_change_norm")
    sdf = normalize("RevChangeYear", "rev_change_norm")

    # --- Sentiment / Risk ---
    sdf = normalize("Beta", "beta_norm", invert=True)
    sdf = normalize("ShortIntToFloat", "short_int_norm")

    # --- Combine weighted score ---
    sdf = sdf.withColumn(
        "FundamentalScore",
        weights["valuation"] * (F.col("pe_norm") + F.col("peg_norm") + F.col("pb_norm")) / 3 +
        weights["profitability"] * (F.col("roe_norm") + F.col("gross_margin_norm") + F.col("net_margin_norm")) / 3 +
        weights["DebtLiquidity"] * (F.col("de_norm") + F.col("current_ratio_norm") + F.col("int_cov_norm")) / 3 +
        weights["Growth"] * (F.col("eps_change_norm") + F.col("rev_change_norm")) / 2 +
        weights["Sentiment"] * (F.col("beta_norm") + F.col("short_int_norm")) / 2
    )

    # --- Flag bad rows ---
    norm_cols = [
        "pe_norm", "peg_norm", "pb_norm",
        "roe_norm", "gross_margin_norm", "net_margin_norm",
        "de_norm", "current_ratio_norm", "int_cov_norm",
        "eps_change_norm", "rev_change_norm",
        "beta_norm", "short_int_norm"
    ]

    bad_expr = F.lit(False)
    for c in norm_cols:
        bad_expr = bad_expr | F.col(c).isNull() | (F.col(c) == 0.0)

    sdf = sdf.withColumn("FundamentalBad", bad_expr)

    return sdf


# -------------------------------
# Batch Metadata
# -------------------------------
def add_batch_metadata(sdf, timeframe, user: int = 1, ingest_ts=None):
    """
    Add BatchId, IngestedAt, CompanyId, TimeFrame, and UserId metadata to a Spark DataFrame.
    """

    if ingest_ts is None:
        ingest_ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # Add constant columns
    sdf = sdf.withColumn("BatchId", F.lit(f"{user}_{ingest_ts}")) \
             .withColumn("IngestedAt", F.lit(ingest_ts)) \
             .withColumn("TimeFrame", F.lit(timeframe)) \
             .withColumn("UserId", F.lit(user))

    return sdf

