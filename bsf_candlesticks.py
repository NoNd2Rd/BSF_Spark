import pandas as pd
import numpy as np
from datetime import datetime
import re
from bsf_settings import load_settings
from operator import itemgetter
import unicodedata
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql import functions as F
from pyspark.sql import Window
from operator import itemgetter
from pyspark.sql import functions as F
from pyspark.sql import Window


# -------------------------------
# Add Signal Columns
# -------------------------------
def generate_signal_columns(sdf, timeframe="Short", user: int = None):
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


# -------------------------------
# Add Candlestick Patterns
# -------------------------------
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
    
   
from pyspark.sql import functions as F, Window
from operator import itemgetter

def add_candle_patterns_fast(sdf, tf_window=5, user: int = 1):
    """
    Spark-native candlestick pattern detection for multiple companies.
    Expects sdf with columns: CompanyId, Open, High, Low, Close, Volume, StockDate
    Returns sdf with new boolean columns for each pattern and PatternCount/PatternType.
    Fully Spark vectorized — no .collect() calls.
    """

    o, h, l, c, v = "Open", "High", "Low", "Close", "Volume"

    # --- Windows ---
    w_tf = Window.orderBy("StockDate").rowsBetween(-(tf_window-1), 0)
    w_shift1 = Window.orderBy("StockDate")
    w_shift2 = Window.orderBy("StockDate")

    # --- Last Close per company ---
    last_close = sdf.agg(F.last("Close").alias("last_close")).first()["last_close"]
    candle_params = get_candle_params(last_close)

    doji_thresh, hammer_thresh, marubozu_thresh, long_body, small_body, shadow_ratio, near_edge, highvol_spike, lowvol_dip, rng_thresh = \
        itemgetter(
            "doji_thresh", "hammer_thresh", "marubozu_thresh", "long_body", "small_body", 
            "shadow_ratio", "near_edge", "highvol_spike", "lowvol_dip", "rng_thresh"
        )(candle_params)

    # --- Rolling calculations ---
    sdf = sdf.withColumn("O_roll", F.first(o).over(w_tf)) \
             .withColumn("C_roll", F.last(c).over(w_tf)) \
             .withColumn("H_roll", F.max(h).over(w_tf)) \
             .withColumn("L_roll", F.min(l).over(w_tf)) \
             .withColumn("V_avg20", F.avg(v).over(Window.partitionBy("CompanyId").orderBy("StockDate").rowsBetween(-19,0)))

    # --- Volume spikes ---
    sdf = sdf.withColumn("HighVolume", F.col(v) > highvol_spike * F.col("V_avg20")) \
             .withColumn("LowVolume", F.col(v) < lowvol_dip * F.col("V_avg20"))

    # --- Body, shadows, range ---
    sdf = sdf.withColumn("Body", F.abs(F.col("C_roll") - F.col("O_roll")) / F.col("C_roll")) \
             .withColumn("UpShadow", (F.col("H_roll") - F.greatest(F.col("O_roll"), F.col("C_roll"))) / F.col("C_roll")) \
             .withColumn("DownShadow", (F.least(F.col("O_roll"), F.col("C_roll")) - F.col("L_roll")) / F.col("C_roll")) \
             .withColumn("Range", (F.col("H_roll") - F.col("L_roll")) / F.col("C_roll")) \
             .withColumn("Bull", F.col("C_roll") > F.col("O_roll")) \
             .withColumn("Bear", F.col("O_roll") > F.col("C_roll"))

    # --- Trend detection ---
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
            sdf = sdf.withColumn(f"{col_name}{lag}", F.lag(f"{col_name}_roll" if col_name in ["O","C","H","L"] else col_name, lag).over(w_shift1))

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
                                           (F.col("O_roll") > F.col("C1")) & (F.col("C_roll") < (F.col("O1") + F.col("C1"))/2) & (F.col("C_roll") > F.col("O1"))) \
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

    # --- PatternCount ---
    pattern_cols = [
        "Doji","Hammer","InvertedHammer","BullishMarubozu","BearishMarubozu","SuspiciousCandle",
        "HangingMan","ShootingStar","SpinningTop","BullishEngulfing","BearishEngulfing",
        "BullishHarami","BearishHarami","HaramiCross","PiercingLine","DarkCloudCover",
        "MorningStar","EveningStar","ThreeWhiteSoldiers","ThreeBlackCrows","TweezerTop",
        "TweezerBottom","InsideBar","OutsideBar","NearHigh","NearLow","DragonflyDoji",
        "GravestoneDoji","LongLeggedDoji","RisingThreeMethods","FallingThreeMethods",
        "GapUp","GapDown","ClimacticCandle"
    ]
    sdf = sdf.withColumn("PatternCount", sum(F.col(c).cast("int") for c in pattern_cols))

    # --- PatternType (first match) using array ---
    sdf = sdf.withColumn("PatternType", F.array(*[
        F.when(F.col(c), c) for c in pattern_cols
    ]))
    sdf = sdf.withColumn("PatternType", F.expr("filter(PatternType, x -> x is not null)[0]"))
    sdf = sdf.withColumn("PatternType", F.coalesce(F.col("PatternType"), F.lit("None")))

    return sdf

# -------------------------------
# Add Trends
# -------------------------------
def add_trend_filters_fast(sdf, timeframe="Daily", user: int = 1):
    """
    Spark version of add_trend_filters.
    Adds moving averages, slopes, returns, volatility, ROC, and trend flags.
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

    # ---------------------------------------
    # Define window specs
    # ---------------------------------------
    w_ma = Window.partitionBy("CompanyId").orderBy("StockDate").rowsBetween(-(ma_window-1), 0)
    w_ret = Window.partitionBy("CompanyId").orderBy("StockDate").rowsBetween(-ret_window, 0)
    w_slope = Window.partitionBy("CompanyId").orderBy("StockDate").rowsBetween(-slope_horizon, 0)
    w_vol = Window.partitionBy("CompanyId").orderBy("StockDate").rowsBetween(-(vol_window-1), 0)
    
    # ---------------------------------------
    # Moving Average and slope
    # ---------------------------------------
    sdf = sdf.withColumn("MA", F.avg(F.col(c)).over(w_ma))
    sdf = sdf.withColumn("MA_slope", (F.col("MA") - F.lag("MA", slope_horizon).over(Window.partitionBy("CompanyId").orderBy("StockDate"))) / F.lag("MA", slope_horizon).over(Window.partitionBy("CompanyId").orderBy("StockDate")))
    sdf = sdf.withColumn("UpTrend_MA", F.col("MA_slope") > 0)
    sdf = sdf.withColumn("DownTrend_MA", F.col("MA_slope") < 0)
    
    # ---------------------------------------
    # Returns and Volatility
    # ---------------------------------------
    sdf = sdf.withColumn("RecentReturn", (F.col(c) - F.lag(c, ret_window).over(Window.partitionBy("CompanyId").orderBy("StockDate"))) / F.lag(c, ret_window).over(Window.partitionBy("CompanyId").orderBy("StockDate")))
    sdf = sdf.withColumn("UpTrend_Return", F.col("RecentReturn") > 0)
    sdf = sdf.withColumn("DownTrend_Return", F.col("RecentReturn") < 0)
    
    # Volatility = rolling std of pct changes
    sdf = sdf.withColumn("ReturnPct", (F.col(c) - F.lag(c).over(Window.partitionBy("CompanyId").orderBy("StockDate"))) / F.lag(c).over(Window.partitionBy("CompanyId").orderBy("StockDate")))
    sdf = sdf.withColumn("Volatility", F.stddev("ReturnPct").over(w_vol))
    
    # Median volatility for dynamic threshold
    vol_med = sdf.approxQuantile("Volatility", [0.5], 0.0)[0]
    sdf = sdf.withColumn("LowVolatility", F.col("Volatility") < vol_med)
    sdf = sdf.withColumn("HighVolatility", F.col("Volatility") > vol_med)
    
    # ---------------------------------------
    # Rate of Change (ROC)
    # ---------------------------------------
    sdf = sdf.withColumn("ROC", (F.col(c) - F.lag(c, ma_window).over(Window.partitionBy("CompanyId").orderBy("StockDate"))) / F.lag(c, ma_window).over(Window.partitionBy("CompanyId").orderBy("StockDate")))
    sdf = sdf.withColumn("MomentumUp", F.col("ROC") > roc_thresh)
    sdf = sdf.withColumn("MomentumDown", F.col("ROC") < -roc_thresh)
    
    # ---------------------------------------
    # Confirmed trends
    # ---------------------------------------
    sdf = sdf.withColumn("ConfirmedUpTrend", F.col("UpTrend_MA") & F.col("UpTrend_Return") & F.col("MomentumUp"))
    sdf = sdf.withColumn("ConfirmedDownTrend", F.col("DownTrend_MA") & F.col("DownTrend_Return") & F.col("MomentumDown"))
    
    # Drop helper column
    sdf = sdf.drop("ReturnPct")
    
    return sdf

# -------------------------------
# Confirmed Signals
# -------------------------------
def add_confirmed_signals(sdf):
    """
    Generate validated candlestick signals based on trend context in Spark.
    E.g., a bullish engulfing is only 'valid' if prior trend was bearish.
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

    for group_name, patterns in signal_groups.items():
        for valid_col, trend_col in patterns.items():
            raw_col = valid_col.replace("Valid", "")
            # Ensure columns exist
            if raw_col not in sdf.columns:
                sdf = sdf.withColumn(raw_col, F.lit(False))
            if trend_col not in sdf.columns:
                sdf = sdf.withColumn(trend_col, F.lit(False))
            sdf = sdf.withColumn(valid_col, F.col(raw_col) & F.col(trend_col))

    # Bullish reversal: DragonflyDoji & DownTrend_MA & HighVolume
    for col in ["DragonflyDoji", "DownTrend_MA", "HighVolume"]:
        if col not in sdf.columns:
            sdf = sdf.withColumn(col, F.lit(False))
    sdf = sdf.withColumn("ValidDragonflyDoji", F.col("DragonflyDoji") & F.col("DownTrend_MA") & F.col("HighVolume"))

    # Bearish reversal: GravestoneDoji & UpTrend_MA & HighVolume
    for col in ["GravestoneDoji", "UpTrend_MA", "HighVolume"]:
        if col not in sdf.columns:
            sdf = sdf.withColumn(col, F.lit(False))
    sdf = sdf.withColumn("ValidGravestoneDoji", F.col("GravestoneDoji") & F.col("UpTrend_MA") & F.col("HighVolume"))

    return sdf


# -------------------------------
# Signal Strength (count-based)
# -------------------------------
from pyspark.sql import functions as F

def add_signal_strength_fast(sdf, directional_groups=None):
    """
    Spark-native signal strength counts and percentages.
    Vectorized and optimized.
    """

    # 1️⃣ Identify all valid signal columns
    valid_cols = [c for c in sdf.columns if c.startswith("Valid")]
    if not valid_cols:
        return sdf.withColumns({
            "SignalStrength": F.lit(0),
            "BullishPctRaw": F.lit(0.0),
            "BearishPctRaw": F.lit(0.0),
            "BullishPctDirectional": F.lit(0.0),
            "BearishPctDirectional": F.lit(0.0)
        })

    # Cast all valid columns to integer
    sdf = sdf.select(
        "*",
        *[F.col(c).cast("int").alias(c) for c in valid_cols]
    )

    # SignalStrength total
    signal_strength_expr = sum(F.col(c) for c in valid_cols)
    sdf = sdf.withColumn("SignalStrengthNonZero", F.when(signal_strength_expr == 0, 1).otherwise(signal_strength_expr))
    sdf = sdf.withColumn("SignalStrength", signal_strength_expr)

    # Bullish / Bearish raw percentages
    bullish_cols = [c for c in valid_cols if c.startswith("ValidBullish")]
    bearish_cols = [c for c in valid_cols if c.startswith("ValidBearish")]

    bullish_sum_expr = sum(F.col(c) for c in bullish_cols) if bullish_cols else F.lit(0)
    bearish_sum_expr = sum(F.col(c) for c in bearish_cols) if bearish_cols else F.lit(0)

    sdf = sdf.withColumns({
        "BullishPctRaw": bullish_sum_expr / F.col("SignalStrengthNonZero"),
        "BearishPctRaw": bearish_sum_expr / F.col("SignalStrengthNonZero")
    })

    # Directional percentages
    if directional_groups is None:
        directional_groups = ["Bullish", "Bearish"]

    directional_cols = [c for c in valid_cols if any(c.startswith(f"Valid{g}") for g in directional_groups)]
    directional_sum_expr = sum(F.col(c) for c in directional_cols) if directional_cols else F.lit(0)
    sdf = sdf.withColumn("DirectionalSumNonZero", F.when(directional_sum_expr == 0, 1).otherwise(directional_sum_expr))

    bullish_dir_cols = [c for c in directional_cols if c.startswith("ValidBullish")]
    bearish_dir_cols = [c for c in directional_cols if c.startswith("ValidBearish")]

    bullish_dir_sum = sum(F.col(c) for c in bullish_dir_cols) if bullish_dir_cols else F.lit(0)
    bearish_dir_sum = sum(F.col(c) for c in bearish_dir_cols) if bearish_dir_cols else F.lit(0)

    sdf = sdf.withColumns({
        "BullishPctDirectional": bullish_dir_sum / F.col("DirectionalSumNonZero"),
        "BearishPctDirectional": bearish_dir_sum / F.col("DirectionalSumNonZero")
    })

    # Drop helper columns
    sdf = sdf.drop("SignalStrengthNonZero", "DirectionalSumNonZero")

    return sdf


# -------------------------------
# Finalize Signals
# -------------------------------
from pyspark.sql import functions as F
from pyspark.sql import Window

def finalize_signals_fast(sdf, tf, tf_window=5, use_fundamentals=True, user: int = 1):
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
    w_lag1 = w.rowsBetween(-1, -1)

    # --- Tomorrow returns / momentum ---
    sdf = sdf.withColumn("TomorrowClose", F.lead("Close").over(w))
    sdf = sdf.withColumn("TomorrowReturn", (F.col("TomorrowClose") - F.col("Close")) / F.col("Close"))

    sdf = sdf.withColumn("Return", (F.col("Close") / F.lag("Close").over(w) - 1).cast("double")).fillna({"Return": 0})

    # Rolling stats for momentum
    sdf = sdf.withColumn("AvgReturn", F.avg("Return").over(w.rowsBetween(-9,0)))
    sdf = sdf.withColumn("Volatility", F.stddev("Return").over(w.rowsBetween(-9,0))).fillna({"Volatility": 1e-8})
    sdf = sdf.withColumn("MomentumZ", (F.col("Return") - F.col("AvgReturn")) / F.col("Volatility"))

    # Momentum thresholds
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
        return sum(F.col(c).cast("double") for c in cols) if cols else F.lit(0)

    sdf = sdf.withColumns({
        "BullScore": pattern_sum(bullish_patterns),
        "BearScore": pattern_sum(bearish_patterns)
    })
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

        sdf = sdf.withColumns({
            "BullishCount": pattern_sum(bull_cols),
            "BearishCount": pattern_sum(bear_cols),
            "MagnitudeStrength": F.abs(F.col("PatternScore")) + F.abs(F.col("MomentumZ"))
        })

        max_vals = sdf.agg(
            F.max("BullishCount").alias("max_bull"),
            F.max("BearishCount").alias("max_bear"),
            F.max("MagnitudeStrength").alias("max_mag")
        ).first()

        sdf = sdf.withColumns({
            "BullishStrengthHybrid": (F.col("BullishCount") / F.lit(max_vals["max_bull"])) +
                                     (F.col("MagnitudeStrength") / F.lit(max_vals["max_mag"])),
            "BearishStrengthHybrid": (F.col("BearishCount") / F.lit(max_vals["max_bear"])) +
                                     (F.col("MagnitudeStrength") / F.lit(max_vals["max_mag"]))
        })

        sdf = sdf.withColumn("SignalStrengthHybrid", F.greatest("BullishStrengthHybrid","BearishStrengthHybrid"))

        # --- ActionConfidence ---
        if use_fundamentals and "FundamentalScore" in sdf.columns:
            sdf = sdf.withColumn("ActionConfidence",
                                 0.6 * F.col("SignalStrengthHybrid") + 0.4 * F.col("FundamentalScore"))
        else:
            sdf = sdf.withColumn("ActionConfidence", F.col("SignalStrengthHybrid"))

        max_conf = sdf.agg(F.max("ActionConfidence").alias("max_conf")).first()["max_conf"]
        sdf = sdf.withColumns({
            "ActionConfidenceNorm": F.col("ActionConfidence") / F.lit(max_conf),
            "SignalDuration": F.sum(F.when(F.col("Action") != F.lag("Action").over(w), 1).otherwise(0)).over(w),
            "ValidAction": F.col("Action").isin("Buy","Sell"),
            "HasValidSignal": F.col("Action").isNotNull() & F.col("TomorrowAction").isNotNull() & F.col("SignalStrengthHybrid").isNotNull()
        })
    else:
        sdf = sdf.withColumn("SignalStrengthHybrid", F.lit(0)).withColumn("ActionConfidence", F.lit(0))

    return sdf


# -------------------------------
# Fundamental information/Score
# -------------------------------
def normalize_scalar(sdf, col_name, invert=False):
    """Compute 0-1 normalized column using scalar min/max."""
    row = sdf.agg(
        F.min(col_name).alias("min_val"),
        F.max(col_name).alias("max_val")
    ).first()
    min_val, max_val = row["min_val"], row["max_val"]

    if min_val is None or max_val is None or min_val == max_val:
        normalized_col = F.lit(0.0)
    else:
        normalized_col = (F.col(col_name) - F.lit(min_val)) / (F.lit(max_val) - F.lit(min_val))
        if invert:
            normalized_col = 1.0 - normalized_col
    return normalized_col
    
def compute_fundamental_score(sdf, user: int = 1):
    """
    Compute a normalized fundamental score for a dataframe of stocks.
    Handles missing or constant columns safely.
    """
    # Load user-specific fundamental weights (merged with defaults)
    user_settings = load_settings(user).get("fundamental_weights", {})
    
    # Fallback to defaults if a key is missing
    weights = {
        "valuation": user_settings.get("valuation", 0.2),
        "profitability": user_settings.get("profitability", 0.3),
        "DebtLiquidity": user_settings.get("DebtLiquidity", 0.2),
        "Growth": user_settings.get("Growth", 0.2),
        "Sentiment": user_settings.get("Sentiment", 0.1),
    }
    
    # --- Valuation (invert ratios) ---
    pe = normalize_scalar(sdf, "PeRatio", invert=True)
    peg = normalize_scalar(sdf, "PegRatio", invert=True)
    pb = normalize_scalar(sdf, "PbRatio", invert=True)

    # --- Profitability ---
    roe = normalize_scalar(sdf, "ReturnOnEquity")
    gross_margin = normalize_scalar(sdf, "GrossMarginTTM")
    net_margin = normalize_scalar(sdf, "NetProfitMarginTTM")

    # --- Debt & Liquidity ---
    de = normalize_scalar(sdf, "TotalDebtToEquity", invert=True)
    current_rat = normalize_scalar(sdf, "CurrentRatio")
    int_cov = normalize_scalar(sdf, "InterestCoverage")

    # --- Growth ---
    eps_change = normalize_scalar(sdf, "EpsChangeYear")
    rev_change = normalize_scalar(sdf, "RevChangeYear")

    # --- Sentiment / Risk ---
    beta = normalize_scalar(sdf, "Beta", invert=True)
    short_int = normalize_scalar(sdf, "ShortIntToFloat")

    # --- Combine weighted score ---
    '''
    FundamentalScore =
    0.2 * (pe + peg + pb)/3        # Valuation
    + 0.3 * (roe + gross_margin + net_margin)/3  # Profitability
    + 0.2 * (de + current_rat + int_cov)/3       # Debt & Liquidity
    + 0.2 * (eps_change + rev_change)/2          # Growth
    + 0.1 * (beta + short_int)/2                # Sentiment / Risk
    '''
    sdf = sdf.withColumn(
        "FundamentalScore",
        weights["valuation"] * (pe + peg + pb) / 3 +
        weights["profitability"] * (roe + gross_margin + net_margin) / 3 +
        weights["DebtLiquidity"] * (de + current_rat + int_cov) / 3 +
        weights["Growth"] * (eps_change + rev_change) / 2 +
        weights["Sentiment"] * (beta + short_int) / 2
    )

    
    # --- Flag bad rows (any component is zero or null) ---
    components = [pe, peg, pb, roe, gross_margin, net_margin,
                  de, current_rat, int_cov, eps_change, rev_change,
                  beta, short_int]
    
    # Build boolean expression
    bad_expr = F.lit(False)
    for comp in components:
        bad_expr = bad_expr | comp.isNull() | (comp == 0.0)
    
    sdf = sdf.withColumn("FundamentalBad", bad_expr)
    
    # optionally drop min_val/max_val afterward
    sdf = sdf.drop("min_val", "max_val")
    
    return sdf
def finalize_signals_optimized(sdf, tf, tf_window=5, use_fundamentals=True, user: int = 1):
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

    # Rolling stats
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
        return sum(F.col(c).cast("double") for c in cols) if cols else F.lit(0)
    sdf = sdf.withColumns({
        "BullScore": pattern_sum(bullish_patterns),
        "BearScore": pattern_sum(bearish_patterns)
    })
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
        sdf = sdf.withColumns({
            "BullishCount": pattern_sum(bull_cols),
            "BearishCount": pattern_sum(bear_cols),
            "MagnitudeStrength": F.abs(F.col("PatternScore")) + F.abs(F.col("MomentumZ"))
        })

        # Use window max instead of agg().first()
        sdf = sdf.withColumns({
            "BullishStrengthHybrid": (F.col("BullishCount") / F.max("BullishCount").over(w_global)) +
                                     (F.col("MagnitudeStrength") / F.max("MagnitudeStrength").over(w_global)),
            "BearishStrengthHybrid": (F.col("BearishCount") / F.max("BearishCount").over(w_global)) +
                                     (F.col("MagnitudeStrength") / F.max("MagnitudeStrength").over(w_global))
        })
        sdf = sdf.withColumn("SignalStrengthHybrid", F.greatest("BullishStrengthHybrid","BearishStrengthHybrid"))

        if use_fundamentals and "FundamentalScore" in sdf.columns:
            sdf = sdf.withColumn("ActionConfidence",
                                 0.6 * F.col("SignalStrengthHybrid") + 0.4 * F.col("FundamentalScore"))
        else:
            sdf = sdf.withColumn("ActionConfidence", F.col("SignalStrengthHybrid"))

        sdf = sdf.withColumn("ActionConfidenceNorm", F.col("ActionConfidence") / F.max("ActionConfidence").over(w_global))
        sdf = sdf.withColumn("SignalDuration", F.sum(F.when(F.col("Action") != F.lag("Action").over(w_company), 1).otherwise(0)).over(w_company))
        sdf = sdf.withColumn("ValidAction", F.col("Action").isin("Buy","Sell"))
        sdf = sdf.withColumn("HasValidSignal", F.col("Action").isNotNull() & F.col("TomorrowAction").isNotNull() & F.col("SignalStrengthHybrid").isNotNull())
    else:
        sdf = sdf.withColumn("SignalStrengthHybrid", F.lit(0)).withColumn("ActionConfidence", F.lit(0))

    return sdf
 
def compute_fundamental_score_optimized(sdf, user: int = 1):
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

    def normalize(col_name, invert=False):
        min_val = F.min(F.col(col_name)).over(w_all)
        max_val = F.max(F.col(col_name)).over(w_all)
        normalized = (F.col(col_name) - min_val) / (max_val - min_val)
        normalized = F.when(max_val == min_val, F.lit(0.0)).otherwise(normalized)
        if invert:
            normalized = 1.0 - normalized
        return normalized

    # --- Valuation ---
    pe = normalize("PeRatio", invert=True)
    pb = normalize("PbRatio", invert=True)
    peg = normalize("PegRatio", invert=True)

    # --- Profitability ---
    roe = normalize("ReturnOnEquity")
    gross_margin = normalize("GrossMarginTTM")
    net_margin = normalize("NetProfitMarginTTM")

    # --- Debt & Liquidity ---
    de = normalize("TotalDebtToEquity", invert=True)
    current_rat = normalize("CurrentRatio")
    int_cov = normalize("InterestCoverage")

    # --- Growth ---
    eps_change = normalize("EpsChangeYear")
    rev_change = normalize("RevChangeYear")

    # --- Sentiment / Risk ---
    beta = normalize("Beta", invert=True)
    short_int = normalize("ShortIntToFloat")

    # --- Combine weighted score ---
    sdf = sdf.withColumn(
        "FundamentalScore",
        weights["valuation"] * (pe + peg + pb) / 3 +
        weights["profitability"] * (roe + gross_margin + net_margin) / 3 +
        weights["DebtLiquidity"] * (de + current_rat + int_cov) / 3 +
        weights["Growth"] * (eps_change + rev_change) / 2 +
        weights["Sentiment"] * (beta + short_int) / 2
    )

    # --- Flag bad rows ---
    components = [pe, peg, pb, roe, gross_margin, net_margin,
                  de, current_rat, int_cov, eps_change, rev_change,
                  beta, short_int]

    bad_expr = F.lit(False)
    for comp in components:
        bad_expr = bad_expr | comp.isNull() | (comp == 0.0)
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

