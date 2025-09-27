# ─── Core CLI and Time Utilities ─────────────────────────────
import argparse
from datetime import datetime, timedelta

# ─── Spark Setup and Utilities ───────────────────────────────
from bsf_env import (
    init_spark,
    init_mariadb_engine,
    set_spark_verbosity
)

# ─── PySpark Functions and Types ─────────────────────────────
from pyspark.sql import functions as F
from pyspark.sql.functions import lit, current_timestamp, broadcast
from pyspark.sql.types import *

# ─── Data Science Stack ──────────────────────────────────────
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

spark = None
engine = None
ingest_ts = None

def prepare_lakehouse_environment(mode: str= None, db_name: str = "bsf"):
    # ─── Setup Environment ──────────────────────────────────────
    global spark, engine, ingest_ts
    spark = init_spark(f"{db_name}_lakehouse_{mode}", log_level="ERROR", show_progress=False, enable_ui=True)
    engine = init_mariadb_engine()
    ingest_ts = spark.sql("SELECT current_timestamp()").collect()[0][0]
    print(f"         ⚡️ Spark initialized >> {db_name}_lakehouse_{mode} <<")
    # ─── Setup Environment ──────────────────────────────────────
    
    if mode == "full":
        #print("         🔍 Inspecting Spark environment before dropping DB...")
        #db_stats(db_name)
        delete_hive_db(db_name)
        print(f"         📁 Ensuring Hive database '{db_name}' exists...")
        spark.sql(f"CREATE DATABASE IF NOT EXISTS {db_name}")


def run_with_logging(func, icon="⏳",  is_subtask=False, title=None, *args, **kwargs):
    bold_title = f"\033[1m**{title}**\033[0m" if title else ''

    if is_subtask:
        # Indented subtask, no extra blank line
        print(f"   ↳ {icon} {func.__name__} is running {bold_title}...")
    else:
        # Main task with leading blank line
        print(f"\n{icon} {func.__name__} is running {bold_title}...")

    # Capture start time
    start_time = datetime.now()
    
    result = func(*args, **kwargs)

    # Capture end time
    end_time = datetime.now()
    total_runtime = end_time - start_time
    if is_subtask:
        # Indented subtask, no extra blank line
        print(f"   ⏱️ {func.__name__} completed in {total_runtime.total_seconds() / 60:.2f} minutes.")
    else:
        # Main task with leading blank line
        print(f"⏱️ {func.__name__} completed in {total_runtime.total_seconds() / 60:.2f} minutes.")
    return result
    
def delete_hive_db(db_name: str):
    print(f"         🧹 Dropping Hive database: {db_name}")
    spark.sql(f"DROP DATABASE IF EXISTS {db_name} CASCADE")

def db_stats(db_name: str):
    # Get and display default locations with description
    #warehouse_dir = spark.conf.get("spark.sql.warehouse.dir", "Not set")
    #delta_base_path = spark.conf.get("spark.delta.basePath", "Not set")
    #filesource_path = spark.conf.get("spark.sql.filesource.path", "Not set")
    #nond2rd_path = spark.conf.get("spark.nond2rd.defaultpath", "Not set")
    
    # Print configuration values
    #print(f"⚡️ spark.sql.warehouse.dir     : {warehouse_dir}")
    #print(f"⚡️ spark.delta.basePath        : {delta_base_path}")
    #print(f"⚡️ spark.sql.filesource.path   : {filesource_path}")
    #print(f"⚡️ spark.nond2rd.defaultpath   : {nond2rd_path}")
    
    # Show all databases in Hive - Using Spark SQL
    #spark.sql("SHOW DATABASES").show(truncate=False)
    
    # List all databases using Spark catalog - Using Catalog API
    db_list = spark.catalog.listDatabases()
             
    # Display databases
    db_found = False
    for db in db_list:
        if db.name == db_name:
            db_found = True
            print(f"            ℹ️ Database Name: {db.name}, Location: {db.locationUri}")
        
    if not db_found:
        print(f"            ⚠️ Database Name: {db_name} was not found")

def write_data(pdf_chunk, overwrite_table: bool = False, show_stats: bool = False):
        # Convert to Spark
        sdf_chunk = spark.createDataFrame(pdf_chunk)
        sdf_chunk = sdf_chunk.withColumn("IngestTime", lit(ingest_ts))
        '''
        # catch 22 - if I repartion the chunks it uses memory instead of letting the write 
        # handle the partioning directly but if I do this then this helps Spark parallelize writes more efficiently.
        sdf_chunk = sdf_chunk.repartition(32, "CompanyId")
        '''
        # Write to Delta 
        #print(f"🧠 Estimated memory footprint: {sdf_chunk.rdd.map(lambda x: len(str(x))).sum()} bytes")
        if show_stats:
            try:
                est_bytes = sdf_chunk.rdd.map(lambda x: len(str(x))).sum()
                est_mib = round(est_bytes / (1024 * 1024), 2)
                print(f"         🧠 Estimated memory footprint: {est_mib} MiB")
            except Exception as e:
                print(f"         ⚠️ Memory estimation failed: {e}")

        sdf_chunk.write.format("delta") \
            .mode("overwrite" if overwrite_table else "append") \
            .partitionBy("CompanyId") \
            .saveAsTable("bsf.companystockhistory")

def optimize_table(ingest_ts: str = None):
    """
    Compact and Z-Order the Delta table.
    If ingest_ts is provided, optimize only that batch.
    """
    if ingest_ts:
        query = f"""
            OPTIMIZE bsf.companystockhistory
            WHERE IngestTime = TIMESTAMP '{ingest_ts}'
            ZORDER BY (StockDate)
        """
        print(f"         🧹 ZORDER on IngestTime = {ingest_ts}")
    else:
        query = """
            OPTIMIZE bsf.companystockhistory
            ZORDER BY (StockDate)
        """
        print(f"         🧹 ZORDER on entire table")

    spark.sql(query)

    print("      ✅ OPTIMIZE/ZORDER Completed on StockDate: bsf.companystockhistory")



# -------------------------------
# 1️⃣ Candlestick Pattern Engine
# -------------------------------
"""
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
"""


def add_candle_patterns(df, o="open", h="high", l="low", c="close",
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

def add_trend_filters(df, c="close"):
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
# 3️⃣ Confirmed Signals
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

 
'''
Step	finalize_signals() version	Your inline version	Match?
Classify action	df["Action"] = df.apply(classify_action, axis=1)	✅ Same	✅
Suppress repeats	np.where(...) with PrevAction	✅ Same	✅
Tomorrow return	(TomorrowClose - close) / close	✅ Same	✅
Tomorrow action label	np.where(... > buy_thresh, ..., < sell_thresh, ...)	Hardcoded thresholds 0.01 and -0.01	✅
Confidence score	Normalized by max	✅ Same	✅
Signal duration	Cumulative change tracker	✅ Same	✅
'''


def finalize_signals(df, buy_thresh=0.01, sell_thresh=-0.01):
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
    df["TomorrowClose"] = df["close"].shift(-1)
    df["TomorrowReturn"] = (df["TomorrowClose"] - df["close"]) / df["close"]
    df["TomorrowReturn"] = df["TomorrowReturn"].fillna(0)

    # Backward-looking momentum
    df["YesterdayReturn"] = df["close"].pct_change().fillna(0)

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

def load_company():
    
    schema = StructType([
        StructField("CompanyId", IntegerType(), True),
        StructField("TradingSymbol", StringType(), True),
        StructField("Name", StringType(), True),
        StructField("ListingExchange", IntegerType(), True),
        StructField("TDAAssetType", IntegerType(), True),
        StructField("IndustryId", IntegerType(), True),
        StructField("MarketSectorId", IntegerType(), True),
        StructField("Cusip", StringType(), True),
        StructField("CIK", StringType(), True),
        StructField("Country", StringType(), True),
        StructField("NbrOfEmployees", IntegerType(), True),
        StructField("IPODate", DateType(), True),
        StructField("FoundedDate", DateType(), True),
        StructField("LogoImage", StringType(), True),
        StructField("LastOpen", FloatType(), True),
        StructField("LastClose", FloatType(), True),
        StructField("LastVolume", FloatType(), True),
        StructField("LastHigh", FloatType(), True),
        StructField("LastLow", FloatType(), True),
        StructField("LastHistoryDate", DateType(), True),
        StructField("LastExtractDate", DateType(), True),
        StructField("ActualHistoryRecord", IntegerType(), True),
        StructField("Active", IntegerType(), True),
        StructField("CreateDate", TimestampType(), True),
        StructField("ChangeDate", TimestampType(), True),
        StructField("ModifiedByProcess", StringType(), True),
        StructField("ModifiedByUserId", IntegerType(), True),
        StructField("SoftDelete", IntegerType(), True)
    ])
    pdf_company = pd.read_sql("SELECT * FROM company WHERE ListingExchange IN (1,2,3,16)", engine)
    #print(f"📊 Rows read from MariaDB: {pdf_company.shape[0]:,}")
    
    # ----------------------------
    # Convert to Spark DataFrame
    # ----------------------------
    # sdf_company = spark.createDataFrame(pdf_company)
    sdf_company = spark.createDataFrame(pdf_company, schema=schema)
    
    # ----------------------------
    # Write to Delta managed table in lakehouse
    # ----------------------------
    print(f"         ⚡️ Start writing delta tables for company.")
    sdf_company.write.format("delta") \
        .mode("overwrite") \
        .partitionBy("ListingExchange") \
        .saveAsTable("bsf.company")
    
    print(f"         ✅ Company table written to Delta lakehouse as bsf.company with {pdf_company.shape[0]:,} rows")

def load_lakehouse():

    # ----------------------------
    # 3️⃣ Chunked read from MariaDB
    # ----------------------------
    chunk_size = 85
    #company_ids = pd.read_sql("SELECT DISTINCT CompanyId FROM company WHERE ListingExchange IN (1,2,3,16)", engine)

    company_ids = (
        spark.sql("""
                    SELECT DISTINCT CompanyId
                    FROM bsf.company
                    WHERE ListingExchange IN (1,2,3,16)
                      AND Active = 1
                      AND LastClose BETWEEN 0.001 AND 0.05
                      AND LastHistoryDate >= date_sub(current_date(), 30)
            """)
            .rdd.flatMap(lambda x: x)
            .collect()
        )
    

    
    # company_batches = [company_ids[i:i+100] for i in range(0, len(company_ids), 100)]
    # try to reduce this error: 25/08/28 11:01:49 WARN TaskSetManager: Stage 111 contains a task of very large size (1087 KiB). The maximum recommended task size is 1000 KiB.
    company_batches = [company_ids[i:i+chunk_size] for i in range(0, len(company_ids), chunk_size)]
    overwrite_table = True
    total_batches = len(company_batches)
    #for idx, batch in enumerate(company_batches, start=1):
    for idx, batch in enumerate(tqdm(company_batches, total=total_batches, desc="📦 Loading batches"), start=1):
        company_list_str = "(" + ",".join(str(cid) for cid in batch) + ")"

        query = f"""
            SELECT csh.CompanyId, csh.StockDate, csh.OpenPrice, csh.HighPrice,
                   csh.LowPrice, csh.ClosePrice, csh.StockVolume
            FROM companystockhistory csh
            WHERE csh.CompanyId IN {company_list_str}
            AND csh.StockDate >= DATE_ADD(NOW(), INTERVAL -410 DAY)
            ORDER BY csh.CompanyId, csh.StockDate
        """


        pdf_chunk = pd.read_sql(query, engine)
        if not pdf_chunk.empty:
            write_data(pdf_chunk, overwrite_table, show_stats=False)
        else:
            print(f"         ⚠️ Skipping empty chunk {idx}/{total_batches} — no matching data")
         

        #print(f"📊 Processing chunk with {pdf_chunk.shape[0]} rows")
        #write_data(pdf_chunk, overwrite_table, show_stats=False)
        overwrite_table = False

        # 🧮 Print progress
        #percent_done = round((idx / total_batches) * 100, 2)
        #print(f"📦 Chunk {idx}/{total_batches} complete — {percent_done}% done")
        
    print("         ✅ All chunks written to Delta table: bsf.companystockhistory")

def write_candidates(final_df, history_df):
    # 💾 Save both to Delta tables
    print(f"         ✅ SCompleted writing delta tables for candidates <daily>.")
    spark.createDataFrame(final_df).write.format("delta") \
        .mode("overwrite") \
        .option("mergeSchema","true") \
        .saveAsTable("bsf.final_daily_signals")
    
    print(f"         ✅ SCompleted writing delta tables for candidates <full>.")
    spark.createDataFrame(history_df).write.format("delta") \
        .mode("overwrite") \
        .option("mergeSchema","true") \
        .saveAsTable("bsf.full_daily_signals")  
    
def load_candidates():
    #print(f"     📦 Load candidates")  
    #company_ids = spark.sql("SELECT DISTINCT CompanyId FROM bsf.companystockhistory order by CompanyId") \
    #                   .rdd.flatMap(lambda x: x).collect()
    sdf = spark.sql("""
            SELECT csh.CompanyId, csh.StockDate, csh.OpenPrice AS open, csh.HighPrice AS high,
                   csh.LowPrice AS low, csh.ClosePrice AS close
            FROM bsf.companystockhistory csh
            ORDER BY CompanyId, StockDate
        """)
    full_pdf = sdf.toPandas()

    latest_signals = []
    company_ids = full_pdf["CompanyId"].unique()
   
    all_history = []       # Stores full processed history
    latest_signals = []    # Stores only the last row per company
    
    for cid in tqdm(company_ids, desc="Processing companies"):
        if len(full_pdf[full_pdf["CompanyId"] == cid]) > 50:  # longest MA window
            company_df = full_pdf[full_pdf["CompanyId"] == cid].copy()
            # Apply signal processing steps
            company_df = add_candle_patterns(company_df)
            company_df = add_trend_filters(company_df)
            company_df = add_confirmed_signals(company_df)
            company_df = add_signal_strength(company_df)
            company_df = finalize_signals(company_df)
            #company_df = add_trend_filters(company_df)
            company_df = add_batch_metadata(company_df, cid)
        
            # Append full history and latest signal
            all_history.append(company_df)
            latest_signals.append(company_df.sort_values("StockDate").tail(1))
    
    # 🔄 Concatenate full history and final signals
    history_df = pd.concat(all_history, ignore_index=True)
    final_df = pd.concat(latest_signals, ignore_index=True)
    
    run_with_logging(
        write_candidates,
        "⏳",
        f"Write Candidate Lakehouse Files",
        final_df=final_df,
        history_df = history_df
    )
  



# 🚀 Main
def main(mode="full"):
        
    #prepare_lakehouse_environment(drop_db=delete_db, mode=mode, db_name = "bsf")
    run_with_logging(
            prepare_lakehouse_environment,
            "⏳",
            f"Run Prepare Lakehouse Environment",
            mode=mode,
            db_name = "bsf"
        )

    if mode == "full":
        run_with_logging(
            load_company,
            "⏳",
            True,
            f"Run Load Company"
        )
        run_with_logging(
                load_lakehouse,
                "⏳",
                True,
                f"Run Lakehouse {mode} Load"
            )
        run_with_logging(
            optimize_table,
            "⏳",
            True,
            f"Run Optimize Table"
        )
    elif mode == "candidates":
        #print(f"📦 Lakehouse Load Candidates")  
        #load_candidates()
        run_with_logging(
                load_candidates,
                "⏳",
                True,
                f"Run Load Candidates"
            )
        #print(f"✅ Lakehouse Load candidates complete")

    
if __name__ == "__main__":
    #print(f"🕒 Ingest timestamp for this process: {ingest_ts}")
    parser = argparse.ArgumentParser(description="Build Lakehouse")
    parser.add_argument("--mode", choices=["full","candidates"], default="full", help="Load mode")

    args = parser.parse_args()
    run_with_logging(
                main,
                "🚀",
                False,
                f"Run Main",
                mode=args.mode
            )
    spark.stop()
    engine.dispose()


