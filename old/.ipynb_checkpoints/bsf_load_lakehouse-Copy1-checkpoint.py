# ─── Core CLI and Time Utilities ─────────────────────────────
import argparse
from datetime import datetime, timedelta

# ─── Spark Setup and Utilities ───────────────────────────────
from bsf_env import (
    init_spark,
    init_mariadb_engine,
    set_spark_verbosity
)
from bsf_db_utilities import DBUtils

from bsf_markers import (
    get_candle_params,
    get_pattern_window,
    add_candle_patterns,
    add_trend_filters,
    add_confirmed_signals,
    add_signal_strength,
    finalize_signals,
    add_batch_metadata,
    generate_signal_columns
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
db = None

def run_with_logging(func, icon="⏳",  is_subtask=False, title=None, *args, **kwargs):
    bold_title = f" \033[1m** {title} **\033[0m" if title else ''

    if is_subtask:
        # Indented subtask, no extra blank line
        print(f"   ➤ {icon} {func.__name__} is running{bold_title}...")
    else:
        # Main task with leading blank line
        print(f"\n{icon} {func.__name__} is running{bold_title}...")

    # Capture start time
    start_time = datetime.now()
    # Run Method
    result = func(*args, **kwargs)
    # Capture end time
    end_time = datetime.now()
    
    total_runtime = end_time - start_time
    if is_subtask:
        # Indented subtask, no extra blank line
        print(f"     ⏰ {func.__name__} completed in {total_runtime.total_seconds() / 60:.2f} minutes.")
    else:
        # Main task with leading blank line
        print(f"⏰ {func.__name__} completed in {total_runtime.total_seconds() / 60:.2f} minutes.")
    return result
    
def prepare_lakehouse_environment(mode: str= None, option: str=None, db_name: str = "bsf"):
    # ─── Setup Environment ──────────────────────────────────────
    global spark, engine, ingest_ts, db

    if mode == 'signals':
        process_option='wide'
    elif mode=='history':
        process_option='tall'
    elif mode=='candidates':
        process_option='wide'
    else:
        process_option='default'
        
    spark = init_spark(f"{db_name}_{mode}_{option}", log_level="ERROR", show_progress=False, enable_ui=True, process_option=process_option)
    engine = init_mariadb_engine()
    ingest_ts = spark.sql("SELECT current_timestamp()").collect()[0][0]
    bold_title = f" \033[1m** {db_name} {mode} >>> {option} <<< **\033[0m"
    print(f"      ⚡️ Spark Session Initialized {bold_title}")
    
    # ─── Setup Database Communications ──────────────────────────────────────
    db = DBUtils(spark, ingest_ts)
    db.spark_stats()

def load_company():
    db.clear_hive_table('bsf','company')

    
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
    
    # ----------------------------
    # Convert to Spark DataFrame
    # ----------------------------
    sdf_company = spark.createDataFrame(pdf_company, schema=schema)
    
    # ----------------------------
    # Write to Delta managed table in lakehouse
    # ----------------------------
    print(f"      ⚡️ Start writing delta tables for company.")
    sdf_company.write.format("delta") \
        .mode("overwrite") \
        .partitionBy("ListingExchange") \
        .saveAsTable("bsf.company")
    
    print(f"      ✅ Company table written to Delta lakehouse as bsf.company with {pdf_company.shape[0]:,} rows")

def load_history(chunk_size=85, option='full', lookback_days=410):
    """
    Loads company stock history from MariaDB into Delta Lake (bsf.companystockhistory)
    with optional incremental loading using a watermark table (bsf.companystockhistory_watermark).

    Parameters:
        chunk_size (int): Number of companies per chunk.
        incremental (bool): If True, only load new data since last watermark.
        lookback_days (int): Number of days to pull for initial full load or fallback.
    """
    incremental = True if option.lower() == "incremental" else False
    from tqdm import tqdm
    import pandas as pd

    # ----------------------------
    # Tables cleanup
    # ----------------------------
    if not incremental:
        db.clear_hive_table('bsf', 'companystockhistory')
        db.clear_hive_table('bsf', 'companystockhistory_watermark')

    # ----------------------------
    # Get active companies
    # ----------------------------
    company_ids = (
        spark.sql("""
            SELECT DISTINCT CompanyId
            FROM bsf.company
            WHERE ListingExchange IN (1,2,3,16)
              AND Active = 1
              AND LastClose BETWEEN 0.001 AND 0.1
              AND LastHistoryDate >= date_sub(current_date(), 30)
        """)
        .rdd.flatMap(lambda x: x)
        .collect()
    )

    company_batches = [company_ids[i:i + chunk_size] for i in range(0, len(company_ids), chunk_size)]
    total_batches = len(company_batches)
    overwrite_table = not incremental

    # ----------------------------
    # Load each batch
    # ----------------------------
    for idx, batch in enumerate(tqdm(company_batches, total=total_batches, desc="🔄 Loading batches"), start=1):
        company_list_str = "(" + ",".join(str(cid) for cid in batch) + ")"

        # ----------------------------
        # Determine date filter per batch
        # ----------------------------
        if incremental:
            # Attempt to read watermark table
            try:
                wm_df = spark.sql(f"""
                    SELECT CompanyId, LastLoadedDate
                    FROM bsf.companystockhistory_watermark
                    WHERE CompanyId IN {company_list_str}
                """).toPandas()
                wm_dict = dict(zip(wm_df.CompanyId, wm_df.LastLoadedDate))
            except Exception:
                # Watermark table missing → fallback to full batch load
                wm_dict = {}

            date_conditions = []
            for cid in batch:
                last_date = wm_dict.get(cid)
                if last_date:
                    date_conditions.append(f"(csh.CompanyId={cid} AND csh.StockDate > '{last_date}')")
                else:
                    date_conditions.append(f"(csh.CompanyId={cid} AND csh.StockDate >= DATE_ADD(NOW(), INTERVAL -{lookback_days} DAY))")
            date_condition = " OR ".join(date_conditions)
        else:
            # Full load: last N days for all companies
            date_condition = f"csh.StockDate >= DATE_ADD(NOW(), INTERVAL -{lookback_days} DAY)  AND csh.CompanyId IN {company_list_str}" #AND csh.StockDate <= DATE_ADD(NOW(), INTERVAL -30 DAY)

        # ----------------------------
        # Fetch chunk
        # ----------------------------
        query = f"""
            SELECT csh.CompanyId, csh.StockDate, csh.OpenPrice, csh.HighPrice,
                   csh.LowPrice, csh.ClosePrice, csh.StockVolume
            FROM companystockhistory csh
            WHERE {date_condition} 
            ORDER BY csh.CompanyId, csh.StockDate
        """
        pdf_chunk = pd.read_sql(query, engine)

        if not pdf_chunk.empty:
            # Write to Delta / Hive
            db.write_history(pdf_chunk, overwrite_table, show_stats=False)
            overwrite_table = False
        else:
            print(f"❗ Skipping empty chunk {idx}/{total_batches} — no matching data")

    print("✅ All batches loaded into bsf.companystockhistory")



    
def load_signals(timeframe=None, option='full'):
    
    incremental = True if option.lower() == "incremental" else False
    if not incremental:
        #db.clear_hive_table('bsf','history_signals_allcol')
        db.clear_hive_table('bsf','history_signals')
        db.clear_hive_table('bsf','history_signals_last_all')
        db.clear_hive_table('bsf','history_signals_last')

    sdf = spark.sql("""
        SELECT csh.CompanyId, csh.StockDate, csh.OpenPrice AS Open, csh.HighPrice AS High,
               csh.LowPrice AS Low, csh.ClosePrice AS Close, csh.StockVolume AS Volume
        FROM bsf.companystockhistory csh
        ORDER BY CompanyId, StockDate
    """)

    full_pdf = sdf.toPandas()
    company_ids = full_pdf["CompanyId"].unique()

    # ✅ Select timeframes
    if timeframe is None:
        timeframes = ["Short", "Swing", "Long", "Daily"]
    else:
        timeframes = [timeframe]

    for tf in timeframes:
        print(f"      ⏳ Processing timeframe: {tf}")
        pattern_window = get_pattern_window(tf)

        # 🔹 Split full PDF by CompanyId only once
        grouped = {cid: df.copy() for cid, df in full_pdf.groupby("CompanyId") if len(df) > 50}
    
        history_signals = []
        latest_signals = []
    
        for cid, company_df in tqdm(grouped.items(), desc=f"      🔄 {tf} companies"):
            # Get the last Close price
            last_close = company_df["Close"].iloc[-1]

            # Get thresholds for this company
            candle_params = get_candle_params(last_close)
            # Apply all the transformations in one Pandas pipeline
            # --- Apply transformations ---
            company_df = (
                    company_df
                    .pipe(add_candle_patterns, pattern_window=pattern_window, **candle_params)
                    .pipe(add_trend_filters, timeframe=tf)
                    .pipe(add_confirmed_signals)
                )
            
            candle_cols, trend_cols, momentum_factor = generate_signal_columns(company_df, tf)
                
            company_df = (
                    company_df
                    .pipe(
                        finalize_signals,
                        pattern_window=pattern_window,
                        bullish_patterns=trend_cols["Bullish"],
                        bearish_patterns=trend_cols["Bearish"],
                        momentum_factor=momentum_factor,
                        candle_columns=candle_cols
                    )
                    .pipe(add_signal_strength)
                    .pipe(add_batch_metadata, cid, tf)
                )
            history_signals.append(company_df)
    
        # ✅ Concatenate once per timeframe (avoid repeated Pandas merges)
        history_df = pd.concat(history_signals, ignore_index=True)
        # Find duplicates based on your merge keys
        dup_keys = history_df[history_df.duplicated(subset=["CompanyId", "StockDate", "TimeFrame"], keep=False)]
        '''
        # Keep only the keys
        dup_keys = dup_keys[["CompanyId", "StockDate", "TimeFrame"]].drop_duplicates()
        
        print("Duplicate keys:")
        print(dup_keys)
        '''
        
        # ✅ Save per timeframe
        run_with_logging(
            db.write_signals,
            icon="⏳",
            is_subtask=True,
            title=f"Write Candidate Lakehouse Partition: ({tf})",
            history_df=history_df,
            timeframe=tf
            )



# 🚀 Main
def main(mode=None, option="full", timeframe=None):
    db_name ="bsf"
    
    run_with_logging(
            prepare_lakehouse_environment,
            "⏳",
            f"Run Prepare Lakehouse Environment",
            mode=mode,
            db_name = db_name,
            option = option
        )

    db.db_stats(db_name)

    if mode == "history":

        run_with_logging(
            load_company,
            "⏳",
            True,
            f"Run Load Company"
        )
        
        run_with_logging(
                load_history,
                "⏳",
                True,
                f"Run Lakehouse History {option} Load",
                chunk_size = 85,
                option = option
            )
        
        run_with_logging(
            db.optimize_table,
            "⏳",
            True,
            f"Run Optimize Table"
        )

    elif mode == "signals":
        run_with_logging(
                load_signals,
                "⏳",
                True,
                f"Run Load Signals",
                timeframe=timeframe,
                option = option
            )
    elif mode == "candidates":
        run_with_logging(
                load_candidates,
                "⏳",
                True,
                f"Run Load Candidates"
            )
    else:
        pass
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Lakehouse")

    parser.add_argument(
        "--mode",
        choices=["history", "signals", "candidates"],
        default="history",
        help="Process mode (load, candidates, signals)"
    )

    parser.add_argument(
        "--option",
        choices=["full", "incremental"],
        default=None,
        help="Load type option (full, incremental)"
    )
    
    parser.add_argument(
        "--timeframe",
        choices=["daily", "swing", "short", "long"],
        default=None,
        help="Timeframe (daily, swing, short, long)"
    )

    args = parser.parse_args()
    mode = args.mode.lower()
    timeframe = args.timeframe.capitalize() if args.timeframe else None
    option = args.option.lower()
    
    run_with_logging(
        main,
        "🚀",
        False,
        f"Run Main for mode: {args.mode} options: {args.option}",
        mode=mode,
        option=option,  
        timeframe=timeframe
    )

    spark.stop()
    engine.dispose()



