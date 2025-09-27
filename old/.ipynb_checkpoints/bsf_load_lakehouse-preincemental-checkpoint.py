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
    
def prepare_lakehouse_environment(mode: str= None, db_name: str = "bsf"):
    # ─── Setup Environment ──────────────────────────────────────
    global spark, engine, ingest_ts, db
    
    spark = init_spark(f"{db_name}_lakehouse_{mode}", log_level="ERROR", show_progress=False, enable_ui=True)
    engine = init_mariadb_engine()
    ingest_ts = spark.sql("SELECT current_timestamp()").collect()[0][0]
    bold_title = f" \033[1m**{db_name}_lakehouse_{mode}**\033[0m"
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

def load_lakehouse(chunk_size = 85):
    # ----------------------------
    # Chunked read from MariaDB
    # ----------------------------        #db.delete_hive_db(db_name)
        
    db.clear_hive_table('bsf','companystockhistory')
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
    # AND CompanyId in (87972,65429,30524 )
    company_batches = [company_ids[i:i+chunk_size] for i in range(0, len(company_ids), chunk_size)]
    overwrite_table = True
    total_batches = len(company_batches)

    for idx, batch in enumerate(tqdm(company_batches, total=total_batches, desc="      🔄 Loading batches"), start=1):
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
            db.write_history(pdf_chunk, overwrite_table, show_stats=False)
        else:
            print(f"      ❗ Skipping empty chunk {idx}/{total_batches} — no matching data")
         
        overwrite_table = False
    print("      ✅ All chunks written to Delta table: bsf.companystockhistory")



def load_lakehouse_new(chunk_size=85, full_load=False):
    """
    Load company stock history from MariaDB into Delta table.

    Parameters:
        chunk_size (int): Number of companies per batch.
        full_load (bool): If True, drop the table and load all history.
                          If False, append only incremental rows after the last date per company.
    """
    table_name = "bsf.companystockhistory"

    # ----------------------------
    # Drop table if full load
    # ----------------------------
    if full_load:
        print("      ⚠️ Full load requested — clearing Delta table")
        db.clear_hive_table('bsf','companystockhistory')

    # ----------------------------
    # Get list of companies to process
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

    company_batches = [company_ids[i:i+chunk_size] for i in range(0, len(company_ids), chunk_size)]
    total_batches = len(company_batches)

    # ----------------------------
    # Incremental logic: fetch last date per company only if incremental
    # ----------------------------
    if not full_load:
        last_dates_df = (
            spark.sql(f"SELECT CompanyId, MAX(Date) as max_date FROM {table_name} GROUP BY CompanyId")
            .toPandas()
        )
        last_dates_map = dict(zip(last_dates_df["CompanyId"], last_dates_df["max_date"]))
    else:
        last_dates_map = {}

    # ----------------------------
    # Process batches
    # ----------------------------
    for idx, batch in enumerate(tqdm(company_batches, total=total_batches, desc="      🔄 Loading batches"), start=1):
        for cid in batch:
            last_date = last_dates_map.get(cid) if not full_load else None

            query = f"""
                SELECT csh.CompanyId, csh.StockDate as Date, csh.OpenPrice, csh.HighPrice,
                       csh.LowPrice, csh.ClosePrice, csh.StockVolume
                FROM companystockhistory csh
                WHERE csh.CompanyId = {cid}
            """

            if last_date:
                query += f" AND csh.StockDate > '{last_date}'"

            query += " ORDER BY csh.StockDate"

            pdf_chunk = pd.read_sql(query, engine)
            if pdf_chunk.empty:
                continue

            sdf_chunk = spark.createDataFrame(pdf_chunk)

            # Append to Delta table
            sdf_chunk.write.format("delta") \
                .mode("append") \
                .partitionBy("CompanyId") \
                .saveAsTable(table_name)

    print(f"      ✅ Load complete: {table_name}")


    
def load_candidates(timeframe=None):
    db.clear_hive_table('bsf','daily_signals_allcol')
    db.clear_hive_table('bsf','daily_signals_last_allcol')
    db.clear_hive_table('bsf','daily_signals')
    db.clear_hive_table('bsf','daily_signals_last')

    sdf = spark.sql("""
        SELECT csh.CompanyId, csh.StockDate, csh.OpenPrice AS Open, csh.HighPrice AS High,
               csh.LowPrice AS Low, csh.ClosePrice AS Close
        FROM bsf.companystockhistory csh
        ORDER BY CompanyId, StockDate
    """)
    #        WHERE csh.CompanyId in (87972,65429,30524 )
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
            '''
             .pipe(
                add_candle_patterns,
                pattern_window=pattern_window,
                **candle_params
            )  
            .pipe(
                add_candle_patterns,
                doji_thresh=candle_params["doji_thresh"],
                hammer_threshold =candle_params["hammer_thresh"],
                marubozu_threshold =candle_params["marubozu_thresh"],
                long_body=candle_params["long_body"],
                shadow_ratio=candle_params["shadow_ratio"],
                small_body=candle_params["small_body"],
                pattern_window=pattern_window,
                near_edge=candle_params["near_edge"],
                rng_thresh=candle_params["rng_thresh"]
            )
            '''
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
    
        # ✅ Save per timeframe
        run_with_logging(
            db.write_candidates,
            icon="⏳",
            is_subtask=True,
            title=f"Write Candidate Lakehouse Partition: ({tf})",
            history_df=history_df,
            timeframe=tf
            )



# 🚀 Main
def main(mode=None, option="short"):
    db_name ="bsf"
    
    run_with_logging(
            prepare_lakehouse_environment,
            "⏳",
            f"Run Prepare Lakehouse Environment",
            mode=mode,
            db_name = db_name
        )

    db.db_stats(db_name)

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
                f"Run Lakehouse {mode} Load",
                chunk_size = 85
            )
        run_with_logging(
            db.optimize_table,
            "⏳",
            True,
            f"Run Optimize Table"
        )
    elif mode == "candidates":
        run_with_logging(
                load_candidates,
                "⏳",
                True,
                f"Run Load Candidates",
                timeframe=option
            )
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Lakehouse")

    # existing
    parser.add_argument(
        "--mode",
        choices=["full", "candidates"],
        default="full",
        help="Load mode"
    )

    # ✅ new timeframe option
    parser.add_argument(
        "--option",
        choices=["short", "swing", "long", "daily"],
        default=None,
        help="Timeframe option (short, swing, long, daily)"
    )

    args = parser.parse_args()
    option = args.option.capitalize() if args.option else None
    
    run_with_logging(
        main,
        "🚀",
        False,
        f"Run Main for mode: {args.mode} options: {option}",
        mode=args.mode,
        option=option   # ✅ pass it through
    )

    spark.stop()
    engine.dispose()



