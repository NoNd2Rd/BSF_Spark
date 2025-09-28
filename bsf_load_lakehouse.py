# ─── Core CLI and Time Utilities ─────────────────────────────
import argparse
from datetime import datetime, timedelta
import time    

# ─── Spark Setup and Utilities ───────────────────────────────
from bsf_config import CONFIG

from bsf_env import (
    init_spark,
    init_mariadb_engine,
    set_spark_verbosity
)
from bsf_dbutilities import DBUtils

from bsf_candlesticks import (
    add_candle_patterns_optimized,
    add_trend_filters_optimized,
    finalize_signals_optimized,
    add_signal_strength_optimized,
    add_batch_metadata_optimized,
    compute_fundamental_score_optimized,
    add_confirmed_signals_optimized
)

from bsf_candidates import (
    phase_1,
    phase_2,
    phase_3
)

# ─── PySpark Functions and Types ─────────────────────────────
from pyspark.sql import functions as F
from pyspark.sql.functions import lit, current_timestamp, broadcast
from pyspark.sql.types import *
from pyspark.sql.window import Window
from pyspark import StorageLevel

# ─── Data Science Stack ──────────────────────────────────────
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import reduce
import math
import sys

# ─── Global ──────────────────────────────────────
spark = None
engine = None
ingest_ts = None
db = None

def run_with_logging(func, icon="⏳",  is_subtask=False, title=None, *args, **kwargs):
    bold_title = f" \033[1m** {title} **\033[0m" if title else ''

    if is_subtask:
        # Indented subtask, no extra blank line
        print(f"  {icon} {func.__name__} is running{bold_title}...")
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
        print(f"  ✅ {func.__name__} completed in {total_runtime.total_seconds() / 60:.2f} minutes.")
    else:
        # Main task with leading blank line
        print(f"✅ {func.__name__} completed in {total_runtime.total_seconds() / 60:.2f} minutes.")
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
    pdf_company = pd.read_sql("SELECT * FROM company WHERE ListingExchange IN (1,2,3,16)", engine)
    db.write_company(pdf_company)
    print(f"      ✔️ Company table written to Delta lakehouse as bsf.company with {pdf_company.shape[0]:,} rows")

    
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
        db.clear_hive_table('bsf', 'companyfundamental')
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
              AND LastClose BETWEEN 0.0001 AND 0.1
              AND LastHistoryDate >= date_sub(current_date(), 30)
        """)
        .rdd.flatMap(lambda x: x)
        .collect()
    )

    company_batches = [company_ids[i:i + chunk_size] for i in range(0, len(company_ids), chunk_size)]
    total_batches = len(company_batches)
    overwrite_table_history = not incremental
    overwrite_table_fund = not incremental
    
    # ----------------------------
    # Load each batch
    # ----------------------------
    for idx, batch in enumerate(tqdm(company_batches, total=total_batches, desc="    🔄 Loading batches"), start=1):
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
        hist_query = f"""
            SELECT csh.CompanyId, csh.StockDate, csh.OpenPrice, csh.HighPrice,
                   csh.LowPrice, csh.ClosePrice, csh.StockVolume
            FROM companystockhistory csh
            WHERE {date_condition} 
            ORDER BY csh.CompanyId, csh.StockDate
        """
        fund_query = f"""
                SELECT 
                    CompanyId, 
                    FundamentalDate, 
                    PeRatio, 
                    PegRatio, 
                    PbRatio, 
                    ReturnOnEquity, 
                    GrossMarginTTM,
                    NetProfitMarginTTM, 
                    TotalDebtToEquity,
                    CurrentRatio, 
                    InterestCoverage, 
                    EpsChangeYear, 
                    RevChangeYear, 
                    Beta, 
                    ShortIntToFloat
                FROM bsf.companyfundamental where CompanyId IN {company_list_str}
                """
        pdf_hist = pd.read_sql(hist_query, engine)
        pdf_fund = pd.read_sql(fund_query, engine)

        if not pdf_hist.empty:
            # Write to Delta / Hive
            db.write_history(pdf_hist, overwrite_table_history, show_stats=False)
            overwrite_table_history = False
        else:
            print(f"    ❗ Skipping empty history chunk {idx}/{total_batches} — no matching data")

        if not pdf_fund.empty:
            # Write to Delta / Hive
            db.write_fundamental(pdf_fund, overwrite_table_fund, show_stats=False)
            overwrite_table_fund = False
        else:
            print(f"    ❗ Skipping empty fundamental chunk {idx}/{total_batches} — no matching data")
            
    print("    ✔️ All batches loaded into bsf.companystockhistory")


def load_candidates():
    user_keys = db.get_users(engine)

    for user_key in user_keys:
        print("Current user: {user_key}")
              
    db.clear_hive_table('bsf','final_candidates_enriched')
    db.clear_hive_table('bsf','final_candidates')
 
    df_last = spark.table("bsf.history_signals_last")
    df_all = spark.table("bsf.history_signals")
    timeframe_dfs_all, timeframe_dfs   = phase_1(spark, df_all, df_last, top_n=30)
    phase2_topN_dfs = phase_2(spark, timeframe_dfs_all, top_n_phase2=15)
    df_phase3_enriched, df_topN_companies, phase3_enriched_dict, topN_companies_dict = phase_3(spark, phase2_topN_dfs, top_n_final=5)
    
    db.write_candidates(df_phase3_enriched, df_topN_companies)
    db.create_bsf(engine, topN_companies_dict)


import math
import pandas as pd
from datetime import datetime

def load_signals(timeframe=None, option="full", batch_size=1000):
    """
    Pandas-based signal processing for small node (4 cores, 6 GB RAM).
    Uses batching per company and .pipe() to chain transformations.
    """
    incremental = True if option.lower() == "incremental" else False
    if not incremental:
        db.clear_hive_table('bsf', 'history_signals')
        db.clear_hive_table('bsf', 'history_signals_last_all')
        db.clear_hive_table('bsf', 'history_signals_last')
    # -------------------------------
    
    # Load stock + fundamentals into pandas
    # -------------------------------
    # -------------------------------
    # Load stock + fundamentals
    # -------------------------------
    sdf_stock = spark.table("bsf.companystockhistory").alias("s")
    sdf_fund  = spark.table("bsf.companyfundamental").alias("f")

    sdf_joined = sdf_stock.join(
        sdf_fund,
        (F.col("s.CompanyId") == F.col("f.CompanyId")) &
        (F.col("f.FundamentalDate") <= F.col("s.StockDate")),
        "left"
    )

    # Latest fundamental per stock row
    w = Window.partitionBy("s.CompanyId", "s.StockDate").orderBy(F.col("f.FundamentalDate").desc())
    sdf_all = (
        sdf_joined
        .withColumn("rn", F.row_number().over(w))
        .filter(F.col("rn") == 1)
        .drop("rn")
        .select(
            F.col("s.CompanyId"),
            F.col("s.StockDate"),
            F.col("s.OpenPrice").alias("Open"),
            F.col("s.HighPrice").alias("High"),
            F.col("s.LowPrice").alias("Low"),
            F.col("s.ClosePrice").alias("Close"),
            F.col("s.StockVolume").alias("Volume"),
            F.col("f.FundamentalDate"),
            F.col("f.PeRatio"),
            F.col("f.PegRatio"),
            F.col("f.PbRatio"),
            F.col("f.ReturnOnEquity"),
            F.col("f.GrossMarginTTM"),
            F.col("f.NetProfitMarginTTM"),
            F.col("f.TotalDebtToEquity"),
            F.col("f.CurrentRatio"),
            F.col("f.InterestCoverage"),
            F.col("f.EpsChangeYear"),
            F.col("f.RevChangeYear"),
            F.col("f.Beta"),
            F.col("f.ShortIntToFloat")
        )
    )

    print(f"⚡️ Loaded full DataFrame: {sdf_all.count():,} rows.")
    df_all = sdf_all.toPandas()
    
    # -------------------------------
    # Users & timeframes
    # -------------------------------
    users = db.get_users(engine)
    if timeframe is None:
        timeframes_items = CONFIG["timeframe_map"].items()
    else:
        timeframes_items = [(timeframe, CONFIG["timeframe_map"].get(timeframe, CONFIG["timeframe_map"]["Daily"]))]

    keep_cols = [
        "UserId","CompanyId","StockDate","Open","High","Low","Close","TomorrowClose","Return","TomorrowReturn",
        "MA","MA_slope","UpTrend_MA","DownTrend_MA","MomentumUp","MomentumDown",
        "ConfirmedUpTrend","ConfirmedDownTrend","Volatility","LowVolatility","HighVolatility","SignalStrength",
        "SignalStrengthHybrid","ActionConfidence",
        "BullishStrengthHybrid","BearishStrengthHybrid","SignalDuration",
        "PatternAction","CandleAction","UpTrend_Return",
        "CandidateAction","Action","TomorrowAction","TimeFrame"
    ]
    # -------------------------------
    # Process all users × timeframes in batches
    # -------------------------------
    company_ids = df_all["CompanyId"].unique()

    # -------------------------------
    # Process each user × timeframe × company
    # -------------------------------
    for user in users:
        for tf, tf_window in timeframes_items:
            print(f"🔄 Processing {tf:<6} for user {user} ...")
    
            df_list = []  # collect per-company results
    
            for cid in company_ids:
                # Filter for a single company
                df_company = df_all[df_all["CompanyId"] == cid].copy()
    
                # Sort to preserve time order
                df_company = df_company.sort_values("StockDate")

    
                # Apply transforms with .pipe()
                df_processed = (
                    df_company
                    .pipe(add_candle_patterns_optimized, tf_window=tf_window, user=user)
                    .pipe(add_trend_filters_optimized, timeframe=tf, user=user)
                    .pipe(add_confirmed_signals_optimized)
                    .pipe(compute_fundamental_score_optimized, user=user)
                    .pipe(finalize_signals_optimized, tf=tf, tf_window=tf_window, use_fundamentals=True, user=user)
                    .pipe(add_signal_strength_optimized)
                    .pipe(add_batch_metadata_optimized, timeframe=tf, user=user, ingest_ts=ingest_ts)
                )
    
                # Trim columns
                df_processed = df_processed[keep_cols]
    
                df_list.append(df_processed)
    
            # Union all companies for this timeframe × user
            if df_list:
                df_final = pd.concat(df_list, ignore_index=True)
   
                # Write signals
                run_with_logging(
                    db.write_signals,
                    icon="⏳",
                    is_subtask=True,
                    title=f"Write Candidate Lakehouse Partition: ({tf})",
                    df=df_final   # pass pandas DataFrame instead of Spark
                )
        
                print(f"✅ Signals written for {tf} / user {user}")

    print("✅ All signals processed.")


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



