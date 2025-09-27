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
    add_candle_patterns,
    add_trend_filters,
    add_confirmed_signals,
    add_signal_strength,
    finalize_signals,
    add_batch_metadata,
    compute_fundamental_score
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

# ─── Data Science Stack ──────────────────────────────────────
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import reduce

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

def process_user_timeframe_1(user, tf, tf_window, sdf, ingest_ts, batch_size):
    history_signals = []
    company_ids = [row.CompanyId for row in sdf.select("CompanyId").distinct().toLocalIterator()]
    start_time = time.time()
    
    tf_aligned = f"{tf:<6}"  # left-align for tqdm
    with tqdm(total=len(company_ids), desc=f"     🔄 {tf_aligned} companies for user {user}") as pbar:
        for batch_start in range(0, len(company_ids), batch_size):
            batch_ids = company_ids[batch_start:batch_start+batch_size]
            sdf_batch = sdf.filter(F.col("CompanyId").isin(batch_ids))

            # All transformations
            sdf_batch = (
                sdf_batch
                .transform(lambda df: add_candle_patterns(df, tf_window=tf_window, user=user))
                .transform(lambda df: add_trend_filters(df, timeframe=tf, user=user))
                .transform(add_confirmed_signals)
                .transform(lambda df: compute_fundamental_score(df, user=user))
                .transform(lambda df: finalize_signals(df, tf=tf, tf_window=tf_window, use_fundamentals=True, user=user))
                .transform(add_signal_strength)
                .transform(lambda df: add_batch_metadata(df, timeframe=tf, user=user, ingest_ts=ingest_ts))
            )

            history_signals.append(sdf_batch)
            pbar.update(len(batch_ids))

    if history_signals:
        # Union in chunks to avoid long lineage
        def union_dfs(dfs):
            while len(dfs) > 1:
                new_dfs = []
                for i in range(0, len(dfs), 5):
                    chunk = dfs[i:i+5]
                    new_dfs.append(reduce(lambda a, b: a.unionByName(b), chunk))
                dfs = new_dfs
            return dfs[0]

        history_df = union_dfs(history_signals)
        run_with_logging(
            db.write_signals,
            icon="⏳",
            is_subtask=True,
            title=f"Write Candidate Lakehouse Partition: ({tf})",
            history_df=history_df
        )

def load_signals_1(timeframe=None, option='full'):
    incremental = option.lower() == "incremental"

    if not incremental:
        db.clear_hive_table('bsf', 'history_signals')
        db.clear_hive_table('bsf', 'history_signals_last_all')
        db.clear_hive_table('bsf', 'history_signals_last')

    sdf = spark.sql("""
        SELECT 
            t.CompanyId,
            t.StockDate,
            t.Open,
            t.High,
            t.Low,
            t.Close,
            t.Volume,
            t.FundamentalDate,
            t.PeRatio,
            t.PegRatio,
            t.PbRatio,
            t.ReturnOnEquity,
            t.GrossMarginTTM,
            t.NetProfitMarginTTM,
            t.TotalDebtToEquity,
            t.CurrentRatio,
            t.InterestCoverage,
            t.EpsChangeYear,
            t.RevChangeYear,
            t.Beta,
            t.ShortIntToFloat
        FROM (
            SELECT 
                s.CompanyId,
                s.StockDate,
                s.OpenPrice AS Open,
                s.HighPrice AS High,
                s.LowPrice  AS Low,
                s.ClosePrice AS Close,
                s.StockVolume AS Volume,
                f.FundamentalDate,
                f.PeRatio,
                f.PegRatio,
                f.PbRatio,
                f.ReturnOnEquity,
                f.GrossMarginTTM,
                f.NetProfitMarginTTM,
                f.TotalDebtToEquity,
                f.CurrentRatio,
                f.InterestCoverage,
                f.EpsChangeYear,
                f.RevChangeYear,
                f.Beta,
                f.ShortIntToFloat,
                ROW_NUMBER() OVER (
                    PARTITION BY s.CompanyId, s.StockDate
                    ORDER BY f.FundamentalDate DESC
                ) AS rn
            FROM bsf.companystockhistory s
            LEFT JOIN bsf.companyfundamental f
              ON s.CompanyId = f.CompanyId
             AND f.FundamentalDate <= s.StockDate
        ) t
        WHERE t.rn = 1
    """)

    pdf = sdf_joined.toPandas()
    sdf.unpersist()
    sdf = spark.createDataFrame(pdf)
    sdf = sdf.repartition("CompanyId").cache()
    sdf.count()  # warm-up

    users = db.get_users(engine)
    if timeframe is None:
        timeframes_items = CONFIG["timeframe_map"].items()
    else:
        timeframes_items = [(timeframe, CONFIG["timeframe_map"].get(timeframe, CONFIG["timeframe_map"]["Daily"]))]

    # Get total companies
    company_ids = [row.CompanyId for row in sdf.select("CompanyId").distinct().toLocalIterator()]
    total_companies = len(company_ids)

    # Get cores from Spark session
    sc = spark.sparkContext
    cores = sc.defaultParallelism

    # Auto batch size (2 batches per core)
    batch_size = max(1, total_companies // (cores * 2))

    # Auto-adjust shuffle partitions
    num_partitions = max(8, min(32, total_companies // 100))
    spark.conf.set("spark.sql.shuffle.partitions", num_partitions)
    spark.conf.set("spark.default.parallelism", num_partitions)

    print(f"Detected {cores} cores, total companies {total_companies}, batch_size {batch_size}, shuffle_partitions {num_partitions}")

    # Run in parallel per user × timeframe
    tasks = []
    with ThreadPoolExecutor(max_workers=cores) as executor:
        for user in users:
            for tf, tf_window in timeframes_items:
                tasks.append(executor.submit(
                    process_user_timeframe, user, tf, tf_window, sdf, ingest_ts, batch_size
                ))

        for future in as_completed(tasks):
            future.result()  # raise exceptions if any

    sdf.unpersist()

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import reduce
from tqdm import tqdm
import time
import pyspark.sql.functions as F

# --- helper to process a batch of companies ---
def process_batch(batch_ids, sdf, tf, tf_window, user, ingest_ts):
    sdf_batch = sdf.filter(F.col("CompanyId").isin(batch_ids))

    # Apply all transformations
    sdf_batch = (
        sdf_batch
        .transform(lambda df: add_candle_patterns(df, tf_window=tf_window, user=user))
        .transform(lambda df: add_trend_filters(df, timeframe=tf, user=user))
        .transform(add_confirmed_signals)
        .transform(lambda df: compute_fundamental_score(df, user=user))
        .transform(lambda df: finalize_signals(df, tf=tf, tf_window=tf_window, use_fundamentals=True, user=user))
        .transform(add_signal_strength)
        .transform(lambda df: add_batch_metadata(df, timeframe=tf, user=user, ingest_ts=ingest_ts))
    )
    return sdf_batch

# --- main per user × timeframe ---
def process_user_timeframe(user, tf, tf_window, sdf, ingest_ts, batch_size, max_batch_threads):
    history_signals = []
    company_ids = [row.CompanyId for row in sdf.select("CompanyId").distinct().toLocalIterator()]
    start_time = time.time()
    tf_aligned = f"{tf:<6}"  # left-align for tqdm

    # Slice company_ids into batches
    batch_slices = [company_ids[i:i + batch_size] for i in range(0, len(company_ids), batch_size)]

    flush_size = 25   # write after this many companies
    history_signals = []
    
    with tqdm(total=len(company_ids), desc=f"     🔄 {tf_aligned} companies for user {user}") as pbar:
        with ThreadPoolExecutor(max_workers=max_batch_threads) as batch_executor:
            future_to_batch = {
                batch_executor.submit(process_batch, batch_ids, sdf, tf, tf_window, user, ingest_ts): len(batch_ids)
                for batch_ids in batch_slices
            }
    
            processed_since_flush = 0
    
            for future in as_completed(future_to_batch):
                sdf_batch = future.result()
                history_signals.append(sdf_batch)
    
                batch_len = future_to_batch[future]
                pbar.update(batch_len)
                processed_since_flush += batch_len
    
                # Show elapsed + ETA per batch
                elapsed = time.time() - start_time
                processed = pbar.n
                remaining = len(company_ids) - processed
                rate = processed / elapsed if elapsed > 0 else 0
                eta = remaining / rate if rate > 0 else 0
                mins, secs = divmod(int(elapsed), 60)
                eta_mins, eta_secs = divmod(int(eta), 60)
                tqdm.write(
                    f"       ⏱ User {user}, {tf_aligned} batch done: {processed}/{len(company_ids)}, "
                    f"elapsed {mins:02d}:{secs:02d}, ETA {eta_mins:02d}:{eta_secs:02d}"
                )
    
                # 🔄 Flush after N companies
                if processed_since_flush >= flush_size:
                    if history_signals:
                        def union_dfs(dfs):
                            while len(dfs) > 1:
                                new_dfs = []
                                for i in range(0, len(dfs), 5):
                                    chunk = dfs[i:i+5]
                                    new_dfs.append(reduce(lambda a, b: a.unionByName(b, allowMissingColumns=True), chunk))
                                dfs = new_dfs
                            return dfs[0]
    
                        history_df = union_dfs(history_signals)
    
                        run_with_logging(
                            db.write_signals,
                            icon="⏳",
                            is_subtask=True,
                            title=f"Write Candidate Lakehouse Partition: ({tf})",
                            history_df=history_df
                        )
    
                    # reset buffer after flush
                    history_signals = []
                    processed_since_flush = 0
    
    # Final flush if leftovers remain
    if history_signals:
        history_df = union_dfs(history_signals)
        run_with_logging(
            db.write_signals,
            icon="⏳",
            is_subtask=True,
            title=f"Write Candidate Lakehouse Partition: ({tf})",
            history_df=history_df
        )

# --- main loader ---
def load_signals(timeframe=None, option='full'):
    incremental = option.lower() == "incremental"

    if not incremental:
        db.clear_hive_table('bsf', 'history_signals')
        db.clear_hive_table('bsf', 'history_signals_last_all')
        db.clear_hive_table('bsf', 'history_signals_last')

    # Load full history
    sdf = spark.sql("""
        SELECT 
            t.CompanyId,
            t.StockDate,
            t.Open,
            t.High,
            t.Low,
            t.Close,
            t.Volume,
            t.FundamentalDate,
            t.PeRatio,
            t.PegRatio,
            t.PbRatio,
            t.ReturnOnEquity,
            t.GrossMarginTTM,
            t.NetProfitMarginTTM,
            t.TotalDebtToEquity,
            t.CurrentRatio,
            t.InterestCoverage,
            t.EpsChangeYear,
            t.RevChangeYear,
            t.Beta,
            t.ShortIntToFloat
        FROM (
            SELECT 
                s.CompanyId,
                s.StockDate,
                s.OpenPrice AS Open,
                s.HighPrice AS High,
                s.LowPrice  AS Low,
                s.ClosePrice AS Close,
                s.StockVolume AS Volume,
                f.FundamentalDate,
                f.PeRatio,
                f.PegRatio,
                f.PbRatio,
                f.ReturnOnEquity,
                f.GrossMarginTTM,
                f.NetProfitMarginTTM,
                f.TotalDebtToEquity,
                f.CurrentRatio,
                f.InterestCoverage,
                f.EpsChangeYear,
                f.RevChangeYear,
                f.Beta,
                f.ShortIntToFloat,
                ROW_NUMBER() OVER (
                    PARTITION BY s.CompanyId, s.StockDate
                    ORDER BY f.FundamentalDate DESC
                ) AS rn
            FROM bsf.companystockhistory s
            LEFT JOIN bsf.companyfundamental f
              ON s.CompanyId = f.CompanyId
             AND f.FundamentalDate <= s.StockDate
        ) t
        WHERE t.rn = 1
    """)

    pdf = sdf_joined.toPandas()
    sdf.unpersist()
    sdf = spark.createDataFrame(pdf)
    sdf = sdf.repartition("CompanyId").cache()
    sdf.count()  # warm-up

    users = db.get_users(engine)
    if timeframe is None:
        timeframes_items = CONFIG["timeframe_map"].items()
    else:
        timeframes_items = [(timeframe, CONFIG["timeframe_map"].get(timeframe, CONFIG["timeframe_map"]["Daily"]))]

    # Auto batch size based on Spark cores
    company_ids = [row.CompanyId for row in sdf.select("CompanyId").distinct().toLocalIterator()]
    total_companies = len(company_ids)
    cores = spark.sparkContext.defaultParallelism
    batch_size = max(1, total_companies // (cores * 2))
    max_batch_threads = cores  # threads per user × timeframe

    tqdm.write(f"Detected {cores} cores, total companies {total_companies}, batch_size {batch_size}")

    # Run in parallel per user × timeframe
    tasks = []
    start_total = time.time()
    with ThreadPoolExecutor(max_workers=cores) as executor:
        for user in users:
            for tf, tf_window in timeframes_items:
                tasks.append(executor.submit(
                    process_user_timeframe, user, tf, tf_window, sdf, ingest_ts, batch_size, max_batch_threads
                ))

        for future in as_completed(tasks):
            future.result()  # raise exceptions if any

    total_elapsed = time.time() - start_total
    mins, secs = divmod(int(total_elapsed), 60)
    tqdm.write(f"✅ Finished all users × timeframes in {mins:02d}:{secs:02d}")

    sdf.unpersist()


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



