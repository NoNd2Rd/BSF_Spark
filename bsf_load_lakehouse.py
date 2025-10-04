import argparse
from datetime import datetime
import time
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import reduce

# Spark imports
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark import StorageLevel
from pyspark.sql.functions import broadcast

# Custom imports
#from bsf_config import CONFIG
from bsf_settings import load_settings
from bsf_env import init_spark, init_mariadb_engine
from bsf_dbutilities import DBUtils
from bsf_candlesticks_simpler import (
    step1_add_candle_patterns,
    step1_add_candle_patterns_dynamic,
    step2_add_trend_filters_optimized,
    step3_add_confirmed_signals_optimized,
    step4_compute_fundamental_score_optimized,
    step5_finalize_signals,
    step6_add_signal_strength_vectorized,
    add_batch_metadata_optimized

)
from bsf_candidates import phase_1, phase_2, phase_3

# Global variables
spark = None
engine = None
ingest_ts = None
db = None

def format_elapsed(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    if hours > 0:
        return f"{hours}h {minutes}m"
    else:
        return f"{minutes}m"



def run_with_logging(func, icon="⏳", is_subtask=False, title=None, *args, **kwargs):
    bold_title = f" \033[1m** {title} **\033[0m" if title else ''
    prefix = "  " if is_subtask else "\n"
    print(f"{prefix}{icon} {func.__name__} is running{bold_title}...")
    
    start_time = time.time()
    try:
        result = func(*args, **kwargs)

        print(f"{prefix}✅ {func.__name__} completed in {format_elapsed(time.time() - start_time)}")
        return result
    except Exception as e:
        print(f"{prefix}❌ {func.__name__} failed: {str(e)}")
        raise

def prepare_lakehouse_environment(mode: str = None, option: str = None, db_name: str = "bsf"):
    global spark, engine, ingest_ts, db
    process_option = {
        'signals': 'wide',
        'history': 'tall',
        'candidates': 'wide'
    }.get(mode, 'default')
    
    spark = init_spark(f"{db_name}_{mode}_{option}", log_level="ERROR", show_progress=False, 
                      enable_ui=True, process_option=process_option)
    engine = init_mariadb_engine()
    ingest_ts = spark.sql("SELECT current_timestamp()").collect()[0][0]
    print(f"    ⚡️ Spark Session Initialized \033[1m** {db_name} {mode} >>> {option} **\033[0m")
    
    db = DBUtils(spark, ingest_ts)
    db.spark_stats(True if mode=='history' else False)

def load_company(chunk_size=2500):
    db.clear_hive_table('bsf', 'company')
    cmp_query = "SELECT * FROM company WHERE ListingExchange IN (1,2,3,16) AND CompanyId = 52"
    pdf_iterator = pd.read_sql(cmp_query, engine, chunksize=chunk_size)
    batch_list = []
    for chunk in tqdm(pdf_iterator, desc="    Processing Company chunks"):
        batch_list.append(chunk)
        if len(batch_list) == 10:
            pdf_batch = pd.concat(batch_list, ignore_index=True)
            db.write_company(pdf_batch, show_stats=False)
            batch_list = []
    
    # Write any remaining chunks
    if batch_list:
        pdf_batch = pd.concat(batch_list, ignore_index=True)
        db.write_company(pdf_batch, show_stats=False)


    print(f"      ✔️ Company table written to Delta lakehouse", flush=True)

def load_history(option='full', chunk_size=10000):
    incremental = option.lower() == "incremental"
    
    if not incremental:
        for table in ['companystockhistory', 'companyfundamental', 'companystockhistory_watermark']:
            db.clear_hive_table('bsf', table)
    


    # ============================================
    # Build Company List
    # ============================================
    company_ids = spark.sql("""
        SELECT DISTINCT CompanyId
        FROM bsf.company
        WHERE ListingExchange IN (1,2,3,16)
          AND Active = 1
          AND LastClose BETWEEN 0.0001 AND 0.1
          AND LastHistoryDate >= date_sub(current_date(), 30)
    """).rdd.flatMap(lambda x: x).collect()

    # Ensure list of ints (safe for Python loops)
    company_ids = list(map(int, company_ids))
    
    # Pre-format string for SQL "IN (...)"
    company_list = f"({','.join(map(str, company_ids))})"
    
    lookback_days = 425
    data_written = False
    # ============================================
    # Build Incremental Date Condition
    # ============================================
    if incremental:
        try:
            wm_df = spark.sql(f"""
                SELECT CompanyId, LastLoadedDate
                FROM bsf.companystockhistory_watermark
                WHERE CompanyId IN {company_list}
            """).toPandas()
            wm_dict = dict(zip(wm_df.CompanyId, wm_df.LastLoadedDate))
        except Exception:
            wm_dict = {}
    
        def normalize_date(val):
            if val is None:
                return None
            return str(val).split(" ")[0]  # keep only YYYY-MM-DD
    
        date_conditions = []
        for cid in company_ids:   # ✅ iterate the list, not the string
            wm_date = normalize_date(wm_dict.get(cid))
            if wm_date:
                date_conditions.append(f"(csh.CompanyId={cid} AND csh.StockDate > '{wm_date}')")
            else:
                date_conditions.append(
                    f"(csh.CompanyId={cid} AND csh.StockDate >= DATE_SUB(CURDATE(), INTERVAL {lookback_days} DAY))"
                )
    
        date_condition = " OR ".join(date_conditions)
    
    else:
        # Non-incremental fallback
        date_condition = f"""
            csh.StockDate >= DATE_SUB(CURDATE(), INTERVAL {lookback_days} DAY)
            AND csh.CompanyId IN {company_list}
        """

    
    hist_query = f"""
        SELECT CompanyId, StockDate, OpenPrice AS Open, HighPrice AS High,
               LowPrice AS Low, ClosePrice AS Close, StockVolume AS Volume
        FROM companystockhistory csh
        WHERE {date_condition}
        ORDER BY CompanyId, StockDate
    """

    fund_query = f"""
        SELECT CompanyId, FundamentalDate, PeRatio, PegRatio, PbRatio, ReturnOnEquity,
               GrossMarginTTM, NetProfitMarginTTM, TotalDebtToEquity, CurrentRatio,
               InterestCoverage, EpsChangeYear, RevChangeYear, Beta, ShortIntToFloat
        FROM companyfundamental
        WHERE CompanyId IN {company_list}
    """
    #print (hist_query)
    pdf_iterator = pd.read_sql(hist_query, engine, chunksize=chunk_size)
    batch_list = []
    for chunk in tqdm(pdf_iterator, desc="    Processing History chunks"):
        batch_list.append(chunk)
        if len(batch_list) == 10:
            pdf_batch = pd.concat(batch_list, ignore_index=True)
            db.write_history(pdf_batch, show_stats=True)
            data_written = True
            batch_list = []
    
    # Write any remaining chunks
    if batch_list:
        pdf_batch = pd.concat(batch_list, ignore_index=True)
        if pdf_batch.empty:
            if not incremental and not data_written:
                print("🚫 Aborting write_history: empty history batch")
                return   # exit the whole function here
            elif incremental and not data_written:
                print("🚫 Skip write_history: empty fundamental batch")
                return   # exit the whole function here
        else:
            db.write_history(pdf_batch, show_stats=True)
    print(f"    ✔️ History table written to Delta lakehouse", flush=True)
    
    #pdf_hist = pd.read_sql(hist_query, engine)
    #db.write_history(pdf_hist, show_stats=False)
    data_written = False
    pdf_iterator = pd.read_sql(fund_query, engine, chunksize=chunk_size)
    batch_list = []
    for chunk in tqdm(pdf_iterator, desc="    Processing Fundamental chunks"):
        batch_list.append(chunk)
        if len(batch_list) == 10:
            pdf_batch = pd.concat(batch_list, ignore_index=True)
            db.write_fundamental(pdf_batch, show_stats=False)
            data_written = True
            batch_list = []
    
    # Write any remaining chunks
    if batch_list:
        pdf_batch = pd.concat(batch_list, ignore_index=True)
        if pdf_batch.empty:
            if not incremental and not data_written:
                print("🚫 Aborting write_fundamental: empty fundamental batch")
                return   # exit the whole function here
            elif incremental and not data_written:
                print("🚫 Skip write_fundamental: empty fundamental batch")
                #return   # exit the whole function here
        else:
            db.write_fundamental(pdf_batch, show_stats=True)
    print(f"    ✔️ Fundamental table written to Delta lakehouse", flush=True)
    
    #pdf_fund = pd.read_sql(fund_query, engine)
    #db.write_fundamental(pdf_fund, show_stats=False)

    print("    ✔️ Processing Signal Driver table")
    table ='history_signal_driver'
    db.clear_hive_table('bsf',table)
    sdf_stock = spark.table("bsf.companystockhistory").alias("s").persist(StorageLevel.MEMORY_AND_DISK)
    sdf_fund = spark.table("bsf.companyfundamental").alias("f").persist(StorageLevel.MEMORY_AND_DISK)


    
    s = sdf_stock.alias("s")
    f = sdf_fund.alias("f")
    
    join_cond = (F.col("s.CompanyId") == F.col("f.CompanyId")) & (F.col("f.FundamentalDate") <= F.col("s.StockDate"))
    sdf_joined = s.join(broadcast(f), join_cond, "left_outer")
    
    w = Window.partitionBy("s.CompanyId", "s.StockDate").orderBy(F.col("f.FundamentalDate").desc())
    
    fundamental_cols = [
        "PeRatio","PegRatio","PbRatio","ReturnOnEquity","GrossMarginTTM",
        "NetProfitMarginTTM","TotalDebtToEquity","CurrentRatio","InterestCoverage",
        "EpsChangeYear","RevChangeYear","Beta","ShortIntToFloat"
    ]
    
    sdf_all = (
        sdf_joined
        .withColumn("rn", F.row_number().over(w))
        .filter(F.col("rn") == 1)
        .drop("rn")
        .select(
            F.col("s.CompanyId"),
            F.col("s.StockDate"),
            F.col("s.Open").alias("Open"),
            F.col("s.High").alias("High"),
            F.col("s.Low").alias("Low"),
            F.col("s.Close").alias("Close"),
            F.col("s.Volume"),
            F.col("f.FundamentalDate"),
            *[F.col(f"f.{c}") for c in fundamental_cols]
        )
    )


    if sdf_all.rdd.isEmpty():
        print("❗ Signal Driver is empty.")
    else:
        db.write_signal_driver(sdf_all, show_stats=True)
        
    sdf_stock.unpersist()
    sdf_fund.unpersist()
    print("    ✔️ Signal tables written to Delta lakehouse")

def load_signals(batch_size=1000):
    for table in ['history_signals', 'history_signals_last_all', 'history_signals_last']:
        db.clear_hive_table('bsf', table)

    keep_cols = [
        "UserId", "Profile", "CompanyId", "StockDate", "Open", "High", "Low", "Close", "TomorrowClose",
        "Return", "TomorrowReturn", "MA", "MA_slope", "UpTrend_MA", "DownTrend_MA",
        "MomentumUp", "MomentumDown", "ConfirmedUpTrend", "ConfirmedDownTrend", "Volatility",
        "LowVolatility", "HighVolatility", "SignalStrength", "SignalStrengthHybrid", "ActionConfidence",
        "BullishStrengthHybrid", "BearishStrengthHybrid", "SignalDuration", "PatternAction",
        "CandleAction", "UpTrend_Return", "CandidateAction", "Action", "TomorrowAction", "TimeFrame"
    ]

    df_all = spark.table("bsf.history_signal_driver").toPandas()
    users = db.get_users(engine)
     
    def process_company(cid, user, profile, tf, tf_window):
        df_company = df_all[df_all["CompanyId"] == cid].copy().sort_values("StockDate")
        df_tf = (
            df_company
            .pipe(add_candle_patterns_optimized, tf_window=tf_window, profile=profile)
            .pipe(add_trend_filters_optimized, timeframe=tf, profile=profile)
            .pipe(add_confirmed_signals_optimized)
            .pipe(compute_fundamental_score_optimized, profile=profile)
            .pipe(finalize_signals_optimized, tf=tf, tf_window=tf_window, use_fundamentals=True, profile=profile)
            .pipe(add_signal_strength_optimized)
            .pipe(add_batch_metadata_optimized, timeframe=tf, user=user, profile=profile, ingest_ts=ingest_ts)
        )
        return df_tf[keep_cols]

    company_ids = df_all["CompanyId"].unique()

    for user in users:
        user_id = user["UserId"]
        profile = user["TemplateProfile"]
        username = user["UserName"]
        timeframes_items = load_settings(str(profile))["timeframe_map"].items()
        
        user_start = time.time()  # Track total time per user
  
        for tf, tf_window in timeframes_items:
            tf_start = time.time()  # Track total time per user
            print(f"🔄 Processing {tf:<6} for user {user_id} - profile {profile.capitalize()} ...")
            df_list = []
            #raise RuntimeError("⚠️ This notebook is blocked. Do NOT run all cells without checking!")
            # Process companies in parallel
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(process_company, cid, user_id, profile, tf, tf_window) for cid in company_ids]

                # Collect results as they complete
                for future in tqdm(as_completed(futures), total=len(futures), desc="    🔄 Processing companies"):
                    df_list.append(future.result())

            # Concatenate and write one DataFrame per user × timeframe
            if df_list:
                df_final = pd.concat(df_list, ignore_index=True)
                run_with_logging(
                    db.write_signals,
                    icon="⏳",
                    is_subtask=True,
                    title=f"Write Candidate Lakehouse Partition: ({tf})",
                    df=df_final
                )
                print(f"✅ Signals written for {tf} / user {user_id}")

            print(f"⏱️ Time for user {user_id} / timeframe {tf} in {format_elapsed(time.time() - tf_start)}")

        print(f"⏱️ Total time for user for user {user_id} in {format_elapsed(time.time() - user_start)}")

    print("✅ All signals processed.")

def load_candlesticks(batch_size=1000):
    for table in ['history_signals', 'history_signals_last_all', 'history_signals_last']:
        db.clear_hive_table('bsf', table)

    keep_cols = [
        "UserId", "Profile", "CompanyId", "StockDate", "Open", "High", "Low", "Close", "TomorrowClose",
        "Return", "TomorrowReturn", "MA", "MA_slope", "UpTrend_MA", "DownTrend_MA",
        "MomentumUp", "MomentumDown", "ConfirmedUpTrend", "ConfirmedDownTrend", "Volatility",
        "LowVolatility", "HighVolatility", "SignalStrength", "SignalStrengthHybrid", "ActionConfidence",
        "BullishStrengthHybrid", "BearishStrengthHybrid", "SignalDuration", "PatternAction",
        "CandleAction", "UpTrend_Return", "CandidateAction", "Action", "TomorrowAction", "TimeFrame"
    ]

    df_all = spark.table("bsf.history_signal_driver").toPandas()
    users = db.get_users(engine)
     
    def process_company(cid, user, profile, tf, tf_window):
        df_company = df_all[df_all["CompanyId"] == cid].copy().sort_values("StockDate")
        df_tf = (
            df_company
            .pipe(step1_add_candle_patterns, tf_window=tf_window, profile=profile)
            .pipe(step2_add_trend_filters_optimized, timeframe=tf, profile=profile)
            .pipe(step3_add_confirmed_signals_optimized)
            .pipe(step4_compute_fundamental_score_optimized, profile=profile)
            .pipe(step5_finalize_signals, tf=tf, tf_window=tf_window, use_fundamentals=True, profile=profile)
            .pipe(step6_add_signal_strength_vectorized)
            .pipe(add_batch_metadata_optimized, timeframe=tf, user=user, profile=profile, ingest_ts=ingest_ts)
        )
        return df_tf[keep_cols]

    company_ids = df_all["CompanyId"].unique()

    for user in users:
        user_id = user["UserId"]
        profile = user["TemplateProfile"]
        username = user["UserName"]
        timeframes_items = load_settings(str(profile))["timeframe_map"].items()
        
        user_start = time.time()  # Track total time per user
  
        for tf, tf_window in timeframes_items:
            tf_start = time.time()  # Track total time per user
            print(f"🔄 Processing {tf:<6} for user {user_id} - profile {profile.capitalize()} ...")
            #df_list = []
            #raise RuntimeError("⚠️ This notebook is blocked. Do NOT run all cells without checking!")
            '''
            # Process companies in parallel
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(process_company, cid, user_id, profile, tf, tf_window) for cid in company_ids]

                # Collect results as they complete
                for future in tqdm(as_completed(futures), total=len(futures), desc="    🔄 Processing companies"):
                    df_list.append(future.result())
            '''
            df_all_companies = df_all.sort_values(["CompanyId", "StockDate"]).copy()

            df_tf = (
                df_all_companies
                .pipe(step1_add_candle_patterns, tf_window=tf_window, profile=profile)
                .pipe(step2_add_trend_filters_optimized, timeframe=tf, profile=profile)
                .pipe(step3_add_confirmed_signals_optimized)
                .pipe(step4_compute_fundamental_score_optimized, profile=profile)
                .pipe(step5_finalize_signals, tf=tf, tf_window=tf_window, use_fundamentals=True, profile=profile)
                .pipe(step6_add_signal_strength_vectorized)
                .pipe(add_batch_metadata_optimized, timeframe=tf, user=user, profile=profile, ingest_ts=ingest_ts)
            )
            # return df_tf[keep_col
            
            # Concatenate and write one DataFrame per user × timeframe
            if not df_tf.empty:
                #df_final = pd.concat(df_list, ignore_index=True)
                run_with_logging(
                    db.write_signals,
                    icon="⏳",
                    is_subtask=True,
                    title=f"Write Candidate Lakehouse Partition: ({tf})",
                    df=df_final
                )
                print(f"✅ Signals written for {tf} / user {user_id}")

            print(f"⏱️ Time for user {user_id} / timeframe {tf} in {format_elapsed(time.time() - tf_start)}")

        print(f"⏱️ Total time for user for user {user_id} in {format_elapsed(time.time() - user_start)}")

    print("✅ All signals processed.")
    
def load_candidates():
    for table in ['final_candidates_enriched', 'final_candidates']:
        db.clear_hive_table('bsf', table)

    users = db.get_users(engine)

    total_start = time.time()

    for user in users:
        user_start = time.time()
        
        
        user_id = user["UserId"]
        profile = user["TemplateProfile"]
        username = user["UserName"]
        print(f"\n🔄 Processing user {user_id} - profile {profile.capitalize()} ...")
        
        # Load user-specific settings
        settings = load_settings(str(profile))["phases"]
        topN_phase1 = settings["phase1"]["topN"]
        topN_phase2 = settings["phase2"]["topN"]
        topN_phase3 = settings["phase3"]["topN"]

        # Load Spark tables filtered by user
        df_last = spark.table("bsf.history_signals_last").filter(F.col("UserId") == user_id)
        df_all  = spark.table("bsf.history_signals").filter(F.col("UserId") == user_id)

        # Phase 1
        timeframe_dfs_all, timeframe_dfs = phase_1(spark, profile, df_all, df_last, topN_phase1)

        # Phase 2
        phase2_topN_dfs = phase_2(spark, profile, timeframe_dfs_all, topN_phase2)

        # Phase 3
        df_phase3_enriched, df_topN_companies, phase3_enriched_dict, topN_companies_dict = phase_3(spark, profile, phase2_topN_dfs, topN_phase3)

        # Write results
        db.write_candidates(df_phase3_enriched, df_topN_companies)
        db.create_bsf(engine, user_id, username, profile, topN_companies_dict)
            

        print(f"✅ User {user_id} processed in {format_elapsed(time.time() - user_start)}")

def main(mode=None, option="full"):
    db_name = "bsf"
    run_with_logging(prepare_lakehouse_environment, "⏳", is_subtask=True, title="Prepare Lakehouse Environment", mode=mode, db_name=db_name, option=option)
    db.db_stats(db_name)

    if mode == "history":
        run_with_logging(load_company, "⏳", True, "Load Company", chunk_size=5000)
        run_with_logging(load_history, "⏳", True, f"Lakehouse History {option} Load",  option=option, chunk_size=10000)
        run_with_logging(db.optimize_table, "⏳", True, "Optimize Table")
    
    elif mode == "signals":
        run_with_logging(load_candlesticks, "⏳", True, "Load Signals",)
    
    elif mode == "candidates":
        run_with_logging(load_candidates, "⏳", True, "Load Candidates")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Lakehouse")
    parser.add_argument(
        "--mode",
        choices=["history", "signals", "candidates"],
        default="history",
        help="Process mode (history, candidates, signals)"
    )
    parser.add_argument(
        "--option",
        choices=["full", "incremental"],
        default="full",
        help="Load type option (full, incremental)"
    )

    args = parser.parse_args()
    mode = args.mode.lower()
    option = args.option.lower()

    try:
        run_with_logging(
            main,
            "🚀",
            False,
            f"Main for mode: {mode} options: {option}",
            mode=mode,
            option=option
        )
    finally:
        if spark:
            spark.stop()
        if engine:
            engine.dispose()