# bsf_db_utilities.py
import os
import sys
import math
import shutil
from datetime import datetime

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import lit, col
from pyspark.sql.types import DoubleType
from pyspark.sql.utils import AnalysisException

from delta.tables import DeltaTable


class DBUtils:
    def __init__(self, spark, ingest_ts):
        self.spark = spark
        self.ingest_ts = ingest_ts

    def get_table_stats(self, table_name: str) -> dict:
            """
            Returns row count and approximate size (in MB) for a Hive-enabled Delta table.
    
            Args:
                table_name (str): Full table name, e.g., "my_database.my_table"
    
            Returns:
                dict: {"rows": int, "size_mb": float, "location": str}
            """
            # 1️⃣ Row count
            num_rows = self.spark.sql(f"SELECT COUNT(*) AS cnt FROM {table_name}").collect()[0]['cnt']
    
            # 2️⃣ Table location
            desc = self.spark.sql(f"DESCRIBE FORMATTED {table_name}").collect()
            location_row = next(row for row in desc if row['col_name'].strip() == 'Location')
            table_location = location_row['data_type'].strip()
    
            # 3️⃣ Approximate size on disk
            def get_size(path):
                total_size = 0
                for dirpath, dirnames, filenames in os.walk(path):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        total_size += os.path.getsize(fp)
                return total_size
    
            size_bytes = get_size(table_location)
            size_mb = size_bytes / (1024 * 1024)
    
            return {"rows": num_rows, "size_mb": size_mb, "location": table_location}

    
    def clear_hive_table(self, db: str, table: str):
        fqtn = f"{db}.{table}"
        print(f"     🧹 Dropping table: {fqtn}")
    
        try:
            # Fetch table details before dropping
            details = self.spark.sql(f"DESCRIBE EXTENDED {fqtn}")
            desc = {row.col_name: row.data_type for row in details.collect()}
            location = desc.get("Location", "").replace("file:", "")
        except AnalysisException:
            location = None
    
        # Drop the Hive table (catalog + files if managed)
        self.spark.sql(f"DROP TABLE IF EXISTS {fqtn}")
    
        # As a safeguard: if location still exists, remove manually
        if location and os.path.exists(location):
            shutil.rmtree(location, ignore_errors=True)
            print(f"     🗑️ Removed leftover files at {location}")
        else:
            print(f"     ✅ Table {fqtn} dropped (no leftover files found)")

    
    def spark_stats(self):
        """
        Print Spark configuration and runtime info using a single SparkConf object.
        """
  
        # Get Spark config once
        conf = self.spark.sparkContext.getConf()
    
        # ----------------------------
        # Basic Spark & environment info
        # ----------------------------
        print(f"       📋 spark.app.name : {conf.get('spark.app.name', 'Not set')}")
        print(f"       📋 Spark Version  : {self.spark.version}")
        print(f"       📋 Python Version : {sys.version.split()[0]}")
        print(f"       📋 User           : {os.getenv('USER', 'Unknown')}")
        print(f"       📋 Scheduler Pool : {conf.get('spark.scheduler.pool', 'default')}")
        #print(f"       ⚡️ spark.master                    : {conf.get('spark.master', 'Not set')}")
        #print(f"       ⚡️ spark.sql.catalogImplementation : {conf.get('spark.sql.catalogImplementation', 'Not set')}")
        #print(f"       ⚡️ spark.sql.catalog.spark_catalog : {conf.get('spark.sql.catalog.spark_catalog', 'Not set')}")
        #print(f"       ⚡️ spark.sql.warehouse.dir         : {conf.get('spark.sql.warehouse.dir', 'Not set')}")
        #print(f"       ⚡️ spark.delta.basePath            : {conf.get('spark.delta.basePath', 'Not set')}")
        #print(f"       ⚡️ spark.sql.filesource.path       : {conf.get('spark.sql.filesource.path', 'Not set')}")
        #print(f"       ⚡️ spark.nond2rd.path              : {conf.get('spark.nond2rd.path', 'Not set')}")
        #print(f"       ⚡️ spark.databricks.delta.retentionDurationCheck.enabled : {conf.get('spark.databricks.delta.retentionDurationCheck.enabled', 'false')}")
        #print(f"       ⚡️ spark.databricks.delta.logStore.class                 : {conf.get('spark.databricks.delta.logStore.class', 'Not set')}")

        # ----------------------------
        # Runtime Spark config check   
        # ----------------------------
        print(f"       🔍 === Spark Runtime Config Check ===")
        print(f"       📋 Max Cores                    : {conf.get('spark.cores.max', 'Not set')}")
        print(f"       📋 Executor Instances           : {conf.get('spark.executor.instances', 'Not set')}")
        print(f"       📋 Executor Cores               : {conf.get('spark.executor.cores', 'Not set')}")
        print(f"       📋 Task Cous                    : {conf.get('spark.task.cpus', 'Not set')}")
        print(f"       📋 Executor Memory              : {conf.get('spark.executor.memory', 'Not set')}")
        print(f"       📋 Driver Memory                : {conf.get('spark.driver.memory', 'Not set')}")
        print(f"       📋 JVM Memory Overhead          : {conf.get('spark.executor.memoryOverhead', 'Not set')}")
        print(f"       📋 Dynamic Allocation Enabled   : {conf.get('spark.dynamicAllocation.enabled', 'Not set')}")
        print(f"       📋 Default Parallelism          : {conf.get('spark.default.parallelism', 'Not set')}")
        print(f"       📋 SQL Shuffle Partitions       : {conf.get('spark.sql.shuffle.partitions', 'Not set')}")
        '''
        # ----------------------------
        # Active executors
        # ----------------------------
        # Get executor memory status as Python dict
        executor_status = spark.sparkContext.getExecutorMemoryStatus()  # {executorId: (maxMem, remainingMem)}
        
        print("       ⚡️ Active Executors & Memory Status:")
        for exec_id, (max_mem, remaining_mem) in executor_status.items():
            print(f"        ⚡️ Executor ID: {exec_id}, Max Memory: {max_mem / 1024**2:.1f} MB, Remaining Memory: {remaining_mem / 1024**2:.1f} MB")
        
        # Number of active executors (includes driver)
        print(f"       ⚡️ Total Executors (including driver): {len(executor_status)}")
        '''

        
    def db_stats(self, db_name: str):
        print(f"     📋 Stats Hive Database: {db_name}")

        '''
        # Show all databases in Hive - Using Spark SQL
        self.spark.sql("SHOW DATABASES").show(truncate=False)
        '''
        # List all databases using Spark catalog - Using Catalog API
        db_list = self.spark.catalog.listDatabases()
                 
        # Display databases
        db_found = False
        for db in db_list:
            if db.name == db_name:
                db_found = True
                print(f"     ✅ Database Name: {db.name}, Location: {db.locationUri}")
                
        if not db_found:
            print(f"     ❗ Database Name: {db_name} was not found")
            

    def write_delta_table(self, df, table_name, partition_cols, partition_filter=None, cast_columns=None):
        """
        Write a Pandas DataFrame to a Delta table with optional numeric casting.
        
        Parameters:
            df (pd.DataFrame): Input DataFrame
            table_name (str): Delta table name
            partition_cols (list): Columns to partition by
            partition_filter (str, optional): replaceWhere filter for append
            cast_columns (list, optional): Columns to cast to DoubleType; skip if None or empty
        """
        if df is None or len(df) == 0:
            print(f"      ❗ Skipping write: No data for {table_name}")
            return
    
        # Create Spark DataFrame
        spark_df = self.spark.createDataFrame(df)
    
        # Conditionally cast columns to DoubleType
        if cast_columns:
            for c in cast_columns:
                if c in spark_df.columns:
                    spark_df = spark_df.withColumn(c, col(c).cast(DoubleType()))
    
        # Smart partitioning based on DataFrame size in memory
        num_rows = df.shape[0]
        mem_mb = df.memory_usage(index=True, deep=True).sum() / (1024 * 1024)
        target_partition_size_mb = 64
        num_partitions = max(1, int(mem_mb / target_partition_size_mb))
    
        if num_partitions > 1:
            if partition_cols:
                spark_df = spark_df.repartition(num_partitions, *partition_cols)
            else:
                spark_df = spark_df.repartition(num_partitions)
    
        print(f"      📤 Writing to Delta table: {table_name} | Rows={num_rows:,} | Size={mem_mb:.1f} MB | Partitions={num_partitions}")
    
        # Check if table exists
        try:
            self.spark.sql(f"DESCRIBE TABLE {table_name}")
            table_exists = True
        except:
            table_exists = False
    
        write_mode = "append" if table_exists and partition_filter else "overwrite"
        options = {"mergeSchema": "false"}
        if write_mode == "append" and partition_filter:
            options["replaceWhere"] = partition_filter
    
        # Write
        writer = spark_df.write.format("delta").mode(write_mode)
        if partition_cols:
            writer = writer.partitionBy(*partition_cols)
        for k, v in options.items():
            writer = writer.option(k, v)
    
        writer.saveAsTable(table_name)
        print(f"      ✅ Completed write to {table_name}")

    from delta.tables import DeltaTable

    

    def merge_delta_table(
        self, df, table_name, merge_keys, 
        company_col="CompanyId", timeframe_col="TimeFrame",
        target_partition_mb=64, overwrite_partition=False
    ):
        """
        Merge a Pandas DataFrame into a Delta table, or overwrite partitions.
    
        Modes:
          - Merge (default): Upsert rows into existing Delta table.
          - Overwrite partition: Drop & replace only the partitions in `df`.
          
        Strategy:
          - Physically partition only by `timeframe_col` (fewer, larger partitions).
          - Use `repartition(N, company_col)` before write to distribute rows evenly.
          - Target ~64 MB parquet file sizes.
        """
    
        # --- Convert Pandas → Spark ---
        sdf = self.spark.createDataFrame(df)
    
        # --- Estimate DataFrame size and rows ---
        mem_bytes = df.memory_usage(index=True, deep=True).sum()
        mem_mb = mem_bytes / (1024 * 1024)
        total_rows = df.shape[0]
    
        # --- Decide partition count ---
        total_cores = self.spark._jsc.sc().defaultParallelism()
        target_partitions = max(
            math.ceil(mem_mb / target_partition_mb),  # ~64 MB chunks
            total_cores                              # at least 1 per core
        )
        target_partitions = min(target_partitions, total_cores * 4)  # cap at 4× cores
    
        # Repartition by company (logical balance), but not physical partitioning
        sdf = sdf.repartition(target_partitions, sdf[company_col])
    
        print(
            f"      📤 Writing to Delta table: {table_name} | "
            f"            Rows={total_rows:,} | Size={mem_mb:.1f} MB | "
            f"            Partitions={sdf.rdd.getNumPartitions()} | "
            f"            Mode={'overwrite_partition' if overwrite_partition else 'merge'}"
        )
    
        # --- Write logic ---
        if self.spark.catalog.tableExists(table_name):
            if overwrite_partition:
                # Extract distinct partitions to overwrite
                partitions = sdf.select(timeframe_col).distinct().collect()
                part_values = [row[timeframe_col] for row in partitions]
    
                print(f"      🔄 Overwriting partitions: {part_values}")
    
                (
                    sdf.write.format("delta")
                        .mode("overwrite")
                        .option("replaceWhere", f"{timeframe_col} IN ({','.join([repr(p) for p in part_values])})")
                        .saveAsTable(table_name)
                )
            else:
                # Merge (upsert)
                target = DeltaTable.forName(self.spark, table_name)
                cond = " AND ".join([f"t.{k} = s.{k}" for k in merge_keys])
                (
                    target.alias("t")
                          .merge(sdf.alias("s"), cond)
                          .whenMatchedUpdateAll()
                          .whenNotMatchedInsertAll()
                          .execute()
                )
        else:
            # First load → physically partition by TimeFrame only
            (
                sdf.write.format("delta")
                    .mode("overwrite")
                    .partitionBy(timeframe_col)
                    .saveAsTable(table_name)
            )
    


    
    def write_history(self, pdf_chunk, overwrite_table: bool = False, show_stats: bool = False):
        # Convert to Spark
        sdf_chunk = self.spark.createDataFrame(pdf_chunk.copy())
    
        if show_stats:
            try:
                est_bytes = sdf_chunk.rdd.map(lambda x: len(str(x))).sum()
                est_mib = round(est_bytes / (1024 * 1024), 2)
                print(f"        🧠 Estimated memory footprint: {est_mib} MiB")
            except Exception as e:
                print(f"        ❗ Memory estimation failed: {e}")
    
        # -------------------------------
        # Write main history table
        # -------------------------------
        sdf_chunk.write.format("delta") \
            .mode("overwrite" if overwrite_table else "append") \
            .partitionBy("CompanyId") \
            .saveAsTable("bsf.companystockhistory")
        
        # -------------------------------
        # Write watermark table
        # -------------------------------   
        watermark_update_df = self.spark.createDataFrame(
            pdf_chunk.groupby("CompanyId")["StockDate"].max().reset_index()
        ).withColumnRenamed("StockDate", "LastLoadedDate")
        
        watermark_table = "bsf.companystockhistory_watermark"
        
        # Check if watermark table exists
        if self.spark.catalog.tableExists(watermark_table):
            existing_wm = self.spark.table(watermark_table)
            merged_wm = existing_wm.union(watermark_update_df)
        else:
            merged_wm = watermark_update_df
        
        # Always collapse to max per CompanyId
        merged_wm = merged_wm.groupBy("CompanyId") \
            .agg(F.max("LastLoadedDate").alias("LastLoadedDate"))
        
        # Overwrite watermark table
        merged_wm.write.format("delta").mode("overwrite").saveAsTable(watermark_table)



         
    def write_candidates(self, history_df=None, timeframe=None):
        if history_df is None or len(history_df) == 0:
            print("⚠️ No history data provided. Skipping write.")
            return
    
        # -------------------------------
        # Columns to keep for research
        # -------------------------------
        good_cols = [
            "CompanyId", "StockDate", "Open", "High", "Low", "Close", "TomorrowClose", "Return", "TomorrowReturn",
            "Doji", "Hammer", "InvertedHammer", "ShootingStar", "BullishEngulfing", "BearishEngulfing", "PiercingLine",
            "DarkCloudCover", "MorningStar", "EveningStar", "ThreeWhiteSoldiers", "ThreeBlackCrows", "TweezerTop",
            "TweezerBottom", "InsideBar", "OutsideBar", "MA", "MA_slope", "UpTrend_MA", "DownTrend_MA", "MomentumUp",
            "MomentumDown", "ConfirmedUpTrend", "ConfirmedDownTrend", "RecentReturn", "UpTrend_Return",
            "DownTrend_Return", "Volatility", "LowVolatility", "HighVolatility", "ROC", "MomentumZ", "SignalStrength",
            "SignalStrengthHybrid", "ActionConfidence", "BullishStrengthHybrid", "BearishStrengthHybrid",
            "SignalDuration", "ValidAction", "HasValidSignal", "MomentumAction", "PatternAction", "CandleAction",
            "CandidateAction", "Action", "TomorrowAction", "TomorrowActionSource", "BatchId", "IngestedAt", "TimeFrame"
        ]

        latest_df = history_df.groupby("CompanyId").tail(1).copy()

        self.merge_delta_table(history_df, "bsf.history_signals_allcol", ["CompanyId", "StockDate", "TimeFrame"])
        self.merge_delta_table(history_df[good_cols], "bsf.history_signals", ["CompanyId", "StockDate", "TimeFrame"])
        
        self.merge_delta_table(latest_df, "bsf.history_signals_allcol_last", ["CompanyId", "StockDate", "TimeFrame"],overwrite_partition=True)
        self.merge_delta_table(latest_df[good_cols], "bsf.history_signals_last", ["CompanyId", "StockDate", "TimeFrame"],overwrite_partition=True)

       
    def optimize_table(self, ingest_ts: str = None):
        query = """
            OPTIMIZE bsf.companystockhistory
            ZORDER BY (StockDate)
        """
        print(f"      📁 ZORDER on StockDate entire table")   
        self.spark.sql(query)
        print("      ✅ OPTIMIZE/ZORDER Completed on StockDate: bsf.companystockhistory")
