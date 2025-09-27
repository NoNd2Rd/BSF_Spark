# bsf_db_utilities.py
import os
from datetime import datetime

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, col
from pyspark.sql.types import DoubleType

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

        
    def delete_hive_db(self, db_name: str):
        print(f"         🧹 Dropping Hive database: {db_name}")
        self.spark.sql(f"DROP DATABASE IF EXISTS {db_name} CASCADE")
    
    def create_hive_db(self, db_name: str):
        print(f"         ✅ Create Hive database: {db_name}")
        self.spark.sql(f"CREATE DATABASE IF NOT EXISTS {db_name}")

    def spark_stats(self):
        print(f"        🔍 Stats Spark:")
         
        # Get and display default locations with description
        warehouse_dir = self.spar.kconf.get("self.sparksql.warehouse.dir", "Not set")
        delta_base_path = self.spark.conf.get("self.sparkdelta.basePath", "Not set")
        filesource_path = self.spark.conf.get("self.sparksql.filesource.path", "Not set")
        nond2rd_path = self.spark.conf.get("self.sparknond2rd.defaultpath", "Not set")
        
        # Print configuration values
        print(f"               ⚡️ self.spark.sql.warehouse.dir     : {warehouse_dir}")
        print(f"               ⚡️ self.spark.delta.basePath        : {delta_base_path}")
        print(f"               ⚡️ self.spark.sql.filesource.path   : {filesource_path}")
        print(f"               ⚡️ self.spark.nond2rd.defaultpath   : {nond2rd_path}")


    def db_stats(self, db_name: str):
        print(f"         🔍 Stats Hive Database: {db_name}")
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
                print(f"         ✅ Database Name: {db.name}, Location: {db.locationUri}")
                
        if not db_found:
            print(f"         ❗ Database Name: {db_name} was not found")
            
    def write_delta_table_bad(self, df, table_name, partition_cols, partition_filter=None):
        if df is None or len(df) == 0:
            print(f"⚠️ Skipping write: No data for {table_name}")
            return
    
        spark_df = self.spark.createDataFrame(df).coalesce(1)
        print(f"📤 Writing to Delta table: {table_name} | Rows={len(df)}")
    
        try:
            self.spark.sql(f"DESCRIBE TABLE {table_name}")
            table_exists = True
        except:
            table_exists = False
    
        write_mode = "append" if table_exists and partition_filter else "overwrite"
        options = {"mergeSchema": "false"}
        if write_mode == "append" and partition_filter:
            options["replaceWhere"] = partition_filter
    
        # 👈 use local spark_df, not self.spark_df
        writer = spark_df.write.format("delta").mode(write_mode).partitionBy(*partition_cols)
        for k, v in options.items():
            writer = writer.option(k, v)
    
        writer.saveAsTable(table_name)
        print(f"✅ Completed write to {table_name}")

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


    def write_delta_table_ok(self, df, table_name, partition_cols, partition_filter=None):
        if df is None or len(df) == 0:
            print(f"⚠️ Skipping write: No data for {table_name}")
            return
    
        # ✅ Create Spark DataFrame without coalesce(1)
        spark_df = self.spark.createDataFrame(df)

        for c in ["BodyRel", "UpperShadowRel", "LowerShadowRel", "RangeRel"]:
            if c in spark_df.columns:
                spark_df = spark_df.withColumn(c, col(c).cast(DoubleType()))
        # ✅ Smart partitioning based on DataFrame size in memory
        num_rows = df.shape[0]
        mem_mb = df.memory_usage(index=True, deep=True).sum() / (1024 * 1024)
    
        # Target max partition size ≈ 64 MB
        target_partition_size_mb = 64
        num_partitions = max(1, int(mem_mb / target_partition_size_mb))
    
        if num_partitions > 1:
            if partition_cols:
                spark_df = spark_df.repartition(num_partitions, *partition_cols)
            else:
                spark_df = spark_df.repartition(num_partitions)
    
        print(f"📤 Writing to Delta table: {table_name} | Rows={num_rows:,} | Size={mem_mb:.1f} MB | Partitions={num_partitions}")
    
        # ✅ Check if table exists
        try:
            self.spark.sql(f"DESCRIBE TABLE {table_name}")
            table_exists = True
        except:
            table_exists = False
    
        write_mode = "append" if table_exists and partition_filter else "overwrite"
        options = {"mergeSchema": "false"}
        if write_mode == "append" and partition_filter:
            options["replaceWhere"] = partition_filter
    
        # ✅ Write
        writer = spark_df.write.format("delta").mode(write_mode)
        if partition_cols:
            writer = writer.partitionBy(*partition_cols)
        for k, v in options.items():
            writer = writer.option(k, v)
    
        writer.saveAsTable(table_name)
        print(f"✅ Completed write to {table_name}")
 
    def write_history(self, pdf_chunk, overwrite_table: bool = False, show_stats: bool = False):
            # Convert to Spark
            sdf_chunk = self.spark.createDataFrame(pdf_chunk.copy())
            #sdf_chunk = sdf_chunk.withColumn("IngestTime", lit(self.ingest_ts))
            '''
            # catch 22 - if I repartion the chunks it uses memory instead of letting the write 
            # handle the partioning directly but if I do this then this helps Spark parallelize writes more efficiently.
            sdf_chunk = sdf_chunk.repartition(32, "CompanyId")
            '''
            if show_stats:
                try:
                    est_bytes = sdf_chunk.rdd.map(lambda x: len(str(x))).sum()
                    est_mib = round(est_bytes / (1024 * 1024), 2)
                    print(f"        🧠 Estimated memory footprint: {est_mib} MiB")
                except Exception as e:
                    print(f"        ❗ Memory estimation failed: {e}")
    
            sdf_chunk.write.format("delta") \
                .mode("overwrite" if overwrite_table else "append") \
                .partitionBy("CompanyId") \
                .saveAsTable("bsf.companystockhistory")
        
    def write_candidates_original(self, latest_df=None, history_df=None, timeframe=None):
    
        # 💾 Save latest_df
        latest_count = len(latest_df)
        print(f"      ❗ Writing Table bsf.final_daily_signals: Rows={latest_count}")
        self.spark.createDataFrame(latest_df).write.format("delta") \
            .mode("overwrite") \
            .partitionBy("TimeFrame") \
            .option("mergeSchema", "true") \
            .saveAsTable("bsf.final_daily_signals")
        print(f"      ✅ Completed writing delta tables for candidates <final>.")
   
        # 💾 Save history_df
        history_count = len(history_df)
        print(f"      ❗ Writing Table bsf.full_daily_signals: Rows={history_count}")
        self.spark.createDataFrame(history_df).write.format("delta") \
            .mode("overwrite") \
            .partitionBy("TimeFrame") \
            .option("mergeSchema", "true") \
            .saveAsTable("bsf.full_daily_signals")
        print(f"      ✅ Completed writing delta tables for candidates <full>.")
 
        
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
    
        # -------------------------------
        # Extract latest row per company
        # -------------------------------
        latest_df = history_df.groupby("CompanyId").tail(1).copy()
         
        # -------------------------------
        # Write full all-column table
        # -------------------------------
        self.write_delta_table(
            df=history_df.copy(),
            table_name="bsf.daily_signals_allcol",
            partition_cols=["TimeFrame", "CompanyId"],
            partition_filter=f"TimeFrame = '{timeframe}'"
        )
         
        # -------------------------------
        # Write last-row all-column table
        # -------------------------------
        self.write_delta_table(
            df=latest_df.copy(),
            table_name="bsf.daily_signals_last_allcol",
            partition_cols=["TimeFrame", "CompanyId"],
            partition_filter=f"TimeFrame = '{timeframe}'"
        )
    
        # -------------------------------
        # Write filtered research tables
        # -------------------------------
        self.write_delta_table(
            df=history_df[good_cols].copy(),
            table_name="bsf.daily_signals",
            partition_cols=["TimeFrame", "CompanyId"],
            partition_filter=f"TimeFrame = '{timeframe}'"
        )
    
        self.write_delta_table(
            df=latest_df[good_cols].copy(),
            table_name="bsf.daily_signals_last",
            partition_cols=["TimeFrame", "CompanyId"],
            partition_filter=f"TimeFrame = '{timeframe}'"
        )

    def write_candidates_old(self, latest_df=None, history_df=None, timeframe=None):
        latest_df = history_df.groupby("CompanyId").tail(1).copy()

        # -------------------------------
        # Columns to keep for research / good columns
        # -------------------------------
        good_cols = [
            "CompanyId", "StockDate",
            "Open", "High", "Low", "Close", "TomorrowClose", "Return", "TomorrowReturn",
            "Doji", "Hammer", "InvertedHammer", "ShootingStar",
            "BullishEngulfing", "BearishEngulfing",
            "PiercingLine", "DarkCloudCover", "MorningStar", "EveningStar",
            "ThreeWhiteSoldiers", "ThreeBlackCrows", "TweezerTop", "TweezerBottom",
            "InsideBar", "OutsideBar",
            "MA", "MA_slope", "UpTrend_MA", "DownTrend_MA",
            "MomentumUp", "MomentumDown", "ConfirmedUpTrend", "ConfirmedDownTrend",
            "RecentReturn", "UpTrend_Return", "DownTrend_Return",
            "Volatility", "LowVolatility", "HighVolatility", "ROC", "MomentumZ",
            "SignalStrength", "SignalStrengthHybrid", "ActionConfidence",
            "BullishStrengthHybrid", "BearishStrengthHybrid", "SignalDuration",
            "ValidAction", "HasValidSignal",
            "MomentumAction", "PatternAction", "CandleAction",
            "CandidateAction", "Action", "TomorrowAction", "TomorrowActionSource",
            "BatchId", "IngestedAt", "TimeFrame"
        ]
        cast_columns=["BodyRel", "UpperShadowRel", "LowerShadowRel", "RangeRel"]
        # -------------------------------
        # Write full tables
        # -------------------------------
        if latest_df is not None and len(latest_df) > 0:
            latest_count = len(latest_df)
            print(f"      ❗ Writing FULL Table bsf.daily_signals_last_allcol: Rows={latest_count}")
            #self.spark.createDataFrame(latest_df).write.format("delta") \
            #    .mode("overwrite") \
            #    .partitionBy("TimeFrame") \
            #    .option("mergeSchema", "false") \
            #    .saveAsTable("bsf.daily_signals_last_allcol")
            #print(f"      ✅ Completed writing FULL delta table for latest candidates.")
            write_delta_table(
                spark=self.spark,
                df=latest_df.copy(),
                table_name="bsf.daily_signals_last_allcol",
                partition_col="TimeFrame",
                partition_value=timeframe,
                cast_columns=cast_columns
            )
        if history_df is not None and len(history_df) > 0:
            history_count = len(history_df)
            print(f"      ❗ Writing FULL Table bsf.daily_signals_allcol: Rows={history_count}")
            #self.spark.createDataFrame(history_df).write.format("delta") \
            #    .mode("overwrite") \
            #    .partitionBy("TimeFrame") \
            #    .option("mergeSchema", "false") \
            #    .saveAsTable("bsf.daily_signals_allcol")
            #print(f"      ✅ Completed writing FULL delta table for history.")
            write_delta_table(
                spark=self.spark,
                df=history_df.copy(),
                table_name="bsf.daily_signals_allcol",
                partition_col="TimeFrame",
                partition_value=timeframe,
                cast_columns=cast_columns
            )    
        # -------------------------------
        # Write filtered "research" tables
        # -------------------------------
        if latest_df is not None and len(latest_df) > 0:
            latest_filtered = latest_df[good_cols].copy()
            latest_count = len(latest_filtered)
            print(f"      ❗ Writing RESEARCH Table bsf.daily_signals_last: Rows={latest_count}")
            #elf.spark.createDataFrame(latest_filtered).write.format("delta") \
            #    .mode("overwrite") \
            #    .partitionBy("TimeFrame") \
            #    .option("mergeSchema", "false") \
            #    .saveAsTable("bsf.daily_signals_last")
            #print(f"      ✅ Completed writing BASE delta table for latest candidates.")
            write_delta_table(
                spark=self.spark,
                df=latest_df[good_cols].copy(),
                table_name="bsf.daily_signals_last",
                partition_col="TimeFrame",
                partition_value=timeframe,
                cast_columns=cast_columns
            )
                
        if history_df is not None and len(history_df) > 0:
            history_filtered = history_df[good_cols].copy()
            history_count = len(history_filtered)
            print(f"      ❗ Writing RESEARCH Table bsf.daily_signals: Rows={history_count}")
            #self.spark.createDataFrame(history_filtered).write.format("delta") \
            #    .mode("overwrite") \
            #    .partitionBy("TimeFrame") \
            #    .option("mergeSchema", "false") \
            #    .saveAsTable("bsf.daily_signals")
            #print(f"      ✅ Completed writing BASE delta table for history.")
            write_delta_table(
                spark=self.spark,
                df=history_df[good_cols].copy(),
                table_name="bsf.daily_signals",
                partition_col="TimeFrame",
                partition_value=timeframe,
                cast_columns=cast_columns
            )

        
    def optimize_table(self, ingest_ts: str = None):
        query = """
            OPTIMIZE bsf.companystockhistory
            ZORDER BY (StockDate)
        """
        print(f"      📁 ZORDER on StockDate entire table")   
        self.spark.sql(query)
        print("      ✅ OPTIMIZE/ZORDER Completed on StockDate: bsf.companystockhistory")
