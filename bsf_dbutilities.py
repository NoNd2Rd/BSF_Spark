# bsf_db_utilities.py
import os
import sys
import math
import shutil
from datetime import datetime, date, timedelta
import requests
import json
from bsf_config import load_settings


import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import lit, col
from pyspark.sql.types import DoubleType, StructType, StructField, StringType, IntegerType, FloatType, DateType, TimestampType
from pyspark.sql.utils import AnalysisException
from pyspark.sql import Row


from delta.tables import DeltaTable
from sqlalchemy import text
from pyspark.sql.window import Window



class DBUtils:
    def __init__(self, spark, ingest_ts):
        self.spark = spark
        self.ingest_ts = ingest_ts
        
    def get_users(self, engine):
        """Return a list of user IDs from aspnetuser."""
        pdf_users = pd.read_sql("SELECT UserId, TemplateProfile, UserName FROM aspnetuser where   AutoCreateTemplate is True ", engine)
        return pdf_users.to_dict(orient="records") 


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

    
    from pyspark.sql.utils import AnalysisException
    import shutil, os
    
    def clear_hive_table(self, db: str, table: str):
        fqtn = f"{db}.{table}"
        print(f"    🧹 Dropping table: {fqtn}")
    
        try:
            location = (
                self.spark.sql(f"DESCRIBE DETAIL {fqtn}")
                .select("location")
                .first()
                .location
            )
        except AnalysisException:
            location = None
    
        # Drop the Hive table (catalog + files if managed)
        self.spark.sql(f"DROP TABLE IF EXISTS {fqtn}")
    
        # As a safeguard: manually delete location if it still exists
        if location and os.path.exists(location):
            shutil.rmtree(location, ignore_errors=True)
            print(f"    🗑️ Removed leftover files at {location}")
        else:
            print(f"    🗑️ Table {fqtn} dropped (no leftover files found)")


    
    def spark_stats(self, verbose: bool=False):
        """
        Print Spark configuration and runtime info using a single SparkConf object.
        """
  
        # Get Spark config once
        conf = self.spark.sparkContext.getConf()
    
        # ----------------------------
        # Basic Spark & environment info
        # ----------------------------
        print(f"    🔍 === Runtime Config Check ===")
        print(f"      📌 User                       : {os.getenv('USER', 'Unknown')}")
        print(f"      📌 Python Version             : {sys.version.split()[0]}")

        # ----------------------------
        # Runtime Spark config check   
        # ----------------------------
        print(f"    🔍 === Spark Runtime Config Check ===")
        print(f"      📌 Name                       : {conf.get('spark.app.name', 'Not set')}")
        print(f"      📌 Master                     : {conf.get('spark.master', 'Not set')}")
        print(f"      📌 Version                    : {conf.get('spark.version', 'Not set')}")
        print(f"      📌 Max Cores                  : {conf.get('spark.cores.max', 'Not set')}")
        print(f"      📌 Executor Instances         : {conf.get('spark.executor.instances', 'Not set')}")
        print(f"      📌 Executor Cores             : {conf.get('spark.executor.cores', 'Not set')}")
        print(f"      📌 Task Cpus                  : {conf.get('spark.task.cpus', 'Not set')}")
        print(f"      📌 Executor Memory            : {conf.get('spark.executor.memory', 'Not set')}")
        print(f"      📌 Driver Memory              : {conf.get('spark.driver.memory', 'Not set')}")
        if verbose:
            print(f"      📌 JVM Memory Overhead        : {conf.get('spark.executor.memoryOverhead', 'Not set')}")
            print(f"      📌 Dynamic Allocation Enabled : {conf.get('spark.dynamicAllocation.enabled', 'Not set')}")
            print(f"      📌 Default Parallelism        : {conf.get('spark.default.parallelism', 'Not set')}")
            print(f"      📌 SQL Shuffle Partitions     : {conf.get('spark.sql.shuffle.partitions', 'Not set')}")
            print(f"      📌 Memory Fraction            : {conf.get('spark.memory.fraction', 'Not set')}")
            print(f"      📌 Memory StorageFraction     : {conf.get('spark.memory.storageFraction', 'Not set')}")
            print(f"      📌 SQL Adaptive               : {conf.get('spark.sql.adaptive.enabled', 'Not set')}")
            print(f"      📌 Shuffle Input Size         : {conf.get('spark.sql.adaptive.shuffle.targetPostShuffleInputSize', 'Not set')}")
            print(f"      📌 Delta OptimizeWrite        : {conf.get('spark.databricks.delta.optimizeWrite.enabled', 'Not set')}")
            print(f"      📌 Delta AutoCompact          : {conf.get('spark.databricks.delta.autoCompact.enabled', 'Not set')}")
            print(f"      📌 SQL Adaptive SkewJoin      : {conf.get('spark.sql.adaptive.skewJoin.enabled', 'Not set')}")
            print(f"      📌 Scheduler Pool             : {conf.get('spark.scheduler.pool', 'default')}")
            print(f"      📌 Sql Catalog Implementation : {conf.get('spark.sql.catalogImplementation', 'Not set')}")
            print(f"      📌 Catalog                    : {conf.get('spark.sql.catalog.spark_catalog', 'Not set')}")
            print(f"      📌 Sql Warehouse Dir          : {conf.get('spark.sql.warehouse.dir', 'Not set')}")
            print(f"      📌 Delta Base Path            : {conf.get('spark.delta.basePath', 'Not set')}")
            print(f"      📌 Filesource Path            : {conf.get('spark.sql.filesource.path', 'Not set')}")
            print(f"      📌 Nond2rd Path               : {conf.get('spark.nond2rd.path', 'Not set')}")
            print(f"      📌 Delta Retention Check      : {conf.get('spark.databricks.delta.retentionDurationCheck.enabled', 'false')}")
            print(f"      📌 Delta LogStore             : {conf.get('spark.databricks.delta.logStore.class', 'Not set')}")
        '''
            .config("spark.memory.fraction", .5)
            .config("spark.memory.storageFraction", .6)
            .config("spark.sql.adaptive.enabled", "true")  
            .config("spark.sql.adaptive.skewJoin.enabled", "true")
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
        print(f"    📋 Stats Hive Database: {db_name}")
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
                print(f"    ✅ Database Name: {db.name}, Location: {db.locationUri}")
                
        if not db_found:
            print(f"    ❗ Database Name: {db_name} was not found")
            
       

    def write_delta_merge(self, sdf, table_name, merge_keys, partition_cols=None):
        """
        Always merge sdf into Delta table using merge keys.
        Supports optional partitioning for table creation.
        """
        if DeltaTable.isDeltaTable(self.spark, table_name):
            # Table exists → merge normally
            target = DeltaTable.forName(self.spark, table_name)
            cond = " AND ".join([f"t.{k} = s.{k}" for k in merge_keys])
            target.alias("t") \
                  .merge(sdf.alias("s"), cond) \
                  .whenMatchedUpdateAll() \
                  .whenNotMatchedInsertAll() \
                  .execute()
        else:
            # Table does not exist yet → first write with partitioning
            writer = sdf.write.format("delta").mode("append")
            if partition_cols:
                writer = writer.partitionBy(*partition_cols)
            writer.saveAsTable(table_name)   

            
    def write_history(self, pdf, show_stats: bool = False):
        # Convert to Spark
        sdf = self.spark.createDataFrame(pdf)
    
        if show_stats:
            try:
                est_bytes = sdf.rdd.map(lambda x: len(str(x))).sum()
                est_mib = round(est_bytes / (1024 * 1024), 2)
                print(f"        🧠 Estimated memory footprint: {est_mib} MiB")
            except Exception as e:
                print(f"        ❗ Memory estimation failed: {e}")
                
        merge_keys = ["CompanyId", "StockDate"]
        table = "bsf.companystockhistory"
        self.write_delta_merge(sdf, table, merge_keys)
                
        # -------------------------------
        # Write watermark table
        # -------------------------------  

        watermark_update_df = sdf.groupBy("CompanyId") \
                               .agg(F.max("StockDate").alias("LastLoadedDate"))
        if show_stats:
            try:
                est_bytes = watermark_update_df.rdd.map(lambda x: len(str(x))).sum()
                est_mib = round(est_bytes / (1024 * 1024), 2)
                print(f"        🧠 Estimated memory footprint: {est_mib} MiB")
            except Exception as e:
                print(f"        ❗ Memory estimation failed: {e}")
                
        watermark_update_df = watermark_update_df.coalesce(1)
        merge_keys = ["CompanyId"]
        table = "bsf.companystockhistory_watermark"
        self.write_delta_merge(watermark_update_df, table, merge_keys)

    
    def write_fundamental(self, pdf, show_stats: bool = False):
        # Convert to Spark
        sdf = self.spark.createDataFrame(pdf)
    
        if show_stats:
            try:
                est_bytes = sdf.rdd.map(lambda x: len(str(x))).sum()
                est_mib = round(est_bytes / (1024 * 1024), 2)
                print(f"        🧠 Estimated memory footprint: {est_mib} MiB")
            except Exception as e:
                print(f"        ❗ Memory estimation failed: {e}")
    
        # -------------------------------
        # Write main funndamental table
        # -------------------------------
        merge_keys = ["CompanyId", "StockDate"]
        table = "bsf.companyfundamental"
        
        self.write_delta_merge(sdf, table, merge_keys)

    def write_signal_driver(self, sdf, show_stats: bool = False):
    
        if show_stats:
            try:
                est_bytes = sdf.rdd.map(lambda x: len(str(x))).sum()
                est_mib = round(est_bytes / (1024 * 1024), 2)
                print(f"        🧠 Estimated memory footprint: {est_mib} MiB")
            except Exception as e:
                print(f"        ❗ Memory estimation failed: {e}")
    
        # -------------------------------
        # Write main funndamental table
        # -------------------------------
        merge_keys = None
        table = "bsf.history_signal_driver"
        self.write_delta_merge(sdf, table, merge_keys)
    
    def write_company(self, pdf, show_stats: bool = False):
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

        # ----------------------------
        # Convert to Spark DataFrame
        # ----------------------------
        sdf = self.spark.createDataFrame(pdf, schema=schema)
        sdf = sdf.coalesce(1)
        
        if show_stats:
            try:
                est_bytes = sdf.rdd.map(lambda x: len(str(x))).sum()
                est_mib = round(est_bytes / (1024 * 1024), 2)
                print(f"        🧠 Estimated memory footprint: {est_mib} MiB")
            except Exception as e:
                print(f"        ❗ Memory estimation failed: {e}")
    

        
        # ----------------------------
        # Write to Delta managed table in lakehouse
        # ----------------------------
        print(f"        ⚡️ Start writing delta tables for company.")
        
        # Force small number of partitions (1 or match your cores)
        #sdf = sdf.coalesce(1)   # <- best if you want a single file
        # or sdf = sdf.repartition(4)  # if you want 4 files max

        merge_keys = ["CompanyId"]
        table = "bsf.company"
        
        self.write_delta_merge(sdf, table, merge_keys)    
         
    
    def write_signals(self, df=None):
        """
        Write signals directly from a Spark DataFrame to Delta tables.
        Accepts sdf: Spark DataFrame with all necessary columns.
        """
        sdf = self.spark.createDataFrame(df)
        
        from pyspark.sql.types import IntegerType
        from pyspark.sql import functions as F
        
        # Cast key columns to Integer
        sdf = sdf.withColumn("UserId", F.col("UserId").cast(IntegerType()))
        sdf = sdf.withColumn("CompanyId", F.col("CompanyId").cast(IntegerType()))
        sdf = sdf.withColumn("TimeFrame", F.col("TimeFrame").cast("string"))
        # Columns to keep
        good_cols = [
            "UserId", "CompanyId", "StockDate", "Open", "High", "Low", "Close", "TomorrowClose", "Return", "TomorrowReturn",
            "Doji", "Hammer", "InvertedHammer", "ShootingStar", "BullishEngulfing", "BearishEngulfing", "PiercingLine",
            "DarkCloudCover", "MorningStar", "EveningStar", "ThreeWhiteSoldiers", "ThreeBlackCrows", "TweezerTop",
            "TweezerBottom", "InsideBar", "OutsideBar", "MA", "MA_slope", "UpTrend_MA", "DownTrend_MA", "MomentumUp",
            "MomentumDown", "ConfirmedUpTrend", "ConfirmedDownTrend", "RecentReturn", "UpTrend_Return",
            "DownTrend_Return", "Volatility", "LowVolatility", "HighVolatility", "ROC", "MomentumZ", "SignalStrength",
            "SignalStrengthHybrid", "ActionConfidence", "BullishStrengthHybrid", "BearishStrengthHybrid",
            "SignalDuration", "ValidAction", "HasValidSignal", "MomentumAction", "PatternAction", "CandleAction",
            "CandidateAction", "Action", "TomorrowAction", "TomorrowActionSource", "BatchId", "IngestedAt", "TimeFrame"
        ]
        
        keep_cols = [
            "UserId","CompanyId", "StockDate", "Open", "High", "Low", "Close", "TomorrowClose", "Return", "TomorrowReturn",
            "MA", "MA_slope", "UpTrend_MA", "DownTrend_MA", "MomentumUp", "MomentumDown",
            "ConfirmedUpTrend", "ConfirmedDownTrend", "Volatility", "LowVolatility", "HighVolatility", "SignalStrength",
            "SignalStrengthHybrid", "ActionConfidence",
            "BullishStrengthHybrid", "BearishStrengthHybrid", "SignalDuration",
            "PatternAction", "CandleAction","UpTrend_Return",
            "CandidateAction", "Action", "TomorrowAction", "TimeFrame"
        ]
    
        # Filter columns
        sdf = sdf.select(*[c for c in keep_cols if c in sdf.columns])
    
        # Latest row per company
        latest_sdf = sdf.withColumn("row_num", F.row_number().over(
            Window.partitionBy("CompanyId").orderBy(F.desc("StockDate"))
        )).filter(F.col("row_num") == 1).drop("row_num")
    
        # Merge main table
        merge_keys = ["UserId", "CompanyId", "StockDate", "TimeFrame"]
        partition_cols = ["UserId", "TimeFrame"]
        table = "bsf.history_signals"
        self.write_delta_merge(sdf, table, merge_keys,partition_cols)    
        # Merge last row table
        merge_keys = ["UserId", "CompanyId", "StockDate", "TimeFrame"]
        partition_cols = ["UserId", "TimeFrame"]
        table = "bsf.history_signals_last"
        self.write_delta_merge(sdf, table, merge_keys,partition_cols)   

        
  
    def write_candidates(self, df_phase3_enriched, df_topN_companies):


        table='bsf.final_candidates_enriched'
        df_phase3_enriched.write.format("delta").mode("append").saveAsTable(f"{table}")
        
        table='bsf.final_candidates'
        df_topN_companies.write.format("delta").mode("append").saveAsTable(f"{table}")
        
        # csv for review
        conf = self.spark.sparkContext.getConf()
        output_path = conf.get("spark.nond2rd.path", "/srv/lakehouse/nond2rd")
        #output_path = "/srv/lakehouse/nond2rd"
        df_topN_companies.toPandas().to_csv(
            f"{output_path}/{'final_candidates'}.csv",
            index=False
        )
        
        return  
     
    def optimize_table(self, ingest_ts: str = None):
        query = """
            OPTIMIZE bsf.companystockhistory
            ZORDER BY (StockDate)
        """
        print(f"      📁 ZORDER on StockDate entire table")   
        self.spark.sql(query)

        print("      ✅ OPTIMIZE/ZORDER Completed on StockDate: bsf.companystockhistory")

    def create_bsf(self, engine, user, username, profile, df_dict):
        settings = load_settings(profile)

        settings_json = json.dumps(template_settings)
        # -----------------------------
        # Optional: show counts
        # -----------------------------
        for tf, records in df_dict.items():
            # Outer loop: per timeframe
            print(f"     ✅ Processing timeframe: {tf}")
            
            if tf == "Daily":
                generate_end_date = date.today() + timedelta(days=3)
                days_to_hold = 1
                seasonal_s_m = 3 #can't be 1
            elif tf == "Short":
                generate_end_date = date.today() + timedelta(days=7)
                days_to_hold = 3
                seasonal_s_m = 3
            elif tf == "Swing":
                generate_end_date = date.today() + timedelta(days=11)
                days_to_hold = 5
                seasonal_s_m = 5
            else:
                generate_end_date = date.today() + timedelta(days=21)
                days_to_hold = 10
                seasonal_s_m = 10
            sql = '''
                INSERT INTO template (
                    UserId, IndustryId, MarketSectorId, ParentTemplateId, Name, Description, ScreenImage,
                    GenerateStartDate, GenerateEndDate, DaysToHold, EmailPortfolio, PortfolioMaxInvestment,
                    MinSharesPerOption, MaxSharesPerOption, MinBidDlrPerOption, MaxBidDlrPerOption,
                    MinShareRoiPercentage, AutoSelectOptions, OptionChoosenBy, TemplateMlType,
                    AutoSelectScaler, Scaler, AutoSelectEstimator, Estimator,
                    BuildStartDate, BuildEndDate, BuildStatus, BuildSentimentModel, BuildClusterModel,
                    NbrOfClusters, TrainDays, TestDays, SeriesLength, WindowSize, Confidence, DaysToForecast,
                    IsAdaptive, ShouldMaintain, ShouldStabilize, OptimizeOrder, OrderAr, OrderI, OrderMa,
                    SeasonalP, SeasonalD, SeasonalQ, SeasonalS, ExperimentRunMinutes, PctToTrain, NbrOfCrossfolds,
                    SelectR2Score, SelectLossFnScore, SelectL1Loss, SelectL2Loss, SelectRMSLoss, PysparkConfig,
                    Status, Active, CreateDate, ChangeDate, ModifiedByProcess, ModifiedByUserId, SoftDelete
                ) VALUES (
                    :UserId, :IndustryId, :MarketSectorId, :ParentTemplateId, :Name, :Description, :ScreenImage,
                    :GenerateStartDate, :GenerateEndDate, :DaysToHold, :EmailPortfolio, :PortfolioMaxInvestment,
                    :MinSharesPerOption, :MaxSharesPerOption, :MinBidDlrPerOption, :MaxBidDlrPerOption,
                    :MinShareRoiPercentage, :AutoSelectOptions, :OptionChoosenBy, :TemplateMlType,
                    :AutoSelectScaler, :Scaler, :AutoSelectEstimator, :Estimator,
                    :BuildStartDate, :BuildEndDate, :BuildStatus, :BuildSentimentModel, :BuildClusterModel,
                    :NbrOfClusters, :TrainDays, :TestDays, :SeriesLength, :WindowSize, :Confidence, :DaysToForecast,
                    :IsAdaptive, :ShouldMaintain, :ShouldStabilize, :OptimizeOrder, :OrderAr, :OrderI, :OrderMa,
                    :SeasonalP, :SeasonalD, :SeasonalQ, :SeasonalS, :ExperimentRunMinutes, :PctToTrain, :NbrOfCrossfolds,
                    :SelectR2Score, :SelectLossFnScore, :SelectL1Loss, :SelectL2Loss, :SelectRMSLoss, :PysparkConfig,
                    :Status, :Active, :CreateDate, :ChangeDate, :ModifiedByProcess, :ModifiedByUserId, :SoftDelete
                )
                '''       
            row_data = {
                "UserId": user,
                "IndustryId": 1,
                "MarketSectorId": 1,
                "ParentTemplateId": None,
                "Name": f"BSF Automatic Build - for {profile.capitalize()} account",
                "Description": f"ML generated template for timeframe: {tf}",
                "ScreenImage": None,
                "GenerateStartDate": date.today() + timedelta(days=1),
                "GenerateEndDate": generate_end_date,
                "DaysToHold": days_to_hold,
                "EmailPortfolio": 1,
                "PortfolioMaxInvestment": 500,
                "MinSharesPerOption": 100,
                "MaxSharesPerOption": 10000,
                "MinBidDlrPerOption": 0.0001,
                "MaxBidDlrPerOption": 1,
                "MinShareRoiPercentage": 0.12,
                "AutoSelectOptions": 0,
                "OptionChoosenBy": 1,
                "TemplateMlType": 1,
                "AutoSelectScaler": 0,
                "Scaler": 3,
                "AutoSelectEstimator": 0,
                "Estimator": 305,
                "BuildStartDate": datetime.now(),
                "BuildEndDate": datetime.now(),
                "BuildStatus": 1,  
                "BuildSentimentModel": 0,
                "BuildClusterModel": 0,
                "NbrOfClusters": 0,
                "TrainDays": 730,
                "TestDays": 21,
                "SeriesLength": 7,
                "WindowSize": 3,
                "Confidence": 0.8,
                "DaysToForecast": 1,
                "IsAdaptive": 1,
                "ShouldMaintain": 1,
                "ShouldStabilize": 1,
                "OptimizeOrder": 0,
                "OrderAr": 0,
                "OrderI": 0,
                "OrderMa": 1,
                "SeasonalP": 0,
                "SeasonalD": 0,
                "SeasonalQ": 0,
                "SeasonalS": seasonal_s_m,
                "ExperimentRunMinutes": 5,
                "PctToTrain": 0.8,
                "NbrOfCrossfolds": 5,
                "SelectR2Score": 0,
                "SelectLossFnScore": 0,
                "SelectL1Loss": 0,
                "SelectL2Loss": 0,
                "SelectRMSLoss": 1,
                "PysparkConfig": settings_json,
                "Status": 1,
                "Active": 1,
                "CreateDate": datetime.now(),
                "ChangeDate": datetime.now(),
                "ModifiedByProcess": "Pyspark Candidates",
                "ModifiedByUserId": 1,
                "SoftDelete": 0
            }
    
    
            table ="Template" 
  
            with engine.connect() as conn:
                trans = conn.begin()          # start transaction
                try:
                    conn.execute(text(sql), row_data)
                    template_id = conn.execute(text("SELECT LAST_INSERT_ID()")).scalar()
                    trans.commit()            # commit manually
                    print(f"       ❗Inserted {table} = {template_id}")
                except:
                    trans.rollback()
                    raise                

            
            for record in records.collect():
                # Inner loop: per record in this timeframe
                print(f"       ❗Processing company: {record['CompanyId']}")
                
                # Do something different per record
                # For example, update values or run a function

                
                sql = """
                INSERT INTO templateoption (
                    TemplateId, CompanyId, UseAsInfluencer, AutoSelectScaler, Scaler, AutoSelectEstimator, Estimator,
                    TrainDays, TestDays, SeriesLength, WindowSize, Confidence, DaysToForecast, IsAdaptive,
                    ShouldMaintain, ShouldStabilize, OptimizeOrder, OrderAr, OrderI, OrderMa,
                    SeasonalP, SeasonalD, SeasonalQ, SeasonalS, ExperimentRunMinutes, PctToTrain, NbrOfCrossfolds,
                    SelectR2Score, SelectLossFnScore, SelectL1Loss, SelectL2Loss, SelectRMSLoss,
                    Status, Active, CreateDate, ChangeDate, ModifiedByProcess, ModifiedByUserId, SoftDelete
                ) VALUES (
                    :TemplateId, :CompanyId, :UseAsInfluencer, :AutoSelectScaler, :Scaler, :AutoSelectEstimator, :Estimator,
                    :TrainDays, :TestDays, :SeriesLength, :WindowSize, :Confidence, :DaysToForecast, :IsAdaptive,
                    :ShouldMaintain, :ShouldStabilize, :OptimizeOrder, :OrderAr, :OrderI, :OrderMa,
                    :SeasonalP, :SeasonalD, :SeasonalQ, :SeasonalS, :ExperimentRunMinutes, :PctToTrain, :NbrOfCrossfolds,
                    :SelectR2Score, :SelectLossFnScore, :SelectL1Loss, :SelectL2Loss, :SelectRMSLoss,
                    :Status, :Active, :CreateDate, :ChangeDate, :ModifiedByProcess, :ModifiedByUserId, :SoftDelete
                )
                """
        
                row_data = {
                    "TemplateId": template_id,                  # FK back to Template table
                    "CompanyId": record["CompanyId"],               # new company
                    "UseAsInfluencer": 0,
                    "AutoSelectScaler": 0,
                    "Scaler": 3,                      # different scaler
                    "AutoSelectEstimator": 0,
                    "Estimator": 305,                 # different estimator
                    "TrainDays": 730,
                    "TestDays": 21,
                    "SeriesLength": 7,
                    "WindowSize": 3,
                    "Confidence": 0.85,               # slightly different
                    "DaysToForecast": 1,
                    "IsAdaptive": 1,
                    "ShouldMaintain": 1,
                    "ShouldStabilize": 1,
                    "OptimizeOrder": 0,
                    "OrderAr": 0,
                    "OrderI": 0,
                    "OrderMa": 1,
                    "SeasonalP": 0,
                    "SeasonalD": 0,
                    "SeasonalQ": 0,
                    "SeasonalS": seasonal_s_m,                   # different seasonal cycle
                    "ExperimentRunMinutes": 10,
                    "PctToTrain": 0.75,
                    "NbrOfCrossfolds": 3,
                    "SelectR2Score": 0,
                    "SelectLossFnScore": 0,
                    "SelectL1Loss": 0,
                    "SelectL2Loss": 0,
                    "SelectRMSLoss": 1,
                    "Status": 1,
                    "Active": 1,
                    "CreateDate": datetime.now(),
                    "ChangeDate": datetime.now(),   # simulate later update
                    "ModifiedByProcess": "Pyspark Candidates",
                    "ModifiedByUserId": 1,
                    "SoftDelete": 0
                }
        
        
                
                table ="TemplateOption"
                with engine.connect() as conn:
                    trans = conn.begin()          # start transaction
                    try:
                        conn.execute(text(sql), row_data)
                        trans.commit()            # commit manually
                        print(f"         ❗Inserted {table}")
                    except:
                        trans.rollback()
                        raise
                sql = """
                    INSERT INTO mlanalysis (
                        AnalysisText, ConsoleAnalysis, AutoML, EstimatorSource, EstimatorType, Estimator, Scaler,
                        AnalysisDate, BuildStartDate, BuildEndDate, BuildStatus,
                        ModelFileName, PlotFileName, ScalerFileName, FeaturesName, Features,
                        LossFunction, RSquared, MeanAbsoluteError, MeanSquaredError, RootMeanSquaredError,
                        AvgLossFunction, AvgRSquared, AvgMeanAbsoluteError, AvgMeanSquaredError, AvgRootMeanSquaredError,
                        Accuracy, AreaUnderRocCurve, AreaUnderPrecisionRecallCurve, F1Score, LogLoss, LogLossReduction,
                        PositivePrecision, PositiveRecall, NegativePrecision, NegativeRecall,
                        AdAreaUnderRocCurve, DetectionRateAtFalsePositiveCount, MacroAccuracy, MicroAccuracy, McLogLoss,
                        PerClassLogLoss1, PerClassLogLoss2, PerClassLogLoss3,
                        MicroAccuraciesStdDeviation, MacroAccuraciesStdDeviation, LogLossStdDeviation, LogLossReductionStdDeviation,
                        AverageDistance, DaviesBouldinIndex, SilhouetteScore, CalinskiHarabaszScore,
                        Homogeneity, Completeness, AdjustedRandIndex, NormalizingDCG,
                        Active, CreateDate, ChangeDate, ModifiedByProcess, ModifiedByUserId, SoftDelete
                    ) VALUES (
                        :AnalysisText, :ConsoleAnalysis, :AutoML, :EstimatorSource, :EstimatorType, :Estimator, :Scaler,
                        :AnalysisDate, :BuildStartDate, :BuildEndDate, :BuildStatus,
                        :ModelFileName, :PlotFileName, :ScalerFileName, :FeaturesName, :Features,
                        :LossFunction, :RSquared, :MeanAbsoluteError, :MeanSquaredError, :RootMeanSquaredError,
                        :AvgLossFunction, :AvgRSquared, :AvgMeanAbsoluteError, :AvgMeanSquaredError, :AvgRootMeanSquaredError,
                        :Accuracy, :AreaUnderRocCurve, :AreaUnderPrecisionRecallCurve, :F1Score, :LogLoss, :LogLossReduction,
                        :PositivePrecision, :PositiveRecall, :NegativePrecision, :NegativeRecall,
                        :AdAreaUnderRocCurve, :DetectionRateAtFalsePositiveCount, :MacroAccuracy, :MicroAccuracy, :McLogLoss,
                        :PerClassLogLoss1, :PerClassLogLoss2, :PerClassLogLoss3,
                        :MicroAccuraciesStdDeviation, :MacroAccuraciesStdDeviation, :LogLossStdDeviation, :LogLossReductionStdDeviation,
                        :AverageDistance, :DaviesBouldinIndex, :SilhouetteScore, :CalinskiHarabaszScore,
                        :Homogeneity, :Completeness, :AdjustedRandIndex, :NormalizingDCG,
                        :Active, :CreateDate, :ChangeDate, :ModifiedByProcess, :ModifiedByUserId, :SoftDelete
                    )
                    """       
                row_data = {
                    "AnalysisText": "Web Training",
                    "ConsoleAnalysis": 0,
                    "AutoML": 0,
                    "EstimatorSource": 2,
                    "EstimatorType": 1,
                    "Estimator": 305,
                    "Scaler": 3,
                    "AnalysisDate": date.today(),
                    "BuildStartDate": datetime.now(),    
                    "BuildEndDate": datetime.now(),      
                    "BuildStatus": 1,
                    "ModelFileName": None,
                    "PlotFileName": None,
                    "ScalerFileName": None,
                    "FeaturesName": None,
                    "Features": None,
                    "LossFunction": 0,
                    "RSquared": 0,
                    "MeanAbsoluteError": 0,
                    "MeanSquaredError": 0,
                    "RootMeanSquaredError": 0,
                    "AvgLossFunction": 0,
                    "AvgRSquared": 0,
                    "AvgMeanAbsoluteError": 0,
                    "AvgMeanSquaredError": 0,
                    "AvgRootMeanSquaredError": 0,
                    "Accuracy": 0,
                    "AreaUnderRocCurve": 0,
                    "AreaUnderPrecisionRecallCurve": 0,
                    "F1Score": 0,
                    "LogLoss": 0,
                    "LogLossReduction": 0,
                    "PositivePrecision": 0,
                    "PositiveRecall": 0,
                    "NegativePrecision": 0,
                    "NegativeRecall": 0,
                    "AdAreaUnderRocCurve": 0,
                    "DetectionRateAtFalsePositiveCount": 0,
                    "MacroAccuracy": 0,
                    "MicroAccuracy": 0,
                    "McLogLoss": 0,
                    "PerClassLogLoss1": 0,
                    "PerClassLogLoss2": 0,
                    "PerClassLogLoss3": 0,
                    "MicroAccuraciesStdDeviation": 0,
                    "MacroAccuraciesStdDeviation": 0,
                    "LogLossStdDeviation": 0,
                    "LogLossReductionStdDeviation": 0,
                    "AverageDistance": 0,
                    "DaviesBouldinIndex": 0,
                    "SilhouetteScore": 0,
                    "CalinskiHarabaszScore": 0,
                    "Homogeneity": 0,
                    "Completeness": 0,
                    "AdjustedRandIndex": 0,
                    "NormalizingDCG": 0,
                    "Active": 1,
                    "CreateDate": datetime.now(),
                    "ChangeDate": datetime.now(),
                    "ModifiedByProcess": "Pyspark Candidates",
                    "ModifiedByUserId": 1,
                    "SoftDelete": 0
                }
                table ="MlAnalysis"
  
                with engine.connect() as conn:
                    trans = conn.begin()          # start transaction
                    try:
                        conn.execute(text(sql), row_data)
                        ml_analysis_id = conn.execute(text("SELECT LAST_INSERT_ID()")).scalar()
                        trans.commit()            # commit manually
                        print(f"         ❗Inserted {table} = {ml_analysis_id}")
                    except:
                        trans.rollback()
                        raise
        
                sql = """
                INSERT INTO templateoptionmlanalysis (
                    TemplateId, CompanyId, MlAnalysisId,
                    Active, CreateDate, ChangeDate, ModifiedByProcess, ModifiedByUserId, SoftDelete
                ) VALUES (
                    :TemplateId, :CompanyId, :MlAnalysisId,
                    :Active, :CreateDate, :ChangeDate, :ModifiedByProcess, :ModifiedByUserId, :SoftDelete
                )
                """                        
                row_data = {
                    "TemplateId": template_id,
                    "CompanyId": record["CompanyId"],
                    "MlAnalysisId": ml_analysis_id,
                    "Active": 1,
                    "CreateDate": datetime.now(),
                    "ChangeDate": datetime.now(),
                    "ModifiedByProcess": "Pyspark Candidates",
                    "ModifiedByUserId": 1,
                    "SoftDelete": 0
                }
        
                table ="TemplateOptionMlAnalysis"

                with engine.connect() as conn:
                    trans = conn.begin()          # start transaction
                    try:
                        conn.execute(text(sql), row_data)
                        trans.commit()            # commit manually
                        print(f"         ❗Inserted {table} ") 
                    except:
                        trans.rollback()
                        raise
                
        

                flask_url=f"http://localhost" 
                flask_port= 6012  
                flask_endpoint= "execute" 
        
                # Base URL
                base_url = f"{flask_url}:{flask_port}/{flask_endpoint}"
               
                results = []
                params = {
                    "forecast_type": "arm",
                    "action": "train",
                    "template_id": template_id,
                    "mlanalysis_id": ml_analysis_id,
                    "company_id": record["CompanyId"]
                }
                try:
                    resp = requests.get(base_url, params=params, timeout=30)
                    resp.raise_for_status()
                    result = resp.json().get("result", None)
                except Exception as e:
                    result = f"Error: {e}"
                    print(f"flask error: {e}")
                # Append the result along with the original row
                results.append(Row(
                    TemplateId=template_id,
                    CompanyId=record["CompanyId"],
                    MlAnalysisId=ml_analysis_id,
                    ApiResult=result
                ))

        return  
         