# -----------------------------
# PySpark
# -----------------------------
from pyspark.sql import SparkSession, DataFrame, functions as F, types as T, Window
from pyspark.sql.functions import lit, current_timestamp, broadcast
from delta.tables import DeltaTable

# -----------------------------
# Data Science & ML
# -----------------------------
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -----------------------------
# Time Series Forecasting
# -----------------------------
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import pmdarima as pm

# -----------------------------
# Visualization
# -----------------------------
import matplotlib.pyplot as plt
from IPython.display import display, HTML

# -----------------------------
# System & Utilities
# -----------------------------
import os
import sys
import re
import warnings
import traceback
import tempfile
from datetime import datetime
from tqdm import tqdm
import joblib

# -----------------------------
# Warnings & Settings
# -----------------------------
warnings.filterwarnings("ignore")


warnings.filterwarnings("ignore", category=ConvergenceWarning)

# -----------------------------
# Hybrid scoring function
# -----------------------------
def hybrid_score(metrics, w_dir=1.0, w_rmse=0.5, w_mape=0.5):
    """
    Combine multiple metrics: higher DirectionAcc, lower RMSE & MAPE
    """
    return w_dir*metrics["DirectionAcc"] - w_rmse*metrics["RMSE"] - w_mape*metrics["MAPE"]

# -----------------------------
# Pick best model using hybrid score
# -----------------------------
def select_best_model(metrics_dict):
    scores = {name: hybrid_score(metrics_dict[name]) for name in metrics_dict}
    best_name = max(scores, key=scores.get)
    return best_name, scores


def phase_1(spark, df_all, df_last,top_n=100):
    # -----------------------------
    # Filter only Buy actions from last-row DF
    # -----------------------------
    # Assign a weighted score
    df_last_buys = df_last.filter(F.col("Action") == "Buy").cache()
    
    df_last_buys = df_last_buys.withColumn(
        "BuyScore",
        # Weighted combination of key signals
        F.col("ActionConfidence") * 0.3 +
        F.col("SignalStrengthHybrid") * 0.2 +
        F.col("BullishStrengthHybrid") * 0.2 +
        F.col("UpTrend_Return").cast("integer") * 0.1
    )
    '''
    +
        F.col("UpTrend_Return").cast("integer") * 0.1
        '''
    df_last_buys = df_last_buys.withColumn(
        "BuyScore",
        F.col("BuyScore") * F.when(F.col("Volatility") > 0.05, 0.8).otherwise(1.0)
    )
    
    # -----------------------------
    # Define ranking window per timeframe
    # -----------------------------
    '''
    w_tf = Window.partitionBy("TimeFrame").orderBy(
        F.desc("ActionConfidence"),
        F.desc("Return")
    )
    '''
    w_tf = Window.partitionBy("TimeFrame").orderBy(
        F.desc("BuyScore")
    )

    
    # -----------------------------
    # Rank and select top N Buy companies per timeframe
    # -----------------------------
    df_ranked_last_top = (
        df_last_buys
        .withColumn("BuyRank", F.row_number().over(w_tf))  # use rank() if you want ties
        .filter(F.col("BuyRank") <= top_n)
        .orderBy(F.col("ActionConfidence").desc(), F.col("BuyRank").asc())
    )
    
    # -----------------------------
    # Extract top companies per timeframe
    # -----------------------------
    top_companies = df_ranked_last_top.select("CompanyId", "TimeFrame").distinct().cache()
    # -----------------------------
    # Filter original last-row DF and full historical DF to include only top Buy companies
    # Cache large DataFrames once
    # -----------------------------
    df_ranked_last_topN = df_last.join(
        broadcast(df_ranked_last_top.select("CompanyId","TimeFrame","BuyRank")),
        on=["CompanyId","TimeFrame"],
        how="inner"
    ).cache()
    
    df_ranked_all_topN = df_all.join(
        broadcast(df_ranked_last_top.select("CompanyId","TimeFrame","BuyRank")),
        on=["CompanyId","TimeFrame"],
        how="inner"
    ).cache()
    
   
    # -----------------------------
    # List of timeframes
    # -----------------------------
    timeframes = ["Short", "Swing", "Long", "Daily"]
    
    # -----------------------------
    # Dictionaries to store per-timeframe DataFrames
    # -----------------------------
    timeframe_dfs = {}
    timeframe_dfs_all = {}
    
    # -----------------------------
    # Efficient per-timeframe splitting
    # -----------------------------
    for tf in timeframes:
        # Last-row top N for this timeframe
        timeframe_dfs[tf] = df_ranked_last_topN.filter(F.col("TimeFrame") == tf)
        
        # Full historical top N for this timeframe
        timeframe_dfs_all[tf] = df_ranked_all_topN.filter(F.col("TimeFrame") == tf)
    
    
    print(f"     ✅ Stage 1 completed: Top {top_n} candidates selected per timeframe")
    return timeframe_dfs_all, timeframe_dfs

def phase_2(spark, timeframe_dfs_all, top_n_phase2=20):
    
    # -----------------------------
    # Parameters
    # -----------------------------
    target_stage2 = "TomorrowClose"
    epsilon = 1e-8
    all_stage2_predictions = []
    
    # -----------------------------
    # Loop over timeframes (Pandas)
    # -----------------------------
    for tf, sdf_tf in timeframe_dfs_all.items():  
        # Convert Spark DF to Pandas once
        df_tf = sdf_tf.toPandas()
        
        companies = df_tf["CompanyId"].unique()
        print(f"       🔄 Phase 2 - Timeframe: {tf}")
        
        for cid in companies:
            # Filter by company
            df_c = df_tf[df_tf["CompanyId"] == cid].copy()
            if df_c.empty:
                continue
            
            # 1️⃣ Identify training rows and future rows
            train_df = df_c[df_c[target_stage2].notna()].copy()
            future_df = df_c[df_c[target_stage2].isna()].copy()
            if train_df.empty or future_df.empty:
                continue
            
            # 2️⃣ Log-transform OHLC columns to stabilize variance
            for col in ["Open", "High", "Low", "Close"]:
                df_c[f"log_{col}"] = np.log(df_c[col].replace(0, epsilon))
            
            # 3️⃣ Select numeric & boolean features (excluding target)
            numeric_cols = train_df.select_dtypes(include=[np.number]).columns.difference([target_stage2]).tolist()
            bool_cols = train_df.select_dtypes(include=["bool"]).columns.tolist()
            all_features = numeric_cols + bool_cols
            
            # 4️⃣ Feature correlation with target
            corr = train_df[all_features + [target_stage2]].corr()[target_stage2].abs()
            threshold = 0.03
            good_features = corr[corr >= threshold].drop(target_stage2).index.tolist()
            if not good_features:
                continue
            
            # 5️⃣ Prepare train dataset
            X_train = train_df[good_features].fillna(0)
            y_train = train_df[target_stage2]
            X_future = future_df[good_features].fillna(0)
            
            # -----------------------------
            # 6️⃣ Train models
            # -----------------------------
            models = {
                "Linear": LinearRegression(),
                "Lasso": Lasso(alpha=0.01),
                "Ridge": Ridge(alpha=1.0, solver="svd"),
                "XGBoost": XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, verbosity=0)
            }
            metrics_dict = {}
            for name, model in models.items():
                model.fit(X_train, y_train)
                pred_train = model.predict(X_train)
                rmse = mean_squared_error(y_train, pred_train, squared=False)
                mae = mean_absolute_error(y_train, pred_train)
                mape = np.mean(np.abs((y_train - pred_train) / (y_train + epsilon)))
                direction = np.mean(np.sign(pred_train[1:] - pred_train[:-1]) == np.sign(y_train.values[1:] - y_train.values[:-1]))
                r2 = r2_score(y_train, pred_train)
                k = X_train.shape[1]
                n = len(y_train)
                adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1)) if n - k - 1 != 0 else 0
                metrics_dict[name] = {"RMSE": rmse, "MAE": mae, "MAPE": mape, "DirectionAcc": direction, "R2": r2, "AdjR2": adj_r2}
            
            # 7️⃣ Pick best model
            #best_name = max(metrics_dict, key=lambda m: hybrid_score(metrics_dict[m]))
            #best_model = models[best_name]
            
            # Pick best model with hybrid score
            best_name, scores = select_best_model(metrics_dict)
            best_model = models[best_name]
    
            
            # 8️⃣ Predict future rows
            for name, model in models.items():
                future_df[f"Pred_{name}"] = model.predict(X_future)
            
            # Weighted ensemble (inverse RMSE)
            #total_inv = sum(1 / metrics_dict[m]["RMSE"] for m in metrics_dict)
            total_inv = sum(1 / (metrics_dict[m]["RMSE"] + epsilon) for m in metrics_dict)
            weights = {m: (1 / metrics_dict[m]["RMSE"]) / total_inv for m in metrics_dict}
            future_df["Pred_Sklearn"] = sum(future_df[f"Pred_{m}"] * w for m, w in weights.items())
            
            # Predicted return
            if "Close" in future_df.columns:
                future_df["PredictedReturn_Sklearn"] = (future_df["Pred_Sklearn"] - future_df["Close"]) / future_df["Close"]
                 # Predicted return for each individual model
                for name in models.keys():
                    future_df[f"PredictedReturn_{name}"] = (future_df[f"Pred_{name}"] - future_df["Close"]) / future_df["Close"]
    
                # Maximum predicted return across all models
                return_cols = [f"PredictedReturn_{name}" for name in models.keys()]
                future_df["MaxPredictedReturn"] = future_df[return_cols].max(axis=1)       
           
            
            # Add identifiers & best model info
            future_df["TimeFrame"] = tf
            future_df["CompanyId"] = cid
            future_df["BestModel"] = best_name
            #future_df["BestModel_RMSE"] = metrics_dict[best_name]["RMSE"]
            #future_df["BestModel_MAPE"] = metrics_dict[best_name]["MAPE"]
            #future_df["BestModel_DirAcc"] = metrics_dict[best_name]["DirectionAcc"]
            # Save all metrics for the best model
            for metric_name, value in metrics_dict[best_name].items():
                col_name = f"BestModel_{metric_name}"   # e.g., BestModel_RMSE, BestModel_MAPE
                future_df[col_name] = value
            # save all scores:
            for model_name, model_metrics in metrics_dict.items():
                for metric_name, value in model_metrics.items():
                    # Create column names like "XGBoost_RMSE", "Linear_MAPE"
                    col_name = f"{model_name}_{metric_name}"
                    future_df[col_name] = value
            '''
            # Raise error if best_name or metrics are missing/null/blank
            if (not best_name) or future_df[["BestModel_RMSE","BestModel_MAPE","BestModel_DirAcc"]].isnull().any().any():
                raise ValueError(
                    f"Missing or null metrics for company {cid}, timeframe {tf}, best_name={best_name} | "
                    f"Metrics: {metrics}"
                )
            if future_df.empty or len(good_features) == 0:
                print(f"Skipping {cid}-{tf} | future rows: {len(future_df)}, good features: {len(good_features)}")
                breakpoint()  # or raise ValueError to stop
            '''
            all_stage2_predictions.append(future_df)
    
    
    
    #top_n_phase2 = 25  # number of top candidates per timeframe
    # -----------------------------
    # Combine all Stage 2 predictions into a single Pandas DF
    # -----------------------------
    if all_stage2_predictions:
        stage2_df = pd.concat(all_stage2_predictions, ignore_index=True)
    else:
        raise ValueError("No Stage 2 predictions generated!")
    # -----------------------------
    # Convert Stage 2 Pandas DF → Spark DF
    # -----------------------------
    spark_stage2_all = (
        spark.createDataFrame(stage2_df)
        .withColumn("CompanyId", F.col("CompanyId").cast("bigint"))
        .withColumn("TimeFrame", F.trim(F.col("TimeFrame")))
    )
    
    # -----------------------------
    # Pick best prediction per CompanyId + TimeFrame
    # -----------------------------
    window_comp = Window.partitionBy("CompanyId", "TimeFrame").orderBy(F.desc("PredictedReturn_Sklearn"))
    
    spark_stage2_best = (
        spark_stage2_all
        .withColumn("row_num", F.row_number().over(window_comp))
        .filter(F.col("row_num") == 1)
        .drop("row_num")
    )
    
    # -----------------------------
    # Assign Phase2 rank per timeframe for all rows
    # -----------------------------
    window_tf = Window.partitionBy("TimeFrame").orderBy(F.desc("PredictedReturn_Sklearn"))
    
    spark_stage2_ranked = (
        spark_stage2_best
        .withColumn("Phase2_Rank", F.row_number().over(window_tf))
    )
    
   
    phase2_top_dfs = {}
    
    # Columns from Stage2 DF we care about
    '''
    pred_cols = [
        "Pred_Linear", "Pred_Lasso", "Pred_Ridge", "Pred_XGBoost",
        "Pred_Sklearn", "PredictedReturn_Sklearn",
        "BestModel", "BestModel_RMSE", "BestModel_MAPE",
        "BestModel_DirAcc", "Phase2_Rank"
    ]
    '''
    
    # Start with columns that are fixed
    pred_cols = ["BestModel"]
    # Optionally, also add BestModel metrics and all scoring columns as before
    best_model_metrics = ["RMSE", "MAE", "MAPE", "DirectionAcc", "R2", "AdjR2"]
    for metric in best_model_metrics:
        pred_cols.append(f"BestModel_{metric}")
    
        
    pred_cols.extend(["Pred_Sklearn", "PredictedReturn_Sklearn"])
    
    
    # Add all metrics for every model dynamically from metrics_dict
    for model_name, model_metrics in metrics_dict.items():
        # Optional: Pred_ columns for each model
        pred_cols.append(f"Pred_{model_name}")
        pred_cols.append(f"PredictedReturn_{model_name}")
        
        # Add all scoring metrics
        for metric_name in model_metrics.keys():
            pred_cols.append(f"{model_name}_{metric_name}")
        
        
    pred_cols.extend(["MaxPredictedReturn","Phase2_Rank"])
    
    
    
    for tf, sdf_tf in timeframe_dfs_all.items():
        # Clean historical DF keys
        sdf_tf_clean = (
            sdf_tf
            .withColumn("CompanyId", F.col("CompanyId").cast("bigint"))
            .withColumn("TimeFrame", F.trim(F.col("TimeFrame")))
        )
    
        # Select only the columns from Stage2 we want
        sdf_stage2 = spark_stage2_ranked.select(
            ["CompanyId", "TimeFrame"] + pred_cols
        )
    
        # Left join
        sdf_enriched = sdf_tf_clean.join(
            F.broadcast(sdf_stage2),
            on=["CompanyId", "TimeFrame"],
            how="left"
        )
    
        phase2_top_dfs[tf] = sdf_enriched
    
    # Top-N filter per timeframe
    phase2_topN_dfs = {}
    for tf, sdf in phase2_top_dfs.items():
        sdf_topN = sdf.filter(F.col("Phase2_Rank") <= top_n_phase2)
        phase2_topN_dfs[tf] = sdf_topN
    
    
    print(f"     ✅ Stage 2 completed: Top {top_n_phase2} candidates selected per timeframe")
    return phase2_topN_dfs


def infer_season_length(ts, max_lag=30, threshold=0.3):
    """
    Infer seasonal period `m` from autocorrelation.

    Parameters:
    -----------
    ts : pandas.Series
        Time series values
    max_lag : int
        Maximum lag to inspect for autocorrelation
    threshold : float
        Minimum autocorrelation to consider a peak as seasonal

    Returns:
    --------
    m : int
        Estimated seasonal period
    """
    ts = ts.dropna()
    if len(ts) < 2:
        return 1  # not enough data to infer
    
    acf_vals = acf(ts, nlags=max_lag, fft=True)
    
    # Ignore lag 0
    acf_vals[0] = 0
    
    # Find first lag where autocorrelation exceeds threshold
    peaks = np.where(acf_vals > threshold)[0]
    
    if len(peaks) == 0:
        return 1  # no strong seasonality detected
    
    # Choose the first peak as seasonal period
    m = int(peaks[0])
    return m

def phase_3(spark, phase2_topN_dfs, top_n_final=10):
    
    # -----------------------------
    # Parameters
    # -----------------------------

    sarimax_order = (1,0,0)
    sarimax_seasonal_order = (0,0,0,0)
    epsilon = 1e-6
    ml_weight = 0.6
    sarimax_weight = 0.4
    
    forecast_steps_map = {
        "Daily": 1,
        "Short": 3,
        "Swing": 5,
        "Long": 10
    }
    

    
    # -----------------------------
    # Phase 3: Loop over companies per timeframe
    # -----------------------------
    phase3_results = []
    
    for tf, sdf_tf in phase2_topN_dfs.items():
        print(f"       🔄 Phase 3 - Timeframe: {tf}")
    
        # Collect companies
        companies = sdf_tf.select("CompanyId").distinct().rdd.flatMap(lambda x: x).collect()
        forecast_horizon = forecast_steps_map.get(tf, 1)
        for cid in companies:
            # Filter Spark DF once, convert to Pandas
            df_c = sdf_tf.filter(F.col("CompanyId") == cid).orderBy("StockDate").toPandas()
            if df_c.empty:
                continue
            
            # -----------------------------
            # SARIMAX Forecast
            # -----------------------------
            y = df_c["Close"].replace(0, epsilon)
    
            last_close = y.iloc[-1]
    
            try:
                #auto_model = fit_auto_arima(y, seasonal=True, m=7)  # m=7 for weekly seasonality on daily data
                m = infer_season_length(y, max_lag=30, threshold=0.3)
                '''
                auto_model = pm.auto_arima(
                    y,
                    start_p=0, start_q=0,  
                    max_p=2, max_q=2,      # keep small since your system has 6GB RAM (3,3)
                    d=None,                # let auto_arima decide
                    start_P=0, start_Q=0,  
                    max_P=1, max_Q=1,      # was (2,2)
                    D=None,
                    m=m,                   # season length (e.g. 7 = weekly seasonality for daily data)
                    seasonal=True,
                    stepwise=True,         # faster stepwise search
                    suppress_warnings=True,
                    error_action="ignore", # continue even if a model fails
                    trace=False             # show models it tries (True)
                )
                '''
    
    
                
                auto_model = pm.auto_arima(
                    y,
                    start_p=0, start_q=0,
                    max_p=2, max_q=2,      # smaller max orders → fewer models
                    d=None,                # let auto_arima decide
                    start_P=0, start_Q=0,
                    max_P=1, max_Q=1,      # smaller seasonal orders
                    D=None,
                    m=m,                   # seasonal length
                    seasonal=True,
                    stepwise=True,         # enable stepwise search (faster than full search)
                    max_order=3,           # sum of p+q+P+Q ≤ 3 → reduces combinations
                    max_d=2,               # restrict differencing search
                    max_D=1,               # restrict seasonal differencing
                    n_jobs=1,              # parallel jobs if supported
                    suppress_warnings=True,
                    error_action="ignore",
                    trace=False            # disable verbose output
                )
              
                if auto_model is not None:
                    # Extract best orders found by auto_arima
                    sarimax_order = auto_model.order
                    sarimax_seasonal_order = auto_model.seasonal_order
                    # Log the AIC
                    #print(f"Best auto_arima model: {auto_model.summary()}")
                    #print(f"AIC: {auto_model.aic()}")
                    aic=auto_model.aic()
                    mltype="automl"
                else:
                    sarimax_order = (1,0,0)
                    sarimax_seasonal_order = (0,0,0,0)
                    aic=0
                    mltype="sarimax"
    
                    
                model = SARIMAX(y,
                                order=sarimax_order,
                                seasonal_order=sarimax_seasonal_order,
                                enforce_stationarity=False,
                                enforce_invertibility=False)
                sarimax_res = model.fit(disp=False)
                forecast = sarimax_res.get_forecast(steps=forecast_horizon)
                pred_price = forecast.predicted_mean.iloc[-1] # iloc[-1] gets the last Value in the prediction results
                last_close = y.iloc[-1]
                sarimax_return = (pred_price - last_close) / last_close
            except Exception as e:
                print(f"⚠️ SARIMAX failed for {cid}-{tf}: {e}")
                pred_price = last_close
                sarimax_return = 0.0
            
            # -----------------------------
            # ML Prediction from existing Phase 2 columns
            # -----------------------------
            ml_return = df_c["MaxPredictedReturn"].iloc[0] if "MaxPredictedReturn" in df_c.columns else 0.0
            
            # -----------------------------
            # Weighted score
            # -----------------------------
            weighted_score = ml_weight * ml_return + sarimax_weight * sarimax_return
            
            # -----------------------------
            # Store enriched data
            # -----------------------------
            df_c["SARIMAX_PredictedClose"] = pred_price
            df_c["SARIMAX_PredictedReturn"] = sarimax_return
            df_c["WeightedScore"] = weighted_score
            df_c["AIC"] = aic
            df_c["MlType"] = mltype
            phase3_results.append(df_c)
    
    
    
    # -----------------------------
    # Combine all companies/timeframes
    # -----------------------------
    df_phase3_full = pd.concat(phase3_results, ignore_index=True)
    df_phase3_spark = spark.createDataFrame(df_phase3_full)
    

    # -----------------------------
    # Enrich all companies with company info
    # -----------------------------
    sdf_company = spark.table("bsf.company").select("CompanyId", "TradingSymbol", "Name")
    df_phase3_enriched = df_phase3_spark.join(F.broadcast(sdf_company), on="CompanyId", how="left")
    
    # 1️⃣ Keep only the latest row per company + timeframe
    window_last = Window.partitionBy("CompanyId", "TimeFrame").orderBy(F.desc("StockDate"))
    df_phase3_enriched = (
        df_phase3_enriched
        .withColumn("rn", F.row_number().over(window_last))
        .filter(F.col("rn") == 1)
        .drop("rn")
    )
    
    # 2️⃣ Rank companies by WeightedScore per timeframe (top N only)
    window_tf = Window.partitionBy("TimeFrame").orderBy(F.desc("WeightedScore"))
    df_topN_companies = (
        df_phase3_enriched
        .withColumn("Phase3_Rank", F.row_number().over(window_tf))
        .filter(F.col("Phase3_Rank") <= top_n_final)
    )
    

    
    timeframes = [row["TimeFrame"] for row in df_phase3_enriched.select("TimeFrame").distinct().collect()]
    
    # Create a dict of DataFrames filtered by timeframe
    phase3_top_dfs = {tf: df_phase3_enriched.filter(F.col("TimeFrame") == tf) for tf in timeframes}
    topN_companies_df = {tf: df_topN_companies.filter(F.col("TimeFrame") == tf) for tf in timeframes}
    # -----------------------------
    # Optional: show counts
    # -----------------------------
    #for tf in timeframes:
    #    print(f"       ❗{tf}: Final top N = {phase3_top_dfs[tf].count()}")
    for tf in timeframes:
        print(f"       ❗{tf}: Final top N = {topN_companies_df[tf].count()}")
    # -----------------------------
    # Generate a timestamp string
    # -----------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df_phase3_enriched = df_phase3_enriched.withColumn("RunTimestamp", F.lit(timestamp))
    cols = ["RunTimestamp"] + [c for c in df_phase3_enriched.columns if c != "RunTimestamp"]
    df_phase3_enriched = df_phase3_enriched.select(cols)
    print(f"     ✅ Stage 3 completed: Latest rows per company + top {top_n_final} candidates selected per timeframe")




    return df_phase3_enriched, df_topN_companies, phase3_top_dfs, topN_companies_df

