# bsf_env.py

from sqlalchemy import create_engine
from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip
import os

# Normal verbose Spark
#spark = init_spark("demo1", log_level="INFO", show_progress=True)

# Mostly quiet (errors only, but progress bar still visible)
#spark = init_spark("demo2", log_level="QUIET", show_progress=True)

# Fully silent (errors only, no progress bar)
#spark = init_spark("build_candidates", log_level="SILENT")

# Explicit "ERROR" level (same as QUIET, just no alias)
#spark = init_spark("demo4", log_level="ERROR", show_progress=False)

# Start normally, perhaps with SILENT
#spark = init_spark("build_candidates", log_level="WARN", show_progress=False)

# Do quiet computations
# ...

# Temporarily restore verbose logs for debugging
#set_spark_verbose(spark, level="INFO", show_progress=True)

# After debugging, silence again
#set_spark_quiet(spark)

def init_spark(app_name: str = None,
               log_level: str = "ERROR",
               show_progress: bool = False,
               enable_ui: bool = True,
               process_option: str = "default") -> SparkSession:
    """
    Initialize a SparkSession with Delta and Hive support.

    Parameters
    ----------
    app_name : str
        Name of the Spark application.
    log_level : str
        Spark log level: ALL, DEBUG, INFO, WARN, ERROR, FATAL, OFF.
        Extra aliases:
            - QUIET  : ERROR level, but respects show_progress
            - SILENT : ERROR level, and disables progress bar
    show_progress : bool
        Whether to show console progress bar (ignored if log_level=SILENT).
    """
    # Normalize log level
    effective_level = log_level.upper()

    if effective_level == "QUIET":
        effective_level = "ERROR"

    elif effective_level == "SILENT":
        effective_level = "ERROR"
        show_progress = False  # enforce fully silent

    postgres_jar = "/opt/spark/jars/postgresql-42.7.5.jar"
    mariadb_jar = "/opt/spark/maria/mariadb-java-client-3.5.2.jar"
    
    # Default job name if not provided
    if app_name is None:
        app_name = "bsf_default_spark_job"

    
    # Build the Spark session
    builder = (
        SparkSession.builder
        .appName(app_name)
        # Optional UI settings
        .config("spark.ui.showConsoleProgress", str(show_progress).lower())
        .config("spark.ui.enabled", str(enable_ui).lower())
    )
    
    # --- Big write optimizations ---
    if process_option == 'wide':
        # Wide: fewer but bigger tasks (less parallelism, heavier memory per task)
        builder = (
            builder
            .config("spark.cores.max", 4)                 # all 4 cores
            .config("spark.executor.instances", 1)        # single executor (best for 1 node)
            .config("spark.executor.cores", 4)            # give executor all 4 cores
            .config("spark.task.cpus", 1)
            .config("spark.executor.memory", "4096m")     # 4 GB executor memory
            .config("spark.executor.memoryOverhead", "1024m")  # 1 GB overhead
            .config("spark.driver.memory", "2048m")       # 2 GB driver memory
            .config("spark.sql.shuffle.partitions", 8)    # 2× cores
            .config("spark.default.parallelism", 8)       # match shuffle partitions
        )
    elif process_option == 'tall':
        # Tall: more parallelism, smaller tasks
        builder = (
            builder
            .config("spark.cores.max", 3)
            .config("spark.executor.instances", 1)
            .config("spark.executor.cores", 3)
            .config("spark.task.cpus", 1)
            .config("spark.executor.memory", "2048m")
            .config("spark.driver.memory", "2048m")
            .config("spark.executor.memoryOverhead", "512m")  # optional, JVM overhead
            .config("spark.sql.shuffle.partitions", 12) # slightly more partitions for parallelism
            .config("spark.default.parallelism", 12)
            .config("spark.scheduler.pool", "highPriority")
        )
    else:
        # fallback: use default settings
        pass

    


    # Create the session
 
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
     
    # Set SparkContext log level
    spark.sparkContext.setLogLevel(effective_level)

    # Suppress noisy subsystems too
    log4j = spark._jvm.org.apache.log4j
    for pkg in [
        "org",
        "akka",
        "org.apache.spark",
        "org.apache.hadoop",
        "org.apache.hive",
        "org.apache.delta"
    ]:
        log4j.LogManager.getLogger(pkg).setLevel(
            getattr(log4j.Level, effective_level)
        )

    print(f"[Spark] Started '{app_name}' "
          f"log_level={log_level.upper()} (effective={effective_level}), "
          f"progress={show_progress}")

    return spark

def set_spark_verbosity(level="ERROR"):
    spark.sparkContext.setLogLevel(level)
    #spark.conf.set("spark.ui.showConsoleProgress", str(show_progress).lower())
    # ─────────────────────────────────────────────
    # 🚀 Usage
    # ─────────────────────────────────────────────
    '''
    set_spark_verbosity("ERROR")  # quiet mode
    set_spark_verbosity("INFO")    # verbose mode
    '''


def init_mariadb_engine(
    host: str = "nond2rd-d19",
    port: int = 3306,
    database: str = "bsf",
    user: str = "bsf",
    password: str = "TMeJNbSTca834uGGy3Gj"
    ):
    """
    Initialize a SQLAlchemy engine for MariaDB.
    """
    url = f"mysql+mariadbconnector://{user}:{password}@{host}:{port}/{database}"
    return create_engine(url)

