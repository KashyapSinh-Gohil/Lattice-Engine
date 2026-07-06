"""
AEGIS historical backfill on Dataproc Managed Spark + NVIDIA Spark RAPIDS.

The nightly/backfill path: re-computes feeder & transformer aggregates over the FULL
history (500M+ rows) in GCS/BigQuery. Same logic as pipeline stages 2-4, expressed in
Spark SQL so the RAPIDS Accelerator executes joins/groupbys on GPUs across the cluster.

Submit with infra/spark_rapids/submit.sh — the cluster is created with GPU workers and
the RAPIDS accelerator init action; spark.plugins=com.nvidia.spark.SQLPlugin does the rest.
Verify GPU execution in the Spark UI: physical plan shows GpuHashAggregate / GpuShuffledHashJoin.
"""
import sys

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

RAW = sys.argv[1] if len(sys.argv) > 1 else "gs://YOUR_BUCKET/raw/city_100m"
OUT = sys.argv[2] if len(sys.argv) > 2 else "gs://YOUR_BUCKET/curated"

spark = (SparkSession.builder.appName("aegis-backfill").getOrCreate())

readings = spark.read.parquet(f"{RAW}/readings_*.parquet")
meters = spark.read.parquet(f"{RAW}/meters.parquet")
tx = spark.read.parquet(f"{RAW}/transformers.parquet")

clean = (readings
         .withColumn("ts", F.date_trunc("minute", F.col("ts")))
         .dropDuplicates(["meter_id", "ts"])
         .withColumn("kwh", F.when(F.col("kwh") > 100, F.col("kwh") / 1000.0)
                              .otherwise(F.col("kwh")))
         .where(F.col("kwh") >= 0)
         .join(meters.select("meter_id", "transformer_id"), "meter_id"))

tx_interval = (clean
               .join(tx.select("transformer_id", "feeder_id", "capacity_kva"), "transformer_id")
               .groupBy("transformer_id", "feeder_id", "capacity_kva", "ts")
               .agg(F.sum(F.col("kwh") * 4).alias("kw"),
                    F.sum(F.when(F.col("voltage") < 207, 1).otherwise(0)).alias("sags"))
               .withColumn("loading", F.col("kw") / (F.col("capacity_kva") * 0.9)))

feeder_daily = (tx_interval
                .withColumn("d", F.to_date("ts"))
                .groupBy("feeder_id", "d")
                .agg(F.max("kw").alias("peak_kw"), F.avg("kw").alias("mean_kw"),
                     F.sum(F.when(F.col("loading") > 1.0, 15).otherwise(0))
                      .alias("overload_minutes"),
                     F.sum("sags").alias("sags")))

tx_interval.write.mode("overwrite").parquet(f"{OUT}/tx_interval")
feeder_daily.write.mode("overwrite").parquet(f"{OUT}/feeder_daily")
print("backfill complete:", feeder_daily.count(), "feeder-days")
spark.stop()
