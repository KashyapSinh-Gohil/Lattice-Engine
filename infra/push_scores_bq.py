"""Push AEGIS pipeline score artifacts to BigQuery (aegis_scores.*) for Looker Studio.

Usage: python infra/push_scores_bq.py --out out/gpu_run --project my-proj
"""
import argparse
import json

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/gpu_run")
    ap.add_argument("--project", required=True)
    a = ap.parse_args()

    from google.cloud import bigquery
    client = bigquery.Client(project=a.project)

    feeders = pd.DataFrame(json.load(open(f"{a.out}/feeders.json")))
    feeders["reason_codes"] = feeders["reason_codes"].apply(", ".join)
    feeders = feeders.drop(columns=["components", "spark_mw", "forecast_mw"], errors="ignore")

    txs = pd.DataFrame(json.load(open(f"{a.out}/transformers.json")))
    txs["reason_codes"] = txs["reason_codes"].apply(", ".join)
    txs = txs.drop(columns=["shap"], errors="ignore")

    for name, df in [("feeder_scores", feeders), ("transformer_scores", txs)]:
        job = client.load_table_from_dataframe(
            df, f"{a.project}.aegis_scores.{name}",
            job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE"))
        job.result()
        print(f"pushed {len(df)} rows -> aegis_scores.{name}")


if __name__ == "__main__":
    main()
