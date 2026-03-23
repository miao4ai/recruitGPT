"""
Upload generated data to BigQuery for tracking and analysis.

Supports three data types:

  distill    — distillation training samples (train_raw / train_clean)
  eval       — LLM judge evaluation results
  pipeline   — end-to-end pipeline run results (match scores, latency)

Tables are created automatically if they don't exist.
Rows are inserted with WRITE_APPEND so re-runs accumulate history.

Setup:
  1. Set GCP_PROJECT and BQ_DATASET in .env (or pass via --project / --dataset)
  2. Authenticate: gcloud auth application-default login

Usage:
  # Upload distillation data after generation
  python scripts/upload_to_bigquery.py distill \
      --input data/generated/train_clean.jsonl \
      --run_id "run_20260322"

  # Upload eval results
  python scripts/upload_to_bigquery.py eval \
      --input eval/eval_results.jsonl \
      --run_id "qwen_v1"

  # Upload pipeline match results
  python scripts/upload_to_bigquery.py pipeline \
      --input data/pipeline_runs/run_001.jsonl \
      --run_id "run_001"
"""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from google.cloud import bigquery

load_dotenv()


# ---------------------------------------------------------------
# BigQuery schemas
# ---------------------------------------------------------------

SCHEMAS = {
    "distill_samples": [
        bigquery.SchemaField("run_id",         "STRING",    mode="REQUIRED"),
        bigquery.SchemaField("uploaded_at",    "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("sample_id",      "STRING",    mode="REQUIRED"),
        bigquery.SchemaField("task",           "STRING",    mode="REQUIRED"),
        bigquery.SchemaField("source_jd_id",   "STRING",    mode="NULLABLE"),
        bigquery.SchemaField("source_resume_id","STRING",   mode="NULLABLE"),
        bigquery.SchemaField("input_text",     "STRING",    mode="REQUIRED"),
        bigquery.SchemaField("output_text",    "STRING",    mode="REQUIRED"),
        bigquery.SchemaField("input_chars",    "INTEGER",   mode="NULLABLE"),
        bigquery.SchemaField("output_chars",   "INTEGER",   mode="NULLABLE"),
        bigquery.SchemaField("passed_filter",  "BOOLEAN",   mode="NULLABLE"),
    ],
    "llm_eval_results": [
        bigquery.SchemaField("run_id",         "STRING",    mode="REQUIRED"),
        bigquery.SchemaField("evaluated_at",   "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("sample_id",      "STRING",    mode="REQUIRED"),
        bigquery.SchemaField("task",           "STRING",    mode="REQUIRED"),
        bigquery.SchemaField("model_path",     "STRING",    mode="NULLABLE"),
        bigquery.SchemaField("judge_model",    "STRING",    mode="NULLABLE"),
        bigquery.SchemaField("accuracy",       "FLOAT64",   mode="NULLABLE"),
        bigquery.SchemaField("format_score",   "FLOAT64",   mode="NULLABLE"),
        bigquery.SchemaField("usefulness",     "FLOAT64",   mode="NULLABLE"),
        bigquery.SchemaField("judge_comment",  "STRING",    mode="NULLABLE"),
    ],
    "pipeline_runs": [
        bigquery.SchemaField("run_id",         "STRING",    mode="REQUIRED"),
        bigquery.SchemaField("run_at",         "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("jd_id",          "STRING",    mode="NULLABLE"),
        bigquery.SchemaField("candidate_id",   "STRING",    mode="NULLABLE"),
        bigquery.SchemaField("rank",           "INTEGER",   mode="NULLABLE"),
        bigquery.SchemaField("retrieval_score","FLOAT64",   mode="NULLABLE"),
        bigquery.SchemaField("reranker_score", "FLOAT64",   mode="NULLABLE"),
        bigquery.SchemaField("graph_score",    "FLOAT64",   mode="NULLABLE"),
        bigquery.SchemaField("final_score",    "FLOAT64",   mode="NULLABLE"),
        bigquery.SchemaField("latency_ms",     "INTEGER",   mode="NULLABLE"),
    ],
}


# ---------------------------------------------------------------
# BQ helpers
# ---------------------------------------------------------------

def get_or_create_table(client: bigquery.Client, dataset_id: str, table_id: str) -> bigquery.Table:
    table_ref = f"{client.project}.{dataset_id}.{table_id}"
    try:
        return client.get_table(table_ref)
    except Exception:
        schema = SCHEMAS[table_id]
        table = bigquery.Table(table_ref, schema=schema)
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field="uploaded_at" if table_id == "distill_samples" else
                  "evaluated_at" if table_id == "llm_eval_results" else "run_at",
        )
        table = client.create_table(table)
        print(f"Created table: {table_ref}")
        return table


def insert_rows(client: bigquery.Client, table: bigquery.Table, rows: list[dict]):
    errors = client.insert_rows_json(table, rows)
    if errors:
        for e in errors:
            print(f"  BQ insert error: {e}")
        raise RuntimeError(f"{len(errors)} rows failed to insert")


# ---------------------------------------------------------------
# Data transformers
# ---------------------------------------------------------------

def _get_messages_texts(sample: dict) -> tuple[str, str]:
    """Extract user input and assistant output from messages list."""
    input_parts, output = [], ""
    messages = sample.get("messages", [])
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in ("system", "user"):
            input_parts.append(content)
        elif role == "assistant":
            output = content  # last assistant turn is the label
    return "\n".join(input_parts), output


def transform_distill(sample: dict, run_id: str, now: str) -> dict:
    input_text, output_text = _get_messages_texts(sample)
    return {
        "run_id":          run_id,
        "uploaded_at":     now,
        "sample_id":       sample.get("id", ""),
        "task":            sample.get("task", ""),
        "source_jd_id":    sample.get("source_jd_id"),
        "source_resume_id":sample.get("source_resume_id"),
        "input_text":      input_text,
        "output_text":     output_text,
        "input_chars":     len(input_text),
        "output_chars":    len(output_text),
        "passed_filter":   sample.get("passed_filter"),
    }


def transform_eval(sample: dict, run_id: str, now: str) -> dict:
    return {
        "run_id":        run_id,
        "evaluated_at":  now,
        "sample_id":     sample.get("sample_id", sample.get("id", "")),
        "task":          sample.get("task", ""),
        "model_path":    sample.get("model_path"),
        "judge_model":   sample.get("judge_model"),
        "accuracy":      sample.get("accuracy"),
        "format_score":  sample.get("format_score"),
        "usefulness":    sample.get("usefulness"),
        "judge_comment": sample.get("judge_comment"),
    }


def transform_pipeline(sample: dict, run_id: str, now: str) -> dict:
    return {
        "run_id":          run_id,
        "run_at":          now,
        "jd_id":           sample.get("jd_id"),
        "candidate_id":    sample.get("candidate_id"),
        "rank":            sample.get("rank"),
        "retrieval_score": sample.get("retrieval_score"),
        "reranker_score":  sample.get("reranker_score"),
        "graph_score":     sample.get("graph_score"),
        "final_score":     sample.get("final_score"),
        "latency_ms":      sample.get("latency_ms"),
    }


TRANSFORMERS = {
    "distill":  (transform_distill,  "distill_samples"),
    "eval":     (transform_eval,     "llm_eval_results"),
    "pipeline": (transform_pipeline, "pipeline_runs"),
}


# ---------------------------------------------------------------
# Main upload
# ---------------------------------------------------------------

def upload(data_type: str, input_path: Path, run_id: str, project: str, dataset: str,
           batch_size: int = 500, dry_run: bool = False):

    transform_fn, table_id = TRANSFORMERS[data_type]
    now = datetime.now(timezone.utc).isoformat()

    # Load records
    records = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    print(f"Loaded {len(records)} records from {input_path}")

    # Transform
    rows = [transform_fn(r, run_id, now) for r in records]

    if dry_run:
        print(f"\n--dry_run: would upload {len(rows)} rows to {project}.{dataset}.{table_id}")
        print("Sample row:")
        print(json.dumps(rows[0], indent=2, default=str))
        return

    # Upload
    client = bigquery.Client(project=project)
    table = get_or_create_table(client, dataset, table_id)

    total = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        insert_rows(client, table, batch)
        total += len(batch)
        print(f"  Uploaded {total}/{len(rows)} rows...")

    print(f"\nDone. {total} rows → {project}.{dataset}.{table_id}")


# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Upload generated data to BigQuery")
    parser.add_argument("type", choices=["distill", "eval", "pipeline"],
                        help="Data type to upload")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input jsonl file")
    parser.add_argument("--run_id", type=str, required=True,
                        help="Run identifier, e.g. 'run_20260322' or 'qwen_v1'")
    parser.add_argument("--project", type=str,
                        default=os.getenv("GCP_PROJECT"),
                        help="GCP project ID (or set GCP_PROJECT in .env)")
    parser.add_argument("--dataset", type=str,
                        default=os.getenv("BQ_DATASET", "recruitgpt_eval"),
                        help="BigQuery dataset ID (or set BQ_DATASET in .env)")
    parser.add_argument("--batch_size", type=int, default=500,
                        help="Rows per BQ insert batch (max 10,000)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Transform and print sample row without uploading")
    args = parser.parse_args()

    if not args.project:
        raise ValueError("GCP project not set — use --project or set GCP_PROJECT in .env")

    upload(
        data_type=args.type,
        input_path=Path(args.input),
        run_id=args.run_id,
        project=args.project,
        dataset=args.dataset,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
