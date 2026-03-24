"""
Cloud Run Job entry point — runs the full data generation pipeline.

Steps:
  1. generate_jds       — synthesise job descriptions via Claude
  2. generate_resumes   — synthesise LinkedIn-style resumes via Claude
  3. distill_data       — run Query Parsing distillation (teacher → student labels)
  4. filter_data        — quality-filter distilled samples
  5. upload_gcs         — push all outputs to GCS
  6. upload_bq          — append distilled samples to BigQuery

Environment variables (set via Cloud Run --set-secrets / --set-env-vars):
  ANTHROPIC_API_KEY     required
  GCP_PROJECT           required
  GCS_BUCKET            required
  BQ_DATASET            default: recruitgpt_eval
  NUM_JDS               default: 200
  NUM_RESUMES           default: 200
  NUM_DISTILL           default: 1500
  DISTILL_TASKS         default: query_parsing
  RUN_ID                default: auto (YYYYMMDD-HHMMSS)
  DISTILL_MODE          default: standard  (or "batch" for async Batch API)
"""

import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def run_step(label: str, cmd: list[str]):
    print(f"\n{'='*60}")
    print(f"STEP: {label}")
    print(f"CMD:  {' '.join(cmd)}")
    print("="*60)
    result = subprocess.run(cmd, check=True)
    return result


def upload_dir_to_gcs(local_dir: str, gcs_prefix: str, bucket: str):
    """Upload a local directory to GCS using gsutil rsync."""
    gcs_path = f"gs://{bucket}/{gcs_prefix}"
    print(f"Uploading {local_dir} → {gcs_path}")
    subprocess.run(
        ["gsutil", "-m", "rsync", "-r", local_dir, gcs_path],
        check=True,
    )


def main():
    # -------------------------------------------------------
    # Config from environment
    # -------------------------------------------------------
    run_id = env("RUN_ID") or datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    num_jds = env("NUM_JDS", "200")
    num_resumes = env("NUM_RESUMES", "200")
    num_distill = env("NUM_DISTILL", "1500")
    distill_tasks = env("DISTILL_TASKS", "query_parsing")
    distill_mode = env("DISTILL_MODE", "standard")
    bucket = env("GCS_BUCKET")
    project = env("GCP_PROJECT")
    bq_dataset = env("BQ_DATASET", "recruitgpt_eval")

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY is not set")
    if not bucket:
        raise ValueError("GCS_BUCKET is not set")
    if not project:
        raise ValueError("GCP_PROJECT is not set")

    print(f"Run ID: {run_id}")
    print(f"Config: {num_jds} JDs, {num_resumes} resumes, {num_distill} distill samples")
    print(f"Tasks:  {distill_tasks}  |  Mode: {distill_mode}")

    # -------------------------------------------------------
    # Step 1: Generate JDs
    # -------------------------------------------------------
    run_step("Generate JDs", [
        "python", "scripts/generate_jds.py",
        "--num", num_jds,
        "--output", "data/jds/",
    ])

    # -------------------------------------------------------
    # Step 2: Generate resumes (only needed for match_explanation)
    # -------------------------------------------------------
    if "match_explanation" in distill_tasks:
        run_step("Generate resumes", [
            "python", "scripts/generate_resumes.py",
            "--num", num_resumes,
            "--output", "data/resumes/",
        ])
    else:
        print("\nSkipping resume generation (not needed for query_parsing only)")

    # -------------------------------------------------------
    # Step 3: Distill data
    # -------------------------------------------------------
    distill_cmd = [
        "python", "scripts/distill_data.py",
        "--tasks", distill_tasks,
        "--num", num_distill,
        "--output", "data/generated/train_raw.jsonl",
        "--resume",
    ]
    if distill_mode == "batch":
        distill_cmd.append("--batch")

    run_step("Distill data", distill_cmd)

    # Batch mode: job ends here; fetch_batch is a separate run
    if distill_mode == "batch":
        print("\nBatch submitted. Re-run with DISTILL_MODE=fetch and BATCH_ID=<id> to collect results.")
        return

    # -------------------------------------------------------
    # Step 4: Filter data
    # -------------------------------------------------------
    run_step("Filter data", [
        "python", "scripts/filter_data.py",
        "--input", "data/generated/train_raw.jsonl",
        "--output", "data/generated/train_clean.jsonl",
    ])

    # -------------------------------------------------------
    # Step 5: Upload to GCS
    # -------------------------------------------------------
    print(f"\n{'='*60}")
    print("STEP: Upload outputs to GCS")
    print("="*60)
    for local, remote in [
        ("data/jds/",       f"runs/{run_id}/jds"),
        ("data/resumes/",   f"runs/{run_id}/resumes"),
        ("data/generated/", f"runs/{run_id}/generated"),
    ]:
        if Path(local).exists():
            upload_dir_to_gcs(local, remote, bucket)

    # -------------------------------------------------------
    # Step 6: Upload distilled samples to BigQuery
    # -------------------------------------------------------
    clean_path = Path("data/generated/train_clean.jsonl")
    if clean_path.exists() and clean_path.stat().st_size > 0:
        run_step("Upload to BigQuery", [
            "python", "scripts/upload_to_bigquery.py", "distill",
            "--input", str(clean_path),
            "--run_id", run_id,
            "--project", project,
            "--dataset", bq_dataset,
        ])
    else:
        print("\nNo clean data found — skipping BigQuery upload")

    print(f"\n{'='*60}")
    print(f"Pipeline complete. Run ID: {run_id}")
    print(f"GCS:       gs://{bucket}/runs/{run_id}/")
    print(f"BigQuery:  {project}.{bq_dataset}.distill_samples")
    print("="*60)


if __name__ == "__main__":
    main()
