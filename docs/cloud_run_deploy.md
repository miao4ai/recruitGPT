# Deploying the Data Pipeline to Cloud Run Jobs

The data generation pipeline (JD synthesis → resume synthesis → distillation → filtering → GCS/BQ upload)
runs as a **Cloud Run Job** — a one-off batch container, not a persistent server.

---

## Prerequisites

```bash
# Install gcloud CLI and authenticate
gcloud auth login
gcloud auth configure-docker us-central1-docker.pkg.dev

# Enable required APIs
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com \
  bigquery.googleapis.com \
  storage.googleapis.com
```

---

## 1. Store secrets in Secret Manager

```bash
# Anthropic API key
echo -n "sk-ant-..." | gcloud secrets create ANTHROPIC_API_KEY --data-file=-

# (Optional) if using OpenAI or DeepSeek as teacher
echo -n "sk-..." | gcloud secrets create OPENAI_API_KEY --data-file=-
```

---

## 2. Create Artifact Registry repo

```bash
gcloud artifacts repositories create recruitgpt \
  --repository-format=docker \
  --location=us-central1
```

---

## 3. Build and push Docker image

```bash
IMAGE=us-central1-docker.pkg.dev/YOUR_PROJECT/recruitgpt/pipeline:latest

docker build -t $IMAGE .
docker push $IMAGE
```

Or use Cloud Build (no local Docker needed):

```bash
gcloud builds submit --tag $IMAGE .
```

---

## 4. Create GCS bucket and BigQuery dataset

```bash
# GCS bucket for outputs
gsutil mb -l us-central1 gs://recruitgpt-dev

# BigQuery dataset (tables are auto-created by upload_to_bigquery.py)
bq mk --dataset --location=US YOUR_PROJECT:recruitgpt_eval
```

---

## 5. Create the Cloud Run Job

```bash
gcloud run jobs create recruitgpt-pipeline \
  --image $IMAGE \
  --region us-central1 \
  --task-timeout 3600 \
  --max-retries 1 \
  --set-secrets ANTHROPIC_API_KEY=ANTHROPIC_API_KEY:latest \
  --set-env-vars \
    GCP_PROJECT=YOUR_PROJECT,\
    GCS_BUCKET=recruitgpt-dev,\
    BQ_DATASET=recruitgpt_eval,\
    NUM_JDS=200,\
    NUM_RESUMES=200,\
    NUM_DISTILL=1500,\
    DISTILL_TASKS=query_parsing,\
    DISTILL_MODE=standard
```

---

## 6. Run the job

```bash
# Execute immediately
gcloud run jobs execute recruitgpt-pipeline --region us-central1

# Watch logs
gcloud run jobs executions list --job recruitgpt-pipeline --region us-central1
gcloud logging read "resource.type=cloud_run_job AND resource.labels.job_name=recruitgpt-pipeline" \
  --limit 100 --format "value(textPayload)"
```

---

## 7. Update config without rebuilding

Change env vars (e.g. increase sample count) without a new Docker build:

```bash
gcloud run jobs update recruitgpt-pipeline \
  --region us-central1 \
  --update-env-vars NUM_DISTILL=3000
```

---

## Pipeline environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | **Required.** Set via Secret Manager |
| `GCP_PROJECT` | — | **Required.** GCP project ID |
| `GCS_BUCKET` | — | **Required.** Output bucket name |
| `BQ_DATASET` | `recruitgpt_eval` | BigQuery dataset |
| `NUM_JDS` | `200` | JDs to synthesise |
| `NUM_RESUMES` | `200` | Resumes to synthesise |
| `NUM_DISTILL` | `1500` | Distillation samples per task |
| `DISTILL_TASKS` | `query_parsing` | `query_parsing` and/or `match_explanation` |
| `DISTILL_MODE` | `standard` | `standard` (sync) or `batch` (async, 50% cheaper) |
| `RUN_ID` | auto (timestamp) | Label for GCS path and BQ run_id |

---

## Cost estimate (one full run)

| Step | Duration | Cost |
|------|----------|------|
| Cloud Run Job (1 vCPU, 2 GB, ~60 min) | ~1 hr | ~$0.05 |
| Claude API — 1,500 query_parsing samples | — | ~$7 |
| Claude API — 200 JDs + 200 resumes | — | ~$1 |
| GCS storage (~50 MB output) | ongoing | <$0.01/mo |
| **Total per run** | | **~$8** |

Use `DISTILL_MODE=batch` to cut the distillation API cost by ~50% (~$3.50 instead of ~$7).
