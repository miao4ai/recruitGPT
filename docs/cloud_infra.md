# Cloud Infrastructure Guide

This document covers the cloud resources needed to run the full recruitGPT training pipeline: distillation data generation, embedding fine-tuning, and LLM fine-tuning.

---

## Overview

```
┌─────────────────────────────────────────────────────┐
│                   Storage Layer                     │
│         GCS Bucket  ·  BigQuery  ·  HuggingFace     │
└──────────────┬─────────────────────────┬────────────┘
               │                         │
       ┌───────▼────────┐       ┌────────▼───────┐
       │  Distillation  │       │  Fine-tuning   │
       │  (API calls)   │       │  (GPU jobs)    │
       │  Cloud Run /   │       │  Vertex AI     │
       │  local script  │       │  Training      │
       └───────┬────────┘       └────────┬───────┘
               │                         │
               └──────────┬──────────────┘
                          │
                 ┌────────▼────────┐
                 │  Model Registry │
                 │  (Vertex AI /   │
                 │   HuggingFace)  │
                 └─────────────────┘
```

---

## 1. Storage

### GCS Bucket Layout

One bucket, three top-level prefixes:

```
gs://recruitgpt-{env}/
│
├── raw/
│   ├── resumes/          # source resume files (txt / pdf)
│   └── jds/              # source job description files
│
├── data/
│   ├── generated/        # distilled LLM outputs (jsonl)
│   │   ├── train_raw.jsonl
│   │   └── train_clean.jsonl
│   ├── pairs/            # embedding triplets
│   │   ├── train_triplets.jsonl
│   │   └── hard_negatives.jsonl
│   └── reranker/         # reranker training pairs
│       ├── train.jsonl
│       └── eval.jsonl
│
├── eval/
│   ├── eval_set.jsonl           # LLM judge evaluation set
│   └── retrieval_benchmark.jsonl
│
└── artifacts/
    ├── bge-recruit/             # fine-tuned BGE checkpoint
    ├── reranker/                # fine-tuned reranker checkpoint
    └── qwen3_5_0_8b-recruit/   # merged QLoRA checkpoint
        └── merged/
```

**Recommended GCS settings:**

| Setting | Value | Reason |
|---------|-------|--------|
| Storage class | Standard | Frequent access during training |
| Location | Same region as Vertex AI jobs | Avoid egress costs |
| Versioning | On for `data/` prefix | Recover from bad distillation runs |
| Lifecycle rule | Move `artifacts/` to Nearline after 90 days | Cost reduction |

### BigQuery (optional, for evaluation tracking)

```sql
-- One table per eval run, partitioned by date
dataset: recruitgpt_eval
tables:
  - llm_eval_results       -- judge scores per sample
  - retrieval_eval_results -- Recall@K, MRR per run
  - pipeline_runs          -- end-to-end latency + quality metrics
```

---

## 2. Distillation Data Generation

Teacher model (DeepSeek / GPT-4o / Claude) generates ~3,000 training samples via API.

### Where to run

**Option A — Local script** (simplest, one-time run)
```bash
python scripts/distill_data.py \
    --teacher deepseek \
    --tasks query_parsing,match_explanation \
    --num_per_task 1500 \
    --output gs://recruitgpt-dev/data/generated/train_raw.jsonl
```

**Option B — Cloud Run job** (if generating at scale or on a schedule)
- Stateless container, no GPU needed
- Set `DEEPSEEK_API_KEY` / `OPENAI_API_KEY` via Secret Manager
- Recommended instance: `1 vCPU, 2 GB RAM`
- Estimated runtime: 30–60 min for 3,000 samples

### Token volume estimate (3,000 samples)

| Task | Samples | Input tokens/sample | Output tokens/sample | Total |
|------|---------|--------------------|--------------------|-------|
| Query Parsing | 1,500 | ~650 (prompt + JD) | ~200 (JSON) | 975K in + 300K out |
| Match Explanation | 1,500 | ~1,350 (prompt + JD + resume) | ~400 (report) | 2.0M in + 600K out |
| **Total** | **3,000** | | | **~3M in + 900K out** |

Match Explanation accounts for ~70% of cost due to the longer input context (JD + resume combined).

### API cost estimate

| Teacher | Input price/M | Output price/M | ~3,000 samples (standard) | Batch mode (50% off) |
|---------|--------------|---------------|--------------------------|---------------------|
| **DeepSeek-V3** | $0.27 | $1.10 | **~$1.8** | ~$0.9 |
| GPT-4o-mini | $0.15 | $0.60 | **~$1.0** | ~$0.5 |
| Claude Haiku | $0.80 | $4.00 | **~$6.0** | ~$3.0 |
| **Claude Sonnet 4.6** | $3.00 | $15.00 | **~$22.5** | **~$11** (Batch API) |
| GPT-4o | $2.50 | $10.00 | **~$16.5** | ~$8 |

> Prices as of March 2026. Always verify at the provider's pricing page before running.

**Recommendation:** Use **DeepSeek-V3** for cost efficiency. If higher output quality is needed (e.g., for match explanations), use **Claude Sonnet via Batch API** (~$11) — distillation data generation is non-realtime and qualifies for batch processing.

### Data format (stored in GCS as jsonl)

```jsonl
{"task": "query_parsing", "input": "<JD text>", "output": {"skills": [...], "seniority": "senior", ...}}
{"task": "match_explanation", "input": "<JD>\n<Resume>", "output": "Candidate demonstrates..."}
```

After generation, run quality filtering before training:
```bash
python scripts/filter_data.py \
    --input gs://recruitgpt-dev/data/generated/train_raw.jsonl \
    --output gs://recruitgpt-dev/data/generated/train_clean.jsonl
```

---

## 3. Embedding Fine-tuning (BGE)

**Base model:** `BAAI/bge-large-zh-v1.5`
**Method:** Contrastive learning, InfoNCE loss, hard negatives

### GPU requirement

| Instance | GPU | VRAM | Cost/hr | Est. Duration |
|----------|-----|------|---------|--------------|
| `a2-highgpu-1g` (GCP) | A100 40GB | 40 GB | ~$3.67 | < 1 hr |
| `g2-standard-4` (GCP) | L4 | 24 GB | ~$0.70 | 1–2 hr |
| RunPod / AutoDL | A6000 | 48 GB | ~$0.50 | < 1 hr |

L4 is the cost-efficient choice for this job.

### Launch command (Vertex AI Custom Job)

```bash
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=bge-finetune \
  --worker-pool-spec=machine-type=g2-standard-4,accelerator-type=NVIDIA_L4,accelerator-count=1,container-image-uri=gcr.io/YOUR_PROJECT/recruitgpt-train:latest \
  --args="python,src/embedding/train_embedding.py,--config,configs/bge_finetune.yaml"
```

### Input / output paths (set in `configs/bge_finetune.yaml`)

```yaml
data:
  train_file: gs://recruitgpt-dev/data/pairs/train_triplets.jsonl
  hard_negatives: gs://recruitgpt-dev/data/pairs/hard_negatives.jsonl
training:
  output_dir: gs://recruitgpt-dev/artifacts/bge-recruit/
```

---

## 4. LLM Fine-tuning (Qwen3.5-0.8B QLoRA)

**Base model:** `Qwen/Qwen3.5-0.8B-Instruct`
**Method:** QLoRA 4-bit, LoRA r=32

### GPU requirement

0.8B is small — a T4 (16 GB) is more than sufficient. No A100 needed.

| Instance | GPU | VRAM | Cost/hr | Est. Duration |
|----------|-----|------|---------|--------------|
| `n1-standard-4` + T4 (GCP) | T4 | 16 GB | ~$0.35 | < 1 hr |
| `g2-standard-4` (GCP) | L4 | 24 GB | ~$0.70 | < 30 min |
| RunPod / AutoDL | RTX 3090 | 24 GB | ~$0.30 | < 30 min |

**Total LLM training cost: < $1** on a T4.

### Launch command (Vertex AI Custom Job)

```bash
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=qwen-qlora-finetune \
  --worker-pool-spec=machine-type=n1-standard-4,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,container-image-uri=gcr.io/YOUR_PROJECT/recruitgpt-train:latest \
  --args="python,src/train.py,--config,configs/qlora_qwen3_5_0_8b.yaml"
```

### Input / output paths (set in `configs/qlora_qwen3_5_0_8b.yaml`)

```yaml
data:
  train_file: gs://recruitgpt-dev/data/generated/train_clean.jsonl
  eval_file:  gs://recruitgpt-dev/eval/eval_set.jsonl
training:
  output_dir: gs://recruitgpt-dev/artifacts/qwen3_5_0_8b-recruit/
merge:
  merged_output_dir: gs://recruitgpt-dev/artifacts/qwen3_5_0_8b-recruit/merged/
```

After training, the merged checkpoint in GCS is the artifact you serve from.

---

## 5. Model Serving

After both fine-tuning jobs complete, you have three artifacts in GCS:

| Artifact | GCS path | Serve via |
|----------|----------|-----------|
| BGE embedding | `artifacts/bge-recruit/` | Cloud Run (encode API) |
| Reranker | `artifacts/reranker/` | Cloud Run (rerank API) |
| Qwen3.5-0.8B merged | `artifacts/qwen3_5_0_8b-recruit/merged/` | Vertex AI Endpoint or Cloud Run |

For the open-recruiter integration, the simplest path is a **Cloud Run container** that loads the merged Qwen3.5-0.8B at startup and exposes a `/parse` endpoint. At 0.8B the model loads in ~2s and fits comfortably in a 16 GB instance.

---

## 6. Cost Summary

| Step | Service | Estimated Cost |
|------|---------|---------------|
| Distill 3,000 samples | DeepSeek API | ~$2–5 |
| Build embedding triplets + hard negatives | T4 GPU (1 hr) | ~$0.35 |
| Fine-tune BGE embedding | L4 GPU (1–2 hr) | ~$0.70–1.40 |
| Fine-tune Qwen3.5-0.8B QLoRA | T4 GPU (< 1 hr) | ~$0.35 |
| GCS storage (ongoing) | ~50 GB data + checkpoints | ~$1/month |
| **Total (one-time training run)** | | **~$4–8** |
