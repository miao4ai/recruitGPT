# RecruitGPT

An open-source AI recruiting pipeline that combines fine-tuned embeddings, cross-encoder reranking, knowledge graph signals, and LLM reasoning to match candidates with jobs.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

## How It Works

RecruitGPT is a 5-stage retrieval-augmented matching pipeline. Each stage narrows and refines the candidate pool, ending with a human-readable explanation.

```
JD / Hiring Query
        │
        ▼
┌───────────────────┐
│  ① Query Parsing  │  LLM extracts structured intent: skills, seniority,
│  (Qwen3.5 0.8B)  │  industry, hard constraints, nice-to-haves
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  ② Retrieval      │  Fine-tuned BGE encodes query → FAISS ANN search
│   (BGE-large)     │  over candidate embeddings → Top-K recall
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  ③ Reranking      │  Cross-encoder scores each (query, candidate) pair
│  (bge-reranker)   │  with full attention → Top-N precision
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  ④ Graph Boost    │  Knowledge graph (skills, companies, industries)
│   (NetworkX)      │  adds structural signals: career similarity,
│                   │  skill adjacency, company-tier overlap
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  ⑤ Explanation    │  LLM generates per-candidate match report:
│  (Qwen3.5 0.8B)  │  strengths, gaps, interview focus areas
└───────────────────┘
```

## Why Fine-tune BGE?

Generic embedding models treat "5 years of distributed systems at a fintech" and "entry-level web developer" as vaguely similar — they're both "software engineering." A fine-tuned BGE model learns the recruiting domain's similarity structure:

- **Seniority matters**: Senior backend ≠ junior backend
- **Skill overlap is nuanced**: "Kubernetes + Go" is closer to "Docker + Rust" than to "Excel + VBA"
- **Context changes meaning**: "Python" in a data science JD ≠ "Python" in a DevOps JD

We fine-tune with contrastive learning on (JD, good-match resume, bad-match resume) triplets, including hard negatives mined from the model itself.

## Quick Start

### Installation

```bash
git clone https://github.com/your-org/recruitGPT.git
cd recruitGPT
pip install -r requirements.txt
```

### Configure API Keys

```bash
cp .env.example .env
# Fill in at least one teacher model key (DeepSeek recommended — cheapest, no license issues)
```

### Step 1 — Generate Training Data via Distillation

A large teacher model (DeepSeek-V3, GPT-4o, or Claude) generates high-quality training data for the smaller student model.

```bash
# Generate LLM training data (query parsing + match explanation)
python scripts/distill_data.py \
    --teacher deepseek \
    --tasks query_parsing,match_explanation \
    --num_per_task 500

# Build embedding triplets
python scripts/build_embedding_pairs.py \
    --resumes data/resumes/ \
    --jds data/jds/ \
    --output data/pairs/train_triplets.jsonl

# Mine hard negatives using current model
python scripts/mine_hard_negatives.py \
    --triplets data/pairs/train_triplets.jsonl \
    --model BAAI/bge-large-zh-v1.5 \
    --output data/pairs/hard_negatives.jsonl

# Quality filtering
python scripts/filter_data.py \
    --input data/generated/train.jsonl \
    --output data/generated/train_clean.jsonl
```

### Step 2 — Fine-tune BGE Embedding

```bash
python src/embedding/train_embedding.py --config configs/bge_finetune.yaml
```

This trains with InfoNCE loss + in-batch negatives + hard negatives. A single A6000 handles it in under an hour for a few thousand triplets.

### Step 3 — Fine-tune LLM (Query Parsing + Explanation)

```bash
python src/train.py --config configs/qlora_qwen3_5_0_8b.yaml
```

QLoRA on Qwen3.5-0.8B — runs on any GPU with 6–8 GB VRAM (RTX 3060, T4, etc.). Merge LoRA weights after training for faster inference.

### Step 4 — Build Index & Run Pipeline

```bash
# Index your candidate pool
python src/pipeline/index.py \
    --resumes data/resumes/ \
    --model outputs/bge-recruit/

# Interactive matching
python src/pipeline/match.py \
    --jd "Your job description here" \
    --top_k 20 \
    --interactive
```

## Project Structure

```
recruitGPT/
│
├── configs/
│   ├── qlora_qwen3_5_0_8b.yaml        # LLM fine-tuning (student model)
│   ├── qlora_qwen7b.yaml              # LLM fine-tuning (teacher reference)
│   ├── qlora_qwen3b.yaml              # LLM low-resource alternative
│   ├── bge_finetune.yaml              # BGE embedding fine-tuning
│   └── reranker_finetune.yaml         # Cross-encoder fine-tuning
│
├── data/
│   ├── seed/                          # Hand-written seed examples
│   ├── pairs/                         # Embedding training triplets
│   ├── reranker/                      # Reranker training pairs
│   ├── resumes/                       # Candidate resume corpus
│   ├── jds/                           # Job description corpus
│   └── generated/                     # Distilled training data
│
├── scripts/
│   ├── distill_data.py                # Teacher → student data generation
│   ├── build_embedding_pairs.py       # Build (query, pos, neg) triplets
│   ├── mine_hard_negatives.py         # Hard negative mining
│   ├── build_reranker_data.py         # Reranker training data
│   ├── build_graph.py                 # Knowledge graph construction
│   ├── filter_data.py                 # Data quality filtering
│   └── convert_format.py             # Format conversion utility
│
├── src/
│   ├── embedding/                     # Stage ②
│   │   ├── train_embedding.py         # BGE contrastive fine-tuning
│   │   ├── eval_embedding.py          # Recall@K, MRR evaluation
│   │   ├── encode.py                  # Encode & retrieve
│   │   └── losses.py                  # InfoNCE, triplet loss
│   │
│   ├── reranker/                      # Stage ③
│   │   ├── train_reranker.py          # Cross-encoder fine-tuning
│   │   ├── eval_reranker.py           # NDCG, MAP evaluation
│   │   └── rerank.py                  # Reranking inference
│   │
│   ├── graph/                         # Stage ④
│   │   ├── schema.py                  # Graph schema definition
│   │   ├── builder.py                 # Build skill/company/industry graph
│   │   └── boost.py                   # Graph signal scoring
│   │
│   ├── pipeline/                      # End-to-end pipeline
│   │   ├── query_parser.py            # Stage ① — LLM query parsing
│   │   ├── retriever.py               # Stage ② — vector retrieval
│   │   ├── reranker_stage.py          # Stage ③ — reranking
│   │   ├── graph_stage.py             # Stage ④ — graph signal
│   │   ├── explainer.py               # Stage ⑤ — LLM explanation
│   │   ├── index.py                   # FAISS index management
│   │   └── match.py                   # Main orchestrator
│   │
│   ├── teacher.py                     # Unified teacher model interface
│   ├── prompts.py                     # All prompt templates
│   ├── train.py                       # LLM QLoRA training (Unsloth)
│   ├── evaluate.py                    # LLM-as-Judge evaluation
│   └── inference.py                   # LLM interactive inference
│
├── eval/
│   ├── eval_set.jsonl                 # LLM evaluation set
│   └── retrieval_benchmark.jsonl      # Embedding retrieval benchmark
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_embedding_analysis.ipynb
│   └── 03_pipeline_demo.ipynb
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Models Used

| Component | Base Model | Fine-tune Method | GPU Requirement |
|-----------|-----------|-----------------|-----------------|
| Query Parser | Qwen/Qwen3.5-0.8B-Instruct | QLoRA (4-bit) | 6–8 GB |
| Explainer (optional) | Qwen/Qwen3.5-0.8B-Instruct | QLoRA (4-bit) | 6–8 GB |
| Embedding | BAAI/bge-large-zh-v1.5 | Contrastive learning | 12–16 GB |
| Reranker | BAAI/bge-reranker-v2-m3 | Cross-encoder | 12–16 GB |
| Graph | NetworkX | No training | CPU only |

> **Teacher models** (for distillation data generation only): DeepSeek-V3, GPT-4o, or Claude via API.

## Cost Estimate

Assuming you use RunPod or AutoDL for GPU rental:

| Step | Estimated Cost |
|------|---------------|
| Distill 3,000 LLM training samples (DeepSeek API) | ~$2–5 |
| Mine hard negatives + build triplets | ~$1–2 (GPU) |
| Fine-tune BGE embedding | ~$1–3 (A6000, <1hr) |
| Fine-tune LLM QLoRA (Qwen3.5-0.8B) | ~$0.5–2 (T4/A10G, <1hr) |
| **Total** | **~$7–18** |

## Evaluation

### Embedding Retrieval

```bash
python src/embedding/eval_embedding.py \
    --model outputs/bge-recruit/ \
    --eval_data data/pairs/eval_triplets.jsonl
# Outputs: Recall@10, Recall@50, MRR
```

### Reranker

```bash
python src/reranker/eval_reranker.py \
    --model outputs/reranker/ \
    --eval_data data/reranker/eval.jsonl
# Outputs: NDCG@5, NDCG@10, MAP
```

### LLM (Judge-based)

```bash
python src/evaluate.py \
    --model_path outputs/qwen3_5_0_8b-recruit/merged \
    --eval_data eval/eval_set.jsonl \
    --judge deepseek
# Outputs: Accuracy, Format, Professionalism, Usefulness (1–5 scale)
```

## Roadmap

- [x] LLM distillation pipeline (query parsing + explanation)
- [x] BGE embedding fine-tuning with hard negative mining
- [x] Cross-encoder reranker
- [x] Skill/company knowledge graph
- [ ] Multi-language support (EN/ZH/JA)
- [ ] Resume PDF parsing (OCR + layout)
- [ ] Real-time indexing API
- [ ] Web UI demo
- [ ] DPO alignment for explanation quality

## MLOps Roadmap (GCP)

This section describes the path to a production-grade MLOps system on Google Cloud Platform.

### Maturity Levels

```
Level 0 (current) → Manual scripts, local GPU
Level 1            → Reproducible ML pipelines, experiment tracking
Level 2            → CI/CD for ML, automated retraining & deployment
```

### Target Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                           CI/CD Layer                           │
│       GitHub → Cloud Build → Artifact Registry → Pipeline      │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                      Data & Experiment Layer                    │
│    GCS (raw/processed/artifacts)   BigQuery   DVC               │
│    Vertex AI Experiments (metrics, hyperparams, artifacts)      │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│              Training Pipeline (Vertex AI Pipelines)            │
│                                                                 │
│  [distill_data] → [build_pairs] → [mine_negatives]             │
│                                          │                      │
│                          ┌───────────────┼───────────────┐      │
│                   [train_bge]   [train_reranker]  [train_llm]  │
│                          └───────────────┼───────────────┘      │
│                                    [evaluate]                   │
│                                          │                      │
│                              [register → Model Registry]        │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                         Serving Layer                           │
│   Vertex AI Endpoints (online)   Batch Prediction (batch)       │
│   Cloud Run (FAISS index API)                                   │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                        Monitoring Layer                         │
│   Vertex AI Model Monitoring   Cloud Monitoring   Looker Studio │
└─────────────────────────────────────────────────────────────────┘
```

### GCP Services by Function

| Function | GCP Service | Purpose |
|----------|------------|---------|
| Raw data & artifacts | Cloud Storage (GCS) | resumes, JDs, model checkpoints |
| Structured metrics | BigQuery | eval results, match history, experiment comparison |
| Data versioning | DVC + GCS backend | track changes to `data/pairs/`, `data/generated/` |
| Experiment tracking | Vertex AI Experiments | loss curves, hyperparams, Recall@K per run |
| GPU training jobs | Vertex AI Training (Custom Jobs) | BGE, reranker, QLoRA fine-tuning |
| Training images | Artifact Registry | versioned Docker images for each training job |
| Pipeline orchestration | Vertex AI Pipelines (KFP v2) | DAG with caching, retry, conditional steps |
| Scheduled retraining | Cloud Scheduler | cron-triggered pipeline runs |
| Model versioning | Vertex AI Model Registry | promote models with eval thresholds |
| Online inference | Vertex AI Endpoints | real-time JD → candidate matching API |
| Batch inference | Vertex AI Batch Prediction | periodic full-pool rescoring |
| FAISS index API | Cloud Run | stateless index serving, loaded from GCS |
| CI/CD trigger | Cloud Build | PR merge → rebuild image → run pipeline |
| Data drift detection | Vertex AI Model Monitoring | embedding distribution shift alerts |
| Dashboards | Looker Studio + BigQuery | matching quality trends, pipeline health |

### Phased Rollout

| Phase | Goal | Key Services |
|-------|------|-------------|
| **Phase 1** | Reproducible training | GCS + Vertex AI Training + Experiments |
| **Phase 2** | Automated pipeline DAG | Vertex AI Pipelines + Model Registry |
| **Phase 3** | CI/CD integration | Cloud Build + Artifact Registry |
| **Phase 4** | Production serving | Vertex AI Endpoints + Cloud Run |
| **Phase 5** | Monitoring & alerting | Model Monitoring + BigQuery + Looker Studio |

### GPU Requirements on GCP

| Training Job | Recommended Instance | Estimated Duration |
|-------------|---------------------|-------------------|
| BGE embedding fine-tune | `a2-highgpu-1g` (A100 40GB) | < 1 hr |
| Cross-encoder reranker | `a2-highgpu-1g` (A100 40GB) | 1–3 hr |
| QLoRA Qwen3.5-0.8B | `n1-standard-4` + T4 (16GB) | < 1 hr |
| Hard negative mining | `n1-standard-8` (CPU) or GPU | < 30 min |

> **Note**: GCP A100 quota is 0 by default. Request an increase via IAM & Admin → Quotas at least 3–5 business days before your training run.

- [ ] Phase 1 — GCS data lake + Vertex AI Training + Experiments
- [ ] Phase 2 — Vertex AI Pipelines DAG + Model Registry
- [ ] Phase 3 — Cloud Build CI/CD + Artifact Registry
- [ ] Phase 4 — Vertex AI Endpoints + Cloud Run serving
- [ ] Phase 5 — Model Monitoring + BigQuery + Looker Studio dashboards

## Contributing

Contributions are welcome. Please open an issue first to discuss what you'd like to change.

## License

[MIT](LICENSE)