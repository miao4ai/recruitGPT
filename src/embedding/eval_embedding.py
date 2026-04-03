"""
Evaluate embedding model on retrieval benchmark.

Usage:
    python src/embedding/eval_embedding.py \
        --model outputs/bge-m3-recruit/best \
        --eval_data eval/retrieval_benchmark.jsonl

Eval data format (one JSON object per line):
    {
        "query": "Senior Backend Engineer Go distributed systems",
        "positive_ids": ["resume_001"],
        "candidate_ids": ["resume_001", "resume_002", ...]
    }
"""

import argparse
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


def load_eval_data(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def compute_recall_at_k(ranked_ids: list[str], positive_ids: set[str], k: int) -> float:
    """Fraction of positives found in top-k results."""
    top_k = set(ranked_ids[:k])
    hits = len(top_k & positive_ids)
    return hits / len(positive_ids) if positive_ids else 0.0


def compute_mrr(ranked_ids: list[str], positive_ids: set[str]) -> float:
    """Mean Reciprocal Rank — 1/rank of the first positive hit."""
    for i, rid in enumerate(ranked_ids):
        if rid in positive_ids:
            return 1.0 / (i + 1)
    return 0.0


def evaluate(model_path: str, eval_data_path: str):
    print(f"Loading model: {model_path}")
    model = SentenceTransformer(model_path)

    samples = load_eval_data(eval_data_path)
    print(f"Loaded {len(samples)} evaluation queries")

    recall_10_scores, recall_50_scores, mrr_scores = [], [], []

    for sample in samples:
        query = sample["query"]
        positive_ids = set(sample["positive_ids"])
        candidate_ids = sample["candidate_ids"]

        # Encode query and candidates
        query_emb = model.encode([query], normalize_embeddings=True)
        candidate_embs = model.encode(candidate_ids, normalize_embeddings=True)

        # Cosine similarity (embeddings are already normalized)
        scores = (query_emb @ candidate_embs.T).flatten()
        ranked_indices = np.argsort(-scores)
        ranked_ids = [candidate_ids[i] for i in ranked_indices]

        recall_10_scores.append(compute_recall_at_k(ranked_ids, positive_ids, 10))
        recall_50_scores.append(compute_recall_at_k(ranked_ids, positive_ids, 50))
        mrr_scores.append(compute_mrr(ranked_ids, positive_ids))

    # Print results
    print("\n" + "=" * 40)
    print(f"{'Metric':<20} {'Score':>10}")
    print("-" * 40)
    print(f"{'Recall@10':<20} {np.mean(recall_10_scores):>10.4f}")
    print(f"{'Recall@50':<20} {np.mean(recall_50_scores):>10.4f}")
    print(f"{'MRR':<20} {np.mean(mrr_scores):>10.4f}")
    print("=" * 40)
    print(f"Evaluated on {len(samples)} queries")


def main():
    parser = argparse.ArgumentParser(description="Evaluate embedding model on retrieval benchmark")
    parser.add_argument("--model", type=str, required=True, help="Path to embedding model")
    parser.add_argument("--eval_data", type=str, default="eval/retrieval_benchmark.jsonl",
                        help="Path to evaluation data (JSONL)")
    args = parser.parse_args()
    evaluate(args.model, args.eval_data)


if __name__ == "__main__":
    main()
