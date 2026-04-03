"""
Evaluate cross-encoder reranker on eval set.

Usage:
    python src/reranker/eval_reranker.py \
        --model outputs/reranker-recruit/best \
        --eval_data data/reranker/eval.jsonl

Eval data format (one JSON object per line):
    {
        "query": "...",
        "candidates": [
            {"id": "resume_001", "text": "...", "relevance": 3},
            {"id": "resume_002", "text": "...", "relevance": 1},
            ...
        ]
    }
    relevance: integer grade (higher = more relevant), used for NDCG/MAP.
"""

import argparse
import json
import math
import numpy as np
from sentence_transformers import CrossEncoder


def load_eval_data(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def dcg_at_k(relevances: list[float], k: int) -> float:
    """Discounted Cumulative Gain at k."""
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances[:k]))


def ndcg_at_k(relevances: list[float], k: int) -> float:
    """Normalized DCG at k."""
    dcg = dcg_at_k(relevances, k)
    ideal = dcg_at_k(sorted(relevances, reverse=True), k)
    return dcg / ideal if ideal > 0 else 0.0


def average_precision(relevances: list[float]) -> float:
    """Average Precision — binary: relevance > 0 counts as relevant."""
    hits, running_sum = 0, 0.0
    for i, rel in enumerate(relevances):
        if rel > 0:
            hits += 1
            running_sum += hits / (i + 1)
    return running_sum / hits if hits > 0 else 0.0


def evaluate(model_path: str, eval_data_path: str):
    print(f"Loading reranker: {model_path}")
    model = CrossEncoder(model_path)

    samples = load_eval_data(eval_data_path)
    print(f"Loaded {len(samples)} evaluation queries")

    ndcg5_scores, ndcg10_scores, map_scores = [], [], []

    for sample in samples:
        query = sample["query"]
        candidates = sample["candidates"]

        # Score each (query, candidate) pair
        pairs = [(query, c["text"]) for c in candidates]
        scores = model.predict(pairs)

        # Rank candidates by model score
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        ranked_relevances = [c["relevance"] for c, _ in scored]

        ndcg5_scores.append(ndcg_at_k(ranked_relevances, 5))
        ndcg10_scores.append(ndcg_at_k(ranked_relevances, 10))
        map_scores.append(average_precision(ranked_relevances))

    # Print results
    print("\n" + "=" * 40)
    print(f"{'Metric':<20} {'Score':>10}")
    print("-" * 40)
    print(f"{'NDCG@5':<20} {np.mean(ndcg5_scores):>10.4f}")
    print(f"{'NDCG@10':<20} {np.mean(ndcg10_scores):>10.4f}")
    print(f"{'MAP':<20} {np.mean(map_scores):>10.4f}")
    print("=" * 40)
    print(f"Evaluated on {len(samples)} queries")


def main():
    parser = argparse.ArgumentParser(description="Evaluate reranker model")
    parser.add_argument("--model", type=str, required=True, help="Path to reranker model")
    parser.add_argument("--eval_data", type=str, default="data/reranker/eval.jsonl",
                        help="Path to evaluation data (JSONL)")
    args = parser.parse_args()
    evaluate(args.model, args.eval_data)


if __name__ == "__main__":
    main()
