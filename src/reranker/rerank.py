"""
Reranking inference utility.

Usage:
    python src/reranker/rerank.py \
        --model outputs/reranker-recruit/best \
        --query "Senior Backend Engineer with 5+ years Go experience" \
        --candidates data/resumes/all_resumes.jsonl

Candidates JSONL format:
    {"id": "resume_001", "text": "Zhang Wei | Backend Engineer ..."}
"""

import argparse
import json
from sentence_transformers import CrossEncoder


def rerank(model_path: str, query: str, candidates: list[dict],
           top_k: int | None = None) -> list[dict]:
    """
    Rerank candidates by cross-encoder relevance score.

    Args:
        model_path: path to cross-encoder model
        query: query text
        candidates: list of dicts with at least "id" and "text" keys
        top_k: if set, return only top-k results

    Returns:
        list of dicts with added "score" field, sorted by descending score
    """
    model = CrossEncoder(model_path)
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)

    for c, score in zip(candidates, scores):
        c["score"] = float(score)

    ranked = sorted(candidates, key=lambda x: x["score"], reverse=True)
    return ranked[:top_k] if top_k else ranked


def load_candidates(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser(description="Rerank candidates with cross-encoder")
    parser.add_argument("--model", type=str, required=True, help="Path to reranker model")
    parser.add_argument("--query", type=str, required=True, help="Query text")
    parser.add_argument("--candidates", type=str, required=True,
                        help="Path to candidates JSONL file")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top results to show")
    args = parser.parse_args()

    candidates = load_candidates(args.candidates)
    print(f"Loaded {len(candidates)} candidates")
    print(f"Query: {args.query!r}\n")

    results = rerank(args.model, args.query, candidates, top_k=args.top_k)

    print(f"Top-{args.top_k} results:")
    for rank, r in enumerate(results, 1):
        text_preview = r["text"][:80].replace("\n", " ")
        print(f"  {rank:>3}. [{r['id']}] score={r['score']:.4f}  {text_preview}...")


if __name__ == "__main__":
    main()
