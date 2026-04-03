"""
Stage 3 - Cross-encoder Reranking: re-score retrieval candidates with a
cross-encoder model to improve precision in the top-N.
"""

from __future__ import annotations

import logging

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class RerankerStage:
    """Cross-encoder reranking of retrieval candidates."""

    def __init__(self, model_path: str) -> None:
        """
        Args:
            model_path: Path or HuggingFace model name for the cross-encoder
                        (e.g. ``BAAI/bge-reranker-v2-m3``).
        """
        logger.info("Loading cross-encoder reranker from %s", model_path)
        self.model = CrossEncoder(model_path, trust_remote_code=True)

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_n: int = 10,
    ) -> list[dict]:
        """
        Re-score each (query, candidate) pair and return the top_n.

        Args:
            query: The query text (e.g. raw JD or serialised parsed query).
            candidates: List of candidate dicts, each must contain a ``text`` key.
            top_n: Number of top candidates to return after reranking.

        Returns:
            Top-N candidates sorted by descending reranker score. Each dict gets
            an added ``reranker_score`` key.
        """
        if not candidates:
            return []

        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)

        for candidate, score in zip(candidates, scores):
            candidate["reranker_score"] = float(score)

        ranked = sorted(candidates, key=lambda c: c["reranker_score"], reverse=True)
        return ranked[:top_n]


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Rerank candidates with a cross-encoder")
    parser.add_argument("--model_path", required=True, help="Path to cross-encoder model")
    parser.add_argument("--query", required=True, help="Query text")
    parser.add_argument(
        "--candidates_file", required=True,
        help="JSONL file of candidate dicts (must have 'text' key)",
    )
    parser.add_argument("--top_n", type=int, default=10)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    candidates = []
    with open(args.candidates_file, "r", encoding="utf-8") as f:
        for line in f:
            candidates.append(json.loads(line))

    reranker = RerankerStage(args.model_path)
    results = reranker.rerank(args.query, candidates, top_n=args.top_n)
    for r in results:
        print(f"[{r['reranker_score']:.4f}] {r.get('id', '?')}: {r['text'][:120]}")
