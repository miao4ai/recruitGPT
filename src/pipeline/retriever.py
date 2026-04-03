"""
Stage 2 - Vector Retrieval: encode a query with fine-tuned BGE-M3 and
search a FAISS index for the most relevant resume candidates.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Retriever:
    """Dense retrieval using a fine-tuned BGE-M3 encoder and a FAISS index."""

    def __init__(self, model_path: str, index_path: str) -> None:
        """
        Args:
            model_path: Path to the fine-tuned BGE-M3 sentence-transformer checkpoint.
            index_path: Directory containing ``index.faiss`` and ``metadata.jsonl``.
        """
        logger.info("Loading retriever model from %s", model_path)
        self.model = SentenceTransformer(model_path, trust_remote_code=True)

        index_dir = Path(index_path)
        faiss_file = index_dir / "index.faiss"
        meta_file = index_dir / "metadata.jsonl"

        logger.info("Loading FAISS index from %s", faiss_file)
        self.index = faiss.read_index(str(faiss_file))

        logger.info("Loading metadata from %s", meta_file)
        self.metadata: list[dict] = []
        with open(meta_file, "r", encoding="utf-8") as f:
            for line in f:
                self.metadata.append(json.loads(line))

        assert len(self.metadata) == self.index.ntotal, (
            f"Metadata count ({len(self.metadata)}) != index size ({self.index.ntotal})"
        )

    def retrieve(self, query_text: str, top_k: int = 50) -> list[dict]:
        """
        Encode query and retrieve top_k candidates from the FAISS index.

        Args:
            query_text: The query string (e.g. serialised parsed JD or raw JD text).
            top_k: Number of candidates to return.

        Returns:
            List of dicts with keys ``id``, ``score``, ``text``, sorted by descending score.
        """
        query_vec = self.model.encode(
            [query_text], normalize_embeddings=True
        ).astype(np.float32)

        scores, indices = self.index.search(query_vec, top_k)

        results: list[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self.metadata[idx]
            results.append(
                {
                    "id": meta.get("id", str(idx)),
                    "score": float(score),
                    "text": meta.get("text", ""),
                }
            )

        # FAISS inner-product returns highest = best; sort descending.
        results.sort(key=lambda r: r["score"], reverse=True)
        return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Retrieve candidates for a query")
    parser.add_argument("--model_path", required=True, help="Path to fine-tuned BGE-M3")
    parser.add_argument("--index_path", required=True, help="Path to FAISS index directory")
    parser.add_argument("--query", required=True, help="Query text")
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    retriever = Retriever(args.model_path, args.index_path)
    hits = retriever.retrieve(args.query, top_k=args.top_k)
    for h in hits:
        print(f"[{h['score']:.4f}] {h['id']}: {h['text'][:120]}")
