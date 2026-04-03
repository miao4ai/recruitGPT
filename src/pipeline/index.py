"""
FAISS index management: build, save, and load a dense vector index
for resume retrieval using a fine-tuned BGE-M3 encoder.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def build_index(
    resumes_path: str,
    model_path: str,
    output_path: str,
    batch_size: int = 64,
) -> None:
    """
    Encode all resumes and build a FAISS index.

    Args:
        resumes_path: Path to a JSONL file where each line has ``{"id": ..., "text": ...}``.
        model_path: Path to the fine-tuned BGE-M3 sentence-transformer checkpoint.
        output_path: Directory to write ``index.faiss`` and ``metadata.jsonl``.
        batch_size: Encoding batch size.
    """
    logger.info("Loading encoder from %s", model_path)
    model = SentenceTransformer(model_path, trust_remote_code=True)

    logger.info("Reading resumes from %s", resumes_path)
    records: list[dict] = []
    with open(resumes_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    texts = [r["text"] for r in records]
    logger.info("Encoding %d resumes (batch_size=%d)...", len(texts), batch_size)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    dim = embeddings.shape[1]
    logger.info("Building FAISS IndexFlatIP (dim=%d, n=%d)", dim, len(texts))
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    faiss_path = out_dir / "index.faiss"
    meta_path = out_dir / "metadata.jsonl"

    faiss.write_index(index, str(faiss_path))
    logger.info("Saved FAISS index to %s", faiss_path)

    with open(meta_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Saved metadata to %s", meta_path)


def load_index(index_path: str) -> faiss.Index:
    """
    Load a FAISS index from disk.

    Args:
        index_path: Path to the ``.faiss`` file.

    Returns:
        The loaded FAISS index.
    """
    logger.info("Loading FAISS index from %s", index_path)
    return faiss.read_index(index_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build FAISS index from resume corpus")
    parser.add_argument("--resumes", required=True, help="Path to resumes JSONL file")
    parser.add_argument("--model_path", required=True, help="Path to fine-tuned BGE-M3")
    parser.add_argument("--output", required=True, help="Output directory for index")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    build_index(args.resumes, args.model_path, args.output, batch_size=args.batch_size)
