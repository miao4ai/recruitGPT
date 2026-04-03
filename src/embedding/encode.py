"""
Encode texts into embeddings and retrieve nearest neighbors.

Usage:
    # Encode a single query
    python src/embedding/encode.py \
        --model outputs/bge-m3-recruit/best \
        --text "Senior Backend Engineer Go"

    # Retrieve from a FAISS index
    python src/embedding/encode.py \
        --model outputs/bge-m3-recruit/best \
        --text "Senior Backend Engineer Go" \
        --index_path outputs/bge-m3-recruit/faiss.index \
        --id_map outputs/bge-m3-recruit/id_map.json \
        --top_k 10
"""

import argparse
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


def encode_texts(model_path: str, texts: list[str]) -> np.ndarray:
    """Encode a list of texts into normalized embeddings."""
    model = SentenceTransformer(model_path)
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=len(texts) > 100)
    return embeddings


def retrieve(model_path: str, query: str, index_path: str, id_map_path: str,
             top_k: int = 10) -> list[tuple[str, float]]:
    """
    Retrieve top-k nearest neighbors from a FAISS index.

    Args:
        model_path: path to sentence-transformers model
        query: query text
        index_path: path to FAISS index file
        id_map_path: path to JSON file mapping index positions to document IDs
        top_k: number of results to return

    Returns:
        list of (document_id, score) tuples sorted by descending score
    """
    import faiss

    model = SentenceTransformer(model_path)
    query_emb = model.encode([query], normalize_embeddings=True).astype(np.float32)

    index = faiss.read_index(index_path)
    with open(id_map_path) as f:
        id_map = json.load(f)

    scores, indices = index.search(query_emb, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        doc_id = id_map[str(idx)] if isinstance(id_map, dict) else id_map[idx]
        results.append((doc_id, float(score)))
    return results


def main():
    parser = argparse.ArgumentParser(description="Encode texts and retrieve nearest neighbors")
    parser.add_argument("--model", type=str, required=True, help="Path to embedding model")
    parser.add_argument("--text", type=str, required=True, help="Text to encode")
    parser.add_argument("--index_path", type=str, default=None, help="Path to FAISS index")
    parser.add_argument("--id_map", type=str, default=None, help="Path to ID map JSON")
    parser.add_argument("--top_k", type=int, default=10, help="Number of results to retrieve")
    args = parser.parse_args()

    if args.index_path and args.id_map:
        results = retrieve(args.model, args.text, args.index_path, args.id_map, args.top_k)
        print(f"Top-{args.top_k} results for: {args.text!r}\n")
        for rank, (doc_id, score) in enumerate(results, 1):
            print(f"  {rank:>3}. {doc_id:<30} score={score:.4f}")
    else:
        emb = encode_texts(args.model, [args.text])
        print(f"Embedding shape: {emb.shape}")
        print(f"Embedding (first 10 dims): {emb[0][:10].tolist()}")
        print(f"L2 norm: {np.linalg.norm(emb[0]):.4f}")


if __name__ == "__main__":
    main()
