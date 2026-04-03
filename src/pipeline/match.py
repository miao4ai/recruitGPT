"""
Main orchestrator: chain all 5 pipeline stages to match candidates to a JD.

Stages:
  1. Query Parsing   - JD -> structured JSON
  2. Retrieval       - vector search over FAISS index
  3. Reranking       - cross-encoder rescoring
  4. Graph Boost     - knowledge-graph signal fusion
  5. Explanation     - generate human-readable match reports

Each stage is optional and skipped when its model/data path is not provided
in the config dict.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Allow importing from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.pipeline.query_parser import QueryParser
from src.pipeline.retriever import Retriever
from src.pipeline.reranker_stage import RerankerStage
from src.pipeline.graph_stage import GraphStage
from src.pipeline.explainer import Explainer


class MatchPipeline:
    """
    End-to-end candidate matching pipeline.

    Config dict keys (all optional -- stages are skipped if path is absent or None):
        query_parser_model : str  - path to fine-tuned Qwen3.5-0.8B for query parsing
        retriever_model    : str  - path to fine-tuned BGE-M3
        index_path         : str  - directory with index.faiss + metadata.jsonl
        reranker_model     : str  - path to cross-encoder model
        graph_path         : str  - path to NetworkX graph file
        explainer_model    : str  - path to fine-tuned Qwen3.5-0.8B for explanations
    """

    def __init__(self, config: dict) -> None:
        self.config = config

        # Stage 1 - Query Parser
        self.query_parser: QueryParser | None = None
        if config.get("query_parser_model"):
            self.query_parser = QueryParser(config["query_parser_model"])

        # Stage 2 - Retriever
        self.retriever: Retriever | None = None
        if config.get("retriever_model") and config.get("index_path"):
            self.retriever = Retriever(config["retriever_model"], config["index_path"])

        # Stage 3 - Reranker
        self.reranker: RerankerStage | None = None
        if config.get("reranker_model"):
            self.reranker = RerankerStage(config["reranker_model"])

        # Stage 4 - Graph Boost
        self.graph_stage: GraphStage | None = None
        if config.get("graph_path"):
            self.graph_stage = GraphStage(config["graph_path"])

        # Stage 5 - Explainer
        self.explainer: Explainer | None = None
        if config.get("explainer_model"):
            self.explainer = Explainer(config["explainer_model"])

    def match(self, jd_text: str, top_k: int = 20) -> list[dict]:
        """
        Run the full matching pipeline on a job description.

        Args:
            jd_text: Raw job description text.
            top_k: Number of final candidates to return.

        Returns:
            List of candidate dicts, each containing at minimum ``id``, ``text``,
            and available scores. If the explainer is loaded, each dict also
            includes an ``explanation`` field.
        """
        # ------------------------------------------------------------------
        # Stage 1: Parse the JD into structured form
        # ------------------------------------------------------------------
        query_parsed: dict | None = None
        if self.query_parser:
            logger.info("Stage 1: Parsing JD...")
            query_parsed = self.query_parser.parse(jd_text)
            logger.info("Parsed query: %s", json.dumps(query_parsed, ensure_ascii=False)[:200])
        else:
            logger.info("Stage 1 skipped (no query_parser_model)")

        # ------------------------------------------------------------------
        # Stage 2: Retrieve candidates from FAISS
        # ------------------------------------------------------------------
        if self.retriever is None:
            logger.warning("Stage 2 skipped (no retriever configured) -- returning empty results")
            return []

        # Use the raw JD as the retrieval query (the encoder handles it)
        retrieval_top_k = max(top_k * 5, 50)  # retrieve more than we need for reranking
        logger.info("Stage 2: Retrieving top %d candidates...", retrieval_top_k)
        candidates = self.retriever.retrieve(jd_text, top_k=retrieval_top_k)
        logger.info("Retrieved %d candidates", len(candidates))

        if not candidates:
            return []

        # ------------------------------------------------------------------
        # Stage 3: Rerank with cross-encoder
        # ------------------------------------------------------------------
        if self.reranker:
            rerank_top_n = max(top_k * 2, 20)
            logger.info("Stage 3: Reranking to top %d...", rerank_top_n)
            candidates = self.reranker.rerank(jd_text, candidates, top_n=rerank_top_n)
        else:
            logger.info("Stage 3 skipped (no reranker_model)")

        # ------------------------------------------------------------------
        # Stage 4: Graph boost
        # ------------------------------------------------------------------
        if self.graph_stage and query_parsed:
            logger.info("Stage 4: Applying graph boost...")
            candidates = self.graph_stage.boost(query_parsed, candidates)
        else:
            logger.info("Stage 4 skipped (no graph or parsed query)")

        # Trim to final top_k
        candidates = candidates[:top_k]

        # ------------------------------------------------------------------
        # Stage 5: Generate explanations for final candidates
        # ------------------------------------------------------------------
        if self.explainer:
            logger.info("Stage 5: Generating explanations for %d candidates...", len(candidates))
            for candidate in candidates:
                candidate["explanation"] = self.explainer.explain(jd_text, candidate["text"])
        else:
            logger.info("Stage 5 skipped (no explainer_model)")

        return candidates


def _load_config(config_path: str) -> dict:
    """Load a JSON config file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the full recruitGPT matching pipeline on a job description"
    )
    parser.add_argument("--jd", required=True, help="Job description text")
    parser.add_argument("--top_k", type=int, default=20, help="Number of candidates to return")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to JSON config file with model/data paths",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    config = _load_config(args.config)
    pipeline = MatchPipeline(config)
    results = pipeline.match(args.jd, top_k=args.top_k)

    print(json.dumps(results, indent=2, ensure_ascii=False))
