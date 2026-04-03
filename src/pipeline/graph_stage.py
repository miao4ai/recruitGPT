"""
Stage 4 - Graph Boost: augment candidate scores with knowledge-graph
signals (company prestige, skill co-occurrence, career transitions, etc.)
using a pre-built NetworkX graph.
"""

from __future__ import annotations

import logging
from pathlib import Path

import networkx as nx

logger = logging.getLogger(__name__)

# Default weights for combining reranker and graph scores
RERANKER_WEIGHT = 0.8
GRAPH_WEIGHT = 0.2


def graph_boost_score(
    graph: nx.Graph,
    query_parsed: dict,
    candidate_text: str,
) -> float:
    """
    Compute a graph-based relevance boost for a single candidate.

    Heuristic: count how many required/nice-to-have skills from the parsed query
    appear as nodes in the graph that are connected (within 1 hop) to tokens
    found in the candidate text.

    Args:
        graph: The pre-built knowledge graph.
        query_parsed: Structured JD dict with ``required_skills``, ``nice_to_have_skills``, etc.
        candidate_text: Raw resume text for the candidate.

    Returns:
        A float score in [0, 1] representing the graph-based match quality.
    """
    required = set(s.lower() for s in query_parsed.get("required_skills", []))
    nice_to_have = set(s.lower() for s in query_parsed.get("nice_to_have_skills", []))
    all_skills = required | nice_to_have
    if not all_skills:
        return 0.0

    candidate_lower = candidate_text.lower()

    # Find graph nodes that appear in the candidate text
    candidate_nodes = set()
    for node in graph.nodes():
        node_str = str(node).lower()
        if node_str in candidate_lower:
            candidate_nodes.add(node)

    # For each query skill, check if it (or a 1-hop neighbour) appears in candidate nodes
    matched = 0.0
    total = 0.0
    for skill in all_skills:
        weight = 2.0 if skill in required else 1.0
        total += weight

        # Direct match
        if skill in candidate_lower:
            matched += weight
            continue

        # Check if any graph neighbour of the skill node is in candidate nodes
        skill_node = None
        for node in graph.nodes():
            if str(node).lower() == skill:
                skill_node = node
                break

        if skill_node is not None and skill_node in graph:
            neighbours = set(graph.neighbors(skill_node))
            if neighbours & candidate_nodes:
                matched += weight * 0.5  # partial credit for neighbour match

    return matched / total if total > 0 else 0.0


class GraphStage:
    """Apply knowledge-graph boost to candidate scores."""

    def __init__(self, graph_path: str) -> None:
        """
        Args:
            graph_path: Path to a serialised NetworkX graph (GraphML, GEXF, or pickle).
        """
        logger.info("Loading knowledge graph from %s", graph_path)
        path = Path(graph_path)
        suffix = path.suffix.lower()

        if suffix == ".graphml":
            self.graph = nx.read_graphml(str(path))
        elif suffix == ".gexf":
            self.graph = nx.read_gexf(str(path))
        elif suffix in (".gpickle", ".pkl", ".pickle"):
            import pickle

            with open(path, "rb") as f:
                self.graph = pickle.load(f)
        else:
            # Default: try GraphML
            self.graph = nx.read_graphml(str(path))

        logger.info(
            "Graph loaded: %d nodes, %d edges",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )

    def boost(
        self,
        query_parsed: dict,
        candidates: list[dict],
        reranker_weight: float = RERANKER_WEIGHT,
        graph_weight: float = GRAPH_WEIGHT,
    ) -> list[dict]:
        """
        Augment each candidate with a graph score and compute a combined score.

        Args:
            query_parsed: Structured JD dict from QueryParser.
            candidates: List of candidate dicts (must have ``text`` and ``reranker_score``).
            reranker_weight: Weight for the reranker score in the combined score.
            graph_weight: Weight for the graph score in the combined score.

        Returns:
            Candidates sorted by descending ``combined_score``, each with added
            ``graph_score`` and ``combined_score`` keys.
        """
        for candidate in candidates:
            g_score = graph_boost_score(self.graph, query_parsed, candidate["text"])
            candidate["graph_score"] = g_score

            reranker_score = candidate.get("reranker_score", candidate.get("score", 0.0))
            candidate["combined_score"] = (
                reranker_weight * reranker_score + graph_weight * g_score
            )

        candidates.sort(key=lambda c: c["combined_score"], reverse=True)
        return candidates
