"""
Build a NetworkX knowledge graph from parsed job queries and resumes.

Usage:
    from src.graph.builder import build_graph, save_graph, load_graph

    G = build_graph(parsed_queries, resumes)
    save_graph(G, "data/graph/recruit.gpickle")
    G = load_graph("data/graph/recruit.gpickle")
"""

from __future__ import annotations

import pickle
from itertools import combinations
from pathlib import Path
from typing import Any

import networkx as nx

from src.graph.schema import (
    COMPANY,
    COMPANY_TO_TIER,
    HIRES,
    IN_TIER,
    NEXT_STEP,
    RELATED_TO,
    ROLE,
    SENIORITY,
    SENIORITY_ORDER,
    SKILL,
    USED_IN,
)


# -------------------------------------------------------------------
# Persistence helpers
# -------------------------------------------------------------------

def load_graph(path: str | Path) -> nx.Graph:
    """Load a graph from a gpickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_graph(graph: nx.Graph, path: str | Path) -> None:
    """Save a graph to a gpickle file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)


# -------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------

def _norm(text: str) -> str:
    """Lowercase-strip normalisation for node IDs."""
    return text.strip().lower()


def _ensure_node(G: nx.Graph, node_id: str, node_type: str, **attrs: Any) -> None:
    """Add a node if it does not already exist; merge attrs otherwise."""
    if G.has_node(node_id):
        G.nodes[node_id].update(attrs)
    else:
        G.add_node(node_id, node_type=node_type, **attrs)


def _inc_edge(G: nx.Graph, u: str, v: str, edge_type: str) -> None:
    """Add or increment the weight on an edge."""
    if G.has_edge(u, v):
        G[u][v]["weight"] += 1
    else:
        G.add_edge(u, v, edge_type=edge_type, weight=1)


def _role_family(role_title: str) -> str:
    """Extract a coarse role family from a title for career-path edges.

    E.g. 'Senior Backend Engineer' -> 'backend engineer'
    """
    t = _norm(role_title)
    # Strip common seniority prefixes
    for prefix in ("senior ", "sr. ", "staff ", "lead ", "principal ",
                   "junior ", "associate ", "intern "):
        if t.startswith(prefix):
            t = t[len(prefix):]
            break
    return t


# -------------------------------------------------------------------
# Main builder
# -------------------------------------------------------------------

def build_graph(
    parsed_queries: list[dict],
    resumes: list[dict] | None = None,
) -> nx.Graph:
    """Construct the knowledge graph.

    Parameters
    ----------
    parsed_queries : list[dict]
        Each dict should have at minimum:
            - role_title: str
            - required_skills: list[str]
            - nice_to_have_skills: list[str]  (optional)
            - seniority: str                  (optional)
            - company: str                    (optional)
    resumes : list[dict] | None
        Each dict should have at minimum:
            - current_title: str
            - seniority: str
            - skills: list[str]              (optional)
            - company: str                   (optional)
    """
    G = nx.Graph()

    # ------------------------------------------------------------------
    # 1. Process job queries
    # ------------------------------------------------------------------
    for q in parsed_queries:
        role_raw = q.get("role_title", "")
        if not role_raw:
            continue
        role_id = _norm(role_raw)
        _ensure_node(G, role_id, ROLE, label=role_raw)

        all_skills: list[str] = []
        for skill in q.get("required_skills", []):
            sid = _norm(skill)
            _ensure_node(G, sid, SKILL, label=skill)
            _inc_edge(G, sid, role_id, USED_IN)
            all_skills.append(sid)

        for skill in q.get("nice_to_have_skills", []):
            sid = _norm(skill)
            _ensure_node(G, sid, SKILL, label=skill)
            _inc_edge(G, sid, role_id, USED_IN)
            all_skills.append(sid)

        # RELATED_TO edges between skills co-occurring in the same JD
        for s1, s2 in combinations(sorted(set(all_skills)), 2):
            _inc_edge(G, s1, s2, RELATED_TO)

        # Company -> tier and company -> role (HIRES)
        company_raw = q.get("company", "")
        if company_raw:
            cid = _norm(company_raw)
            tier = COMPANY_TO_TIER.get(cid, "general")
            _ensure_node(G, cid, COMPANY, label=company_raw, tier=tier)
            _inc_edge(G, cid, role_id, HIRES)
            # IN_TIER edge
            tier_node = f"tier:{tier}"
            _ensure_node(G, tier_node, SENIORITY, label=tier)
            _inc_edge(G, cid, tier_node, IN_TIER)

    # ------------------------------------------------------------------
    # 2. Process resumes (adds more skill + role + company nodes)
    # ------------------------------------------------------------------
    if resumes:
        for r in resumes:
            role_raw = r.get("current_title", "")
            if role_raw:
                role_id = _norm(role_raw)
                _ensure_node(G, role_id, ROLE, label=role_raw)

                for skill in r.get("skills", []):
                    sid = _norm(skill)
                    _ensure_node(G, sid, SKILL, label=skill)
                    _inc_edge(G, sid, role_id, USED_IN)

            company_raw = r.get("company", "")
            if company_raw:
                cid = _norm(company_raw)
                tier = COMPANY_TO_TIER.get(cid, "general")
                _ensure_node(G, cid, COMPANY, label=company_raw, tier=tier)
                if role_raw:
                    _inc_edge(G, cid, _norm(role_raw), HIRES)

    # ------------------------------------------------------------------
    # 3. NEXT_STEP edges for career progression
    # ------------------------------------------------------------------
    # Group role nodes by family, then connect adjacent seniority levels.
    role_nodes = [
        (nid, data) for nid, data in G.nodes(data=True)
        if data.get("node_type") == ROLE
    ]

    # Build family -> {seniority: node_id}
    family_map: dict[str, dict[str, str]] = {}
    for nid, data in role_nodes:
        family = _role_family(data.get("label", nid))
        # Infer seniority from the label
        label_lower = data.get("label", nid).lower()
        seniority = "mid"  # default
        for level in reversed(SENIORITY_ORDER):
            if level in label_lower:
                seniority = level
                break
        # Additional heuristic keywords
        if "staff" in label_lower or "principal" in label_lower:
            seniority = "staff"
        elif "senior" in label_lower or "sr." in label_lower:
            seniority = "senior"
        elif "lead" in label_lower:
            seniority = "lead"
        elif "manager" in label_lower:
            seniority = "manager"
        elif "director" in label_lower:
            seniority = "director"
        elif "junior" in label_lower or "associate" in label_lower:
            seniority = "junior"
        elif "intern" in label_lower:
            seniority = "intern"

        family_map.setdefault(family, {})[seniority] = nid

    for family, levels in family_map.items():
        ordered = sorted(levels.items(), key=lambda kv: SENIORITY_ORDER.index(kv[0])
                         if kv[0] in SENIORITY_ORDER else 99)
        for i in range(len(ordered) - 1):
            _inc_edge(G, ordered[i][1], ordered[i + 1][1], NEXT_STEP)

    return G
