"""
Score candidates using knowledge-graph signals.

Usage:
    from src.graph.boost import graph_boost_score

    score = graph_boost_score(G, query_parsed, candidate_parsed)
    # returns float in [0.0, 1.0]
"""

from __future__ import annotations

import networkx as nx

from src.graph.schema import (
    COMPANY_TO_TIER,
    NEXT_STEP,
    RELATED_TO,
    SENIORITY_ORDER,
    SENIORITY_RANK,
    TIER_RANK,
)


def _norm(text: str) -> str:
    return text.strip().lower()


# -------------------------------------------------------------------
# Component scorers
# -------------------------------------------------------------------

def _skill_adjacency_score(
    G: nx.Graph,
    required_skills: list[str],
    candidate_skills: set[str],
) -> float:
    """Partial credit for candidate skills that are RELATED_TO required skills.

    For each required skill the candidate lacks, check whether the candidate
    has any neighbour connected via RELATED_TO.  Credit is weighted by the
    edge weight (higher co-occurrence = stronger signal).

    Returns a value in [0.0, 1.0].
    """
    if not required_skills:
        return 0.0

    credit = 0.0
    for skill in required_skills:
        sid = _norm(skill)
        if sid in candidate_skills:
            # Direct match -- full credit for this skill
            credit += 1.0
            continue

        # Check graph neighbours
        if sid not in G:
            continue
        best = 0.0
        for neighbour in G.neighbors(sid):
            edge_data = G[sid][neighbour]
            if edge_data.get("edge_type") != RELATED_TO:
                continue
            if neighbour in candidate_skills:
                # Partial credit scaled by weight (cap at 0.5 per adjacent skill)
                weight = edge_data.get("weight", 1)
                best = max(best, min(0.5, 0.1 * weight))
        credit += best

    return min(1.0, credit / len(required_skills))


def _career_path_score(
    G: nx.Graph,
    target_role: str,
    candidate_role: str,
) -> float:
    """Bonus if the candidate's current role is one NEXT_STEP away from the
    target role (in either direction on the career ladder).

    Returns 0.0 or a bonus up to 0.15.
    """
    tid = _norm(target_role)
    cid = _norm(candidate_role)

    if tid == cid:
        return 0.15  # exact role match

    if tid not in G or cid not in G:
        return 0.0

    # Check direct NEXT_STEP edge
    if G.has_edge(tid, cid) and G[tid][cid].get("edge_type") == NEXT_STEP:
        return 0.10

    # Check 2-hop path via NEXT_STEP
    for mid_node in G.neighbors(tid):
        if G[tid][mid_node].get("edge_type") != NEXT_STEP:
            continue
        if G.has_edge(mid_node, cid) and G[mid_node][cid].get("edge_type") == NEXT_STEP:
            return 0.05

    return 0.0


def _company_tier_score(
    candidate_company: str,
    target_company: str | None = None,
) -> float:
    """Small bonus if candidate's company is in the same or higher prestige tier.

    Returns 0.0 to 0.10.
    """
    c_tier = COMPANY_TO_TIER.get(_norm(candidate_company))
    if c_tier is None:
        return 0.0

    c_rank = TIER_RANK.get(c_tier, 0)

    if target_company:
        t_tier = COMPANY_TO_TIER.get(_norm(target_company))
        if t_tier is not None:
            t_rank = TIER_RANK.get(t_tier, 0)
            if c_rank >= t_rank:
                return 0.10
            elif c_rank >= t_rank - 1:
                return 0.05
            return 0.0

    # No target company -- reward higher-tier companies slightly
    return min(0.10, c_rank * 0.015)


def _seniority_fit_score(
    target_seniority: str | None,
    candidate_seniority: str | None,
) -> float:
    """Bonus when candidate seniority is close to the target.

    Returns 0.0 to 0.10.
    """
    if not target_seniority or not candidate_seniority:
        return 0.0

    t = SENIORITY_RANK.get(target_seniority.lower())
    c = SENIORITY_RANK.get(candidate_seniority.lower())
    if t is None or c is None:
        return 0.0

    diff = abs(t - c)
    if diff == 0:
        return 0.10
    elif diff == 1:
        return 0.05
    return 0.0


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------

def graph_boost_score(
    graph: nx.Graph,
    query_parsed: dict,
    candidate_parsed: dict,
) -> float:
    """Compute a graph-based boost score for a candidate against a query.

    Parameters
    ----------
    graph : nx.Graph
        The knowledge graph built by ``builder.build_graph``.
    query_parsed : dict
        Parsed job query with keys:
            - required_skills: list[str]
            - nice_to_have_skills: list[str]  (optional)
            - role_title: str
            - seniority: str                  (optional)
            - company: str                    (optional)
    candidate_parsed : dict
        Parsed candidate/resume with keys:
            - skills: list[str]
            - current_title: str
            - seniority: str                  (optional)
            - company: str                    (optional)

    Returns
    -------
    float
        Score in [0.0, 1.0].  Intended to be added as a boost to the base
        retrieval score, not used as a standalone ranking signal.
    """
    candidate_skills = {_norm(s) for s in candidate_parsed.get("skills", [])}

    required = query_parsed.get("required_skills", [])
    nice = query_parsed.get("nice_to_have_skills", [])

    # Skill adjacency (up to 0.65 weight)
    skill_score = _skill_adjacency_score(graph, required + nice, candidate_skills)

    # Career path (up to 0.15)
    career = _career_path_score(
        graph,
        query_parsed.get("role_title", ""),
        candidate_parsed.get("current_title", ""),
    )

    # Company tier (up to 0.10)
    company = _company_tier_score(
        candidate_parsed.get("company", ""),
        query_parsed.get("company"),
    )

    # Seniority fit (up to 0.10)
    seniority = _seniority_fit_score(
        query_parsed.get("seniority"),
        candidate_parsed.get("seniority"),
    )

    # Weighted combination -- weights sum to 1.0
    total = (0.65 * skill_score) + career + company + seniority

    return max(0.0, min(1.0, total))
