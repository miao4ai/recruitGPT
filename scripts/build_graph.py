"""
Build a knowledge graph from JDs and resumes using NetworkX.

Input:
  data/generated/train_clean.jsonl  — parsed JD data
  data/resumes/all_resumes.jsonl    — resume corpus

Output:
  data/graph/recruit_graph.gpickle

Node types: Skill, Role, Company, Industry, Seniority
Edge types:
  Skill  --RELATED_TO-->  Skill     (co-occurrence in same JD/resume)
  Skill  --USED_IN-->     Role
  Company --IN_TIER-->    Tier      (uses COMPANIES tiers from generate_resumes.py)
  Role   --NEXT_STEP-->   Role      (seniority progression)
  Industry --HIRES-->     Role

Usage:
  python scripts/build_graph.py
  python scripts/build_graph.py --output data/graph/recruit_graph.gpickle
"""

import argparse
import json
import pickle
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import networkx as nx

# ---------------------------------------------------------------
# Company tier mapping (from generate_resumes.py)
# ---------------------------------------------------------------

COMPANIES = {
    "tier1_tech": [
        "Google", "Meta", "Apple", "Amazon", "Microsoft", "Netflix",
        "ByteDance", "Alibaba", "Tencent", "Baidu",
    ],
    "tier2_tech": [
        "Stripe", "Airbnb", "Uber", "Lyft", "Snap", "Twitter/X",
        "Shopify", "Atlassian", "Databricks", "Figma", "Notion",
    ],
    "fintech": [
        "Ant Group", "Revolut", "Robinhood", "Coinbase", "Plaid",
        "Wise", "Chime", "Nubank", "Klarna",
    ],
    "startup": [
        "a Series A startup", "a Series B SaaS company",
        "an early-stage fintech startup", "a growth-stage e-commerce company",
    ],
    "finance": [
        "Goldman Sachs", "JP Morgan", "Morgan Stanley", "BlackRock",
        "Deloitte", "PwC", "EY", "KPMG", "Citadel", "Bridgewater",
    ],
    "healthcare": [
        "Mayo Clinic", "Kaiser Permanente", "Cleveland Clinic",
        "Johns Hopkins Hospital", "HCA Healthcare", "Pfizer", "Johnson & Johnson",
    ],
    "trades": [
        "Turner Construction", "Bechtel", "Fluor Corporation",
        "AECOM", "a regional electrical contractor", "a commercial HVAC company",
    ],
    "general": [
        "Walmart", "Target", "Costco", "McKinsey", "BCG",
        "Salesforce", "HubSpot", "Nike", "Disney", "Marriott",
    ],
}

# Invert to company -> tier
COMPANY_TO_TIER: dict[str, str] = {}
for tier, companies in COMPANIES.items():
    for company in companies:
        COMPANY_TO_TIER[company.lower()] = tier

# ---------------------------------------------------------------
# Seniority progression chains per category
# ---------------------------------------------------------------

SENIORITY_PROGRESSIONS = {
    "tech": [
        ("Intern", "Junior Engineer"),
        ("Junior Engineer", "Mid Engineer"),
        ("Mid Engineer", "Senior Engineer"),
        ("Senior Engineer", "Staff Engineer"),
        ("Staff Engineer", "Principal Engineer"),
        ("Senior Engineer", "Engineering Manager"),
        ("Engineering Manager", "Director of Engineering"),
    ],
    "finance": [
        ("Analyst", "Senior Analyst"),
        ("Senior Analyst", "Manager"),
        ("Manager", "Director"),
        ("Director", "VP"),
    ],
    "healthcare": [
        ("Registered Nurse", "Senior Nurse"),
        ("Senior Nurse", "Nurse Practitioner"),
        ("Nurse Practitioner", "Clinical Lead"),
    ],
    "trades": [
        ("Apprentice", "Journeyman"),
        ("Journeyman", "Master"),
        ("Master", "Site Supervisor"),
    ],
    "general": [
        ("Associate", "Senior"),
        ("Senior", "Manager"),
        ("Manager", "Director"),
    ],
}

# Map role_category -> progression key
CATEGORY_TO_PROGRESSION = {
    "backend": "tech", "frontend": "tech", "data": "tech",
    "infra": "tech", "mobile": "tech", "management": "tech",
    "finance": "finance",
    "healthcare": "healthcare",
    "trades": "trades",
    "product": "general", "sales": "general", "marketing": "general",
    "operations": "general", "legal": "general", "hr": "general",
    "education": "general", "design": "general",
}

# Map role_category -> industry
CATEGORY_TO_INDUSTRY = {
    "backend": "Technology", "frontend": "Technology", "data": "Technology",
    "infra": "Technology", "mobile": "Technology", "management": "Technology",
    "product": "Technology", "design": "Technology",
    "finance": "Finance", "healthcare": "Healthcare", "trades": "Trades",
    "sales": "Sales & Marketing", "marketing": "Sales & Marketing",
    "operations": "Operations & Logistics", "legal": "Legal",
    "hr": "Human Resources", "education": "Education",
}


# ---------------------------------------------------------------
# Load data (same loaders as build_embedding_pairs.py)
# ---------------------------------------------------------------

def load_parsed_queries(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            sample = json.loads(line)
            if sample.get("task") != "query_parsing":
                continue

            assistant_content = None
            for msg in reversed(sample.get("messages", [])):
                if msg.get("role") == "assistant":
                    assistant_content = msg["content"]
                    break

            if not assistant_content:
                continue

            try:
                parsed = json.loads(assistant_content)
            except json.JSONDecodeError:
                continue

            records.append(parsed)
    return records


def load_resumes(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def extract_skills_from_resume(resume: dict) -> list[str]:
    """Pull skill list from a resume's text (Skills section)."""
    text = resume.get("text", "")
    lower = text.lower()
    for marker in ["**skills**", "**skills**\n", "skills\n"]:
        idx = lower.find(marker)
        if idx != -1:
            section = text[idx:idx + 500]
            lines = section.split("\n")
            for line in lines[1:]:
                line = line.strip()
                if line:
                    return [s.strip() for s in line.split(",") if s.strip()]
    return []


# ---------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------

def add_node(G: nx.Graph, name: str, node_type: str) -> None:
    if not G.has_node(name):
        G.add_node(name, type=node_type)


def add_edge(G: nx.Graph, src: str, dst: str, edge_type: str) -> None:
    if G.has_edge(src, dst):
        G[src][dst]["weight"] = G[src][dst].get("weight", 1) + 1
    else:
        G.add_edge(src, dst, type=edge_type, weight=1)


def build_graph(jd_records: list[dict], resumes: list[dict]) -> nx.Graph:
    G = nx.Graph()

    # ----------------------------------------------------------
    # 1. Process JDs: skills, roles, industries
    # ----------------------------------------------------------
    for parsed in jd_records:
        role_title = parsed.get("role_title", "")
        skills = parsed.get("required_skills", [])
        industry = parsed.get("industry_context", "")
        seniority = parsed.get("seniority", "")

        if role_title:
            add_node(G, role_title, "Role")

        if seniority:
            add_node(G, seniority, "Seniority")

        # Skill nodes + USED_IN edges
        for skill in skills:
            add_node(G, skill, "Skill")
            if role_title:
                add_edge(G, skill, role_title, "USED_IN")

        # Skill co-occurrence (RELATED_TO)
        for s1, s2 in combinations(skills, 2):
            add_edge(G, s1, s2, "RELATED_TO")

        # Industry -> Role (HIRES)
        if industry:
            add_node(G, industry, "Industry")
            if role_title:
                add_edge(G, industry, role_title, "HIRES")

    # ----------------------------------------------------------
    # 2. Process resumes: skills, companies, roles
    # ----------------------------------------------------------
    for resume in resumes:
        role_cat = resume.get("role_category", "")
        current_title = resume.get("current_title", "")
        skills = extract_skills_from_resume(resume)

        if current_title:
            add_node(G, current_title, "Role")

        # Skill nodes + USED_IN edges
        for skill in skills:
            add_node(G, skill, "Skill")
            if current_title:
                add_edge(G, skill, current_title, "USED_IN")

        # Skill co-occurrence from resume
        for s1, s2 in combinations(skills, 2):
            add_edge(G, s1, s2, "RELATED_TO")

        # Extract company mentions and add Company --IN_TIER--> Tier
        text_lower = resume.get("text", "").lower()
        for company, tier in COMPANY_TO_TIER.items():
            if company in text_lower:
                # Use original-case company name from COMPANIES dict
                orig = next(
                    (c for cs in COMPANIES.values() for c in cs if c.lower() == company),
                    company,
                )
                add_node(G, orig, "Company")
                add_node(G, tier, "Tier")
                add_edge(G, orig, tier, "IN_TIER")

        # Industry --HIRES--> Role
        industry = CATEGORY_TO_INDUSTRY.get(role_cat, "")
        if industry and current_title:
            add_node(G, industry, "Industry")
            add_edge(G, industry, current_title, "HIRES")

    # ----------------------------------------------------------
    # 3. Seniority progressions: Role --NEXT_STEP--> Role
    # ----------------------------------------------------------
    seen_progressions = set()
    for prog_key, chain in SENIORITY_PROGRESSIONS.items():
        for role_from, role_to in chain:
            if (role_from, role_to) not in seen_progressions:
                add_node(G, role_from, "Role")
                add_node(G, role_to, "Role")
                add_edge(G, role_from, role_to, "NEXT_STEP")
                seen_progressions.add((role_from, role_to))

    return G


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build a recruitment knowledge graph from JDs and resumes",
    )
    parser.add_argument("--queries", default="data/generated/train_clean.jsonl",
                        help="Path to distilled query parsing data")
    parser.add_argument("--resumes", default="data/resumes/all_resumes.jsonl",
                        help="Path to resume corpus")
    parser.add_argument("--output", default="data/graph/recruit_graph.gpickle",
                        help="Output path for the graph pickle")
    args = parser.parse_args()

    # Load data
    print(f"Loading JD records from {args.queries}")
    jd_records = load_parsed_queries(args.queries)
    print(f"  Loaded {len(jd_records)} parsed JDs")

    print(f"Loading resumes from {args.resumes}")
    resumes = load_resumes(args.resumes)
    print(f"  Loaded {len(resumes)} resumes")

    # Build graph
    print("\nBuilding knowledge graph...")
    G = build_graph(jd_records, resumes)

    # Stats
    node_types = defaultdict(int)
    for _, data in G.nodes(data=True):
        node_types[data.get("type", "unknown")] += 1

    edge_types = defaultdict(int)
    for _, _, data in G.edges(data=True):
        edge_types[data.get("type", "unknown")] += 1

    print(f"\nGraph stats:")
    print(f"  Nodes: {G.number_of_nodes()}")
    for ntype, count in sorted(node_types.items()):
        print(f"    {ntype}: {count}")
    print(f"  Edges: {G.number_of_edges()}")
    for etype, count in sorted(edge_types.items()):
        print(f"    {etype}: {count}")

    # Top-connected nodes
    top_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\n  Top-10 connected nodes:")
    for name, degree in top_nodes:
        ntype = G.nodes[name].get("type", "?")
        print(f"    {name} ({ntype}): degree={degree}")

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nWrote graph -> {output_path}")


if __name__ == "__main__":
    main()
