"""
Build (query, positive, negative) triplets for BGE embedding fine-tuning.

Input:
  data/generated/train_clean.jsonl  — distilled query parsing outputs (for structured queries)
  data/jds/all_jds.jsonl            — raw JD texts
  data/resumes/all_resumes.jsonl    — raw resume texts

Output:
  data/pairs/train_triplets.jsonl   — one triplet per line

Each triplet:
  {
    "query": "<structured query string from parsed JD>",
    "positive": "<resume text that matches>",
    "negative": "<resume text that does NOT match>"
  }

Matching logic:
  - positive: same role_category AND similar seniority (±1 level)
  - easy negative: different role_category
  - semi-hard negative: same role_category but seniority gap >= 2

Usage:
  python scripts/build_embedding_pairs.py
  python scripts/build_embedding_pairs.py --output data/pairs/train_triplets.jsonl
  python scripts/build_embedding_pairs.py --negatives_per_positive 3
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------
# Seniority ordering (for proximity matching)
# ---------------------------------------------------------------

SENIORITY_ORDER = {
    "intern": 0, "junior": 1, "mid": 2, "senior": 3,
    "staff": 4, "lead": 5, "manager": 6, "director": 7,
}


def seniority_distance(s1: str, s2: str) -> int:
    a = SENIORITY_ORDER.get(s1, 2)
    b = SENIORITY_ORDER.get(s2, 2)
    return abs(a - b)


# ---------------------------------------------------------------
# Build structured query from parsed JD
# ---------------------------------------------------------------

def build_query_string(parsed: dict) -> str:
    """
    Convert a query_parsing JSON output into a flat text query
    that BGE can encode. Includes the most discriminative fields.
    """
    parts = []

    if parsed.get("role_title"):
        parts.append(parsed["role_title"])

    if parsed.get("seniority"):
        parts.append(f"Seniority: {parsed['seniority']}")

    skills = parsed.get("required_skills", [])
    if skills:
        parts.append(f"Skills: {', '.join(skills)}")

    if parsed.get("industry_context"):
        parts.append(f"Industry: {parsed['industry_context']}")

    constraints = parsed.get("hard_constraints", [])
    if constraints:
        parts.append(f"Requirements: {'; '.join(constraints)}")

    return " | ".join(parts)


# ---------------------------------------------------------------
# Load data
# ---------------------------------------------------------------

def load_parsed_queries(path: str) -> list[dict]:
    """Load query_parsing samples and extract the parsed JSON + metadata."""
    records = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            sample = json.loads(line)
            if sample.get("task") != "query_parsing":
                continue

            # Extract the assistant output (parsed JSON)
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

            records.append({
                "jd_id": sample.get("source_jd_id", ""),
                "parsed": parsed,
                "query": build_query_string(parsed),
                "role_category": infer_category(parsed),
                "seniority": parsed.get("seniority", "mid"),
            })
    return records


def infer_category(parsed: dict) -> str:
    """Best-effort mapping from parsed JD to role category."""
    title = (parsed.get("role_title", "") or "").lower()
    skills = [s.lower() for s in parsed.get("required_skills", [])]
    all_text = title + " " + " ".join(skills)

    category_signals = {
        "backend": ["backend", "go", "rust", "java", "microservices", "api engineer"],
        "frontend": ["frontend", "react", "vue", "ui engineer", "next.js"],
        "data": ["data engineer", "data scientist", "ml engineer", "spark", "airflow", "pytorch"],
        "infra": ["devops", "sre", "infrastructure", "terraform", "kubernetes"],
        "product": ["product manager", "product owner", "growth pm"],
        "design": ["ux", "ui/ux", "product designer"],
        "mobile": ["ios", "android", "mobile", "react native", "flutter"],
        "management": ["engineering manager", "tech lead", "director of engineering"],
        "finance": ["financial", "accountant", "portfolio", "risk analyst", "controller", "investment"],
        "healthcare": ["nurse", "physician", "pharmacist", "therapist", "clinical", "patient care"],
        "trades": ["electrician", "plumber", "hvac", "welder", "cnc", "construction"],
        "sales": ["account executive", "sales", "customer success", "solutions engineer"],
        "marketing": ["marketing", "seo", "content strategist", "brand manager"],
        "operations": ["operations manager", "supply chain", "logistics", "warehouse", "procurement"],
        "legal": ["counsel", "paralegal", "compliance", "attorney", "contract manager"],
        "hr": ["hr", "talent acquisition", "recruiter", "people operations"],
        "education": ["teacher", "instructional", "curriculum", "trainer", "academic"],
    }

    for cat, signals in category_signals.items():
        for signal in signals:
            if signal in all_text:
                return cat
    return "backend"  # fallback


def load_resumes(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------
# Triplet construction
# ---------------------------------------------------------------

def build_triplets(
    queries: list[dict],
    resumes: list[dict],
    negatives_per_positive: int = 2,
) -> list[dict]:
    """
    For each parsed JD query, find matching and non-matching resumes.
    Returns a list of {"query", "positive", "negative"} dicts.
    """

    # Index resumes by category
    by_category: dict[str, list[dict]] = defaultdict(list)
    for r in resumes:
        cat = r.get("role_category", "unknown")
        by_category[cat].append(r)

    all_categories = list(by_category.keys())
    triplets = []

    for q in queries:
        q_cat = q["role_category"]
        q_seniority = q["seniority"]

        # --- Find positives: same category, seniority distance <= 1 ---
        candidates_same = by_category.get(q_cat, [])
        positives = [
            r for r in candidates_same
            if seniority_distance(q_seniority, r.get("seniority", "mid")) <= 1
        ]

        if not positives:
            # Fallback: any resume in the same category
            positives = candidates_same

        if not positives:
            continue  # no match possible

        for _ in range(negatives_per_positive):
            pos = random.choice(positives)

            # --- Easy negative: different category ---
            other_cats = [c for c in all_categories if c != q_cat]
            if other_cats:
                neg_cat = random.choice(other_cats)
                easy_neg = random.choice(by_category[neg_cat])
                triplets.append({
                    "query": q["query"],
                    "positive": pos["text"],
                    "negative": easy_neg["text"],
                    "type": "easy",
                    "query_category": q_cat,
                    "pos_category": pos.get("role_category"),
                    "neg_category": easy_neg.get("role_category"),
                })

            # --- Semi-hard negative: same category, seniority gap >= 2 ---
            hard_negs = [
                r for r in candidates_same
                if seniority_distance(q_seniority, r.get("seniority", "mid")) >= 2
            ]
            if hard_negs:
                hard_neg = random.choice(hard_negs)
                triplets.append({
                    "query": q["query"],
                    "positive": pos["text"],
                    "negative": hard_neg["text"],
                    "type": "semi_hard",
                    "query_category": q_cat,
                    "pos_category": pos.get("role_category"),
                    "neg_category": hard_neg.get("role_category"),
                })

    random.shuffle(triplets)
    return triplets


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build embedding training triplets")
    parser.add_argument("--queries", default="data/generated/train_clean.jsonl",
                        help="Path to distilled query parsing data")
    parser.add_argument("--resumes", default="data/resumes/all_resumes.jsonl",
                        help="Path to resume corpus")
    parser.add_argument("--output", default="data/pairs/train_triplets.jsonl")
    parser.add_argument("--negatives_per_positive", type=int, default=2,
                        help="Number of negative samples per positive")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Load data
    print(f"Loading queries from {args.queries}")
    queries = load_parsed_queries(args.queries)
    print(f"  Loaded {len(queries)} query parsing samples")

    print(f"Loading resumes from {args.resumes}")
    resumes = load_resumes(args.resumes)
    print(f"  Loaded {len(resumes)} resumes")

    # Category distribution
    q_cats = defaultdict(int)
    for q in queries:
        q_cats[q["role_category"]] += 1
    r_cats = defaultdict(int)
    for r in resumes:
        r_cats[r.get("role_category", "unknown")] += 1

    print(f"\nQuery categories: {dict(q_cats)}")
    print(f"Resume categories: {dict(r_cats)}")

    # Build triplets
    triplets = build_triplets(queries, resumes, args.negatives_per_positive)

    # Stats
    easy = sum(1 for t in triplets if t["type"] == "easy")
    semi_hard = sum(1 for t in triplets if t["type"] == "semi_hard")
    print(f"\nBuilt {len(triplets)} triplets:")
    print(f"  Easy negatives:      {easy}")
    print(f"  Semi-hard negatives: {semi_hard}")

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for t in triplets:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"\nWrote → {output_path}")


if __name__ == "__main__":
    main()
