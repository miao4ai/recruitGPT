"""
Build (query, candidate, score) pairs for cross-encoder reranker training.

Uses rule-based scoring — no trained model needed.

Input:
  data/generated/train_clean.jsonl  — distilled query parsing outputs
  data/resumes/all_resumes.jsonl    — resume corpus

Output:
  data/reranker/train.jsonl  — 80% of scored pairs
  data/reranker/eval.jsonl   — 20% of scored pairs

Scoring rules:
  1.0 — same role_category + seniority distance <= 1 + skill overlap > 50%
  0.5 — same role_category but seniority gap >= 2 OR skill overlap 20-50%
  0.0 — different role_category

For each query we generate ~3 scored pairs (1 high, 1 medium, 1 low).

Usage:
  python scripts/build_reranker_data.py
  python scripts/build_reranker_data.py --train_ratio 0.9
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------
# Seniority ordering (shared with build_embedding_pairs.py)
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
    that a cross-encoder can consume.
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
# Category inference
# ---------------------------------------------------------------

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
    return "backend"


# ---------------------------------------------------------------
# Skill overlap computation
# ---------------------------------------------------------------

def extract_skills_from_resume(resume: dict) -> set[str]:
    """Pull a normalised skill set from a resume record."""
    text = resume.get("text", "").lower()
    # Check the Skills section first; fall back to full text tokens
    skills_section = ""
    for marker in ["**skills**", "**skills**\n", "skills\n"]:
        idx = text.find(marker)
        if idx != -1:
            skills_section = text[idx:idx + 500]
            break

    if skills_section:
        # Grab the first line after the marker
        lines = skills_section.split("\n")
        for line in lines[1:]:
            line = line.strip()
            if line:
                return {s.strip() for s in line.split(",") if s.strip()}
    # Fallback: return empty (will score via category/seniority only)
    return set()


def skill_overlap_ratio(query_skills: list[str], resume_skills: set[str]) -> float:
    """Fraction of query skills found in resume skill set."""
    if not query_skills:
        return 0.0
    query_lower = {s.lower() for s in query_skills}
    matches = query_lower & resume_skills
    return len(matches) / len(query_lower)


# ---------------------------------------------------------------
# Score a (query, resume) pair
# ---------------------------------------------------------------

def score_pair(query: dict, resume: dict, resume_skills: set[str]) -> float:
    """
    Rule-based relevance score.
      1.0 — same category + seniority <= 1 + skill overlap > 50%
      0.5 — same category but seniority >= 2 OR skill overlap 20-50%
      0.0 — different category
    """
    q_cat = query["role_category"]
    r_cat = resume.get("role_category", "unknown")

    if q_cat != r_cat:
        return 0.0

    sen_dist = seniority_distance(
        query["seniority"], resume.get("seniority", "mid"),
    )
    overlap = skill_overlap_ratio(query["skills"], resume_skills)

    if sen_dist <= 1 and overlap > 0.5:
        return 1.0
    # Same category but weaker match
    return 0.5


# ---------------------------------------------------------------
# Load data
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

            records.append({
                "jd_id": sample.get("source_jd_id", ""),
                "parsed": parsed,
                "query": build_query_string(parsed),
                "role_category": infer_category(parsed),
                "seniority": parsed.get("seniority", "mid"),
                "skills": parsed.get("required_skills", []),
            })
    return records


def load_resumes(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------
# Pair construction
# ---------------------------------------------------------------

def build_scored_pairs(
    queries: list[dict],
    resumes: list[dict],
) -> list[dict]:
    """
    For each query, pick ~3 resumes that yield one high (1.0),
    one medium (0.5), and one low (0.0) score.
    """

    # Index resumes by category
    by_category: dict[str, list[dict]] = defaultdict(list)
    for r in resumes:
        by_category[r.get("role_category", "unknown")].append(r)

    # Pre-compute resume skill sets
    skill_cache: dict[str, set[str]] = {}
    for r in resumes:
        skill_cache[r["id"]] = extract_skills_from_resume(r)

    all_categories = list(by_category.keys())
    pairs = []

    for q in queries:
        q_cat = q["role_category"]
        same_cat = by_category.get(q_cat, [])
        if not same_cat:
            continue

        # --- Try to find a 1.0 candidate ---
        high_candidates = [
            r for r in same_cat
            if seniority_distance(q["seniority"], r.get("seniority", "mid")) <= 1
            and skill_overlap_ratio(q["skills"], skill_cache[r["id"]]) > 0.5
        ]
        if high_candidates:
            pick = random.choice(high_candidates)
            pairs.append({
                "query": q["query"],
                "candidate": pick["text"],
                "score": 1.0,
                "query_category": q_cat,
                "candidate_category": pick.get("role_category"),
            })

        # --- 0.5 candidate: same category, weaker match ---
        mid_candidates = [
            r for r in same_cat
            if score_pair(q, r, skill_cache[r["id"]]) == 0.5
        ]
        if mid_candidates:
            pick = random.choice(mid_candidates)
            pairs.append({
                "query": q["query"],
                "candidate": pick["text"],
                "score": 0.5,
                "query_category": q_cat,
                "candidate_category": pick.get("role_category"),
            })

        # --- 0.0 candidate: different category ---
        other_cats = [c for c in all_categories if c != q_cat]
        if other_cats:
            neg_cat = random.choice(other_cats)
            pick = random.choice(by_category[neg_cat])
            pairs.append({
                "query": q["query"],
                "candidate": pick["text"],
                "score": 0.0,
                "query_category": q_cat,
                "candidate_category": pick.get("role_category"),
            })

    random.shuffle(pairs)
    return pairs


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build (query, candidate, score) pairs for cross-encoder reranker training",
    )
    parser.add_argument("--queries", default="data/generated/train_clean.jsonl",
                        help="Path to distilled query parsing data")
    parser.add_argument("--resumes", default="data/resumes/all_resumes.jsonl",
                        help="Path to resume corpus")
    parser.add_argument("--output_dir", default="data/reranker",
                        help="Directory for train.jsonl and eval.jsonl")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Fraction of pairs for training (rest goes to eval)")
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

    # Build scored pairs
    pairs = build_scored_pairs(queries, resumes)

    # Stats by score
    score_counts = defaultdict(int)
    for p in pairs:
        score_counts[p["score"]] += 1

    print(f"\nBuilt {len(pairs)} scored pairs:")
    for score in sorted(score_counts.keys(), reverse=True):
        print(f"  score={score:.1f}: {score_counts[score]}")

    # Train / eval split
    split_idx = int(len(pairs) * args.train_ratio)
    train_pairs = pairs[:split_idx]
    eval_pairs = pairs[split_idx:]

    print(f"\nSplit: {len(train_pairs)} train, {len(eval_pairs)} eval")

    # Write output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.jsonl"
    eval_path = output_dir / "eval.jsonl"

    for path, data in [(train_path, train_pairs), (eval_path, eval_pairs)]:
        with open(path, "w") as f:
            for p in data:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"Wrote {len(data)} pairs -> {path}")


if __name__ == "__main__":
    main()
