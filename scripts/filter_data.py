"""
Quality filtering for distillation training data.

Reads:  data/generated/train_raw.jsonl
Writes: data/generated/train_clean.jsonl

Filters applied per task:

  query_parsing:
    - Output is valid JSON
    - All required schema fields are present
    - required_skills is a non-empty list
    - seniority is a known value
    - Total text length is within bounds (not too short / not truncated)

  match_explanation:
    - All four required sections are present (Strengths, Gaps, Interview Focus, Recommendation)
    - No section is empty
    - Recommendation is a single sentence

Usage:
    python scripts/filter_data.py
    python scripts/filter_data.py --input data/generated/train_raw.jsonl --output data/generated/train_clean.jsonl
    python scripts/filter_data.py --stats   # print filter breakdown without writing
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------
# Constants
# ---------------------------------------------------------------

REQUIRED_QP_FIELDS = {
    "role_title",
    "seniority",
    "required_skills",
    "nice_to_have_skills",
    "industry_context",
    "hard_constraints",
    "soft_signals",
    "team_context",
}

VALID_SENIORITY = {
    "intern", "junior", "mid", "senior", "staff", "lead", "manager", "director",
}

REQUIRED_ME_SECTIONS = [
    "**Strengths**",
    "**Gaps**",
    "**Interview Focus**",
    "**Recommendation**",
]

# Approximate token counts via char proxy (1 token ≈ 4 chars)
MIN_TOTAL_CHARS = 400    # too short → likely truncated or empty output
MAX_TOTAL_CHARS = 16000  # too long → likely repeated / runaway generation


# ---------------------------------------------------------------
# Per-task validators
# ---------------------------------------------------------------

def _get_assistant_output(sample: dict) -> str:
    """Extract the last assistant message content."""
    for msg in reversed(sample.get("messages", [])):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""


def _get_full_text(sample: dict) -> str:
    return " ".join(
        msg.get("content", "") for msg in sample.get("messages", [])
    )


def validate_query_parsing(sample: dict) -> tuple[bool, str]:
    output = _get_assistant_output(sample)

    if not output:
        return False, "empty_output"

    # Valid JSON
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        return False, "invalid_json"

    if not isinstance(parsed, dict):
        return False, "output_not_dict"

    # Required fields present
    missing = REQUIRED_QP_FIELDS - parsed.keys()
    if missing:
        return False, f"missing_fields:{','.join(sorted(missing))}"

    # required_skills non-empty list
    skills = parsed.get("required_skills")
    if not isinstance(skills, list) or len(skills) == 0:
        return False, "empty_required_skills"

    # Valid seniority
    seniority = parsed.get("seniority", "")
    if seniority not in VALID_SENIORITY:
        return False, f"invalid_seniority:{seniority}"

    # Length sanity check on full text
    total_chars = len(_get_full_text(sample))
    if total_chars < MIN_TOTAL_CHARS:
        return False, "too_short"
    if total_chars > MAX_TOTAL_CHARS:
        return False, "too_long"

    return True, "ok"


def validate_match_explanation(sample: dict) -> tuple[bool, str]:
    output = _get_assistant_output(sample)

    if not output:
        return False, "empty_output"

    # All four sections present
    for section in REQUIRED_ME_SECTIONS:
        if section not in output:
            return False, f"missing_section:{section}"

    # No section is empty (next section or end of string immediately follows header)
    lines = output.split("\n")
    section_indices = {}
    for i, line in enumerate(lines):
        for section in REQUIRED_ME_SECTIONS:
            if line.strip().startswith(section):
                section_indices[section] = i

    for section, idx in section_indices.items():
        # Collect lines between this section header and the next
        next_indices = [j for j in section_indices.values() if j > idx]
        end = min(next_indices) if next_indices else len(lines)
        content_lines = [l.strip() for l in lines[idx+1:end] if l.strip()]
        if not content_lines:
            return False, f"empty_section:{section}"

    # Recommendation is a single sentence (no newlines in content)
    rec_idx = section_indices.get("**Recommendation**")
    if rec_idx is not None:
        rec_lines = [l.strip() for l in lines[rec_idx+1:] if l.strip()]
        if len(rec_lines) > 2:
            return False, "recommendation_too_long"

    # Length sanity
    total_chars = len(_get_full_text(sample))
    if total_chars < MIN_TOTAL_CHARS:
        return False, "too_short"
    if total_chars > MAX_TOTAL_CHARS:
        return False, "too_long"

    return True, "ok"


VALIDATORS = {
    "query_parsing": validate_query_parsing,
    "match_explanation": validate_match_explanation,
}


# ---------------------------------------------------------------
# Main filter loop
# ---------------------------------------------------------------

def filter_dataset(input_path: Path, output_path: Path, stats_only: bool = False):
    records = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    print(f"Loaded {len(records)} raw samples from {input_path}")

    passed = []
    reject_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    task_totals: dict[str, int] = defaultdict(int)

    for sample in records:
        task = sample.get("task", "unknown")
        task_totals[task] += 1

        validator = VALIDATORS.get(task)
        if validator is None:
            reject_counts[task]["unknown_task"] += 1
            continue

        ok, reason = validator(sample)
        if ok:
            passed.append(sample)
        else:
            reject_counts[task][reason] += 1

    # Print stats
    print(f"\n{'='*50}")
    print(f"Filter results")
    print(f"{'='*50}")
    for task, total in task_totals.items():
        n_passed = sum(1 for s in passed if s.get("task") == task)
        n_rejected = total - n_passed
        pct = 100 * n_passed / total if total else 0
        print(f"\n{task}:")
        print(f"  Total:    {total}")
        print(f"  Passed:   {n_passed}  ({pct:.1f}%)")
        print(f"  Rejected: {n_rejected}")
        if reject_counts[task]:
            for reason, count in sorted(reject_counts[task].items(), key=lambda x: -x[1]):
                print(f"    {reason}: {count}")

    print(f"\n{'='*50}")
    print(f"Total passed: {len(passed)} / {len(records)}  ({100*len(passed)/len(records):.1f}%)")

    if stats_only:
        print("\n--stats mode: no output written.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for sample in passed:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(passed)} clean samples → {output_path}")


# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Filter distillation training data")
    parser.add_argument("--input", type=str, default="data/generated/train_raw.jsonl")
    parser.add_argument("--output", type=str, default="data/generated/train_clean.jsonl")
    parser.add_argument("--stats", action="store_true",
                        help="Print filter breakdown without writing output file")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} not found. Run scripts/distill_data.py first."
        )

    filter_dataset(input_path, output_path, stats_only=args.stats)


if __name__ == "__main__":
    main()
