"""
Generate synthetic LinkedIn-style resumes using Claude as teacher model.

Usage:
    python scripts/generate_resumes.py --num 200 --output data/resumes/
    python scripts/generate_resumes.py --num 100 --roles "backend,data" --output data/resumes/

Output format (one JSON per file + consolidated data/resumes/all_resumes.jsonl):
    {
        "id": "resume_0001",
        "name": "Alex Chen",
        "role_category": "backend",
        "current_title": "Senior Backend Engineer",
        "seniority": "senior",
        "years_of_experience": 7,
        "text": "<full LinkedIn-style resume text>"
    }
"""

import anthropic
import argparse
import json
import os
import random
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-sonnet-4-6"

# ---------------------------------------------------------------
# Diversity matrix
# ---------------------------------------------------------------

ROLE_TITLES = {
    "backend": [
        "Backend Engineer", "Senior Backend Engineer", "Staff Engineer",
        "Platform Engineer", "Software Engineer", "Senior Software Engineer",
    ],
    "frontend": [
        "Frontend Engineer", "Senior Frontend Engineer", "UI Engineer",
        "Full-Stack Engineer", "React Developer",
    ],
    "data": [
        "Data Engineer", "Senior Data Engineer", "Analytics Engineer",
        "Data Scientist", "ML Engineer", "Senior ML Engineer",
    ],
    "infra": [
        "DevOps Engineer", "Site Reliability Engineer", "Senior SRE",
        "Cloud Engineer", "Infrastructure Engineer",
    ],
    "product": [
        "Product Manager", "Senior Product Manager", "Technical PM",
        "Associate PM", "Group Product Manager",
    ],
    "mobile": [
        "iOS Engineer", "Android Engineer", "Senior Mobile Engineer",
        "React Native Engineer",
    ],
    "management": [
        "Engineering Manager", "Tech Lead", "Senior Engineering Manager",
    ],
}

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
}

UNIVERSITIES = [
    "MIT", "Stanford University", "Carnegie Mellon University",
    "UC Berkeley", "University of Waterloo", "ETH Zurich",
    "Tsinghua University", "Peking University", "NUS",
    "University of Michigan", "Georgia Tech", "Columbia University",
]

DEGREES = ["B.S. Computer Science", "B.Eng. Software Engineering",
           "M.S. Computer Science", "B.S. Mathematics", "M.S. Data Science"]

NAMES = [
    "Alex Chen", "Jamie Liu", "Sam Park", "Jordan Wang", "Taylor Zhang",
    "Morgan Li", "Casey Kim", "Riley Zhao", "Drew Wu", "Avery Huang",
    "Blake Ng", "Cameron Sun", "Dana Zhou", "Evan Lim", "Fiona Tan",
    "Grace Xu", "Henry Ma", "Iris Guo", "Jason Fu", "Karen Ye",
    "Leo Han", "Mia Song", "Nathan Bai", "Olivia Dai", "Peter Lu",
    "Quinn Shi", "Rachel Jin", "Steven Feng", "Tina Jiang", "Victor Wei",
]

SYSTEM_PROMPT = """\
You are generating realistic synthetic LinkedIn profiles for software engineering candidates.
Write profiles that read like real LinkedIn pages — specific, varied, and credible.
Include real-sounding company names, project details, and quantified achievements.
Do not use generic filler phrases like "passionate about technology" or "team player".\
"""

RESUME_TEMPLATE = """\
Generate a LinkedIn-style resume/profile with these characteristics:

Name: {name}
Current title: {current_title}
Years of experience: {years_exp}
Previous companies: {companies}
University: {university} — {degree}
Role focus: {role_category}
Key skills to include (use 5-7 of these, add 1-2 of your own): {skills}

Format:
## {name}
{current_title}

**Summary**
2-3 sentences. Specific, no generic buzzwords.

**Experience**
For each role (2-3 roles total):
### [Job Title] | [Company] | [Start Year] – [End Year or Present]
- 2-3 bullet points with specific achievements and numbers
- e.g. "Reduced P99 latency from 800ms to 120ms by migrating from REST to gRPC"

**Skills**
Comma-separated list

**Education**
### {degree} | {university} | [Graduation Year]

Rules:
- Make achievements specific and quantified where possible
- Vary writing style between candidates
- Keep total length 300–450 words
- Write in English
- Do NOT include contact info, photos, or links\
"""

SKILL_POOLS = {
    "backend": ["Go", "Python", "Java", "Rust", "Node.js", "PostgreSQL", "MySQL",
                "Redis", "Kafka", "gRPC", "Kubernetes", "Docker", "AWS", "GCP",
                "microservices", "distributed systems", "REST API"],
    "frontend": ["React", "TypeScript", "Next.js", "Vue.js", "GraphQL",
                 "CSS", "Webpack", "Jest", "Figma", "Storybook"],
    "data": ["Python", "SQL", "Spark", "Airflow", "dbt", "BigQuery", "Snowflake",
             "Pandas", "PyTorch", "TensorFlow", "scikit-learn", "Kafka", "Flink"],
    "infra": ["Kubernetes", "Terraform", "AWS", "GCP", "Docker", "CI/CD",
              "Prometheus", "Grafana", "Linux", "Ansible", "Helm"],
    "product": ["SQL", "A/B testing", "user research", "Figma", "Jira",
                "OKRs", "roadmap planning", "stakeholder management"],
    "mobile": ["Swift", "Kotlin", "React Native", "Flutter", "iOS SDK",
               "Android SDK", "REST API", "CI/CD"],
    "management": ["team building", "technical roadmap", "Agile", "mentorship",
                   "hiring", "OKRs", "cross-functional collaboration"],
}


def build_spec(role_category: str) -> dict:
    title = random.choice(ROLE_TITLES[role_category])

    # Years of experience from title
    title_lower = title.lower()
    if any(x in title_lower for x in ["staff", "principal", "director"]):
        years_exp = random.randint(8, 15)
    elif any(x in title_lower for x in ["senior", "sr.", "manager", "lead"]):
        years_exp = random.randint(4, 9)
    else:
        years_exp = random.randint(1, 4)

    # Pick 2 previous companies
    all_companies = (
        COMPANIES["tier1_tech"] + COMPANIES["tier2_tech"] +
        COMPANIES["fintech"] + COMPANIES["startup"]
    )
    companies = random.sample(all_companies, k=2)

    skills = random.sample(SKILL_POOLS.get(role_category, SKILL_POOLS["backend"]), k=6)

    return {
        "name": random.choice(NAMES),
        "current_title": title,
        "years_exp": years_exp,
        "companies": " and ".join(companies),
        "university": random.choice(UNIVERSITIES),
        "degree": random.choice(DEGREES),
        "role_category": role_category,
        "skills": ", ".join(skills),
    }


def generate_resume(client: anthropic.Anthropic, spec: dict) -> str:
    prompt = RESUME_TEMPLATE.format(**spec)
    message = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()


def infer_seniority(title: str) -> str:
    t = title.lower()
    if any(x in t for x in ["staff", "principal", "director"]):
        return "staff"
    if any(x in t for x in ["senior", "sr.", "lead"]):
        return "senior"
    if any(x in t for x in ["manager"]):
        return "manager"
    return "mid"


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic LinkedIn resumes via Claude")
    parser.add_argument("--num", type=int, default=200, help="Total resumes to generate")
    parser.add_argument(
        "--roles",
        type=str,
        default="backend,frontend,data,infra,product,mobile",
        help="Comma-separated role categories",
    )
    parser.add_argument("--output", type=str, default="data/resumes/")
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "all_resumes.jsonl"

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set — add it to .env")
    client = anthropic.Anthropic(api_key=api_key)

    role_categories = [r.strip() for r in args.roles.split(",")]
    per_category = args.num // len(role_categories)
    specs = []
    for cat in role_categories:
        if cat not in ROLE_TITLES:
            print(f"Unknown category '{cat}', skipping")
            continue
        for _ in range(per_category):
            specs.append(build_spec(cat))
    while len(specs) < args.num:
        specs.append(build_spec(random.choice(role_categories)))
    random.shuffle(specs)

    print(f"Generating {len(specs)} resumes → {output_dir}")
    generated = []
    failed = 0

    for i, spec in enumerate(specs):
        resume_id = f"resume_{i+1:04d}"
        try:
            text = generate_resume(client, spec)
            record = {
                "id": resume_id,
                "name": spec["name"],
                "role_category": spec["role_category"],
                "current_title": spec["current_title"],
                "seniority": infer_seniority(spec["current_title"]),
                "years_of_experience": spec["years_exp"],
                "text": text,
            }
            generated.append(record)

            with open(output_dir / f"{resume_id}.json", "w") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)

            print(f"[{i+1}/{len(specs)}] {resume_id}: {spec['name']} — {spec['current_title']}")
            time.sleep(args.delay)

        except Exception as e:
            print(f"[{i+1}/{len(specs)}] {resume_id} FAILED: {e}")
            failed += 1
            time.sleep(2)

    with open(jsonl_path, "w") as f:
        for record in generated:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nDone. {len(generated)} resumes saved to {output_dir}")
    print(f"Consolidated: {jsonl_path}")
    if failed:
        print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
