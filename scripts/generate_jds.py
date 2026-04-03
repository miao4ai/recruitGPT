"""
Generate synthetic job descriptions using Claude as teacher model.

Usage:
    python scripts/generate_jds.py --num 200 --output data/jds/
    python scripts/generate_jds.py --num 50 --roles "backend,data" --output data/jds/

Output format (one JSON per file, also appended to data/jds/all_jds.jsonl):
    {
        "id": "jd_0001",
        "role": "Senior Backend Engineer",
        "industry": "Fintech",
        "seniority": "senior",
        "skills_focus": ["Go", "Kubernetes", "PostgreSQL"],
        "text": "<full JD text>"
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

# ---------------------------------------------------------------------------
# Diversity matrix — Claude will pick from these when generating each JD
# ---------------------------------------------------------------------------

ROLES = {
    # --- Tech ---
    "backend": [
        "Backend Engineer", "Senior Backend Engineer", "Staff Backend Engineer",
        "Platform Engineer", "API Engineer", "Distributed Systems Engineer",
    ],
    "frontend": [
        "Frontend Engineer", "Senior Frontend Engineer", "UI Engineer",
        "React Developer", "Full-Stack Engineer",
    ],
    "data": [
        "Data Engineer", "Senior Data Engineer", "Analytics Engineer",
        "Data Scientist", "ML Engineer", "MLOps Engineer",
    ],
    "infra": [
        "DevOps Engineer", "Site Reliability Engineer", "Cloud Infrastructure Engineer",
        "Platform Engineer", "Security Engineer",
    ],
    "product": [
        "Product Manager", "Senior Product Manager", "Technical Product Manager",
        "Growth PM",
    ],
    "design": [
        "UX Designer", "Product Designer", "Senior UI/UX Designer",
    ],
    "mobile": [
        "iOS Engineer", "Android Engineer", "React Native Engineer",
        "Senior Mobile Engineer",
    ],
    "management": [
        "Engineering Manager", "Tech Lead", "Director of Engineering",
    ],
    # --- Finance ---
    "finance": [
        "Financial Analyst", "Senior Financial Analyst", "Investment Banking Analyst",
        "Portfolio Manager", "Risk Analyst", "Compliance Officer",
        "Accountant", "Senior Accountant", "Controller", "FP&A Manager",
    ],
    # --- Healthcare ---
    "healthcare": [
        "Registered Nurse", "Nurse Practitioner", "ICU Nurse",
        "Physician Assistant", "Medical Technologist", "Clinical Research Coordinator",
        "Pharmacist", "Physical Therapist", "Occupational Therapist",
    ],
    # --- Trades / Skilled Labor ---
    "trades": [
        "Electrician", "Master Electrician", "Plumber", "HVAC Technician",
        "Welder", "CNC Machinist", "Industrial Maintenance Technician",
        "Construction Project Manager", "Site Supervisor",
    ],
    # --- Sales & Marketing ---
    "sales": [
        "Account Executive", "Senior Account Executive", "Sales Development Representative",
        "Enterprise Sales Manager", "Customer Success Manager",
        "Solutions Engineer", "Sales Operations Manager",
    ],
    "marketing": [
        "Marketing Manager", "Digital Marketing Specialist", "Content Strategist",
        "SEO Specialist", "Growth Marketing Manager", "Brand Manager",
        "Social Media Manager",
    ],
    # --- Operations & Logistics ---
    "operations": [
        "Operations Manager", "Supply Chain Manager", "Logistics Coordinator",
        "Warehouse Manager", "Procurement Specialist", "Quality Assurance Manager",
    ],
    # --- Legal ---
    "legal": [
        "Corporate Counsel", "Paralegal", "Compliance Manager",
        "Contract Manager", "IP Attorney", "Legal Operations Manager",
    ],
    # --- HR ---
    "hr": [
        "HR Business Partner", "Talent Acquisition Specialist", "Recruiter",
        "Compensation & Benefits Analyst", "HR Manager", "People Operations Manager",
    ],
    # --- Education ---
    "education": [
        "Teacher", "Curriculum Designer", "Instructional Designer",
        "Academic Advisor", "Training Manager", "Corporate Trainer",
    ],
}

INDUSTRIES = [
    # Tech
    "Fintech", "E-commerce", "HealthTech", "EdTech", "SaaS / B2B",
    "Gaming", "Social Media", "Logistics / Supply Chain", "Cybersecurity",
    "AI / Machine Learning", "Climate Tech", "HR Tech",
    # Non-tech
    "Banking / Financial Services", "Insurance", "Investment Management",
    "Hospital / Health System", "Pharmaceutical", "Biotech",
    "Construction", "Manufacturing", "Energy / Utilities",
    "Retail", "Real Estate", "Legal Services",
    "Government / Public Sector", "Nonprofit",
    "Consulting", "Media / Entertainment", "Hospitality / Travel",
    "Telecommunications", "Agriculture / Food Tech",
]

SENIORITY_LEVELS = ["junior", "mid", "senior", "staff", "lead", "manager"]

COMPANY_SIZES = ["early-stage startup (20–50 people)", "Series B startup (100–300 people)",
                 "growth-stage company (500–1000 people)", "large tech company (1000+ people)"]

SKILL_POOLS = {
    # --- Tech ---
    "backend": ["Go", "Python", "Java", "Rust", "Node.js", "PostgreSQL", "MySQL",
                "Redis", "Kafka", "gRPC", "REST API", "Kubernetes", "Docker",
                "AWS", "GCP", "microservices", "distributed systems"],
    "frontend": ["React", "TypeScript", "Next.js", "Vue.js", "CSS", "Webpack",
                 "GraphQL", "REST API", "Jest", "Storybook", "Figma"],
    "data": ["Python", "SQL", "Spark", "Airflow", "dbt", "BigQuery", "Snowflake",
             "Pandas", "PyTorch", "TensorFlow", "scikit-learn", "Kafka", "Flink"],
    "infra": ["Kubernetes", "Terraform", "AWS", "GCP", "Azure", "Docker", "CI/CD",
              "Prometheus", "Grafana", "Linux", "Ansible", "Helm"],
    "product": ["product roadmap", "OKRs", "A/B testing", "user research",
                "SQL", "data analysis", "stakeholder management", "Agile"],
    "design": ["Figma", "user research", "prototyping", "design systems",
               "accessibility", "usability testing"],
    "mobile": ["Swift", "Kotlin", "React Native", "Flutter", "iOS SDK",
               "Android SDK", "REST API", "CI/CD", "App Store"],
    "management": ["team building", "technical roadmap", "cross-functional collaboration",
                   "Agile", "mentorship", "OKRs", "hiring"],
    # --- Finance ---
    "finance": ["financial modeling", "Excel", "Bloomberg Terminal", "SQL",
                "risk management", "derivatives", "GAAP", "IFRS", "valuation",
                "M&A", "budgeting", "forecasting", "SAP", "QuickBooks"],
    # --- Healthcare ---
    "healthcare": ["patient care", "EMR/EHR", "HIPAA compliance", "BLS/ACLS",
                   "medication administration", "clinical assessment", "triage",
                   "care coordination", "medical terminology", "IV therapy",
                   "wound care", "patient education"],
    # --- Trades ---
    "trades": ["NEC code", "blueprint reading", "electrical wiring", "PLC programming",
               "OSHA safety", "welding (MIG/TIG)", "CNC programming", "AutoCAD",
               "HVAC systems", "plumbing codes", "project estimation",
               "preventive maintenance", "lockout/tagout"],
    # --- Sales ---
    "sales": ["Salesforce", "CRM management", "pipeline management", "cold outreach",
              "consultative selling", "contract negotiation", "account management",
              "revenue forecasting", "SaaS sales", "enterprise sales cycle"],
    # --- Marketing ---
    "marketing": ["Google Analytics", "SEO/SEM", "content marketing", "email marketing",
                  "social media management", "HubSpot", "copywriting", "A/B testing",
                  "marketing automation", "brand strategy", "paid media"],
    # --- Operations ---
    "operations": ["supply chain management", "inventory management", "ERP systems",
                   "lean manufacturing", "Six Sigma", "vendor management",
                   "warehouse management", "logistics planning", "KPI tracking"],
    # --- Legal ---
    "legal": ["contract drafting", "legal research", "regulatory compliance",
              "corporate governance", "IP law", "litigation support",
              "due diligence", "Westlaw/LexisNexis", "risk assessment"],
    # --- HR ---
    "hr": ["talent acquisition", "HRIS", "employee relations", "compensation analysis",
            "performance management", "onboarding", "labor law", "diversity & inclusion",
            "benefits administration", "Workday", "ADP"],
    # --- Education ---
    "education": ["curriculum development", "instructional design", "LMS platforms",
                  "assessment design", "classroom management", "differentiated instruction",
                  "student engagement", "educational technology", "accreditation"],
}

SYSTEM_PROMPT = """You are an expert recruiter and technical writer.
Generate realistic, detailed job descriptions that read like real postings from tech companies.
Be specific about requirements, avoid generic filler. Vary the writing style between postings."""

JD_TEMPLATE = """Generate a job description with these characteristics:

Role: {role}
Industry: {industry}
Seniority: {seniority}
Company size: {company_size}
Key skills to emphasize (pick 4-6 of these, add 1-2 of your own): {skills}

Requirements:
- Length: 300–500 words
- Include: about the company (2-3 sentences), responsibilities (5-7 bullets), requirements (4-6 bullets), nice-to-haves (2-3 bullets)
- Make it specific and realistic — avoid generic phrases like "fast-paced environment"
- Write in English
- Do NOT include salary or location

Output the job description text only, no extra commentary."""


def build_prompt(role_category: str) -> dict:
    """Sample one concrete JD spec from the diversity matrix."""
    role = random.choice(ROLES[role_category])
    industry = random.choice(INDUSTRIES)
    company_size = random.choice(COMPANY_SIZES)
    skills = random.sample(SKILL_POOLS.get(role_category, SKILL_POOLS["backend"]), k=6)

    # Infer seniority from role title
    title_lower = role.lower()
    if any(x in title_lower for x in ["staff", "principal", "director"]):
        seniority = "staff"
    elif any(x in title_lower for x in ["senior", "sr."]):
        seniority = "senior"
    elif any(x in title_lower for x in ["manager", "lead", "tech lead"]):
        seniority = "manager"
    elif any(x in title_lower for x in ["junior", "jr.", "associate"]):
        seniority = "junior"
    else:
        seniority = "mid"

    return {
        "role": role,
        "industry": industry,
        "seniority": seniority,
        "company_size": company_size,
        "skills": ", ".join(skills),
        "role_category": role_category,
        "skills_focus": skills,
    }


def generate_jd(client: anthropic.Anthropic, spec: dict) -> str:
    """Call Claude to generate one JD."""
    prompt = JD_TEMPLATE.format(
        role=spec["role"],
        industry=spec["industry"],
        seniority=spec["seniority"],
        company_size=spec["company_size"],
        skills=spec["skills"],
    )

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic JDs via Claude")
    parser.add_argument("--num", type=int, default=200, help="Total JDs to generate")
    parser.add_argument(
        "--roles",
        type=str,
        default="backend,frontend,data,infra,product,mobile,finance,healthcare,trades,sales,marketing,operations,legal,hr,education",
        help="Comma-separated role categories to include",
    )
    parser.add_argument("--output", type=str, default="data/jds/", help="Output directory")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between API calls")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "all_jds.jsonl"

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set — add it to .env")
    client = anthropic.Anthropic(api_key=api_key)

    role_categories = [r.strip() for r in args.roles.split(",")]
    # Distribute evenly across role categories
    per_category = args.num // len(role_categories)
    specs = []
    for cat in role_categories:
        if cat not in ROLES:
            print(f"Unknown role category '{cat}', skipping")
            continue
        for _ in range(per_category):
            specs.append(build_prompt(cat))
    # Fill remainder
    while len(specs) < args.num:
        specs.append(build_prompt(random.choice(role_categories)))
    random.shuffle(specs)

    print(f"Generating {len(specs)} JDs → {output_dir}")
    generated = []
    failed = 0

    for i, spec in enumerate(specs):
        jd_id = f"jd_{i+1:04d}"
        jd_file = output_dir / f"{jd_id}.json"

        # Skip if already exists
        if jd_file.exists():
            with open(jd_file) as f:
                generated.append(json.load(f))
            print(f"[{i+1}/{len(specs)}] {jd_id} — skip (exists)")
            continue

        try:
            text = generate_jd(client, spec)
            record = {
                "id": jd_id,
                "role": spec["role"],
                "industry": spec["industry"],
                "seniority": spec["seniority"],
                "skills_focus": spec["skills_focus"],
                "text": text,
            }
            generated.append(record)

            # Write individual file
            with open(jd_file, "w") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)

            print(f"[{i+1}/{len(specs)}] {jd_id}: {spec['role']} @ {spec['industry']}")
            time.sleep(args.delay)

        except Exception as e:
            print(f"[{i+1}/{len(specs)}] {jd_id} FAILED: {e}")
            failed += 1
            time.sleep(2)  # back off on error

    # Write consolidated jsonl
    with open(jsonl_path, "w") as f:
        for record in generated:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nDone. {len(generated)} JDs saved to {output_dir}")
    print(f"Consolidated: {jsonl_path}")
    if failed:
        print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
