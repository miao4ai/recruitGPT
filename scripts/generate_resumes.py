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
    # --- Tech ---
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
    # --- Finance ---
    "finance": [
        "Financial Analyst", "Senior Financial Analyst", "Investment Analyst",
        "Portfolio Manager", "Risk Analyst", "Accountant", "Senior Accountant",
        "FP&A Manager", "Controller",
    ],
    # --- Healthcare ---
    "healthcare": [
        "Registered Nurse", "Nurse Practitioner", "ICU Nurse",
        "Physician Assistant", "Pharmacist", "Physical Therapist",
        "Medical Technologist", "Clinical Research Coordinator",
    ],
    # --- Trades ---
    "trades": [
        "Electrician", "Master Electrician", "Plumber", "HVAC Technician",
        "Welder", "CNC Machinist", "Maintenance Technician",
        "Construction Project Manager", "Site Supervisor",
    ],
    # --- Sales ---
    "sales": [
        "Account Executive", "Senior Account Executive", "Sales Development Rep",
        "Enterprise Sales Manager", "Customer Success Manager", "Solutions Engineer",
    ],
    # --- Marketing ---
    "marketing": [
        "Marketing Manager", "Digital Marketing Specialist", "Content Strategist",
        "SEO Specialist", "Growth Marketing Manager", "Brand Manager",
    ],
    # --- Operations ---
    "operations": [
        "Operations Manager", "Supply Chain Manager", "Logistics Coordinator",
        "Warehouse Manager", "Procurement Specialist",
    ],
    # --- Legal ---
    "legal": [
        "Corporate Counsel", "Paralegal", "Compliance Manager",
        "Contract Manager", "Legal Operations Manager",
    ],
    # --- HR ---
    "hr": [
        "HR Business Partner", "Talent Acquisition Specialist", "Recruiter",
        "HR Manager", "People Operations Manager",
    ],
    # --- Education ---
    "education": [
        "Teacher", "Curriculum Designer", "Instructional Designer",
        "Training Manager", "Corporate Trainer",
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

UNIVERSITIES = [
    "MIT", "Stanford University", "Carnegie Mellon University",
    "UC Berkeley", "University of Waterloo", "ETH Zurich",
    "Tsinghua University", "Peking University", "NUS",
    "University of Michigan", "Georgia Tech", "Columbia University",
]

DEGREES = {
    "default": ["B.S. Computer Science", "B.Eng. Software Engineering",
                "M.S. Computer Science", "B.S. Mathematics", "M.S. Data Science"],
    "finance": ["B.S. Finance", "B.S. Economics", "MBA", "M.S. Financial Engineering",
                "B.S. Accounting", "CFA Charterholder"],
    "healthcare": ["B.S.N. Nursing", "M.S.N. Nursing", "Doctor of Nursing Practice",
                   "Pharm.D.", "B.S. Biology", "DPT Physical Therapy"],
    "trades": ["Journeyman Electrician License", "Associate Degree — Electrical Technology",
               "HVAC Certification", "Welding Certification", "B.S. Construction Management"],
    "sales": ["B.A. Business Administration", "B.S. Marketing", "MBA"],
    "marketing": ["B.A. Marketing", "B.A. Communications", "M.S. Digital Marketing", "MBA"],
    "operations": ["B.S. Supply Chain Management", "B.S. Industrial Engineering",
                   "MBA — Operations", "Six Sigma Black Belt"],
    "legal": ["J.D.", "B.A. Political Science", "LL.M.", "Paralegal Certificate"],
    "hr": ["B.A. Human Resources", "B.S. Psychology", "MBA — HR Management",
            "SHRM-CP Certification"],
    "education": ["B.A. Education", "M.Ed.", "Ed.D.", "Teaching Credential",
                  "M.A. Curriculum & Instruction"],
}

NAMES = [
    "Alex Chen", "Jamie Liu", "Sam Park", "Jordan Wang", "Taylor Zhang",
    "Morgan Li", "Casey Kim", "Riley Zhao", "Drew Wu", "Avery Huang",
    "Blake Ng", "Cameron Sun", "Dana Zhou", "Evan Lim", "Fiona Tan",
    "Grace Xu", "Henry Ma", "Iris Guo", "Jason Fu", "Karen Ye",
    "Leo Han", "Mia Song", "Nathan Bai", "Olivia Dai", "Peter Lu",
    "Quinn Shi", "Rachel Jin", "Steven Feng", "Tina Jiang", "Victor Wei",
]

SYSTEM_PROMPT = """\
You are generating realistic synthetic LinkedIn profiles for professionals across all industries.
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
    # --- Tech ---
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
    # --- Finance ---
    "finance": ["financial modeling", "Excel", "Bloomberg Terminal", "SQL",
                "risk management", "GAAP", "IFRS", "valuation", "budgeting",
                "forecasting", "SAP", "M&A"],
    # --- Healthcare ---
    "healthcare": ["patient care", "EMR/EHR", "HIPAA", "BLS/ACLS",
                   "medication administration", "clinical assessment", "triage",
                   "care coordination", "IV therapy", "wound care"],
    # --- Trades ---
    "trades": ["NEC code", "blueprint reading", "electrical wiring", "PLC programming",
               "OSHA safety", "welding (MIG/TIG)", "CNC programming", "AutoCAD",
               "HVAC systems", "plumbing codes", "preventive maintenance"],
    # --- Sales ---
    "sales": ["Salesforce", "CRM", "pipeline management", "cold outreach",
              "consultative selling", "contract negotiation", "account management",
              "revenue forecasting", "SaaS sales"],
    # --- Marketing ---
    "marketing": ["Google Analytics", "SEO/SEM", "content marketing", "HubSpot",
                  "copywriting", "A/B testing", "marketing automation",
                  "social media management", "paid media"],
    # --- Operations ---
    "operations": ["supply chain management", "ERP systems", "lean manufacturing",
                   "Six Sigma", "vendor management", "logistics planning",
                   "inventory management", "KPI tracking"],
    # --- Legal ---
    "legal": ["contract drafting", "legal research", "regulatory compliance",
              "corporate governance", "litigation support", "due diligence",
              "Westlaw/LexisNexis", "risk assessment"],
    # --- HR ---
    "hr": ["talent acquisition", "HRIS", "employee relations", "compensation analysis",
            "performance management", "labor law", "Workday", "ADP",
            "diversity & inclusion", "onboarding"],
    # --- Education ---
    "education": ["curriculum development", "instructional design", "LMS platforms",
                  "assessment design", "classroom management", "educational technology",
                  "student engagement", "differentiated instruction"],
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

    # Pick 2 previous companies — match industry to role category
    CATEGORY_COMPANY_POOLS = {
        "finance": ["finance", "general"],
        "healthcare": ["healthcare", "general"],
        "trades": ["trades", "general"],
        "sales": ["general", "tier2_tech", "fintech"],
        "marketing": ["general", "tier2_tech"],
        "operations": ["general", "trades"],
        "legal": ["finance", "general"],
        "hr": ["general", "tier1_tech", "tier2_tech"],
        "education": ["general"],
    }
    pools = CATEGORY_COMPANY_POOLS.get(
        role_category, ["tier1_tech", "tier2_tech", "fintech", "startup"]
    )
    all_companies = []
    for pool in pools:
        all_companies.extend(COMPANIES.get(pool, []))
    companies = random.sample(all_companies, k=min(2, len(all_companies)))

    skills = random.sample(SKILL_POOLS.get(role_category, SKILL_POOLS["backend"]), k=6)

    return {
        "name": random.choice(NAMES),
        "current_title": title,
        "years_exp": years_exp,
        "companies": " and ".join(companies),
        "university": random.choice(UNIVERSITIES),
        "degree": random.choice(DEGREES.get(role_category, DEGREES["default"])),
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
