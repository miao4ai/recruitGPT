"""
All prompt templates for recruitGPT.

Two tasks:
  1. Query Parsing  — JD text → structured JSON (used in Stage ①)
  2. Match Explanation — JD + resume → human-readable report (used in Stage ⑤)

Each task has:
  - SYSTEM_PROMPT   : sets the role and output contract
  - build_*_prompt(): formats the user turn for a given input
  - OUTPUT_SCHEMA   : JSON schema comment for documentation / validation

These prompts are used by:
  - scripts/distill_data.py  (Teacher / Claude generates training data)
  - src/pipeline/query_parser.py  (Student / fine-tuned model at inference)
  - src/pipeline/explainer.py
"""

# ============================================================
# Task 1: Query Parsing
# ============================================================

QUERY_PARSER_SYSTEM = """\
You are a recruiting intelligence system. Your job is to read a job description \
and extract structured hiring intent as a JSON object.

Output ONLY valid JSON — no markdown fences, no explanation, no extra text.

JSON schema:
{
  "role_title": string,            // canonical role name, e.g. "Senior Backend Engineer"
  "seniority": string,             // one of: "intern", "junior", "mid", "senior", "staff", "lead", "manager", "director"
  "required_skills": [string],     // hard requirements — must-haves
  "nice_to_have_skills": [string], // optional / bonus skills
  "industry_context": string,      // domain the company operates in, e.g. "Fintech", "E-commerce"
  "hard_constraints": [string],    // non-negotiable requirements, e.g. "5+ years backend experience"
  "soft_signals": [string],        // culture / work style signals, e.g. "fast-paced", "ownership mindset"
  "team_context": string | null    // what team / product area this role sits in, if mentioned
}

Rules:
- required_skills: only list skills explicitly stated as required; do not infer
- seniority: infer from title and years-of-experience requirements
- hard_constraints: include years of experience, location, education if mentioned
- All string values in English
- If a field cannot be determined, use null for strings or [] for arrays\
"""

QUERY_PARSER_FEW_SHOT = [
    {
        "role": "user",
        "content": """\
Job Description:
---
We are looking for a Senior Backend Engineer to join our Payments team at FinFlow, \
a Series B fintech startup. You will design and own core payment processing services \
handling millions of transactions per day.

Responsibilities:
- Design and build high-throughput, low-latency payment APIs in Go
- Own the reliability and scalability of our transaction pipeline
- Collaborate with product and data teams to ship new payment features

Requirements:
- 5+ years of backend engineering experience
- Strong proficiency in Go or Rust
- Experience with PostgreSQL and Redis at scale
- Understanding of distributed systems and event-driven architecture (Kafka)

Nice to have:
- Prior experience in payments or financial services
- Familiarity with PCI-DSS compliance
- Experience with Kubernetes and GCP
---""",
    },
    {
        "role": "assistant",
        "content": """\
{
  "role_title": "Senior Backend Engineer",
  "seniority": "senior",
  "required_skills": ["Go", "PostgreSQL", "Redis", "distributed systems", "Kafka", "REST API"],
  "nice_to_have_skills": ["Rust", "Kubernetes", "GCP", "PCI-DSS", "payments domain"],
  "industry_context": "Fintech",
  "hard_constraints": ["5+ years backend engineering experience"],
  "soft_signals": ["ownership mindset", "high-throughput systems", "cross-functional collaboration"],
  "team_context": "Payments team — core payment processing services"
}\
""",
    },
]


def build_query_parser_prompt(jd_text: str) -> str:
    """Format the user turn for query parsing inference."""
    return f"Job Description:\n---\n{jd_text.strip()}\n---"


# ============================================================
# Task 2: Match Explanation
# ============================================================

MATCH_EXPLAINER_SYSTEM = """\
You are a senior technical recruiter writing concise candidate assessment notes \
for a hiring manager.

Given a job description and a candidate's resume, write a structured match report.

Output format (plain text, use the exact section headers below):

**Strengths**
- <bullet: specific skill or experience that directly matches a JD requirement>
- ...

**Gaps**
- <bullet: requirement in the JD that the candidate does not clearly demonstrate>
- ...

**Interview Focus**
- <bullet: specific area to probe in the interview to resolve uncertainty>
- ...

**Recommendation**
<One sentence: "Strong match", "Potential match — validate X", or "Likely mismatch — missing Y">

Rules:
- Be specific — reference actual skills, years, company names from the resume
- Strengths: only list things the JD explicitly requires and the resume clearly shows
- Gaps: only list hard requirements that are missing or unclear; do not penalise nice-to-haves
- Recommendation must be exactly one sentence\
"""

MATCH_EXPLAINER_FEW_SHOT = [
    {
        "role": "user",
        "content": """\
Job Description:
---
Senior Backend Engineer — Payments team. Requires 5+ years backend, Go or Rust, \
PostgreSQL/Redis at scale, Kafka. Fintech domain preferred.
---

Resume:
---
Zhang Wei | Backend Engineer
Experience:
- ByteDance (2020–2024): Built recommendation system backend in Go, handling 500K QPS. \
  Used Redis for caching, MySQL for storage. Led a team of 3.
- Meituan (2018–2020): Python microservices for logistics. Kafka-based event pipeline.
Skills: Go, Python, Redis, MySQL, Kafka, Docker, gRPC
Education: B.S. Computer Science, Tsinghua University
---""",
    },
    {
        "role": "assistant",
        "content": """\
**Strengths**
- 6 years backend experience (ByteDance + Meituan), exceeds 5-year requirement
- Go expertise at scale: 500K QPS recommendation system at ByteDance
- Kafka experience from Meituan logistics event pipeline
- Redis at scale: caching layer in high-throughput ByteDance system

**Gaps**
- No PostgreSQL experience listed — uses MySQL; may need ramp-up on Postgres-specific features
- No fintech or payments domain background

**Interview Focus**
- PostgreSQL vs MySQL: probe depth of relational DB knowledge and migration comfort
- Distributed transactions: how did they handle consistency in the payments-adjacent logistics work?
- Leadership: led a team of 3 at ByteDance — explore scope and impact

**Recommendation**
Potential match — validate PostgreSQL depth and comfort moving into a regulated fintech environment.\
""",
    },
]


def build_match_explainer_prompt(jd_text: str, resume_text: str) -> str:
    """Format the user turn for match explanation inference."""
    return (
        f"Job Description:\n---\n{jd_text.strip()}\n---\n\n"
        f"Resume:\n---\n{resume_text.strip()}\n---"
    )


# ============================================================
# Shared helpers
# ============================================================

# Training samples are stored as:
# {
#   "task": "query_parsing" | "match_explanation",
#   "messages": [
#     {"role": "system",    "content": <system prompt>},
#     {"role": "user",      "content": <user turn>},      ← few-shot pairs
#     {"role": "assistant", "content": <few-shot answer>},
#     ...
#     {"role": "user",      "content": <actual input>},
#     {"role": "assistant", "content": <teacher output>}  ← loss computed here
#   ]
# }

_PROMPTS = {
    "query_parsing": {
        "system": QUERY_PARSER_SYSTEM,
        "few_shot": QUERY_PARSER_FEW_SHOT,
    },
    "match_explanation": {
        "system": MATCH_EXPLAINER_SYSTEM,
        "few_shot": MATCH_EXPLAINER_FEW_SHOT,
    },
}


def format_training_sample(task: str, user_content: str, assistant_content: str) -> dict:
    """
    Wrap a (input, output) pair into the standard training sample format.
    Used by scripts/distill_data.py after Claude generates the output.
    """
    if task not in _PROMPTS:
        raise ValueError(f"Unknown task '{task}'. Choose from: {list(_PROMPTS)}")
    p = _PROMPTS[task]
    messages = [{"role": "system", "content": p["system"]}]
    messages.extend(p["few_shot"])
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": assistant_content})
    return {"task": task, "messages": messages}


def get_inference_messages(task: str, user_content: str) -> list[dict]:
    """
    Build the messages list for inference (no assistant turn).
    Used by query_parser.py and explainer.py at runtime.
    """
    if task not in _PROMPTS:
        raise ValueError(f"Unknown task '{task}'. Choose from: {list(_PROMPTS)}")
    p = _PROMPTS[task]
    messages = [{"role": "system", "content": p["system"]}]
    messages.extend(p["few_shot"])
    messages.append({"role": "user", "content": user_content})
    return messages
