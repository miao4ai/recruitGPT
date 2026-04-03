"""
Knowledge-graph schema constants for Stage 4 of the recruitGPT pipeline.

Defines node types, edge types, seniority progression, and company tier
mappings used by builder.py and boost.py.
"""

# ---------------------------------------------------------------------------
# Node types
# ---------------------------------------------------------------------------
SKILL = "SKILL"
ROLE = "ROLE"
COMPANY = "COMPANY"
INDUSTRY = "INDUSTRY"
SENIORITY = "SENIORITY"

NODE_TYPES = {SKILL, ROLE, COMPANY, INDUSTRY, SENIORITY}

# ---------------------------------------------------------------------------
# Edge types
# ---------------------------------------------------------------------------
RELATED_TO = "RELATED_TO"   # skill <-> skill  (co-occur in same JD)
USED_IN = "USED_IN"         # skill  -> role
IN_TIER = "IN_TIER"         # company -> tier label
NEXT_STEP = "NEXT_STEP"     # seniority(role) -> seniority(role)
HIRES = "HIRES"             # company -> role

EDGE_TYPES = {RELATED_TO, USED_IN, IN_TIER, NEXT_STEP, HIRES}

# ---------------------------------------------------------------------------
# Seniority progression  (intern -> ... -> director)
# ---------------------------------------------------------------------------
SENIORITY_ORDER = [
    "intern",
    "junior",
    "mid",
    "senior",
    "staff",
    "lead",
    "manager",
    "director",
]

SENIORITY_RANK = {level: i for i, level in enumerate(SENIORITY_ORDER)}

# ---------------------------------------------------------------------------
# Company tiers  (mirrored from scripts/generate_resumes.py)
# ---------------------------------------------------------------------------
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
    "finance": [
        "Goldman Sachs", "JP Morgan", "Morgan Stanley", "BlackRock",
        "Deloitte", "PwC", "EY", "KPMG", "Citadel", "Bridgewater",
    ],
    "healthcare": [
        "Mayo Clinic", "Kaiser Permanente", "Cleveland Clinic",
        "Johns Hopkins Hospital", "HCA Healthcare", "Pfizer",
        "Johnson & Johnson",
    ],
    "trades": [
        "Turner Construction", "Bechtel", "Fluor Corporation",
        "AECOM", "a regional electrical contractor",
        "a commercial HVAC company",
    ],
    "general": [
        "Walmart", "Target", "Costco", "McKinsey", "BCG",
        "Salesforce", "HubSpot", "Nike", "Disney", "Marriott",
    ],
}

# Reverse lookup: company name -> tier
COMPANY_TO_TIER: dict[str, str] = {}
for _tier, _names in COMPANIES.items():
    for _name in _names:
        COMPANY_TO_TIER[_name.lower()] = _tier

# Tier prestige ordering (higher index = more prestigious for tech roles)
TIER_RANK = {
    "trades": 0,
    "healthcare": 1,
    "general": 2,
    "finance": 3,
    "fintech": 4,
    "tier2_tech": 5,
    "tier1_tech": 6,
}
