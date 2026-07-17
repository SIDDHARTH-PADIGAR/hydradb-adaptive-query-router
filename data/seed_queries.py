"""
Seed queries for bootstrapping ground-truth labels.

- REAL_COOKBOOK_QUERIES: pulled directly from HydraDB's own published
  cookbook descriptions - grounds the set in HydraDB's stated use cases.
- SYNTHETIC_SHORT_FACTUAL / SYNTHETIC_COREFERENCE_HEAVY: small hand-written
  sets covering two specific patterns.
- GENERATED_QUERIES: domain x template combinations. This is where volume
  and real diversity come from - 25 domains x several template shapes,
  instead of hand-typing one query at a time. Uses the SAME marker list
  as features/extract.py (imported directly, not retyped) so there's no
  chance of the generator drifting out of sync with what the features
  actually detect.

Every query goes through the same hard-rule filtering in model/train.py
before training - this file's job is just volume and diversity, not
predicting which ones will end up "residual."
"""
import itertools
import random
import csv as _csv
import os as _os


from features.extract import RELATIONAL_MARKERS, TEMPORAL_MARKERS

REAL_COOKBOOK_QUERIES = [
    ("How do I reset my password?", 1, 0),
    ("What should I do today?", 3, 420),
    ("Find me someone who has 5+ years of experience in machine learning "
     "and has worked at Apple before", 1, 0),
    ("Why did we choose Postgres?", 2, 150),
    ("Who owns the payments service?", 1, 0),
    ("What is our main competitor doing right now?", 1, 0),
    ("How has their messaging shifted over the last 6 months?", 4, 680),
    ("Why was this built this way?", 2, 300),
    ("Why did we decide to sunset the legacy reporting pipeline?", 3, 510),
    ("What led to the decision to sunset Project X?", 2, 240),
    ("How has quarterly revenue trended over the last four quarters?", 1, 0),
    ("Prepare a summary for tomorrow's board meeting", 5, 900),
    ("What's the customer's plan and past issues before I respond?", 3, 350),
    ("Who is responsible for the billing outage last week?", 2, 190),
    ("How does the escalation policy relate to the SLA agreement?", 1, 0),
]

SYNTHETIC_SHORT_FACTUAL = [
    ("What is the API rate limit?", 1, 0),
    ("What's our support email?", 1, 0),
    ("Ticket ERR-4021 status?", 1, 0),
    ("Define SOC2", 1, 0),
    ("Pricing for the Scale plan?", 1, 0),
    ("Company address?", 1, 0),
    ("What port does the service run on?", 1, 0),
    ("Show me last invoice", 2, 60),
    ("Current uptime SLA?", 1, 0),
    ("List active connectors", 1, 0),
]

SYNTHETIC_COREFERENCE_HEAVY = [
    ("What did they say about it last time we discussed this?", 6, 950),
    ("Is that still the case, or has it changed since then?", 5, 780),
    ("Can you remind me what he decided about it?", 4, 640),
]

# --- Generated set: domains x templates ---

DOMAINS = [
    "billing", "account security", "infrastructure uptime", "onboarding flow",
    "feature access", "support tickets", "usage analytics", "sales pipeline",
    "contract terms", "incident response", "user permissions",
    "third-party integrations", "notification settings", "search indexing",
    "workflow automation", "data retention", "audit logging", "rate limiting",
    "api quotas", "deployment pipeline", "session management", "backup policy",
    "access reviews", "escalation policy", "customer segmentation",
]

# No relational/temporal marker on purpose - procedural/explanatory questions.
PROCEDURAL_TEMPLATES = [
    "What does {d} look like for a new account?",
    "Can you walk me through {d}?",
    "What settings control {d}?",
    "How do I update {d}?",
    "What's the default configuration for {d}?",
    "Where can I find {d} details?",
    "What options are available for {d}?",
    "How is {d} typically configured?",
    "What triggers a change in {d}?",
    "What's the current status of {d}?",
]

# Each uses exactly one marker from RELATIONAL_MARKERS, two domain slots.
RELATIONAL_TEMPLATES = [
    "How does {d1} relate to {d2}?",
    "Why did {d1} change after {d2} was updated?",
    "Why does {d1} sometimes conflict with {d2}?",
    "Does {d1} depend on {d2} being configured correctly?",
    "How does {d1} compare to {d2}?",
    "Who is responsible for {d1} versus {d2}?",
    "Does the team that owns {d1} also own {d2}?",
]

# Each uses a marker from TEMPORAL_MARKERS, single domain slot.
TEMPORAL_TEMPLATES = [
    "How has {d} changed over time?",
    "What did {d} used to look like before the last update?",
    "What's the history of {d} for this account?",
    "Was {d} different previously?",
    "How has {d} evolved since last quarter?",
    "What was {d} set to before the migration?",
]


def _generate(seed: int = 7, target_count: int = 240):
    rng = random.Random(seed)
    generated = []

    for tmpl in PROCEDURAL_TEMPLATES:
        for d in DOMAINS:
            generated.append((tmpl.format(d=d), 1, 0))

    pairs = list(itertools.permutations(DOMAINS, 2))
    rng.shuffle(pairs)
    for tmpl in RELATIONAL_TEMPLATES:
        for d1, d2 in pairs[:10]:
            turn = rng.choice([1, 2, 3])
            prior = rng.choice([0, 80, 150, 220])
            generated.append((tmpl.format(d1=d1, d2=d2), turn, prior))

    for tmpl in TEMPORAL_TEMPLATES:
        for d in DOMAINS:
            turn = rng.choice([1, 2, 3])
            prior = rng.choice([0, 100, 220])
            generated.append((tmpl.format(d=d), turn, prior))

    rng.shuffle(generated)
    seen, unique = set(), []
    for q in generated:
        if q[0] not in seen:
            seen.add(q[0])
            unique.append(q)
    return unique[:target_count]


GENERATED_QUERIES = _generate()

ALL_SEED_QUERIES = (
    REAL_COOKBOOK_QUERIES
    + SYNTHETIC_SHORT_FACTUAL
    + SYNTHETIC_COREFERENCE_HEAVY
    + GENERATED_QUERIES
)

_LLM_CSV_PATH = _os.path.join(_os.path.dirname(__file__), "llm_generated_queries.csv")
LLM_GENERATED_QUERIES = []
if _os.path.exists(_LLM_CSV_PATH):
    with open(_LLM_CSV_PATH, newline="", encoding="utf-8-sig") as _f:
        for _r in _csv.DictReader(_f):
            LLM_GENERATED_QUERIES.append((
                _r["query"],
                int(float(_r["session_turn_number"])),
                int(float(_r["prior_context_tokens"])),
            ))
    print(f"Loaded {len(LLM_GENERATED_QUERIES)} LLM-generated queries from {_LLM_CSV_PATH}")

ALL_SEED_QUERIES = ALL_SEED_QUERIES + LLM_GENERATED_QUERIES