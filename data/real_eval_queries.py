"""
Queries matched to everything actually ingested into hydra_router_poc:
sample_knowledge.txt (payments/billing), security_incident.txt,
sales_contract_scenario.txt.
"""

REAL_EVAL_QUERIES = [
    # --- fast: direct lookups, one doc each ---
    ("What plan includes dedicated infrastructure?", 1, 0),
    ("Does the scale plan include priority support?", 1, 0),
    ("What database does the payments service use?", 1, 0),
    ("What access level do new engineers get by default?", 1, 0),
    ("When was the enterprise contract for Acme Corp signed?", 1, 0),
    ("How many support tickets came from the June incident?", 1, 0),
    ("How often does the certificate rotation policy run?", 1, 0),
    ("When is Acme Corp's contract renewal date?", 1, 0),
    ("What percentage of users were affected by the login issue?", 1, 0),
    ("Does the standard enterprise contract include custom clauses?", 1, 0),

    # --- thinking: genuinely relational, causal, cross-entity ---
    ("Who owns the payments service?", 1, 0),
    ("What does the payments service depend on?", 1, 0),
    ("Why did the payments incident happen last quarter?", 1, 0),
    ("What caused the platform team to migrate the database?", 1, 0),
    ("Who needs to approve access to the payments service?", 1, 0),
    ("What caused the authentication service to reject valid tokens?", 1, 0),
    ("Why wasn't the certificate rotation issue caught earlier?", 1, 0),
    ("Which teams were involved in resolving the June incident?", 2, 90),
    ("Why were enterprise customers notified but not free tier customers?", 1, 0),
    ("What changes were made to the rotation process after the incident?", 2, 120),
    ("Why did Acme Corp's overage take two weeks to resolve?", 1, 0),
    ("What's different about Acme Corp's contract versus a standard one?", 1, 0),
    ("Why did legal get involved in Acme Corp's overage?", 1, 0),
    ("What should change in Acme Corp's renewal based on this issue?", 2, 100),
    ("How does the custom SLA clause relate to the overage policy conflict?", 1, 0),
]