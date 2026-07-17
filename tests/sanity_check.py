"""
Sanity check with hand-picked queries where we know what SHOULD happen,
for two different reasons:

  RULE CASES: the hard rules in rules/hard_rules.py are plain if/else code,
  not ML. Their output is 100% deterministic and we can assert on it
  exactly - if these fail, something is broken in the code, not "the
  model needs more data."

  MODEL CASES: these fall through to the classifier, where we genuinely
  don't have ground truth (that's the whole point - if we knew the answer
  for certain, we wouldn't need a model). We can't assert these are
  "correct," only check whether the decision is defensible and look at
  the SHAP explanation for whether the reasoning makes sense.

  KEYWORD-BLIND CASES: the real test. Queries that are conceptually
  relational/temporal but deliberately avoid every word in
  RELATIONAL_MARKERS / TEMPORAL_MARKERS. If the model only ever predicts
  "fast" on these, that confirms it hasn't learned anything beyond the
  keyword list both mock.py and features/extract.py already encode -
  which is the honest limitation of training on mock-bootstrapped data.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from router import decide

print("=" * 70)
print("RULE CASES - deterministic, must match exactly")
print("=" * 70)

rule_cases = [
    # (query, prior_context_tokens, expected_mode, expected_rule)
    ("What's the status of ticket ERR-4021?", 0,
     "fast", "literal_token_exact_match"),
    ("What's our support email?", 0,
     "fast", "short_direct_lookup"),
    ("How does the retry policy relate to timeout settings, and why "
     "does it depend on the connection pool configuration?", 0,
     "thinking", "strong_relational_signal"),
]

rule_failures = 0
for query, prior_tokens, expected_mode, expected_rule in rule_cases:
    d = decide(query, session_turn_number=1, prior_context_tokens=prior_tokens)
    ok = d.mode == expected_mode and d.rule_triggered == expected_rule
    status = "PASS" if ok else "FAIL"
    if not ok:
        rule_failures += 1
    print(f"[{status}] \"{query[:60]}...\"" if len(query) > 60 else f"[{status}] \"{query}\"")
    print(f"       expected: mode={expected_mode}, rule={expected_rule}")
    print(f"       got:      mode={d.mode}, rule={d.rule_triggered}, graph_context={d.graph_context}")

# coreference case needs prior_context_tokens > 200 to trip, pass separately
d = decide("What did they say about it, and is that still true?",
           session_turn_number=6, prior_context_tokens=300)
ok = d.mode == "thinking" and d.rule_triggered == "coreference_heavy_needs_session_graph"
if not ok:
    rule_failures += 1
print(f"[{'PASS' if ok else 'FAIL'}] \"What did they say about it, and is that still true?\" (prior=300)")
print(f"       expected: mode=thinking, rule=coreference_heavy_needs_session_graph")
print(f"       got:      mode={d.mode}, rule={d.rule_triggered}")

print()
print("=" * 70)
print("KEYWORD-BLIND CASES - conceptually relational, no marker words used")
print("=" * 70)
print("No 'expected' here - this is checking whether the model learned")
print("anything beyond the keyword list, or is blind without it.\n")

keyword_blind_cases = [
    "What connects the retry logic to the timeout configuration?",
    "Is there a link between deploy frequency and incident count?",
    "What's tying the billing spike to the new pricing rollout?",
    "Something changed about the access policy since last month - what?",
]

for query in keyword_blind_cases:
    d = decide(query, session_turn_number=1, prior_context_tokens=0)
    print(f"\"{query}\"")
    print(f"  -> mode={d.mode}, confidence={d.confidence}, rule={d.rule_triggered}")
    if d.shap_top_features:
        print(f"     top features: {d.shap_top_features}")
    print()

print("=" * 70)
print(f"RULE CASES: {len(rule_cases) + 1 - rule_failures}/{len(rule_cases) + 1} passed")
print("If any rule case failed, that's a real bug - check rules/hard_rules.py.")
print("If every keyword-blind case came back 'fast', that confirms the model")
print("is keyword-matching, not reasoning - expected given how mock labels")
print("are generated, and the reason real API data matters for the real thing.")
print("=" * 70)