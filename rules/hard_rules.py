"""
Hard rules catch the obvious cases so the classifier only has to earn its
keep on genuinely ambiguous queries - same pattern as the "catastrophic
query -> High Risk immediately" layer in the SQL admission-control project.

Each rule returns a full or partial decision override plus the reason it
fired, or None if it doesn't apply. Rules are checked in order; first match
short-circuits the classifier entirely for the fields it sets.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from features.extract import QueryFeatures


@dataclass
class RuleOverride:
    mode: Optional[str] = None
    graph_context: Optional[bool] = None
    query_by: Optional[str] = None
    alpha: Optional[float] = None
    reason: str = ""


def apply_hard_rules(query: str, feats: QueryFeatures) -> Optional[RuleOverride]:
    # 1. Literal token present (error code, SKU, ticket ID) -> exact match,
    #    no point paying for semantic/graph work.
    if feats.contains_literal_token:
        return RuleOverride(
            mode="fast",
            graph_context=False,
            query_by="text",
            reason="literal_token_exact_match",
        )

    # 2. Explicit multi-hop / relational language -> force thinking + graph.
    #    Their own docs: "graph_context: pair with mode=thinking - in fast
    #    mode the graph slice is shallow." No point asking for graph_context
    #    without paying for thinking mode to actually traverse it.
    if feats.relational_keyword_count >= 2 or (
        feats.relational_keyword_count >= 1 and feats.entity_count_estimate >= 1
    ):
        return RuleOverride(
            mode="thinking",
            graph_context=True,
            reason="strong_relational_signal",
        )

    # 3. Very short, no pronouns, no temporal/relational language -> almost
    #    certainly a direct factual lookup.
    if feats.token_count <= 6 and not feats.has_relational_keywords and not feats.has_temporal_keywords:
        return RuleOverride(
            mode="fast",
            graph_context=False,
            reason="short_direct_lookup",
        )

    # 4. High pronoun density with long prior context -> coreference-heavy,
    #    needs the graph to resolve "it"/"they" against session history.
    if feats.pronoun_density > 0.15 and feats.prior_context_tokens > 200:
        return RuleOverride(
            mode="thinking",
            graph_context=True,
            reason="coreference_heavy_needs_session_graph",
        )

    return None  # no rule fired - defer to the classifier
