"""
Cheap, pre-execution feature extraction. Everything here must be fast -
the whole premise of admission control is that deciding costs far less
than executing. Semantic similarity (below) adds a few milliseconds,
which is still orders of magnitude cheaper than the 90ms-5s a real
HydraDB call costs, so it stays within budget - just no longer
microsecond-cheap like the pure regex checks.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, asdict

from sentence_transformers import SentenceTransformer
import numpy as np

RELATIONAL_MARKERS = (
    "relate", "relationship", "own", "owns", "depend", "why did", "why does",
    "who is responsible", "who owns", "how does", "compare", "versus", " vs ",
)
TEMPORAL_MARKERS = (
    "used to", "history of", "over time", "changed", "previously",
    "last time", "before", "since", "as of", "used to be", "evolved",
)
LITERAL_TOKEN_PATTERN = re.compile(
    r"\b([A-Z0-9]{2,}-\d+|[A-Z]{2,}\d{2,}|#\d+|\berr[_-]?\d+\b)\b", re.IGNORECASE
)
PRONOUN_PATTERN = re.compile(r"\b(he|she|they|it|his|her|their|this|that)\b", re.IGNORECASE)

# Small, hand-picked anchor sets for semantic similarity. These capture
# the CONCEPT of relational/causal/comparative intent vs. plain factual
# lookup, without requiring an exact keyword match - "what connects X to
# Y" scores similar to these anchors even though "connects" isn't in
# RELATIONAL_MARKERS. This is what a pure keyword list structurally can't
# do, no matter how much more data trains against it.
RELATIONAL_ANCHORS = [
    "how does this relate to that",
    "what connects these two things",
    "why does one depend on the other",
    "is there a link between these",
    "what's tying this to that",
    "how do these compare",
    "who is responsible for this versus that",
    "what caused this change",
    "why did this happen because of that",
    "what's the connection between these",
    "something broke after a recent change and I need to figure out what caused it",
    "two things happened around the same time and I want to know if they're connected",
    "this started failing right after another part of the system changed",
    "I need the full picture of everything affected before this gets resolved",
    "one thing worked before and now it doesn't, after something else changed",
    "this looks like it cascaded from an earlier event elsewhere",
    "I need to trace back what led to the current state",
    "one person could still do this while another person could not, under the same conditions",
]
FACTUAL_ANCHORS = [
    "what is the value of this setting",
    "what is the current status",
    "where can I find this",
    "what is the definition of this term",
    "show me the current configuration",
    "what is the price of this plan",
    "what port does this run on",
    "what is the support email",
    "I want to check what my current plan includes",
    "can you tell me the exact date my subscription renews",
    "I need to see the full list of active users on my account",
    "what is the exact error message shown on this failed request",
    "I'd like to confirm the current status of my support ticket",
    "I just need to look up a single value on my account",
]

# Real pretrained sentence embeddings, not TF-IDF - this is what actually
# understands "connects" and "relate" refer to the same concept. Loaded
# once at import time (~1-2s), reused for every call after that. First
# run on this machine will download the model (~90MB) from HuggingFace.
_ANCHOR_TEXTS = RELATIONAL_ANCHORS + FACTUAL_ANCHORS
_N_RELATIONAL_ANCHORS = len(RELATIONAL_ANCHORS)
_model = SentenceTransformer("all-MiniLM-L6-v2")
_anchor_embeddings = _model.encode(_ANCHOR_TEXTS, normalize_embeddings=True)

# Single source of truth for feature order - router.py and model/train.py
# both import this instead of retyping their own copy, so they can't
# silently drift out of sync with each other again.
FEATURE_COLUMNS = [
    "token_count", "has_relational_keywords", "relational_keyword_count",
    "has_temporal_keywords", "temporal_keyword_count", "entity_count_estimate",
    "contains_literal_token", "pronoun_density", "session_turn_number",
    "prior_context_tokens", "is_question",
    "relational_semantic_score", "factual_semantic_score",
]


@dataclass
class QueryFeatures:
    token_count: int
    has_relational_keywords: bool
    relational_keyword_count: int
    has_temporal_keywords: bool
    temporal_keyword_count: int
    entity_count_estimate: int
    contains_literal_token: bool
    pronoun_density: float
    session_turn_number: int
    prior_context_tokens: int
    is_question: bool
    relational_semantic_score: float
    factual_semantic_score: float

    def to_dict(self) -> dict:
        return asdict(self)


def _estimate_entity_count(query: str) -> int:
    words = query.split()
    caps = 0
    for i, w in enumerate(words):
        clean = w.strip(".,?!\"'")
        if i > 0 and clean[:1].isupper() and clean.lower() not in ("i",):
            caps += 1
    quoted = len(re.findall(r'"[^"]+"|\'[^\']+\'', query))
    return caps + quoted


def _semantic_scores(query: str) -> tuple[float, float]:
    """Max cosine similarity to the relational anchor set and the factual
    anchor set, using real sentence embeddings - catches paraphrases a
    keyword list or TF-IDF can't, because it's based on meaning, not
    shared vocabulary. Costs a few milliseconds, still trivially cheap
    next to the 90ms-5s HydraDB call it's deciding whether to trigger."""
    q_vec = _model.encode([query], normalize_embeddings=True)[0]
    sims = _anchor_embeddings @ q_vec
    relational_score = float(sims[:_N_RELATIONAL_ANCHORS].max())
    factual_score = float(sims[_N_RELATIONAL_ANCHORS:].max())
    return round(relational_score, 4), round(factual_score, 4)


def extract_features(
    query: str,
    session_turn_number: int = 1,
    prior_context_tokens: int = 0,
) -> QueryFeatures:
    q_lower = query.lower()
    tokens = query.split()

    relational_hits = sum(1 for m in RELATIONAL_MARKERS if m in q_lower)
    temporal_hits = sum(1 for m in TEMPORAL_MARKERS if m in q_lower)
    pronouns = len(PRONOUN_PATTERN.findall(query))
    relational_score, factual_score = _semantic_scores(query)

    return QueryFeatures(
        token_count=len(tokens),
        has_relational_keywords=relational_hits > 0,
        relational_keyword_count=relational_hits,
        has_temporal_keywords=temporal_hits > 0,
        temporal_keyword_count=temporal_hits,
        entity_count_estimate=_estimate_entity_count(query),
        contains_literal_token=bool(LITERAL_TOKEN_PATTERN.search(query)),
        pronoun_density=round(pronouns / max(len(tokens), 1), 3),
        session_turn_number=session_turn_number,
        prior_context_tokens=prior_context_tokens,
        is_question="?" in query or q_lower.startswith(
            ("what", "who", "why", "how", "when", "where", "which", "does", "is", "are")
        ),
        relational_semantic_score=relational_score,
        factual_semantic_score=factual_score,
    )