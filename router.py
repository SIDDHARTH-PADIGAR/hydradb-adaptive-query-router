"""
The actual router: given a query, decides {mode, graph_context, alpha,
max_results} before any expensive call happens.

Decision order (mirrors the SQL admission-control project):
  1. Extract cheap features
  2. Hard rules - catch obvious cases, skip the model entirely
  3. Trained classifier - only runs when no rule fired
  4. SHAP explanation of whichever path was taken
  5. alpha / max_results are set by simple heuristics, not the classifier -
     these are lower-stakes knobs and don't need a learned model (see README
     for why this is a deliberate scope cut, not an oversight)
"""
from __future__ import annotations
import os
from dataclasses import dataclass, asdict
from typing import Optional

from features.extract import extract_features, QueryFeatures, FEATURE_COLUMNS
from rules.hard_rules import apply_hard_rules

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "router_model.json")

_booster = None
_explainer = None


def _load_model():
    """Lazy-load so the API can start even if the model hasn't been
    trained yet (falls back to rules + a conservative default)."""
    global _booster, _explainer
    if _booster is not None:
        return _booster
    try:
        import xgboost as xgb
        booster = xgb.Booster()
        booster.load_model(os.path.abspath(MODEL_PATH))
        _booster = booster
        try:
            import shap
            _explainer = shap.TreeExplainer(booster)
        except ImportError:
            _explainer = None
    except Exception:
        _booster = False  # sentinel: tried and failed, don't retry every call
    return _booster


@dataclass
class RoutingDecision:
    mode: str
    graph_context: bool
    alpha: float
    max_results: int
    confidence: Optional[float]
    rule_triggered: Optional[str]
    shap_top_features: list
    query_by: str = "hybrid"

    def to_dict(self):
        return asdict(self)


def _alpha_and_max_results(feats: QueryFeatures, mode: str) -> tuple[float, int]:
    """Simple heuristic knobs, per HydraDB's own documented tuning advice:
    lower alpha toward keyword-matching for literal tokens, higher for
    conceptual queries; more max_results when in thinking mode since
    downstream reranking benefits from a larger candidate pool."""
    alpha = 0.5 if feats.contains_literal_token else 0.8
    max_results = 20 if mode == "thinking" else 10
    return alpha, max_results


def decide(
    query: str,
    session_turn_number: int = 1,
    prior_context_tokens: int = 0,
) -> RoutingDecision:
    feats = extract_features(query, session_turn_number, prior_context_tokens)

    rule = apply_hard_rules(query, feats)
    if rule is not None:
        mode = rule.mode or "fast"
        graph_context = rule.graph_context if rule.graph_context is not None else False
        alpha, max_results = _alpha_and_max_results(feats, mode)
        if rule.alpha is not None:
            alpha = rule.alpha
        return RoutingDecision(
            mode=mode,
            graph_context=graph_context,
            alpha=alpha,
            max_results=max_results,
            confidence=1.0,
            rule_triggered=rule.reason,
            shap_top_features=[],
            query_by=rule.query_by or "hybrid",
        )

    booster = _load_model()
    if not booster:
        # No trained model available yet - conservative default: fast mode,
        # no graph. Never silently guess "thinking" (expensive) with no
        # signal to justify it.
        alpha, max_results = _alpha_and_max_results(feats, "fast")
        return RoutingDecision(
            mode="fast", graph_context=False, alpha=alpha, max_results=max_results,
            confidence=None, rule_triggered="no_model_conservative_default",
            shap_top_features=[],
        )

    import xgboost as xgb
    import numpy as np
    row = np.array([[getattr(feats, c) if not isinstance(getattr(feats, c), bool)
                      else float(getattr(feats, c)) for c in FEATURE_COLUMNS]])
    dmatrix = xgb.DMatrix(row, feature_names=FEATURE_COLUMNS)
    prob_thinking = float(booster.predict(dmatrix)[0])
    mode = "thinking" if prob_thinking > 0.5 else "fast"
    graph_context = mode == "thinking"

    shap_top = []
    if _explainer is not None:
        try:
            shap_values = _explainer.shap_values(row)
            pairs = sorted(
                zip(FEATURE_COLUMNS, shap_values[0].tolist()),
                key=lambda p: abs(p[1]), reverse=True,
            )
            shap_top = [{"feature": f, "impact": round(v, 4)} for f, v in pairs[:3]]
        except Exception:
            shap_top = []

    alpha, max_results = _alpha_and_max_results(feats, mode)
    return RoutingDecision(
        mode=mode,
        graph_context=graph_context,
        alpha=alpha,
        max_results=max_results,
        confidence=round(prob_thinking if mode == "thinking" else 1 - prob_thinking, 3),
        rule_triggered=None,
        shap_top_features=shap_top,
    )
