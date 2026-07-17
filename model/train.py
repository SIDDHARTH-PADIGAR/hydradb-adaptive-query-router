"""
Trains a classifier to predict {mode, graph_context} from pre-execution
query features, using the bootstrapped labels from eval/bootstrap.py.

Run:
    python -m eval.bootstrap --client mock   # produces data/bootstrapped_labels.csv
    python -m model.train                    # trains + saves model/router_model.json
"""
from __future__ import annotations
import csv
import json
import os

import numpy as np

from features.extract import QueryFeatures, FEATURE_COLUMNS
from rules.hard_rules import apply_hard_rules

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "bootstrapped_labels.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "router_model.json")
META_PATH = os.path.join(os.path.dirname(__file__), "router_model_meta.json")

def _to_float(v: str):
    if v in ("True", "False"):
        return 1.0 if v == "True" else 0.0
    return float(v)


def _row_to_features(r: dict) -> QueryFeatures:
    """Rebuild a QueryFeatures object from a CSV row so we can re-run the
    same hard rules the live router would apply, and filter to only the
    queries the classifier actually gets asked to decide."""
    return QueryFeatures(
        token_count=int(float(r["token_count"])),
        has_relational_keywords=r["has_relational_keywords"] == "True",
        relational_keyword_count=int(float(r["relational_keyword_count"])),
        has_temporal_keywords=r["has_temporal_keywords"] == "True",
        temporal_keyword_count=int(float(r["temporal_keyword_count"])),
        entity_count_estimate=int(float(r["entity_count_estimate"])),
        contains_literal_token=r["contains_literal_token"] == "True",
        pronoun_density=float(r["pronoun_density"]),
        session_turn_number=int(float(r["session_turn_number"])),
        prior_context_tokens=int(float(r["prior_context_tokens"])),
        is_question=r["is_question"] == "True",
        relational_semantic_score=float(r["relational_semantic_score"]),
        factual_semantic_score=float(r["factual_semantic_score"]),
    )


def load_dataset():
    """Loads bootstrapped labels, then DROPS any row a hard rule would
    already resolve. The classifier should only ever be trained and graded
    on the population it actually decides for in production - the
    ambiguous residual, not the easy cases the rules already catch."""
    with open(os.path.abspath(DATA_PATH)) as f:
        all_rows = list(csv.DictReader(f))

    # Override mock's keyword-derived labels with hand labels where they
    # exist - hand labels reflect real judgment about whether a query
    # needs relational reasoning, not just whether it contains a keyword.
    hand_labels_path = os.path.join(os.path.dirname(DATA_PATH), "hand_labels.csv")
    if os.path.exists(os.path.abspath(hand_labels_path)):
        with open(os.path.abspath(hand_labels_path)) as f:
            overrides = {r["query"]: r["label_mode"] for r in csv.DictReader(f)}
        n_overridden = 0
        for r in all_rows:
            if r["query"] in overrides and overrides[r["query"]] != r["label_mode"]:
                n_overridden += 1
            if r["query"] in overrides:
                r["label_mode"] = overrides[r["query"]]
        print(f"Applied {len(overrides)} hand labels "
              f"({n_overridden} disagreed with the mock's keyword-based label).")

    # Same mechanism, second source: LLM-generated labels. GPT's judgment
    # isn't derived from RELATIONAL_MARKERS, so it doesn't carry the same
    # keyword-matching bias the mock's own labels do.
    llm_labels_path = os.path.join(os.path.dirname(DATA_PATH), "llm_generated_queries.csv")
    if os.path.exists(os.path.abspath(llm_labels_path)):
        with open(os.path.abspath(llm_labels_path), encoding="utf-8-sig") as f:
            llm_overrides = {r["query"]: r["label_mode"] for r in csv.DictReader(f)}
        n_llm_overridden = 0
        for r in all_rows:
            if r["query"] in llm_overrides and llm_overrides[r["query"]] != r["label_mode"]:
                n_llm_overridden += 1
            if r["query"] in llm_overrides:
                r["label_mode"] = llm_overrides[r["query"]]
        print(f"Applied {len(llm_overrides)} LLM-generated labels "
              f"({n_llm_overridden} disagreed with the mock's keyword-based label).")

    residual_rows = []
    dropped = 0
    for r in all_rows:
        feats = _row_to_features(r)
        if apply_hard_rules(r["query"], feats) is not None:
            dropped += 1
            continue
        residual_rows.append(r)

    print(f"Loaded {len(all_rows)} total rows. "
          f"{dropped} were already resolved by hard rules and dropped. "
          f"Training/evaluating on {len(residual_rows)} residual (ambiguous) rows - "
          f"that's the actual population the classifier operates on in production.")

    X = np.array([[_to_float(r[c]) for c in FEATURE_COLUMNS] for r in residual_rows])
    y = np.array([1 if r["label_mode"] == "thinking" else 0 for r in residual_rows])
    return X, y, residual_rows


def _class_weight(y: np.ndarray) -> float:
    pos = max(int(y.sum()), 1)
    neg = max(len(y) - pos, 1)
    return neg / pos


def _stratified_kfold_indices(y: np.ndarray, k: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    class0_idx = np.where(y == 0)[0]
    class1_idx = np.where(y == 1)[0]
    rng.shuffle(class0_idx)
    rng.shuffle(class1_idx)
    folds0 = np.array_split(class0_idx, k) if len(class0_idx) >= k else [class0_idx]
    folds1 = np.array_split(class1_idx, k) if len(class1_idx) >= k else [class1_idx]
    n_folds = min(len(folds0), len(folds1)) if folds0 and folds1 else 1
    for i in range(n_folds):
        test_idx = np.concatenate([folds0[i], folds1[i]])
        train_idx = np.setdiff1d(np.arange(len(y)), test_idx)
        if len(train_idx) > 0 and len(test_idx) > 0:
            yield train_idx, test_idx


def train():
    if not HAS_XGB:
        raise SystemExit(
            "xgboost not installed. Run: pip install xgboost shap --break-system-packages"
        )

    X, y, rows = load_dataset()

    if len(y) < 10:
        raise SystemExit(
            f"Only {len(y)} residual (non-hard-rule) examples after filtering - "
            f"too few to train or evaluate anything meaningful."
        )
    if len(set(y.tolist())) < 2:
        raise SystemExit(
            "All residual examples landed on the same label after hard-rule "
            "filtering - there's nothing for the classifier to learn to distinguish."
        )

    pos_weight = _class_weight(y)
    params = {
        "objective": "binary:logistic", "max_depth": 3, "eta": 0.3,
        "eval_metric": "logloss", "scale_pos_weight": pos_weight,
    }

    k = min(5, int(y.sum()), int((len(y) - y.sum())))
    k = max(k, 2)
    fold_accuracies = []
    for train_idx, test_idx in _stratified_kfold_indices(y, k):
        dtr = xgb.DMatrix(X[train_idx], label=y[train_idx], feature_names=FEATURE_COLUMNS)
        dte = xgb.DMatrix(X[test_idx], feature_names=FEATURE_COLUMNS)
        fold_model = xgb.train(params, dtr, num_boost_round=40)
        preds = (fold_model.predict(dte) > 0.5).astype(int)
        fold_accuracies.append(float((preds == y[test_idx]).mean()))

    cv_mean = float(np.mean(fold_accuracies)) if fold_accuracies else None
    cv_std = float(np.std(fold_accuracies)) if fold_accuracies else None

    dall = xgb.DMatrix(X, label=y, feature_names=FEATURE_COLUMNS)
    booster = xgb.train(params, dall, num_boost_round=40)
    booster.save_model(os.path.abspath(MODEL_PATH))

    meta = {
        "feature_columns": FEATURE_COLUMNS,
        "n_residual_examples": len(y),
        "class_balance": {"fast": int((y == 0).sum()), "thinking": int((y == 1).sum())},
        "scale_pos_weight_used": pos_weight,
        "cv_folds": k,
        "cv_accuracy_mean": cv_mean,
        "cv_accuracy_std": cv_std,
        "cv_accuracy_per_fold": fold_accuracies,
        "label_positive_class": "thinking",
    }
    with open(os.path.abspath(META_PATH), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Trained final model on all {len(y)} residual examples "
          f"({meta['class_balance']}).")
    print(f"Cross-validated accuracy ({k}-fold): {cv_mean:.3f} +/- {cv_std:.3f} "
          f"(per fold: {[round(a, 2) for a in fold_accuracies]})")
    print(f"Model saved to {MODEL_PATH}")
    return booster, meta


if __name__ == "__main__":
    train()