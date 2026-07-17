"""
Bootstraps ground truth for training the router.

Run:
    python -m eval.bootstrap --client mock
    python -m eval.bootstrap --client real --db-name hydra_router_poc --limit 15
    python -m eval.bootstrap --client real --db-name hydra_router_poc
"""
from __future__ import annotations
from data.real_eval_queries import REAL_EVAL_QUERIES
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.seed_queries import ALL_SEED_QUERIES
from features.extract import extract_features
from hydra_client.mock import MockHydraClient

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "bootstrapped_labels.csv")


def build_client(kind: str, database: str | None):
    if kind == "mock":
        return MockHydraClient(seed=42), "mock_db"
    elif kind == "real":
        from hydra_client.real import RealHydraClient
        api_key = os.environ.get("HYDRA_DB_API_KEY")
        if not api_key:
            raise SystemExit("HYDRA_DB_API_KEY not set. Export your key or run with --client mock.")
        if not database:
            raise SystemExit("--db-name is required when --client real")
        return RealHydraClient(api_key=api_key), database
    raise ValueError(kind)


def bootstrap(client_kind: str = "mock", database: str | None = None, limit: int | None = None):
    client, db = build_client(client_kind, database)

    queries = REAL_EVAL_QUERIES if client_kind == "real" else (ALL_SEED_QUERIES[:limit] if limit else ALL_SEED_QUERIES)
    total = len(queries)
    print(f"Running {total} queries against client={client_kind}"
          f"{' (LIMITED subset for a quick check)' if limit else ''}...\n")

    rows = []
    skipped = 0
    start_time = time.time()

    for i, (query, turn, prior_tokens) in enumerate(queries):
        elapsed = time.time() - start_time
        eta = (elapsed / (i + 1)) * (total - i - 1) if i > 0 else 0
        print(f"[{i+1}/{total}] ({elapsed:.0f}s elapsed, ~{eta:.0f}s left) {query[:55]}")

        feats = extract_features(query, session_turn_number=turn, prior_context_tokens=prior_tokens)
        try:
            fast = client.query(query, database=db, mode="fast", graph_context=True)
            thinking = client.query(query, database=db, mode="thinking", graph_context=True)
        except Exception as e:
            print(f"  SKIPPING after repeated failures: {e}")
            skipped += 1
            continue

        if fast.graph_context_nonempty or (
            not feats.has_relational_keywords and not feats.has_temporal_keywords
        ):
            label_mode, label_graph = "fast", fast.graph_context_nonempty
        elif thinking.graph_context_nonempty:
            label_mode, label_graph = "thinking", True
        else:
            label_mode, label_graph = "fast", False

        row = feats.to_dict()
        row.update({
            "query": query,
            "label_mode": label_mode,
            "label_graph_context": label_graph,
            "fast_latency_ms": round(fast.latency_ms, 1),
            "thinking_latency_ms": round(thinking.latency_ms, 1),
            "fast_graph_nonempty": fast.graph_context_nonempty,
            "thinking_graph_nonempty": thinking.graph_context_nonempty,
        })
        rows.append(row)

    if not rows:
        print("\nEverything failed - nothing written. Check connectivity/API key.")
        return []

    fieldnames = list(rows[0].keys())
    out_path = os.path.abspath(OUTPUT_PATH)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} bootstrapped labels to {out_path} ({skipped} skipped)")
    print(f"Label distribution: "
          f"fast={sum(1 for r in rows if r['label_mode']=='fast')}, "
          f"thinking={sum(1 for r in rows if r['label_mode']=='thinking')}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client", choices=["mock", "real"], default="mock")
    parser.add_argument("--db-name", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N queries (quick check)")
    args = parser.parse_args()
    bootstrap(client_kind=args.client, database=args.db_name, limit=args.limit)