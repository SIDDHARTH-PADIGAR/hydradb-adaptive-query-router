"""
Compares three strategies across the seed query set:
  1. always_fast     - naive baseline, everything forced to mode=fast
  2. always_thinking - naive baseline, everything forced to mode=thinking
  3. router          - our decision layer picks per-query

Run:
    python -m eval.benchmark --client mock
    python -m eval.benchmark --client real --db-name hydra_router_poc
"""
from __future__ import annotations
from data.real_eval_queries import REAL_EVAL_QUERIES
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.seed_queries import ALL_SEED_QUERIES
from router import decide
from hydra_client.mock import MockHydraClient


def build_client(kind: str, database: str | None):
    if kind == "mock":
        return MockHydraClient(seed=7), "mock_db"
    from hydra_client.real import RealHydraClient
    api_key = os.environ.get("HYDRA_DB_API_KEY")
    if not api_key:
        raise SystemExit("HYDRA_DB_API_KEY not set. Use --client mock instead.")
    return RealHydraClient(api_key=api_key), database or "router_poc_db"


def run_benchmark(client_kind: str = "mock", database: str | None = None):
    client, db = build_client(client_kind, database)

    totals = {"always_fast": [], "always_thinking": [], "router": []}
    graph_hits = {"always_fast": 0, "always_thinking": 0, "router": 0}
    skipped = 0

    queries = REAL_EVAL_QUERIES if client_kind == "real" else ALL_SEED_QUERIES
    for i, (query, turn, prior_tokens) in enumerate(queries):
        print(f"[{i+1}/{len(queries)}] {query[:60]}")
        try:
            # naive baseline 1: everything forced fast
            r_fast = client.query(query, database=db, mode="fast", graph_context=True)
            # naive baseline 2: everything forced thinking
            r_think = client.query(query, database=db, mode="thinking", graph_context=True)
            # router: let the decision layer choose
            decision = decide(query, turn, prior_tokens)
            r_router = client.query(
                query, database=db, mode=decision.mode,
                graph_context=decision.graph_context,
                alpha=decision.alpha, max_results=decision.max_results,
            )
        except Exception as e:
            print(f"  SKIPPING after repeated failures: {e}")
            skipped += 1
            continue

        totals["always_fast"].append(r_fast.latency_ms)
        graph_hits["always_fast"] += int(r_fast.graph_context_nonempty)

        totals["always_thinking"].append(r_think.latency_ms)
        graph_hits["always_thinking"] += int(r_think.graph_context_nonempty)

        totals["router"].append(r_router.latency_ms)
        graph_hits["router"] += int(r_router.graph_context_nonempty)

    n = len(totals["router"])
    if n == 0:
        print("\nEvery query failed - nothing to report. Check connectivity/API key.")
        return

    print(f"\nBenchmark over {n} completed queries out of {len(ALL_SEED_QUERIES)} "
          f"({skipped} skipped after retries, client={client_kind})\n")
    print(f"{'strategy':<18}{'avg_latency_ms':>16}{'graph_hit_rate':>18}")
    for strategy in ("always_fast", "always_thinking", "router"):
        avg = sum(totals[strategy]) / n
        hit_rate = graph_hits[strategy] / n
        print(f"{strategy:<18}{avg:>16.1f}{hit_rate:>17.1%}")

    saved_vs_thinking = (
        (sum(totals["always_thinking"]) - sum(totals["router"]))
        / sum(totals["always_thinking"]) * 100
    )
    print(f"\nRouter saves {saved_vs_thinking:.1f}% latency vs always-thinking, "
          f"at {graph_hits['router']}/{n} graph_context hit rate "
          f"(vs {graph_hits['always_thinking']}/{n} for always-thinking).")

    if client_kind == "mock":
        print("\nNOTE: numbers above are meaningless as a founder-facing claim until "
              "--client real is used with a real API key.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client", choices=["mock", "real"], default="mock")
    parser.add_argument("--db-name", default=None)
    args = parser.parse_args()
    run_benchmark(client_kind=args.client, database=args.db_name)