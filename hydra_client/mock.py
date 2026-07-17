"""
Simulates HydraDB's /query behavior based on numbers published in their
own docs:
  - fast mode:      <=500ms, single-pass, shallow/often-empty graph slice
  - thinking mode:  3-5s, multi-hop traversal + rerank, richer graph_context

IMPORTANT: this is a stand-in, not ground truth. It exists so the rest of
the pipeline (features -> rules -> classifier -> eval harness) can be built
and tested end to end before a real API key is wired in. The moment a real
key exists, swap MockHydraClient for RealHydraClient in main.py/api and
re-run eval/bootstrap.py to regenerate labels from real measurements.

FIX: this used to define its own separate RELATIONAL_MARKERS tuple that
didn't match features/extract.py's list (missing "why does" and "who
owns"), which meant labels generated here could contradict the features
the model was trained on. Now imports the same list features.extract uses,
so there's exactly one definition, not two that can drift apart.
"""
from __future__ import annotations
import random
import time
from typing import Optional

from hydra_client.base import HydraClient, QueryResult
from features.extract import RELATIONAL_MARKERS


class MockHydraClient(HydraClient):
    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

    def query(
        self,
        query: str,
        database: str,
        mode: str = "fast",
        graph_context: bool = True,
        alpha: float | str = 0.8,
        max_results: int = 10,
        query_by: str = "hybrid",
        collection: Optional[str] = None,
    ) -> QueryResult:
        q_lower = query.lower()
        relational = any(m in q_lower for m in RELATIONAL_MARKERS)
        length_factor = min(len(query.split()) / 20.0, 1.0)

        if mode == "thinking":
            base = self._rng.uniform(2800, 4600)
            base += 400 if relational else 0
            simulated_ms = base + length_factor * 300
        else:
            base = self._rng.uniform(90, 420)
            base += 60 if relational else 0
            simulated_ms = min(base + length_factor * 80, 500)

        chunks = [
            {"id": f"chunk_{i}", "score": round(self._rng.uniform(0.4, 0.95), 3)}
            for i in range(min(max_results, 5))
        ]

        graph_ctx = None
        if graph_context:
            if mode == "thinking" and relational:
                graph_ctx = {
                    "query_paths": [
                        {
                            "triplets": [{"source": {"name": "entity_a"},
                                          "relation": {"canonical_predicate": "relates_to"},
                                          "target": {"name": "entity_b"}}],
                            "relevancy_score": round(self._rng.uniform(0.5, 0.9), 2),
                            "group_id": "p_0",
                        }
                    ],
                    "chunk_relations": [],
                }
            elif mode == "thinking":
                graph_ctx = {"query_paths": [], "chunk_relations": []} if self._rng.random() > 0.3 else None
            else:
                graph_ctx = None

        return QueryResult(
            chunks=chunks,
            graph_context=graph_ctx,
            latency_ms=simulated_ms,
            mode_used=mode,
            raw={"simulated": True, "query": query},
        )