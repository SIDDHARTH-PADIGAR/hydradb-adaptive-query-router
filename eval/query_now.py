"""
Ingestion already completed (confirmed: indexing_status == "completed").
Just queries directly instead of re-running the whole upload.

Run:
    python -m eval.query_now
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hydra_client.real import RealHydraClient

TENANT_ID = "hydra_router_poc"

api_key = os.environ.get("HYDRA_DB_API_KEY")
client = RealHydraClient(api_key=api_key)

for q, mode in [
    ("Who owns the payments service?", "fast"),
    ("Why did the payments incident happen?", "thinking"),
]:
    print(f"\n--- query='{q}' mode={mode} ---")
    result = client.query(q, database=TENANT_ID, mode=mode, graph_context=True)
    print(f"latency: {result.latency_ms:.0f}ms")
    print(json.dumps(result.raw, indent=2))