"""
Run: uvicorn api.main:app --reload --port 8000

POST /route          -> routing decision only, no HydraDB call
POST /route/execute   -> routing decision + actual HydraDB /query call

Client selection happens in ONE place (get_client, below): if
HYDRA_DB_API_KEY is set in the environment, real calls are used;
otherwise it falls back to the mock so the API always runs.
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from pydantic import BaseModel, Field

from router import decide, RoutingDecision
from hydra_client.mock import MockHydraClient

app = FastAPI(title="HydraDB Query Router", version="0.1.0")

_client_cache = {}


def get_client():
    if "client" in _client_cache:
        return _client_cache["client"], _client_cache["database"]

    api_key = os.environ.get("HYDRA_DB_API_KEY")
    if api_key:
        from hydra_client.real import RealHydraClient
        database = os.environ.get("HYDRA_DB_DATABASE", "router_poc_db")
        client = RealHydraClient(api_key=api_key)
        source = "real"
    else:
        client = MockHydraClient(seed=None)
        database = "mock_db"
        source = "mock"

    _client_cache.update(client=client, database=database, source=source)
    return client, database


class RouteRequest(BaseModel):
    query: str
    session_turn_number: int = Field(default=1, ge=1)
    prior_context_tokens: int = Field(default=0, ge=0)


@app.post("/route")
def route(req: RouteRequest) -> dict:
    decision: RoutingDecision = decide(
        req.query, req.session_turn_number, req.prior_context_tokens
    )
    return decision.to_dict()


@app.post("/route/execute")
def route_execute(req: RouteRequest) -> dict:
    decision: RoutingDecision = decide(
        req.query, req.session_turn_number, req.prior_context_tokens
    )
    client, database = get_client()
    result = client.query(
        req.query,
        database=database,
        mode=decision.mode,
        graph_context=decision.graph_context,
        alpha=decision.alpha,
        max_results=decision.max_results,
        query_by=decision.query_by,
    )
    return {
        "decision": decision.to_dict(),
        "client_source": _client_cache.get("source"),
        "result": {
            "latency_ms": round(result.latency_ms, 1),
            "mode_used": result.mode_used,
            "graph_context_nonempty": result.graph_context_nonempty,
            "chunk_count": len(result.chunks),
        },
    }


@app.get("/health")
def health() -> dict:
    _, database = get_client()
    return {"status": "ok", "client_source": _client_cache.get("source"), "database": database}
