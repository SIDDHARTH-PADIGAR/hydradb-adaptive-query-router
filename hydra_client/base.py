"""
Common interface for anything that can execute a HydraDB /query call.

Both MockHydraClient (hydra_client/mock.py) and RealHydraClient
(hydra_client/real.py) implement this. Nothing else in the codebase
should import either concrete class directly except main.py / api,
where the choice of which to use is made in ONE place.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class QueryResult:
    """Normalized shape returned by any client implementation."""
    chunks: list[dict[str, Any]] = field(default_factory=list)
    graph_context: Optional[dict[str, Any]] = None
    latency_ms: float = 0.0
    mode_used: str = "fast"
    raw: Optional[dict[str, Any]] = None

    @property
    def graph_context_nonempty(self) -> bool:
        """Real API responses populate chunk_relations with actual
        triplets even when query_paths is empty - confirmed against a
        real response. Check both, not just query_paths."""
        if not self.graph_context:
            return False
        paths = self.graph_context.get("query_paths") or []
        relations = self.graph_context.get("chunk_relations") or []
        return len(paths) > 0 or len(relations) > 0


class HydraClient(ABC):
    @abstractmethod
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
        """Execute a single POST /query call and return a normalized result."""
        raise NotImplementedError
