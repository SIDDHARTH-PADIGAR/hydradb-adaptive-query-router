"""
Thin wrapper around HydraDB's REAL API, confirmed against their live docs
(docs.hydradb.com/quickstart) - NOT the guessed /query + database shape
this file used before. Actual shape:

  POST /tenants/create              - create an isolated workspace
  GET  /tenants/infra/status        - poll until provisioning finishes
  POST /memories/add_memory         - ingest plain-text content
  POST /recall/full_recall          - the actual query call

Base URL: https://api.hydradb.com
Auth: Authorization: Bearer <api_key>
"""
from __future__ import annotations
import time
from typing import Optional

import requests

from hydra_client.base import HydraClient, QueryResult

BASE_URL = "https://api.hydradb.com"


class RealHydraClient(HydraClient):
    def __init__(self, api_key: str, base_url: str = BASE_URL,
                 timeout_fast: float = 15.0, timeout_thinking: float = 30.0,
                 max_retries: int = 2):
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout_fast = timeout_fast
        self._timeout_thinking = timeout_thinking
        self._max_retries = max_retries

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def create_tenant(self, tenant_id: str) -> dict:
        resp = requests.post(
            f"{self._base_url}/tenants/create",
            headers=self._headers(),
            json={"tenant_id": tenant_id},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def wait_for_tenant_ready(self, tenant_id: str, timeout_s: float = 60.0):
        """Tenant creation is async - poll until infra is provisioned."""
        start = time.time()
        while time.time() - start < timeout_s:
            resp = requests.get(
                f"{self._base_url}/tenants/infra/status",
                headers=self._headers(),
                params={"tenant_id": tenant_id},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            infra = data.get("infra", {})
            graph_ok = infra.get("graph_status")
            vector_ok = infra.get("vectorstore_status")
            vector_ok = all(vector_ok) if isinstance(vector_ok, list) else bool(vector_ok)
            if graph_ok and vector_ok:
                return data
            time.sleep(2)
        raise TimeoutError(f"Tenant {tenant_id} did not finish provisioning in {timeout_s}s")
    
    def upload_knowledge_file(self, tenant_id: str, file_path: str) -> dict:
        """POST /ingestion/upload_knowledge - actual file upload, not memories.
        Returns file_ids we then poll with verify_processing."""
        with open(file_path, "rb") as f:
            resp = requests.post(
                f"{self._base_url}/ingestion/upload_knowledge",
                headers={"Authorization": f"Bearer {self._api_key}"},  # no Content-Type - multipart sets its own
                data={"tenant_id": tenant_id},
                files={"files": (file_path.split("\\")[-1].split("/")[-1], f, "text/plain")},
                timeout=30,
            )
        resp.raise_for_status()
        return resp.json()

    def verify_processing(self, tenant_id: str, file_ids: list[str]) -> dict:
        resp = requests.post(
            f"{self._base_url}/ingestion/verify_processing",
            headers=self._headers(),
            params={"tenant_id": tenant_id, "file_ids": file_ids},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()
    
    def add_memory(self, tenant_id: str, text: str, sub_tenant_id: str = "poc_content",
                   infer: bool = False) -> dict:
        resp = requests.post(
            f"{self._base_url}/memories/add_memory",
            headers=self._headers(),
            json={
                "tenant_id": tenant_id,
                "sub_tenant_id": sub_tenant_id,
                "memories": [{"text": text, "infer": infer}],
            },
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()

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
        payload = {
            "tenant_id": database,
            "query": query,
            "mode": mode,
            "graph_context": graph_context,
            "max_results": max_results,
        }

        timeout = self._timeout_thinking if mode == "thinking" else self._timeout_fast

        last_error = None
        for attempt in range(self._max_retries + 1):
            try:
                start = time.perf_counter()
                resp = requests.post(
                    f"{self._base_url}/recall/full_recall",
                    headers=self._headers(),
                    json=payload,
                    timeout=timeout,
                )
                elapsed_ms = (time.perf_counter() - start) * 1000
                resp.raise_for_status()
                data = resp.json()
                chunks = data.get("chunks") or data.get("results") or data.get("data") or []
                return QueryResult(
                    chunks=chunks,
                    graph_context=data.get("graph_context"),
                    latency_ms=elapsed_ms,
                    mode_used=mode,
                    raw=data,
                )
            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
                last_error = e
                if attempt < self._max_retries:
                    wait = 2 ** attempt  # 1s, then 2s
                    print(f"  (timeout on '{query[:40]}...', retry {attempt+1}/{self._max_retries} in {wait}s)")
                    time.sleep(wait)
                    continue
                raise last_error