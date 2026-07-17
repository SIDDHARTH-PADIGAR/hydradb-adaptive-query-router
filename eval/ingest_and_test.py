"""
Ingests real content via the actual file-upload path (not memories, not
the console UI), waits for processing to genuinely complete, then queries
it - full round trip, fully visible at every step.

Run:
    python -m eval.ingest_and_test
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hydra_client.real import RealHydraClient

TENANT_ID = "hydra_router_poc"
SAMPLE_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "sample_knowledge.txt")

api_key = os.environ.get("HYDRA_DB_API_KEY")
if not api_key:
    raise SystemExit("Set $env:HYDRA_DB_API_KEY first.")

if not os.path.exists(os.path.abspath(SAMPLE_FILE)):
    raise SystemExit(
        f"Create {SAMPLE_FILE} first - the sample_knowledge.txt content "
        f"I gave you a few messages ago, saved into data/."
    )

client = RealHydraClient(api_key=api_key)

print(f"Ensuring tenant '{TENANT_ID}' is ready...")
try:
    client.create_tenant(TENANT_ID)
except Exception as e:
    print(f"  (probably already exists: {e})")
client.wait_for_tenant_ready(TENANT_ID)
print("Tenant ready.\n")

print("Uploading sample_knowledge.txt via /ingestion/upload_knowledge...")
upload_result = client.upload_knowledge_file(TENANT_ID, os.path.abspath(SAMPLE_FILE))
print(json.dumps(upload_result, indent=2))

file_ids = [r.get("file_id") or r.get("source_id") for r in upload_result.get("results", upload_result.get("files", []))]
file_ids = [f for f in file_ids if f]
print(f"\nFile IDs to track: {file_ids}\n")

print("Polling verify_processing every 5s, up to 90s...")
completed = False
for attempt in range(18):
    time.sleep(5)
    status = client.verify_processing(TENANT_ID, file_ids)
    print(f"  [{(attempt+1)*5}s] {json.dumps(status)}")
    statuses = [item.get("status") for item in status.get("results", status.get("files", []))]
    if statuses and all(s == "completed" for s in statuses):
        completed = True
        break
    if any(s == "errored" for s in statuses):
        print("A file errored during processing - stopping.")
        break

print()
if not completed:
    print("Did not reach 'completed' status in 90s - either it's genuinely")
    print("slow, or something's wrong. Full status output is above.")
else:
    print("Processing completed. Querying now...\n")
    result = client.query("Who owns the payments service?", database=TENANT_ID, mode="thinking", graph_context=True)
    print(json.dumps(result.raw, indent=2))