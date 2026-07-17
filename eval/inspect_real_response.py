"""
Run this to see the actual shape of a real HydraDB response, and to
diagnose why content isn't showing up in queries.

Run:
    python -m eval.inspect_real_response
    python -m eval.inspect_real_response --tenant-id your_console_tenant_id
"""
import argparse
import json
import os
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hydra_client.real import RealHydraClient

api_key = os.environ.get("HYDRA_DB_API_KEY")
if not api_key:
    raise SystemExit("Set $env:HYDRA_DB_API_KEY first.")

parser = argparse.ArgumentParser()
parser.add_argument("--tenant-id", default="hydra_router_poc")
args = parser.parse_args()
TENANT_ID = args.tenant_id

client = RealHydraClient(api_key=api_key)

print(f"Using tenant '{TENANT_ID}'")
print("Creating tenant (safe to skip if it already exists)...")
try:
    client.create_tenant(TENANT_ID)
except Exception as e:
    print(f"  (create_tenant returned: {e} - probably already exists, continuing)")

print("Waiting for tenant to finish provisioning...")
client.wait_for_tenant_ready(TENANT_ID)
print("Tenant ready.\n")

print("Adding one test memory - printing the FULL raw response this time...")
add_result = client.add_memory(TENANT_ID, "The payments service is owned by the platform team.", infer=False)
print(json.dumps(add_result, indent=2))
print()

print("Polling every 5s, up to 30s, until a query returns something...")
found = False
for attempt in range(6):
    time.sleep(5)
    try:
        result = client.query("Who owns the payments service?", database=TENANT_ID, mode="thinking", graph_context=True)
    except requests.exceptions.HTTPError as e:
        print(f"  [{(attempt+1)*5}s] HTTP {e.response.status_code} error. Response body:")
        print(f"    {e.response.text[:500]}")
        continue
    has_content = bool(result.chunks) or bool(result.raw.get("graph_context", {}).get("query_paths"))
    print(f"  [{(attempt+1)*5}s] chunks={len(result.chunks)}, "
          f"query_paths={len(result.raw.get('graph_context', {}).get('query_paths', []))}")
    if has_content:
        found = True
        break

print()
if found:
    print("Got content back. Full response:")
    print(json.dumps(result.raw, indent=2))
else:
    print("Still empty after 60s. Something beyond simple indexing delay is going on -")
    print("check the add_memory response above for any error/warning field, and double")
    print("check --tenant-id matches your console's actual tenant.")
    
print("\n" + "=" * 70)
print("Checking existing KNOWLEDGE sources on this tenant (the console upload)")
print("=" * 70)
resp = requests.post(
    f"{client._base_url}/list/data",
    headers=client._headers(),
    json={"tenant_id": TENANT_ID, "kind": "knowledge", "page": 1, "page_size": 20},
    timeout=10,
)
print(f"status: {resp.status_code}")
print(json.dumps(resp.json(), indent=2))

print("\n" + "=" * 70)
print("Checking existing MEMORY sources on this tenant (what our script added)")
print("=" * 70)
resp2 = requests.post(
    f"{client._base_url}/list/data",
    headers=client._headers(),
    json={"tenant_id": TENANT_ID, "kind": "memory", "page": 1, "page_size": 20},
    timeout=10,
)
print(f"status: {resp2.status_code}")
print(json.dumps(resp2.json(), indent=2))