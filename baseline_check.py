#!/usr/bin/env python3
"""
SZCA Media Gateway — Baseline & Production Verification Script

Validates:
 1. Gateway health endpoint GET /health (HTTP 200 OK)
 2. Pool saturation and replica counts GET /v1/pools
 3. Prometheus metrics export GET /metrics
"""

import sys
import json
import urllib.request
import urllib.error

HOST = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"

def check_endpoint(path, name):
    url = f"{HOST}{path}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = resp.read().decode('utf-8')
            print(f"[PASS] {name} ({path}) -> HTTP {resp.status}")
            return resp.status == 200, data
    except urllib.error.HTTPError as e:
        print(f"[FAIL] {name} ({path}) -> HTTP {e.code}")
        return False, ""
    except Exception as e:
        print(f"[FAIL] {name} ({path}) -> {e}")
        return False, ""

def main():
    print(f"=== SZCA Gateway Baseline Health Verification ({HOST}) ===\n")
    
    # 1. Health check
    h_ok, _ = check_endpoint("/health", "Health Endpoint")
    
    # 2. Pools health check
    p_ok, p_data = check_endpoint("/v1/pools", "Stage Pools Status")
    if p_ok:
        try:
            pools_json = json.loads(p_data)
            print(f"       Pool Metrics: {json.dumps(pools_json, indent=2)}")
        except Exception:
            pass

    # 3. Metrics export
    m_ok, _ = check_endpoint("/metrics", "Prometheus Metrics")

    print("\n-----------------------------------------")
    if h_ok and p_ok and m_ok:
        print("RESULT: All baseline health checks PASSED.")
        sys.exit(0)
    else:
        print("RESULT: One or more baseline checks FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    main()
