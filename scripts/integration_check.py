"""Lab 4 — integration test against a running container.

    python scripts/integration_check.py --base http://localhost:8080 --expect-version <sha>

Asserts on what a caller depends on, not on "it answered": the response shape, that the
version it reports is the one CI built, and that a bad row is refused with a usable error.
A grep for "probability" passes on an error body that happens to mention the word.
"""
from __future__ import annotations

import argparse
import sys
import time

import requests

VALID = {
    "temp_c": 78.4, "vibration_mm_s": 3.1, "pressure_kpa": 315.2,
    "hours_since_service": 4200.0, "load_pct": 68.0, "ambient_humidity": 55.0,
}


def wait_ready(base: str, timeout_s: float) -> dict:
    deadline = time.monotonic() + timeout_s
    last = "no response"
    while time.monotonic() < deadline:
        try:
            r = requests.get(f"{base}/ready", timeout=2)
            if r.status_code == 200:
                return r.json()
            last = f"{r.status_code} {r.text[:200]}"
        except requests.RequestException as exc:
            last = type(exc).__name__
        time.sleep(1)
    raise AssertionError(f"not ready after {timeout_s:.0f}s: {last}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8080")
    ap.add_argument("--expect-version", required=True)
    ap.add_argument("--timeout", type=float, default=90)
    args = ap.parse_args()

    ready = wait_ready(args.base, args.timeout)
    assert ready["model_version"] == args.expect_version, f"/ready reports {ready}"

    r = requests.post(f"{args.base}/predict", json=VALID, timeout=10)
    assert r.status_code == 200, f"/predict {r.status_code}: {r.text[:300]}"
    body = r.json()
    assert set(body) == {"probability", "model_version", "request_id"}, f"shape: {sorted(body)}"
    assert 0.0 <= body["probability"] <= 1.0, body
    assert body["model_version"] == args.expect_version, (
        f"served {body['model_version']!r}, CI built {args.expect_version!r}")
    assert r.headers.get("x-model-version") == args.expect_version, dict(r.headers)

    bad = requests.post(f"{args.base}/predict", json={**VALID, "temp_c": "hot"}, timeout=10)
    assert bad.status_code == 422, f"bad row got {bad.status_code}"
    assert bad.json()["detail"][0]["field"] == "temp_c", bad.text[:300]

    print(f"PASS  version={body['model_version']} probability={body['probability']:.4f} "
          f"request_id={body['request_id']}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as exc:
        print(f"FAIL  {exc}")
        sys.exit(1)
