"""Lab 4 — the service's own metrics, in Prometheus text format.

Hand-written rather than prometheus_client: requirements-serve.txt is hash-compiled inside
the amd64 image and a new dependency costs a ten-minute recompile, while the exposition
format is a page of text. Five signals, the five the dashboard needs:

  http_requests_total{path,status,status_class}   rate, and errors split 4xx / 5xx
  request_latency_ms_bucket{path,le}              p50 / p95 / p99 via histogram_quantile
  model_version_info{version}                     which model is answering
  feature_window_psi{feature}, feature_window_mean{feature}
                                                  the inputs over the last WINDOW rows,
                                                  scored against the training profile

The window is the same size the scheduled detector uses (monitoring/drift.WINDOW), so the
dashboard panel and the alert are the same statistic over the same amount of evidence.
"""
from __future__ import annotations

import json
import os
import threading
from collections import deque
from pathlib import Path

import numpy as np

from monitoring.drift import WINDOW, psi_from_profile

# Milliseconds. Dense around the 25 ms the model costs and the 50 ms service p95, because
# histogram_quantile interpolates inside a bucket and a coarse bucket there would make the
# p95 panel read whatever the bucket edges are.
BUCKETS_MS = (5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300, 500, 1000, 2500, 5000)

# Bounded label values. An unbounded path label is a cardinality bomb the first time a
# scanner walks the API.
KNOWN_PATHS = {"/predict", "/predict/batch", "/ready", "/health", "/metrics"}

DEFAULT_PROFILE = Path(__file__).resolve().parents[1] / "monitoring" / "reference_profile.json"


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _labels(**labels: str) -> str:
    return "{" + ",".join(f'{k}="{_escape(str(v))}"' for k, v in labels.items()) + "}"


class Metrics:
    def __init__(self, profile_path: Path | None = None, window: int = WINDOW) -> None:
        self._lock = threading.Lock()
        self._requests: dict[tuple[str, str], int] = {}
        self._hist: dict[str, list[int]] = {}
        self._sum: dict[str, float] = {}
        self._rows: deque[dict[str, float]] = deque(maxlen=window)
        path = profile_path or Path(os.environ.get("REFERENCE_PROFILE", DEFAULT_PROFILE))
        self.profile = json.loads(path.read_text()) if path.exists() else None

    def observe_request(self, path: str, status: int, latency_ms: float) -> None:
        path = path if path in KNOWN_PATHS else "other"
        with self._lock:
            key = (path, str(status))
            self._requests[key] = self._requests.get(key, 0) + 1
            counts = self._hist.setdefault(path, [0] * (len(BUCKETS_MS) + 1))
            counts[int(np.searchsorted(BUCKETS_MS, latency_ms, side="left"))] += 1
            self._sum[path] = self._sum.get(path, 0.0) + latency_ms

    def observe_rows(self, rows: list[dict[str, float]]) -> None:
        with self._lock:
            self._rows.extend(rows)

    def render(self, model_version: str) -> str:
        with self._lock:
            requests = dict(self._requests)
            hist = {p: list(c) for p, c in self._hist.items()}
            sums = dict(self._sum)
            rows = list(self._rows)

        out = ["# HELP http_requests_total Requests by path and status.",
               "# TYPE http_requests_total counter"]
        for (path, status), n in sorted(requests.items()):
            out.append(f"http_requests_total{_labels(path=path, status=status, status_class=status[0] + 'xx')} {n}")

        out += ["# HELP request_latency_ms Server-side request latency in milliseconds.",
                "# TYPE request_latency_ms histogram"]
        for path, counts in sorted(hist.items()):
            cumulative = 0
            for edge, n in zip(BUCKETS_MS, counts):
                cumulative += n
                out.append(f"request_latency_ms_bucket{_labels(path=path, le=edge)} {cumulative}")
            cumulative += counts[-1]
            out.append(f"request_latency_ms_bucket{_labels(path=path, le='+Inf')} {cumulative}")
            out.append(f"request_latency_ms_sum{_labels(path=path)} {sums[path]:.3f}")
            out.append(f"request_latency_ms_count{_labels(path=path)} {cumulative}")

        out += ["# HELP model_version_info The model version this process serves.",
                "# TYPE model_version_info gauge",
                f"model_version_info{_labels(version=model_version)} 1"]

        out += ["# HELP feature_window_rows Rows in the rolling input window.",
                "# TYPE feature_window_rows gauge",
                f"feature_window_rows {len(rows)}"]
        if rows:
            out += ["# HELP feature_window_mean Mean of each feature over the rolling window.",
                    "# TYPE feature_window_mean gauge"]
            features = list(rows[0])
            columns = {f: np.array([r[f] for r in rows], dtype=float) for f in features}
            for f in features:
                out.append(f"feature_window_mean{_labels(feature=f)} {columns[f].mean():.6f}")
            if self.profile:
                out += ["# HELP feature_window_psi PSI of the rolling window against the training profile.",
                        "# TYPE feature_window_psi gauge"]
                for f, ref in self.profile["features"].items():
                    if f in columns:
                        score = psi_from_profile(ref, columns[f], self.profile["rows"])
                        out.append(f"feature_window_psi{_labels(feature=f)} {score:.6f}")
        return "\n".join(out) + "\n"
