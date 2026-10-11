"""Lab 3/4 — service contract tests.

Two kinds of test here. The contract tests use one shared client. The lifecycle tests
each build their own app, because "loaded once at startup", "ready only after it can
score", and "alive while loading" are claims about startup, and a shared client has
already started up.
"""
from __future__ import annotations

import importlib
import io
import json
import logging
import os
import threading
from contextlib import contextmanager

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from src import config, data, seeds  # noqa: E402

RAW = config.REPO_ROOT / "data" / "raw" / "sensors.csv"
VALID = {
    "temp_c": 78.4, "vibration_mm_s": 3.1, "pressure_kpa": 315.2,
    "hours_since_service": 4200.0, "load_pct": 68.0, "ambient_humidity": 55.0,
}


def _local_model_path():
    """A model on disk for the tests to serve. The deployed service loads a registered
    version through the adapter; this exists so CI needs no cloud account."""
    if not RAW.exists():
        pytest.skip("run `make data` first")
    model_path = config.REPO_ROOT / "reports" / "model.joblib"
    if not model_path.exists():
        import joblib
        from sklearn.ensemble import RandomForestClassifier

        seed = seeds.set_all()
        df = data.load_raw(RAW)
        train_df, _, _ = data.split(df, seed=seed)
        model = RandomForestClassifier(n_estimators=60, max_depth=8, random_state=seed)
        model.fit(train_df[data.FEATURES], train_df[data.TARGET])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
    return model_path


@contextmanager
def app_module(**env: str):
    """A freshly imported service.app with this environment, restored afterwards."""
    saved = dict(os.environ)
    os.environ["MODEL_PATH"] = str(_local_model_path())
    os.environ["MODEL_VERSION"] = "test-1"
    os.environ.pop("MODEL_REGISTRY_NAME", None)
    os.environ.update(env)
    try:
        import service.app as module

        yield importlib.reload(module)
    finally:
        os.environ.clear()
        os.environ.update(saved)


@pytest.fixture(scope="module")
def client():
    with app_module() as module, TestClient(module.app) as c:
        yield c


# --- contract ----------------------------------------------------------------

def test_health_and_ready_are_different(client):
    """Liveness passes whenever the process is up. Readiness only when it can score."""
    assert client.get("/health").json()["status"] == "alive"
    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"


def test_predict_returns_probability_and_version(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["model_version"] == "test-1"
    assert body["request_id"] == r.headers["x-request-id"]
    assert r.headers["x-model-version"] == "test-1"


def test_out_of_range_input_is_rejected(client):
    bad = {**VALID, "load_pct": 250.0}
    assert client.post("/predict", json=bad).status_code == 422


def test_unknown_field_is_rejected(client):
    """extra='forbid'. A silently ignored field is how a caller ends up sending a feature
    you never read while believing it matters."""
    assert client.post("/predict", json={**VALID, "surprise": 1}).status_code == 422


def test_missing_field_is_rejected(client):
    incomplete = {k: v for k, v in VALID.items() if k != "temp_c"}
    assert client.post("/predict", json=incomplete).status_code == 422


def test_rejection_names_the_field_and_echoes_the_value(client):
    """A 422 that does not say which field is a support ticket, not an error message."""
    body = client.post("/predict", json={**VALID, "load_pct": 250.0}).json()
    assert body["error"] == "validation_error"
    offending = [d for d in body["detail"] if d["field"] == "load_pct"]
    assert offending and offending[0]["received"] == 250.0
    assert body["request_id"] and body["model_version"] == "test-1"


def test_schema_bounds_match_the_data_contract():
    """service/schemas.py and src/data.PLAUSIBLE_RANGES must not drift apart: the service
    would accept rows the training data never contained, and nothing would say so."""
    from service.schemas import PredictRequest

    for feature, (low, high) in data.PLAUSIBLE_RANGES.items():
        meta = PredictRequest.model_fields[feature].metadata
        bounds = {type(m).__name__: getattr(m, "ge", getattr(m, "le", None)) for m in meta}
        assert bounds.get("Ge") == low, feature
        assert bounds.get("Le") == high, feature


def test_batch_matches_singles(client):
    rows = [VALID, {**VALID, "temp_c": 92.0}]
    batch = client.post("/predict/batch", json={"rows": rows}).json()["probabilities"]
    singles = [client.post("/predict", json=r).json()["probability"] for r in rows]
    assert batch == pytest.approx(singles, abs=1e-9)


def test_batch_size_limit_enforced(client):
    r = client.post("/predict/batch", json={"rows": [VALID] * 101})
    assert r.status_code == 422


def test_rejection_does_not_echo_the_whole_payload_back(client):
    """An oversized batch must not come back as an oversized error body: that is the
    request amplified at the sender, and at whoever stores the logs."""
    body = client.post("/predict/batch", json={"rows": [VALID] * 101}).json()
    received = body["detail"][0]["received"]
    assert received == {"type": "array", "length": 101}
    assert len(json.dumps(body)) < 1000


def test_request_id_from_the_caller_is_kept(client):
    """A caller-supplied id is what joins their trace to our logs. Minting a new one
    breaks the join silently."""
    r = client.post("/predict", json=VALID, headers={"x-request-id": "caller-abc"})
    assert r.headers["x-request-id"] == "caller-abc"
    assert r.json()["request_id"] == "caller-abc"


def test_every_log_line_is_one_json_object(client):
    """The log format is only structured if it survives a message containing quotes."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    import service.app as module

    handler.setFormatter(module.JsonFormatter())
    logging.getLogger().addHandler(handler)
    try:
        client.post("/predict", json=VALID)
        client.post("/predict", json={**VALID, "surprise": '"quoted"\nnewline'})
    finally:
        logging.getLogger().removeHandler(handler)

    lines = [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]
    requests = [rec for rec in lines if rec["msg"] == "request"]
    assert len(requests) == 2
    for rec in requests:
        assert rec["request_id"] != "-"
        assert rec["model_version"] == "test-1"
        assert rec["latency_ms"] >= 0
        assert rec["path"] == "/predict"
    assert requests[0]["rows"] == 1
    assert requests[1]["status"] == 422


# --- lifecycle ---------------------------------------------------------------

def test_logs_go_to_stdout_with_a_severity(capsys):
    """A platform reads a container's stderr as errors. Logging INFO there makes a healthy
    endpoint report a steady stream of errors from its own health checks, which is how an
    error-rate alert becomes noise before it is ever used."""
    import service.app as module

    saved = logging.getLogger().handlers
    log = module._configure_logging()
    try:
        capsys.readouterr()
        log.info("probe", extra={"fields": {"path": "/ready", "status": 200}})
        captured = capsys.readouterr()
    finally:
        logging.getLogger().handlers = saved

    assert captured.err == ""
    record = json.loads(captured.out.strip().splitlines()[-1])
    assert record["severity"] == record["level"] == "INFO"
    assert record["path"] == "/ready"


def test_model_is_loaded_once_at_startup():
    """Per-request loading is the commonest cause of a p99 that looks nothing like p50,
    and it is invisible until you count."""
    with app_module() as module:
        with TestClient(module.app) as c:
            for _ in range(5):
                assert c.post("/predict", json=VALID).status_code == 200
        assert module.STATE.loads == 1


def test_alive_while_loading_but_not_ready():
    """The handout's diagram: /health 200 and /ready 503 in the window before the model
    exists. Only reproducible because the loader runs off the startup path."""
    with app_module(MODEL_LOAD_WAIT_S="0") as module:
        release = threading.Event()
        real_load = module._load_model
        module._load_model = lambda: (release.wait(5), real_load())[1]
        try:
            with TestClient(module.app) as c:
                assert c.get("/health").json()["status"] == "alive"
                not_ready = c.get("/ready")
                assert not_ready.status_code == 503
                assert not_ready.json()["phase"] == "loading"
                assert c.post("/predict", json=VALID).status_code == 503

                release.set()
                for _ in range(100):
                    if c.get("/ready").status_code == 200:
                        break
                    threading.Event().wait(0.05)
                assert c.get("/ready").status_code == 200
                assert c.post("/predict", json=VALID).status_code == 200
        finally:
            module._load_model = real_load


def test_a_failed_load_stays_alive_and_never_ready():
    """Liveness must not fail here: restarting the container repeats the same failure and
    loses the logs that explain it."""
    with app_module(MODEL_PATH="reports/does-not-exist.joblib") as module:
        with TestClient(module.app) as c:
            assert c.get("/health").json()["status"] == "alive"
            ready = c.get("/ready")
            assert ready.status_code == 503
            assert ready.json()["phase"] == "failed"
            assert "does-not-exist" in ready.json()["reason"]
            assert c.post("/predict", json=VALID).status_code == 503


def test_readiness_requires_scoring_not_just_unpickling():
    """A model that loads but cannot predict is the failure that looks healthy."""
    class Broken:
        def predict_proba(self, frame):
            raise ValueError("feature names should match those that were passed during fit")

    with app_module() as module:
        module._load_model = lambda: (Broken(), {"source": "file", "version": "broken",
                                                 "artifact_uri": "", "aliases": []})
        with TestClient(module.app) as c:
            assert c.get("/health").json()["status"] == "alive"
            assert c.get("/ready").status_code == 503
            assert "feature names" in c.get("/ready").json()["reason"]


# --- Lab 4: metrics --------------------------------------------------------------

def _sample(text: str, name: str, **labels: str) -> float:
    want = ",".join(f'{k}="{v}"' for k, v in labels.items())
    for line in text.splitlines():
        if line.startswith(f"{name}{{{want}}} ") or (not labels and line.startswith(f"{name} ")):
            return float(line.rsplit(" ", 1)[1])
    raise AssertionError(f"{name}{{{want}}} not exposed")


def test_metrics_split_4xx_from_2xx_and_expose_the_version():
    with app_module() as module, TestClient(module.app) as c:
        for _ in range(3):
            c.post("/predict", json=VALID)
        c.post("/predict", json={**VALID, "load_pct": 250.0})
        text = c.get("/metrics").text
    assert _sample(text, "http_requests_total", path="/predict", status="200", status_class="2xx") == 3
    assert _sample(text, "http_requests_total", path="/predict", status="422", status_class="4xx") == 1
    assert _sample(text, "request_latency_ms_count", path="/predict") == 4
    assert _sample(text, "model_version_info", version="test-1") == 1


def test_rejected_rows_never_enter_the_drift_window():
    """A 422 is a caller problem, not an input distribution. Counting it would let one
    misbehaving client move the drift panel."""
    with app_module() as module, TestClient(module.app) as c:
        c.post("/predict", json={**VALID, "load_pct": 250.0})
        c.post("/predict", json=VALID)
        text = c.get("/metrics").text
    assert _sample(text, "feature_window_rows") == 1
    assert _sample(text, "feature_window_mean", feature="load_pct") == VALID["load_pct"]


def test_window_psi_moves_when_the_inputs_shift():
    import numpy as np

    rng = np.random.default_rng(0)
    df = data.load_raw(RAW).sample(400, random_state=1)
    rows = df[data.FEATURES].to_dict("records")
    with app_module() as module, TestClient(module.app) as c:
        c.post("/predict/batch", json={"rows": rows[:100]})
        before = _sample(c.get("/metrics").text, "feature_window_psi", feature="temp_c")
        hot = [{**r, "temp_c": r["temp_c"] + 15 + rng.normal()} for r in rows[100:400]]
        for i in range(0, 300, 100):
            c.post("/predict/batch", json={"rows": hot[i:i + 100]})
        after = _sample(c.get("/metrics").text, "feature_window_psi", feature="temp_c")
    assert after > 0.25 > before, f"PSI {before:.3f} -> {after:.3f}"


def test_unknown_paths_share_one_label():
    with app_module() as module, TestClient(module.app) as c:
        c.get("/wp-admin")
        c.get("/.env")
        text = c.get("/metrics").text
    assert _sample(text, "http_requests_total", path="other", status="404", status_class="4xx") == 2
