"""Lab 3 — inference service.

Provider-neutral by construction: the model arrives through the adapter, and the same
container image deploys to SageMaker, Azure ML, or Vertex AI. Route paths differ per
platform; that difference belongs in cloudlayer/, never here.

Run locally:  make serve VERSION=1          # the registered version, through the adapter
              make serve-local              # a local file, development only

Three decisions in this file are the lab's:

  * The model is resolved through `adapter.get_model(name, version)` and loaded once at
    startup. Nothing here knows a bucket name or a provider hostname — the artifact URI
    is whatever the registry returns, and `make portability-audit` enforces that.
  * Loading runs on a background thread, so liveness answers while readiness is still
    false. That is the sequence in the handout's diagram, and it is only reproducible if
    the process serves the port before the model exists.
  * Readiness is earned by scoring a row, not by a non-None object. A model that
    unpickles but cannot score is the failure that looks healthy.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse

from service.metrics import Metrics
from service.schemas import BatchRequest, BatchResponse, PredictRequest, PredictResponse
from src.data import FEATURES, PLAUSIBLE_RANGES

# Per-request context. The middleware sets both; handlers add to EXTRA so a fact known
# only inside the handler (how many rows were scored) still reaches the one line that
# describes the request.
REQUEST_ID: ContextVar[str] = ContextVar("request_id", default="-")
EXTRA: ContextVar[dict[str, Any] | None] = ContextVar("extra", default=None)


def _extra() -> dict[str, Any]:
    """The current request's log fields. Outside a request this is a throwaway dict, so a
    handler called directly never mutates a shared default."""
    fields = EXTRA.get()
    if fields is None:
        fields = {}
        EXTRA.set(fields)
    return fields


class JsonFormatter(logging.Formatter):
    """One JSON object per line, built with json.dumps.

    Assembling JSON with a format string works until a message contains a quote or a
    newline — which is exactly what an exception message does — and then the collector
    silently drops the line you needed. Escaping is not optional in a log format.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
                  + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            # Several log collectors — Cloud Logging and anything reading the Bunyan or
            # GELF conventions — key alerting off a field called `severity` and ignore
            # `level`. Emitting both costs nine characters a line.
            "severity": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": REQUEST_ID.get(),
            "model_version": STATE.version,
        }
        payload.update(getattr(record, "fields", {}))
        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info).splitlines()[-1]
        return json.dumps(payload, default=str)


def _configure_logging() -> logging.Logger:
    # stdout, not the default stderr. Measured, not preferred: the first deployment sent
    # every line to stderr, and Vertex files a prediction container's stderr as severity
    # ERROR — so a healthy endpoint reported ~360 errors an hour from its readiness probe
    # alone, and any alert on error rate was worthless before Lab 4 began. The same
    # convention holds on the other two platforms: stdout is INFO, stderr is an error.
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
    # uvicorn ships its own handlers; drop them so every line on stdout is parseable.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uv = logging.getLogger(name)
        uv.handlers = []
        uv.propagate = True
    return logging.getLogger("service")


def _log(level: int, msg: str, **fields: Any) -> None:
    log.log(level, msg, extra={"fields": fields})


class ModelState:
    """What the process knows about the model, and how it came to know it.

    `loads` exists to be asserted on: "loaded once at startup" is a claim a test can
    check, and per-request loading is the commonest cause of a p99 that looks nothing
    like p50.
    """

    def __init__(self) -> None:
        self.model: Any = None
        self.status: str = "loading"          # loading | ready | failed
        self.reason: str = ""
        # MODEL_VERSION may be an alias ("production"). `version` becomes the resolved id
        # once the registry answers, because an alias moves and a response body that says
        # "production" names nothing a rollback can be argued about afterwards.
        self.requested: str = os.environ.get("MODEL_VERSION", "unknown")
        self.version: str = self.requested
        self.source: str = ""
        self.artifact_uri: str = ""
        self.load_seconds: float = 0.0
        self.loads: int = 0

    def ready(self) -> bool:
        return self.status == "ready"


STATE = ModelState()
METRICS = Metrics()
log = _configure_logging()

# Midpoint of every plausible range: a row that is valid by construction and stays valid
# if the schema moves, because both come from src.data.
SELFTEST_ROW = {f: (lo + hi) / 2 for f, (lo, hi) in PLAUSIBLE_RANGES.items()}


def _load_from_registry(name: str, version: str) -> tuple[Any, dict[str, Any]]:
    """Resolve name + version through the adapter, then load what it points at.

    This is the same path `scripts/reload_check.py` proves in Lab 2, and it is why the
    service needs no provider SDK: the adapter returns a URI, MLflow reads it.
    """
    # MLflow writes an advisory line on import and a tqdm progress bar while it downloads
    # artifacts. Both go to the same stream as the logs, and one unparseable line is
    # enough to make a collector drop the batch it arrives in. A terminal is not the
    # audience for a serving container's stdout.
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
    import mlflow.sklearn

    from cloudlayer.factory import get_adapter
    from src import config

    cfg = config.load()
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    entry = get_adapter(cfg).get_model(name, version)
    model = mlflow.sklearn.load_model(entry["artifact_uri"])
    return model, {
        "source": f"registry:{name}",
        "version": str(entry.get("version", version)),
        "artifact_uri": entry.get("artifact_uri", ""),
        "aliases": list(entry.get("aliases", [])),
    }


def _load_from_uri(uri: str) -> tuple[Any, dict[str, Any]]:
    """Load the artifact the platform handed this container.

    Every managed endpoint gives the container its model rather than making it ask the
    registry: SageMaker unpacks it to a local directory, Azure ML sets one, Vertex passes
    a storage URI. The provider env var names live in the serving shim under cloudlayer/;
    by the time it reaches here it is just a URI, local or remote, and MLflow reads both.

    This is still "by version": the endpoint deploys one registered version, and the
    version it is serving arrives as MODEL_VERSION alongside the artifact.
    """
    import mlflow.sklearn

    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
    return mlflow.sklearn.load_model(uri), {
        "source": "platform",
        "version": STATE.requested,
        "artifact_uri": uri,
        "aliases": [],
    }


def _load_from_path(path_str: str) -> tuple[Any, dict[str, Any]]:
    """Development and CI only. A submitted Lab 3 loads a registered version."""
    from pathlib import Path

    import joblib

    path = Path(path_str)
    if not path.exists():
        raise RuntimeError(f"MODEL_PATH={path} does not exist.")
    return joblib.load(path), {
        "source": "file",
        "version": STATE.requested,
        "artifact_uri": str(path),
        "aliases": [],
    }


def _load_model() -> tuple[Any, dict[str, Any]]:
    """Choose the source explicitly, in this order:

        MODEL_PATH   a local file             — development and CI
        MODEL_URI    an artifact handed over  — a managed endpoint, through the shim
        registry     name + version           — a laptop or a VM asking the registry itself

    Explicit rather than fallback: silently dropping from a failed registry lookup to a
    stale local file is how a service ends up serving a model nobody registered."""
    local = os.environ.get("MODEL_PATH")
    if local:
        return _load_from_path(local)

    supplied = os.environ.get("MODEL_URI")
    if supplied:
        return _load_from_uri(supplied)

    from src import config

    cfg = config.load(strict=False)
    name = os.environ.get("MODEL_REGISTRY_NAME") or cfg.model_registry_name
    version = os.environ.get("MODEL_VERSION", "")
    if not (name and version):
        raise RuntimeError(
            "No model source. Set MODEL_VERSION (a version id or an alias) with "
            "MODEL_REGISTRY_NAME, or MODEL_URI for an artifact the platform supplies, "
            "or MODEL_PATH for local development."
        )
    return _load_from_registry(name, version)


def _selftest(model: Any) -> float:
    """Score one row before declaring readiness.

    An artifact can unpickle and still be unable to predict — a feature list that moved, a
    scikit-learn that no longer has the estimator's internals. Finding that here costs one
    prediction; finding it on the first real request costs an incident.
    """
    frame = pd.DataFrame([SELFTEST_ROW], columns=FEATURES)
    return float(model.predict_proba(frame)[0, 1])


def load_once() -> None:
    """Runs on the loader thread. Sets STATE, and never raises into the server."""
    started = time.perf_counter()
    try:
        model, meta = _load_model()
        probability = _selftest(model)
        STATE.model = model
        STATE.loads += 1
        STATE.source = meta["source"]
        STATE.version = str(meta["version"])
        STATE.artifact_uri = meta["artifact_uri"]
        STATE.load_seconds = time.perf_counter() - started
        STATE.status, STATE.reason = "ready", ""
        _log(logging.INFO, "model loaded", source=STATE.source,
             requested_version=STATE.requested, model_version=STATE.version,
             artifact_uri=STATE.artifact_uri, aliases=meta.get("aliases", []),
             load_seconds=round(STATE.load_seconds, 3), selftest_probability=probability,
             loads=STATE.loads)
    except Exception as exc:
        STATE.model = None
        STATE.status, STATE.reason = "failed", f"{type(exc).__name__}: {exc}"
        STATE.load_seconds = time.perf_counter() - started
        # Liveness stays true: the process is fine, and killing it would only restart a
        # container that will fail the same way. Readiness stays false, so no traffic
        # arrives, and the reason is on stdout where the platform's log view shows it.
        _log(logging.ERROR, "model load failed", requested_version=STATE.requested,
             reason=STATE.reason, load_seconds=round(STATE.load_seconds, 3))


@asynccontextmanager
async def lifespan(app: FastAPI):
    thread = threading.Thread(target=load_once, name="model-loader", daemon=True)
    thread.start()
    # How long startup waits for the loader before opening the port.
    #   0  — serve immediately: /health answers alive, /ready answers 503 until the model
    #        is up. This is what the container sets, and what the platform's probes expect.
    #   >0 — block, so a local run and CI are deterministic rather than racing the thread.
    wait_s = float(os.environ.get("MODEL_LOAD_WAIT_S", "30"))
    if wait_s > 0:
        thread.join(wait_s)
    yield
    STATE.model = None


app = FastAPI(title="ITCS355 inference", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def add_request_context(request: Request, call_next):
    """One line per request, and the id that ties it to the caller's own trace."""
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    token, extra_token = REQUEST_ID.set(request_id), EXTRA.set({})
    started = time.perf_counter()

    def elapsed_ms() -> float:
        return round((time.perf_counter() - started) * 1000, 2)

    try:
        response = await call_next(request)
    except BaseException as exc:
        # Nothing should escape the handlers below, but a request that dies here must
        # still leave a line behind: an unlogged request is one nobody can investigate.
        latency = elapsed_ms()
        METRICS.observe_request(request.url.path, 500, latency)
        _log(logging.ERROR, "request failed", method=request.method,
             path=request.url.path, latency_ms=latency,
             error=f"{type(exc).__name__}: {exc}", **_extra())
        raise
    else:
        response.headers["x-request-id"] = request_id
        response.headers["x-model-version"] = STATE.version
        latency = elapsed_ms()
        METRICS.observe_request(request.url.path, response.status_code, latency)
        _log(logging.INFO, "request", method=request.method, path=request.url.path,
             status=response.status_code, latency_ms=latency,
             model_status=STATE.status, **_extra())
        return response
    finally:
        REQUEST_ID.reset(token)
        EXTRA.reset(extra_token)


def _envelope(status_code: int, error: str, detail: Any) -> JSONResponse:
    """Every error body carries the request id and the model version, for the same reason
    every success body does: a client report that quotes neither cannot be traced."""
    return JSONResponse(
        status_code=status_code,
        content={"error": error, "detail": detail,
                 "request_id": REQUEST_ID.get(), "model_version": STATE.version},
    )


def _echo(value: Any) -> Any:
    """What to quote back to the caller as the offending value.

    Echoing it verbatim is how a 101-row batch becomes a 101-row error body — the request
    amplified back at the sender, and into the logs. Scalars come back whole; anything
    larger comes back described.
    """
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    if isinstance(value, str):
        return value if len(value) <= 120 else value[:117] + "..."
    if isinstance(value, (list, tuple)):
        return {"type": "array", "length": len(value)}
    if isinstance(value, dict):
        return {"type": "object", "keys": sorted(value)[:12]}
    return type(value).__name__


@app.exception_handler(RequestValidationError)
async def on_validation_error(request: Request, exc: RequestValidationError):
    """422 with a message a caller can act on: which field, what was wrong, what was sent.

    FastAPI's default body is a list of loc/msg pairs, which is accurate and unreadable at
    3am. Naming the field and echoing the offending value turns a support round-trip into
    a fix.
    """
    detail = [
        {
            "field": ".".join(str(p) for p in err["loc"][1:]) or "body",
            "message": err["msg"],
            "type": err["type"],
            "received": _echo(err.get("input")),
        }
        for err in exc.errors()
    ]
    _extra()["validation_errors"] = [d["field"] for d in detail]
    _log(logging.WARNING, "request rejected", fields_rejected=[d["field"] for d in detail])
    return _envelope(422, "validation_error", detail)


@app.exception_handler(HTTPException)
async def on_http_error(request: Request, exc: HTTPException):
    return _envelope(exc.status_code, "http_error", exc.detail)


@app.exception_handler(Exception)
async def on_unhandled_error(request: Request, exc: Exception):
    """A stack trace belongs in the logs, not in the response body."""
    _log(logging.ERROR, "unhandled error", error=f"{type(exc).__name__}: {exc}")
    return _envelope(500, "internal_error", "prediction failed; see logs for the request id")


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness. The process is up. Says nothing about whether it can serve.

    It deliberately does not touch the model: a liveness probe that fails while a model
    loads gets the container killed and restarted forever.
    """
    return {"status": "alive"}


@app.get("/ready")
def ready():
    """Readiness. The model is loaded and has scored a row.

    These two are genuinely different, and confusing them causes a specific production
    failure: traffic routed to a container whose model has not finished loading. All three
    providers distinguish them, and Quiz 3 asks about it.
    """
    if not STATE.ready():
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "phase": STATE.status,
                     "reason": STATE.reason or "model still loading",
                     "requested_version": STATE.requested},
        )
    return {
        "status": "ready",
        "model_version": STATE.version,
        "requested_version": STATE.requested,
        "source": STATE.source,
        "load_seconds": round(STATE.load_seconds, 3),
    }


def _score(rows: list[dict]) -> list[float]:
    if not STATE.ready():
        raise HTTPException(
            status_code=503,
            detail=f"model not ready ({STATE.status}): {STATE.reason or 'still loading'}",
        )
    _extra()["rows"] = len(rows)
    frame = pd.DataFrame(rows, columns=FEATURES)
    scores = [float(p) for p in STATE.model.predict_proba(frame)[:, 1]]
    # After scoring, so the window holds only inputs the model actually answered — a row
    # that failed validation never reaches here, and is counted as a 4xx instead.
    METRICS.observe_rows(rows)
    return scores


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    """Prometheus scrape target. Unauthenticated by design on a private port; the managed
    endpoints expose only the predict and health routes, so this never faces the internet
    unless someone routes it there."""
    return METRICS.render(STATE.version)


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    score = _score([payload.model_dump()])[0]
    return PredictResponse(probability=score, model_version=STATE.version,
                           request_id=REQUEST_ID.get())


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(payload: BatchRequest) -> BatchResponse:
    scores = _score([row.model_dump() for row in payload.rows])
    return BatchResponse(probabilities=scores, model_version=STATE.version,
                         request_id=REQUEST_ID.get())
