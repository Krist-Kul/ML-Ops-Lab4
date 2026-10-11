"""The portability seam.

Eleven methods as issued. Every managed ML platform sells you the same operations under
different names; writing this once is the difference between knowing a product and
knowing the category. `generate` is the newest of them and the one the vendors are
currently busiest renaming.

Lab 2 added two more — `get_model` and `promote_model`. Reading a version back and moving
an alias are as provider-specific as writing one, and leaving them out would have pushed a
`gs://` path into scripts/, which is the leak this file exists to prevent.

You implement exactly ONE of aws.py, azure.py, or gcp.py. Lab 1 needs only `upload`,
`download`, and `push_image`. The rest raise NotImplementedError until the lab that
needs them, which is deliberate — do not implement ahead.

Nothing outside cloudlayer/ may import a provider SDK.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class CloudAdapter(ABC):
    """Provider-neutral interface. The grader only ever calls these."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    # --- Lab 1 ---------------------------------------------------------------
    @abstractmethod
    def upload(self, local_path: str, key: str) -> str:
        """Upload a file under BLOB_URI. Returns the full URI of the stored object."""

    @abstractmethod
    def download(self, uri: str, local_path: str) -> None:
        """Fetch an object to a local path. Creates parent directories."""

    @abstractmethod
    def push_image(self, local_tag: str) -> str:
        """Push a locally built image to CONTAINER_REGISTRY. Returns the remote reference,
        which must be digest-pinned (repo@sha256:...), not tag-pinned."""

    # --- Lab 2 ---------------------------------------------------------------
    def submit_training(self, image_uri: str, args: dict[str, Any]) -> str:
        raise NotImplementedError("Lab 2")

    def wait_training(self, job_id: str) -> dict[str, Any]:
        raise NotImplementedError("Lab 2")

    def register_model(self, model_uri: str, name: str,
                       lineage: dict[str, str] | None = None) -> str:
        """Register the model stored at `model_uri` under `name`. Returns the version.

        `lineage` is the eight fields Lab 2 requires, and they belong on the VERSION, not
        on the tracking run: a run records how a model was made, a version is what gets
        promoted, and at promotion time the run may not be the thing in front of you.
        """
        raise NotImplementedError("Lab 2")

    def get_model(self, name: str, version: str) -> dict[str, Any]:
        """The read side of register_model: resolve name + version to at least

            {"artifact_uri": str, "aliases": [str], "lineage": {...}}

        Without this the registry is write-only, and `scripts/reload_check.py` would have
        to fall back to a path someone remembered — which is the failure the lab is about.
        """
        raise NotImplementedError("Lab 2")

    def promote_model(self, name: str, version: str, alias: str) -> dict[str, Any]:
        """Move `alias` (staging, production, ...) to this version. Returns the version's
        aliases after the move. Promotion is a separate operation from registration on
        purpose: registering is a record, promoting is a decision."""
        raise NotImplementedError("Lab 2")

    # --- Lab 3 ---------------------------------------------------------------
    def serving_spec(self, env: dict[str, str] | None = None) -> dict[str, Any]:
        """How THIS provider's endpoint talks to the serving container: the routes it
        calls, the port it calls them on, and the command that starts the process.

        Added in Lab 3 for the same reason `get_model` was added in Lab 2. The three
        providers disagree about exactly this — SageMaker insists on /ping and
        /invocations, Azure ML wants a scoring script and its own liveness route, Vertex
        makes you declare both on the model resource — and the alternative was a
        provider's route names sitting in `scripts/`, which is the leak the seam exists
        to prevent. The container image stays identical across all three.
        """
        raise NotImplementedError("Lab 3")

    def deploy(self, model_ref: str, endpoint: str, instance: str) -> str:
        raise NotImplementedError("Lab 3")

    def invoke(self, endpoint: str, payload: dict[str, Any],
               timeout_s: float = 30.0) -> dict[str, Any]:
        """Call the endpoint once, with a deadline.

        `timeout_s` is not optional decoration. An unbounded predict call inside a loop that
        owns a billed resource is a cost bug: Lab 3 Task 4 lost 19 hours and about 194 THB to
        one that never returned after the host slept.
        """
        raise NotImplementedError("Lab 3")

    def list_deployments(self, endpoint: str) -> list[dict[str, Any]]:
        """What is deployed on this endpoint, with each one's traffic share."""
        raise NotImplementedError("Lab 3")

    def undeploy(self, endpoint: str, deployed_model_id: str) -> str:
        """Detach one deployed model and release its replicas.

        Separate from moving traffic, because the two are separate on every provider and
        the difference is the bill: a deployment at 0% traffic keeps its machines.
        """
        raise NotImplementedError("Lab 3")

    def split_traffic(self, endpoint: str, split: dict[str, int]) -> dict[str, int]:
        """Set the traffic share of deployments that are ALREADY attached. Returns the split.

        `deploy(traffic=N)` can only set a share while attaching something new, which covers
        starting a canary and nothing else. Changing the split of two live deployments — the
        rollback — needs this, and so does taking a canary to 0% before detaching it.

        `split` maps deployment id to percent and must total 100: every provider rejects a
        split that does not, and rejecting it here gives a readable error instead of theirs.

        Moving traffic is NOT the same as releasing machines. A deployment left at 0% keeps
        its replicas and keeps billing, so a rollback is this call followed by `undeploy`.
        The vocabulary differs — production variant weights, deployment traffic percentages,
        traffic split across deployed models — and the concept does not.
        """
        raise NotImplementedError("Lab 3")

    # --- Lab 4 ---------------------------------------------------------------
    def emit_metric(self, name: str, value: float, unit: str = "None",
                    labels: dict[str, str] | None = None) -> None:
        """Write one point of a custom metric. `labels` splits one metric into series —
        drift_psi{feature=temp_c} — so an alert can be "any series above X" rather than
        one alert per feature."""
        raise NotImplementedError("Lab 4")

    def download_dir(self, uri: str, local_dir: str) -> int:
        """Copy every object under `uri` into `local_dir`, keeping relative paths. Returns
        the number of files. A registered model is a directory, not one object."""
        raise NotImplementedError("Lab 4")

    def delete_prefix(self, key_prefix: str) -> int:
        """Delete every object under BLOB_URI/key_prefix. Returns the number deleted."""
        raise NotImplementedError("Lab 4")

    def ensure_metric_alert(self, metric: str, thresholds: dict[str, float], email: str,
                            tags: dict[str, str], label: str = "feature") -> dict[str, str]:
        """Create, or update in place, an alert that fires when the series of `metric`
        whose `label` is k exceeds thresholds[k], for any k, notifying `email`. Idempotent: running it twice must not
        produce two alerts, or the second email teaches people to ignore the first."""
        raise NotImplementedError("Lab 4")

    # --- Lab 5 ---------------------------------------------------------------
    def generate(self, prompt: str, params: dict[str, Any]) -> dict[str, Any]:
        """Call a managed LLM endpoint once. Returns at least:

            {"response": str, "input_tokens": int,
             "output_tokens": int, "latency_ms": int}

        Token counts must be read from the provider's own usage fields. Estimating them
        by counting words produces a cost report built on a number you made up, and
        every provider tokenises differently.

        `params` carries the knobs worth exposing at the seam — `max_output_tokens`
        above all, since output tokens dominate the bill. Provider-specific options
        belong here in Layer 3, never in the caller.
        """
        raise NotImplementedError("Lab 5")

    def teardown(self, tags: dict[str, str]) -> list[str]:
        """Delete every resource carrying these tags. Returns what was deleted.

        Deletion is asynchronous on all three providers — returning successfully does
        not mean the resource is gone. Re-check, and check the bill.
        """
        raise NotImplementedError("Lab 5")


class LocalAdapter(CloudAdapter):
    """Filesystem stand-in so Lab 1 runs before your cloud account is ready.

    Acceptable for developing Lab 1. NOT acceptable for submission — your submitted
    Lab 1 must upload to real object storage and push to a real registry.
    """

    def upload(self, local_path: str, key: str) -> str:
        import shutil
        from pathlib import Path

        dest = Path(self.cfg.data_dir) / "_local_blob" / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dest)
        return f"local://{dest}"

    def download(self, uri: str, local_path: str) -> None:
        import shutil
        from pathlib import Path

        src = uri.removeprefix("local://")
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, local_path)

    def push_image(self, local_tag: str) -> str:
        raise NotImplementedError(
            "LocalAdapter cannot push images. Implement your provider's adapter for Lab 1 submission."
        )
