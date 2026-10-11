"""GCP adapter. Implements upload/download/push_image for Lab 1 and
submit_training/wait_training (Vertex AI custom jobs) for Lab 2.

SDK:  pip install google-cloud-storage google-cloud-aiplatform
Docs: storage.Client for GCS; Artifact Registry push goes through `docker push` after
      `gcloud auth configure-docker <region>-docker.pkg.dev`.

Layer 2 rules this file obeys:
  * BLOB_URI looks like gs://bucket/prefix — parsed here, never in src/.
  * Artifact Registry paths are region-scoped:
        <region>-docker.pkg.dev/<project>/<repo>/<image>
    A common first failure is pushing to gcr.io out of habit; it is a different service.
  * push_image returns the digest reference, not the tag.
  * GCP calls them labels, not tags, and they must be lowercase with no spaces.
    cfg.tags(1) already satisfies that constraint — the values are used as they stand.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from cloudlayer.base import CloudAdapter
from src import traffic as traffic_split


def _split_gs_uri(uri: str) -> tuple[str, str]:
    """gs://bucket/some/prefix -> ('bucket', 'some/prefix'). Prefix may be empty."""
    if not uri.startswith("gs://"):
        raise ValueError(f"Not a GCS URI: {uri!r}. Expected gs://bucket[/prefix].")
    bucket, _, prefix = uri[len("gs://"):].partition("/")
    if not bucket:
        raise ValueError(f"No bucket in {uri!r}.")
    return bucket, prefix.strip("/")


def _run(cmd: list[str]) -> str:
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{out.stderr.strip()}")
    return out.stdout.strip()


class GcpAdapter(CloudAdapter):
    def _bucket(self):
        from google.cloud import storage

        bucket_name, prefix = _split_gs_uri(self.cfg.blob_uri)
        client = storage.Client(project=self.cfg.project_id)
        return client.bucket(bucket_name), prefix

    def upload(self, local_path: str, key: str) -> str:
        bucket, prefix = self._bucket()
        blob_name = f"{prefix}/{key.lstrip('/')}" if prefix else key.lstrip("/")
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(local_path)
        return f"gs://{bucket.name}/{blob_name}"

    def download(self, uri: str, local_path: str) -> None:
        from google.cloud import storage

        bucket_name, blob_name = _split_gs_uri(uri)
        if not blob_name:
            raise ValueError(f"No object path in {uri!r}.")
        target = Path(local_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        client = storage.Client(project=self.cfg.project_id)
        client.bucket(bucket_name).blob(blob_name).download_to_filename(str(target))

    def download_dir(self, uri: str, local_dir: str) -> int:
        from google.cloud import storage

        bucket_name, prefix = _split_gs_uri(uri)
        client = storage.Client(project=self.cfg.project_id)
        count = 0
        for blob in client.list_blobs(bucket_name, prefix=prefix.rstrip("/") + "/"):
            if blob.name.endswith("/"):
                continue
            target = Path(local_dir) / blob.name[len(prefix):].lstrip("/")
            target.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(target))
            count += 1
        return count

    def delete_prefix(self, key_prefix: str) -> int:
        bucket, prefix = self._bucket()
        full = f"{prefix}/{key_prefix.lstrip('/')}" if prefix else key_prefix.lstrip("/")
        blobs = list(bucket.client.list_blobs(bucket.name, prefix=full))
        for blob in blobs:
            blob.delete()
        return len(blobs)

    def push_image(self, local_tag: str) -> str:
        registry = self.cfg.container_registry.rstrip("/")
        host = registry.split("/", 1)[0]
        remote = f"{registry}/{local_tag.rsplit('/', 1)[-1]}"

        # Idempotent, and cheap if the helper is already configured.
        _run(["gcloud", "auth", "configure-docker", host, "--quiet"])
        _run(["docker", "tag", local_tag, remote])
        _run(["docker", "push", remote])

        # The tag we just pushed can be moved later; the digest cannot. Read it back
        # from the registry rather than trusting the local daemon's view.
        repo = remote.rsplit(":", 1)[0]
        digest = _run([
            "gcloud", "artifacts", "docker", "images", "describe", remote,
            "--project", self.cfg.project_id, "--format", "value(image_summary.digest)",
        ])
        return f"{repo}@{digest}"

    # --- Lab 2 ---------------------------------------------------------------
    # Vertex custom jobs through the low-level v1 client. The high-level
    # aiplatform.CustomContainerTrainingJob wrapper wants a staging bucket and uploads a
    # model on success; neither is wanted here.

    def _jobs(self):
        from google.cloud import aiplatform_v1

        return aiplatform_v1.JobServiceClient(
            client_options={"api_endpoint": f"{self.cfg.region}-aiplatform.googleapis.com"})

    def _mounted(self, key: str) -> str:
        """Where BLOB_URI/<key> appears inside a Vertex job. Vertex mounts buckets at
        /gcs/<bucket> with Cloud Storage FUSE, so the image reads and writes object storage
        as plain files and needs no storage SDK."""
        bucket, prefix = _split_gs_uri(self.cfg.blob_uri)
        return "/".join(p for p in ("/gcs", bucket, prefix, key.strip("/")) if p)

    def submit_training(self, image_uri: str, args: dict[str, Any]) -> str:
        """args: job_name, data_key, output_key (both relative to BLOB_URI), and optionally
        command (replaces the image entrypoint), args (extra container args), instance,
        spot, env, timeout_s, lab. Returns the job resource name."""
        from google.cloud import aiplatform_v1 as v
        from google.protobuf import duration_pb2

        container = v.ContainerSpec(
            image_uri=image_uri,
            command=args.get("command", []),
            args=[*args.get("args", []),
                  "--data-path", self._mounted(args["data_key"]),
                  "--artifact-dir", self._mounted(args["output_key"])],
            env=[v.EnvVar(name=k, value=str(val)) for k, val in args.get("env", {}).items()],
        )
        pool = v.WorkerPoolSpec(
            replica_count=1, container_spec=container,
            machine_spec=v.MachineSpec(machine_type=args.get("instance", "n1-standard-4")),
        )
        # The run-time identity. Without one Vertex falls back to its own service agent,
        # which is not the identity you granted bucket access to.
        identity = self.cfg.identity_ref.removeprefix("serviceAccount:")
        # The timeout is a spend cap: a hung job is killed instead of billed until noticed.
        # On spot, a reclaimed VM restarts the container, and the job resumes from its
        # checkpoint rather than failing.
        spot = bool(args.get("spot"))
        scheduling = v.Scheduling(
            timeout=duration_pb2.Duration(seconds=int(args.get("timeout_s", 3600))),
            strategy=v.Scheduling.Strategy.SPOT if spot else v.Scheduling.Strategy.STRATEGY_UNSPECIFIED,
            restart_job_on_worker_restart=spot,
        )
        spec = v.CustomJobSpec(
            worker_pool_specs=[pool], scheduling=scheduling,
            service_account=identity if identity.endswith(".gserviceaccount.com") else "",
        )
        job = v.CustomJob(display_name=args["job_name"], job_spec=spec,
                          labels=self.cfg.tags(args.get("lab", 2)))
        parent = f"projects/{self.cfg.project_id}/locations/{self.cfg.region}"
        return self._jobs().create_custom_job(parent=parent, custom_job=job).name

    def wait_training(self, job_id: str, poll_s: float = 20.0) -> dict[str, Any]:
        import time

        from google.cloud import aiplatform_v1 as v

        S = v.JobState
        terminal = {S.JOB_STATE_SUCCEEDED, S.JOB_STATE_FAILED, S.JOB_STATE_CANCELLED,
                    S.JOB_STATE_EXPIRED, S.JOB_STATE_PARTIALLY_SUCCEEDED}
        client, last = self._jobs(), None
        while True:
            job = client.get_custom_job(name=job_id)
            if job.state != last:
                print(f"  {time.strftime('%H:%M:%S')}  {job.state.name}", flush=True)
                last = job.state
            if job.state in terminal:
                break
            time.sleep(poll_s)

        raw = v.CustomJob.pb(job)
        ran = raw.HasField("start_time") and raw.HasField("end_time")
        pool = job.job_spec.worker_pool_specs[0]
        return {
            "job_id": job.name,
            "state": job.state.name.removeprefix("JOB_STATE_").lower(),
            "succeeded": job.state == S.JOB_STATE_SUCCEEDED,
            "error": job.error.message,
            # RUNNING to end. Provisioning before start_time is excluded, so treat cost
            # derived from this as a floor and reconcile it against the bill.
            "duration_s": (job.end_time - job.start_time).total_seconds() if ran else 0.0,
            "instance": pool.machine_spec.machine_type,
            "spot": job.job_spec.scheduling.strategy == v.Scheduling.Strategy.SPOT,
            "logs": f"gcloud ai custom-jobs stream-logs {job.name.rsplit('/', 1)[-1]} "
                    f"--region {self.cfg.region} --project {self.cfg.project_id}",
        }

    # --- Lab 2: Vertex Model Registry ---------------------------------------
    # A "registered model" in Vertex is one Model resource with many versions; each
    # version is addressed as <model resource>@<version_id>. Lineage goes on the version:
    # labels for the fields that survive GCP's label rules (lowercase, [a-z0-9_-], 63
    # chars, no ':') and version_description for the full values, because a sha256 digest
    # is 71 characters with its prefix and no label will hold it.

    LINEAGE_FIELDS = ("git_commit", "data_version", "mlflow_run_id", "training_job_id",
                      "image_digest", "seed", "metric_val", "metric_test")

    def _models(self):
        from google.cloud import aiplatform_v1

        return aiplatform_v1.ModelServiceClient(
            client_options={"api_endpoint": f"{self.cfg.region}-aiplatform.googleapis.com"})

    def _registry_parent(self) -> str:
        return f"projects/{self.cfg.project_id}/locations/{self.cfg.region}"

    def _find_model(self, name: str) -> str | None:
        """Resource name of the registered model called `name`, or None."""
        from google.cloud import aiplatform_v1 as v

        models = list(self._models().list_models(v.ListModelsRequest(
            parent=self._registry_parent(), filter=f'display_name="{name}"')))
        if not models:
            return None
        # The list returns the default version of each model; strip any @version suffix.
        return models[0].name.split("@", 1)[0]

    @staticmethod
    def _label_value(value: str) -> str:
        """GCP labels: lowercase letters, digits, dash, underscore; 63 characters.
        Anything else is dropped rather than mangled into something that looks real."""
        import re

        out = re.sub(r"[^a-z0-9_-]", "-", str(value).lower())
        return out[:63]

    def register_model(self, model_uri: str, name: str,
                       lineage: dict[str, str] | None = None,
                       serving_image: str | None = None,
                       serving: dict[str, Any] | None = None) -> str:
        """Upload the model directory at `model_uri` (gs://...) as a new version of the
        registered model `name`. Returns the version id, e.g. "1".

        Vertex refuses a model with no container_spec ("container_spec must be set for
        custom model"), so a version always names an environment. The default is the
        image that trained it — measured reason: the newest prebuilt sklearn serving
        container is sklearn-cpu.1-6 and this model is pickled by scikit-learn 1.8, so a
        prebuilt container would record an environment that cannot load the artifact.
        `serving` is Lab 3's addition: the routes, port, command and environment Vertex
        needs to call the container. It is provider shape — predict_route, health_route,
        ports — and it is why this argument exists here rather than in scripts/."""
        import json

        from google.cloud import aiplatform_v1 as v

        lineage = {k: str(lineage.get(k, "")) for k in self.LINEAGE_FIELDS} if lineage else {}
        missing = [k for k, val in lineage.items() if not val]
        if lineage and missing:
            raise ValueError(f"lineage is incomplete: {missing}. A half-traced model is "
                             "worse than an untraced one, because it looks traced.")

        image = serving_image or lineage.get("image_digest", "")
        if not image:
            raise ValueError("no serving_image and no image_digest in lineage: Vertex "
                             "requires a container for every registered version.")
        model = v.Model(
            display_name=name,
            artifact_uri=model_uri,
            container_spec=self._container_spec(image, serving),
            # Lineage twice, on purpose: labels are queryable in the console and in
            # `gcloud ai models list --filter`, the description holds the untruncated
            # values a rebuild actually needs.
            labels={**self.cfg.tags(2),
                    **{k: self._label_value(val) for k, val in lineage.items()}},
            version_description=json.dumps(lineage, sort_keys=True) if lineage else "",
        )
        parent_model = self._find_model(name)
        request = v.UploadModelRequest(parent=self._registry_parent(), model=model,
                                       parent_model=parent_model or None)
        result = self._models().upload_model(request=request).result()
        return result.model_version_id

    def _container_spec(self, image: str, serving: dict[str, Any] | None):
        """The container a registered version names.

        Without `serving` this is just an image, which is all Lab 2 needed: Vertex refuses
        a version with no container, but nothing ever calls it. With it, the version is
        deployable — Vertex will POST to `predict_route` and poll `health_route` on
        `ports[0]`, and passes all three back to the container as AIP_* variables.
        """
        from google.cloud import aiplatform_v1 as v
        from google.protobuf import duration_pb2

        if not serving:
            return v.ModelContainerSpec(image_uri=image)
        return v.ModelContainerSpec(
            image_uri=image,
            command=list(serving.get("command", [])),
            args=list(serving.get("args", [])),
            env=[v.EnvVar(name=k, value=str(val))
                 for k, val in serving.get("env", {}).items()],
            ports=[v.Port(container_port=int(serving.get("port", 8080)))],
            predict_route=serving.get("predict_route", "/v1/predict"),
            health_route=serving.get("health_route", "/ready"),
            # How long a replica has to answer the health route before Vertex gives up.
            # The model is downloaded and unpickled inside that window, so it has to be
            # bigger than the cold start, not bigger than the process start.
            deployment_timeout=duration_pb2.Duration(
                seconds=int(serving.get("deployment_timeout_s", 900))),
        )

    def get_model(self, name: str, version: str) -> dict[str, Any]:
        """Resolve name + version to the artifact and the lineage recorded with it.
        `version` may be a version id ("1") or an alias ("production")."""
        import json

        resource = self._find_model(name)
        if resource is None:
            raise LookupError(f"No registered model named {name!r} in "
                              f"{self._registry_parent()}. Run `make register` first.")
        model = self._models().get_model(name=f"{resource}@{version}")
        try:
            recorded = json.loads(model.version_description or "{}")
        except json.JSONDecodeError:  # written by hand, or by an earlier version of this code
            recorded = {"version_description": model.version_description}
        return {
            "name": name,
            "resource": model.name,
            "version": model.version_id,
            "artifact_uri": model.artifact_uri,
            "aliases": list(model.version_aliases),
            "lineage": recorded,
            "labels": dict(model.labels),
            "created": model.version_create_time.rfc3339(),
        }

    def promote_model(self, name: str, version: str, alias: str) -> dict[str, Any]:
        """Point `alias` at this version, taking it off whichever version holds it now.
        Vertex aliases are unique per model, so the move is the API's own behaviour."""
        resource = self._find_model(name)
        if resource is None:
            raise LookupError(f"No registered model named {name!r} to promote.")
        model = self._models().get_model(name=f"{resource}@{version}")
        self._models().merge_version_aliases(
            name=f"{resource}@{model.version_id}", version_aliases=[alias])
        return self.get_model(name, model.version_id)

    # --- Lab 3: Vertex Endpoints --------------------------------------------
    # An Endpoint is the billed object: it owns replicas and a traffic split, and it keeps
    # costing money whether or not anyone calls it. A DeployedModel is one model version
    # attached to an endpoint with its own machines. Deploying twice to one endpoint is
    # what makes Task 4's canary a traffic-split change rather than a redeploy.

    def serving_spec(self, env: dict[str, str] | None = None) -> dict[str, Any]:
        """Vertex's half of the serving contract.

        The image's own CMD runs the neutral app, which knows nothing about Vertex. Here
        the command is replaced with the shim in `cloudlayer/gcp_serving.py`, which speaks
        the {"instances": [...]} envelope and translates AIP_STORAGE_URI into the
        neutral MODEL_URI. Same image, same bytes, provider-shaped entrypoint.

        Health is readiness, not liveness. Pointing Vertex at /health would mean traffic
        arrives as soon as the process is up and fails until the model finishes loading —
        the exact bug the two routes exist to prevent.
        """
        return {
            "command": ["uvicorn"],
            "args": ["cloudlayer.gcp_serving:app", "--host", "0.0.0.0", "--port", "8080",
                     "--workers", "1"],
            "env": dict(env or {}),
            "port": 8080,
            "predict_route": "/v1/predict",
            "health_route": "/ready",
            "deployment_timeout_s": 900,
        }

    def _endpoints(self):
        from google.cloud import aiplatform_v1

        return aiplatform_v1.EndpointServiceClient(
            client_options={"api_endpoint": f"{self.cfg.region}-aiplatform.googleapis.com"})

    def _find_endpoint(self, name: str) -> str | None:
        """Resource name of the endpoint called `name`, or None. `name` may already be a
        full resource name, in which case it is returned unchanged."""
        from google.cloud import aiplatform_v1 as v

        if name.startswith("projects/"):
            return name
        found = list(self._endpoints().list_endpoints(v.ListEndpointsRequest(
            parent=self._registry_parent(), filter=f'display_name="{name}"')))
        return found[0].name if found else None

    def _service_account(self) -> str:
        """The endpoint's run-time identity, or "" to accept the Vertex service agent.

        The container reads the model artifact from the bucket, so this is the identity
        that needs object read there — not the CLI identity that ran the deploy.
        """
        identity = self.cfg.identity_ref.removeprefix("serviceAccount:")
        return identity if identity.endswith(".gserviceaccount.com") else ""

    def deploy(self, model_ref: str, endpoint: str, instance: str,
               min_replicas: int = 1, max_replicas: int = 1, spot: bool = False,
               traffic: int = 100, lab: int = 3, timeout_s: int = 2400) -> str:
        """Deploy `model_ref` ("<name>@<version>", or "<name>" for the default version) to
        the endpoint called `endpoint`, creating it if it does not exist. Returns the
        endpoint resource name.

        `traffic` is the share this deployment takes. The remainder is divided among the
        deployments already attached, in proportion to what they hold now. 100 is the Task 2
        case — first model, all the traffic — and it takes every other deployment to 0%.

        The remainder has to be computed and sent explicitly. Vertex rejects a traffic_split
        that does not total 100 outright, with "The traffic percentage values must add up to
        100": it does not treat the map as a patch that leaves unnamed deployments alone.
        This method used to send only {"0": traffic} and claim the rest stayed where it was,
        which worked for the only case Task 2 exercised — the first deployment, taking
        everything — and failed the first time a canary asked for 10%.
        """
        from google.cloud import aiplatform_v1 as v

        name, _, version = model_ref.partition("@")
        resource = self._find_model(name)
        if resource is None:
            raise LookupError(f"No registered model named {name!r}. Run `make register`.")
        model = self._models().get_model(name=f"{resource}@{version or 'default'}")

        endpoints = self._endpoints()
        target = self._find_endpoint(endpoint)
        if target is None:
            created = endpoints.create_endpoint(
                parent=self._registry_parent(),
                endpoint=v.Endpoint(display_name=endpoint, labels=self.cfg.tags(lab)),
            ).result(timeout=timeout_s)
            target = created.name

        deployed = v.DeployedModel(
            model=f"{resource}@{model.version_id}",
            display_name=f"{name}-v{model.version_id}",
            dedicated_resources=v.DedicatedResources(
                machine_spec=v.MachineSpec(machine_type=instance),
                min_replica_count=min_replicas,
                max_replica_count=max_replicas,
                # Spot replicas are evicted without warning. Cheap for a demo, dishonest
                # under a load test: the p99 you measure is then the platform's eviction
                # behaviour, not your service's.
                spot=spot,
            ),
            service_account=self._service_account(),
        )
        # "0" is how the API names the model being deployed by this request; every other key
        # is an existing deployment id. The whole map is replaced, so it must total 100.
        response = endpoints.deploy_model(
            endpoint=target, deployed_model=deployed,
            traffic_split=self._split_for_new_deployment(target, int(traffic)),
        ).result(timeout=timeout_s)
        self._last_deployed_model_id = response.deployed_model.id
        return target

    def _split_for_new_deployment(self, endpoint_resource: str, traffic: int) -> dict[str, int]:
        """The complete traffic split to send with a deploy that takes `traffic` percent.

        Reads what is attached and hands the division to `src.traffic`, which is where the
        arithmetic lives because the arithmetic is not provider-specific. What IS specific is
        that the map's key for the deployment being created is its index in this request.
        """
        live = self._endpoints().get_endpoint(name=endpoint_resource)
        return traffic_split.split_for_new_deployment(
            current=dict(live.traffic_split),
            deployment_ids=[d.id for d in live.deployed_models],
            traffic=traffic)

    def list_deployments(self, endpoint: str) -> list[dict[str, Any]]:
        """What is attached to this endpoint, and what share of traffic each one has.

        Traffic and replicas are different facts, and conflating them is expensive: a
        deployment at 0% traffic still has its machines, and still bills for them.
        """
        target = self._find_endpoint(endpoint)
        if target is None:
            raise LookupError(f"No endpoint named {endpoint!r}.")
        live = self._endpoints().get_endpoint(name=target)
        split = dict(live.traffic_split)
        return [{
            "deployed_model_id": deployed.id,
            "display_name": deployed.display_name,
            "model": deployed.model,
            "model_version_id": deployed.model_version_id,
            "traffic_percent": split.get(deployed.id, 0),
            "instance": deployed.dedicated_resources.machine_spec.machine_type,
            "min_replicas": deployed.dedicated_resources.min_replica_count,
            "max_replicas": deployed.dedicated_resources.max_replica_count,
            "created": deployed.create_time.rfc3339() if deployed.create_time else "",
        } for deployed in live.deployed_models]

    def undeploy(self, endpoint: str, deployed_model_id: str, timeout_s: int = 1800) -> str:
        """Detach one deployed model and release its replicas. Returns its id.

        This is the call that stops the meter for that deployment. Moving traffic to 0
        does not: Vertex keeps the machines, ready to take traffic back, and keeps
        charging for them.
        """
        target = self._find_endpoint(endpoint)
        if target is None:
            raise LookupError(f"No endpoint named {endpoint!r}.")
        self._endpoints().undeploy_model(
            endpoint=target, deployed_model_id=deployed_model_id).result(timeout=timeout_s)
        return deployed_model_id

    def split_traffic(self, endpoint: str, split: dict[str, int]) -> dict[str, int]:
        """Set the traffic split across deployments already attached to this endpoint.

        Vertex has no call that changes one deployment's share; the split is a field on the
        Endpoint and is replaced wholesale, so every deployment that should keep traffic has
        to appear in `split`. Anything omitted goes to 0% — which is why this validates the
        total rather than trusting the caller, and why the ids are checked against what is
        actually attached: a typo in a deployment id would otherwise silently route 100% of
        production traffic to nothing.
        """
        from google.api_core import exceptions as gexc
        from google.cloud import aiplatform_v1 as v
        from google.protobuf import field_mask_pb2

        target = self._find_endpoint(endpoint)
        if target is None:
            raise LookupError(f"No endpoint named {endpoint!r}.")

        # Validated before the call so a bad split is a readable local error rather than a
        # provider 400, and so the two failures that matter — a total that is not 100, and an
        # id that is not attached — are named for what they are.
        traffic_split.validate(
            split, attached={d["deployed_model_id"] for d in self.list_deployments(endpoint)})

        try:
            live = self._endpoints().update_endpoint(
                endpoint=v.Endpoint(name=target, traffic_split={k: int(x) for k, x in split.items()}),
                update_mask=field_mask_pb2.FieldMask(paths=["traffic_split"]),
            )
        except gexc.GoogleAPICallError as exc:  # pragma: no cover - needs a live endpoint
            raise RuntimeError(f"Vertex refused the split {split}: {exc.message}") from exc
        return dict(live.traffic_split)

    def _prediction_client(self):
        """One PredictionServiceClient, reused.

        Built per call this used to add a gRPC channel and a TLS handshake to every request.
        That is invisible on a three-payload smoke test and ruinous on a canary run of a few
        thousand: it caps throughput, and because `invoke` reports the round trip it goes
        looks like it belongs to the service — a latency signal built on it would be
        measuring channel setup. Cached under a lock because the canary driver calls `invoke`
        from a thread pool; the client itself is documented as thread-safe.
        """
        import threading

        from google.cloud import aiplatform_v1 as v

        existing = getattr(self, "_prediction_client_cache", None)
        if existing is not None:
            return existing
        lock = self.__dict__.setdefault("_prediction_client_lock", threading.Lock())
        with lock:
            existing = getattr(self, "_prediction_client_cache", None)
            if existing is None:
                existing = v.PredictionServiceClient(client_options={
                    "api_endpoint": f"{self.cfg.region}-aiplatform.googleapis.com"})
                self._prediction_client_cache = existing
        return existing

    def _resolve_endpoint_cached(self, name: str) -> str | None:
        """`_find_endpoint`, memoised for the hot path.

        Resolving a display name costs a list_endpoints call, and on the canary driver's few
        thousand invocations that doubles the API traffic and the wall clock for a lookup
        whose answer does not change. Only `invoke` uses this; deploy, undeploy, split and
        teardown keep the uncached lookup, because those are the operations that create and
        destroy the thing being resolved and a stale answer there would be acted on. If an
        endpoint is deleted underneath a cached name the next call fails NotFound, which is
        the loud failure rather than the quiet one.
        """
        cache = self.__dict__.setdefault("_endpoint_name_cache", {})
        if name not in cache:
            cache[name] = self._find_endpoint(name)
        return cache[name]

    def invoke(self, endpoint: str, payload: dict[str, Any],
               timeout_s: float = 30.0) -> dict[str, Any]:
        """Call the endpoint once. `payload` is either one row, or {"instances": [...]}.

        Returns the predictions plus what the platform says about who answered — the
        deployed model id and the model version id. Those come from Vertex, not from the
        container, which is what makes them usable as evidence in Task 4: a container can
        claim any version it likes.
        """
        import time

        from google.cloud import aiplatform_v1 as v
        from google.protobuf import json_format, struct_pb2

        target = self._resolve_endpoint_cached(endpoint)
        if target is None:
            raise LookupError(f"No endpoint named {endpoint!r} in {self._registry_parent()}.")

        rows = payload.get("instances") if isinstance(payload, dict) and "instances" in payload \
            else [payload]
        instances = [json_format.ParseDict(row, struct_pb2.Value()) for row in rows]

        client = self._prediction_client()
        started = time.perf_counter()
        # A deadline, because without one this call can block forever and did: during Task 4
        # the laptop slept mid-run, the gRPC connections died under it, and a predict call
        # never returned or raised. The driver process stayed alive holding a canary up for
        # 19 hours at 10.18 THB/h. An unbounded network call inside a loop that owns a billed
        # resource is a cost bug, not a reliability nicety.
        response = client.predict(endpoint=target, instances=instances, timeout=timeout_s)
        latency_ms = (time.perf_counter() - started) * 1000

        # Convert the whole response rather than each prediction: proto-plus has already
        # turned the repeated Values into its own map/list wrappers, which MessageToDict
        # cannot take one at a time. Going through the raw message gives plain JSON.
        raw = json_format.MessageToDict(v.PredictResponse.pb(response))
        return {
            "endpoint": target,
            "predictions": raw.get("predictions", []),
            "deployed_model_id": response.deployed_model_id,
            "model_version_id": response.model_version_id,
            "model": response.model,
            # Round trip from this machine, so it includes the internet. The endpoint's own
            # latency is what the load test measures; this number is for smoke tests.
            "latency_ms": round(latency_ms, 2),
        }

    def teardown(self, tags: dict[str, str]) -> list[str]:
        """Delete every endpoint carrying these labels, undeploying its models first.

        Vertex refuses to delete an endpoint that still has traffic on it, which is the
        API telling you the truth: the billed thing is the replicas, and they go first.
        Registered models and their artifacts are left alone — they cost storage, not
        compute, and Lab 4 needs them.
        """
        from google.cloud import aiplatform_v1 as v

        selector = " AND ".join(f'labels.{k}="{val}"' for k, val in tags.items())
        endpoints = self._endpoints()
        removed: list[str] = []
        for endpoint in endpoints.list_endpoints(v.ListEndpointsRequest(
                parent=self._registry_parent(), filter=selector)):
            for deployed in endpoint.deployed_models:
                endpoints.undeploy_model(
                    endpoint=endpoint.name, deployed_model_id=deployed.id,
                ).result(timeout=1800)
                removed.append(f"deployed_model {deployed.id} ({deployed.display_name})")
            endpoints.delete_endpoint(name=endpoint.name).result(timeout=1800)
            removed.append(f"endpoint {endpoint.name}")

        # Lab 4: alert policies and their notification channels carry the same labels,
        # and a schedule that outlives them pages nobody while it keeps writing points.
        for kind, name in self._monitoring_resources(tags):
            self._monitoring().delete(f"{self.MONITORING_API}/{name}",
                                      params={"force": "true"} if kind == "channel" else None
                                      ).raise_for_status()
            removed.append(f"{kind} {name}")
        return removed

    # --- Lab 4: Cloud Monitoring, over REST ----------------------------------
    # google-auth and requests are already pinned in requirements.txt; the monitoring
    # client library would add a dozen packages to write one number.

    MONITORING_API = "https://monitoring.googleapis.com/v3"
    METRIC_PREFIX = "custom.googleapis.com/itcs355"

    def _monitoring(self):
        import google.auth
        from google.auth.transport.requests import AuthorizedSession

        if getattr(self, "_monitoring_session", None) is None:
            creds, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"])
            self._monitoring_session = AuthorizedSession(creds)
        return self._monitoring_session

    def emit_metric(self, name: str, value: float, unit: str = "None",
                    labels: dict[str, str] | None = None) -> None:
        import datetime as dt

        now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        body = {"timeSeries": [{
            "metric": {"type": f"{self.METRIC_PREFIX}/{name}", "labels": labels or {}},
            # `global` because the writer is a scheduled job, not a resource: the series
            # must survive the endpoint, the runner and the laptop all being gone.
            "resource": {"type": "global", "labels": {"project_id": self.cfg.project_id}},
            "points": [{"interval": {"endTime": now}, "value": {"doubleValue": float(value)}}],
        }]}
        r = self._monitoring().post(
            f"{self.MONITORING_API}/projects/{self.cfg.project_id}/timeSeries", json=body)
        if r.status_code != 200:
            raise RuntimeError(f"emit_metric {name} {labels}: {r.status_code} {r.text[:300]}")

    def _monitoring_resources(self, tags: dict[str, str]) -> list[tuple[str, str]]:
        found = []
        base = f"{self.MONITORING_API}/projects/{self.cfg.project_id}"
        for kind, path, key in (("policy", "alertPolicies", "alertPolicies"),
                                ("channel", "notificationChannels", "notificationChannels")):
            r = self._monitoring().get(f"{base}/{path}")
            r.raise_for_status()
            for item in r.json().get(key, []):
                if all(item.get("userLabels", {}).get(k) == v for k, v in tags.items()):
                    found.append((kind, item["name"]))
        return found

    def ensure_metric_alert(self, metric: str, thresholds: dict[str, float], email: str,
                            tags: dict[str, str], label: str = "feature") -> dict[str, str]:
        base = f"{self.MONITORING_API}/projects/{self.cfg.project_id}"
        session = self._monitoring()
        existing = dict((kind, name) for kind, name in self._monitoring_resources(tags))

        channel = {"type": "email", "displayName": f"itcs355 drift -> {email}",
                   "labels": {"email_address": email}, "userLabels": tags}
        if "channel" in existing:
            channel_name = existing["channel"]
        else:
            r = session.post(f"{base}/notificationChannels", json=channel)
            r.raise_for_status()
            channel_name = r.json()["name"]

        metric_type = f"{self.METRIC_PREFIX}/{metric}"
        # Declare the metric before alerting on it. A policy on a metric nobody has written
        # yet is refused for up to ten minutes, and "set up the alert after the first
        # breach" is backwards. Creating an existing descriptor is a no-op.
        r = session.post(f"{base}/metricDescriptors", json={
            "type": metric_type, "metricKind": "GAUGE", "valueType": "DOUBLE",
            "description": f"Lab 4: {metric} per {label}, written by the scheduled detector.",
            "labels": [{"key": label, "valueType": "STRING"}]})
        if r.status_code not in (200, 409):
            raise RuntimeError(f"metric descriptor: {r.status_code} {r.text[:300]}")

        # One condition per label value, ORed. Cloud Monitoring allows six conditions per
        # policy, which is exactly this model's feature count — a seventh feature would
        # need a second policy, and this refuses rather than silently dropping one.
        if len(thresholds) > 6:
            raise ValueError(f"{len(thresholds)} thresholds; one policy holds at most 6")
        conditions = [{
            "displayName": f"{metric}{{{label}={key}}} > {limit}",
            "conditionThreshold": {
                "filter": (f'metric.type="{metric_type}" AND resource.type="global" '
                           f'AND metric.label.{label}="{key}"'),
                # The writer is a schedule, so points are sparse. ALIGN_MAX over 5 min
                # keeps one breaching point visible to the evaluator; a mean would dilute
                # it against the empty periods on either side.
                "aggregations": [{"alignmentPeriod": "300s", "perSeriesAligner": "ALIGN_MAX"}],
                "comparison": "COMPARISON_GT",
                "thresholdValue": limit,
                "duration": "0s",
                "trigger": {"count": 1},
            },
        } for key, limit in sorted(thresholds.items())]
        policy = {
            "displayName": f"itcs355 {metric} above its per-{label} limit",
            "userLabels": tags,
            "combiner": "OR",
            "documentation": {"mimeType": "text/markdown", "content": (
                f"`{metric}` crossed its limit for at least one {label}. Before retraining, "
                "check whether the schema or null rate changed: a broken upstream pipeline "
                "looks exactly like drift here, and retraining on it destroys the last good "
                "model. Runbook: README, Lab 4 Task 6.")},
            "conditions": conditions,
            "alertStrategy": {"autoClose": "1800s"},
            "notificationChannels": [channel_name],
            "enabled": True,
        }
        import time

        # A new descriptor takes minutes to become visible to the policy validator, which
        # answers 404 "Cannot find metric(s)" meanwhile. Bounded: ten minutes, then fail.
        for _ in range(21):
            if "policy" in existing:
                r = session.patch(f"{self.MONITORING_API}/{existing['policy']}", json=policy)
            else:
                r = session.post(f"{base}/alertPolicies", json=policy)
            if not (r.status_code == 404 and "Cannot find metric" in r.text):
                break
            time.sleep(30)
        if r.status_code != 200:
            raise RuntimeError(f"alert policy: {r.status_code} {r.text[:400]}")
        return {"policy": r.json()["name"], "channel": channel_name}

    # generate                          -> Lab 5 (managed LLM endpoint; read usageMetadata for tokens)
    # generate                          -> Lab 5 (managed LLM endpoint; read usageMetadata for tokens)
