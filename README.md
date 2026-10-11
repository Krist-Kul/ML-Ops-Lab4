# ITCS355 Lab 1 — Reproducible Training

> **Course materials live in [`course/`](course/README.md)** — syllabus, slides, the faculty
> specification, all five lab handouts, and the project brief. Every document is Markdown and
> renders on GitHub, diagrams included. New to the repo? Start with the
> [portability reference](course/reference/cloud-portability-reference.md).
> Keep this block when you edit the rest of this file; it is not part of the Lab 1 deliverable.

Predicting machine failure within 7 days from sensor readings. The model is not the point;
whether a stranger can reproduce it is.

> **This README is graded.** A grader with Docker and nothing else from this setup runs one
> command and compares the result against the claim below.

---

## Reproduce

```bash
make reproduce
```

expected test_roc_auc: 0.848 ± 0.010

Runtime: about 40 seconds on 4 cores. No cloud account or credentials needed for this command —
that is deliberate, and it is why a grader can run it.

The tolerance comes from measured spread, not from caution. Held at the default seed
(`20260101`), repeated runs on this machine agree to six decimal places, and the containerised
`linux/amd64` run agrees with the host to within 1.3e-5. Changing the seed is a different matter:
seeds 20260101 / 7 / 1234 / 99 give 0.8483 / 0.8503 / 0.8610 / 0.8440, a spread of 0.017. That
is the split moving, not the model — `data.split` groups by `machine_id`, so a new seed deals
different machines into test. The claim above is a fixed-seed claim, and the tolerance is sized
for cross-machine floating-point drift only.

---

## The problem

240 machines, 25 readings each, 6 sensor features, binary target `failed_within_7d` with a
positive rate near 12%.

Machines have persistent characteristics — a hot-running machine reads hot in every row. So the
train/validation/test split is **grouped by `machine_id`**: every reading from one machine lands
in exactly one partition. Splitting row-wise instead lets the model memorise the machine and
reports a validation score that will never survive production. `tests/test_data.py` asserts this
property holds, and Lab 4 turns it into a CI gate.

Bringing your own dataset is allowed. Replace `scripts/make_dataset.py`, update the schema in
`src/data.py`, and keep every test passing.

---

## Layout

```
src/          Layer 1 — provider-neutral. No SDKs, no bucket names, no absolute paths.
cloudlayer/   Layer 3 — the only place a provider SDK may be imported.
scripts/      Dataset generation, cloud check, portability audit, metric verification.
tests/        Data contract tests and split property tests.
```

`src/config.py` is the single point of environment knowledge. Everything else reads from it.
`make portability-audit` enforces the rule; it fails the build if a provider string appears in
`src/` or `tests/`.

---

## Setup

```bash
cp cloud.env.example cloud.env      # fill in, never commit
make setup
make cloud-check                    # eight slots, all PASS
make data                           # generate the dataset
make test                           # 10 tests, all passing
```

Post your `make cloud-check` output in the course channel before Session 1.

---

## What you must finish

Four `TODO` markers are left in the repo deliberately. Each is a graded decision, not busywork.

| Where | What |
|---|---|
| `requirements.txt` | Compiled with `pip-compile --generate-hashes`, inside `python:3.11-slim` so the lock matches the image |
| `Dockerfile` | Base pinned by multi-arch index digest; `pip install --require-hashes` |
| `cloudlayer/gcp.py` | `upload`, `download`, `push_image` implemented; the other seven still raise `NotImplementedError` |
| This README | Claim line re-measured, trade-off answered below |

Provider plumbing, for the record:

```bash
make image-push        # linux/amd64 image, pushed digest-pinned to Artifact Registry
dvc remote modify storage url ${BLOB_URI}/dvc
dvc add data/raw && dvc push
```

Five tracked runs are in the `itcs355-lab1` experiment, varying tree depth
(4 / 8 / 16), forest size (200 / 600), and leaf size (5 / 25) — not five seeds of one
configuration. Depth is the parameter that moves the metric; 600 trees buys nothing over 200.

---

## Reproducibility trade-off

Drop the seeds first.

Losing them costs comparability: two runs of the same code give slightly different numbers, and
here a seed change moves test ROC AUC by up to 0.017 because the seed also deals the grouped
split. That is a measurement problem, and it is visible — the number is simply different, and it
is recoverable by re-running with a seed fixed.

The other two fail silently. An unhashed dependency lets a republished wheel change what the
build installs with no commit; a tag-pinned base changes the interpreter and system libraries
under the same `FROM` line. Both produce a build that stopped being the build you tested, with
nothing in Git to show for it.

---

## Lab 2 — tracking, study, registry

```bash
make train-remote                 # the Lab 1 container as a Vertex custom job
make tune                         # 12-trial study, spot, checkpointed to object storage
make compare                      # reports/lab2-comparison.md, ranked by cost per point
make register                     # retrain the chosen trial at HEAD, register, stage it
make reload-check VERSION=1       # load that version FROM the registry and score rows
make promote VERSION=1 ALIAS=production
```

Reports: `reports/lab2-study.md` (jobs, trials, the interruption drill),
`lab2-comparison.md` (table plus the 200-word justification), `lab2-seed-variance.md`,
`lab2-prices.md` (how the rates in `src/costs.py` were verified), and
`lab2-registry.md` (the registered version and its lineage).

`make register` refuses to run from a dirty tree. The study itself was run from one, so its
trials record `<sha>-dirty`, and a lineage field naming a commit nobody can check out is the
failure the lab is about. The registered version is therefore a retrain of the chosen trial's
configuration from a clean commit, on the same data and seed, and its metrics are compared
against the trial's — they match exactly, which is the evidence that the retrain is the same
model and not merely a similar one.

Lineage lives on the registered **version**, not on the MLflow run: Vertex keeps labels and the
version description per version, checked by registering a second version with a different commit
and reading the first one back unchanged. Labels are capped at 63 characters, so the full
`image_digest` is in the version description and truncated in the label.

Registered: `itcs355-6688053` version **1**, aliases `default`, `staging`, `production`,
trained by Vertex job `4570562909159030784` from commit `6173dd6` on data
`422cccb9136e8140` — val ROC AUC 0.8405, test 0.8543, identical to the study's `trial-00`.
`make reload-check VERSION=1` loads it from the registry and scores five held-out rows.

### Lab 2 deliverables

- [x] `submit_training()` / `wait_training()` — `make train-remote`, 91 s, test ROC AUC
      0.84824, the same number the local container produced
- [x] 12 trials on spot, checkpointed to object storage, all tracked in `itcs355-lab2`
- [x] Interruption survived: `make tune INTERRUPT_AT=5` killed trial 5 mid-run; Vertex
      restarted the container and the study resumed from the checkpoint 7 s later
- [x] Comparison table and the 200-word justification in `reports/lab2-comparison.md`
- [x] Eight lineage fields on the registered **version** — `reports/lab2-registry.md`
- [x] Promotion performed (`staging`, then `production`), with the ownership rule below
- [x] `reload_check.py` loads by version from the registry and scores rows
- [x] Cost per trial recorded; 0.51 THB spent of the 150 THB budget, repriced at the
      verified rates — 0.23 THB for the first managed run (on demand, before spot was
      wired up), 0.23 THB for the 12-trial study on spot, 0.05 THB for the registration
      run. The figures logged at the time are 25% lower because they used the handout's
      unverified 7.6 THB/h and flat 30% spot rule
- [ ] `make teardown` — implemented in Lab 5. No compute is left running (all Vertex jobs
      terminal), and the handout keeps the registered model and its artifacts for Lab 3

The permission that failed first, which Drill 2 asks for: the job's run-time identity had
no access to the bucket, and Vertex reported it as `OSError: [Errno 5] Input/output error`
from the `/gcs` FUSE mount — not as an IAM error. Fixed with `roles/storage.objectUser`
on the bucket for `itcs355-trainer@...`, which is the job's `IDENTITY_REF`, not the CLI
identity that submitted it.

### Who may promote a model

Not the person who trained it, and not a rule that only says "a senior engineer". In an
organisation this repository's model belongs to — maintenance scheduling, where a false negative
is an unplanned line stop and a false positive is a wasted technician visit — promotion to
`production` should need two roles:

| Role | What they own |
|---|---|
| The model owner (an ML engineer who did **not** train this version) | Reads the lineage and re-runs the reload check against the version. They are checking the record, not the metric |
| The maintenance lead who consumes the predictions | Accepts the operating point: at the threshold in use, this many extra callouts per week, that many misses |

Evidence they should require, and refuse the promotion without:

1. **All eight lineage fields resolve.** The commit checks out, the image digest exists in the
   registry, the data hash matches the versioned dataset, the run opens. A field that is present
   but stale is worse than a missing one.
2. **A reload from the registry by version**, on a machine that never trained it —
   `make reload-check VERSION=<n>` — with scores, not just a successful import.
3. **Test metrics from a split the candidate never saw**, next to the incumbent's on the same
   split, and a statement of whether the difference exceeds the seed variance measured in
   `reports/lab2-seed-variance.md`. For this study it does not, which is an argument for leaving
   the incumbent alone.
4. **A rollback path**: the version the alias currently points at, and the command that moves it
   back. Vertex aliases are unique per model, so `make promote VERSION=<old> ALIAS=production`
   is that command, and it is one API call.
5. **A stated cost**, training and serving, because promotion is what starts the meter.

In this course the same person does all of it, which is exactly why the rule has to be written
down rather than felt. Note also that the `default` alias stays on version 1 unless moved:
registering a newer version does not promote it, and a service that loads "the latest model"
has decided to promote every training run automatically.

---

## Lab 3 — serving (Task 1: the inference service)

```bash
make serve VERSION=1              # serve the REGISTERED version, resolved through the adapter
make serve VERSION=production     # an alias resolves too; responses report the version it names
make serve-local                  # a locally trained file; development only
make smoke-local                  # hit all four routes and print the bodies
make serve-image                  # build the serving image, base pinned by digest
pytest -q tests/test_service.py   # 16 tests: the contract, and the startup lifecycle
```

| Route | Purpose |
|---|---|
| `GET /health` | Liveness. The process is up. Never touches the model |
| `GET /ready` | Readiness. The model is loaded **and has scored a row** |
| `POST /predict` | One row |
| `POST /predict/batch` | 1–100 rows |

The model is resolved by `adapter.get_model(name, version)` and loaded **once**, on a
background thread, so the port opens before the model exists: `/health` answers alive while
`/ready` answers 503 for the 7.3 s the artifact takes to download and unpickle. That window is
the point of the two routes, and it only exists if the load is off the startup path. Readiness
is then earned by scoring a midpoint row — an artifact that unpickles but cannot predict is the
failure that looks healthy.

Every response, success or error, carries the **resolved** registry version and the request id;
`production` never appears in a response body, because an alias moves and Task 4 will move it.
`MODEL_VERSION` is deployment configuration, never baked into the image, so a rollback is a
config change rather than a rebuild.

Logs are one JSON object per line built with `json.dumps` — 13 of 13 lines on the last run,
uvicorn's own included. Two libraries had to be silenced to keep that true
(`MLFLOW_DISABLE_AGENT_HINT`, `MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR`): a log line a collector
cannot parse takes the whole batch with it.

Evidence, decisions, and the cold-start and latency figures that Task 3 starts from:
`reports/lab3-service.md`.

### Task 2 — deployed

```bash
make deploy              # push the serving image, register a serving version, deploy it
make smoke               # three known payloads, checked against the registry
make endpoint-status     # what is attached, its traffic share, what it costs per hour
make teardown            # delete it
```

`itcs355-6688053-endpoint` serves registry version **3** on `n1-standard-2`. Version 3 is
version 1's *bytes* — the same artifact object, not a retrain — registered again with the
serving image as its container, because a Vertex version carries the container that serves it
and version 1 names the training image. Lineage is copied across unchanged: a different web
server in front of the model did not change the commit that produced it.

Every provider-shaped part sits in `cloudlayer/`: `gcp_serving.py` translates Vertex's
`{"instances": […]}` envelope and its `AIP_STORAGE_URI` into the neutral `MODEL_URI`, and
`adapter.serving_spec()` declares the routes, port and command. `service/app.py` is untouched
by any of it, and the image's default `CMD` still runs the neutral app. The one provider
dependency in the image is a storage client, and only because Vertex hands the container a
`gs://` URI where SageMaker and Azure ML hand it a local directory.

`make smoke` scores three known payloads and compares each against the same version loaded
locally from the registry — all three match exactly — then sends a row that must be refused,
and it is, with the service's own 422 body. Health is readiness: the Model resource declares
`/ready`, and Vertex polls it about 360 times an hour.

Two findings the deployment produced and local testing could not. Every log line arrived in
Cloud Logging as **ERROR**, because the handler wrote to stderr and the platform reads a
prediction container's stderr as errors — a healthy endpoint reporting 360 errors an hour from
its own health checks. Now on stdout, with a `severity` field. And a deployment at **0% traffic
still bills**: after the second deploy both versions held replicas at 10.18 THB/h until the old
one was undeployed. Moving traffic is not releasing machines, which is the bill waiting at the
end of Task 4.

**Cost: 5.09 THB/h, 122 THB/day** — verified against the Cloud Billing Catalog, now in
`src/costs.py` as `SERVING_PRICE_TABLE`. A prediction vCPU-hour is 13% dearer than the training
vCPU-hour for the same machine, and E2 is 30% cheaper than N1. With 149 THB of budget left,
this endpoint spends it in 29 hours. Full write-up: `reports/lab3-deploy.md`.

Tasks 3–5 (load test, canary and rollback, cost per 1,000) are written up in
`reports/lab3-load.md`, `reports/lab3-canary.md` and `reports/lab3-cost.md`.

---

## Lab 4 — CI/CD, observability, drift

### Task 1 — tests that can fail

| category | where | runs in CI as |
|---|---|---|
| unit | `tests/test_drift.py`, `test_costs.py`, `test_canary.py`, `test_traffic_split.py`, `test_llm_eval.py` | `Unit tests`, before the dataset exists |
| data contract | `tests/test_data.py` (contract half) | `Data contract tests` |
| model behaviour | `tests/test_model_behaviour.py` | `Model behaviour tests` |
| integration | `scripts/integration_check.py` against the built serving image | `Integration test` in the build job |

**The incident each data contract test would have caught.** Each was checked by corrupting the
real file the way the incident would, and running the four tests against it:

| test | incident it catches | Fahrenheit | % as fraction | 2% nulls | dropped column |
|---|---|---|---|---|---|
| `test_schema_columns_present_and_typed` | upstream renames or drops a column (`ambient_humidity` removed by a gateway update); training fails with a `KeyError` far from the cause, serving returns 422 to every caller | pass | pass | pass | **FAIL** |
| `test_null_rate_within_threshold` | a sensor gateway outage writes empty fields. scikit-learn forests accept NaN since 1.4, so this would not crash: the model learns "missing" as a signal | pass | pass | **FAIL** | error |
| `test_features_within_plausible_ranges` | sensor firmware switches `temp_c` to Fahrenheit; every machine reads 160+ and is predicted to fail | **FAIL** | pass | pass | error |
| `test_feature_means_within_reference_band` | `load_pct` arrives as 0.68 instead of 68. Every row passes the [0, 100] range check; only the bulk shift gives it away | **FAIL** | **FAIL** | pass | error |
| `test_target_is_binary_and_not_degenerate` | the label join breaks and every row gets 0; accuracy looks excellent | | | | |
| `test_no_machine_leaks_across_splits` | Lab 1's leakage: a row-wise split reports a score that never survives production | | | | |

The range test and the mean band look redundant until the fraction case: no single row is
implausible, so only a test about the distribution can see it. Thresholds are stated in
`tests/test_data.py`: **null rate 0** (no imputer exists, so any null is learned rather than
handled), **mean within 0.5 reference std** (the whole file differs from the training split by at
most 0.087 std; every unit error worth naming is more than 3 std away).

**Model behaviour.** Valid probabilities, a known-healthy machine scores below 0.5, risk does not
fall as hours since service rise, predictions are not constant, and a single-row prediction
stays under **25 ms** — half the 50 ms service p95 in `loadtest/k6.js`, because Lab 3 measured
the model as ~25 of the service's 26.3 ms p50. Measured locally: 13.0 ms.

**Integration.** `scripts/integration_check.py` starts nothing itself; CI runs the image it just
built, tagged with the commit SHA, and the script asserts the `/predict` body has exactly
`probability`, `model_version`, `request_id`; that the version served is the SHA CI built (in
the body and the `x-model-version` header); and that a bad row gets a 422 naming the field. The
grep for `probability` it replaced would pass on an error body that mentions the word.

### Task 2 — the pipeline

Repository: <https://github.com/Krist-Kul/ML-Ops-Lab4>. `ci.yml` runs on every pull request and
push to main; `cd.yml` runs only when CI completes green on a push to main, through a `staging`
environment that accepts deployments from `main` alone.

```
CI   lint → portability audit → unit → data contract → model behaviour → service
     → build images tagged ${{ github.sha }} → integration test against that image
CD   (main, CI green) OIDC → build → push by SHA, register by digest → staging alias → smoke test
```

- **No stored keys.** `infra/ci_identity.sh` creates a workload identity pool whose provider accepts
  a GitHub token only when `repository == Krist-Kul/ML-Ops-Lab4` **and** `ref == refs/heads/main`.
  A pull request cannot get credentials at all — the bad-commit PR in Task 3 could not push or
  deploy even if the workflow tried. The service account holds four roles, each scoped as narrowly
  as the platform allows (registry writer on one repository, object viewer on one bucket, metric
  writer, Vertex user); the script says why each one.
- **Images are tagged with the commit SHA, registered by digest.** The SHA names the commit; the
  digest names the bytes, and only the digest goes into the model registry.
- **CD ships the commit CI tested.** A `workflow_run` trigger checks out the default branch's head
  by default, which may already be a newer commit; `cd.yml` checks out `workflow_run.head_sha`.
- **Staging is a registry alias, not a warm endpoint.** CD registers a serving version (this image,
  the production version's model bytes), moves `staging` onto it, then pulls the image back by
  digest and runs `integration_check.py` against that exact pair. One replica bills 122 THB/day
  whether called or not and the Lab 3 budget is spent; `STAGING_ENDPOINT=true` as a repository
  variable adds the endpoint deploy through the same `deploy.py` Lab 3 used.
- **Installs are hash-checked everywhere.** CI uses `requirements.txt`; CD and the drift job use
  `requirements-cd.txt`, compiled with `requirements.txt` as a constraint so every package CI tested
  resolves to the same version.

### Task 4 — dashboard and SLO

The service exposes `/metrics` (Prometheus text, hand-written in `service/metrics.py` rather than a
new dependency in the hash-compiled serving requirements). `make monitor-up` runs the CI-built
serving image with the registered production model, Prometheus and Grafana; the dashboard is
**`monitoring/dashboard.json`, provisioned read-only** — the committed file is the dashboard.

| panel | query source | note |
|---|---|---|
| request rate | `http_requests_total` | prediction routes only; probes would otherwise dominate |
| error rate by class | `status_class` label, 4xx and 5xx separate | 4xx is excluded from the availability SLO |
| latency p50 / p95 / p99 | `request_latency_ms` histogram, server-side | buckets dense around 25 and 50 ms, where interpolation matters |
| feature drift | `feature_window_psi` over the last 500 accepted inputs | same statistic and window as the scheduled detector; dashed lines are the alert limits |
| rolling mean | `feature_window_mean{feature="temp_c"}` | a shift moves it; a spread or fleet-mix change barely does, which is why PSI is the alerting signal |
| model version | `model_version_info` | one value; two outside a canary means something is wrong |

Rejected rows never enter the rolling window: a 422 is a caller problem, and counting it would let
one misbehaving client move the drift panel. Tests pin that, the 4xx/2xx split, and that the window
PSI moves when the inputs shift.

**SLO** (`monitoring/slo.yaml`; one sentence each):

| objective | target and window | when the budget is spent |
|---|---|---|
| availability | 99.5% of non-4xx responses over 30 days | freeze every deploy that is not a rollback or reliability fix, model versions included, until the budget recovers, and post-mortem which deploys spent it; if a provider outage spent it, record that and do not freeze |
| latency | p95 of `/predict` ≤ 200 ms at the endpoint over 7 days | if request rate is near the 10.4 req/s one replica saturates at, raise `max_replicas` to 2 (pre-approved, 5.09 THB/h); otherwise freeze deploys and bisect the last change — never relax the target |
| freshness | production model ≤ 30 days old | trigger the Lab 5 retrain, promote only if the gate passes and the new window shows no schema or null-rate change; otherwise keep the stale model with a dated waiver |

99.5% is defended against what this deployment can do (one replica, 37–42 s cold start, a 36.6 s
scripted rollback); 99.9% would be spent by one slow incident and would mean paying for a second
replica nobody downstream would notice. The latency target is the 200 ms end-to-end promise in
`loadtest/k6.js`; the 50 ms there is the container's share of it, which Lab 3 measured
`n1-standard-2` missing (123.5 ms). Both are kept as they were committed. And the freshness
objective has a date on it: production is version 1, registered 2026-09-20, and breaches on
**2026-10-20** — so Lab 5's schedule must retrain at least every 28 days, or the SLO and the
pipeline contradict each other.

---

## Notes for the grader

- `make reproduce` needs Docker only. No `cloud.env`, no credentials, no network beyond the
  base image pull. `make data` regenerates `data/raw/sensors.csv` deterministically from
  `scripts/make_dataset.py`, so a `dvc pull` is not required to reproduce the metric — DVC here
  versions the dataset, it is not a dependency of the graded command.
- The base image is pinned to the **multi-arch index digest**, not the amd64 manifest digest, so
  the same `FROM` line resolves on an arm64 laptop and on an amd64 grader while still naming
  exact bytes. `make image` passes `--platform linux/amd64` regardless; on Apple Silicon that
  runs under emulation and takes several minutes.
- `requirements.txt` was compiled inside `python:3.11-slim` rather than on the host (Python
  3.13), so the resolved versions and hashes are the ones the image actually installs.
- `push_image` returns `repo@sha256:...` read back from Artifact Registry with
  `gcloud artifacts docker images describe`, not the local daemon's `RepoDigests`, and not the
  tag it pushed.
- The DVC remote is `${BLOB_URI}/dvc` on GCS. It is private; a grader who wants the data
  needs a reader grant on the bucket, or can regenerate it with `make data`.
- MLflow tracks to `sqlite:///mlflow.db` in the repo root — local by design in Lab 1, moved to a
  server in Lab 2. Inside the container the tracking DB is redirected to the mounted
  `reports/mlflow.db`, while model artifacts go to `/app/mlruns`, created and owned by the
  non-root `runner` user. Artifacts are discarded with the container; `reports/metrics.json` is
  what `make verify` reads.

---

## Checklist before you submit

- [x] `make reproduce` works from a fresh clone, on a machine that is not yours
- [x] `make verify` passes against the claim line
- [x] `make test` — all tests pass
- [x] `make portability-audit` — clean
- [x] Image builds for `linux/amd64` and is pushed, digest-pinned
- [x] `dvc push` completed; a grader with bucket access can `dvc pull`
- [x] Five or more tracked runs with params, metrics, data fingerprint, and commit SHA
- [x] `git log -p | grep -i -E "secret|password|AKIA|BEGIN PRIVATE"` returns nothing

That last check is not optional. A credential in Git history is an automatic deduction in this
course, and rotating it is your responsibility, not the grader's.
