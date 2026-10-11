# ITCS355 Lab 1
# `make reproduce` is the one command a grader runs. Keep it working.

SHELL := /bin/bash
IMAGE ?= itcs355-lab1
TAG   ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo dev)
# Baked into the image. "-dirty" marks a build from uncommitted changes, which no commit
# can reproduce.
GIT_COMMIT ?= $(shell git rev-parse HEAD 2>/dev/null || echo unknown)$(shell git diff --quiet HEAD 2>/dev/null || echo -dirty)
PLATFORM ?= linux/amd64
SEED ?= 20260101

.PHONY: help setup cloud-check data test portability-audit train image image-push reproduce verify clean teardown \
        train-remote tune tune-local compare register promote reload-check serve serve-local serve-image smoke-local deploy smoke endpoint-status loadtest drift inject-drift pipeline cost cost-per-1k swap-check llm-eval llm-gate

help:
	@grep -E "^[a-zA-Z_-]+:.*?## .*$$" $(MAKEFILE_LIST) | awk -F":.*?## " "{printf \"  %-20s %s\\n\", \$$1, \$$2}"

setup: ## Install dependencies and print environment status
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	@echo "environment ok"

cloud-check: ## Resolve the eight capability slots
	python scripts/cloud_check.py

data: ## Generate the default dataset (deterministic)
	python scripts/make_dataset.py --seed $(SEED)

test: ## Run data contract and split property tests
	pytest -q tests/

portability-audit: ## Fail if provider strings leak into src/
	python scripts/portability_audit.py

train: ## Train locally, outside the container
	python -m src.train --seed $(SEED) --metrics-out reports/metrics.json

image: ## Build the training image for linux/amd64
	docker buildx build --platform $(PLATFORM) --build-arg GIT_COMMIT=$(GIT_COMMIT) \
	  -t $(IMAGE):$(TAG) --load .

image-push: image ## Push to CONTAINER_REGISTRY via your adapter
	python -c "from src import config; from cloudlayer.factory import get_adapter; \
	print(get_adapter(config.load()).push_image(\"$(IMAGE):$(TAG)\"))"

reproduce: data image ## THE ONE COMMAND. Grader runs this.
	docker run --rm \
	  -v "$$PWD/data:/app/data:ro" \
	  -v "$$PWD/reports:/app/reports" \
	  -e MLFLOW_TRACKING_URI=sqlite:////app/reports/mlflow.db \
	  $(IMAGE):$(TAG) --seed $(SEED) --metrics-out /app/reports/metrics.json

verify: ## Check the produced metric against the README claim
	python scripts/verify_metric.py

# LAB selects which lab's resources go. Lab 3 is the one that creates a billed endpoint.
LAB ?= 3
teardown: ## Delete every resource tagged course=itcs355 for LAB (default 3)
	python scripts/teardown.py --lab $(LAB)

clean: ## Remove local artifacts
	rm -rf mlruns mlartifacts mlflow.db reports/metrics.json .pytest_cache

# --- Lab 2 -------------------------------------------------------------------
# Builds and pushes the current tree unless IMAGE_URI names an image already pushed.
train-remote: $(if $(IMAGE_URI),,image) ## Run the training image as a managed job
	python scripts/train_remote.py \
	  $(if $(IMAGE_URI),--image $(IMAGE_URI),--local-tag $(IMAGE):$(TAG)) \
	  $(if $(SPOT),--spot) -- --seed $(SEED)

# STUDY=<name> resumes a study; INTERRUPT_AT=<n> kills trial n mid-run once, as a drill.
tune: $(if $(IMAGE_URI),,image) ## Budgeted study on managed spot compute (>=12 trials)
	python scripts/tune_remote.py \
	  $(if $(IMAGE_URI),--image $(IMAGE_URI),--local-tag $(IMAGE):$(TAG)) \
	  --trials 12 --budget-thb 150 --spot --seed $(SEED) \
	  $(if $(STUDY),--study $(STUDY)) $(if $(INTERRUPT_AT),--interrupt-at $(INTERRUPT_AT))

tune-local: ## Same study on this machine at zero cost (development only)
	python -m src.tune --trials 12 --budget-thb 150 --seed $(SEED)

compare: ## Rank runs by metric and by cost per point
	python scripts/compare_runs.py --experiment itcs355-lab2

# TRIAL is the study trial whose configuration gets registered; it is retrained from the
# current commit first, so the lineage names a commit that exists.
TRIAL   ?= trial-00
ALIAS   ?= staging
VERSION ?= 1

register: $(if $(IMAGE_URI),,image) ## Register the chosen model with lineage, then stage it
	python scripts/register_model.py \
	  $(if $(IMAGE_URI),--image $(IMAGE_URI),--local-tag $(IMAGE):$(TAG)) \
	  --trial $(TRIAL) --promote $(ALIAS) $(if $(SPOT),--spot)

promote: ## Move an alias to an existing version: make promote VERSION=1 ALIAS=production
	python scripts/register_model.py --promote-only $(VERSION) --promote $(ALIAS)

reload-check: ## Load the registered model by version and score rows
	python scripts/reload_check.py --version $(VERSION)

# --- Lab 3 -------------------------------------------------------------------
# VERSION is a registry version id or an alias: make serve VERSION=production
serve: ## Serve the REGISTERED version locally on :8080, resolved through the adapter
	MODEL_VERSION=$(VERSION) uvicorn service.app:app --port 8080

serve-local: ## Serve a locally trained file instead. Development only — not submittable
	python scripts/export_model.py --out reports/model.joblib
	MODEL_PATH=reports/model.joblib MODEL_VERSION=local uvicorn service.app:app --port 8080

smoke-local: ## Hit all four routes against a running local service, as JSON evidence
	bash scripts/smoke_service.sh $(or $(TARGET),http://127.0.0.1:8080)

ENDPOINT ?= $(shell grep -E "^MODEL_REGISTRY_NAME=" cloud.env 2>/dev/null | cut -d= -f2)-endpoint
INSTANCE ?= n1-standard-2

deploy: serve-image ## Push the serving image, register a serving version, deploy the endpoint
	python scripts/deploy.py --local-tag itcs355-serve:$(TAG) \
	  --endpoint $(ENDPOINT) --instance $(INSTANCE) \
	  $(if $(SOURCE_VERSION),--source-version $(SOURCE_VERSION)) \
	  $(if $(TRAFFIC),--traffic $(TRAFFIC))

endpoint-status: ## What is deployed, its traffic share, and what it costs per hour
	python scripts/endpoint_status.py --endpoint $(ENDPOINT) $(if $(UNDEPLOY),--undeploy $(UNDEPLOY))

smoke: ## Call the endpoint with three known payloads and check them against the registry
	python scripts/smoke_endpoint.py --endpoint $(ENDPOINT) --rejects

serve-image: ## Build the serving image
	docker buildx build --platform $(PLATFORM) -f service/Dockerfile.serve -t itcs355-serve:$(TAG) --load .

loadtest: ## Load test at three concurrency levels
	@for vus in 1 10 50; do \
	  echo "=== $$vus VUs ==="; \
	  k6 run -e TARGET=$(TARGET) -e VUS=$$vus loadtest/k6.js || true; \
	done

CANARY_VERSION ?=
canary-reference: ## Measure what normal looks like, BEFORE any canary exists
	python scripts/canary.py reference --endpoint $(ENDPOINT)

canary-deploy: ## Attach the canary at TRAFFIC% (default 10)
	python scripts/canary.py deploy --endpoint $(ENDPOINT) \
	  --canary-version $(CANARY_VERSION) --traffic $(or $(TRAFFIC),10) --instance $(INSTANCE)

canary-watch: ## Drive traffic and detect the degradation from metrics alone
	python scripts/canary.py watch --endpoint $(ENDPOINT)

canary-rollback: ## Traffic back to production, then release the canary's machines
	python scripts/canary.py rollback --endpoint $(ENDPOINT)

canary-status: ## Who is attached, what share each holds, what it costs
	python scripts/canary.py status --endpoint $(ENDPOINT)

# --- Lab 4 -------------------------------------------------------------------
MAGNITUDE ?= 4
LAB4_REPO ?= Krist-Kul/ML-Ops-Lab4
EMAIL ?=

inject-drift: ## Shift temp_c by MAGNITUDE (default 4 C, the target the threshold was set for)
	python scripts/inject_drift.py --feature temp_c --mode shift --magnitude $(MAGNITUDE)

drift: ## Score drift against the reference window
	python -m monitoring.drift --current data/current.csv

drift-threshold: ## Derive the per-feature PSI limits and window from no-drift simulation
	python scripts/drift_threshold.py

drift-alerting: ## Create or update the drift alert policy and its email channel (EMAIL=...)
	@test -n "$(EMAIL)" || { echo "usage: make drift-alerting EMAIL=you@example.com"; exit 2; }
	python scripts/drift_alerting.py --email $(EMAIL)

monitor-up: ## Service + Prometheus + Grafana locally; dashboard at localhost:3000
	python -c "from src import config; from cloudlayer.factory import get_adapter; \
	a = get_adapter(config.load()); e = a.get_model(config.load().model_registry_name, 'production'); \
	print(a.download_dir(e['artifact_uri'], 'reports/prod-model'))"
	GIT_SHA=$$(git rev-parse --short HEAD) docker compose -f monitoring/stack/compose.yml up -d --build

monitor-down: ## Stop the local monitoring stack
	docker compose -f monitoring/stack/compose.yml down

replay: ## Production traffic to the local service, shipping the input window for the detector
	python scripts/replay_traffic.py --rate 4 --minutes 10

teardown-lab4: ## Disable the drift schedule FIRST, then delete the alert policy, channel and inputs
	gh workflow disable Drift -R $(LAB4_REPO) || true
	python scripts/teardown.py --lab 4 --yes
	python -c "from src import config; from cloudlayer.factory import get_adapter; \
	print(get_adapter(config.load()).delete_prefix('lab4/'))"
	docker compose -f monitoring/stack/compose.yml down

# --- Lab 5 -------------------------------------------------------------------
pipeline: ## Compile pipeline/pipeline.yaml for your provider
	python -c "from cloudlayer.pipelines import compile_for; from src import config; \
	compile_for(config.load().provider)"

llm-eval: ## Run the LLM golden set against recorded responses (offline, free)
	python scripts/llm_eval.py --out reports/llm_eval-baseline.json

llm-gate: ## Prove the gate fails on a degraded set — expected to exit non-zero
	python scripts/llm_eval.py --out reports/llm_eval-baseline.json >/dev/null
	python scripts/llm_eval.py --responses evals/fixtures/triage-regressed.jsonl \
	  --out reports/llm_eval.json --baseline reports/llm_eval-baseline.json

cost: ## Build the cost report
	python scripts/cost_report.py --estimate $(EST) --actual $(ACT) --rps $(RPS) --instance $(INSTANCE)

cost-per-1k: ## Lab 3 Task 5 — cost per 1,000 predictions and the batch break-even (offline)
	python scripts/cost_per_1k.py

swap-check: ## Prove the portability seam against a second provider
	python scripts/portability_swap_check.py --second-provider $(SECOND)
