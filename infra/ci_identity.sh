#!/usr/bin/env bash
# Lab 4 — keyless identity for GitHub Actions (GCP workload identity federation).
#
#   bash infra/ci_identity.sh            # idempotent; prints what to put in repo variables
#
# No service-account key is ever created. GitHub mints an OIDC token per job; GCP trusts it
# only when the token says it came from THIS repository AND from refs/heads/main. A pull
# request — including one from a fork — cannot obtain credentials, so the bad-commit PR in
# Task 3 runs every test and could not push or deploy even if the workflow tried.
#
# Roles, and why each is the narrowest that works:
#   artifactregistry.writer   on the one repository, not the project — push SHA images
#   aiplatform.user           register the serving version and move the `staging` alias.
#                             Broader than wanted (it could also deploy); Vertex has no
#                             narrower predefined role for model upload.
#   monitoring.metricWriter   write drift_psi; cannot read or change alert policies
#   storage.objectViewer      on the bucket — read the production input window
set -euo pipefail

PROJECT=${PROJECT:-itcs355-6688053}
REGION=${REGION:-asia-southeast1}
REPO=${REPO:-Krist-Kul/ML-Ops-Lab4}
BUCKET=${BUCKET:-itcs355-6688053}
AR_REPO=${AR_REPO:-itcs355}
POOL=github
PROVIDER=ml-ops-lab4
SA=itcs355-ci
SA_EMAIL="$SA@$PROJECT.iam.gserviceaccount.com"
NUMBER=$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')

gcloud services enable iamcredentials.googleapis.com sts.googleapis.com --project "$PROJECT"

gcloud iam workload-identity-pools describe "$POOL" --location=global --project "$PROJECT" >/dev/null 2>&1 \
  || gcloud iam workload-identity-pools create "$POOL" --location=global --project "$PROJECT" \
       --display-name="GitHub Actions"

gcloud iam workload-identity-pools providers describe "$PROVIDER" --location=global \
    --workload-identity-pool="$POOL" --project "$PROJECT" >/dev/null 2>&1 \
  || gcloud iam workload-identity-pools providers create-oidc "$PROVIDER" --location=global \
       --workload-identity-pool="$POOL" --project "$PROJECT" \
       --issuer-uri="https://token.actions.githubusercontent.com" \
       --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.ref=assertion.ref" \
       --attribute-condition="assertion.repository=='$REPO' && assertion.ref=='refs/heads/main'"

gcloud iam service-accounts describe "$SA_EMAIL" --project "$PROJECT" >/dev/null 2>&1 \
  || gcloud iam service-accounts create "$SA" --project "$PROJECT" \
       --display-name="ITCS355 CI/CD (GitHub OIDC, no keys)"

gcloud iam service-accounts add-iam-policy-binding "$SA_EMAIL" --project "$PROJECT" \
  --role=roles/iam.workloadIdentityUser \
  --member="principalSet://iam.googleapis.com/projects/$NUMBER/locations/global/workloadIdentityPools/$POOL/attribute.repository/$REPO" \
  --condition=None >/dev/null

gcloud artifacts repositories add-iam-policy-binding "$AR_REPO" --location="$REGION" --project "$PROJECT" \
  --role=roles/artifactregistry.writer --member="serviceAccount:$SA_EMAIL" --condition=None >/dev/null
for role in roles/aiplatform.user roles/monitoring.metricWriter; do
  gcloud projects add-iam-policy-binding "$PROJECT" --role="$role" \
    --member="serviceAccount:$SA_EMAIL" --condition=None >/dev/null
done
gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" \
  --role=roles/storage.objectViewer --member="serviceAccount:$SA_EMAIL" >/dev/null

echo "GCP_WIF_PROVIDER=projects/$NUMBER/locations/global/workloadIdentityPools/$POOL/providers/$PROVIDER"
echo "GCP_DEPLOY_SA=$SA_EMAIL"
