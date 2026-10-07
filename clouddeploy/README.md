# chora-model-gateway — Cloud Deploy ops runbook

> Operational reference for engineers + ops + auditors. Companion to
> services/chora-model-gateway/cloudbuild.yaml + services/chora-model-gateway/clouddeploy/{skaffold,dev,prod}.

## Identity

- Service: chora-model-gateway
- Team: team-3-platform
- Domain: ai-kernel (Tier 1 D1 supporting domain)
- Track: M15 LLM-routing (ADR-163 — chora-model-gateway on GKE asia-southeast1)
- Cloud Build trigger: chora-model-gateway-main-dev (TF-managed, **DISABLED on initial create** per [[feedback-cicd-no-mass-trip]]; user trips manually)
- Cloud Deploy pipeline: chora-model-gateway-pipeline
- Cloud Deploy targets:
  - chora-model-gateway-dev → ns `ai-kernel-dev` (auto-deploy, verify on)
  - chora-model-gateway-prod → ns `ai-kernel` (requireApproval: false — SOFT per Wave-2 default 2026-05-17; verify off — PSA-restricted ns)
- Cloud Deploy Automation: chora-model-gateway-pipeline/promote-dev-to-prod (promoteReleaseRule, wait 1m settle)
- Cloud Deploy executionTimeout: 7200s
- Runtime GSA: chora-model-gateway@chora-489812.iam.gserviceaccount.com (per m15-model-gateway-iam module)
- Workload Identity binding: chora-489812.svc.id.goog[ai-kernel-dev/chora-model-gateway] AND [ai-kernel/chora-model-gateway]

## Image

- Registry: asia-southeast1-docker.pkg.dev/chora-489812/chora-services/chora-model-gateway
- Image vulnerability tab: https://console.cloud.google.com/artifacts/docker/chora-489812/asia-southeast1/chora-services/chora-model-gateway?project=chora-489812
- Cosign verify (KMS):
  ```bash
  cosign verify asia-southeast1-docker.pkg.dev/chora-489812/chora-services/chora-model-gateway:<SHORT_SHA> \
    --key=gcpkms://projects/chora-489812/locations/asia-southeast1/keyRings/chora-keys/cryptoKeys/chora-binauthz-signer
  ```

## Evidence

- GCS evidence bucket: gs://chora-489812-cloudbuild-evidence/chora-model-gateway/<SHORT_SHA>/
  - chora-model-gateway-lint-report.txt
  - chora-model-gateway-cover.out
  - chora-model-gateway-cover-integration.out
  - chora-model-gateway-gosec.sarif
  - chora-model-gateway-govulncheck.json + -summary.txt
  - chora-model-gateway-trivy.sarif
  - chora-model-gateway-sbom.spdx.json

## One-time prerequisites (before first dev rollout)

These prereqs must land BEFORE the first `gcloud builds triggers run` fires.
The trigger is DISABLED on initial create; user flips to enabled only after:

1. **Namespace `ai-kernel-dev` provisioned** (does NOT exist as of bring-up commit):
   ```bash
   kubectl create namespace ai-kernel-dev
   kubectl label namespace ai-kernel-dev \
     chora.dev/team=team-3-platform \
     chora.dev/environment=dev \
     chora.dev/data-classification=development \
     app.kubernetes.io/managed-by=terraform
   # NOTE: omit pod-security.kubernetes.io/* labels — dev ns stays permissive
   # so the verify-Job pod can run with default security context.
   # NOTE: omit istio-injection=enabled — mirrors chora-identity-dev. Istio
   # sidecar on the verify-Job pod runs FOREVER (never terminates after the
   # main container completes), causing Cloud Deploy verify-Job timeout.
   # Dev pods run sidecarless (PeerAuthentication is PERMISSIVE in prod ns
   # so the dev pod can still dial inter-service in plaintext if needed).
   # Prod ns (ai-kernel) keeps istio-injection=enabled.
   ```
2. **KSA + WIF binding in ai-kernel-dev**:
   ```bash
   kubectl -n ai-kernel-dev create serviceaccount chora-model-gateway
   gcloud iam service-accounts add-iam-policy-binding \
     chora-model-gateway@chora-489812.iam.gserviceaccount.com \
     --role=roles/iam.workloadIdentityUser \
     --member='serviceAccount:chora-489812.svc.id.goog[ai-kernel-dev/chora-model-gateway]' \
     --account=dale-cli@chora-489812.iam.gserviceaccount.com \
     --project=chora-489812
   ```
3. **Secret Manager IAM grant in dev**: the gateway boots fail-loud against
   `chora-dev-cloudsql-chora_observability-app_rw-dsn`; ensure the dev DSN
   secret IAM has chora-model-gateway@ as `roles/secretmanager.secretAccessor`.
   (Likely already in place via m15-model-gateway-iam — verify.)

## Watch / verify commands

```bash
# Trigger fire history (last 10 builds)
gcloud builds list --region=asia-southeast1 --filter='tags:service-chora-model-gateway' --limit=10 \
  --project=chora-489812

# Latest Cloud Deploy release
gcloud deploy releases list --delivery-pipeline=chora-model-gateway-pipeline \
  --region=asia-southeast1 --limit=1 --project=chora-489812

# Rollout state on the latest release (replace <RELEASE>)
gcloud deploy rollouts list --release=<RELEASE> \
  --delivery-pipeline=chora-model-gateway-pipeline --region=asia-southeast1 \
  --project=chora-489812

# Pod state in dev + prod
kubectl -n ai-kernel-dev get pods,deploy,hpa
kubectl -n ai-kernel get pods,deploy,hpa -l app.kubernetes.io/name=chora-model-gateway

# Smoke (gRPC health check from inside cluster)
kubectl run --rm -i --tty grpcprobe \
  --image=ghcr.io/grpc-ecosystem/grpc-health-probe:v0.4.24 \
  --restart=Never -- \
  -addr=chora-model-gateway.ai-kernel.svc.cluster.local:9090
```

## Observability

- Cloud Trace service map (filtered): https://console.cloud.google.com/traces/list?project=chora-489812
- Cloud Logging: https://console.cloud.google.com/logs/query?project=chora-489812&query=resource.labels.namespace_name%3D%22ai-kernel%22%20AND%20resource.labels.container_name%3D%22service%22
- TokenUsageLedger (canonical billing-grade): chora_observability.token_usage_ledger (per [[ai-cost-tracking]] skill)
- chora.observability.token_usage.recorded.v1 Pub/Sub topic (per [[pub-sub-topology]] skill)

## Console links

- Trigger: https://console.cloud.google.com/cloud-build/triggers;region=asia-southeast1?project=chora-489812 (filter by chora-model-gateway)
- Build history: https://console.cloud.google.com/cloud-build/builds;region=asia-southeast1?project=chora-489812
- Cloud Deploy pipeline: https://console.cloud.google.com/deploy/delivery-pipelines/asia-southeast1/chora-model-gateway-pipeline?project=chora-489812

## Approval gate posture

SOFT (auto-approve) — `requireApproval: false` on chora-model-gateway-prod target + Automation auto-advance enabled (1m wait after dev SUCCEEDED). Per user directive 2026-05-26 (Wave-2 default to support concurrent SIT dev). Restore manual approval by flipping `requireApproval: true` in chora-infra/clouddeploy/targets/chora-model-gateway-prod.yaml + re-applying via `gcloud deploy apply`.

Note: ADR-159 PROPOSED 2026-05-18 originally specified manual approval for the M15 LLM-routing track. User chose Wave-2 default at bring-up; if the M15 track later goes into a higher-risk phase, flip to manual.

## Trigger gate posture

DISABLED on initial create — per [[feedback-cicd-no-mass-trip]]. The `disabled` field in the chora-infra `service_repos` Terraform variable is set to `true` for chora-model-gateway in `chora-infra/terraform/environments/dev/terraform.tfvars`. To enable normal CI on push to main:

```bash
# Manual flip — edit terraform.tfvars: chora-model-gateway.disabled = false
terraform -chdir=chora-infra/terraform/environments/dev plan
terraform -chdir=chora-infra/terraform/environments/dev apply -target=module.m10_platform_services.google_cloudbuild_trigger.service_main_push
```

OR (one-off, NON-TF, drift-warning): `gcloud builds triggers update chora-model-gateway-main-dev --no-disabled --region=asia-southeast1 --project=chora-489812`. NB: any subsequent `terraform apply` will re-disable if the tfvars entry stays `disabled = true`.

## Known runtime gotchas

- **PSA=restricted on prod ns `ai-kernel`** — Cloud Deploy verify on prod set to `false`. The skaffold `VerifyContainer` schema does not expose `securityContext` so verify can't satisfy the PSA constraint at the prod target. Verify-on-dev-only is the Wave-2 default. Restoring prod verify requires ASM nativeSidecar GA OR a manual PeerAuthentication selector exception.
- **NEG quota silently fails Cloud Deploy rollouts**: see `feedback_neg_quota_orphan_zones` memory. Symptom: rollout FAILED with EXECUTION_FAILED, but `kubectl get pod` shows the new pod Ready 3/3. Check Cloud Console NEG quota for `asia-southeast1`.
- **Cosmetic FAILURE at trailing PUSH** (cover-integration.out missing): if no integration-tagged tests exist, the build writes a placeholder marker so the artifacts.objects upload doesn't cosmetic-FAIL. Promote to a real integration smoke once Pub/Sub + Cloud SQL fixtures are wired.
- **Defense-in-depth check (Phase 3.1 inheritance)** — the gateway's `:9090` ingress at `gateway.chora.site` sits behind Cloud Armor + Cloud Model Armor (PRE + POST). The CICD pipeline does NOT exercise either; it only smokes `grpc.health.v1.Health/Check`. End-to-end attack-payload defense is verified separately via the `docs/m15/option-b-cloud-armor-cutover-2026-05-26.md` test plan.

## See also

- Canonical pipeline spec: ../../../docs/architecture.md §6.8
- Cross-service evidence index: ../../../docs/cicd/EVIDENCE_PACK_INDEX_2026-05-17.md
- Cross-service URL cheat sheet: ../../../docs/cicd/EVIDENCE_URL_CHEAT_SHEET.md
- Cookbook handoff: ../../../docs/m15/handoff-model-gateway-cicd-bring-up-2026-05-26.md
- ADR-163: ../../../docs/architecture/adrs/adr-163-model-gateway-on-gke-asia-southeast1.md
