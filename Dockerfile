# syntax=docker/dockerfile:1.6
#
# chora-model-gateway Dockerfile — Go service (M15 / ADR-163).
# Mirrors chora-infra/templates/Dockerfile.go-service with one addition:
# chora-contracts/gen/go/ is copied because the gateway imports the
# generated model_gateway_service stubs (mgv1).
#
# Build context = repo root (the monorepo). Standard invocation:
#
#   docker buildx build --platform=linux/amd64 \
#     -f services/chora-model-gateway/Dockerfile \
#     --build-arg SERVICE_NAME=chora-model-gateway \
#     --build-arg GIT_SHA=$(git rev-parse --short HEAD) \
#     --build-arg BUILD_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ) \
#     -t asia-southeast1-docker.pkg.dev/chora-489812/chora-services/chora-model-gateway:${TAG} \
#     --push \
#     .
#
# --platform=linux/amd64 is MANDATORY on darwin/arm64 dev hosts per
# feedback_buildx_platform_amd64 — arm64 binaries fail silently on GKE
# linux/amd64 nodes.

ARG GO_VERSION=1.26.6
ARG ALPINE_VERSION=3.23
ARG SERVICE_NAME=chora-model-gateway
ARG GIT_SHA=unknown
ARG BUILD_TIME=unknown

# ----------------------------------------------------------------------------
# Stage 1 — build
# ----------------------------------------------------------------------------
FROM golang:${GO_VERSION}-alpine${ALPINE_VERSION} AS builder

ARG SERVICE_NAME
ARG GIT_SHA
ARG BUILD_TIME

WORKDIR /src

RUN apk add --no-cache ca-certificates git

# Shared libs first (cache-friendly across rebuilds).
COPY libs/chora-go-common/ ./libs/chora-go-common/

# Generated proto stubs — gateway imports
# github.com/locoroco-git/Chora-LMS/chora-contracts/gen/go/chora/services/model_gateway/v1.
COPY chora-contracts/gen/go/ ./chora-contracts/gen/go/

# ADR-152 Cloud Model Armor agent → template-tier lookup table.
COPY chora-contracts/yaml/agent-guardrail-mapping.yaml ./chora-contracts/yaml/agent-guardrail-mapping.yaml

# This service.
COPY services/${SERVICE_NAME}/ ./services/${SERVICE_NAME}/

# Synthesise a minimal go.work — service + shared lib + contracts only.
RUN cat > /src/go.work <<EOWORK
go 1.26.1

use (
	./libs/chora-go-common
	./chora-contracts/gen/go
	./services/${SERVICE_NAME}
)
EOWORK

WORKDIR /src/services/${SERVICE_NAME}

RUN go mod download all 2>/dev/null || go mod download

ENV CGO_ENABLED=0 \
    GOOS=linux \
    GOARCH=amd64
RUN go build -trimpath \
      -ldflags "-s -w \
        -X main.serviceName=${SERVICE_NAME} \
        -X main.gitSHA=${GIT_SHA} \
        -X main.buildTime=${BUILD_TIME}" \
      -o /out/service \
      ./cmd/server

# ----------------------------------------------------------------------------
# Stage 2 — runtime
# ----------------------------------------------------------------------------
FROM gcr.io/distroless/static-debian12:nonroot

ARG SERVICE_NAME
ARG GIT_SHA
ARG BUILD_TIME

LABEL org.opencontainers.image.title="${SERVICE_NAME}" \
      org.opencontainers.image.source="https://github.com/5007-Capstone/chora" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.created="${BUILD_TIME}" \
      org.opencontainers.image.vendor="Chora Platform" \
      org.opencontainers.image.licenses="UNLICENSED" \
      io.chora.service="${SERVICE_NAME}" \
      io.chora.git-sha="${GIT_SHA}" \
      io.chora.build-time="${BUILD_TIME}"

WORKDIR /

COPY --from=builder /out/service /service
COPY --from=builder /src/chora-contracts/yaml/agent-guardrail-mapping.yaml /etc/chora/agent-guardrail-mapping.yaml

USER nonroot:nonroot
ENTRYPOINT ["/service"]
