# syntax=docker/dockerfile:1.6
#
# chora-model-gateway Dockerfile — Go service (M15 / ADR-163).
#
# Build context = a workspace holding this repo and its two replaced siblings
# as FLAT neighbours, which is the layout the service's go.mod already declares
# (`replace … => ../chora-contracts/gen/go`, `replace … => ../chora-common`) and
# the layout the repos have on a dev machine. .github/workflows/docker-publish.yml
# assembles exactly that:
#
#   chora-model-gateway/   <- this repo
#   chora-contracts/       <- apollo-chora/chora-contracts
#   chora-common/          <- apollo-chora/chora-common
#
# The previous revision COPYed the monorepo paths (services/, libs/, plus a
# synthesised go.work), so every `Publish Docker image` run failed at
# `COPY services/chora-model-gateway/: not found` — and even with those paths
# present the committed go.mod's `../` replaces do not resolve from
# services/chora-model-gateway (that build needed a hand-patched go.mod).
#
# Standard invocation (from the workspace root):
#
#   docker buildx build --platform=linux/amd64,linux/arm64 \
#     -f chora-model-gateway/Dockerfile \
#     --build-arg SERVICE_NAME=chora-model-gateway \
#     --build-arg GIT_SHA=$(git rev-parse --short HEAD) \
#     --build-arg BUILD_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ) \
#     -t walfa/chora-model-gateway:${TAG} \
#     --push \
#     .
#
# GOARCH follows the target platform (buildx supplies TARGETARCH); pass
# --build-arg GOARCH=arm64 to force one. The apollo deployment runs arm64, and
# an amd64 binary there dies with `exec format error`.

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

# The three flat neighbours. The replaced modules are copied as sources: their
# committed pseudo-versions are placeholders (`v0.0.0-00010101…`) that only
# resolve through these replace targets.
COPY chora-contracts/ ./chora-contracts/
COPY chora-common/ ./chora-common/
COPY chora-model-gateway/ ./chora-model-gateway/

WORKDIR /src/chora-model-gateway

RUN go mod download

ARG TARGETARCH
ARG GOARCH=${TARGETARCH}
ENV CGO_ENABLED=0 \
    GOOS=linux \
    GOARCH=${GOARCH}

RUN go build -trimpath -ldflags "-s -w" -o /out/service ./cmd/server

# ----------------------------------------------------------------------------
# Stage 2 — runtime
# ----------------------------------------------------------------------------
FROM gcr.io/distroless/static-debian12:nonroot

ARG SERVICE_NAME
ARG GIT_SHA
ARG BUILD_TIME

LABEL org.opencontainers.image.title="${SERVICE_NAME}" \
      org.opencontainers.image.source="https://github.com/apollo-chora/chora-model-gateway" \
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
