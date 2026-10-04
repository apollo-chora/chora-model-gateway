# syntax=docker/dockerfile:1.6
#
# chora-model-gateway — multi-stage build.
#
# BUILD CONTEXT IS THIS REPOSITORY'S ROOT:
#
#   docker build -t chora-model-gateway .
#
# The context is the root because go.mod `replace`s the generated gRPC stubs to
# ./gen, which is a sibling module. Exactly two things are copied in — ./gen and
# ./services' own source — so the context is small and self-contained: a fresh
# clone builds with nothing else checked out.
#
# Multi-arch (amd64 + arm64), which is what the registry image is:
#
#   docker buildx build --platform linux/amd64,linux/arm64 \
#     -t <registry>/chora-model-gateway:<tag> --push .
#
# The binary is CGO_ENABLED=0 and static, so a single build serves both.

ARG GO_VERSION=1.26.6
ARG ALPINE_VERSION=3.23
ARG SERVICE_NAME=chora-model-gateway

# ----------------------------------------------------------------------------
# Stage 1 — build
# ----------------------------------------------------------------------------
FROM golang:${GO_VERSION}-alpine${ALPINE_VERSION} AS builder

ARG SERVICE_NAME
ARG GIT_SHA=unknown
ARG BUILD_TIME=unknown

RUN apk add --no-cache ca-certificates git

WORKDIR /src

# The generated protobuf/gRPC stubs for the ModelGatewayService contract,
# vendored under ./gen (see go.mod for how to re-sync them).
COPY gen/ ./gen/

# This service.
COPY . .

# Synthesise a two-module workspace so the replace directive resolves. Without
# it the ./gen module is only found via the replace, which works but pulls in a
# second resolution path; the workspace makes the intended one explicit.
RUN printf 'go 1.26.1\n\nuse (\n\t./gen\n\t.\n)\n' > /src/go.work

WORKDIR /src

ENV CGO_ENABLED=0 \
    GOOS=linux
RUN go build -trimpath \
      -ldflags "-s -w \
        -X main.gitSHA=${GIT_SHA} \
        -X main.buildTime=${BUILD_TIME}" \
      -o /out/service \
      ./cmd/server

# ----------------------------------------------------------------------------
# Stage 2 — runtime
# ----------------------------------------------------------------------------
FROM gcr.io/distroless/static-debian12:nonroot

ARG GIT_SHA=unknown
ARG BUILD_TIME=unknown

LABEL org.opencontainers.image.title="chora-model-gateway" \
      org.opencontainers.image.source="https://github.com/apollo-chora/chora-model-gateway" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.created="${BUILD_TIME}" \
      org.opencontainers.image.vendor="Chora Platform" \
      org.opencontainers.image.licenses="UNLICENSED"

WORKDIR /

COPY --from=builder /out/service /service

# The registry is read at boot and never written, so it ships in the image at a
# fixed path. docker-compose mounts over it to change the catalogue without a
# rebuild.
COPY config/models.yaml /etc/chora/models.yaml

USER nonroot:nonroot

ENV CHORA_MODEL_REGISTRY=/etc/chora/models.yaml \
    CHORA_GRPC_PORT=9090 \
    CHORA_HTTP_PORT=8080

EXPOSE 9090 8080

ENTRYPOINT ["/service"]