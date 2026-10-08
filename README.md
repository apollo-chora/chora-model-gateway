# chora-model-gateway

## About

`chora-model-gateway` is a Go service that presents Chora model access through OpenAI-compatible HTTP endpoints and the Chora `ModelGatewayService` gRPC contract. It resolves logical model IDs from environment-based role settings and `config/models.yaml`, dispatches to OpenAI-compatible, Anthropic, Gemini, or Vertex upstreams, and supports text, image generation, embeddings, and hosted web search. The gateway also enforces tenant budgets and writes metered usage events to the `chora_observability` database's transactional outbox.

## Quick start

### Prerequisites

- Go 1.26+
- Docker and Docker Compose for the containerized deployment
- Checked-out sibling repositories for the `replace` directives in `go.mod`:
  - `../chora-contracts` (generated gRPC stubs)
  - `../chora-common` (shared library, module `github.com/apollo-chora/chora-common`)
- A migrated `chora_observability` PostgreSQL database
- Credentials for any upstream model providers you configure

From the parent directory:

```bash
git clone https://github.com/apollo-chora/chora-model-gateway.git
git clone https://github.com/apollo-chora/chora-contracts.git
git clone https://github.com/apollo-chora/chora-common.git
cd chora-model-gateway
```

For the Compose deployment:

```bash
cp .env.example .env
```

Set at least `CHORA_DATABASE_URL`, `CHORA_DEFAULT_TENANT_ID`, and the model role variables you need. The sample file uses these role settings:

```env
TEXT_LLM_BASE_URL=https://api.meta.ai/v1
TEXT_LLM_FORMAT=responses
TEXT_LLM_MODEL=muse-spark-1.3-contributor
TEXT_LLM_API_KEY=

IMAGE_LLM_BASE_URL=https://api.meta.ai/v1
IMAGE_LLM_FORMAT=images
IMAGE_LLM_MODEL=muse-image-1.0
IMAGE_LLM_API_KEY=

EMBEDDING_LLM_BASE_URL=https://api.openai.com/v1
EMBEDDING_LLM_FORMAT=embeddings
EMBEDDING_LLM_MODEL=text-embedding-3-small
EMBEDDING_LLM_API_KEY=
```

Start the gateway with:

```bash
docker compose up -d --build
```

The Compose file maps HTTP port `8080` and gRPC port `9090` by default. Check startup and readiness with:

```bash
docker compose logs -f gateway
curl http://localhost:8080/healthz
curl http://localhost:8080/readyz
curl http://localhost:8080/v1/models
```

To run locally without Docker:

```bash
go run ./cmd/server
```

The gateway itself does not run PostgreSQL migrations. Its Compose configuration expects `chora_observability` to already exist and be migrated.

## Usage

### HTTP

The HTTP service listens on `8080` by default.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/healthz` | Process health check |
| `GET` | `/readyz` | PostgreSQL readiness check |
| `GET` | `/v1/models` | List configured logical model IDs |
| `GET` | `/v1/models/{model}` | Look up one logical model |
| `POST` | `/v1/responses` | Responses-compatible text requests, including configured web search |
| `POST` | `/v1/chat/completions` | Chat Completions-compatible requests |
| `POST` | `/v1/completions` | Compatibility alias for Chat Completions |
| `POST` | `/v1/images/generations` | Image generation |
| `POST` | `/v1/images/edits` | Image compatibility surface |
| `POST` | `/v1/embeddings` | Embeddings |

A basic chat request:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "your-logical-model-id",
    "messages": [
      {"role": "user", "content": "Hello"}
    ]
  }'
```

The gateway resolves tenant and request context from Chora headers and environment defaults. `x-chora-tenant-id` may also be supplied as the `tenant` query parameter.

### gRPC

The gRPC server listens on `9090` by default and implements the `ModelGatewayService` declared in `chora-contracts/proto/services/model_gateway_service.proto`.

Implemented RPCs:

- `Invoke`
- `Embed`
- `GroundedSearch`

Grounded requests can also use `Invoke` with `response_modality=GROUNDED` or the HTTP `/v1/responses` endpoint.

### Model configuration

There are two configuration paths.

Environment role settings define one text, image, and embedding model with variables such as:

```text
TEXT_LLM_BASE_URL
TEXT_LLM_FORMAT
TEXT_LLM_MODEL
TEXT_LLM_API_KEY
TEXT_LLM_SUPPORT_GROUNDING

IMAGE_LLM_BASE_URL
IMAGE_LLM_FORMAT
IMAGE_LLM_MODEL
IMAGE_LLM_API_KEY

EMBEDDING_LLM_BASE_URL
EMBEDDING_LLM_FORMAT
EMBEDDING_LLM_MODEL
EMBEDDING_LLM_API_KEY
```

For multiple logical models, aliases, fallbacks, capabilities, endpoint overrides, and pricing, use `config/models.yaml`. Each registry entry declares a provider (`openai` or `anthropic`), upstream model, base URL, capabilities, and optional fallback, pricing, credential, and grounding settings.

For a self-hosted OpenAI-compatible server running on the host, Compose provides `host.docker.internal`. For example:

```env
TEXT_LLM_BASE_URL=http://host.docker.internal:11434/v1
TEXT_LLM_FORMAT=chat_completions
TEXT_LLM_MODEL=llama3.1:8b
TEXT_LLM_API_KEY=
TEXT_LLM_SUPPORT_GROUNDING=false
```

## Development

Build the service:

```bash
go build ./cmd/server
```

Run the quality checks:

```bash
go build ./...
go vet ./...
go test ./...
```

The main source tree is organized as follows:

```text
cmd/server/           Service entrypoint and wiring

internal/
  api/openai/         OpenAI-compatible HTTP API
  adapter/
    grpc/             ModelGatewayService implementation
    pg/               PostgreSQL budget, claim, and outbox operations
    vendors/          openai, anthropic, gemini, vertexembed upstreams
    clients/          Upstream provider clients
    middleware/       Metering and companion suspension
    modelarmor/       Cloud Model Armor policy integration
    registry/         Model registry resolver
    secrets/          Secret resolution
  domain/             Model resolution, budgets, fallbacks, and metering
  executor/           Dispatch executor
  registry/           Model registry loading

config/
  models.yaml         Logical model registry
```
