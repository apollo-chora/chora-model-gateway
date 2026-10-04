# chora-model-gateway

A local-first, provider-neutral model gateway for Chora.

The gateway exposes OpenAI-compatible HTTP endpoints and the existing Chora gRPC contract while centralising model configuration, tenant budgets, usage accounting, fallback routing, and grounded search.

The current runtime is Python 3.13 with **BlackSheep**, **Granian**, **Strands Agents**, **structlog**, **uv**, and PostgreSQL.

## Local Docker deployment

The recommended way to run the gateway is Docker Compose. A local deployment consists of:

- `gateway` — the model gateway on HTTP `8080` and gRPC `9090`
- `postgres` — local PostgreSQL with the Chora budget and usage-ledger schema

You only need Docker and credentials for the models you want to call.

### 1. Clone and configure

```bash
git clone https://github.com/apollo-chora/chora-model-gateway.git
cd chora-model-gateway
cp .env.example .env
```

Edit `.env` and configure the model roles you need.

A minimal text-only setup looks like:

```env
GATEWAY_HTTP_PORT=8080
GATEWAY_GRPC_PORT=9090
POSTGRES_HOST_PORT=5433

CHORA_DEFAULT_TENANT_ID=00000000-0000-7000-8000-000000000001
CHORA_DEFAULT_GCID=00000000-0000-7000-8000-000000000002

TEXT_LLM_BASE_URL=https://api.meta.ai/v1
TEXT_LLM_FORMAT=responses
TEXT_LLM_MODEL=muse-spark-1.3
TEXT_LLM_API_KEY=your-key
TEXT_LLM_SUPPORT_GROUNDING=true
```

The gateway does **not** infer providers from variable names such as `OPENAI_API_KEY` or `META_API_KEY`. The role configuration explicitly says which endpoint, protocol, and model to use.

### 2. Start it

```bash
docker compose up -d --build
```

Follow startup logs with:

```bash
docker compose logs -f gateway
```

Check the containers:

```bash
docker compose ps
```

The default local addresses are:

| Service | Address |
| --- | --- |
| HTTP API | `http://localhost:8080` |
| gRPC | `localhost:9090` |
| PostgreSQL | `localhost:5433` |
| Health | `http://localhost:8080/healthz` |
| Readiness | `http://localhost:8080/readyz` |

Change the host ports with `GATEWAY_HTTP_PORT`, `GATEWAY_GRPC_PORT`, and `POSTGRES_HOST_PORT`.

### 3. Verify the gateway

```bash
curl http://localhost:8080/healthz
curl http://localhost:8080/readyz
curl http://localhost:8080/v1/models
```

A text request:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "muse-spark-1.3",
    "messages": [
      {"role": "user", "content": "Hello from the local Chora gateway"}
    ]
  }'
```

Or use an OpenAI client:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-used-by-the-local-gateway",
)

response = client.chat.completions.create(
    model="muse-spark-1.3",
    messages=[{"role": "user", "content": "Hello"}],
)

print(response.choices[0].message.content)
```

The API key used by the gateway to contact the upstream model comes from the gateway's environment, not from the caller's OpenAI `Authorization` header.

## Model roles

For a straightforward local deployment, configure models by role.

### Text

```env
TEXT_LLM_BASE_URL=https://api.meta.ai/v1
TEXT_LLM_FORMAT=responses
TEXT_LLM_MODEL=muse-spark-1.3
TEXT_LLM_API_KEY=...
TEXT_LLM_SUPPORT_GROUNDING=true
```

Supported text formats are:

| `TEXT_LLM_FORMAT` | Upstream protocol |
| --- | --- |
| `responses` | OpenAI Responses-compatible API |
| `chat_completions` | OpenAI Chat Completions-compatible API |
| `messages` | Anthropic Messages-compatible API |

An OpenAI-compatible endpoint does not have to be OpenAI itself. Meta, self-hosted servers, proxies, and other compatible services can use the same protocol adapter.

### Images

```env
IMAGE_LLM_BASE_URL=https://api.meta.ai/v1
IMAGE_LLM_FORMAT=images
IMAGE_LLM_MODEL=muse-image-1.0
IMAGE_LLM_API_KEY=...
```

Image generation is exposed at:

```text
POST /v1/images/generations
POST /v1/images/edits
```

Generated image data is returned as `b64_json`.

### Embeddings

```env
EMBEDDING_LLM_BASE_URL=https://api.openai.com/v1
EMBEDDING_LLM_FORMAT=embeddings
EMBEDDING_LLM_MODEL=text-embedding-3-small
EMBEDDING_LLM_API_KEY=...
```

Embeddings are exposed at:

```text
POST /v1/embeddings
```

Both a single string and an array of strings are accepted.

You can configure only the roles you actually use.

## Grounded web search

Grounding is part of the text-model configuration:

```env
TEXT_LLM_SUPPORT_GROUNDING=true
```

When the configured model supports native hosted search, the gateway uses that capability directly. For an OpenAI Responses-compatible model this means a native `web_search` tool can be sent with the model request.

Example:

```bash
curl http://localhost:8080/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "muse-spark-1.3",
    "input": "Who won the most recent Formula 1 race?",
    "tools": [{"type": "web_search"}]
  }'
```

The default external search provider setting is:

```env
SEARCH_PROVIDER=exa
EXA_API_KEY=
```

The intent is **native grounding first**. Grounding requires the configured
model to support hosted web search natively: the gateway sends the provider's
own `web_search` tool and never silently downgrades a grounded request to an
ungrounded one. A model that does not advertise the `web_search` capability is
refused with `grounding_unavailable` (or `wrong_grounding_surface` when the
model grounds on a different surface), matching the gateway's historical
behavior. `SEARCH_PROVIDER` / `EXA_API_KEY` are retained in the deployment
configuration for an external search fallback, but no fallback is dispatched
unless the model can serve the request.

## Using the model registry

Role-based environment configuration is the easiest deployment path, but `config/models.yaml` remains available for installations that need multiple logical models, aliases, fallbacks, capabilities, or per-model pricing.

The registry is mounted read-only by Compose:

```text
./config/models.yaml -> /app/config/models.yaml
```

After editing it:

```bash
docker compose restart gateway
```

Environment role models are loaded alongside registry models and take precedence for the same model ID.

Legacy registry entries can still refer to credential variables such as `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `META_API_KEY`. Those variables exist for registry compatibility; new simple deployments should normally prefer the role-specific `*_LLM_API_KEY` settings.

## Self-hosted models

Compose maps `host.docker.internal` back to the Docker host, including on Linux:

```yaml
extra_hosts:
  - host.docker.internal:host-gateway
```

That lets the gateway container call Ollama, vLLM, LM Studio, or another OpenAI-compatible server running directly on your machine.

For example:

```env
TEXT_LLM_BASE_URL=http://host.docker.internal:11434/v1
TEXT_LLM_FORMAT=chat_completions
TEXT_LLM_MODEL=llama3.1:8b
TEXT_LLM_API_KEY=
TEXT_LLM_SUPPORT_GROUNDING=false
```

Then:

```bash
docker compose up -d --build
```

No cloud service is required.

## HTTP API

The existing public HTTP surface is retained:

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/v1/responses` | Responses-compatible text and grounded requests |
| `POST` | `/v1/chat/completions` | Chat Completions-compatible requests |
| `POST` | `/v1/completions` | compatibility alias |
| `POST` | `/v1/images/generations` | image generation |
| `POST` | `/v1/images/edits` | image compatibility surface |
| `POST` | `/v1/embeddings` | embeddings |
| `GET` | `/v1/models` | available logical model IDs |
| `GET` | `/v1/models/{model}` | model lookup |
| `GET` | `/healthz` | process health |
| `GET` | `/readyz` | always `ready` once the process is serving |

Streaming is currently not implemented. Requests with `stream: true` return an explicit `501 streaming_unsupported` error instead of pretending to provide SSE.

## gRPC compatibility

The container also listens on `9090` and retains the existing `ModelGatewayService` contract.

Supported RPCs:

- `Invoke`
- `Embed`

The old dedicated `GroundedSearch` RPC remains `UNIMPLEMENTED`. Grounding now goes through ordinary `Invoke` with the grounded modality or through `/v1/responses`.

The protobuf source is in `proto/model_gateway_service.proto`; Python stubs are generated during the Docker build.

## PostgreSQL, budgets, and usage

The Compose deployment initializes PostgreSQL from `migrations/`.

The important tables are:

| Table | Purpose |
| --- | --- |
| `per_tenant_llm_budget` | tenant budget and exhaustion policy |
| `token_usage_ledger` | usage/accounting record for completed model calls |
| `invoke_debit_claims` | idempotent debit claims |

The gateway connects as the non-superuser `chora_app` role so PostgreSQL row-level security remains effective.

Inspect recent ledger entries:

```bash
docker compose exec -T -e PGPASSWORD=chora_app postgres \
  psql -h 127.0.0.1 -U chora_app -d chora_observability \
  -c "BEGIN;
      SET LOCAL chora.tenant_id='00000000-0000-7000-8000-000000000001';
      SELECT model_id, input_tokens, output_tokens, cost_usd_micros, recorded_at
      FROM token_usage_ledger
      ORDER BY recorded_at DESC
      LIMIT 10;
      COMMIT;"
```

Budget policies are `block`, `downgrade`, and `alert`.

## Common Docker operations

Start or rebuild:

```bash
docker compose up -d --build
```

View gateway logs:

```bash
docker compose logs -f gateway
```

Restart after configuration changes:

```bash
docker compose restart gateway
```

Stop:

```bash
docker compose down
```

Stop and delete the local PostgreSQL data volume:

```bash
docker compose down -v
```

The last command permanently removes local gateway database data.

## Troubleshooting

### Gateway is not ready

```bash
docker compose ps
docker compose logs postgres
docker compose logs gateway
curl -i http://localhost:8080/readyz
```

`/readyz` answers `ready` once the process is serving. Budget and ledger
writes go to PostgreSQL; if the database is unreachable the gateway still
serves requests, but settlements fail (see the logs).

### Upstream model cannot be reached

Remember that `localhost` inside the gateway container means the gateway container itself.

For a model server running on your host, use:

```text
http://host.docker.internal:<port>
```

### Model is not found

Check:

```bash
curl http://localhost:8080/v1/models
```

The requested `model` must resolve either to a role-configured model or an entry/alias in `config/models.yaml`.

### Provider authentication fails

For role-based configuration, check the corresponding variable:

```text
TEXT_LLM_API_KEY
IMAGE_LLM_API_KEY
EMBEDDING_LLM_API_KEY
```

For registry entries, check the variable named by that entry's `api_key_env`.

### Start from a clean database

```bash
docker compose down -v
docker compose up -d --build
```

Migrations under `migrations/` are applied by PostgreSQL only when the data directory is first initialized.

## Development

Dependencies and tooling are managed by `uv`:

```bash
uv sync --all-groups
```

Run the same quality checks used in CI:

```bash
uv run ruff check app tests
uv run ruff format --check app tests
uv run mypy
uv run bandit -c pyproject.toml -r app
uv run pytest --cov=app --cov-report=term-missing
```

The production container is served by Granian:

```text
granian --interface asgi --host 0.0.0.0 --port 8080 app.gateway:app
```

## Project layout

```text
app/
  config.py         environment + registry configuration
  database.py       PostgreSQL budget, ledger, and idempotency operations
  runtime.py        provider dispatch (chat/responses/messages/images/embeddings)
  service.py        gateway orchestration, budgets, fallbacks, accounting
  http.py           OpenAI-compatible BlackSheep HTTP surface
  grpc_server.py    Chora gRPC compatibility surface
  gateway.py        application startup and shutdown

config/models.yaml
migrations/
proto/
tests/
Dockerfile
docker-compose.yml
pyproject.toml
```

## Docker image

The repository publishes:

```text
walfa/chora-model-gateway
```

For local development, `docker compose up --build` builds from the checked-out source. To use a published image instead, remove or override the Compose `build: .` setting and use the image tag you want.

---

The deployment is intentionally local-first: PostgreSQL, explicit environment configuration, and model endpoints you control. There is no required GCP runtime, Secret Manager, workload identity, or managed Pub/Sub dependency.
