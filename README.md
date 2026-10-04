# chora-model-gateway

A provider-agnostic LLM gateway. Every model call on the platform goes through
it, so it is the one place where cost is metered, budgets are enforced, and
usage is recorded.

It speaks **two** surfaces:

| Surface | Port | What it is |
|---|---|---|
| gRPC | `9090` | `ModelGatewayService.Invoke` + `.Embed` |
| HTTP | `8080` | an **OpenAI-compatible** API — point any OpenAI client at it |

Its infrastructure is **Postgres plus the environment**. No cloud account, no
credential file, no cluster.

> This service was a GCP/GKE workload (Vertex AI + Cloud Model Armor + Secret
> Manager + Pub/Sub) until 2026-10-04, when the GCP path was removed outright.
> See [Retired from GCP](#retired-from-gcp) for what changed and why.

---

## Quick start

```bash
cp .env.example .env      # fill in whichever provider keys you have
docker compose up --build
```

This repository is self-contained: a fresh clone builds, tests and runs with
nothing else checked out. The one piece of upstream coupling - the generated
`ModelGatewayService` stubs - is vendored under [`gen/`](gen/), with the
re-sync command documented in [`go.mod`](go.mod).

Then:

```bash
curl http://localhost:18080/v1/models
```

Host ports are `18080` (HTTP) and `19090` (gRPC) by default — change them in
`.env` if those are taken. The container always listens on `8080`/`9090`.

### Point a real OpenAI client at it

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:18080/v1",
    api_key="not-used-but-required-by-the-sdk",
)

resp = client.chat.completions.create(
    model="gpt-4o",              # a name from config/models.yaml
    messages=[{"role": "user", "content": "hello"}],
)
print(resp.choices[0].message.content)
```

---

## The OpenAI-compatible API

| Method | Path | Notes |
|---|---|---|
| `POST` | `/v1/responses` | the Responses API — **this is where hosted web search lives** |
| `POST` | `/v1/chat/completions` | text, tool calls, vision content parts |
| `POST` | `/v1/images/generations` | returns `b64_json`; `response_format: url` is refused |
| `POST` | `/v1/embeddings` | string or array input; `encoding_format: base64` is refused |
| `GET` | `/v1/models` | the registry as OpenAI model objects |
| `GET` | `/v1/models/{id}` | one entry |
| `GET` | `/healthz`, `/readyz` | probes |

Everything not in the table above (`/v1/completions` as an alias,
`/v1/images/edits`) maps onto the same handlers.

Hosted **web search** is available too, but it is opt-in per model rather than
universal — see [Grounded search](#grounded-search-optional).

### What is deliberately NOT supported

Each of these returns an explicit error rather than a plausible-looking wrong
answer:

- **Streaming.** `stream: true` → `501 streaming_unsupported`. Returning a
  non-streaming body to a client expecting SSE produces a hang, not an error.
- **URL image responses.** The gateway returns bytes and hosts nothing, so a
  URL could only point somewhere that does not exist.
- **Base64 embeddings.** Float arrays are what clients actually handle.
- **Token-array embedding input.** Only strings and string arrays.

### Errors

Every non-2xx response uses the OpenAI error envelope, so a client parses
errors the same way it parses successes:

```json
{"error": {"message": "...", "type": "...", "code": "..."}}
```

Status mapping:

| Situation | Status | `code` |
|---|---|---|
| Model not in the registry | `404` | `model_not_found` |
| Malformed or incomplete request | `400` | `invalid_request` / `invalid_json` |
| Tenant budget exhausted | `402` | `budget_exhausted` |
| Upstream said 401 / 429 / 404 | *relayed as-is* | `upstream_error` |
| Gateway misconfiguration (e.g. a configured-but-empty API key) | `500` | `gateway_misconfigured` |
| Provider unreachable, chain exhausted | `502` | `vendor_error` |
| Unsupported feature (streaming, url format) | `501` | `streaming_unsupported` etc. |

A provider's own status is **relayed rather than flattened**: a `401` from the
provider means the key is wrong, and reporting that as `502 bad gateway` sends
the caller looking in the wrong place.

### Attribution

Every request is attributed to a tenant and an actor, because the budget row is
tenant-scoped and an unattributed call cannot be settled.

| Header / field | Default |
|---|---|
| `X-Chora-Tenant-Id` | `CHORA_DEFAULT_TENANT_ID` |
| `X-Chora-Gcid` | `CHORA_DEFAULT_GCID` |
| `user` (the OpenAI-native field) | becomes the ledger row's role tag |

---

## Grounded search (optional)

Web search is **opt-in per model**. A registry entry without a `grounding:`
block refuses a grounded request rather than running an ungrounded call that
would return an answer which only *looks* researched.

There is no single "web search" wire format — OpenAI puts its tool on
`/v1/responses` as `{"type":"web_search"}`, Anthropic puts a differently-named
server tool on `/v1/messages` as
`{"type":"web_search_20250305","name":"web_search","max_uses":5}`, and the
OpenAI-compatible aggregators pick either. So you **ask** for `web_search` in the
abstract and the **registry** decides what actually goes on the wire. A client
written against the gateway keeps working when you swap providers.

### Calling it

Exactly the request from the docs, pointed at the gateway:

```bash
curl -X POST "http://localhost:18080/v1/responses" \
  -H "Authorization: Bearer $MODEL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "muse-spark",
    "input": "Who won the most recent Formula 1 race?",
    "tools": [{"type": "web_search"}]
  }'
```

The reply is the Responses shape — a `web_search_call` item, then a `message`
with the answer and its `url_citation` annotations — so an unmodified OpenAI
Responses client reads it as normal:

```json
{
  "id": "01a1...", "object": "response", "model": "muse-spark-1.3",
  "output": [
    {"type": "web_search_call", "status": "completed",
     "action": {"type": "search", "query": "most recent Formula 1 race"}},
    {"type": "message", "role": "assistant", "content": [{
      "type": "output_text",
      "text": "…",
      "annotations": [{"type": "url_citation", "url": "https://…", "title": "…"}]}]}
  ],
  "chora_gateway": {
    "grounded": true,
    "grounding_surface": "responses",
    "citations": [{"url": "https://…", "title": "…"}],
    "search_queries": ["most recent Formula 1 race"]
  }
}
```

`chora_gateway.citations` is the **normalised** list. OpenAI's annotations and
Anthropic's `citations` are structurally different; both are parsed into this
one shape so you don't need a branch per vendor.

### Configuring it

```yaml
- id: claude-sonnet-5-search
  provider: anthropic
  base_url: https://api.anthropic.com
  api_key_env: ANTHROPIC_API_KEY
  capabilities: [chat, tools, vision, web_search]
  grounding:
    surface: messages          # responses | chat_completions | messages
    tool_type: web_search_20250305
    tool_name: web_search
    max_uses: 5               # a SPEND control: web search is priced per search
    extra_tool_fields:        # provider options the gateway does not model
      allowed_domains: ["example.com"]
```

Every field is optional and the defaults follow the provider family, so an
`openai` entry can just be:

```yaml
capabilities: [chat, web_search]
grounding: {}
```

…which grounds on `/v1/responses` with `{"type":"web_search"}`. An `anthropic`
entry grounds on `/v1/messages` with its dated server tool.

The shipped registry has worked examples for OpenAI, Anthropic, Meta's
`api.meta.ai` (`muse-spark` + `muse-image`), and a self-hosted server on the
chat-completions surface.

### Verified against a live provider

Everything below was observed against `https://api.meta.ai/v1` on 2026-10-04,
not inferred from docs — and three of the four points contradict what the
OpenAI documentation says:

- **`muse-spark-1.3` is a reasoning model.** A 16-token request came back with
  13 reasoning tokens, `content: null` and `finish_reason: "length"`. An output
  ceiling is load-bearing: set it low and every call returns an empty answer.
- **`/v1/responses` returns `output[0]` as a `reasoning` item**, with the
  message later. This is why the gateway parses the array *by type*.
- **The images endpoint returns WebP, not PNG**, despite what OpenAI documents.
  The gateway sniffs the magic bytes and reports `mime_type`, so a caller is
  never handed a WebP labelled `image/png`.
- **A grounded run returns several `message` items**, not one: the model
  narrates between its searches and answers last. They are preserved as separate
  output items with per-message citations — joining them yields narration soup
  with no answer in it.

A multi-step grounded run also needs a large ceiling. A population question
needed three searches plus two page opens and did **not** finish inside 1500
output tokens; 4000 completed. When it does not finish, the provider returns
`status: "incomplete"`, which the gateway mirrors rather than reporting a
confident-looking non-answer.

### Notes worth knowing

- **Grounding is a normal dispatch.** It goes through the same budget gate,
  ledger and fallback chain. A budget block returns `402` **before** the search
  is sent — verified, not assumed.
- **The ledger records `modality = 'GROUNDED'`** so a search-heavy tenant is
  identifiable. Most providers price search *per request*, not per token, so a
  token-only cost table under-reports it; the token figure is a floor.
- **`max_uses` is a spend control** on Anthropic, which bills per search. An
  unbounded tool on a single request is unbounded spend.
- **Grounding from `/v1/chat/completions`** works only for an entry configured
  with `surface: chat_completions`. OpenAI does *not* offer hosted search there,
  so asking on the wrong surface returns `400 wrong_grounding_surface` pointing
  you at `/v1/responses` rather than dispatching a tool the provider ignores.
- **Anthropic's server tool needs no separate endpoint** — it is a tool on
  `/v1/messages`, so the request path is unchanged.
- **Reasoning items are not reproduced.** The gateway models the text and the
  citations, not the chain of thought, so a Responses client gets the answer and
  citations but no `reasoning` output items.
- **Citation URLs expire.** OpenAI's are redirect links with a ~30-day life.
  Treat them as display-only.

---

## Configuring models

**`config/models.yaml` is the catalogue.** Adding a model is an edit to that
file — not a code change and not a rebuild. It is mounted read-only, so
editing it is a `docker compose restart gateway`.

```yaml
models:
  - id: gpt-4o                    # the name callers pass as `model`
    provider: openai              # openai | anthropic
    upstream_model: gpt-4o-mini    # the name sent upstream (defaults to id)
    base_url: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY   # omit entirely for a server needing no key
    context_window: 128000
    max_output_tokens: 16384      # the ceiling; callers are clamped to it
    capabilities: [chat, tools, vision, embeddings]
    fallback_ids: [gpt-4o]        # tried in order if this one fails
    aliases: [quick]              # extra names this entry answers to
    pricing:                      # USD micros per million tokens
      input_per_mtok_usd_micros: 150000
      output_per_mtok_usd_micros: 600000
      cached_per_mtok_usd_micros: 75000
      cache_write_per_mtok_usd_micros: 187500
```

Recognised capabilities: `chat`, `tools`, `vision`, `image`, `embeddings`.
Omitting the list means chat-only. A registry file with an unknown provider, an
unknown capability, or an unresolvable endpoint **fails at boot** rather than on
the first request that happens to touch it.

### Endpoint resolution — including a fully custom endpoint

There are two ways to name a server, and this is the part most providers get
subtle.

**1. `base_url` + a relative path** (the usual case) — the path is joined:

```yaml
base_url: https://api.openai.com/v1
# -> https://api.openai.com/v1/chat/completions
```

**2. A fully custom chat-completions endpoint** — put an **absolute URL** in
the path field and `base_url` is ignored for that surface:

```yaml
base_url: https://ignored.example.com/v1
chat_completions_path: https://llm.internal.corp:8443/generate
# -> https://llm.internal.corp:8443/generate
```

Only `http://` and `https://` count as absolute overrides. A `grpc://` path
passes no validation and is rejected at boot, because it would otherwise be
silently concatenated onto `base_url` and produce a nonsense URL that only
fails at dispatch time — far from the config error.

Per-surface paths: `chat_completions_path`, `messages_path` (Anthropic),
`images_path`, `embeddings_path`.

An Anthropic entry's primary endpoint is `/v1/messages`, **not**
`/chat/completions` — the boot log prints the URL that will really be called.

### Self-hosted models

`local-llama` and `local-vllm` in the shipped registry point at
`host.docker.internal`, which `docker-compose.yml` maps so a containerised
gateway can reach a server on your machine:

| Server | Start it | Registry entry |
|---|---|---|
| Ollama | `ollama serve` | `local-llama` |
| vLLM | `vllm serve Qwen/Qwen2.5-7B-Instruct` | `local-vllm` |
| LM Studio | enable its local server | add a `base_url` pointing at it |

These declare no `api_key_env`, which is exactly how the gateway knows they
need no credential.

---

## Cost ledger and budget

Three tables, all in `migrations/`:

| Table | Holds |
|---|---|
| `per_tenant_llm_budget` | the spend ceiling, read under `FOR UPDATE` |
| `token_usage_ledger` | one row per completed dispatch |
| `invoke_debit_claims` | keyed idempotency: a redelivered dispatch bills once |

### Two invariants

**The debit and the ledger row commit together.** `Repo.Settle` opens one
transaction. A crash between two separate transactions would leave either
invisible spend or free spend, and neither is recoverable after the fact.

**Row-level security is real, not decorative.** Every transaction begins with
`SET LOCAL chora.tenant_id`, and the policies in `0001_initial.sql` filter on
that GUC.

> ⚠️ **The gateway must NOT connect as a superuser.** PostgreSQL exempts
> superusers from RLS entirely, regardless of `FORCE ROW LEVEL SECURITY` — a
> gateway running as the compose file's `chora` user would see every tenant's
> budget, and the isolation would be a comment. `0002_app_role.sql` creates
> `chora_app`, an ordinary role with no `BYPASSRLS`, and `docker-compose.yml`
> connects as that. The migration asserts the exemption is absent, so a future
> grant fails at migration time rather than leaking in production.

### Inspecting the ledger

```bash
docker compose exec -T -e PGPASSWORD=chora_app postgres \
  psql -h 127.0.0.1 -U chora_app -d chora_observability \
  -c "BEGIN; SET LOCAL chora.tenant_id='00000000-0000-7000-8000-000000000001';
      SELECT model_id, input_tokens, output_tokens, cost_usd_micros, recorded_at
      FROM token_usage_ledger ORDER BY recorded_at DESC LIMIT 10; COMMIT;"
```

The `SET LOCAL` is required — that is RLS working, not ceremony.

### Budget policies

| `policy` | Behaviour when the budget is exhausted |
|---|---|
| `block` (and any unrecognised value) | refuse with `402`; the provider is never called |
| `downgrade` | re-resolve `downgrade_to_logical_model_id` and dispatch that |
| `alert` | proceed |

An unrecognised policy defaults to **block** — refusing is the safe direction.
A downgrade to a model that is not in the registry fails loudly rather than
silently dispatching the expensive model the budget exists to avoid.

---

## Environment variables

| Variable | Required | Default | Meaning |
|---|---|---|---|
| `CHORA_DATABASE_URL` | **yes** | — | Postgres DSN |
| `CHORA_DEFAULT_TENANT_ID` | **yes** | — | tenant for calls that do not name one |
| `CHORA_DEFAULT_GCID` | no | tenant id | actor for calls that do not name one |
| `CHORA_MODEL_REGISTRY` | no | `config/models.yaml` | the catalogue |
| `CHORA_GRPC_PORT` | no | `9090` | gRPC listen port |
| `CHORA_HTTP_PORT` | no | `8080` | HTTP listen port |
| `CHORA_BOOTSTRAP_TIMEOUT` | no | `30s` | how long boot waits for the database |
| `CHORA_VENDOR_HTTP_TIMEOUT` | no | `120s` | per-call upstream budget |
| `CHORA_DEFAULT_AGENT_ID` | no | `openai_compat` | ledger role for unattributed calls |
| `OPENAI_API_KEY` etc. | per entry | — | named by a registry entry's `api_key_env` |

A registry entry whose `api_key_env` is **set but empty** is refused at request
time with `gateway_misconfigured`. That is deliberate: silently sending an
unauthenticated request to a public endpoint is worse than a clear refusal.

---

## Verifying it works

The shipped registry includes `stub` and `stub-authed`, which point at a local
stub rather than a real provider, so the whole path can be exercised with no
credentials:

```bash
python3 tools/stub_openai.py &          # or: python3 -m http.server style stub
docker compose up -d

curl localhost:18080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"stub","messages":[{"role":"user","content":"hi"}]}'
```

```bash
# Unit + integration
go test ./...
go test -race -count=1 ./...

# Coverage, per the repo floors (domain 85%, adapters 60%)
go test -coverprofile=/tmp/c.out -coverpkg=./internal/domain ./internal/domain
go tool cover -func=/tmp/c.out | tail -1
```

---

## Layout

```
cmd/server/main.go              composition root — no cloud wiring
internal/
  domain/                       pure logic, zero infra imports
    service.go                  the Invoke flow
    service_embed.go            the Embed flow
    policy.go                   TargetModel + endpoint resolution
    ports.go                    the boundary interfaces
    budget.go                   budget policy decisions
  adapter/
    httpapi/                    the OpenAI-compatible surface
    grpc/                       the proto surface
    pg/                         budget + ledger + claims
    policy/                     registry → domain.PolicyLoader
    secrets/                    env-backed credential resolution
    vendors/openai/             chat + images + embeddings
    vendors/anthropic/          the messages API
config/models.yaml              the model catalogue
migrations/                     the schema
```

The dependency direction is adapter → domain and never the reverse; `domain/`
imports no infrastructure at all.

---

## Retired from GCP

Removed on 2026-10-04, with the GKE manifests, the `m15-model-gateway-iam`
Terraform module, the Cloud Build/Cloud Deploy configs, and the OpenAI/Anthropic
BYOA secret plumbing:

| Was | Now |
|---|---|
| Vertex AI Gemini + Gemma LoRA | registry-driven OpenAI-compatible providers |
| Cloud Model Armor (4 screening legs) | none — the guardrail tiering never had a working caller anyway |
| Secret Manager | environment variables named by the registry |
| Pub/Sub `chora.observability.token_usage.recorded.v1` | the `token_usage_ledger` table, read directly |
| Cloud Trace OTLP | structured JSON logs |
| identity `ManaService` metering | `per_tenant_llm_budget` |
| Cloud SQL + `cloudsql-proxy` sidecar | plain Postgres |
| `GroundedSearch` RPC | removed with its Google-Search-grounding provider |

**Capability note — grounding came back.** The old `GroundedSearch` RPC is gone,
because it was Google Search grounding routed through Gemini and there is no
equivalent on the providers now supported. Grounded search is instead a
**registry-driven capability** on the ordinary dispatch path
(`response_modality: GROUNDED`, `/v1/responses`), which is strictly better: it
works on any provider whose API offers a hosted search tool, and it no longer
needs a bespoke RPC or a dedicated Gemini dependency. It returns gRPC
`Unimplemented` only on the old RPC name. See
[Grounded search](#grounded-search-optional).

`Embed` survives, reimplemented against the OpenAI-compatible `/embeddings`
surface.