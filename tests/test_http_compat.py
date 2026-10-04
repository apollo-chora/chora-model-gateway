from __future__ import annotations

import base64

import pytest

from tests.conftest import Client, body, make_app, make_models


@pytest.fixture
async def client(models, db, provider, chat_completion):
    provider.route("https://api.example.test/v1/chat/completions", chat_completion)
    provider.route("https://api.example.test/v1/responses", chat_completion)
    provider.route(
        "https://api.example.test/v1/images/generations",
        {
            "created": 1,
            "data": [{"b64_json": base64.b64encode(b"\x89PNG-bytes").decode()}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
        },
    )
    provider.route(
        "https://api.example.test/v1/embeddings",
        {
            "model": "embedder-upstream",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.5, 0.25]}],
            "usage": {"prompt_tokens": 4, "total_tokens": 4},
        },
    )
    app = make_app(models, db, provider)
    await app.start()
    try:
        yield Client(app)
    finally:
        await app.stop()


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_healthz(client):
    response = await client.get("/healthz")
    assert response.status == 200
    assert response.content.body == b"ok"


@pytest.mark.anyio
async def test_readyz_is_always_ready_even_without_a_database(client):
    # The old gateway's readyz never consults the database.
    response = await client.get("/readyz")
    assert response.status == 200
    assert response.content.body == b"ready"


@pytest.mark.anyio
async def test_nosniff_header_on_every_response(client):
    for path in ("/healthz", "/v1/models"):
        response = await client.get(path)
        assert b"x-content-type-options" in response.headers
        assert response.headers.get_first(b"x-content-type-options") == b"nosniff"


# ---------------------------------------------------------------------------
# GET /v1/models
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_models_list_carries_the_additive_metadata(client):
    response = await client.get("/v1/models")
    assert response.status == 200
    data = body(response)
    assert data["object"] == "list"
    assert len(data["data"]) == len(make_models())
    chatty = next(m for m in data["data"] if m["id"] == "chatty")
    assert chatty["object"] == "model"
    assert chatty["created"] == 0
    assert chatty["owned_by"] == "openai"
    assert chatty["context_window"] == 8000
    assert chatty["max_output_tokens"] == 512
    assert chatty["capabilities"] == ["chat", "tools"]
    assert chatty["base_url"] == "https://api.example.test/v1"


@pytest.mark.anyio
async def test_models_single_retrieval(client):
    response = await client.get("/v1/models/chatty")
    assert response.status == 200
    assert body(response)["id"] == "chatty"


@pytest.mark.anyio
async def test_models_unknown_model_is_404(client):
    response = await client.get("/v1/models/ghost")
    assert response.status == 404
    assert body(response)["error"]["code"] == "model_not_found"


@pytest.mark.anyio
async def test_models_wrong_method_is_405_envelope(client):
    response = await client.post("/v1/models", payload={"x": 1})
    assert response.status == 405
    error = body(response)["error"]
    assert error["code"] == "method_not_allowed"
    assert error["type"] == "invalid_request_error"


# ---------------------------------------------------------------------------
# POST /v1/chat/completions
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_chat_happy_path(client):
    response = await client.post(
        "/v1/chat/completions", payload={"model": "chatty", "messages": [{"role": "user", "content": "hi"}]}
    )
    assert response.status == 200
    data = body(response)
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["usage"]["total_tokens"] == 5
    assert data["chora_gateway"]["vendor"] == "openai"
    assert data["chora_gateway"]["invocation_id"] == data["id"]


@pytest.mark.anyio
async def test_chat_strips_system_and_developer_prompts(client, provider):
    await client.post(
        "/v1/chat/completions",
        payload={
            "model": "chatty",
            "messages": [
                {"role": "system", "content": "be terse"},
                {"role": "developer", "content": "and precise"},
                {"role": "user", "content": "hi"},
            ],
        },
    )
    _, payload, _, _ = provider.last
    assert payload["messages"] == [{"role": "user", "content": "hi"}]


@pytest.mark.anyio
async def test_chat_forwards_generation_config(client, provider):
    await client.post(
        "/v1/chat/completions",
        payload={
            "model": "chatty",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": 123,
            "seed": 7,
            "stop": ["END"],
        },
    )
    _, payload, _, _ = provider.last
    assert payload["temperature"] == 0.2
    assert payload["top_p"] == 0.9
    assert payload["max_tokens"] == 123
    assert payload["seed"] == 7
    assert payload["stop"] == ["END"]
    assert payload["stream"] is False


@pytest.mark.anyio
async def test_chat_stop_string_form(client, provider):
    await client.post(
        "/v1/chat/completions",
        payload={
            "model": "chatty",
            "messages": [{"role": "user", "content": "hi"}],
            "stop": "END",
        },
    )
    _, payload, _, _ = provider.last
    assert payload["stop"] == "END"


@pytest.mark.anyio
async def test_chat_max_completion_tokens_becomes_max_tokens(client, provider):
    await client.post(
        "/v1/chat/completions",
        payload={
            "model": "chatty",
            "messages": [{"role": "user", "content": "hi"}],
            "max_completion_tokens": 55,
        },
    )
    _, payload, _, _ = provider.last
    assert payload["max_tokens"] == 55


@pytest.mark.anyio
async def test_chat_forwards_tools(client, provider):
    await client.post(
        "/v1/chat/completions",
        payload={
            "model": "chatty",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
        },
    )
    _, payload, _, _ = provider.last
    assert payload["tools"][0]["function"]["name"] == "lookup"


@pytest.mark.anyio
async def test_chat_tool_calls_surface(client, provider):
    provider.route(
        "https://api.example.test/v1/chat/completions",
        {
            "id": "chat-2",
            "model": "chatty-upstream",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        },
    )
    response = await client.post(
        "/v1/chat/completions",
        payload={
            "model": "chatty",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    data = body(response)
    assert data["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "lookup"
    # The old facade maps every non-length finish reason to "stop".
    assert data["choices"][0]["finish_reason"] == "stop"


@pytest.mark.anyio
async def test_chat_attribution_headers_and_user(client, db, provider):
    app = make_app(make_models(), db, provider)
    await app.start()
    try:
        response = await Client(app).post(
            "/v1/chat/completions",
            payload={"model": "chatty", "messages": [{"role": "user", "content": "hi"}], "user": "alice"},
            headers={"X-Chora-Tenant-Id": "tenant-42", "X-Chora-Gcid": "gcid-99"},
        )
    finally:
        await app.stop()
    assert response.status == 200
    event = db.ledger[0]
    assert event["tenant_id"] == "tenant-42"
    assert event["gcid"] == "gcid-99"
    assert event["agent_role"] == "alice"


@pytest.mark.anyio
async def test_chat_tenant_query_fallback(client, db, provider):
    app = make_app(make_models(), db, provider)
    await app.start()
    try:
        response = await Client(app).post(
            "/v1/chat/completions?tenant=tenant-77",
            payload={"model": "chatty", "messages": [{"role": "user", "content": "hi"}]},
        )
    finally:
        await app.stop()
    assert response.status == 200
    assert db.ledger[0]["tenant_id"] == "tenant-77"


@pytest.mark.anyio
async def test_chat_trace_headers_reach_the_provider(client, provider):
    await client.post(
        "/v1/chat/completions",
        payload={"model": "chatty", "messages": [{"role": "user", "content": "hi"}]},
        headers={"traceparent": "00-abc-def-01", "tracestate": "foo=bar"},
    )
    _, _, headers, _ = provider.last
    assert headers["traceparent"] == "00-abc-def-01"
    assert headers["tracestate"] == "foo=bar"


@pytest.mark.anyio
async def test_chat_unknown_model_is_404(client):
    response = await client.post(
        "/v1/chat/completions", payload={"model": "ghost", "messages": [{"role": "user", "content": "hi"}]}
    )
    assert response.status == 404
    error = body(response)["error"]
    assert error["code"] == "model_not_found"
    assert "GET /v1/models" in error["message"]


@pytest.mark.anyio
async def test_chat_streaming_is_refused_explicitly(client, db):
    response = await client.post(
        "/v1/chat/completions",
        payload={"model": "chatty", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert response.status == 501
    error = body(response)["error"]
    assert error["code"] == "streaming_unsupported"
    assert error["type"] == "invalid_request_error"
    assert db.ledger == []


@pytest.mark.anyio
async def test_chat_system_only_is_rejected(client, db):
    response = await client.post(
        "/v1/chat/completions", payload={"model": "chatty", "messages": [{"role": "system", "content": "be terse"}]}
    )
    assert response.status == 400
    assert body(response)["error"]["code"] == "empty_conversation"
    assert db.ledger == []


@pytest.mark.anyio
async def test_chat_missing_fields(client):
    for payload in (
        {"messages": [{"role": "user", "content": "hi"}]},
        {"model": "chatty"},
        {"model": "chatty", "messages": []},
    ):
        response = await client.post("/v1/chat/completions", payload=payload)
        assert response.status == 400, payload


@pytest.mark.anyio
async def test_chat_malformed_json(client):
    response = await client.post("/v1/chat/completions", content=b'{"model":')
    assert response.status == 400
    error = body(response)["error"]
    assert error["code"] == "invalid_json"
    assert error["message"].startswith("could not parse request body:")


@pytest.mark.anyio
async def test_chat_wrong_method_is_405(client):
    response = await client.get("/v1/chat/completions")
    assert response.status == 405


@pytest.mark.anyio
async def test_legacy_completions_path(client):
    response = await client.post(
        "/v1/completions", payload={"model": "chatty", "messages": [{"role": "user", "content": "hi"}]}
    )
    assert response.status == 200


@pytest.mark.anyio
async def test_chat_budget_block_is_402(client, db, provider):
    db.budget_state = {
        "budget_usd_micros": 100,
        "spent_usd_micros": 200,
        "policy": "block",
        "downgrade_to_logical_model_id": None,
    }
    response = await client.post(
        "/v1/chat/completions", payload={"model": "chatty", "messages": [{"role": "user", "content": "hi"}]}
    )
    assert response.status == 402
    error = body(response)["error"]
    assert error["code"] == "budget_exhausted"
    assert error["type"] == "insufficient_quota"
    assert db.ledger == []


@pytest.mark.anyio
async def test_chat_config_error_is_500(client, models, db, provider):
    # A configured-but-empty credential is a misconfiguration, not a 502.
    models["chatty"] = make_models()["chatty"]
    models["chatty"].api_key_env = "MISSING_KEY"
    response = await client.post(
        "/v1/chat/completions", payload={"model": "chatty", "messages": [{"role": "user", "content": "hi"}]}
    )
    assert response.status == 500
    error = body(response)["error"]
    assert error["code"] == "gateway_misconfigured"
    assert error["type"] == "server_error"


@pytest.mark.anyio
async def test_chat_upstream_status_is_relayed(client, provider):
    for status in (401, 403, 404, 429, 502):
        provider.route("https://api.example.test/v1/chat/completions", {"error": "boom"}, status=status)
        response = await client.post(
            "/v1/chat/completions", payload={"model": "chatty", "messages": [{"role": "user", "content": "hi"}]}
        )
        assert response.status == status, status


@pytest.mark.anyio
async def test_chat_upstream_error_types(client, provider):
    provider.route("https://api.example.test/v1/chat/completions", {"error": "boom"}, status=401)
    error = body(
        await client.post(
            "/v1/chat/completions", payload={"model": "chatty", "messages": [{"role": "user", "content": "hi"}]}
        )
    )["error"]
    assert error["type"] == "authentication_error"
    assert error["code"] == "upstream_error"

    provider.route("https://api.example.test/v1/chat/completions", {"error": "boom"}, status=429)
    error = body(
        await client.post(
            "/v1/chat/completions", payload={"model": "chatty", "messages": [{"role": "user", "content": "hi"}]}
        )
    )["error"]
    assert error["type"] == "rate_limit_error"

    provider.route("https://api.example.test/v1/chat/completions", {"error": "boom"}, status=404)
    error = body(
        await client.post(
            "/v1/chat/completions", payload={"model": "chatty", "messages": [{"role": "user", "content": "hi"}]}
        )
    )["error"]
    assert error["type"] == "invalid_request_error"


@pytest.mark.anyio
async def test_chat_chain_exhaustion_relays_the_upstream_status(client, provider):
    # The old facade relays the provider's own status even when the whole
    # chain is exhausted with an upstream error.
    provider.route("https://api.example.test/v1/chat/completions", {"error": "down"}, status=503)
    response = await client.post(
        "/v1/chat/completions", payload={"model": "chatty", "messages": [{"role": "user", "content": "hi"}]}
    )
    assert response.status == 503
    error = body(response)["error"]
    assert error["code"] == "upstream_error"
    assert error["type"] == "upstream_error"


@pytest.mark.anyio
async def test_chat_chain_exhaustion_without_upstream_status_is_502(client, provider):
    # A non-upstream failure (the provider is unreachable) is a plain 502.
    provider.fail_with["https://api.example.test/v1/chat/completions"] = ConnectionError("provider unreachable")
    response = await client.post(
        "/v1/chat/completions", payload={"model": "chatty", "messages": [{"role": "user", "content": "hi"}]}
    )
    assert response.status == 502
    error = body(response)["error"]
    assert error["code"] == "vendor_error"
    assert error["type"] == "upstream_error"


# ---------------------------------------------------------------------------
# Grounding on the chat surface
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_chat_grounding_honoured_on_the_configured_surface(client):
    response = await client.post(
        "/v1/chat/completions",
        payload={
            "model": "chatsearcher",
            "messages": [{"role": "user", "content": "who won"}],
            "tools": [{"type": "web_search"}],
        },
    )
    assert response.status == 200


@pytest.mark.anyio
async def test_chat_grounding_refused_on_the_wrong_surface(client, db):
    response = await client.post(
        "/v1/chat/completions",
        payload={
            "model": "searcher",
            "messages": [{"role": "user", "content": "who won"}],
            "tools": [{"type": "web_search"}],
        },
    )
    assert response.status == 400
    error = body(response)["error"]
    assert error["code"] == "wrong_grounding_surface"
    assert "/v1/responses" in error["message"]
    assert db.ledger == []


@pytest.mark.anyio
async def test_chat_grounding_refused_on_a_non_grounded_model(client, db):
    response = await client.post(
        "/v1/chat/completions",
        payload={
            "model": "plain",
            "messages": [{"role": "user", "content": "who won"}],
            "tools": [{"type": "web_search"}],
        },
    )
    assert response.status == 400
    assert body(response)["error"]["code"] == "grounding_unavailable"
    assert db.ledger == []


@pytest.mark.anyio
async def test_chat_plain_call_unaffected_by_grounding_capability(client):
    response = await client.post(
        "/v1/chat/completions", payload={"model": "searcher", "messages": [{"role": "user", "content": "hi"}]}
    )
    assert response.status == 200


# ---------------------------------------------------------------------------
# POST /v1/images/generations
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_image_happy_path(client, provider):
    response = await client.post(
        "/v1/images/generations",
        payload={"model": "painter", "prompt": "a cat", "size": "1024x1024", "quality": "high"},
    )
    assert response.status == 200
    data = body(response)
    assert len(data["data"]) == 1
    assert base64.b64decode(data["data"][0]["b64_json"]) == b"\x89PNG-bytes"
    assert data["mime_type"] == "image/png"
    assert data["usage"]["prompt_tokens"] == 5
    _, payload, _, _ = provider.last
    assert payload["size"] == "1024x1024"
    assert payload["quality"] == "high"


@pytest.mark.anyio
async def test_image_url_format_is_refused_before_dispatch(client, db):
    response = await client.post(
        "/v1/images/generations", payload={"model": "painter", "prompt": "cat", "response_format": "url"}
    )
    assert response.status == 501
    assert body(response)["error"]["code"] == "url_response_unsupported"
    assert db.ledger == []


@pytest.mark.anyio
async def test_image_requires_prompt(client):
    response = await client.post("/v1/images/generations", payload={"model": "painter"})
    assert response.status == 400
    assert body(response)["error"]["code"] == "missing_parameter"


@pytest.mark.anyio
async def test_image_no_image_returned_is_502(client, provider):
    provider.route("https://api.example.test/v1/images/generations", {"created": 1, "data": [], "usage": {}})
    response = await client.post("/v1/images/generations", payload={"model": "painter", "prompt": "cat"})
    assert response.status == 502
    assert body(response)["error"]["code"] == "no_image"


@pytest.mark.anyio
async def test_image_edits_path_alias(client):
    response = await client.post("/v1/images/edits", payload={"model": "painter", "prompt": "a cat"})
    assert response.status == 200


# ---------------------------------------------------------------------------
# POST /v1/embeddings
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_embeddings_single_and_batch(client):
    response = await client.post("/v1/embeddings", payload={"model": "embedder", "input": "hello"})
    assert response.status == 200
    data = body(response)
    assert data["object"] == "list"
    assert data["model"] == "embedder"
    assert len(data["data"]) == 1
    assert data["data"][0]["object"] == "embedding"
    assert data["data"][0]["embedding"] == [0.5, 0.25]
    assert data["usage"] == {"prompt_tokens": 4, "completion_tokens": 0, "total_tokens": 4}

    response = await client.post("/v1/embeddings", payload={"model": "embedder", "input": ["a", "b", "c"]})
    data = body(response)
    assert len(data["data"]) == 3
    assert [row["index"] for row in data["data"]] == [0, 1, 2]


@pytest.mark.anyio
async def test_embeddings_base64_is_refused(client):
    response = await client.post(
        "/v1/embeddings", payload={"model": "embedder", "input": "hi", "encoding_format": "base64"}
    )
    assert response.status == 501
    assert body(response)["error"]["code"] == "base64_unsupported"


@pytest.mark.anyio
async def test_embeddings_bad_input(client):
    for payload in (
        {"model": "embedder"},
        {"model": "embedder", "input": ""},
        {"model": "embedder", "input": []},
        {"model": "embedder", "input": [1, 2, 3]},
        {"input": "hi"},
    ):
        response = await client.post("/v1/embeddings", payload=payload)
        assert response.status == 400, payload


@pytest.mark.anyio
async def test_embeddings_unknown_model_is_404(client):
    response = await client.post("/v1/embeddings", payload={"model": "ghost", "input": "hi"})
    assert response.status == 404
    assert body(response)["error"]["code"] == "model_not_found"


@pytest.mark.anyio
async def test_embeddings_dimensions_default_to_768(client, provider):
    await client.post("/v1/embeddings", payload={"model": "embedder", "input": "hi"})
    _, payload, _, _ = provider.last
    assert payload["dimensions"] == 768


# ---------------------------------------------------------------------------
# Error envelope
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_error_envelope_shape(client):
    response = await client.post(
        "/v1/chat/completions", payload={"model": "ghost", "messages": [{"role": "user", "content": "hi"}]}
    )
    assert response.headers.get_first(b"content-type") == b"application/json"
    error = body(response)["error"]
    assert error["message"] and error["type"] and error["code"]
