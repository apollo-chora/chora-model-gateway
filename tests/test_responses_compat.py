from __future__ import annotations

import pytest

from tests.conftest import Client, body, make_app, make_models


@pytest.fixture
async def client(models, db, provider, grounded_response):
    provider.route("https://api.example.test/v1/responses", grounded_response)
    provider.route("https://api.example.test/v1/chat/completions", grounded_response)
    app = make_app(models, db, provider)
    await app.start()
    try:
        yield Client(app)
    finally:
        await app.stop()


@pytest.mark.anyio
async def test_responses_grounded_request(client, provider):
    response = await client.post(
        "/v1/responses",
        payload={
            "model": "searcher",
            "input": "Who won the most recent Formula 1 race?",
            "tools": [{"type": "web_search"}],
        },
    )
    assert response.status == 200
    data = body(response)
    assert data["object"] == "response"
    assert data["model"] == "searcher-upstream"
    assert data["status"] == "completed"
    assert len(data["output"]) == 2
    assert data["output"][0]["type"] == "web_search_call"
    assert data["output"][1]["type"] == "message"
    assert data["usage"]["total_tokens"] == 37
    # The search call carries the queries that actually ran.
    assert data["output"][0]["action"]["query"] == "most recent Formula 1 race winner"


@pytest.mark.anyio
async def test_responses_citations_appear_natively_and_normalised(client):
    response = await client.post(
        "/v1/responses",
        payload={
            "model": "searcher",
            "input": "who won",
            "tools": [{"type": "web_search"}],
        },
    )
    data = body(response)
    content = data["output"][1]["content"][0]
    annotations = content["annotations"]
    assert len(annotations) == 2
    assert annotations[0]["type"] == "url_citation"
    assert annotations[0]["url"] == "https://f1.example/race-report"
    assert annotations[0]["start_index"] == 0
    # Absent offsets are omitted, never emitted as a misleading 0..0.
    assert "start_index" not in annotations[1]

    meta = data["chora_gateway"]
    assert meta["grounded"] is True
    assert len(meta["citations"]) == 2
    assert meta["grounding_surface"] == "responses"
    assert meta["search_queries"] == ["most recent Formula 1 race winner"]


@pytest.mark.anyio
async def test_responses_no_tools_is_a_plain_call(client, provider):
    response = await client.post("/v1/responses", payload={"model": "searcher", "input": "just answer"})
    assert response.status == 200
    data = body(response)
    assert data["output"][0]["type"] == "message"
    assert data["chora_gateway"]["grounded"] is False
    # omitempty: a non-grounded call omits the key entirely.
    assert "grounding_surface" not in data["chora_gateway"]


@pytest.mark.anyio
async def test_responses_non_search_tools_do_not_ground(client):
    response = await client.post(
        "/v1/responses",
        payload={
            "model": "searcher",
            "input": "hi",
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
        },
    )
    assert response.status == 200
    assert body(response)["chora_gateway"]["grounded"] is False


@pytest.mark.anyio
async def test_responses_grounding_refused_on_a_non_grounded_model(client, db):
    response = await client.post(
        "/v1/responses",
        payload={
            "model": "plain",
            "input": "who won",
            "tools": [{"type": "web_search"}],
        },
    )
    assert response.status == 400
    error = body(response)["error"]
    assert error["code"] == "grounding_unavailable"
    assert "web_search" in error["message"]
    assert db.ledger == []


@pytest.mark.anyio
async def test_responses_web_fetch_also_counts_as_grounding(client):
    response = await client.post(
        "/v1/responses",
        payload={
            "model": "searcher",
            "input": "what is this page",
            "tools": [{"type": "web_fetch"}],
        },
    )
    assert response.status == 200
    assert body(response)["chora_gateway"]["grounded"] is True


@pytest.mark.anyio
async def test_responses_stream_is_refused(client, db):
    response = await client.post(
        "/v1/responses",
        payload={
            "model": "searcher",
            "input": "hi",
            "stream": True,
        },
    )
    assert response.status == 501
    assert body(response)["error"]["code"] == "streaming_unsupported"
    assert db.ledger == []


@pytest.mark.anyio
async def test_responses_wrong_method_is_405(client):
    response = await client.get("/v1/responses")
    assert response.status == 405


@pytest.mark.anyio
async def test_responses_missing_fields(client):
    for payload in (
        {"input": "hi", "tools": [{"type": "web_search"}]},
        {"model": "searcher", "tools": [{"type": "web_search"}]},
        {"model": "searcher", "input": "", "tools": [{"type": "web_search"}]},
        {"model": "searcher"},
    ):
        response = await client.post("/v1/responses", payload=payload)
        assert response.status == 400, payload


@pytest.mark.anyio
async def test_responses_unknown_model_is_404(client):
    response = await client.post(
        "/v1/responses",
        payload={
            "model": "ghost",
            "input": "hi",
            "tools": [{"type": "web_search"}],
        },
    )
    assert response.status == 404
    assert body(response)["error"]["code"] == "model_not_found"


@pytest.mark.anyio
async def test_responses_budget_block_is_402(client, db, provider):
    db.budget_state = {
        "budget_usd_micros": 100,
        "spent_usd_micros": 200,
        "policy": "block",
        "downgrade_to_logical_model_id": None,
    }
    response = await client.post(
        "/v1/responses",
        payload={
            "model": "searcher",
            "input": "hi",
            "tools": [{"type": "web_search"}],
        },
    )
    assert response.status == 402
    assert body(response)["error"]["code"] == "budget_exhausted"


@pytest.mark.anyio
async def test_responses_instructions_become_the_system_prompt(client, provider):
    await client.post(
        "/v1/responses",
        payload={
            "model": "searcher",
            "input": "hi",
            "instructions": "be terse",
            "tools": [{"type": "web_search"}],
        },
    )
    _, payload, _, _ = provider.last
    assert payload["instructions"] == "be terse"


@pytest.mark.anyio
async def test_responses_generation_config_forwarded(client, provider):
    await client.post(
        "/v1/responses",
        payload={
            "model": "searcher",
            "input": "hi",
            "max_output_tokens": 1500,
            "reasoning_effort": "high",
            "tools": [{"type": "web_search"}],
        },
    )
    _, payload, _, _ = provider.last
    assert payload["max_output_tokens"] == 1500
    assert payload["reasoning_effort"] == "high"


@pytest.mark.anyio
async def test_responses_conversation_reaches_the_provider(client, provider):
    await client.post(
        "/v1/responses",
        payload={
            "model": "searcher",
            "input": [
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
                {"role": "user", "content": "three"},
            ],
            "tools": [{"type": "web_search"}],
        },
    )
    _, payload, _, _ = provider.last
    assert payload["input"][2]["content"] == "three"


@pytest.mark.anyio
async def test_responses_attribution(client, db, provider):
    app = make_app(make_models(), db, provider)
    await app.start()
    try:
        response = await Client(app).post(
            "/v1/responses",
            payload={"model": "searcher", "input": "hi", "user": "alice", "tools": [{"type": "web_search"}]},
            headers={"X-Chora-Tenant-Id": "tenant-7"},
        )
    finally:
        await app.stop()
    assert response.status == 200
    event = db.ledger[0]
    assert event["tenant_id"] == "tenant-7"
    assert event["agent_role"] == "alice"


@pytest.mark.anyio
async def test_responses_grounding_tool_goes_on_the_wire(client, provider):
    await client.post(
        "/v1/responses",
        payload={
            "model": "searcher",
            "input": "hi",
            "tools": [{"type": "web_search"}],
        },
    )
    _, payload, _, _ = provider.last
    assert payload["tools"] == [{"type": "web_search"}]
    assert payload["stream"] is False


@pytest.mark.anyio
async def test_responses_anthropic_surface_uses_the_server_tool(client, provider):
    provider.route(
        "https://api.example.test/v1/messages",
        {
            "id": "msg-1",
            "model": "claude-upstream",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "the answer"}],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        },
    )
    response = await client.post(
        "/v1/responses",
        payload={
            "model": "claude-searcher",
            "input": "hi",
            "tools": [{"type": "web_search"}],
        },
    )
    assert response.status == 200
    _, payload, headers, _ = provider.last
    assert payload["tools"][0]["type"] == "web_search_20250305"
    assert payload["tools"][0]["name"] == "web_search"
    # A credential-less target ships neither x-api-key nor Authorization.
    assert "x-api-key" not in headers
    assert "authorization" not in {k.lower() for k in headers}


@pytest.mark.anyio
async def test_responses_incomplete_status_is_not_reported_as_success(client, provider):
    provider.route(
        "https://api.example.test/v1/responses",
        {
            "id": "resp-2",
            "model": "searcher-upstream",
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [
                {
                    "type": "message",
                    "id": "m",
                    "status": "in_progress",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "half an answer"}],
                }
            ],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "input_tokens_details": {"cached_tokens": 0},
            },
        },
    )
    response = await client.post(
        "/v1/responses",
        payload={
            "model": "searcher",
            "input": "hi",
            "tools": [{"type": "web_search"}],
        },
    )
    assert response.status == 200
    data = body(response)
    assert data["chora_gateway"]["finish_reason"] == "max_tokens"
    assert "truncated" in data["chora_gateway"]["finish_detail"]


@pytest.mark.anyio
async def test_responses_multiple_messages_stay_separate(client, provider):
    provider.route(
        "https://api.example.test/v1/responses",
        {
            "id": "resp-3",
            "model": "searcher-upstream",
            "status": "completed",
            "output": [
                {
                    "type": "web_search_call",
                    "id": "ws",
                    "status": "completed",
                    "action": {"type": "search", "query": "q1"},
                },
                {
                    "type": "message",
                    "id": "m0",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "I'll look that up."}],
                },
                {
                    "type": "message",
                    "id": "m1",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "6.21 million.",
                            "annotations": [
                                {
                                    "type": "url_citation",
                                    "url": "https://pop.gov.sg/pib",
                                    "start_index": 0,
                                    "end_index": 4,
                                }
                            ],
                        }
                    ],
                },
            ],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "input_tokens_details": {"cached_tokens": 0},
            },
        },
    )
    response = await client.post(
        "/v1/responses",
        payload={
            "model": "searcher",
            "input": "population of Singapore?",
            "tools": [{"type": "web_search"}],
        },
    )
    data = body(response)
    output = data["output"]
    assert len(output) == 3
    assert output[0]["type"] == "web_search_call"
    # The answer is the LAST message; narration carries no citations.
    assert output[1]["content"][0]["text"] == "I'll look that up."
    assert output[1]["content"][0]["annotations"] == []
    assert output[2]["content"][0]["annotations"][0]["url"] == "https://pop.gov.sg/pib"
    # The completion is the answer, not narration soup.
    assert data["chora_gateway"]["messages"][-1]["text"] == "6.21 million."
