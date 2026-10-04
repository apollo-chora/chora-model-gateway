from __future__ import annotations

import grpc
import pytest
from grpc.aio import AioRpcError

import model_gateway_service_pb2 as pb
import model_gateway_service_pb2_grpc as pbg
from app.grpc_server import start_grpc
from tests.conftest import make_gateway, make_models


@pytest.fixture
async def grpc_channel(models, db, provider, chat_completion):
    provider.route("https://api.example.test/v1/chat/completions", chat_completion)
    provider.route("https://api.example.test/v1/responses", chat_completion)
    provider.route(
        "https://api.example.test/v1/embeddings",
        {
            "model": "embedder-upstream",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.5, 0.25]}],
            "usage": {"prompt_tokens": 4, "total_tokens": 4},
        },
    )
    gateway = make_gateway(models, db, provider)
    server, port = await start_grpc(gateway, 0)
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    stub = pbg.ModelGatewayServiceStub(channel)
    try:
        yield stub
    finally:
        await channel.close()
        await server.stop(5)


def call(stub, **kwargs):
    return stub.Invoke(pb.InvokeRequest(**kwargs))


# ---------------------------------------------------------------------------
# Invoke mapping
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_invoke_happy_path(grpc_channel):
    response = await call(
        grpc_channel,
        invocation_id="inv-1",
        tenant_id="t",
        gcid="g",
        agent_id="openai_compat",
        logical_model_id="chatty",
        prompt="hi",
    )
    assert response.invocation_id == "inv-1"
    assert response.completion == "hello from the stub"
    assert response.vendor == "openai"
    assert response.model_version == "chatty-upstream"
    assert response.usage.input_tokens == 3
    assert response.usage.output_tokens == 2
    assert response.usage.cost_micros > 0
    assert response.finish_reason == pb.FINISH_REASON_COMPLETE
    assert response.gateway_version == "test"
    assert response.fallback_chain == ["openai:chatty-upstream"]


@pytest.mark.anyio
async def test_invoke_generates_an_invocation_id(grpc_channel):
    response = await call(
        grpc_channel,
        tenant_id="t",
        gcid="g",
        agent_id="a",
        logical_model_id="chatty",
        prompt="hi",
    )
    assert response.invocation_id


@pytest.mark.anyio
async def test_invoke_generation_config_struct(grpc_channel):
    from google.protobuf.struct_pb2 import Struct

    config = Struct()
    config.update({"temperature": 0.2, "max_tokens": 100})
    response = await call(
        grpc_channel,
        tenant_id="t",
        gcid="g",
        agent_id="a",
        logical_model_id="chatty",
        prompt="hi",
        generation_config=config,
    )
    assert response.completion == "hello from the stub"


@pytest.mark.anyio
async def test_invoke_contents_and_tools_json(grpc_channel):
    response = await call(
        grpc_channel,
        tenant_id="t",
        gcid="g",
        agent_id="a",
        logical_model_id="chatty",
        contents_json='[{"role":"user","content":"hi"}]',
        tools_json='[{"type":"function","function":{"name":"lookup"}}]',
    )
    assert response.completion == "hello from the stub"


@pytest.mark.anyio
async def test_invoke_malformed_contents_json_degrades_gracefully(grpc_channel):
    # The Go adapter passes the raw string to the domain and the vendor falls
    # back to the prompt; a malformed payload must not escape as gRPC UNKNOWN.
    response = await call(
        grpc_channel,
        tenant_id="t",
        gcid="g",
        agent_id="a",
        logical_model_id="chatty",
        prompt="hi",
        contents_json="{not valid json",
    )
    assert response.completion == "hello from the stub"


@pytest.mark.anyio
async def test_invoke_fallback_ids(grpc_channel, provider, models):
    # The primary fails; the caller-declared fallback (on its own host) answers.
    models["textonly"] = make_models()["textonly"]
    models["textonly"].base_url = "https://fallback.example.test/v1"
    provider.route("https://api.example.test/v1/chat/completions", {"error": "down"}, status=503)
    provider.route(
        "https://fallback.example.test/v1/chat/completions",
        {
            "id": "chat-fb",
            "model": "textonly-upstream",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "from the fallback"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        },
    )
    response = await call(
        grpc_channel,
        tenant_id="t",
        gcid="g",
        agent_id="a",
        logical_model_id="chatty",
        prompt="hi",
        fallback_logical_model_ids=["textonly"],
    )
    assert response.completion == "from the fallback"
    assert response.fallback_chain == ["openai:chatty-upstream", "openai:textonly-upstream"]


@pytest.mark.anyio
async def test_invoke_idempotency_key_skips_the_debit_but_still_ledgers(grpc_channel, db):
    await call(
        grpc_channel,
        tenant_id="t",
        gcid="g",
        agent_id="a",
        logical_model_id="chatty",
        prompt="hi",
        dispatch_idempotency_key="dispatch-1",
    )
    event = db.ledger[0]
    assert event["debit_deduped"] is False
    assert db.settle_calls[0][1] == event["cost_usd_micros"] > 0

    db.budget_state = {
        "budget_usd_micros": 100,
        "spent_usd_micros": 0,
        "policy": "block",
        "downgrade_to_logical_model_id": None,
    }
    await call(
        grpc_channel,
        tenant_id="t",
        gcid="g",
        agent_id="a",
        logical_model_id="chatty",
        prompt="hi",
        dispatch_idempotency_key="dispatch-1",
    )
    event = db.ledger[-1]
    assert event["debit_deduped"] is True
    # The ledger row still carries the real cost; only the debit is suppressed.
    assert event["cost_usd_micros"] > 0
    assert db.settle_calls[-1][1] == 0


@pytest.mark.anyio
async def test_invoke_traceparent_explicit(grpc_channel, provider):
    await call(
        grpc_channel,
        tenant_id="t",
        gcid="g",
        agent_id="a",
        logical_model_id="chatty",
        prompt="hi",
        traceparent="00-explicit-01",
    )
    _, _, headers, _ = provider.last
    assert headers["traceparent"] == "00-explicit-01"


@pytest.mark.anyio
async def test_invoke_traceparent_metadata_fallback(grpc_channel, provider):
    response = await grpc_channel.Invoke(
        pb.InvokeRequest(
            tenant_id="t",
            gcid="g",
            agent_id="a",
            logical_model_id="chatty",
            prompt="hi",
        ),
        metadata=(("traceparent", "00-from-metadata-02"),),
    )
    assert response.completion == "hello from the stub"
    _, _, headers, _ = provider.last
    assert headers["traceparent"] == "00-from-metadata-02"


@pytest.mark.anyio
async def test_invoke_budget_block_is_a_normal_response(grpc_channel, db):
    db.budget_state = {
        "budget_usd_micros": 100,
        "spent_usd_micros": 200,
        "policy": "block",
        "downgrade_to_logical_model_id": None,
    }
    response = await call(
        grpc_channel,
        tenant_id="t",
        gcid="g",
        agent_id="a",
        logical_model_id="chatty",
        prompt="hi",
    )
    # The old gateway answers budget blocks with a normal response carrying
    # finish_reason BUDGET_BLOCK, not a gRPC status error.
    assert response.finish_reason == pb.FINISH_REASON_BUDGET_BLOCK
    assert "budget" in response.finish_detail
    # The generated invocation id is preserved even when the caller omitted one.
    assert response.invocation_id


# ---------------------------------------------------------------------------
# Status mapping
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_invoke_upstream_status_mapping(grpc_channel, provider):
    cases = {
        401: grpc.StatusCode.UNAUTHENTICATED,
        403: grpc.StatusCode.UNAUTHENTICATED,
        404: grpc.StatusCode.NOT_FOUND,
        429: grpc.StatusCode.RESOURCE_EXHAUSTED,
        503: grpc.StatusCode.UNAVAILABLE,
    }
    for status, expected in cases.items():
        provider.route("https://api.example.test/v1/chat/completions", {"error": "boom"}, status=status)
        with pytest.raises(AioRpcError) as exc_info:
            await call(grpc_channel, tenant_id="t", gcid="g", agent_id="a", logical_model_id="chatty", prompt="hi")
        assert exc_info.value.code() == expected, status


@pytest.mark.anyio
async def test_invoke_config_error_is_failed_precondition(grpc_channel, models, db, provider):
    models["chatty"].api_key_env = "MISSING_KEY"
    with pytest.raises(AioRpcError) as exc_info:
        await call(grpc_channel, tenant_id="t", gcid="g", agent_id="a", logical_model_id="chatty", prompt="hi")
    assert exc_info.value.code() == grpc.StatusCode.FAILED_PRECONDITION


@pytest.mark.anyio
async def test_invoke_capability_error_is_failed_precondition(grpc_channel):
    with pytest.raises(AioRpcError) as exc_info:
        await call(
            grpc_channel,
            tenant_id="t",
            gcid="g",
            agent_id="a",
            logical_model_id="chatty",
            prompt="hi",
            response_modality="IMAGE",
        )
    assert exc_info.value.code() == grpc.StatusCode.FAILED_PRECONDITION


@pytest.mark.anyio
async def test_invoke_validation_error_is_unavailable(grpc_channel):
    with pytest.raises(AioRpcError) as exc_info:
        await call(grpc_channel, tenant_id="", gcid="g", agent_id="a", logical_model_id="chatty", prompt="hi")
    assert exc_info.value.code() == grpc.StatusCode.UNAVAILABLE


@pytest.mark.anyio
async def test_invoke_unknown_model_is_unavailable(grpc_channel):
    with pytest.raises(AioRpcError) as exc_info:
        await call(grpc_channel, tenant_id="t", gcid="g", agent_id="a", logical_model_id="ghost", prompt="hi")
    assert exc_info.value.code() == grpc.StatusCode.UNAVAILABLE


# ---------------------------------------------------------------------------
# Embed
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_embed_happy_path(grpc_channel, db):
    response = await grpc_channel.Embed(
        pb.EmbedRequest(
            invocation_id="emb-1",
            tenant_id="t",
            gcid="g",
            agent_id="a",
            logical_model_id="embedder",
            text="hello",
        )
    )
    assert response.invocation_id == "emb-1"
    assert list(response.values) == [0.5, 0.25]
    assert response.vendor == "openai"
    assert response.model_version == "embedder-upstream"
    assert response.usage.input_tokens == 4
    # Embeddings are unpriced: the ledger row exists with a zero cost.
    assert db.ledger[0]["cost_usd_micros"] == 0
    assert db.ledger[0]["input_tokens"] == 4


@pytest.mark.anyio
async def test_embed_missing_fields_is_invalid_argument(grpc_channel):
    with pytest.raises(AioRpcError) as exc_info:
        await grpc_channel.Embed(pb.EmbedRequest(tenant_id="", gcid="g", agent_id="a", text="hi"))
    assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT


@pytest.mark.anyio
async def test_embed_non_embedding_model_is_failed_precondition(grpc_channel):
    with pytest.raises(AioRpcError) as exc_info:
        await grpc_channel.Embed(
            pb.EmbedRequest(
                tenant_id="t",
                gcid="g",
                agent_id="a",
                logical_model_id="chatty",
                text="hi",
            )
        )
    assert exc_info.value.code() == grpc.StatusCode.FAILED_PRECONDITION


@pytest.mark.anyio
async def test_embed_default_model_when_omitted(grpc_channel, db):
    response = await grpc_channel.Embed(
        pb.EmbedRequest(
            tenant_id="t",
            gcid="g",
            agent_id="a",
            text="hi",
        )
    )
    assert response.model_version == "embedder-upstream"


# ---------------------------------------------------------------------------
# GroundedSearch
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_grounded_search_is_unimplemented(grpc_channel):
    with pytest.raises(AioRpcError) as exc_info:
        await grpc_channel.GroundedSearch(
            pb.GroundedSearchRequest(
                tenant_id="t",
                gcid="g",
                agent_id="a",
                directive="who won",
            )
        )
    assert exc_info.value.code() == grpc.StatusCode.UNIMPLEMENTED
