from __future__ import annotations

import pytest

from app.service import BudgetBlock, GatewayError
from tests.conftest import FakeDB, make_gateway, make_spec


def budget_state(policy="block", spent=200, ceiling=100, downgrade_to=None):
    return {
        "budget_usd_micros": ceiling,
        "spent_usd_micros": spent,
        "policy": policy,
        "downgrade_to_logical_model_id": downgrade_to,
    }


async def invoke(gateway, db, **overrides):
    request = {
        "tenant_id": "00000000-0000-7000-8000-000000000001",
        "gcid": "00000000-0000-7000-8000-000000000002",
        "agent_id": "openai_compat",
        "model": "chatty",
        "prompt": "hi",
        "modality": "TEXT",
    }
    request.update(overrides)
    return await gateway.invoke(request)


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_no_budget_row_allows(models, db, provider):
    db.budget_state = None
    response = await invoke(make_gateway(models, db, provider), db)
    assert response.text == "hello from the stub"


@pytest.mark.anyio
async def test_budget_under_the_ceiling_allows(models, db, provider):
    db.budget_state = budget_state(spent=50, ceiling=100)
    response = await invoke(make_gateway(models, db, provider), db)
    assert response.text == "hello from the stub"


@pytest.mark.anyio
async def test_budget_block_refuses_before_dispatch(models, db, provider):
    db.budget_state = budget_state(policy="block", spent=200, ceiling=100)
    with pytest.raises(BudgetBlock):
        await invoke(make_gateway(models, db, provider), db)
    assert provider.requests == []
    assert db.ledger == []


@pytest.mark.anyio
async def test_budget_block_at_exact_ceiling(models, db, provider):
    db.budget_state = budget_state(policy="block", spent=100, ceiling=100)
    with pytest.raises(BudgetBlock):
        await invoke(make_gateway(models, db, provider), db)


@pytest.mark.anyio
async def test_budget_zero_ceiling_is_exhausted(models, db, provider):
    db.budget_state = budget_state(policy="block", spent=0, ceiling=0)
    with pytest.raises(BudgetBlock):
        await invoke(make_gateway(models, db, provider), db)


@pytest.mark.anyio
async def test_budget_unknown_policy_fails_safe(models, db, provider):
    # An unrecognised policy must refuse rather than allow.
    db.budget_state = budget_state(policy="something-new", spent=200, ceiling=100)
    with pytest.raises(BudgetBlock):
        await invoke(make_gateway(models, db, provider), db)


@pytest.mark.anyio
async def test_budget_empty_policy_fails_safe(models, db, provider):
    db.budget_state = budget_state(policy="", spent=200, ceiling=100)
    with pytest.raises(BudgetBlock):
        await invoke(make_gateway(models, db, provider), db)


@pytest.mark.anyio
async def test_budget_alert_proceeds(models, db, provider):
    db.budget_state = budget_state(policy="alert", spent=200, ceiling=100)
    response = await invoke(make_gateway(models, db, provider), db)
    assert response.text == "hello from the stub"


@pytest.mark.anyio
async def test_budget_downgrade_reresolves_the_target(models, db, provider):
    db.budget_state = budget_state(policy="downgrade", spent=200, ceiling=100, downgrade_to="textonly")
    gateway = make_gateway(models, db, provider)
    await invoke(gateway, db)
    # The downgrade target was dispatched: the wire carried textonly's model.
    _, payload, _, _ = provider.last
    assert payload["model"] == "textonly-upstream"


@pytest.mark.anyio
async def test_budget_downgrade_to_unknown_model_is_a_config_error(models, db, provider):
    db.budget_state = budget_state(policy="downgrade", spent=200, ceiling=100, downgrade_to="model-typo")
    with pytest.raises(GatewayError) as exc_info:
        await invoke(make_gateway(models, db, provider), db)
    assert exc_info.value.status == 500
    assert exc_info.value.code == "gateway_misconfigured"


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_cost_uses_term_wise_floor_division(models, db, provider):
    spec = make_spec(
        pricing={
            "input_per_mtok_usd_micros": 1_000_000,
            "output_per_mtok_usd_micros": 2_000_000,
            "cached_per_mtok_usd_micros": 100_000,
        }
    )
    # 1000 in (500 cached) + 2000 out at $1/$2/$0.10 per Mtok.
    assert spec.cost_micros(1000 - 500, 2000, 500) == 500 + 4000 + 50


@pytest.mark.anyio
async def test_cost_cached_tokens_are_not_double_counted(models):
    spec = make_spec(
        pricing={
            "input_per_mtok_usd_micros": 1_000_000,
            "cached_per_mtok_usd_micros": 0,
        }
    )
    assert spec.cost_micros(0, 0, 1000) == 0
    # A nonsensical cached > input must not produce a negative debit.
    assert spec.cost_micros(0, 0, 500) == 0


@pytest.mark.anyio
async def test_cost_unpriced_model_is_zero(models):
    spec = make_spec(pricing={})
    assert spec.cost_micros(1000, 2000, 500) == 0


@pytest.mark.anyio
async def test_cost_ledger_carries_the_real_cost_when_deduped(models, db, provider):
    gateway = make_gateway(models, db, provider)
    await invoke(gateway, db, dispatch_idempotency_key="k1")
    await invoke(gateway, db, dispatch_idempotency_key="k1")
    first, second = db.ledger
    # Both events carry the real cost; only the first moved the budget.
    assert first["cost_micros"] == second["cost_micros"] > 0
    assert db.settle_calls[0][1] == first["cost_micros"]
    assert db.settle_calls[1][1] == 0


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_duplicate_dispatch_key_does_not_debit_twice(models, db, provider):
    gateway = make_gateway(models, db, provider)
    await invoke(gateway, db, dispatch_idempotency_key="dup")
    await invoke(gateway, db, dispatch_idempotency_key="dup")
    assert len(db.ledger) == 2
    assert db.settle_calls[0][1] > 0
    assert db.settle_calls[1][1] == 0


@pytest.mark.anyio
async def test_claim_key_includes_the_action_code(models, db, provider):
    gateway = make_gateway(models, db, provider)
    await invoke(gateway, db, dispatch_idempotency_key="k", action_code="alice")
    await invoke(gateway, db, dispatch_idempotency_key="k", action_code="bob")
    # Different action codes are different claims: both debit.
    assert db.settle_calls[0][1] > 0
    assert db.settle_calls[1][1] > 0


@pytest.mark.anyio
async def test_no_dispatch_key_means_no_claim(models, db, provider):
    gateway = make_gateway(models, db, provider)
    await invoke(gateway, db)
    await invoke(gateway, db)
    # Without a key a plain retried call debits again.
    assert db.settle_calls[0][1] > 0
    assert db.settle_calls[1][1] > 0


# ---------------------------------------------------------------------------
# Fallbacks
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_caller_fallbacks_come_before_registry_fallbacks(models, db, provider):
    models["chatty"] = make_spec(fallback_ids=["textonly"])
    models["textonly"] = make_spec(
        id="textonly", model="textonly-upstream", base_url="https://fallback.example.test/v1"
    )
    gateway = make_gateway(models, db, provider)
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
    response = await invoke(gateway, db, fallback_ids=["textonly"])
    assert response.fallback_chain == ["openai:chatty-upstream", "openai:textonly-upstream"]


@pytest.mark.anyio
async def test_unknown_caller_fallback_fails_the_resolve(models, db, provider):
    gateway = make_gateway(models, db, provider)
    with pytest.raises(GatewayError) as exc_info:
        await invoke(gateway, db, fallback_ids=["ghost"])
    assert exc_info.value.status == 502
    assert exc_info.value.code == "vendor_error"


@pytest.mark.anyio
async def test_unknown_registry_fallback_fails_at_boot():
    import os
    import tempfile

    import yaml

    from app.config import ConfigError, load_models

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(
            {
                "models": [
                    {"id": "a", "provider": "openai", "base_url": "https://x.test/v1", "fallback_ids": ["missing"]},
                ]
            },
            handle,
        )
        path = handle.name
    try:
        with pytest.raises(ConfigError):
            load_models(path)
    finally:
        os.unlink(path)


def _registry_with_path(path_value):
    import tempfile

    import yaml

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(
            {
                "models": [
                    {
                        "id": "a",
                        "provider": "openai",
                        "base_url": "https://x.test/v1",
                        "chat_completions_path": path_value,
                    },
                ]
            },
            handle,
        )
        return handle.name


@pytest.mark.parametrize("bad_path", ["//host/path", "ftp://x.test/v1", "grpc://x.test"])
def test_registry_rejects_invalid_endpoint_overrides_at_boot(bad_path):
    from app.config import ConfigError, load_models

    path = _registry_with_path(bad_path)
    try:
        with pytest.raises(ConfigError):
            load_models(path)
    finally:
        import os

        os.unlink(path)


@pytest.mark.parametrize("good_path", ["/custom/chat", "https://x.test/v1/custom/chat"])
def test_registry_accepts_valid_endpoint_overrides(good_path):
    from app.config import load_models

    path = _registry_with_path(good_path)
    try:
        models = load_models(path)
        assert models["a"].chat_completions_path == good_path
    finally:
        import os

        os.unlink(path)


@pytest.mark.anyio
async def test_fallback_can_cross_provider_boundaries(models, db, provider):
    models["chatty"] = make_spec(fallback_ids=["claude-searcher"])
    gateway = make_gateway(models, db, provider)
    provider.route("https://api.example.test/v1/chat/completions", {"error": "down"}, status=503)
    provider.route(
        "https://api.example.test/v1/messages",
        {
            "id": "m",
            "model": "claude-upstream",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "from anthropic"}],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        },
    )
    response = await invoke(gateway, db)
    assert response.text == "from anthropic"
    assert response.vendor == "anthropic"
    assert response.fallback_chain == ["openai:chatty-upstream", "anthropic:claude-upstream"]


# ---------------------------------------------------------------------------
# Capabilities and validation
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_image_request_to_non_image_model_is_refused(models, db, provider):
    gateway = make_gateway(models, db, provider)
    with pytest.raises(GatewayError) as exc_info:
        await invoke(gateway, db, modality="IMAGE")
    assert exc_info.value.status == 500
    assert exc_info.value.code == "gateway_misconfigured"
    assert "image" in exc_info.value.message


@pytest.mark.anyio
async def test_grounded_request_to_non_grounded_model_is_refused(models, db, provider):
    gateway = make_gateway(models, db, provider)
    with pytest.raises(GatewayError) as exc_info:
        await invoke(gateway, db, model="plain", modality="GROUNDED")
    assert exc_info.value.status == 500
    assert "web_search" in exc_info.value.message


@pytest.mark.anyio
async def test_grounded_request_without_a_grounding_block_is_refused(models, db, provider):
    models["nocaps"] = make_spec(id="nocaps", model="nocaps-upstream", capabilities=["chat", "web_search"])
    gateway = make_gateway(models, db, provider)
    with pytest.raises(GatewayError) as exc_info:
        await invoke(gateway, db, model="nocaps", modality="GROUNDED")
    assert exc_info.value.status == 500
    assert "grounding" in exc_info.value.message


@pytest.mark.anyio
async def test_empty_request_is_refused(models, db, provider):
    gateway = make_gateway(models, db, provider)
    with pytest.raises(GatewayError) as exc_info:
        await invoke(gateway, db, prompt="")
    assert exc_info.value.status == 502


@pytest.mark.anyio
async def test_missing_tenant_is_refused(models, db, provider):
    gateway = make_gateway(models, db, provider)
    with pytest.raises(GatewayError):
        await invoke(gateway, db, tenant_id="")


@pytest.mark.anyio
async def test_bad_modality_is_refused(models, db, provider):
    gateway = make_gateway(models, db, provider)
    with pytest.raises(GatewayError):
        await invoke(gateway, db, modality="AUDIO")


# ---------------------------------------------------------------------------
# Output ceiling
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_output_ceiling_clamps_max_tokens(models, db, provider):
    gateway = make_gateway(models, db, provider)
    await invoke(gateway, db, params={"max_tokens": 9999})
    _, payload, _, _ = provider.last
    assert payload["max_tokens"] == 512  # the registry ceiling


@pytest.mark.anyio
async def test_output_ceiling_honours_a_smaller_request(models, db, provider):
    gateway = make_gateway(models, db, provider)
    await invoke(gateway, db, params={"max_tokens": 100})
    _, payload, _, _ = provider.last
    assert payload["max_tokens"] == 100


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_configured_but_empty_credential_is_a_misconfiguration(models, db, provider):
    models["chatty"].api_key_env = "MISSING_KEY"
    gateway = make_gateway(models, db, provider)
    with pytest.raises(GatewayError) as exc_info:
        await invoke(gateway, db)
    assert exc_info.value.status == 500
    assert exc_info.value.code == "gateway_misconfigured"
    assert "MISSING_KEY" in exc_info.value.message


@pytest.mark.anyio
async def test_credential_travels_on_the_request(models, db, provider, monkeypatch):
    monkeypatch.setenv("CHATTY_KEY", "sk-test")
    models["chatty"].api_key_env = "CHATTY_KEY"
    gateway = make_gateway(models, db, provider)
    await invoke(gateway, db)
    _, _, headers, _ = provider.last
    assert headers["Authorization"] == "Bearer sk-test"


@pytest.mark.anyio
async def test_no_credential_ref_sends_no_auth_header(models, db, provider):
    gateway = make_gateway(models, db, provider)
    await invoke(gateway, db)
    _, _, headers, _ = provider.last
    assert "Authorization" not in headers


# ---------------------------------------------------------------------------
# Settle
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_settle_failure_fails_the_call(models, db, provider):
    class FailingDB(FakeDB):
        async def debit_and_enqueue(self, event, debit):
            raise RuntimeError("deadlock detected")

    gateway = make_gateway(models, FailingDB(), provider)
    with pytest.raises(GatewayError) as exc_info:
        await invoke(gateway, db)
    assert exc_info.value.status == 502
    assert "settle" in exc_info.value.message


@pytest.mark.anyio
async def test_ledger_row_carries_the_trace_context(models, db, provider):
    gateway = make_gateway(models, db, provider)
    await invoke(gateway, db, traceparent="00-abc", tracestate="foo=bar", surface="openai_compat")
    event = db.ledger[0]
    assert event["traceparent"] == "00-abc"
    assert event["tracestate"] == "foo=bar"
    assert event["gateway_version"] == "test"


# ---------------------------------------------------------------------------
# Registry endpoint paths
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_custom_chat_completions_path_is_used(models, db, provider):
    # The Go registry honors a per-entry chat_completions_path override; the
    # runtime must dispatch to it rather than the default /chat/completions.
    models["chatty"] = make_spec(chat_completions_path="/custom/chat")
    gateway = make_gateway(models, db, provider)
    await invoke(gateway, db)
    path, _, _, url = provider.last
    assert path == "/v1/custom/chat"
    assert url == "https://api.example.test/v1/custom/chat"


@pytest.mark.anyio
async def test_absolute_chat_completions_path_is_used_verbatim(models, db, provider):
    models["chatty"] = make_spec(chat_completions_path="https://llm.internal.example.com/generate")
    gateway = make_gateway(models, db, provider)
    await invoke(gateway, db)
    path, _, _, url = provider.last
    assert url == "https://llm.internal.example.com/generate"


@pytest.mark.anyio
async def test_custom_messages_path_is_used_for_anthropic(models, db, provider):
    models["claude"] = make_spec(
        id="claude",
        vendor="anthropic",
        format="messages",
        model="claude-upstream",
        messages_path="/custom/messages",
    )
    gateway = make_gateway(models, db, provider)
    provider.route(
        "https://api.example.test/v1/custom/messages",
        {
            "id": "m",
            "model": "claude-upstream",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "hi"}],
            "usage": {
                "input_tokens": 1,
                "output_tokens": 1,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        },
    )
    response = await invoke(gateway, db, model="claude")
    assert response.text == "hi"
    path, _, _, url = provider.last
    assert path == "/v1/custom/messages"


# ---------------------------------------------------------------------------
# Error paths (found in the ChatGPT review)
# ---------------------------------------------------------------------------


class _DBErrorDB(FakeDB):
    """A FakeDB whose claim() or budget() raises, to exercise the wrapping."""

    def __init__(self, fail_claim=False, fail_budget=False):
        super().__init__()
        self._fail_claim = fail_claim
        self._fail_budget = fail_budget

    async def claim(self, gcid, key, action, invocation_id):
        if self._fail_claim:
            raise RuntimeError("db down")
        return await super().claim(gcid, key, action, invocation_id)

    async def budget(self, tenant_id):
        if self._fail_budget:
            raise RuntimeError("db down")
        return await super().budget(tenant_id)


@pytest.mark.anyio
async def test_claim_db_failure_is_a_vendor_error(models, provider):
    gateway = make_gateway(models, _DBErrorDB(fail_claim=True), provider)
    with pytest.raises(GatewayError) as exc_info:
        await invoke(gateway, _DBErrorDB(fail_claim=True), dispatch_idempotency_key="k")
    assert exc_info.value.status == 502
    assert exc_info.value.code == "vendor_error"


@pytest.mark.anyio
async def test_budget_db_failure_is_a_vendor_error(models, provider):
    gateway = make_gateway(models, _DBErrorDB(fail_budget=True), provider)
    with pytest.raises(GatewayError) as exc_info:
        await invoke(gateway, _DBErrorDB(fail_budget=True))
    assert exc_info.value.status == 502
    assert exc_info.value.code == "vendor_error"


@pytest.mark.anyio
async def test_stale_upstream_status_does_not_misclassify_exhausted_chain(models, db, provider):
    # Primary returns 429, fallback fails with a connection error: the final
    # error is the LAST target's (no upstream status), so it is a 502, not 429.
    models["chatty"] = make_spec(base_url="https://api.example.test/v1")
    models["textonly"] = make_spec(
        id="textonly", model="textonly-upstream", base_url="https://fallback.example.test/v1"
    )
    provider.route("https://api.example.test/v1/chat/completions", {"error": "rate limited"}, status=429)
    provider.fail_with["https://fallback.example.test/v1/chat/completions"] = ConnectionError("unreachable")
    gateway = make_gateway(models, db, provider)
    with pytest.raises(GatewayError) as exc_info:
        await invoke(gateway, db, fallback_ids=["textonly"])
    assert exc_info.value.status == 502
    assert exc_info.value.upstream_status is None


@pytest.mark.anyio
async def test_last_target_upstream_status_wins(models, db, provider):
    # Primary fails with a connection error, fallback returns 429: the final
    # error relays the LAST target's upstream status.
    models["chatty"] = make_spec(base_url="https://api.example.test/v1")
    models["textonly"] = make_spec(
        id="textonly", model="textonly-upstream", base_url="https://fallback.example.test/v1"
    )
    provider.fail_with["https://api.example.test/v1/chat/completions"] = ConnectionError("unreachable")
    provider.route("https://fallback.example.test/v1/chat/completions", {"error": "rate limited"}, status=429)
    gateway = make_gateway(models, db, provider)
    with pytest.raises(GatewayError) as exc_info:
        await invoke(gateway, db, fallback_ids=["textonly"])
    # The chain-exhausted error is a vendor error (502); the facade relays the
    # last target's upstream status (429) to the caller.
    assert exc_info.value.status == 502
    assert exc_info.value.upstream_status == 429


@pytest.mark.anyio
async def test_output_ceiling_reapplies_per_fallback_target(models, db, provider):
    # The primary has a high ceiling and fails; the fallback has a low ceiling
    # and must re-clamp the caller's max_tokens to its own ceiling.
    models["chatty"] = make_spec(max_output_tokens=8192, base_url="https://api.example.test/v1")
    models["textonly"] = make_spec(
        id="textonly",
        model="textonly-upstream",
        max_output_tokens=1024,
        base_url="https://fallback.example.test/v1",
    )
    provider.route("https://api.example.test/v1/chat/completions", {"error": "down"}, status=503)
    provider.route(
        "https://fallback.example.test/v1/chat/completions",
        {
            "id": "c",
            "model": "textonly-upstream",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        },
    )
    gateway = make_gateway(models, db, provider)
    response = await invoke(gateway, db, params={"max_tokens": 6000}, fallback_ids=["textonly"])
    assert response.text == "ok"
    _, payload, _, _ = provider.last
    assert payload["max_tokens"] == 1024
