"""The protobuf contract must stay wire-compatible with the original gateway.

Every field number, enum value, service name and RPC name below is pinned to
the old generated contract (chora.services.model_gateway.v1). A renumber or a
reused number breaks every existing gRPC client.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PROTO = Path("proto/model_gateway_service.proto").read_text()


def _message_fields(name: str) -> dict[str, int]:
    match = re.search(rf"message {name}\{{(.*?)\}}", PROTO)
    assert match, f"message {name} not found"
    return {field: int(number) for field, number in re.findall(r"\w+ (\w+)=(\d+);", match.group(1))}


EXPECTED = {
    "InvokeRequest": {
        "invocation_id": 1,
        "tenant_id": 2,
        "gcid": 3,
        "agent_id": 4,
        "crew_kind": 5,
        "logical_model_id": 6,
        "prompt": 7,
        "system_prompt": 8,
        "generation_config": 9,
        "traceparent": 10,
        "tracestate": 11,
        "fallback_logical_model_ids": 12,
        "response_modality": 13,
        "action_code": 14,
        "contents_json": 15,
        "tools_json": 16,
        "surface": 17,
        "dispatch_idempotency_key": 18,
    },
    "TokenUsage": {"input_tokens": 1, "output_tokens": 2, "cached_tokens": 3, "cost_micros": 4},
    "InvokeResponse": {
        "invocation_id": 1,
        "completion": 2,
        "usage": 3,
        "vendor": 4,
        "model_version": 5,
        "model_armor_verdict_pre": 6,
        "model_armor_verdict_post": 7,
        "fallback_chain": 8,
        "latency_ms": 9,
        "finish_reason": 10,
        "finish_detail": 11,
        "completed_at": 12,
        "gateway_version": 13,
        "image_bytes": 14,
        "image_mime_type": 15,
        "tool_calls_json": 16,
    },
    "GroundedSearchRequest": {
        "invocation_id": 1,
        "tenant_id": 2,
        "gcid": 3,
        "agent_id": 4,
        "crew_kind": 5,
        "directive": 6,
        "max_results": 7,
        "logical_model_id": 8,
        "action_code": 9,
        "traceparent": 10,
        "tracestate": 11,
    },
    "GroundedSearchResponse": {
        "invocation_id": 1,
        "citations": 2,
        "grounded_answer": 3,
        "search_entry_point_html": 4,
        "web_search_queries": 5,
        "usage": 6,
        "vendor": 7,
        "model_version": 8,
        "model_armor_verdict_pre": 9,
        "model_armor_verdict_post": 10,
        "finish_reason": 11,
        "finish_detail": 12,
        "completed_at": 13,
        "gateway_version": 14,
    },
    "EmbedRequest": {
        "invocation_id": 1,
        "tenant_id": 2,
        "gcid": 3,
        "agent_id": 4,
        "crew_kind": 5,
        "logical_model_id": 6,
        "text": 7,
        "task_type": 8,
        "output_dimensions": 9,
        "traceparent": 10,
        "tracestate": 11,
    },
    "EmbedResponse": {
        "invocation_id": 1,
        "values": 2,
        "vendor": 3,
        "model_version": 4,
        "usage": 5,
        "completed_at": 6,
        "gateway_version": 7,
    },
}


@pytest.mark.parametrize("message", sorted(EXPECTED))
def test_proto_field_numbers(message):
    assert _message_fields(message) == EXPECTED[message]


def test_proto_enum_values():
    match = re.search(r"enum FinishReason\{(.*?)\}", PROTO)
    assert match
    values = dict(re.findall(r"(\w+)=(\d+)", match.group(1)))
    assert values == {
        "FINISH_REASON_UNSPECIFIED": "0",
        "FINISH_REASON_COMPLETE": "1",
        "FINISH_REASON_MAX_TOKENS": "2",
        "FINISH_REASON_MODEL_ARMOR_BLOCK": "3",
        "FINISH_REASON_BUDGET_BLOCK": "4",
        "FINISH_REASON_VENDOR_ERROR": "5",
        "FINISH_REASON_MANA_BLOCK": "6",
    }


def test_proto_service_and_rpc_names():
    assert "package chora.services.model_gateway.v1;" in PROTO
    match = re.search(r"service ModelGatewayService\{(.*?)\}", PROTO)
    assert match
    rpcs = re.findall(r"rpc (\w+)", match.group(1))
    assert rpcs == ["Invoke", "GroundedSearch", "Embed"]


def test_proto_no_removed_field_numbers_reused():
    # Field 16 on InvokeResponse is tool_calls_json; the old contract never
    # reused a retired number for a different purpose.
    response = _message_fields("InvokeResponse")
    assert response["tool_calls_json"] == 16
    assert response["image_mime_type"] == 15
