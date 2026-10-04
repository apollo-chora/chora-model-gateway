from __future__ import annotations

import json
from typing import Any

import grpc
from google.protobuf.json_format import MessageToDict
from google.protobuf.timestamp_pb2 import Timestamp

import model_gateway_service_pb2 as pb
import model_gateway_service_pb2_grpc as pbg

from .service import BudgetBlock, Gateway, GatewayError

_FINISH_REASON_MAP = {
    "unspecified": pb.FINISH_REASON_UNSPECIFIED,
    "complete": pb.FINISH_REASON_COMPLETE,
    "max_tokens": pb.FINISH_REASON_MAX_TOKENS,
    "budget_block": pb.FINISH_REASON_BUDGET_BLOCK,
    "vendor_error": pb.FINISH_REASON_VENDOR_ERROR,
}


def _timestamp() -> Timestamp:
    value = Timestamp()
    value.GetCurrentTime()
    return value


class ModelGateway(pbg.ModelGatewayServiceServicer):
    def __init__(self, gateway: Gateway) -> None:
        self.gateway = gateway

    async def Invoke(self, request: Any, context: Any) -> Any:
        try:
            response = await self.gateway.invoke(
                {
                    "invocation_id": request.invocation_id,
                    "tenant_id": request.tenant_id,
                    "gcid": request.gcid,
                    "agent_id": request.agent_id,
                    "logical_model_id": request.logical_model_id,
                    "prompt": request.prompt,
                    "system": request.system_prompt,
                    "messages": json.loads(request.contents_json) if request.contents_json else None,
                    "tools": json.loads(request.tools_json) if request.tools_json else None,
                    "params": (
                        MessageToDict(request.generation_config, preserving_proto_field_name=True)
                        if request.HasField("generation_config")
                        else {}
                    ),
                    "modality": request.response_modality or "TEXT",
                    "action_code": request.action_code,
                    "surface": request.surface,
                    "dispatch_idempotency_key": request.dispatch_idempotency_key,
                    "fallback_ids": list(request.fallback_logical_model_ids),
                    "traceparent": request.traceparent or _metadata(context, "traceparent"),
                    "tracestate": request.tracestate,
                }
            )
        except BudgetBlock as exc:
            return pb.InvokeResponse(
                invocation_id=request.invocation_id,
                finish_reason=pb.FINISH_REASON_BUDGET_BLOCK,
                finish_detail=exc.detail,
                completed_at=_timestamp(),
                gateway_version=self.gateway.settings.service_version,
            )
        except GatewayError as exc:
            await context.abort(*_map_error(exc))
            raise

        return pb.InvokeResponse(
            invocation_id=response.id,
            completion=response.text,
            usage=pb.TokenUsage(
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                cached_tokens=response.cached_tokens,
                cost_micros=response.cost,
            ),
            vendor=response.vendor,
            model_version=response.model,
            fallback_chain=response.fallback_chain,
            latency_ms=response.latency_ms,
            finish_reason=_FINISH_REASON_MAP.get(response.finish, pb.FINISH_REASON_UNSPECIFIED),
            finish_detail=response.finish_detail,
            completed_at=_timestamp(),
            gateway_version=self.gateway.settings.service_version,
            image_bytes=response.image,
            image_mime_type=response.image_mime,
            tool_calls_json=json.dumps(response.tool_calls) if response.tool_calls else "",
        )

    async def Embed(self, request: Any, context: Any) -> Any:
        if not request.tenant_id or not request.gcid or not request.agent_id or not request.text:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "embed: tenant_id, gcid, agent_id and text required",
            )
            raise ValueError("embed: tenant_id, gcid, agent_id and text required")
        try:
            response = await self.gateway.embed(
                {
                    "invocation_id": request.invocation_id,
                    "tenant_id": request.tenant_id,
                    "gcid": request.gcid,
                    "agent_id": request.agent_id,
                    "logical_model_id": request.logical_model_id,
                    "text": request.text,
                    "dimensions": request.output_dimensions,
                    "traceparent": request.traceparent or _metadata(context, "traceparent"),
                    "tracestate": request.tracestate,
                }
            )
        except GatewayError as exc:
            await context.abort(*_map_error(exc))
            raise
        return pb.EmbedResponse(
            invocation_id=response["id"],
            values=response["values"],
            vendor=response["vendor"],
            model_version=response["model"],
            usage=pb.TokenUsage(input_tokens=response["input_tokens"]),
            completed_at=_timestamp(),
            gateway_version=self.gateway.settings.service_version,
        )

    async def GroundedSearch(self, request: Any, context: Any) -> Any:
        await context.abort(
            grpc.StatusCode.UNIMPLEMENTED,
            "GroundedSearch is retired; use Invoke response_modality=GROUNDED or /v1/responses",
        )
        raise ValueError("GroundedSearch is retired")


def _map_error(exc: GatewayError) -> tuple[Any, str]:
    """Translate a gateway error to a gRPC status, mirroring the old adapter:
    a vendor failure carrying an upstream HTTP status is relayed rather than
    flattened; a gateway misconfiguration is FailedPrecondition; an invoke
    failure is Unavailable; a bare error (the embed guards) is InvalidArgument."""
    if exc.upstream_status is not None:
        if exc.upstream_status in (401, 403):
            return grpc.StatusCode.UNAUTHENTICATED, f"upstream credential rejected: {exc.message}"
        if exc.upstream_status == 404:
            return grpc.StatusCode.NOT_FOUND, f"upstream model or endpoint not found: {exc.message}"
        if exc.upstream_status == 429:
            return grpc.StatusCode.RESOURCE_EXHAUSTED, f"upstream rate limited: {exc.message}"
        return grpc.StatusCode.UNAVAILABLE, f"upstream provider error: {exc.message}"
    if exc.kind == "config":
        return grpc.StatusCode.FAILED_PRECONDITION, exc.message
    if exc.kind == "bare":
        return grpc.StatusCode.INVALID_ARGUMENT, exc.message
    return grpc.StatusCode.UNAVAILABLE, exc.message


def _metadata(context: Any, name: str) -> str:
    for item in context.invocation_metadata():
        if item.key == name and item.value:
            return item.value
    return ""


async def start_grpc(gateway: Gateway, port: int) -> tuple[Any, int]:
    server = grpc.aio.server()
    pbg.add_ModelGatewayServiceServicer_to_server(ModelGateway(gateway), server)
    assigned = server.add_insecure_port(f"[::]:{port}")
    await server.start()
    return server, assigned
