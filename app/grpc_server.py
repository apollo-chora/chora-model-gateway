from __future__ import annotations

import json
from typing import Any

import grpc
import model_gateway_service_pb2 as pb
import model_gateway_service_pb2_grpc as pbg
from google.protobuf.json_format import MessageToDict
from google.protobuf.timestamp_pb2 import Timestamp

from .service import Gateway, GatewayError


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
                    "params": MessageToDict(
                        request.generation_config, preserving_proto_field_name=True
                    )
                    if request.HasField("generation_config")
                    else {},
                    "modality": request.response_modality or "TEXT",
                    "action_code": request.action_code,
                    "surface": request.surface,
                    "dispatch_idempotency_key": request.dispatch_idempotency_key,
                    "fallback_ids": list(request.fallback_logical_model_ids),
                    "traceparent": request.traceparent,
                    "tracestate": request.tracestate,
                }
            )
        except GatewayError as exc:
            code = (
                grpc.StatusCode.RESOURCE_EXHAUSTED
                if exc.status in (402, 429)
                else grpc.StatusCode.UNAVAILABLE
            )
            await context.abort(code, str(exc))
            raise

        result = response["result"]
        return pb.InvokeResponse(
            invocation_id=response["id"],
            completion=result.text,
            usage=pb.TokenUsage(
                input_tokens=response["usage"]["input"],
                output_tokens=response["usage"]["output"],
                cached_tokens=response["usage"]["cached"],
                cost_micros=response["usage"]["cost"],
            ),
            vendor=response["vendor"],
            model_version=response["model"],
            fallback_chain=response["fallback_chain"],
            latency_ms=response["latency_ms"],
            finish_reason=pb.FINISH_REASON_COMPLETE,
            completed_at=_timestamp(),
            gateway_version=self.gateway.settings.service_version,
            image_bytes=result.image,
            image_mime_type=result.mime,
            tool_calls_json=json.dumps(result.tool_calls) if result.tool_calls else "",
        )

    async def Embed(self, request: Any, context: Any) -> Any:
        if not request.tenant_id or not request.gcid or not request.agent_id or not request.text:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "embed: tenant_id, gcid, agent_id and text required",
            )
        response = await self.gateway.embed(
            {
                "invocation_id": request.invocation_id,
                "tenant_id": request.tenant_id,
                "gcid": request.gcid,
                "agent_id": request.agent_id,
                "logical_model_id": request.logical_model_id,
                "text": request.text,
                "dimensions": request.output_dimensions,
                "traceparent": request.traceparent,
                "tracestate": request.tracestate,
            }
        )
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


async def start_grpc(gateway: Gateway, port: int) -> Any:
    server = grpc.aio.server()
    pbg.add_ModelGatewayServiceServicer_to_server(ModelGateway(gateway), server)
    server.add_insecure_port(f"[::]:{port}")
    await server.start()
    return server
