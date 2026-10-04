from __future__ import annotations

import base64
import time
from typing import Any

from blacksheep import Application, Request
from blacksheep.server.responses import json as reply
from blacksheep.server.responses import text

from .compat import embedding_inputs, flatten_responses_input, generation_params, model_entry, wants_grounding
from .service import Gateway, GatewayError


def create_app(gateway: Gateway, db: Any, models: dict[str, Any]) -> Application:
    app = Application()

    @app.exception_handler(GatewayError)
    async def handle_gateway_error(_: Any, exc: GatewayError) -> Any:
        return reply(
            {
                "error": {
                    "message": str(exc),
                    "type": "invalid_request_error" if exc.status < 500 else "upstream_error",
                    "code": exc.code,
                }
            },
            status=exc.status,
        )

    @app.router.get("/healthz")
    async def health() -> Any:
        return text("ok")

    @app.router.get("/readyz")
    async def ready() -> Any:
        ok = await db.ping()
        return text("ready" if ok else "not_ready", status=200 if ok else 503)

    @app.router.get("/v1/models")
    async def list_models() -> Any:
        data = [
            model_entry(models[model_id]) for model_id in dict.fromkeys(models)
        ]
        return reply({"object": "list", "data": data})

    @app.router.get("/v1/models/{model}")
    async def get_model(model: str) -> Any:
        spec = gateway.model(model)
        return reply(model_entry(spec))

    def attributes(request: Request) -> dict[str, Any]:
        def header(name: bytes) -> str:
            value = request.get_first_header(name)
            return value.decode() if value else ""

        return {
            "tenant_id": header(b"x-chora-tenant-id") or request.query.get("tenant") or None,
            "gcid": header(b"x-chora-gcid") or None,
            "traceparent": header(b"traceparent"),
            "tracestate": header(b"tracestate"),
        }

    @app.router.post("/v1/chat/completions")
    @app.router.post("/v1/completions")
    async def chat(request: Request) -> Any:
        body = await request.json()
        if body.get("stream"):
            raise GatewayError(
                501,
                "streaming_unsupported",
                'this gateway does not implement SSE streaming; retry with "stream": false',
            )
        if not body.get("model"):
            raise GatewayError(400, "missing_parameter", '"model" is required and must name an entry in the model registry')
        messages = body.get("messages") or ([{"role": "user", "content": body.get("prompt", "")}] if body.get("prompt") else [])
        if not messages:
            raise GatewayError(400, "missing_parameter", '"messages" must contain at least one message')
        prompt = ""
        system = ""
        for input_message in messages:
            if input_message.get("role") in ("system", "developer"):
                system += str(input_message.get("content", ""))
            elif input_message.get("role") == "user":
                prompt = str(input_message.get("content", ""))

        tools = body.get("tools", [])
        grounded = wants_grounding(tools)
        if not any(m.get("role") not in ("system", "developer") for m in messages):
            raise GatewayError(400, "empty_conversation", "messages contains only a system prompt; add at least one user message")
        response = await gateway.invoke(
            attributes(request)
            | {
                "model": body.get("model"),
                "prompt": prompt,
                "system": system,
                "messages": messages,
                "tools": tools,
                "params": generation_params(body),
                "modality": "GROUNDED" if grounded else "TEXT",
                "action_code": body.get("user"),
                "surface": "openai_compat",
            }
        )
        result = response["result"]
        message: dict[str, Any] = {"role": "assistant", "content": result.text}
        if result.tool_calls:
            message["tool_calls"] = result.tool_calls
        return reply(
            {
                "id": response["id"],
                "object": "chat.completion",
                "created": int(time.time()),
                "model": response["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if result.tool_calls else "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": response["usage"]["input"],
                    "completion_tokens": response["usage"]["output"],
                    "total_tokens": response["usage"]["input"] + response["usage"]["output"],
                },
                "chora_gateway": {
                    "vendor": response["vendor"],
                    "fallback_chain": response["fallback_chain"],
                    "latency_ms": response["latency_ms"],
                    "invocation_id": response["id"],
                    "grounded": grounded,
                    "citations": result.citations,
                    "search_queries": result.search_queries,
                },
            }
        )

    @app.router.post("/v1/responses")
    async def responses(request: Request) -> Any:
        body = await request.json()
        if body.get("stream"):
            raise GatewayError(
                501,
                "streaming_unsupported",
                'this gateway does not implement SSE streaming; retry with "stream": false',
            )
        grounded = wants_grounding(body.get("tools", []))
        try:
            prompt, response_messages = flatten_responses_input(body.get("input"))
        except ValueError as exc:
            raise GatewayError(400, "invalid_input", str(exc)) from exc
        response = await gateway.invoke(
            attributes(request)
            | {
                "model": body.get("model"),
                "prompt": prompt,
                "messages": response_messages,
                "system": body.get("instructions", ""),
                "params": {
                    key: body[key]
                    for key in (
                        "temperature", "top_p", "max_output_tokens",
                        "max_tokens", "seed", "reasoning_effort",
                    )
                    if key in body
                },
                "modality": "GROUNDED" if grounded else "TEXT",
                "action_code": body.get("user"),
                "surface": "responses",
            }
        )
        result = response["result"]
        output: list[dict[str, Any]] = []
        if grounded:
            output.append({"type": "web_search_call", "id": "ws_" + response["id"], "status": "completed"})
        output.append(
            {
                "type": "message", "id": "msg_" + response["id"], "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": result.text, "annotations": result.citations}],
            }
        )
        return reply(
            {
                "id": response["id"], "object": "response", "created_at": int(time.time()),
                "model": response["model"], "status": "completed", "output": output,
                "usage": {
                    "prompt_tokens": response["usage"]["input"],
                    "completion_tokens": response["usage"]["output"],
                    "total_tokens": response["usage"]["input"] + response["usage"]["output"],
                },
                "chora_gateway": {
                    "vendor": response["vendor"], "grounded": grounded,
                    "citations": result.citations, "search_queries": result.search_queries,
                    "fallback_chain": response["fallback_chain"], "latency_ms": response["latency_ms"],
                    "invocation_id": response["id"],
                    "grounding_surface": "responses" if grounded else "",
                },
            }
        )

    @app.router.post("/v1/images/generations")
    @app.router.post("/v1/images/edits")
    async def image(request: Request) -> Any:
        body = await request.json()
        if not body.get("prompt"):
            raise GatewayError(400, "missing_parameter", '"prompt" is required')
        if not body.get("model"):
            raise GatewayError(400, "missing_parameter", '"model" is required and must name an image-capable entry in the model registry')
        if body.get("response_format") == "url":
            raise GatewayError(501, "url_response_unsupported", "this gateway returns bytes, not hosted URLs")
        response = await gateway.invoke(
            attributes(request)
            | {
                "model": body.get("model"), "prompt": body.get("prompt", ""),
                "params": {key: body[key] for key in ("size", "quality", "style") if key in body},
                "modality": "IMAGE", "action_code": body.get("user"), "surface": "openai_compat",
            }
        )
        result = response["result"]
        return reply(
            {
                "created": int(time.time()),
                "data": [{"b64_json": base64.b64encode(result.image).decode(),
                          "revised_prompt": result.revised_prompt}],
                "usage": {
                    "prompt_tokens": response["usage"]["input"],
                    "completion_tokens": response["usage"]["output"],
                    "total_tokens": response["usage"]["input"] + response["usage"]["output"],
                },
                "mime_type": result.mime,
            }
        )

    @app.router.post("/v1/embeddings")
    async def embeddings(request: Request) -> Any:
        body = await request.json()
        if str(body.get("encoding_format", "")).lower() == "base64":
            raise GatewayError(501, "base64_unsupported", '"encoding_format": "base64" is not implemented; omit it or send "float"')
        try:
            inputs = embedding_inputs(body.get("input"))
        except ValueError as exc:
            raise GatewayError(400, "invalid_parameter", str(exc)) from exc
        rows: list[dict[str, Any]] = []
        total = 0
        for index, input_text in enumerate(inputs):
            embedded = await gateway.embed(
                attributes(request)
                | {
                    "model": body.get("model"), "text": input_text,
                    "dimensions": int(body.get("dimensions", 0)),
                    "agent_id": gateway.settings.default_agent_id,
                }
            )
            rows.append({"object": "embedding", "index": index, "embedding": embedded["values"]})
            total += embedded["input_tokens"]
        return reply(
            {
                "object": "list", "model": body.get("model"), "data": rows,
                "usage": {"prompt_tokens": total, "total_tokens": total},
            }
        )

    return app
