from __future__ import annotations

import base64
import time
from typing import Any

from blacksheep import Application, Request
from blacksheep.server.responses import json as reply
from blacksheep.server.responses import text
from blacksheep.server.routing import Router

from .compat import embedding_inputs, flatten_responses_input, generation_params, model_entry, wants_grounding
from .service import BudgetBlock, Gateway, GatewayError, Invocation

NOSNIFF = (b"x-content-type-options", b"nosniff")


def _json(data: Any, status: int = 200) -> Any:
    response = reply(data, status=status)
    response.add_header(*NOSNIFF)
    return response


def _text(value: str, status: int = 200) -> Any:
    response = text(value, status=status)
    response.add_header(*NOSNIFF)
    return response


def _error(status: int, err_type: str, code: str, message: str) -> Any:
    return _json({"error": {"message": message, "type": err_type, "code": code}}, status=status)


def _upstream_error_type(status: int) -> str:
    if status in (401, 403):
        return "authentication_error"
    if status == 429:
        return "rate_limit_error"
    if status == 404:
        return "invalid_request_error"
    return "upstream_error"


def create_app(gateway: Gateway, db: Any, models: dict[str, Any]) -> Application:
    # A dedicated router per application: BlackSheep's default router is shared
    # process-wide, and a second Application would reset it.
    app = Application(router=Router())

    @app.exception_handler(GatewayError)
    async def handle_gateway_error(_: Any, __: Any, exc: GatewayError) -> Any:
        if exc.upstream_status is not None and exc.upstream_status >= 400:
            return _error(exc.upstream_status, _upstream_error_type(exc.upstream_status), "upstream_error", exc.message)
        return _error(exc.status, exc.err_type, exc.code, exc.message)

    @app.exception_handler(BudgetBlock)
    async def handle_budget_block(_: Any, __: Any, exc: BudgetBlock) -> Any:
        return _error(
            402,
            "insufficient_quota",
            "budget_exhausted",
            exc.detail or "the tenant LLM budget for this period is exhausted",
        )

    # ------------------------------------------------------------------
    # Probes
    # ------------------------------------------------------------------

    @app.router.get("/healthz")
    async def health() -> Any:
        return _text("ok")

    @app.router.get("/readyz")
    async def ready() -> Any:
        return _text("ready")

    # ------------------------------------------------------------------
    # GET /v1/models
    # ------------------------------------------------------------------

    @app.router.get("/v1/models")
    async def list_models() -> Any:
        data = [model_entry(models[model_id]) for model_id in dict.fromkeys(models)]
        return _json({"object": "list", "data": data})

    @app.router.get("/v1/models/{model}")
    async def get_model(model: str) -> Any:
        try:
            spec = gateway.model(model)
        except GatewayError:
            return _error(
                404,
                "invalid_request_error",
                "model_not_found",
                f'model "{model}" is not in the registry',
            )
        return _json(model_entry(spec))

    # ------------------------------------------------------------------
    # Shared request assembly
    # ------------------------------------------------------------------

    def attributes(request: Request) -> dict[str, Any]:
        def header(name: bytes) -> str:
            value = request.get_first_header(name)
            return value.decode() if value else ""

        def query(name: str) -> str:
            values = request.query.get(name) or []
            return values[0] if values else ""

        return {
            "tenant_id": header(b"x-chora-tenant-id") or query("tenant") or gateway.settings.default_tenant_id,
            "gcid": header(b"x-chora-gcid") or gateway.settings.default_gcid,
            "agent_id": gateway.settings.default_agent_id,
            "traceparent": header(b"traceparent"),
            "tracestate": header(b"tracestate"),
        }

    async def _body(request: Request) -> dict[str, Any]:
        try:
            body = await request.json()
        except Exception as exc:
            raise ValueError(f"could not parse request body: {exc}") from exc
        if not isinstance(body, dict):
            raise ValueError("could not parse request body: request body must be a JSON object")
        return body

    def grounding_surface_for(model: str) -> tuple[str, Any]:
        """Returns (surface, error). Mirrors the old facade's grounding gate."""
        try:
            spec = gateway.model(model)
        except GatewayError:
            return "", _error(
                400,
                "invalid_request_error",
                "grounding_unavailable",
                f'model "{model}" is not in the registry',
            )
        if not spec.supports("web_search"):
            return "", _error(
                400,
                "invalid_request_error",
                "grounding_unavailable",
                f'model "{model}" does not advertise the "web_search" capability '
                f"(it has: {', '.join(spec.capabilities)})",
            )
        if spec.grounding is None:
            return "", _error(
                400,
                "invalid_request_error",
                "grounding_unavailable",
                f'model "{model}" advertises "web_search" but its registry entry configures no grounding endpoint',
            )
        return spec.grounding.effective_surface(spec.vendor), None

    # ------------------------------------------------------------------
    # POST /v1/chat/completions and /v1/completions
    # ------------------------------------------------------------------

    @app.router.post("/v1/chat/completions")
    @app.router.post("/v1/completions")
    async def chat(request: Request) -> Any:
        try:
            body = await _body(request)
        except ValueError as exc:
            return _error(400, "invalid_request_error", "invalid_json", f"could not parse request body: {exc}")
        if body.get("stream"):
            return _error(
                501,
                "invalid_request_error",
                "streaming_unsupported",
                'this gateway does not implement SSE streaming; retry with "stream": false',
            )
        model_name = body.get("model") or ""
        if not model_name:
            return _error(
                400,
                "invalid_request_error",
                "missing_parameter",
                '"model" is required and must name an entry in the model registry',
            )
        if model_name.strip().lower() not in models:
            return _error(
                404,
                "invalid_request_error",
                "model_not_found",
                f'model "{model_name}" is not in the registry; GET /v1/models lists what is',
            )
        messages = body.get("messages") or []
        if not messages:
            return _error(
                400,
                "invalid_request_error",
                "missing_parameter",
                '"messages" must contain at least one message',
            )

        tools = body.get("tools") or []
        grounded = False
        if wants_grounding(tools):
            surface, error = grounding_surface_for(model_name)
            if error is not None:
                return error
            if surface != "chat_completions":
                return _error(
                    400,
                    "invalid_request_error",
                    "wrong_grounding_surface",
                    f'model "{model_name}" is configured to ground on the {surface} surface; '
                    "use POST /v1/responses for hosted web search",
                )
            grounded = True

        system_parts: list[str] = []
        conversation: list[Any] = []
        for message in messages:
            role = str(message.get("role", "")).lower()
            if role in ("system", "developer"):
                text = _raw_to_text(message.get("content"))
                if text:
                    system_parts.append(text)
            else:
                conversation.append(message)
        if not conversation:
            return _error(
                400,
                "invalid_request_error",
                "empty_conversation",
                "messages contains only a system prompt; add at least one user message",
            )

        response = await gateway.invoke(
            attributes(request)
            | {
                "model": model_name,
                "system": "\n\n".join(system_parts),
                "messages": conversation,
                "tools": tools,
                "params": generation_params(body),
                "modality": "GROUNDED" if grounded else "TEXT",
                "action_code": body.get("user"),
                "surface": "openai_compat",
            }
        )
        return _json(_chat_response(response, grounded))

    # ------------------------------------------------------------------
    # POST /v1/responses
    # ------------------------------------------------------------------

    @app.router.post("/v1/responses")
    async def responses(request: Request) -> Any:
        try:
            body = await _body(request)
        except ValueError as exc:
            return _error(400, "invalid_request_error", "invalid_json", f"could not parse request body: {exc}")
        if body.get("stream"):
            return _error(
                501,
                "invalid_request_error",
                "streaming_unsupported",
                'this gateway does not implement SSE streaming; retry with "stream": false',
            )
        model_name = body.get("model") or ""
        if not model_name:
            return _error(
                400,
                "invalid_request_error",
                "missing_parameter",
                '"model" is required and must name an entry in the model registry',
            )
        if model_name.strip().lower() not in models:
            return _error(
                404,
                "invalid_request_error",
                "model_not_found",
                f'model "{model_name}" is not in the registry; GET /v1/models lists what is',
            )

        tools = body.get("tools") or []
        grounded = wants_grounding(tools)
        if grounded:
            _, error = grounding_surface_for(model_name)
            if error is not None:
                return error

        try:
            prompt, response_messages = flatten_responses_input(body.get("input"))
        except ValueError as exc:
            return _error(400, "invalid_request_error", "invalid_input", str(exc))

        params = {
            key: body[key]
            for key in ("temperature", "top_p", "max_output_tokens", "max_tokens", "seed", "reasoning_effort")
            if key in body
        }
        response = await gateway.invoke(
            attributes(request)
            | {
                "model": model_name,
                "prompt": prompt,
                "messages": response_messages,
                "system": body.get("instructions", ""),
                "params": params,
                "modality": "GROUNDED" if grounded else "TEXT",
                "action_code": body.get("user"),
                "surface": "responses",
            }
        )
        return _json(_responses_response(response, grounded))

    # ------------------------------------------------------------------
    # POST /v1/images/generations and /v1/images/edits
    # ------------------------------------------------------------------

    @app.router.post("/v1/images/generations")
    @app.router.post("/v1/images/edits")
    async def image(request: Request) -> Any:
        try:
            body = await _body(request)
        except ValueError as exc:
            return _error(400, "invalid_request_error", "invalid_json", f"could not parse request body: {exc}")
        if not body.get("prompt"):
            return _error(400, "invalid_request_error", "missing_parameter", '"prompt" is required')
        model_name = body.get("model") or ""
        if not model_name:
            return _error(
                400,
                "invalid_request_error",
                "missing_parameter",
                '"model" is required and must name an image-capable entry in the model registry',
            )
        if model_name.strip().lower() not in models:
            return _error(
                404,
                "invalid_request_error",
                "model_not_found",
                f'model "{model_name}" is not in the registry; GET /v1/models lists what is',
            )
        if str(body.get("response_format", "")).lower() == "url":
            return _error(
                501,
                "invalid_request_error",
                "url_response_unsupported",
                'this gateway returns bytes, not hosted URLs; use "response_format": "b64_json" (the default)',
            )
        response = await gateway.invoke(
            attributes(request)
            | {
                "model": model_name,
                "prompt": body.get("prompt", ""),
                "params": {key: body[key] for key in ("size", "quality", "style") if key in body},
                "modality": "IMAGE",
                "action_code": body.get("user"),
                "surface": "openai_compat",
            }
        )
        if response.finish != "complete" or not response.image:
            return _error(
                502,
                "upstream_error",
                "no_image",
                response.finish_detail or "the model returned no image",
            )
        return _json(
            {
                "created": int(time.time()),
                "data": [
                    {
                        "b64_json": base64.b64encode(response.image).decode(),
                        **({"revised_prompt": response.revised_prompt} if response.revised_prompt else {}),
                    }
                ],
                "usage": {
                    "prompt_tokens": response.input_tokens,
                    "completion_tokens": response.output_tokens,
                    "total_tokens": response.input_tokens + response.output_tokens,
                },
                **({"mime_type": response.image_mime} if response.image_mime else {}),
            }
        )

    # ------------------------------------------------------------------
    # POST /v1/embeddings
    # ------------------------------------------------------------------

    @app.router.post("/v1/embeddings")
    async def embeddings(request: Request) -> Any:
        try:
            body = await _body(request)
        except ValueError as exc:
            return _error(400, "invalid_request_error", "invalid_json", f"could not parse request body: {exc}")
        model_name = body.get("model") or ""
        if not model_name:
            return _error(400, "invalid_request_error", "missing_parameter", '"model" is required')
        if model_name.strip().lower() not in models:
            return _error(
                404,
                "invalid_request_error",
                "model_not_found",
                f'model "{model_name}" is not in the registry',
            )
        if str(body.get("encoding_format", "")).lower() == "base64":
            return _error(
                501,
                "invalid_request_error",
                "base64_unsupported",
                '"encoding_format": "base64" is not implemented; omit it or send "float"',
            )
        try:
            inputs = embedding_inputs(body.get("input"))
        except ValueError as exc:
            return _error(400, "invalid_request_error", "invalid_parameter", str(exc))

        rows: list[dict[str, Any]] = []
        total = 0
        for index, input_text in enumerate(inputs):
            embedded = await gateway.embed(
                attributes(request)
                | {
                    "model": model_name,
                    "text": input_text,
                    "dimensions": int(body.get("dimensions") or 0),
                    "agent_id": gateway.settings.default_agent_id,
                }
            )
            rows.append({"object": "embedding", "index": index, "embedding": embedded["values"]})
            total += embedded["input_tokens"]
        return _json(
            {
                "object": "list",
                "model": model_name,
                "data": rows,
                "usage": {"prompt_tokens": total, "completion_tokens": 0, "total_tokens": total},
            }
        )

    # ------------------------------------------------------------------
    # Method mismatches — the old facade answers 405 with the OpenAI envelope
    # ------------------------------------------------------------------

    async def _method_not_allowed(request: Request) -> Any:
        allowed = "POST" if request.method in ("GET", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS") else "GET"
        return _error(405, "invalid_request_error", "method_not_allowed", f"{allowed} only")

    for _path in (
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/responses",
        "/v1/images/generations",
        "/v1/images/edits",
        "/v1/embeddings",
    ):
        for _method in ("get", "put", "patch", "delete", "head", "options"):
            _register_method_guard(app, _path, _method, _method_not_allowed)

    for _method in ("post", "put", "patch", "delete", "head", "options"):
        _register_method_guard(app, "/v1/models", _method, _method_not_allowed)
        _register_method_guard(app, "/v1/models/{model}", _method, _method_not_allowed)
        _register_method_guard(app, "/healthz", _method, _method_not_allowed)
        _register_method_guard(app, "/readyz", _method, _method_not_allowed)

    return app


def _register_method_guard(app: Application, path: str, method: str, handler: Any) -> None:
    route = getattr(app.router, method)
    route(path)(handler)


# ---------------------------------------------------------------------------
# Response shapes
# ---------------------------------------------------------------------------


def _chat_response(response: Invocation, grounded: bool) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": response.text}
    if response.tool_calls:
        message["tool_calls"] = response.tool_calls
    meta: dict[str, Any] = {
        "vendor": response.vendor,
        "latency_ms": response.latency_ms,
        "invocation_id": response.id,
    }
    if response.fallback_chain:
        meta["fallback_chain"] = response.fallback_chain
    if grounded:
        meta["grounded"] = True
    if response.citations:
        meta["citations"] = response.citations
    if response.search_queries:
        meta["search_queries"] = response.search_queries
    return {
        "id": response.id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": response.model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": _openai_finish_reason(response.finish),
            }
        ],
        "usage": {
            "prompt_tokens": response.input_tokens,
            "completion_tokens": response.output_tokens,
            "total_tokens": response.input_tokens + response.output_tokens,
        },
        "chora_gateway": meta,
    }


def _openai_finish_reason(finish: str) -> str:
    if finish == "max_tokens":
        return "length"
    return "stop"


def _responses_response(response: Invocation, grounded: bool) -> dict[str, Any]:
    output: list[dict[str, Any]] = []
    if grounded:
        action: dict[str, Any] = {"type": "search"}
        if len(response.search_queries) == 1:
            action["query"] = response.search_queries[0]
        elif response.search_queries:
            action["queries"] = response.search_queries
        output.append(
            {
                "type": "web_search_call",
                "id": "ws_" + response.id,
                "status": "completed",
                "action": action,
            }
        )
    messages = response.messages or [{"text": response.text, "citations": response.citations}]
    for index, msg in enumerate(messages):
        output.append(
            {
                "type": "message",
                "id": f"msg_{response.id}_{index}",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": msg["text"],
                        "annotations": _annotations_to_json(msg["citations"]),
                    }
                ],
            }
        )
    meta: dict[str, Any] = {
        "vendor": response.vendor,
        "grounded": grounded,
        "citations": response.citations,
        "latency_ms": response.latency_ms,
        "invocation_id": response.id,
    }
    if response.search_queries:
        meta["search_queries"] = response.search_queries
    if response.fallback_chain:
        meta["fallback_chain"] = response.fallback_chain
    if response.grounding_surface:
        meta["grounding_surface"] = response.grounding_surface
    if response.finish_detail:
        meta["finish_reason"] = response.finish
        meta["finish_detail"] = response.finish_detail
    if response.messages:
        meta["messages"] = response.messages
    return {
        "id": response.id,
        "object": "response",
        "created_at": int(time.time()),
        "model": response.model,
        "status": "completed",
        "output": output,
        "usage": {
            "prompt_tokens": response.input_tokens,
            "completion_tokens": response.output_tokens,
            "total_tokens": response.input_tokens + response.output_tokens,
        },
        "chora_gateway": meta,
    }


def _annotations_to_json(citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for citation in citations:
        annotation: dict[str, Any] = {"type": "url_citation", "url": citation["url"]}
        if citation.get("title"):
            annotation["title"] = citation["title"]
        if citation.get("start_index") or citation.get("end_index"):
            annotation["start_index"] = int(citation.get("start_index") or 0)
            annotation["end_index"] = int(citation.get("end_index") or 0)
        out.append(annotation)
    return out


def _raw_to_text(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        return "".join(
            part.get("text", "") for part in raw if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    return ""
