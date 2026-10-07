from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import GroundingSpec, ModelSpec

logger = logging.getLogger(__name__)

PostFn = Callable[[str, dict[str, Any], dict[str, str]], Awaitable[dict[str, Any]]]
FetchFn = Callable[[str], Awaitable[tuple[bytes, str]]]

FINISH_COMPLETE = "complete"
FINISH_MAX_TOKENS = "max_tokens"
FINISH_VENDOR_ERROR = "vendor_error"
FINISH_UNSPECIFIED = "unspecified"


def _attachment_for_bytes(mime: str, b64: str, uri: str) -> dict[str, Any]:
    """Build the OpenAI content part for a fetched grounding blob.

    Images ride as ``image_url`` data URLs; everything else rides as a ``file``
    part (OpenRouter/OpenAI accept ``data:`` URLs there, which is how a PDF
    reaches a multimodal model without the provider needing object-store
    access).
    """
    kind = (mime or "").split(";")[0].strip().lower()
    if kind.startswith("image/"):
        return {"type": "image_url", "image_url": {"url": f"data:{kind};base64,{b64}"}}
    name = uri.rsplit("/", 1)[-1] if uri else "material"
    if not name:
        name = "material"
    if kind:
        name = f"{name}.{kind.split('/')[-1]}" if "." not in name else name
    return {
        "type": "file",
        "file": {"filename": name, "file_data": f"data:{kind or 'application/octet-stream'};base64,{b64}"},
    }


def _fetch_object(uri: str) -> bytes:
    """Fetch an ``s3://bucket/key`` object (the deployment's object store).

    Runs in a worker thread — boto3 is synchronous. Credentials come from the
    standard S3_* env the rest of the stack uses.
    """
    import os

    import boto3  # lazy: keeps the test path SDK-free

    if not uri.startswith("s3://"):
        raise ValueError(f"unsupported object uri: {uri}")
    rest = uri[len("s3://") :]
    bucket, _, key = rest.partition("/")
    if not bucket or not key:
        raise ValueError(f"malformed object uri: {uri}")
    client = boto3.client(
        "s3",
        endpoint_url=os.getenv("S3_ENDPOINT") or None,
        aws_access_key_id=os.getenv("S3_ACCESS_KEY_ID") or os.getenv("S3_ACCESS_KEY") or None,
        aws_secret_access_key=os.getenv("S3_SECRET_ACCESS_KEY") or os.getenv("S3_SECRET_KEY") or None,
        region_name=os.getenv("S3_REGION") or "us-east-1",
    )
    obj = client.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()


class ProviderError(Exception):
    """A non-2xx response from the provider. Carries the provider's own HTTP
    status so the facade can relay it instead of flattening everything."""

    def __init__(self, status: int, endpoint: str, body: str) -> None:
        super().__init__(f"upstream {status} from {endpoint}: {body}")
        self.status = status
        self.endpoint = endpoint
        self.body = body

    def upstream_status(self) -> int:
        return self.status


@dataclass(slots=True)
class Result:
    text: str = ""
    model: str = ""
    vendor: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cost: int = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    image: bytes = b""
    mime: str = ""
    revised_prompt: str = ""
    finish: str = FINISH_COMPLETE
    finish_detail: str = ""


def sniff_image_mime(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if len(data) >= 3 and data[0] == 0xFF and data[1] == 0xD8 and data[2] == 0xFF:
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    return "image/png"


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _first_non_empty(*values: str) -> str:
    for value in values:
        if value:
            return value
    return ""


class _HttpTransport:
    """Default transport: httpx with one client per (base_url, auth) pair.

    The timeout is generous on purpose: a grounded reasoning run makes several
    upstream calls (search, page reads, then the answer) and can take well over
    a minute. The old gateway's HTTP server used a 5-minute write timeout.
    """

    def __init__(self, timeout: float = 300.0) -> None:
        self._timeout = timeout
        self._clients: dict[tuple[str, str, str], httpx.AsyncClient] = {}

    def _client(self, spec: ModelSpec) -> httpx.AsyncClient:
        auth = spec.api_key
        key = (spec.base_url, auth, json.dumps(spec.extra_headers, sort_keys=True))
        client = self._clients.get(key)
        if client is None:
            headers: dict[str, str] = {}
            # Anthropic authenticates with x-api-key, not an Authorization bearer.
            if auth and not _is_anthropic(spec):
                headers["Authorization"] = f"Bearer {auth}"
            headers.update(spec.extra_headers)
            client = httpx.AsyncClient(
                base_url=spec.base_url,
                headers=headers,
                timeout=httpx.Timeout(self._timeout, connect=10.0),
            )
            self._clients[key] = client
        return client

    async def post(
        self, spec: ModelSpec, url: str, payload: dict[str, Any], traceparent: str, tracestate: str
    ) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if traceparent:
            headers["traceparent"] = traceparent
        if tracestate:
            headers["tracestate"] = tracestate
        client = self._client(spec)
        response = await client.post(url, content=json.dumps(payload), headers=headers)
        if response.status_code >= 400:
            raise ProviderError(response.status_code, url, _truncate(response.text.strip(), 2048))
        return response.json()

    async def post_anthropic(
        self,
        spec: ModelSpec,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        traceparent: str,
        tracestate: str,
    ) -> dict[str, Any]:
        request_headers = dict(headers)
        if traceparent:
            request_headers["traceparent"] = traceparent
        if tracestate:
            request_headers["tracestate"] = tracestate
        response = await self._client(spec).post(url, content=json.dumps(payload), headers=request_headers)
        if response.status_code >= 400:
            raise ProviderError(response.status_code, url, _truncate(response.text.strip(), 2048))
        return response.json()

    async def fetch(self, url: str) -> tuple[bytes, str]:
        # The download URL is provider-minted and pre-signed: no credential is
        # attached, so a leak to a third-party host is impossible.
        response = await httpx.AsyncClient().get(url)
        if response.status_code >= 400:
            raise ProviderError(response.status_code, url, f"image fetch failed: {response.status_code}")
        return response.content[: 32 << 20], (response.headers.get("Content-Type") or "image/png")


class Runtime:
    """Single-shot provider dispatch. One request in, one completion out —
    the gateway is a gateway, not an agent loop."""

    def __init__(
        self,
        search_provider: str = "exa",
        post: PostFn | None = None,
        fetch: FetchFn | None = None,
    ) -> None:
        self.search_provider = search_provider
        self._transport = _HttpTransport()
        self._post = post
        self._fetch = fetch

    async def _dispatch(
        self, spec: ModelSpec, url: str, payload: dict[str, Any], traceparent: str, tracestate: str
    ) -> dict[str, Any]:
        if self._post is not None:
            return await self._post(url, payload, self._headers(spec, traceparent, tracestate))
        return await self._transport.post(spec, url, payload, traceparent, tracestate)

    def _headers(self, spec: ModelSpec, traceparent: str, tracestate: str) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if spec.api_key:
            headers["Authorization"] = f"Bearer {spec.api_key}"
        headers.update(spec.extra_headers)
        if traceparent:
            headers["traceparent"] = traceparent
        if tracestate:
            headers["tracestate"] = tracestate
        return headers

    async def _fetch_image(self, url: str) -> tuple[bytes, str]:
        if self._fetch is not None:
            return await self._fetch(url)
        return await self._transport.fetch(url)

    # ------------------------------------------------------------------
    # Text dispatch
    # ------------------------------------------------------------------

    async def generate(
        self,
        spec: ModelSpec,
        prompt: str,
        system: str = "",
        messages: Any = None,
        tools: Any = None,
        params: dict[str, Any] | None = None,
        grounded: bool = False,
        traceparent: str = "",
        tracestate: str = "",
    ) -> Result:
        if _is_anthropic(spec):
            return await self._anthropic(
                spec, prompt, system, messages, tools, params or {}, grounded, traceparent, tracestate
            )
        surface = spec.grounding.effective_surface(spec.vendor) if spec.grounding else "responses"
        if grounded and surface == "chat_completions":
            return await self._openai_chat(
                spec,
                prompt,
                system,
                messages,
                tools,
                params or {},
                grounded=True,
                traceparent=traceparent,
                tracestate=tracestate,
            )
        if grounded:
            return await self._openai_responses(spec, prompt, system, messages, params or {}, traceparent, tracestate)
        return await self._openai_chat(
            spec,
            prompt,
            system,
            messages,
            tools,
            params or {},
            traceparent=traceparent,
            tracestate=tracestate,
        )

    # ------------------------------------------------------------------
    # OpenAI-shaped: chat completions
    # ------------------------------------------------------------------

    def _chat_url(self, spec: ModelSpec) -> str:
        return _resolve_endpoint(spec.base_url, spec.chat_completions_path, "/chat/completions")

    async def _build_chat_body(
        self,
        spec: ModelSpec,
        prompt: str,
        system: str,
        messages: Any,
        tools: Any,
        params: dict[str, Any],
        grounded: bool,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"model": spec.model, "stream": False}
        normalized = await self._normalize_chat_messages(messages) if isinstance(messages, list) else []
        if normalized:
            body["messages"] = normalized
        else:
            chat_messages: list[dict[str, Any]] = []
            if system:
                chat_messages.append({"role": "system", "content": system})
            chat_messages.append({"role": "user", "content": prompt})
            body["messages"] = chat_messages
        if tools:
            body["tools"] = tools
        if grounded and spec.grounding and spec.grounding.effective_surface(spec.vendor) == "chat_completions":
            body["tools"] = list(body.get("tools") or []) + [self._grounding_tool(spec.grounding, "chat_completions")]
        self._apply_generation_config(body, params)
        return body

    async def _normalize_chat_messages(self, messages: list[Any]) -> list[dict[str, Any]]:
        """Coerce ADK/genai-shaped entries into the OpenAI chat shape.

        The Go client sends ``contents_json`` as genai ``[]*Content``
        (``{"role": "user", "parts": [...]}``) because that is what the gemini
        adapter consumes. The OpenAI chat surface expects ``content``, so
        forwarding those entries verbatim makes the provider reject the whole
        request with ``Missing required parameter: messages[0].content``.

        Parts are translated rather than dropped: the grounding plugin attaches
        the uploaded batch material as a ``fileData`` (object-store URI) part and
        the grounding contract is that THIS gateway dereferences it (see
        chora-adk-common/groundingplugin) — no provider can fetch an ``s3://``
        URI. Images become ``image_url`` data URLs, documents become ``file``
        parts, and text-bearing material is inlined as text. Entries with no
        usable part are dropped so the caller falls back to the flat prompt.
        """
        out: list[dict[str, Any]] = []
        for entry in messages:
            if not isinstance(entry, dict):
                continue
            if isinstance(entry.get("content"), (str, list)) and entry.get("content"):
                out.append(entry)
                continue
            parts = entry.get("parts")
            if not isinstance(parts, list):
                continue
            text = "".join(str(p.get("text", "")) for p in parts if isinstance(p, dict))
            attachments = await self._attachments_from_parts(parts)
            if not text and not attachments:
                continue
            role = str(entry.get("role") or "user")
            if role == "model":  # genai names the assistant turn "model"
                role = "assistant"
            if attachments:
                content: Any = ([{"type": "text", "text": text}] if text else []) + attachments
            else:
                content = text
            out.append({"role": role, "content": content})
        return out

    async def _attachments_from_parts(self, parts: list[Any]) -> list[dict[str, Any]]:
        """Translate genai ``inlineData`` / ``fileData`` parts into OpenAI
        content parts, fetching object-store URIs on the way."""
        out: list[dict[str, Any]] = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            inline = part.get("inlineData") or part.get("inline_data")
            filedata = part.get("fileData") or part.get("file_data")
            if isinstance(inline, dict):
                mime = str(inline.get("mimeType") or inline.get("mime_type") or "")
                data = str(inline.get("data") or "")
                if data:
                    out.append(_attachment_for_bytes(mime, data, ""))
            elif isinstance(filedata, dict):
                uri = str(filedata.get("fileUri") or filedata.get("file_uri") or "")
                mime = str(filedata.get("mimeType") or filedata.get("mime_type") or "")
                if not uri:
                    continue
                try:
                    raw = await asyncio.to_thread(_fetch_object, uri)
                except Exception:  # noqa: BLE001 — a missing blob must not kill the call
                    logger.warning("grounding.object_fetch_failed", extra={"uri": uri})
                    continue
                out.append(_attachment_for_bytes(mime, base64.b64encode(raw).decode("ascii"), uri))
        return out

    def _apply_generation_config(self, body: dict[str, Any], cfg: dict[str, Any]) -> None:
        for key, value in cfg.items():
            if value is None:
                continue
            if key in ("temperature", "top_p"):
                number = _number(value)
                if number is not None:
                    body[key] = number
            elif key in ("max_tokens", "max_completion_tokens"):
                number = _number(value)
                if number is not None:
                    body["max_tokens"] = int(number)
            elif key == "n":
                number = _number(value)
                if number is not None:
                    body["n"] = int(number)
            elif key == "seed":
                number = _number(value)
                if number is not None:
                    body["seed"] = int(number)
            elif key == "stop":
                if isinstance(value, str):
                    body["stop"] = value
                elif isinstance(value, list):
                    body["stop"] = [item for item in value if isinstance(item, str)]
            elif key == "tool_choice":
                body["tool_choice"] = value
            elif key == "response_format":
                body["response_format"] = value

    async def _openai_chat(
        self,
        spec: ModelSpec,
        prompt: str,
        system: str,
        messages: Any,
        tools: Any,
        params: dict[str, Any],
        grounded: bool = False,
        traceparent: str = "",
        tracestate: str = "",
    ) -> Result:
        body = await self._build_chat_body(spec, prompt, system, messages, tools, params, grounded)
        raw = await self._dispatch(spec, self._chat_url(spec), body, traceparent, tracestate)
        return self._chat_to_result(spec, raw)

    def _chat_to_result(self, spec: ModelSpec, raw: dict[str, Any]) -> Result:
        usage = raw.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        cached = int(details.get("cached_tokens") or 0)
        input_tokens = int(usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("completion_tokens") or 0)
        model = _first_non_empty(raw.get("model", ""), spec.model)
        result = Result(
            model=model,
            vendor=spec.vendor,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached,
            cost=spec.cost_micros(input_tokens - cached, output_tokens, cached),
            finish=FINISH_COMPLETE,
        )
        choices = raw.get("choices") or []
        if not choices:
            result.finish = FINISH_UNSPECIFIED
            result.finish_detail = "upstream returned no choices"
            return result
        choice = choices[0]
        message = choice.get("message") or {}
        result.text = self._content_to_text(message.get("content"))
        result.finish = self._map_finish_reason(choice.get("finish_reason") or "")
        result.citations = self._citations_from_annotations(message.get("annotations") or [])
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            result.tool_calls = [
                {
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": (tc.get("function") or {}).get("name", ""),
                        "arguments": (tc.get("function") or {}).get("arguments", ""),
                    },
                }
                for tc in tool_calls
            ]
        return result

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                part.get("text", "") for part in content if isinstance(part, dict) and isinstance(part.get("text"), str)
            )
        return ""

    @staticmethod
    def _map_finish_reason(reason: str) -> str:
        if reason in ("stop", "tool_calls", "function_call"):
            return FINISH_COMPLETE
        if reason == "length":
            return FINISH_MAX_TOKENS
        if reason == "content_filter":
            return FINISH_UNSPECIFIED
        return FINISH_COMPLETE

    @staticmethod
    def _citations_from_annotations(annotations: list[Any]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for annotation in annotations:
            if not isinstance(annotation, dict):
                continue
            url = annotation.get("url")
            if not url:
                continue
            kind = annotation.get("type") or ""
            if kind and kind != "url_citation":
                continue
            citation: dict[str, Any] = {"url": url}
            if annotation.get("title"):
                citation["title"] = annotation["title"]
            start = annotation.get("start_index")
            end = annotation.get("end_index")
            if start or end:
                citation["start_index"] = int(start or 0)
                citation["end_index"] = int(end or 0)
            out.append(citation)
        return out

    # ------------------------------------------------------------------
    # OpenAI-shaped: responses (hosted web search)
    # ------------------------------------------------------------------

    def _responses_url(self, spec: ModelSpec) -> str:
        path = spec.grounding.responses_path if spec.grounding else ""
        return _resolve_endpoint(spec.base_url, path, "/responses")

    def _grounding_tool(self, grounding: GroundingSpec, surface: str) -> dict[str, Any]:
        tool: dict[str, Any] = {"type": grounding.effective_tool_type(surface)}
        if surface == "messages":
            tool["name"] = grounding.effective_tool_name()
        if grounding.max_uses > 0:
            tool["max_uses"] = grounding.max_uses
        for key, value in grounding.extra_tool_fields.items():
            if key == "type":
                continue
            tool[key] = value
        return tool

    def _build_responses_body(
        self,
        spec: ModelSpec,
        prompt: str,
        system: str,
        messages: Any,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"model": spec.model, "stream": False}
        if isinstance(messages, list) and messages:
            body["input"] = messages
        else:
            body["input"] = prompt
        if system:
            body["instructions"] = system
        if spec.grounding:
            body["tools"] = [self._grounding_tool(spec.grounding, spec.grounding.effective_surface(spec.vendor))]
        else:
            body["tools"] = None
        for key, value in params.items():
            if key in (
                "temperature",
                "top_p",
                "max_output_tokens",
                "max_tokens",
                "seed",
                "reasoning_effort",
                "parallel_tool_calls",
            ):
                body[key] = value
        return body

    async def _openai_responses(
        self,
        spec: ModelSpec,
        prompt: str,
        system: str,
        messages: Any,
        params: dict[str, Any],
        traceparent: str,
        tracestate: str,
    ) -> Result:
        body = self._build_responses_body(spec, prompt, system, messages, params)
        raw = await self._dispatch(spec, self._responses_url(spec), body, traceparent, tracestate)
        return self._responses_to_result(spec, raw)

    def _responses_to_result(self, spec: ModelSpec, raw: dict[str, Any]) -> Result:
        usage = raw.get("usage") or {}
        details = usage.get("input_tokens_details") or {}
        cached = int(details.get("cached_tokens") or 0)
        input_tokens = int(usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        result = Result(
            model=_first_non_empty(raw.get("model", ""), spec.model),
            vendor=spec.vendor,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached,
            cost=spec.cost_micros(input_tokens - cached, output_tokens, cached),
            finish=FINISH_COMPLETE,
        )

        status = raw.get("status") or ""
        if status in ("", "completed"):
            pass
        elif status == "incomplete":
            reason = "the provider stopped before finishing"
            incomplete = raw.get("incomplete_details") or {}
            if incomplete.get("reason"):
                reason = incomplete["reason"]
            result.finish = FINISH_MAX_TOKENS
            result.finish_detail = (
                f"the response was truncated ({reason}); "
                "a grounded run needs enough max_output_tokens to finish its searches"
            )
        elif status == "failed":
            error = raw.get("error") or {}
            result.finish_detail = "the provider reported a failure: " + str(
                error.get("message") or "the provider reported a failure"
            )
            result.finish = FINISH_VENDOR_ERROR
        elif status == "cancelled":
            result.finish = FINISH_VENDOR_ERROR
            result.finish_detail = "the response was cancelled upstream"
        else:
            result.finish = FINISH_UNSPECIFIED
            result.finish_detail = f"the response finished in status {status}"

        for item in raw.get("output") or []:
            if not isinstance(item, dict):
                continue
            kind = item.get("type") or ""
            if kind == "web_search_call":
                result.search_queries.extend(self._search_queries_from(item))
            elif kind == "message":
                text_parts: list[str] = []
                citations: list[dict[str, Any]] = []
                for part in item.get("content") or []:
                    if not isinstance(part, dict):
                        continue
                    part_type = part.get("type") or ""
                    if part_type not in ("output_text", "text", "") and not part.get("text"):
                        continue
                    if isinstance(part.get("text"), str):
                        text_parts.append(part["text"])
                    citations.extend(self._citations_from_annotations(part.get("annotations") or []))
                text = "".join(text_parts)
                if not text and not citations:
                    continue
                result.messages.append({"text": text, "citations": citations})
                result.citations.extend(citations)

        if result.messages:
            result.text = result.messages[-1]["text"]
        if result.finish == FINISH_COMPLETE and not result.text:
            result.finish = FINISH_UNSPECIFIED
            result.finish_detail = "the provider returned no output_text content"
        return result

    @staticmethod
    def _search_queries_from(item: dict[str, Any]) -> list[str]:
        action = item.get("action") or {}
        kind = action.get("type") or ""
        if kind not in ("search", ""):
            return []
        queries: list[str] = []
        if action.get("query"):
            queries.append(action["query"])
        queries.extend(action.get("queries") or [])
        return queries

    # ------------------------------------------------------------------
    # Anthropic-shaped: messages
    # ------------------------------------------------------------------

    def _messages_url(self, spec: ModelSpec) -> str:
        return _resolve_endpoint(spec.base_url, spec.messages_path, "/v1/messages")

    def _build_messages_body(
        self,
        spec: ModelSpec,
        prompt: str,
        system: str,
        messages: Any,
        tools: Any,
        params: dict[str, Any],
        grounded: bool,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": spec.model,
            "system": system,
            "max_tokens": spec.max_output_tokens or 4096,
        }
        if isinstance(messages, list) and messages:
            body["messages"] = messages
        else:
            body["messages"] = [{"role": "user", "content": prompt}]
        if tools:
            body["tools"] = tools
        if grounded and spec.grounding:
            body["tools"] = list(body.get("tools") or []) + [self._grounding_tool(spec.grounding, "messages")]
        for key, value in params.items():
            if value is None:
                continue
            if key in ("temperature", "top_p"):
                number = _number(value)
                if number is not None:
                    body[key] = number
            elif key in ("max_tokens", "max_completion_tokens"):
                number = _number(value)
                if number is not None:
                    body["max_tokens"] = int(number)
            elif key in ("stop", "stop_sequences"):
                if isinstance(value, str):
                    body["stop_sequences"] = [value]
                elif isinstance(value, list):
                    body["stop_sequences"] = [item for item in value if isinstance(item, str)]
        return body

    async def _anthropic(
        self,
        spec: ModelSpec,
        prompt: str,
        system: str,
        messages: Any,
        tools: Any,
        params: dict[str, Any],
        grounded: bool,
        traceparent: str,
        tracestate: str,
    ) -> Result:
        body = self._build_messages_body(spec, prompt, system, messages, tools, params, grounded)
        headers = self._headers(spec, traceparent, tracestate)
        headers["anthropic-version"] = "2023-06-01"
        if spec.api_key:
            headers["x-api-key"] = spec.api_key
            headers.pop("Authorization", None)
        if self._post is not None:
            raw = await self._post(self._messages_url(spec), body, headers)
        else:
            raw = await self._transport.post_anthropic(
                spec, self._messages_url(spec), body, headers, traceparent, tracestate
            )
        return self._messages_to_result(spec, raw)

    def _messages_to_result(self, spec: ModelSpec, raw: dict[str, Any]) -> Result:
        usage = raw.get("usage") or {}
        cached = int(usage.get("cache_read_input_tokens") or 0)
        cache_writes = int(usage.get("cache_creation_input_tokens") or 0)
        input_tokens = int(usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        plain_input = max(0, input_tokens - cache_writes)
        result = Result(
            model=_first_non_empty(raw.get("model", ""), spec.model),
            vendor=spec.vendor,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached,
            cost=spec.cost_micros(plain_input - cached, output_tokens, cached, cache_writes),
            finish=FINISH_COMPLETE,
        )
        for block in raw.get("content") or []:
            if not isinstance(block, dict):
                continue
            kind = block.get("type") or ""
            if kind == "text":
                if isinstance(block.get("text"), str):
                    result.text += block["text"]
                for citation in block.get("citations") or []:
                    if not isinstance(citation, dict) or not citation.get("url"):
                        continue
                    result.citations.append(
                        {
                            "url": citation["url"],
                            "title": citation.get("title") or "",
                            "snippet": citation.get("cited_text") or "",
                        }
                    )
            elif kind == "server_tool_use":
                query = (block.get("input") or {}).get("query")
                if query:
                    result.search_queries.append(query)
        stop_reason = raw.get("stop_reason") or ""
        if stop_reason in ("end_turn", "stop_sequence", "tool_use", ""):
            result.finish = FINISH_COMPLETE
        elif stop_reason == "max_tokens":
            result.finish = FINISH_MAX_TOKENS
        else:
            result.finish = FINISH_COMPLETE
        return result

    # ------------------------------------------------------------------
    # Images
    # ------------------------------------------------------------------

    def _images_url(self, spec: ModelSpec) -> str:
        return _resolve_endpoint(spec.base_url, spec.images_path, "/images/generations")

    async def image(
        self,
        spec: ModelSpec,
        prompt: str,
        params: dict[str, Any],
        traceparent: str = "",
        tracestate: str = "",
    ) -> Result:
        if not prompt:
            raise ValueError("image generation requires a prompt")
        body: dict[str, Any] = {"model": spec.model, "prompt": prompt, "n": 1}
        for key in ("size", "quality", "style", "n", "response_format", "user"):
            if params.get(key) is not None:
                body[key] = params[key]
        raw = await self._dispatch(spec, self._images_url(spec), body, traceparent, tracestate)
        return await self._image_to_result(spec, raw, prompt)

    async def _image_to_result(self, spec: ModelSpec, raw: dict[str, Any], prompt: str) -> Result:
        usage = raw.get("usage") or {}
        result = Result(
            model=spec.model,
            vendor=spec.vendor,
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            cost=0,
            finish=FINISH_COMPLETE,
        )
        data = raw.get("data") or []
        if not data:
            result.finish = FINISH_UNSPECIFIED
            result.finish_detail = "upstream returned no image data"
            return result
        first = data[0] or {}
        result.revised_prompt = first.get("revised_prompt") or ""
        if first.get("b64_json"):
            try:
                result.image = base64.b64decode(first["b64_json"])
            except Exception as exc:
                raise ValueError(f"decode image b64_json: {exc}") from exc
            result.mime = sniff_image_mime(result.image)
        elif first.get("url"):
            result.image, result.mime = await self._fetch_image(first["url"])
        else:
            raise ValueError("image response carried neither b64_json nor url")
        return result

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def _embeddings_url(self, spec: ModelSpec) -> str:
        return _resolve_endpoint(spec.base_url, spec.embeddings_path, "/embeddings")

    async def embed(
        self,
        spec: ModelSpec,
        text: str,
        dimensions: int = 0,
        traceparent: str = "",
        tracestate: str = "",
    ) -> tuple[list[float], int, str]:
        body: dict[str, Any] = {"model": spec.model, "input": text}
        if dimensions > 0:
            body["dimensions"] = dimensions
        raw = await self._dispatch(spec, self._embeddings_url(spec), body, traceparent, tracestate)
        data = raw.get("data") or []
        if not data:
            raise ValueError("embeddings response carried no data")
        values = data[0].get("embedding") or []
        usage = raw.get("usage") or {}
        return (
            [float(v) for v in values],
            int(usage.get("prompt_tokens") or 0),
            _first_non_empty(raw.get("model", ""), spec.model),
        )


def _resolve_endpoint(base: str, path: str, default_path: str) -> str:
    if not path:
        path = default_path
    if path.startswith("http://") or path.startswith("https://"):
        return path
    if not base:
        return path
    return base.rstrip("/") + "/" + path.lstrip("/")


def _is_anthropic(spec: ModelSpec) -> bool:
    return spec.vendor == "anthropic" or spec.format == "messages"
