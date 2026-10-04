from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from .config import DEFAULT_EMBEDDING_DIMENSIONS, DEFAULT_EMBEDDING_MODEL_ID, ModelSpec
from .runtime import Result, Runtime

FINISH_BUDGET_BLOCK = "budget_block"


class GatewayError(Exception):
    """A terminal invoke failure. Carries the transport status, the OpenAI
    error type, the machine code, and optionally the provider's own HTTP
    status so the facade can relay it instead of flattening it.

    `kind` mirrors the old gateway's error taxonomy for gRPC status mapping:
    "config" (FailedPrecondition), "invoke" (Unavailable), "bare" (InvalidArgument).
    """

    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        err_type: str = "upstream_error",
        upstream_status: int | None = None,
        inner: Exception | None = None,
        kind: str = "invoke",
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.err_type = err_type
        self.upstream_status = upstream_status
        self.inner = inner
        self.kind = kind

    def __str__(self) -> str:
        return self.message


class BudgetBlock(Exception):
    """A budget refusal. The HTTP facade maps it to 402; gRPC answers with a
    normal response carrying finish_reason BUDGET_BLOCK.

    Carries the resolved invocation id so the gRPC response echoes the id the
    gateway generated (matching the Go domain, which returns it on a block)."""

    def __init__(self, detail: str, invocation_id: str = "") -> None:
        super().__init__(detail)
        self.detail = detail
        self.invocation_id = invocation_id


@dataclass(slots=True)
class Invocation:
    id: str
    text: str = ""
    model: str = ""
    vendor: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cost: int = 0
    fallback_chain: list[str] = field(default_factory=list)
    latency_ms: int = 0
    finish: str = "complete"
    finish_detail: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    image: bytes = b""
    image_mime: str = ""
    revised_prompt: str = ""
    grounding_surface: str = ""


def _new_invocation_id() -> str:
    generator = getattr(uuid, "uuid7", None)
    if generator is not None:
        return str(generator())
    return str(uuid.uuid4())


def _is_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
    except (ValueError, AttributeError, TypeError):
        return False
    return True


class Gateway:
    def __init__(self, settings: Any, models: dict[str, ModelSpec], db: Any) -> None:
        self.settings = settings
        self.models = models
        self.db = db
        self.runtime = Runtime(settings.search_provider)

    def model(self, name: str | None) -> ModelSpec:
        spec = self.models.get((name or "").strip().lower())
        if not spec:
            raise GatewayError(
                502,
                "vendor_error",
                f'policy: model "{name}" is not in the registry',
                err_type="upstream_error",
            )
        return spec

    def _resolve(self, model: str | None, caller_fallbacks: list[str]) -> tuple[ModelSpec, list[ModelSpec]]:
        """Resolve the primary target and the fallback chain. The caller's own
        chain comes first; the registry's declared chain follows. An unknown
        name fails the whole resolve rather than being silently dropped."""
        primary = self.model(model)
        chain: list[ModelSpec] = []
        seen = {primary.id}
        for name in caller_fallbacks:
            candidate = self.model(name)
            if candidate.id in seen:
                continue
            seen.add(candidate.id)
            chain.append(candidate)
        for name in primary.fallback_ids:
            registry_candidate = self.models.get(name.strip().lower())
            if registry_candidate is None:
                registry_candidate = self.model(name)
            if registry_candidate.id in seen:
                continue
            seen.add(registry_candidate.id)
            chain.append(registry_candidate)
        return primary, chain

    @staticmethod
    def _apply_output_ceiling(params: dict[str, Any], ceiling: int) -> dict[str, Any]:
        if ceiling <= 0 or not params:
            return params
        requested = params.get("max_tokens")
        if requested is None:
            return params
        try:
            value = float(requested)
        except (TypeError, ValueError):
            return params
        if value <= ceiling:
            return params
        out = dict(params)
        out["max_tokens"] = ceiling
        return out

    async def invoke(self, request: dict[str, Any]) -> Invocation:
        started = time.monotonic()

        tenant = (request.get("tenant_id") or "").strip()
        gcid = (request.get("gcid") or "").strip()
        agent = (request.get("agent_id") or "").strip()
        model_id = (request.get("model") or request.get("logical_model_id") or "").strip()
        modality = request.get("modality") or "TEXT"
        prompt = request.get("prompt") or ""
        contents = request.get("contents_json") or request.get("messages")

        # Step 1 — validate the envelope. A missing tenant or model is a caller
        # bug, and discovering it at the vendor would mean a billable request
        # with no attribution.
        if not tenant:
            raise GatewayError(502, "vendor_error", "tenant_id required", inner=ValueError("tenant_id required"))
        if not gcid:
            raise GatewayError(502, "vendor_error", "gcid required", inner=ValueError("gcid required"))
        if not agent:
            raise GatewayError(502, "vendor_error", "agent_id required", inner=ValueError("agent_id required"))
        if not model_id:
            raise GatewayError(
                502, "vendor_error", "logical_model_id required", inner=ValueError("logical_model_id required")
            )
        if modality not in ("", "TEXT", "IMAGE", "GROUNDED"):
            raise GatewayError(
                502,
                "vendor_error",
                f'response_modality "{modality}" is not one of TEXT, IMAGE, GROUNDED',
                inner=ValueError("bad modality"),
            )
        if not prompt and not contents:
            raise GatewayError(
                502,
                "vendor_error",
                "prompt or contents_json required (an empty request is still billed by most providers)",
                inner=ValueError("empty request"),
            )

        invocation_id = request.get("invocation_id") or _new_invocation_id()

        # Step 3 — resolve the registry entry and the fallback chain.
        spec, chain = self._resolve(model_id, list(request.get("fallback_ids") or []))

        # Step 4 — keyed idempotency. A redelivered dispatch with the same key
        # must bill once, so the claim is taken BEFORE any spend.
        deduped = False
        dispatch_key = request.get("dispatch_idempotency_key") or ""
        if dispatch_key:
            claimed = await self.db.claim(gcid, dispatch_key, request.get("action_code") or agent)
            deduped = not claimed

        # Step 5 — load the tenant budget.
        budget = await self.db.budget(tenant)

        # Step 6 — decide the budget action. An unrecognised policy fails safe:
        # refuse rather than allow.
        policy = (budget or {}).get("policy") or ""
        exhausted = bool(budget) and (budget["spent_usd_micros"] >= budget["budget_usd_micros"])
        if exhausted:
            if policy == "downgrade":
                downgrade_id = budget.get("downgrade_to_logical_model_id") or ""
                try:
                    spec, chain = self._resolve(downgrade_id, [])
                except GatewayError as exc:
                    raise GatewayError(
                        500,
                        "gateway_misconfigured",
                        f"budget downgrade target is not in the model registry: {downgrade_id}",
                        err_type="server_error",
                        inner=exc,
                        kind="config",
                    ) from exc
            elif policy != "alert":
                raise BudgetBlock("tenant LLM budget exhausted; policy=block", invocation_id=invocation_id)

        # Step 7 + 8 — resolve the credential and dispatch, walking the chain.
        action_code = request.get("action_code") or agent
        if modality == "IMAGE":
            action_code = request.get("action_code") or f"{self.settings.default_agent_id}_image"
        params = self._apply_output_ceiling(dict(request.get("params") or {}), spec.max_output_tokens)
        grounded = modality == "GROUNDED"
        traceparent = request.get("traceparent") or ""
        tracestate = request.get("tracestate") or ""

        result: Result | None = None
        used: ModelSpec | None = None
        tried: list[str] = []
        last_error: Exception | None = None
        upstream_status: int | None = None

        for candidate in [spec, *chain]:
            tried.append(f"{candidate.vendor}:{candidate.model}")
            if candidate.api_key_env and not candidate.api_key:
                raise GatewayError(
                    500,
                    "gateway_misconfigured",
                    f'credential reference "{candidate.api_key_env}" is set but resolves to an empty value',
                    err_type="server_error",
                    kind="config",
                )
            required = "image" if modality == "IMAGE" else ("web_search" if grounded else "chat")
            if not candidate.supports(required):
                raise GatewayError(
                    500,
                    "gateway_misconfigured",
                    f'model "{candidate.id}" does not advertise the "{required}" capability '
                    f"(it has: {', '.join(candidate.capabilities)})",
                    err_type="server_error",
                    kind="config",
                )
            if grounded and candidate.grounding is None:
                raise GatewayError(
                    500,
                    "gateway_misconfigured",
                    f'model "{candidate.id}" advertises "web_search" but its registry entry '
                    "configures no grounding endpoint; add a `grounding:` block",
                    err_type="server_error",
                    kind="config",
                )
            try:
                if modality == "IMAGE":
                    result = await self.runtime.image(
                        candidate, prompt, request.get("params") or {}, traceparent, tracestate
                    )
                else:
                    result = await self.runtime.generate(
                        candidate,
                        prompt,
                        request.get("system") or "",
                        request.get("messages"),
                        request.get("tools"),
                        params,
                        grounded,
                        traceparent,
                        tracestate,
                    )
                used = candidate
                break
            except Exception as exc:  # noqa: BLE001 — the chain walks on any failure
                last_error = exc
                status_getter = getattr(exc, "upstream_status", None)
                if callable(status_getter):
                    upstream_status = status_getter()

        if result is None or used is None:
            detail = f"all targets exhausted: {last_error}" if last_error else "all targets exhausted"
            raise GatewayError(
                502,
                "vendor_error",
                detail,
                inner=last_error,
                upstream_status=upstream_status,
            )

        # Step 9 — settle the budget + the ledger atomically. A deduped
        # dispatch still ledgers the real cost; only the debit is suppressed.
        charge = result.cost
        debit = 0 if deduped else charge
        event = {
            "invocation_id": invocation_id,
            "tenant_id": tenant,
            "gcid": gcid,
            "model_id": result.model or used.model,
            "vendor": result.vendor or used.vendor,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "cached_tokens": result.cached_tokens,
            "agent_role": action_code,
            "surface": request.get("surface"),
            "modality": modality or "TEXT",
            "fallback_chain": tried,
            "debit_deduped": deduped,
            "traceparent": traceparent,
            "tracestate": tracestate,
            "gateway_version": self.settings.service_version,
            "cost_usd_micros": charge,
        }
        try:
            await self.db.settle(event, debit)
        except Exception as exc:
            raise GatewayError(502, "vendor_error", f"settle failed: {exc}", inner=exc) from exc

        return Invocation(
            id=invocation_id,
            text=result.text,
            model=result.model or used.model,
            vendor=result.vendor or used.vendor,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cached_tokens=result.cached_tokens,
            cost=charge,
            fallback_chain=tried,
            latency_ms=int((time.monotonic() - started) * 1000),
            finish=result.finish,
            finish_detail=result.finish_detail,
            tool_calls=result.tool_calls,
            citations=result.citations,
            search_queries=result.search_queries,
            messages=result.messages,
            image=result.image,
            image_mime=result.mime,
            revised_prompt=result.revised_prompt,
            grounding_surface=(used.grounding.effective_surface(used.vendor) if grounded and used.grounding else ""),
        )

    async def embed(self, request: dict[str, Any]) -> dict[str, Any]:
        tenant = (request.get("tenant_id") or "").strip()
        gcid = (request.get("gcid") or "").strip()
        agent = (request.get("agent_id") or "").strip()
        text = request.get("text") or ""
        if not tenant:
            raise GatewayError(
                400, "invalid_request", "embed: tenant_id required", err_type="invalid_request_error", kind="bare"
            )
        if not gcid:
            raise GatewayError(
                400, "invalid_request", "embed: gcid required", err_type="invalid_request_error", kind="bare"
            )
        if not agent:
            raise GatewayError(
                400, "invalid_request", "embed: agent_id required", err_type="invalid_request_error", kind="bare"
            )
        if not text:
            raise GatewayError(
                400, "invalid_request", "embed: text required", err_type="invalid_request_error", kind="bare"
            )

        invocation_id = request.get("invocation_id") or _new_invocation_id()
        model_id = (request.get("model") or request.get("logical_model_id") or "").strip()
        if not model_id:
            model_id = DEFAULT_EMBEDDING_MODEL_ID
        spec = self.model(model_id)
        if not spec.supports("embeddings"):
            raise GatewayError(
                500,
                "gateway_misconfigured",
                f'embed: model "{spec.id}" does not advertise the "embeddings" capability',
                err_type="server_error",
                kind="config",
            )
        dimensions = int(request.get("dimensions") or 0)
        if dimensions <= 0:
            dimensions = DEFAULT_EMBEDDING_DIMENSIONS
        if spec.api_key_env and not spec.api_key:
            raise GatewayError(
                500,
                "gateway_misconfigured",
                f'credential reference "{spec.api_key_env}" is set but resolves to an empty value',
                err_type="server_error",
                kind="config",
            )
        try:
            values, input_tokens, model = await self.runtime.embed(
                spec, text, dimensions, request.get("traceparent") or "", request.get("tracestate") or ""
            )
        except Exception as exc:
            status_getter = getattr(exc, "upstream_status", None)
            raise GatewayError(
                502,
                "vendor_error",
                f"embed: vendor dispatch: {exc}",
                inner=exc,
                upstream_status=status_getter() if callable(status_getter) else None,
                kind="bare",
            ) from exc

        event = {
            "invocation_id": invocation_id,
            "tenant_id": tenant,
            "gcid": gcid,
            "model_id": model,
            "vendor": spec.vendor,
            "input_tokens": input_tokens,
            "output_tokens": 0,
            "cached_tokens": 0,
            "agent_role": agent,
            "modality": "TEXT",
            "fallback_chain": [f"{spec.vendor}:{spec.model}"],
            "traceparent": request.get("traceparent") or "",
            "tracestate": request.get("tracestate") or "",
            "gateway_version": self.settings.service_version,
            "cost_usd_micros": 0,
        }
        await self.db.settle(event, 0)
        return {
            "id": invocation_id,
            "values": values,
            "model": model,
            "vendor": spec.vendor,
            "input_tokens": input_tokens,
        }
