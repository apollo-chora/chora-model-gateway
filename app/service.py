from __future__ import annotations

import time
import uuid
from typing import Any

from .runtime import Result, Runtime


class GatewayError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


class Gateway:
    def __init__(self, settings: Any, models: dict[str, Any], db: Any) -> None:
        self.settings = settings
        self.models = models
        self.db = db
        self.runtime = Runtime(settings.search_provider)

    def model(self, name: str | None) -> Any:
        spec = self.models.get((name or "").lower())
        if not spec:
            raise GatewayError(404, "model_not_found", f'model "{name}" is not in the registry')
        return spec

    @staticmethod
    def cost(spec: Any, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> int:
        pricing = spec.pricing
        return (
            input_tokens * int(pricing.get("input_per_mtok_usd_micros", 0))
            + output_tokens * int(pricing.get("output_per_mtok_usd_micros", 0))
            + cached_tokens * int(pricing.get("cached_per_mtok_usd_micros", 0))
        ) // 1_000_000

    async def invoke(self, request: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        invocation_id = request.get("invocation_id") or str(uuid.uuid4())
        spec = self.model(request.get("model") or request.get("logical_model_id"))
        modality = request.get("modality") or "TEXT"
        required = "image" if modality == "IMAGE" else ("web_search" if modality == "GROUNDED" else "chat")
        if required not in spec.capabilities:
            raise GatewayError(400, "grounding_unavailable" if required == "web_search" else "invalid_model_capability", f'model "{spec.id}" does not advertise the "{required}" capability')
        tenant = request.get("tenant_id") or self.settings.default_tenant_id
        gcid = request.get("gcid") or self.settings.default_gcid
        agent = request.get("agent_id") or self.settings.default_agent_id
        budget = await self.db.budget(tenant)

        if budget and budget["spent_usd_micros"] >= budget["budget_usd_micros"]:
            if budget["policy"] == "block":
                raise GatewayError(402, "budget_exhausted", "tenant LLM budget exhausted")
            if budget["policy"] == "downgrade" and budget["downgrade_to_logical_model_id"]:
                spec = self.model(budget["downgrade_to_logical_model_id"])

        claimed = await self.db.claim(
            gcid, request.get("dispatch_idempotency_key", ""), request.get("action_code", agent)
        )
        chain = [spec] + [
            self.model(model_id)
            for model_id in request.get("fallback_ids", [])
            if model_id.lower() in self.models
        ]
        result: Result | None = None
        last_error: Exception | None = None
        tried: list[str] = []

        for candidate in chain:
            tried.append(f"{candidate.format}:{candidate.model}")
            try:
                if request.get("modality") == "IMAGE":
                    result = await self.runtime.image(
                        candidate, request.get("prompt", ""), request.get("params", {})
                    )
                else:
                    result = await self.runtime.generate(
                        candidate,
                        request.get("prompt", ""),
                        request.get("system", ""),
                        request.get("messages"),
                        request.get("tools"),
                        request.get("params", {}),
                        request.get("modality") == "GROUNDED",
                    )
                spec = candidate
                break
            except Exception as exc:
                last_error = exc

        if result is None:
            raise GatewayError(502, "vendor_error", f"provider chain exhausted: {last_error}")

        charge = self.cost(spec, result.input_tokens, result.output_tokens, result.cached_tokens) if claimed else 0
        event = {
            "invocation_id": invocation_id, "tenant_id": tenant, "gcid": gcid,
            "model_id": result.model or spec.model, "vendor": result.vendor,
            "input_tokens": result.input_tokens, "output_tokens": result.output_tokens,
            "cached_tokens": result.cached_tokens, "agent_role": request.get("action_code") or agent,
            "surface": request.get("surface"), "modality": request.get("modality") or "TEXT",
            "fallback_chain": tried, "debit_deduped": not claimed,
            "traceparent": request.get("traceparent"), "tracestate": request.get("tracestate"),
            "gateway_version": self.settings.service_version,
        }
        await self.db.settle(event, charge)
        return {
            "id": invocation_id, "result": result, "model": result.model or spec.model,
            "vendor": result.vendor,
            "usage": {"input": result.input_tokens, "output": result.output_tokens,
                      "cached": result.cached_tokens, "cost": charge},
            "fallback_chain": tried, "latency_ms": int((time.monotonic() - started) * 1000),
        }

    async def embed(self, request: dict[str, Any]) -> dict[str, Any]:
        invocation_id = request.get("invocation_id") or str(uuid.uuid4())
        spec = self.model(request.get("model") or request.get("logical_model_id") or "text-embedding-004")
        if "embeddings" not in spec.capabilities and spec.kind != "embedding":
            raise GatewayError(500, "gateway_misconfigured", f'embed: model "{spec.id}" does not advertise the "embeddings" capability')
        tenant = request.get("tenant_id") or self.settings.default_tenant_id
        gcid = request.get("gcid") or self.settings.default_gcid
        agent = request.get("agent_id") or self.settings.default_agent_id
        values, input_tokens, model = await self.runtime.embed(
            spec, request["text"], int(request.get("dimensions", 0))
        )
        await self.db.settle(
            {
                "invocation_id": invocation_id, "tenant_id": tenant, "gcid": gcid,
                "model_id": model, "vendor": "embeddings", "input_tokens": input_tokens,
                "output_tokens": 0, "cached_tokens": 0, "agent_role": agent, "modality": "TEXT",
                "fallback_chain": [f"embeddings:{model}"], "traceparent": request.get("traceparent"),
                "tracestate": request.get("tracestate"), "gateway_version": self.settings.service_version,
            },
            0,
        )
        return {"id": invocation_id, "values": values, "model": model, "vendor": "embeddings", "input_tokens": input_tokens}
