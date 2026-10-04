from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

KNOWN_CAPABILITIES = ("chat", "tools", "vision", "image", "embeddings", "web_search")

DEFAULT_EMBEDDING_MODEL_ID = "text-embedding-004"
DEFAULT_EMBEDDING_DIMENSIONS = 768


class ConfigError(Exception):
    """A registry/deployment misconfiguration. Mirrors domain.ConfigError."""


@dataclass(slots=True)
class GroundingSpec:
    """Hosted web search configuration for one registry entry."""

    surface: str = ""  # "" | responses | chat_completions | messages
    responses_path: str = ""
    tool_type: str = ""
    tool_name: str = ""
    max_uses: int = 0
    extra_tool_fields: dict[str, Any] = field(default_factory=dict)

    def effective_surface(self, vendor: str) -> str:
        if self.surface:
            return self.surface
        return "messages" if vendor == "anthropic" else "responses"

    def effective_tool_type(self, surface: str) -> str:
        if self.tool_type:
            return self.tool_type
        return {
            "messages": "web_search_20250305",
            "chat_completions": "web_search_preview",
        }.get(surface, "web_search")

    def effective_tool_name(self) -> str:
        return self.tool_name or "web_search"


@dataclass(slots=True)
class ModelSpec:
    id: str
    kind: str = "text"
    vendor: str = "openai"
    format: str = "chat_completions"
    model: str = ""
    base_url: str = ""
    api_key_env: str = ""
    capabilities: list[str] = field(default_factory=lambda: ["chat"])
    fallback_ids: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    context_window: int = 0
    max_output_tokens: int = 0
    pricing: dict[str, Any] = field(default_factory=dict)
    extra_headers: dict[str, str] = field(default_factory=dict)
    grounding: GroundingSpec | None = None

    @property
    def api_key(self) -> str:
        return os.getenv(self.api_key_env, "").strip() if self.api_key_env else ""

    def supports(self, capability: str) -> bool:
        if not self.capabilities:
            return capability == "chat"
        return capability in self.capabilities

    def cost_micros(self, billable_input: int, output: int, cached: int = 0, cache_writes: int = 0) -> int:
        """USD micros, term-wise floored — the same arithmetic the ledger stores."""
        pricing = self.pricing
        micros = (billable_input * int(pricing.get("input_per_mtok_usd_micros", 0))) // 1_000_000
        micros += (cached * int(pricing.get("cached_per_mtok_usd_micros", 0))) // 1_000_000
        micros += (cache_writes * int(pricing.get("cache_write_per_mtok_usd_micros", 0))) // 1_000_000
        micros += (output * int(pricing.get("output_per_mtok_usd_micros", 0))) // 1_000_000
        return micros


@dataclass(slots=True)
class Settings:
    database_url: str
    default_tenant_id: str
    default_gcid: str
    default_agent_id: str
    http_port: int
    grpc_port: int
    registry_path: str
    service_version: str
    search_provider: str

    @classmethod
    def load(cls) -> Settings:
        tenant = os.environ["CHORA_DEFAULT_TENANT_ID"]
        return cls(
            database_url=os.environ["CHORA_DATABASE_URL"],
            default_tenant_id=tenant,
            default_gcid=os.getenv("CHORA_DEFAULT_GCID", tenant),
            default_agent_id=os.getenv("CHORA_DEFAULT_AGENT_ID", "openai_compat"),
            http_port=int(os.getenv("CHORA_HTTP_PORT", "8080")),
            grpc_port=int(os.getenv("CHORA_GRPC_PORT", "9090")),
            registry_path=os.getenv("CHORA_MODEL_REGISTRY", "config/models.yaml"),
            service_version=os.getenv("SERVICE_VERSION", "chora-model-gateway:local"),
            search_provider=os.getenv("SEARCH_PROVIDER", "exa"),
        )


def _env(prefix: str, kind: str, default_format: str) -> ModelSpec | None:
    model = os.getenv(f"{prefix}_LLM_MODEL", "").strip()
    if not model:
        return None
    grounding = os.getenv(f"{prefix}_LLM_SUPPORT_GROUNDING", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    capabilities = {"text": ["chat", "tools"], "image": ["image"], "embedding": ["embeddings"]}[kind]
    if kind == "text" and grounding:
        capabilities = [*capabilities, "web_search"]
    spec = ModelSpec(
        id=model,
        kind=kind,
        vendor="anthropic" if os.getenv(f"{prefix}_LLM_FORMAT", default_format) == "messages" else "openai",
        format=os.getenv(f"{prefix}_LLM_FORMAT", default_format),
        model=model,
        base_url=os.getenv(f"{prefix}_LLM_BASE_URL", ""),
        api_key_env=f"{prefix}_LLM_API_KEY",
        capabilities=capabilities,
    )
    if grounding:
        spec.grounding = GroundingSpec()
    return spec


def _parse_grounding(row: dict[str, Any]) -> GroundingSpec:
    block = row.get("grounding")
    if block is None:
        return None  # type: ignore[return-value]
    surface = str(block.get("surface", "")).lower()
    if surface not in ("", "responses", "chat_completions", "messages"):
        raise ConfigError(
            f"grounding surface {block.get('surface')!r} is not one of: responses, chat_completions, messages"
        )
    max_uses = int(block.get("max_uses", 0) or 0)
    if max_uses < 0:
        raise ConfigError("grounding max_uses cannot be negative")
    return GroundingSpec(
        surface=surface,
        responses_path=str(block.get("responses_path", "")),
        tool_type=str(block.get("tool_type", "")),
        tool_name=str(block.get("tool_name", "")),
        max_uses=max_uses,
        extra_tool_fields=dict(block.get("extra_tool_fields", {})),
    )


def load_models(path: str) -> dict[str, ModelSpec]:
    """Parse the registry YAML. Boot-time validation mirrors the old gateway:
    a typo must fail at boot, not on the first request that touches it."""
    registry = Path(path)
    specs: list[ModelSpec] = []
    if registry.exists():
        rows = (yaml.safe_load(registry.read_text()) or {}).get("models", [])
        for row in rows:
            caps = list(row.get("capabilities") or ["chat"])
            for cap in caps:
                if cap not in KNOWN_CAPABILITIES:
                    raise ConfigError(f"capability {cap!r} is not one of {', '.join(KNOWN_CAPABILITIES)}")
            provider = str(row.get("provider", "")).lower()
            if provider not in ("openai", "anthropic"):
                raise ConfigError(f"provider {provider!r} is not one of: openai, anthropic")
            if "image" in caps and "chat" not in caps:
                kind = "image"
            elif "embeddings" in caps and "chat" not in caps:
                kind = "embedding"
            else:
                kind = "text"
            grounding = _parse_grounding(row)
            if grounding is not None and "web_search" not in caps:
                raise ConfigError("a grounding block is configured but `capabilities` does not include 'web_search'")
            context_window = int(row.get("context_window", 0) or 0)
            max_output = int(row.get("max_output_tokens", 0) or 0)
            if context_window < 0:
                raise ConfigError("context_window cannot be negative")
            if max_output < 0:
                raise ConfigError("max_output_tokens cannot be negative")
            if max_output > 0 and context_window > 0 and max_output > context_window:
                raise ConfigError(f"max_output_tokens ({max_output}) exceeds context_window ({context_window})")
            specs.append(
                ModelSpec(
                    id=str(row["id"]),
                    kind=kind,
                    vendor=provider,
                    format="messages" if provider == "anthropic" else "chat_completions",
                    model=str(row.get("upstream_model") or row["id"]),
                    base_url=str(row.get("base_url", "")),
                    api_key_env=str(row.get("api_key_env", "")),
                    capabilities=caps,
                    fallback_ids=[str(fb) for fb in row.get("fallback_ids", [])],
                    aliases=[str(a) for a in row.get("aliases", [])],
                    context_window=context_window,
                    max_output_tokens=max_output,
                    pricing=dict(row.get("pricing", {})),
                    extra_headers=dict(row.get("extra_headers", {})),
                    grounding=grounding,
                )
            )

    # The role-based environment configuration is the primary deployment path;
    # it wins over a registry row that happens to use the same id.
    for spec in (
        _env("TEXT", "text", "responses"),
        _env("IMAGE", "image", "images"),
        _env("EMBEDDING", "embedding", "embeddings"),
    ):
        if spec:
            specs.append(spec)

    models: dict[str, ModelSpec] = {}
    upstreams: dict[str, str] = {}
    for spec in specs:
        for name in [spec.id, *spec.aliases]:
            key = name.strip().lower()
            if not key:
                continue
            existing = upstreams.get(key)
            if existing and existing != spec.model:
                raise ConfigError(f"model name {name!r} is declared twice with different upstreams ({existing!r})")
            upstreams[key] = spec.model
            models[key] = spec

    for spec in specs:
        for fb in spec.fallback_ids:
            if fb.strip().lower() not in models:
                raise ConfigError(f"model {spec.id!r} lists fallback {fb!r}, which is not in the registry")

    if not models:
        raise ConfigError("no models configured")
    return models
