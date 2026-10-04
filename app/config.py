from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ModelSpec:
    id: str
    kind: str = "text"
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
    support_grounding: bool = False

    @property
    def api_key(self) -> str:
        return os.getenv(self.api_key_env, "") if self.api_key_env else ""


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
        "1", "true", "yes", "on"
    }
    capabilities = [kind] if kind != "text" else ["chat", "tools"] + (["web_search"] if grounding else [])
    return ModelSpec(
        id=model,
        kind=kind,
        format=os.getenv(f"{prefix}_LLM_FORMAT", default_format),
        model=model,
        base_url=os.getenv(f"{prefix}_LLM_BASE_URL", ""),
        api_key_env=f"{prefix}_LLM_API_KEY",
        capabilities=capabilities,
        support_grounding=grounding,
    )


def load_models(path: str) -> dict[str, ModelSpec]:
    specs: list[ModelSpec] = []
    registry = Path(path)
    if registry.exists():
        rows = (yaml.safe_load(registry.read_text()) or {}).get("models", [])
        for row in rows:
            caps = row.get("capabilities", ["chat"])
            provider = row.get("provider", "")
            fmt = row.get("format") or ("messages" if provider == "anthropic" else "chat_completions")
            if "image" in caps and "chat" not in caps:
                kind = "image"
            elif "embeddings" in caps and "chat" not in caps:
                kind = "embedding"
            else:
                kind = "text"
            specs.append(
                ModelSpec(
                    id=row["id"],
                    kind=kind,
                    format=fmt,
                    model=row.get("upstream_model", row["id"]),
                    base_url=row.get("base_url", ""),
                    api_key_env=row.get("api_key_env", ""),
                    capabilities=caps,
                    fallback_ids=row.get("fallback_ids", []),
                    aliases=row.get("aliases", []),
                    context_window=row.get("context_window", 0),
                    max_output_tokens=row.get("max_output_tokens", 0),
                    pricing=row.get("pricing", {}),
                    extra_headers=row.get("extra_headers", {}),
                    support_grounding="web_search" in caps,
                )
            )

    for spec in (
        _env("TEXT", "text", "responses"),
        _env("IMAGE", "image", "images"),
        _env("EMBEDDING", "embedding", "embeddings"),
    ):
        if spec:
            specs.insert(0, spec)

    models: dict[str, ModelSpec] = {}
    for spec in specs:
        models[spec.id.lower()] = spec
        for alias in spec.aliases:
            models[alias.lower()] = spec
    if not models:
        raise RuntimeError("no models configured")
    return models
