from __future__ import annotations

import base64
import json
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlsplit

import pytest
from blacksheep.contents import Content
from blacksheep.testing import TestClient

from app.config import GroundingSpec, ModelSpec
from app.http import create_app
from app.runtime import ProviderError, Runtime
from app.service import Gateway


class FakeDB:
    """In-memory stand-in for the Postgres budget/claim/ledger ports."""

    def __init__(self, budget: dict[str, Any] | None = None) -> None:
        self.budget_state = budget
        self.claims: dict[tuple[str, str, str], int] = {}
        self.ledger: list[dict[str, Any]] = []
        self.settle_calls: list[tuple[dict[str, Any], int]] = []

    async def ping(self) -> bool:
        return True

    async def budget(self, tenant_id: str) -> dict[str, Any] | None:
        return self.budget_state

    async def claim(self, gcid: str, key: str, action: str, invocation_id: str) -> bool:
        if not key:
            return True
        claim_key = (gcid, key, action or "unspecified")
        if claim_key in self.claims:
            return False
        self.claims[claim_key] = 1
        return True

    async def debit_and_enqueue(self, event: dict[str, Any], debit: int) -> None:
        self.settle_calls.append((event, debit))
        self.ledger.append(event)


class FakeProvider:
    """Routes provider HTTP by URL path and returns canned JSON, recording every
    request so tests can assert on the wire shape."""

    def __init__(self) -> None:
        self.routes: dict[str, tuple[int, dict[str, Any]]] = {}
        self.fail_with: dict[str, Exception] = {}
        self.requests: list[tuple[str, dict[str, Any], dict[str, str], str]] = []

    def route(self, path: str, body: dict[str, Any], status: int = 200) -> None:
        self.routes[path] = (status, body)

    async def post(self, url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        path = urlsplit(url).path
        self.requests.append((path, payload, headers, url))
        if url in self.fail_with:
            raise self.fail_with[url]
        status, body = self.routes.get(url, (200, {}))
        if status >= 400:
            raise ProviderError(status, url, json.dumps(body))
        return body

    @property
    def last(self) -> tuple[str, dict[str, Any], dict[str, str], str]:
        return self.requests[-1]


def make_spec(**kw: Any) -> ModelSpec:
    base: dict[str, Any] = {
        "id": "chatty",
        "vendor": "openai",
        "format": "chat_completions",
        "model": "chatty-upstream",
        "base_url": "https://api.example.test/v1",
        "capabilities": ["chat", "tools"],
        "context_window": 8000,
        "max_output_tokens": 512,
        "pricing": {
            "input_per_mtok_usd_micros": 1_000_000,
            "output_per_mtok_usd_micros": 2_000_000,
        },
    }
    base.update(kw)
    return ModelSpec(**base)


def make_models() -> dict[str, ModelSpec]:
    return {
        "chatty": make_spec(),
        "painter": make_spec(id="painter", model="painter-upstream", capabilities=["image"], kind="image"),
        "embedder": make_spec(id="embedder", model="embedder-upstream", capabilities=["embeddings"], kind="embedding"),
        "text-embedding-004": make_spec(
            id="text-embedding-004",
            model="text-embedding-004",
            capabilities=["embeddings"],
            kind="embedding",
        ),
        "textonly": make_spec(id="textonly", model="textonly-upstream", capabilities=["chat"]),
        "searcher": make_spec(
            id="searcher",
            model="searcher-upstream",
            capabilities=["chat", "web_search"],
            grounding=GroundingSpec(surface="responses", tool_type="web_search"),
        ),
        "chatsearcher": make_spec(
            id="chatsearcher",
            model="chatsearcher-upstream",
            capabilities=["chat", "web_search"],
            grounding=GroundingSpec(surface="chat_completions", tool_type="web_search"),
        ),
        "claude-searcher": make_spec(
            id="claude-searcher",
            vendor="anthropic",
            format="messages",
            model="claude-upstream",
            base_url="https://api.example.test",
            capabilities=["chat", "web_search"],
            grounding=GroundingSpec(surface="messages", tool_type="web_search_20250305"),
        ),
        "plain": make_spec(id="plain", model="plain-upstream", capabilities=["chat"]),
    }


def make_settings() -> SimpleNamespace:
    return SimpleNamespace(
        database_url="postgres://fake",
        default_tenant_id="00000000-0000-7000-8000-000000000001",
        default_gcid="00000000-0000-7000-8000-000000000002",
        default_agent_id="openai_compat",
        http_port=8080,
        grpc_port=9090,
        registry_path="",
        service_version="test",
        search_provider="exa",
    )


def make_gateway(models: dict[str, ModelSpec], db: FakeDB, provider: FakeProvider) -> Gateway:
    gateway = Gateway(make_settings(), models, db)
    gateway.runtime = Runtime(post=provider.post)
    return gateway


def make_app(models: dict[str, ModelSpec], db: FakeDB, provider: FakeProvider):
    gateway = make_gateway(models, db, provider)
    return create_app(gateway, db, models)


@pytest.fixture
def models() -> dict[str, ModelSpec]:
    return make_models()


class Client:
    """JSON-friendly wrapper over BlackSheep's TestClient."""

    def __init__(self, app: Any) -> None:
        self._client = TestClient(app)

    async def __aenter__(self) -> Client:
        simulator: Any = self._client._test_simulator
        self._app = simulator.app
        await self._app.start()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self._app.stop()
        return None

    async def get(self, path: str, headers: dict[str, str] | None = None, query: Any = None) -> Any:
        return await self._client.get(path, headers=headers, query=query)

    async def post(
        self,
        path: str,
        payload: Any = None,
        headers: dict[str, str] | None = None,
        content: bytes | None = None,
        query: Any = None,
    ) -> Any:
        body = None
        if content is not None:
            body = Content(b"application/json", content)
        elif payload is not None:
            body = Content(b"application/json", json.dumps(payload).encode())
        return await self._client.post(path, headers=headers, content=body, query=query)

    async def put(self, path: str, **kw: Any) -> Any:
        return await self._client.put(path, **kw)


def body(response: Any) -> dict[str, Any]:
    return json.loads(response.content.body)


@pytest.fixture
def db() -> FakeDB:
    return FakeDB()


def register_standard_routes(provider: FakeProvider) -> FakeProvider:
    provider.route(
        "https://api.example.test/v1/chat/completions",
        {
            "id": "chat-1",
            "model": "chatty-upstream",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hello from the stub"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        },
    )
    provider.route(
        "https://api.example.test/v1/responses",
        {
            "id": "resp-1",
            "model": "chatty-upstream",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "id": "m",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "hello from the stub"}],
                }
            ],
            "usage": {
                "input_tokens": 3,
                "output_tokens": 2,
                "total_tokens": 5,
                "input_tokens_details": {"cached_tokens": 0},
            },
        },
    )
    provider.route(
        "https://api.example.test/v1/images/generations",
        {
            "created": 1,
            "data": [{"b64_json": base64.b64encode(b"\x89PNG-bytes").decode()}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
        },
    )
    provider.route(
        "https://api.example.test/v1/embeddings",
        {
            "model": "embedder-upstream",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.5, 0.25]}],
            "usage": {"prompt_tokens": 4, "total_tokens": 4},
        },
    )
    return provider


@pytest.fixture
def provider() -> FakeProvider:
    return register_standard_routes(FakeProvider())


@pytest.fixture
def chat_completion() -> dict[str, Any]:
    return {
        "id": "chat-1",
        "model": "chatty-upstream",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hello from the stub"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


@pytest.fixture
def grounded_response() -> dict[str, Any]:
    return {
        "id": "resp-1",
        "model": "searcher-upstream",
        "status": "completed",
        "output": [
            {
                "type": "web_search_call",
                "id": "ws_1",
                "status": "completed",
                "action": {"type": "search", "query": "most recent Formula 1 race winner"},
            },
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Max Verstappen won.",
                        "annotations": [
                            {
                                "type": "url_citation",
                                "url": "https://f1.example/race-report",
                                "title": "Race report",
                                "start_index": 0,
                                "end_index": 3,
                            },
                            {"type": "url_citation", "url": "https://f1.example/standings"},
                        ],
                    }
                ],
            },
        ],
        "usage": {
            "input_tokens": 25,
            "output_tokens": 12,
            "total_tokens": 37,
            "input_tokens_details": {"cached_tokens": 0},
        },
    }
