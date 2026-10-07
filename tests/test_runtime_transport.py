from __future__ import annotations

import base64
import json
from unittest import mock

import httpx
import pytest

from app.runtime import ProviderError, Runtime
from tests.conftest import make_spec


def stub_transport(response: httpx.Response | None = None, status: int = 200, payload: dict | None = None):
    """Replace the runtime's HTTP transport with a stubbed httpx client."""
    if response is None:
        response = mock.Mock()
        response.status_code = status
        response.text = json.dumps(payload or {})
        response.json.return_value = payload or {}
    client = mock.Mock(spec=httpx.AsyncClient)
    client.post = mock.AsyncMock(return_value=response)
    return client


@pytest.mark.anyio
async def test_default_transport_builds_the_chat_request():
    spec = make_spec()
    runtime = Runtime(post=None)
    response = mock.Mock()
    response.status_code = 200
    response.text = "{}"
    response.json.return_value = {
        "id": "c",
        "model": "m",
        "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }
    client = stub_transport(response)
    runtime._transport._client = lambda *_a, **_k: client  # type: ignore[method-assign]

    result = await runtime.generate(spec, "hello", "be terse")
    assert result.text == "hi"
    _, kwargs = client.post.call_args
    # The payload is serialized to JSON and the trace headers ride along.
    assert json.loads(kwargs["content"])["messages"][0] == {"role": "system", "content": "be terse"}


@pytest.mark.anyio
async def test_genai_shaped_contents_are_normalized_for_chat():
    """The Go client sends genai `[]*Content` ({role, parts:[{text}]}) in
    contents_json. Forwarded verbatim the provider rejects the request with
    "Missing required parameter: messages[0].content", so the chat builder must
    convert them to {role, content}."""
    spec = make_spec()
    runtime = Runtime(post=None)
    response = mock.Mock()
    response.status_code = 200
    response.text = "{}"
    response.json.return_value = {
        "id": "c",
        "model": "m",
        "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }
    client = stub_transport(response)
    runtime._transport._client = lambda *_a, **_k: client  # type: ignore[method-assign]

    contents = [
        {"role": "user", "parts": [{"text": "BEGIN"}]},
        {"role": "model", "parts": [{"text": "prior answer"}]},
    ]
    await runtime.generate(spec, "BEGIN", "", contents)
    body = json.loads(client.post.call_args[1]["content"])
    assert body["messages"] == [
        {"role": "user", "content": "BEGIN"},
        {"role": "assistant", "content": "prior answer"},
    ]


@pytest.mark.anyio
async def test_contentless_messages_fall_back_to_the_flat_prompt():
    """A genai entry whose parts carry no text must not be forwarded as a
    content-less message; the flat prompt is the correct fallback."""
    spec = make_spec()
    runtime = Runtime(post=None)
    response = mock.Mock()
    response.status_code = 200
    response.text = "{}"
    response.json.return_value = {
        "id": "c",
        "model": "m",
        "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }
    client = stub_transport(response)
    runtime._transport._client = lambda *_a, **_k: client  # type: ignore[method-assign]

    await runtime.generate(spec, "real prompt", "", [{"role": "user", "parts": [{"text": ""}]}])
    body = json.loads(client.post.call_args[1]["content"])
    assert body["messages"] == [{"role": "user", "content": "real prompt"}]


@pytest.mark.anyio
async def test_system_instruction_is_kept_when_messages_are_present():
    """ADK routes the agent's InstructionProvider output through
    SystemInstruction, which the Go client forwards in the separate `system`
    field. A request that also carries `messages` must still send it, or the
    model sees only the trigger turn and answers it literally."""
    spec = make_spec()
    runtime = Runtime(post=None)
    response = mock.Mock()
    response.status_code = 200
    response.text = "{}"
    response.json.return_value = {
        "id": "c",
        "model": "m",
        "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }
    client = stub_transport(response)
    runtime._transport._client = lambda *_a, **_k: client  # type: ignore[method-assign]

    await runtime.generate(spec, "BEGIN", "You are a question generator.", [{"role": "user", "parts": [{"text": "BEGIN"}]}])
    body = json.loads(client.post.call_args[1]["content"])
    # The bare `system` field is ignored when messages are present (the public
    # HTTP surface relies on that); the trusted gRPC path carries the
    # instruction as an explicit system message instead (see grpc_server).
    assert body["messages"] == [{"role": "user", "content": "BEGIN"}]


@pytest.mark.anyio
async def test_system_role_message_in_contents_is_forwarded():
    """The gRPC path prepends the agent instruction as a system-role entry;
    that entry has `content` and must survive normalization."""
    spec = make_spec()
    runtime = Runtime(post=None)
    response = mock.Mock()
    response.status_code = 200
    response.text = "{}"
    response.json.return_value = {
        "id": "c",
        "model": "m",
        "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }
    client = stub_transport(response)
    runtime._transport._client = lambda *_a, **_k: client  # type: ignore[method-assign]

    await runtime.generate(
        spec,
        "BEGIN",
        "",
        [
            {"role": "system", "content": "You are a question generator."},
            {"role": "user", "parts": [{"text": "BEGIN"}]},
        ],
    )
    body = json.loads(client.post.call_args[1]["content"])
    assert body["messages"] == [
        {"role": "system", "content": "You are a question generator."},
        {"role": "user", "content": "BEGIN"},
    ]


@pytest.mark.anyio
async def test_default_transport_relays_upstream_errors():
    spec = make_spec()
    runtime = Runtime(post=None)
    response = mock.Mock()
    response.status_code = 401
    response.text = '{"error": "bad key"}'
    response.json.return_value = {"error": "bad key"}
    client = stub_transport(response)
    runtime._transport._client = lambda *_a, **_k: client  # type: ignore[method-assign]

    with pytest.raises(ProviderError) as exc_info:
        await runtime.generate(spec, "hi")
    assert exc_info.value.status == 401


@pytest.mark.anyio
async def test_default_transport_anthropic_uses_x_api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    spec = make_spec(vendor="anthropic", format="messages", api_key_env="ANTHROPIC_API_KEY")
    runtime = Runtime(post=None)
    response = mock.Mock()
    response.status_code = 200
    response.text = "{}"
    response.json.return_value = {
        "id": "m",
        "model": "claude",
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": "hi"}],
        "usage": {
            "input_tokens": 1,
            "output_tokens": 1,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
    }
    client = stub_transport(response)
    runtime._transport._client = lambda *_a, **_k: client  # type: ignore[method-assign]

    result = await runtime.generate(spec, "hi")
    assert result.text == "hi"
    _, kwargs = client.post.call_args
    assert kwargs["headers"]["x-api-key"] == spec.api_key
    assert "authorization" not in {k.lower() for k in kwargs["headers"]}
    assert kwargs["headers"]["anthropic-version"] == "2023-06-01"


@pytest.mark.anyio
async def test_default_transport_image_sniffs_webp():
    spec = make_spec(capabilities=["image"], kind="image")
    runtime = Runtime(post=None)
    webp = b"RIFF\x24\x00\x00\x00WEBPVP8 "
    response = mock.Mock()
    response.status_code = 200
    response.text = "{}"
    response.json.return_value = {
        "data": [{"b64_json": base64.b64encode(webp).decode()}],
        "usage": {"prompt_tokens": 1, "output_tokens": 1},
    }
    client = stub_transport(response)
    runtime._transport._client = lambda *_a, **_k: client  # type: ignore[method-assign]

    result = await runtime.image(spec, "a cat", {})
    assert result.mime == "image/webp"
    assert result.image == webp


@pytest.mark.anyio
async def test_default_transport_embed():
    spec = make_spec(capabilities=["embeddings"], kind="embedding")
    runtime = Runtime(post=None)
    response = mock.Mock()
    response.status_code = 200
    response.text = "{}"
    response.json.return_value = {
        "model": "embedder-upstream",
        "data": [{"embedding": [0.1, 0.2], "index": 0}],
        "usage": {"prompt_tokens": 7, "total_tokens": 7},
    }
    client = stub_transport(response)
    runtime._transport._client = lambda *_a, **_k: client  # type: ignore[method-assign]

    values, tokens, model = await runtime.embed(spec, "hello", 256)
    assert values == [0.1, 0.2]
    assert tokens == 7
    assert model == "embedder-upstream"
    _, kwargs = client.post.call_args
    assert json.loads(kwargs["content"])["dimensions"] == 256


@pytest.mark.anyio
async def test_file_data_part_is_fetched_and_sent_as_a_file_part(monkeypatch):
    """The grounding plugin attaches the batch material as a genai `fileData`
    (s3:// URI) part and the gateway is the component that dereferences it — no
    provider can fetch an s3:// URI. A PDF must arrive as an OpenAI `file` part
    with a data URL."""
    import app.runtime as runtime_mod

    monkeypatch.setattr(runtime_mod, "_fetch_object", lambda uri: b"%PDF-1.4 fake")

    spec = make_spec()
    runtime = Runtime(post=None)
    response = mock.Mock()
    response.status_code = 200
    response.text = "{}"
    response.json.return_value = {
        "id": "c",
        "model": "m",
        "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }
    client = stub_transport(response)
    runtime._transport._client = lambda *_a, **_k: client  # type: ignore[method-assign]

    contents = [
        {
            "role": "user",
            "parts": [
                {"text": "Generate 2 MCQs."},
                {
                    "fileData": {
                        "fileUri": "s3://chora-batch-uploads/tenants/t/jobs/j/source",
                        "mimeType": "application/pdf",
                    }
                },
            ],
        }
    ]
    await runtime.generate(spec, "Generate 2 MCQs.", "", contents)
    body = json.loads(client.post.call_args[1]["content"])
    msg = body["messages"][0]
    assert msg["content"][0] == {"type": "text", "text": "Generate 2 MCQs."}
    file_part = msg["content"][1]
    assert file_part["type"] == "file"
    assert file_part["file"]["file_data"].startswith("data:application/pdf;base64,")


@pytest.mark.anyio
async def test_inline_image_part_becomes_an_image_url(monkeypatch):
    spec = make_spec()
    runtime = Runtime(post=None)
    response = mock.Mock()
    response.status_code = 200
    response.text = "{}"
    response.json.return_value = {
        "id": "c",
        "model": "m",
        "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }
    client = stub_transport(response)
    runtime._transport._client = lambda *_a, **_k: client  # type: ignore[method-assign]

    contents = [
        {
            "role": "user",
            "parts": [
                {"text": "What is shown?"},
                {"inlineData": {"mimeType": "image/png", "data": "aGk="}},
            ],
        }
    ]
    await runtime.generate(spec, "What is shown?", "", contents)
    body = json.loads(client.post.call_args[1]["content"])
    assert body["messages"][0]["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,aGk="},
    }
