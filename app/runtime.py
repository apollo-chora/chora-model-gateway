from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass, field
from typing import Any

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from strands import Agent
from strands.models.openai import OpenAIModel
from strands.models.openai_responses import OpenAIResponsesModel


@dataclass(slots=True)
class Result:
    text: str = ""
    model: str = ""
    vendor: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    image: bytes = b""
    mime: str = ""
    revised_prompt: str = ""
    finish: str = "stop"


class Runtime:
    def __init__(self, search_provider: str = "exa") -> None:
        self.search_provider = search_provider

    def args(self, spec: Any) -> dict[str, Any]:
        args: dict[str, Any] = {"api_key": spec.api_key or "not-used"}
        if spec.base_url:
            args["base_url"] = spec.base_url
        if spec.extra_headers:
            args["default_headers"] = spec.extra_headers
        return args

    async def generate(
        self, spec: Any, prompt: str, system: str = "", messages: Any = None,
        tools: Any = None, params: dict[str, Any] | None = None, grounded: bool = False,
    ) -> Result:
        values = dict(params or {})
        if spec.format == "messages":
            return await self.anthropic(spec, prompt, system, messages, tools, values, grounded)

        if spec.format == "responses":
            response_model: Any = OpenAIResponsesModel(
                model_id=spec.model,
                client_args=self.args(spec),
                params=values | ({"tools": [{"type": "web_search"}]} if grounded and spec.support_grounding else {}),
            )
        else:
            response_model = OpenAIModel(model_id=spec.model, client_args=self.args(spec), params=values)

        # Strands' Agent constructor accepts client-side tools via `tools`.
        # Provider-native search is already configured on the model above.
        agent = Agent(model=response_model, system_prompt=system or None, tools=[] if grounded and not spec.support_grounding else None)
        result = await asyncio.to_thread(agent, prompt)
        return Result(text=str(result), model=spec.model, vendor=spec.format)

    async def anthropic(
        self, spec: Any, prompt: str, system: str, messages: Any, tools: Any,
        params: dict[str, Any], grounded: bool,
    ) -> Result:
        client = AsyncAnthropic(
            api_key=spec.api_key,
            base_url=spec.base_url or None,
            default_headers=spec.extra_headers or None,
        )
        tool_list = list(tools or [])
        if grounded:
            tool_list.append({"type": "web_search_20250305", "name": "web_search", "max_uses": 5})
        kwargs: dict[str, Any] = {
            "model": spec.model,
            "messages": messages or [{"role": "user", "content": prompt}],
            "max_tokens": int(params.get("max_tokens", spec.max_output_tokens or 4096)),
        }
        if system:
            kwargs["system"] = system
        if tool_list:
            kwargs["tools"] = tool_list
        response = await client.messages.create(**kwargs)
        text = "".join(getattr(block, "text", "") for block in response.content)
        tool_calls = [
            {
                "id": block.id,
                "type": "function",
                "function": {"name": block.name, "arguments": json.dumps(block.input)},
            }
            for block in response.content
            if getattr(block, "type", "") == "tool_use"
        ]
        return Result(
            text=text, model=response.model, vendor="messages",
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            tool_calls=tool_calls, finish=response.stop_reason or "stop",
        )

    async def image(self, spec: Any, prompt: str, params: dict[str, Any]) -> Result:
        response = await AsyncOpenAI(**self.args(spec)).images.generate(
            model=spec.model, prompt=prompt,
            **{key: value for key, value in params.items() if key in {"size", "quality", "style"}},
        )
        item = response.data[0]
        raw = base64.b64decode(item.b64_json or "")
        return Result(
            model=spec.model, vendor="images", image=raw,
            mime="image/webp" if raw[:4] == b"RIFF" else "image/png",
            revised_prompt=item.revised_prompt or "",
        )

    async def embed(self, spec: Any, text: str, dims: int = 0) -> tuple[list[float], int, str]:
        kwargs: dict[str, Any] = {"model": spec.model, "input": text}
        if dims:
            kwargs["dimensions"] = dims
        response = await AsyncOpenAI(**self.args(spec)).embeddings.create(**kwargs)
        return response.data[0].embedding, response.usage.prompt_tokens, response.model
