from __future__ import annotations

from typing import Any


def model_entry(spec: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": spec.id,
        "object": "model",
        "created": 0,
        "owned_by": spec.vendor,
    }
    if spec.context_window:
        out["context_window"] = spec.context_window
    if spec.max_output_tokens:
        out["max_output_tokens"] = spec.max_output_tokens
    if spec.capabilities:
        out["capabilities"] = spec.capabilities
    if spec.base_url:
        out["base_url"] = spec.base_url
    return out


def generation_params(body: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        key: body[key] for key in ("temperature", "top_p", "stop", "n", "seed") if key in body and body[key] is not None
    }
    maximum = body.get("max_tokens")
    if maximum is None:
        maximum = body.get("max_completion_tokens")
    if maximum is not None:
        out["max_tokens"] = maximum
    return out


def wants_grounding(tools: list[dict[str, Any]]) -> bool:
    for tool in tools:
        for key in ("type", "name"):
            value = str(tool.get(key, "")).lower()
            if "web_search" in value or "web_fetch" in value:
                return True
    return False


def _input_item_text(item: Any) -> str:
    if not isinstance(item, dict):
        raise ValueError('each entry of "input" must be an object')
    content = item.get("content")
    if content is None:
        text = item.get("text")
        if isinstance(text, str) and text:
            return text
        raise ValueError("an input item needs content or text")
    if isinstance(content, str):
        if content:
            return content
        raise ValueError("the input item carried no text")
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            kind = part.get("type", "")
            if kind not in ("", "input_text", "text", "output_text"):
                raise ValueError("unsupported non-text input part")
            text = part.get("text")
            if isinstance(text, str):
                parts.append(text)
        if parts:
            return "".join(parts)
        raise ValueError("the input item carried no text")
    raise ValueError("invalid input item content")


def flatten_responses_input(value: Any) -> tuple[str, list[dict[str, Any]] | None]:
    if value is None:
        raise ValueError('"input" is required')
    if isinstance(value, str):
        if not value.strip():
            raise ValueError('"input" must not be empty')
        return value, None
    if not isinstance(value, list):
        raise ValueError('"input" must be a string or an array of input items')
    if not value:
        raise ValueError('"input" must not be empty')
    if len(value) == 1:
        return _input_item_text(value[0]), None

    messages: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"input[{index}] must be an object")
        messages.append(
            {
                "role": item.get("role") or "user",
                "content": _input_item_text(item),
            }
        )
    return "", messages


def embedding_inputs(value: Any) -> list[str]:
    if isinstance(value, str):
        if not value:
            raise ValueError('"input" must not be empty')
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        if not value:
            raise ValueError('"input" must not be empty')
        for index, item in enumerate(value):
            if not item:
                raise ValueError(f'"input"[{index}] is empty')
        return value
    raise ValueError('"input" must be a string or an array of strings; token arrays are not supported')
