#!/usr/bin/env python3
"""Minimal OpenAI-compatible stub, for exercising the gateway without a provider.

Speaks just enough of the spec for the smoke test:
  POST /v1/chat/completions
  POST /v1/images/generations
  POST /v1/embeddings
"""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = 11434


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _read(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def _send(self, code, body):
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        req = self._read()
        # Echo the auth header back in the content so the test can prove the
        # gateway forwarded the registry's credential.
        auth = self.headers.get("Authorization", "(none)")

        if self.path.endswith("/chat/completions"):
            self._send(
                200,
                {
                    "id": "chatcmpl-stub",
                    "object": "chat.completion",
                    "model": req.get("model", "unknown"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": f"stub reply; auth={auth}; "
                                f"messages={len(req.get('messages', []))}; "
                                f"max_tokens={req.get('max_tokens')}",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
                },
            )
        elif self.path.endswith("/images/generations"):
            # 1x1 transparent PNG.
            png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
            self._send(200, {"created": 1, "data": [{"b64_json": png, "revised_prompt": "a stub cat"}]})
        elif self.path.endswith("/responses"):
            # The Responses API shape, including a web_search_call item and
            # url_citation annotations, so the grounded path can be exercised
            # without a real provider.
            wants_search = any("web_search" in (t.get("type") or "") for t in req.get("tools", []))
            output = []
            if wants_search:
                output.append(
                    {
                        "type": "web_search_call",
                        "id": "ws_stub",
                        "status": "completed",
                        "action": {"type": "search", "query": req.get("input", "")},
                    }
                )
            content = {
                "type": "output_text",
                "text": f"stub responses reply; auth={auth}; input={req.get('input')}",
                "annotations": [],
            }
            if wants_search:
                content["annotations"] = [
                    {
                        "type": "url_citation",
                        "url": "https://stub.example/source",
                        "title": "Stub Source",
                        "start_index": 0,
                        "end_index": 4,
                    },
                ]
            output.append(
                {"type": "message", "id": "msg_stub", "status": "completed", "role": "assistant", "content": [content]}
            )
            self._send(
                200,
                {
                    "id": "resp_stub",
                    "object": "response",
                    "created_at": 1,
                    "model": req.get("model", "unknown"),
                    "status": "completed",
                    "output": output,
                    "usage": {"input_tokens": 21, "output_tokens": 13, "total_tokens": 34},
                },
            )
        elif self.path.endswith("/embeddings"):
            self._send(
                200,
                {
                    "model": req.get("model", "unknown"),
                    "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3, 0.4]}],
                    "usage": {"prompt_tokens": 3, "total_tokens": 3},
                },
            )
        else:
            self._send(404, {"error": {"message": f"no stub for {self.path}"}})


if __name__ == "__main__":
    print(f"stub OpenAI-compatible server on :{PORT}", flush=True)
    HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
