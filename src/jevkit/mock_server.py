"""A local mock of OpenRouter's System One endpoint (`POST /api/v1/systemone`).

It lets the real `OpenRouterTransport` run end to end (JSON over HTTP, auth header,
status codes, response validation) with no key or network. Answers come from any
`Transport`, typically an `OfflineTransport` with hand-written rules.

    with MockSystemOneServer(offline_rules) as server:
        jev = Jev(OpenRouterTransport(api_key="mock", base_url=server.base_url))
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType

from pydantic import ValidationError

from .client import Transport
from .wire import DecisionRequest


class MockSystemOneServer:
    def __init__(self, answerer: Transport, host: str = "127.0.0.1", port: int = 0) -> None:
        self.answerer = answerer
        self.requests: list[DecisionRequest] = []
        handler = self._handler()
        self._server = ThreadingHTTPServer((host, port), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host!s}:{port}/api"

    def __enter__(self) -> MockSystemOneServer:
        self._thread.start()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        self._server.shutdown()
        self._server.server_close()

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        mock = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:  # keep demo output clean
                pass

            def _reply(self, code: int, body: dict[str, object]) -> None:
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self) -> None:
                if self.path != "/api/v1/systemone":
                    return self._reply(404, {"error": {"message": "Not Found", "code": 404}})
                if not self.headers.get("Authorization", "").startswith("Bearer "):
                    return self._reply(401, {"error": {"message": "Missing API key",
                                                       "code": 401}})
                raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                try:
                    request = DecisionRequest.model_validate_json(raw)
                except ValidationError as err:
                    return self._reply(400, {"error": {"message": str(err), "code": 400}})
                mock.requests.append(request)
                response = mock.answerer.send(request)
                body = response.model_dump(mode="json", exclude_none=True)
                body.setdefault("id", f"gen-dec-mock-{len(mock.requests)}")
                body["model"] = "mock/jev-1.14"
                body["provider"] = "mock"
                self._reply(200, body)

        return Handler
