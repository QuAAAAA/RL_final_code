from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable


Handler = Callable[[dict[str, Any]], dict[str, Any]]


class JsonApiServer:
    def __init__(self) -> None:
        self.routes: dict[str, Handler] = {"/health": lambda _payload: {"ok": True}}

    def route(self, path: str) -> Callable[[Handler], Handler]:
        def decorator(func: Handler) -> Handler:
            self.routes[path] = func
            return func

        return decorator

    def serve(self, host: str, port: int) -> None:
        routes = self.routes

        class RequestHandler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length).decode("utf-8") if length else "{}"
                try:
                    payload = json.loads(raw) if raw else {}
                    if self.path not in routes:
                        self._send(404, {"ok": False, "error": f"unknown route: {self.path}"})
                        return
                    result = routes[self.path](payload)
                    self._send(200, result)
                except Exception as exc:
                    self._send(500, {"ok": False, "error": str(exc), "type": type(exc).__name__})

            def log_message(self, fmt: str, *args: object) -> None:
                print(f"[api] {self.address_string()} {fmt % args}")

            def _send(self, status: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        ThreadingHTTPServer((host, port), RequestHandler).serve_forever()


def server_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    return parser.parse_args()

