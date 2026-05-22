from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any


class ServiceError(RuntimeError):
    pass


class JsonServiceClient:
    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ServiceError(f"{self.base_url}{path} failed: {exc.code} {detail}") from exc
        except urllib.error.URLError as exc:
            raise ServiceError(f"{self.base_url}{path} unavailable: {exc}") from exc
        result = json.loads(body) if body else {}
        if isinstance(result, dict) and result.get("ok") is False:
            raise ServiceError(f"{self.base_url}{path} returned error: {result}")
        return result

    def health(self) -> bool:
        try:
            result = self.post("/health", {})
        except ServiceError:
            return False
        return bool(result.get("ok", True))

    def wait_until_ready(self, deadline_sec: float = 60.0) -> None:
        deadline = time.time() + deadline_sec
        while time.time() < deadline:
            if self.health():
                return
            time.sleep(0.5)
        raise ServiceError(f"{self.base_url} did not become ready within {deadline_sec}s")

