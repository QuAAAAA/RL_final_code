from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .http_client import JsonServiceClient
from .schemas import ServiceSpec


@dataclass
class ManagedProcess:
    name: str
    process: subprocess.Popen

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()


class ServiceManager:
    def __init__(self, specs: list[ServiceSpec]) -> None:
        self.specs = specs
        self.processes: list[ManagedProcess] = []

    def start(self) -> None:
        for spec in self.specs:
            if not spec.command:
                continue
            env = os.environ.copy()
            if spec.env:
                env.update(spec.env)
            process = subprocess.Popen(
                spec.command,
                cwd=str(spec.cwd) if spec.cwd else None,
                env=env,
            )
            self.processes.append(ManagedProcess(spec.name, process))
            if spec.url:
                JsonServiceClient(spec.url).wait_until_ready()

    def stop(self) -> None:
        for process in reversed(self.processes):
            process.stop()

    def __enter__(self) -> "ServiceManager":
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()


def spec_from_config(name: str, item: dict) -> ServiceSpec:
    cwd = Path(item["cwd"]).expanduser().resolve() if item.get("cwd") else None
    return ServiceSpec(
        name=name,
        url=item.get("url"),
        mock=bool(item.get("mock", False)),
        command=item.get("command"),
        cwd=cwd,
        env=item.get("env"),
    )

