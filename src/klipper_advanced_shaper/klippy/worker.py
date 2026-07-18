"""Supervised low-priority process runner for analysis and artifact work."""

from __future__ import annotations

import multiprocessing
import os
import queue
import traceback
from typing import Any, Callable, Mapping


def _entry(
    output: Any,
    function: Callable[..., Any],
    arguments: Mapping[str, Any],
    memory_mb: int,
    cpu_seconds: int,
) -> None:
    try:
        if hasattr(os, "nice"):
            os.nice(10)
        try:
            import resource

            if memory_mb:
                limit = memory_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
            if cpu_seconds:
                resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
        except (ImportError, OSError, ValueError):
            pass
        output.put((True, function(**dict(arguments))))
    except BaseException:
        output.put((False, traceback.format_exc()))


class SupervisedWorker:
    def __init__(
        self,
        reactor: Any,
        *,
        timeout: float = 600.0,
        memory_mb: int = 1536,
        cpu_seconds: int = 300,
    ) -> None:
        self.reactor = reactor
        self.timeout = float(timeout)
        self.memory_mb = int(memory_mb)
        self.cpu_seconds = int(cpu_seconds)

    def run(
        self,
        function: Callable[..., Any],
        arguments: Mapping[str, Any],
        checkpoint: Callable[[], None],
    ) -> Any:
        methods = multiprocessing.get_all_start_methods()
        context = multiprocessing.get_context("fork" if "fork" in methods else "spawn")
        output = context.Queue(maxsize=1)
        process = context.Process(
            target=_entry,
            args=(output, function, arguments, self.memory_mb, self.cpu_seconds),
            daemon=True,
        )
        process.start()
        started = self.reactor.monotonic()
        try:
            while process.is_alive():
                checkpoint()
                now = self.reactor.monotonic()
                if now - started > self.timeout:
                    raise RuntimeError("analysis worker timed out")
                self.reactor.pause(now + 0.05)
            process.join(timeout=1.0)
            try:
                success, value = output.get_nowait()
            except queue.Empty as error:
                raise RuntimeError("analysis worker exited without a result") from error
            if not success:
                raise RuntimeError("analysis worker failed:\n%s" % value)
            return value
        finally:
            if process.is_alive():
                process.terminate()
            process.join(timeout=1.0)
            output.close()
