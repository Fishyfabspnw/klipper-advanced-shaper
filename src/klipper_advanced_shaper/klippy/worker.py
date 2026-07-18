"""Supervised low-priority process runner for analysis and artifact work."""

from __future__ import annotations

import multiprocessing
import os
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
        output.send((True, function(**dict(arguments))))
    except BaseException:
        try:
            output.send((False, traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            # The parent may have cancelled or timed out while work was running.
            pass
    finally:
        output.close()


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
        # A Queue owns a feeder thread.  A child returning a sufficiently large
        # value can block during interpreter shutdown until the parent drains
        # that feeder, while the old parent loop waited for child shutdown
        # before reading the Queue.  A one-way Connection plus concurrent
        # polling avoids that circular wait and does not need a feeder thread.
        output, child_output = context.Pipe(duplex=False)
        process = context.Process(
            target=_entry,
            args=(child_output, function, arguments, self.memory_mb, self.cpu_seconds),
            daemon=True,
        )
        process.start()
        child_output.close()
        started = self.reactor.monotonic()
        result: Any = None
        received = False
        try:
            while True:
                checkpoint()
                now = self.reactor.monotonic()
                if now - started > self.timeout:
                    raise RuntimeError("analysis worker timed out")
                if not received and output.poll(0.0):
                    try:
                        result = output.recv()
                    except (EOFError, OSError) as error:
                        raise RuntimeError(
                            "analysis worker closed its result channel"
                        ) from error
                    received = True
                if not process.is_alive():
                    break
                self.reactor.pause(now + 0.05)
            process.join(timeout=1.0)
            if not received and output.poll(0.1):
                try:
                    result = output.recv()
                except (EOFError, OSError) as error:
                    raise RuntimeError(
                        "analysis worker closed its result channel"
                    ) from error
                received = True
            if not received:
                raise RuntimeError(
                    "analysis worker exited without a result (exit code %s)"
                    % process.exitcode
                )
            success, value = result
            if not success:
                raise RuntimeError("analysis worker failed:\n%s" % value)
            return value
        finally:
            if process.is_alive():
                process.terminate()
            process.join(timeout=1.0)
            output.close()
