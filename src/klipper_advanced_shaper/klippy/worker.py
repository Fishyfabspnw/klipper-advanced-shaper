"""Supervised external-interpreter runner for analysis and artifact work."""

from __future__ import annotations

import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Optional


def _private_pickle(path: Path, value: Any) -> None:
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _stop_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2.0)


class SupervisedWorker:
    def __init__(
        self,
        reactor: Any,
        *,
        timeout: float = 600.0,
        memory_mb: int = 1536,
        cpu_seconds: int = 300,
        temporary_root: Optional[str] = None,
    ) -> None:
        self.reactor = reactor
        self.timeout = float(timeout)
        self.memory_mb = int(memory_mb)
        self.cpu_seconds = int(cpu_seconds)
        self.temporary_root = temporary_root

    def run(
        self,
        function: Callable[..., Any],
        arguments: Mapping[str, Any],
        checkpoint: Callable[[], None],
    ) -> Any:
        # A standalone interpreter avoids inheriting Klippy's greenlet, thread,
        # allocator, and numerical-library state.  Private files also avoid the
        # bounded-pipe shutdown deadlocks possible with multiprocessing IPC.
        work_dir = Path(
            tempfile.mkdtemp(prefix="advanced-shaper-worker-", dir=self.temporary_root)
        )
        os.chmod(work_dir, 0o700)
        input_path = work_dir / "request.pickle"
        result_path = work_dir / "result.pickle"
        stderr_path = work_dir / "stderr.log"
        process: Optional[subprocess.Popen[Any]] = None
        stderr_handle = None
        try:
            _private_pickle(input_path, (function, dict(arguments)))
            stderr_descriptor = os.open(
                str(stderr_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
            stderr_handle = os.fdopen(stderr_descriptor, "wb")
            command = [
                sys.executable,
                "-m",
                "klipper_advanced_shaper.worker_child",
                "--input",
                str(input_path),
                "--output",
                str(result_path),
                "--memory-mb",
                str(self.memory_mb),
                "--cpu-seconds",
                str(self.cpu_seconds),
            ]
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=stderr_handle,
                shell=False,
                close_fds=True,
            )
            started = self.reactor.monotonic()
            while process.poll() is None:
                checkpoint()
                now = self.reactor.monotonic()
                if now - started > self.timeout:
                    raise RuntimeError("analysis worker timed out")
                self.reactor.pause(now + 0.05)

            stderr_handle.close()
            stderr_handle = None
            if process.returncode != 0:
                detail = stderr_path.read_text(encoding="utf-8", errors="replace")[-8192:]
                suffix = ":\n%s" % detail if detail else ""
                raise RuntimeError(
                    "analysis worker exited with code %s%s" % (process.returncode, suffix)
                )
            if not result_path.is_file():
                raise RuntimeError("analysis worker exited without a result")
            with result_path.open("rb") as stream:
                success, value = pickle.load(stream)
            if not success:
                raise RuntimeError("analysis worker failed:\n%s" % value)
            return value
        finally:
            try:
                if process is not None:
                    _stop_process(process)
            finally:
                if stderr_handle is not None:
                    stderr_handle.close()
                # Cleanup failure is intentionally not hidden: raw calibration
                # inputs must not be silently left behind on the printer host.
                shutil.rmtree(work_dir)
