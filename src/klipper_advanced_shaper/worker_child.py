"""Isolated subprocess entry point used by the supervised Klippy worker."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Optional, Sequence


def _apply_limits(memory_mb: int, cpu_seconds: int) -> None:
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


def _atomic_pickle(path: Path, value: Any) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    temporary = Path(temporary_name)
    os.chmod(temporary, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def diagnostic_numpy_payload(size: int) -> dict[str, Any]:
    """Zero-motion numerical task used to verify the interpreter boundary."""
    import numpy as np

    values = np.linspace(0.0, 16.0 * np.pi, size, dtype=np.float64)
    payload = np.sin(values)
    return {"pid": os.getpid(), "mean": float(np.mean(payload)), "payload": payload}


def diagnostic_sum(values: Sequence[float]) -> float:
    return sum(values)


def diagnostic_failure(message: str) -> None:
    raise ValueError(message)


def diagnostic_sleep(seconds: float) -> None:
    time.sleep(seconds)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--memory-mb", type=int, default=0)
    parser.add_argument("--cpu-seconds", type=int, default=0)
    parser.add_argument("--diagnostic", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.diagnostic:
        _apply_limits(args.memory_mb, args.cpu_seconds)
        result = diagnostic_numpy_payload(32768)
        print(
            json.dumps(
                {
                    "ok": True,
                    "boundary": "external-interpreter",
                    "pid": result["pid"],
                    "numpy_samples": len(result["payload"]),
                    "mean": result["mean"],
                }
            )
        )
        return 0
    if not args.input or not args.output:
        raise SystemExit("--input and --output are required")

    _apply_limits(args.memory_mb, args.cpu_seconds)
    try:
        with Path(args.input).open("rb") as stream:
            function, arguments = pickle.load(stream)
        envelope = (True, function(**dict(arguments)))
    except BaseException:
        envelope = (False, traceback.format_exc())
    try:
        _atomic_pickle(Path(args.output), envelope)
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
