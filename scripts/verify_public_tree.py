"""Fail CI if files resemble private captures, credentials, or agent state."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PATH_PARTS = {
    ".codex",
    ".ssh",
    "agent_handoff",
    "captures",
    "printer_data",
    "secrets",
}
FORBIDDEN_SUFFIXES = {".stdata", ".pem", ".key"}
SECRET_PATTERNS = (
    re.compile("-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE" + " KEY-----"),
    re.compile(r"(?i)password\s*[:=]\s*[^\s${][^\s]*"),
    re.compile(r"(?i)ssh" + r"pass\b"),
)
TEXT_SUFFIXES = {
    ".cfg",
    ".html",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def tracked_files() -> list[Path]:
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True
    )
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    names = set(tracked.stdout.splitlines()) | set(untracked.stdout.splitlines())
    return [ROOT / line for line in sorted(names) if line]


def violations(paths: list[Path]) -> list[str]:
    found: list[str] = []
    for path in paths:
        try:
            relative = path.relative_to(ROOT)
        except ValueError:
            relative = Path(path.name)
        lowered_parts = {part.lower() for part in relative.parts}
        if lowered_parts & FORBIDDEN_PATH_PARTS or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            found.append(f"forbidden public path: {relative}")
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES or not path.is_file():
            continue
        content = path.read_text(encoding="utf-8", errors="replace")
        for pattern in SECRET_PATTERNS:
            if pattern.search(content):
                found.append(f"credential-like content in: {relative}")
                break
    return found


def main() -> int:
    found = violations(tracked_files())
    if found:
        print("Public-tree verification failed:")
        print("\n".join(f"- {item}" for item in found))
        return 1
    print("Public-tree verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
