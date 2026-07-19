#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! git -C "${repo_dir}" diff --quiet || ! git -C "${repo_dir}" diff --cached --quiet; then
  echo "Refusing to update a checkout with local changes." >&2
  exit 1
fi

git -C "${repo_dir}" pull --ff-only
exec "${repo_dir}/scripts/install.sh"
