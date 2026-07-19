#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
klipper_dir="${KLIPPER_DIR:-${HOME}/klipper}"
klipper_venv="${KLIPPER_VENV:-${HOME}/klippy-env}"
extras_dir="${klipper_dir}/klippy/extras"
loader_source="${repo_dir}/scripts/advanced_input_shaper.py"
loader_target="${extras_dir}/advanced_input_shaper.py"

if [[ ! -x "${klipper_venv}/bin/python" || ! -d "${extras_dir}" ]]; then
  echo "Klipper or its Python environment was not found." >&2
  echo "Set KLIPPER_DIR and KLIPPER_VENV to the correct absolute paths." >&2
  exit 1
fi

"${klipper_venv}/bin/python" -m pip install --upgrade "${repo_dir}"
if [[ -f "${loader_target}" ]] && ! cmp -s "${loader_source}" "${loader_target}"; then
  cp -p "${loader_target}" "${loader_target}.previous"
  echo "Preserved the previous loader as ${loader_target}.previous"
fi
install -m 0644 "${loader_source}" "${loader_target}"

echo "Installed Klipper Advanced Shaper. Add [advanced_input_shaper] to printer.cfg."
echo "Restart the Klipper host service while idle (often: sudo systemctl restart klipper)."
echo "Klipper's G-code RESTART does not reload updated Python package code."
