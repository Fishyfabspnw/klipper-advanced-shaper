#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
klipper_dir="${KLIPPER_DIR:-${HOME}/klipper}"
klipper_venv="${KLIPPER_VENV:-${HOME}/klippy-env}"
extras_dir="${klipper_dir}/klippy/extras"
loader_source="${repo_dir}/scripts/advanced_input_shaper.py"
loader_target="${extras_dir}/advanced_input_shaper.py"
previous_loader="${loader_target}.previous"

if [[ ! -x "${klipper_venv}/bin/python" || ! -d "${extras_dir}" ]]; then
  echo "Klipper or its Python environment was not found." >&2
  echo "Set KLIPPER_DIR and KLIPPER_VENV to the correct absolute paths." >&2
  exit 1
fi

if [[ -f "${loader_target}" ]] && ! cmp -s "${loader_source}" "${loader_target}"; then
  echo "Refusing to remove a loader that differs from this checkout:" >&2
  echo "${loader_target}" >&2
  exit 1
fi

if [[ -f "${loader_target}" ]]; then
  rm -- "${loader_target}"
fi
if [[ -f "${previous_loader}" ]]; then
  mv -- "${previous_loader}" "${loader_target}"
  echo "Restored the previous loader at ${loader_target}"
fi

"${klipper_venv}/bin/python" -m pip uninstall --yes klipper-advanced-shaper

echo "Uninstalled Klipper Advanced Shaper. Remove [advanced_input_shaper] from printer.cfg."
echo "Restart the Klipper host service while idle (often: sudo systemctl restart klipper)."
echo "The repository checkout and result files were left in place."
