#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
klipper_dir="${KLIPPER_DIR:-${HOME}/klipper}"
klipper_venv="${KLIPPER_VENV:-${HOME}/klippy-env}"
extras_dir="${klipper_dir}/klippy/extras"
config_dir="${KLIPPER_CONFIG_DIR:-${HOME}/printer_data/config}"
loader_source="${repo_dir}/scripts/advanced_input_shaper.py"
loader_target="${extras_dir}/advanced_input_shaper.py"
macros_source="${repo_dir}/config/advanced_shaper_macros.cfg"
macros_target="${config_dir}/advanced_shaper_macros.cfg"

if [[ ! -x "${klipper_venv}/bin/python" || ! -d "${extras_dir}" ]]; then
  echo "Klipper or its Python environment was not found." >&2
  echo "Set KLIPPER_DIR and KLIPPER_VENV to the correct absolute paths." >&2
  exit 1
fi

# First resolve any missing dependencies using pip's normal only-if-needed
# behavior. Then replace this alpha package explicitly: local development
# builds intentionally share a version and --upgrade alone may keep old code.
"${klipper_venv}/bin/python" -m pip install --upgrade "${repo_dir}"
"${klipper_venv}/bin/python" -m pip install \
  --force-reinstall --no-deps "${repo_dir}"
if [[ -f "${loader_target}" ]] && ! cmp -s "${loader_source}" "${loader_target}"; then
  cp -p "${loader_target}" "${loader_target}.previous"
  echo "Preserved the previous loader as ${loader_target}.previous"
fi
install -m 0644 "${loader_source}" "${loader_target}"

if [[ -d "${config_dir}" ]]; then
  if [[ -f "${macros_target}" ]] && ! cmp -s "${macros_source}" "${macros_target}"; then
    cp -p "${macros_target}" "${macros_target}.previous"
    echo "Preserved the previous macro file as ${macros_target}.previous"
  fi
  install -m 0644 "${macros_source}" "${macros_target}"
  echo "Installed Mainsail macros as ${macros_target}."
else
  echo "Printer config directory not found; copy ${macros_source} manually."
  echo "Set KLIPPER_CONFIG_DIR if the printer config directory is elsewhere."
fi

echo "Installed Klipper Advanced Shaper. Add [advanced_input_shaper] to printer.cfg."
echo "The local package was force-reinstalled; dependencies were not force-reinstalled."
echo "Restart the Klipper host service while idle (often: sudo systemctl restart klipper)."
echo "Klipper's G-code RESTART does not reload updated Python package code."
