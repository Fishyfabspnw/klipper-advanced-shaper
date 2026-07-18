"""Private, atomic result artifacts for calibration review and replay."""

from __future__ import annotations

import hashlib
import html
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")


class ArtifactWriter:
    def __init__(self, root: str | os.PathLike[str], keep_raw: bool = True) -> None:
        self.root = Path(root).expanduser()
        self.keep_raw = bool(keep_raw)

    def write(
        self,
        result_id: str,
        report: Mapping[str, Any],
        raw_groups: Mapping[str, Mapping[str, list[Any]]] | None = None,
    ) -> dict[str, Any]:
        destination = self.root / result_id
        destination.mkdir(parents=True, exist_ok=True)
        report_path = destination / "result.json"
        _atomic_bytes(report_path, _json_bytes(report))
        svg_path = destination / "summary.svg"
        _atomic_bytes(svg_path, self._svg(report).encode("utf-8"))
        png_path = destination / "summary.png"
        self._png(report, png_path)
        html_path = destination / "report.html"
        _atomic_bytes(
            html_path,
            self._html(report, svg_path.name, png_path.name).encode("utf-8"),
        )
        artifacts: dict[str, Any] = {
            "json": str(report_path),
            "svg": str(svg_path),
            "png": str(png_path),
            "html": str(html_path),
        }
        if self.keep_raw and raw_groups:
            raw_path = destination / "captures.npz"
            arrays = {}
            for group, axes in raw_groups.items():
                for axis, captures in axes.items():
                    for index, capture in enumerate(captures):
                        arrays["%s_%s_%d" % (group, axis, index)] = np.asarray(
                            capture["samples"], dtype=float
                        )
            with tempfile.NamedTemporaryFile(
                dir=destination, suffix=".npz", delete=False
            ) as stream:
                temporary = Path(stream.name)
            try:
                np.savez_compressed(temporary, **arrays)
                os.replace(temporary, raw_path)
            finally:
                if temporary.exists():
                    temporary.unlink()
            artifacts["raw"] = str(raw_path)
        artifacts["sha256"] = {
            key: hashlib.sha256(Path(value).read_bytes()).hexdigest()
            for key, value in artifacts.items()
            if key != "sha256"
        }
        manifest_path = destination / "manifest.json"
        _atomic_bytes(
            manifest_path,
            _json_bytes(
                {
                    "schema_version": report.get("schema_version"),
                    "result_id": result_id,
                    "sha256": artifacts["sha256"],
                }
            ),
        )
        artifacts["manifest"] = str(manifest_path)
        return artifacts

    @staticmethod
    def _png(report: Mapping[str, Any], path: Path) -> None:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        labels = []
        accelerations = []
        colors = []
        spectra = []
        for axis, details in report.get("axes", {}).items():
            for candidate in details.get("candidates", []):
                labels.append("%s %s" % (axis, candidate["name"]))
                accelerations.append(float(candidate["max_accel"]))
                colors.append(
                    "#7c3aed" if candidate["name"] == details.get("selected") else "#94a3b8"
                )
            spectrum = details.get("spectrum", {})
            if spectrum:
                spectra.append((axis, spectrum["frequency_hz"], spectrum["psd"]))
        figure, plots = plt.subplots(3, 1, figsize=(11, 12))
        for axis, frequencies, psd in spectra:
            plots[0].semilogy(frequencies, np.maximum(psd, 1e-18), label=axis)
        plots[0].set(xlabel="Frequency (Hz)", ylabel="PSD", title="Robust repeat spectrum")
        if spectra:
            plots[0].legend()
        plots[1].bar(labels or ["no result"], accelerations or [0.0], color=colors or ["#94a3b8"])
        plots[1].set_ylabel("Smoothing-derived max acceleration (mm/s²)")
        plots[1].set_title("Native candidates and Pareto selection")
        plots[1].tick_params(axis="x", rotation=35)
        validation = report.get("validation", {}).get("axes", {})
        validation_labels = []
        before = []
        after = []
        for axis, values in validation.items():
            validation_labels.append(axis)
            before.append(values["baseline_energy"])
            after.append(values["shaped_energy"])
        positions = np.arange(len(validation_labels) or 1)
        plots[2].bar(positions - 0.18, before or [0.0], 0.36, label="reference")
        plots[2].bar(positions + 0.18, after or [0.0], 0.36, label="candidate")
        plots[2].set_xticks(positions, validation_labels or ["pending"])
        plots[2].set(ylabel="Resonant-band energy", title="Held-out before/after validation")
        plots[2].legend()
        figure.tight_layout()
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".png", delete=False) as stream:
            temporary = Path(stream.name)
        try:
            figure.savefig(temporary, dpi=160)
            plt.close(figure)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def _svg(report: Mapping[str, Any]) -> str:
        axes = report.get("axes", {})
        rows = []
        y = 55
        for axis, details in axes.items():
            selected = details.get("selected", "abstained")
            modes = ", ".join("%.1f Hz" % item["frequency"] for item in details.get("modes", []))
            rows.append(
                '<text x="30" y="%d">%s: %s — modes %s</text>'
                % (y, html.escape(axis), html.escape(selected), html.escape(modes))
            )
            y += 32
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="%d" role="img">'
            '<rect width="100%%" height="100%%" fill="#111827"/>'
            '<g fill="#e5e7eb" font-family="sans-serif" font-size="20">'
            '<text x="30" y="28" font-weight="bold">Klipper Advanced Shaper</text>%s</g></svg>'
            % (max(110, y), "".join(rows))
        )

    @staticmethod
    def _html(report: Mapping[str, Any], svg_name: str, png_name: str) -> str:
        payload = html.escape(json.dumps(report, indent=2, sort_keys=True))
        return (
            "<!doctype html><meta charset='utf-8'><title>Advanced Shaper Report</title>"
            "<style>body{font-family:system-ui;max-width:1100px;margin:2rem auto;"
            "background:#f8fafc}img{max-width:100%}pre{background:#111827;"
            "color:#e5e7eb;padding:1rem;overflow:auto}</style>"
            "<h1>Klipper Advanced Shaper</h1><img src='"
            + html.escape(svg_name)
            + "' alt='Calibration summary'>"
            + "<img src='"
            + html.escape(png_name)
            + "' alt='Spectrum, candidates, and validation'><pre>"
            + payload
            + "</pre>"
        )
