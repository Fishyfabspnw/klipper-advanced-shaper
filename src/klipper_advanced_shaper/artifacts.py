# ruff: noqa: E501
"""Private, atomic result artifacts for calibration review and replay."""

from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

BLUE = "#2563eb"
ORANGE = "#ea580c"
GOLD = "#ca8a04"
SLATE = "#64748b"
INK = "#172033"


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


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _number(value: Any, digits: int = 1, fallback: str = "Not available") -> str:
    number = _finite(value)
    return fallback if number is None else f"{number:,.{digits}f}"


def _percent(value: Any, digits: int = 1, fallback: str = "Not available") -> str:
    number = _finite(value)
    return fallback if number is None else f"{number * 100:.{digits}f}%"


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _axes(report: Mapping[str, Any]) -> Mapping[str, Any]:
    axes = report.get("axes", {})
    return axes if isinstance(axes, Mapping) else {}


def _validation_axes(report: Mapping[str, Any]) -> Mapping[str, Any]:
    validation = report.get("validation", {})
    if not isinstance(validation, Mapping):
        return {}
    axes = validation.get("axes", {})
    return axes if isinstance(axes, Mapping) else {}


def _selected_candidate(details: Mapping[str, Any]) -> Mapping[str, Any]:
    selected = str(details.get("selected", "")).lower()
    candidates = details.get("candidates", [])
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        return {}
    for candidate in candidates:
        if isinstance(candidate, Mapping) and str(candidate.get("name", "")).lower() == selected:
            return candidate
    return {}


def _positive_frequency_max(values: Any) -> Optional[float]:
    try:
        frequencies = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    frequencies = frequencies[np.isfinite(frequencies) & (frequencies > 0.0)]
    return float(np.max(frequencies)) if frequencies.size else None


def _plot_frequency_ceiling(
    details: Mapping[str, Any], fallback_frequencies: Any
) -> Optional[float]:
    """Bound display axes to measured sweep evidence, not sensor Nyquist bins."""
    spectrogram = details.get("spectrogram", {})
    if isinstance(spectrogram, Mapping) and bool(spectrogram.get("available")):
        limit = _positive_frequency_max(spectrogram.get("frequency_hz", []))
        if limit is not None:
            return limit + max(5.0, limit * 0.05)
    evidence_limits: list[float] = []
    native_candidates = details.get("native_candidates", [])
    if isinstance(native_candidates, Sequence) and not isinstance(
        native_candidates, (str, bytes)
    ):
        for candidate in native_candidates:
            if not isinstance(candidate, Mapping):
                continue
            response = candidate.get("native_frequency_response", {})
            if not isinstance(response, Mapping):
                continue
            limit = _positive_frequency_max(response.get("frequency_hz", []))
            if limit is not None:
                evidence_limits.append(limit)
    if evidence_limits:
        measured_limit = max(evidence_limits)
        return measured_limit + max(5.0, measured_limit * 0.05)
    return _positive_frequency_max(fallback_frequencies)


def _plot_candidate_subset(
    details: Mapping[str, Any], limit: int = 5
) -> tuple[Mapping[str, Any], ...]:
    """Keep plots legible while full candidate data remains in JSON/CSV/HTML."""
    candidates = details.get("candidates", [])
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        return ()
    usable = [
        candidate
        for candidate in candidates
        if isinstance(candidate, Mapping) and _finite(candidate.get("max_accel")) is not None
    ]
    if limit <= 0 or len(usable) <= limit:
        return tuple(usable)
    selected = str(details.get("selected", "")).lower()
    ordered = sorted(
        usable,
        key=lambda candidate: (
            str(candidate.get("name", "")).lower() != selected,
            -float(candidate["max_accel"]),
            str(candidate.get("name", "")),
        ),
    )
    return tuple(ordered[:limit])


def _plot_candidate_label(axis: str, name: str) -> str:
    if "(" in name and name.endswith(")"):
        family, arguments = name.split("(", 1)
        return f"{axis} · {family}\n({arguments}"
    return f"{axis} · {name}"


def _selected_modeled_response(
    details: Mapping[str, Any], frequencies: Any
) -> Optional[np.ndarray]:
    """Return the selected shaper response using recorded design parameters."""
    grid = np.asarray(frequencies, dtype=float).reshape(-1)
    if not grid.size or np.any(~np.isfinite(grid)) or np.any(grid < 0.0):
        return None
    selected = str(details.get("selected", ""))
    selected_lower = selected.lower()
    for candidate in details.get("native_candidates", []):
        if not isinstance(candidate, Mapping):
            continue
        if str(candidate.get("name", "")).lower() != selected_lower:
            continue
        response = candidate.get("native_frequency_response", {})
        if not isinstance(response, Mapping):
            continue
        response_frequency = np.asarray(response.get("frequency_hz", []), dtype=float)
        response_ratio = np.asarray(response.get("response_ratio", []), dtype=float)
        if len(response_frequency) and len(response_frequency) == len(response_ratio):
            return np.interp(grid, response_frequency, response_ratio)
    selected_candidate = _selected_candidate(details)
    metadata = selected_candidate.get("metadata", {})
    metadata = metadata if isinstance(metadata, Mapping) else {}
    if not bool(metadata.get("parameterized")):
        return None
    try:
        from .analysis.experimental import generalized_mzv_pulses, oscillator_response
        from .shapers import parse_shaper_identifier

        identifier = parse_shaper_identifier(selected)
        if identifier.family != "mzv":
            return None
        arguments = identifier.argument_map()
        amplitudes, pulse_times = generalized_mzv_pulses(
            int(arguments["n"]),
            identifier.mzv_spacing(),
            float(selected_candidate["frequency"]),
            float(metadata["design_damping_ratio"]),
        )
        return oscillator_response(
            amplitudes,
            pulse_times,
            grid,
            float(metadata["design_damping_ratio"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _status(report: Mapping[str, Any]) -> tuple[str, str, str]:
    value = str(report.get("status", "")).lower()
    validation = report.get("validation", {})
    validation = validation if isinstance(validation, Mapping) else {}
    if value == "rejected":
        reason = report.get("reason") or validation.get("reason") or "Acceptance gates were not met."
        return "REJECTED", str(reason), "rejected"
    if value == "accepted":
        return "ACCEPTED FOR REVIEW", "The measured acceptance gates passed.", "accepted"
    if report.get("abstain"):
        return "NO RECOMMENDATION", str(report.get("reason", "Analysis abstained.")), "pending"
    return "ANALYSIS REPORT", "Validation has not produced an acceptance decision.", "pending"


def _csv_bytes(rows: Sequence[Sequence[Any]]) -> bytes:
    stream = io.StringIO(newline="")
    csv.writer(stream, lineterminator="\n").writerows(rows)
    return stream.getvalue().encode("utf-8")


class ArtifactWriter:
    def __init__(self, root: str | os.PathLike[str], keep_raw: bool = True) -> None:
        self.root = Path(root).expanduser()
        self.keep_raw = bool(keep_raw)

    def write(
        self,
        result_id: str,
        report: Mapping[str, Any],
        raw_groups: Optional[Mapping[str, Mapping[str, list[Any]]]] = None,
    ) -> dict[str, Any]:
        destination = self.root / result_id
        destination.mkdir(parents=True, exist_ok=True)
        report_path = destination / "result.json"
        _atomic_bytes(report_path, _json_bytes(report))
        svg_path = destination / "summary.svg"
        _atomic_bytes(svg_path, self._svg(report).encode("utf-8"))
        png_path = destination / "summary.png"
        self._png(report, png_path)
        shaper_png_path = destination / "input_shaper.png"
        shaper_svg_path = destination / "input_shaper.svg"
        self._input_shaper_plots(report, shaper_png_path, shaper_svg_path)
        html_path = destination / "report.html"
        _atomic_bytes(
            html_path,
            self._html(
                report,
                svg_path.name,
                png_path.name,
                shaper_png_path.name,
                shaper_svg_path.name,
            ).encode("utf-8"),
        )
        artifacts: dict[str, Any] = {
            "json": str(report_path),
            "svg": str(svg_path),
            "png": str(png_path),
            "html": str(html_path),
            "input_shaper_png": str(shaper_png_path),
            "input_shaper_svg": str(shaper_svg_path),
        }
        candidate_rows = self._candidate_csv(report)
        if len(candidate_rows) > 1:
            candidate_path = destination / "candidates.csv"
            _atomic_bytes(candidate_path, _csv_bytes(candidate_rows))
            artifacts["candidates_csv"] = str(candidate_path)
        validation_rows = self._validation_csv(report)
        if len(validation_rows) > 1:
            validation_path = destination / "validation.csv"
            _atomic_bytes(validation_path, _csv_bytes(validation_rows))
            artifacts["validation_csv"] = str(validation_path)
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
    def _candidate_csv(report: Mapping[str, Any]) -> list[list[Any]]:
        rows: list[list[Any]] = [[
            "axis", "candidate", "frequency_hz", "max_accel_mm_s2", "smoothing",
            "residual_vibration_fraction", "repeatability", "cross_axis_energy",
            "sensitivity", "residual_metric", "acceleration_evidence",
            "resonance_validated_accel_mm_s2", "print_validated_accel_mm_s2",
            "pareto", "selected",
        ]]
        for axis, details in _axes(report).items():
            if not isinstance(details, Mapping):
                continue
            selected = str(details.get("selected", "")).lower()
            pareto = {str(item).lower() for item in details.get("pareto", [])}
            for candidate in details.get("candidates", []):
                if not isinstance(candidate, Mapping):
                    continue
                name = str(candidate.get("name", ""))
                metadata = candidate.get("metadata", {})
                metadata = metadata if isinstance(metadata, Mapping) else {}
                rows.append([
                    axis, name, candidate.get("frequency", ""), candidate.get("max_accel", ""),
                    candidate.get("smoothing", ""), candidate.get("residual_vibration", ""),
                    candidate.get("repeatability", ""), candidate.get("cross_axis_energy", ""),
                    candidate.get("sensitivity", ""), metadata.get("residual_metric", ""),
                    metadata.get("acceleration_evidence", ""),
                    metadata.get("resonance_validated_acceleration_mm_s2", ""),
                    metadata.get("print_validated_acceleration_mm_s2", ""),
                    str(name.lower() in pareto).lower(),
                    str(name.lower() == selected).lower(),
                ])
        return rows

    @staticmethod
    def _validation_csv(report: Mapping[str, Any]) -> list[list[Any]]:
        rows: list[list[Any]] = [[
            "axis", "baseline_energy", "shaped_energy", "attenuation_ci95_low",
            "attenuation_ci95_high", "reference_cross_axis_energy",
            "candidate_cross_axis_energy", "cross_axis_regression", "qc_passed", "passed",
        ]]
        for axis, values in _validation_axes(report).items():
            if not isinstance(values, Mapping):
                continue
            ci = values.get("improvement_ci_95", ["", ""])
            if not isinstance(ci, Sequence) or isinstance(ci, (str, bytes)):
                ci = ["", ""]
            rows.append([
                axis, values.get("baseline_energy", ""), values.get("shaped_energy", ""),
                ci[0] if len(ci) else "", ci[1] if len(ci) > 1 else "",
                values.get("reference_cross_axis_energy", ""),
                values.get("candidate_cross_axis_energy", ""),
                values.get("cross_axis_regression", ""), values.get("qc_passed", ""),
                values.get("passed", ""),
            ])
        return rows

    @staticmethod
    def _png(report: Mapping[str, Any], path: Path) -> None:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        plt.rcParams.update({
            "font.size": 9, "axes.titleweight": "bold", "axes.titlesize": 12,
            "axes.labelcolor": INK, "axes.edgecolor": "#cbd5e1", "text.color": INK,
            "xtick.color": "#475569", "ytick.color": "#475569", "figure.facecolor": "white",
            "axes.facecolor": "#f8fafc", "grid.color": "#e2e8f0", "grid.linewidth": 0.7,
        })
        figure, plots = plt.subplots(3, 1, figsize=(12, 13), constrained_layout=True)
        figure.suptitle("Klipper Advanced Shaper · Technical calibration summary", fontsize=17,
                        fontweight="bold", x=0.06, ha="left")

        spectra = []
        for axis, details in _axes(report).items():
            if not isinstance(details, Mapping):
                continue
            spectrum = details.get("spectrum", {})
            if isinstance(spectrum, Mapping):
                frequencies = np.asarray(spectrum.get("frequency_hz", []), dtype=float)
                psd = np.asarray(spectrum.get("psd", []), dtype=float)
                if len(frequencies) and len(frequencies) == len(psd):
                    spectra.append(
                        (
                            str(axis),
                            frequencies,
                            psd,
                            details.get("modes", []),
                            _plot_frequency_ceiling(details, frequencies),
                        )
                    )
        for index, (axis, frequencies, psd, modes, _ceiling) in enumerate(spectra):
            color = BLUE if index == 0 else ORANGE
            plots[0].semilogy(frequencies, np.maximum(psd, 1e-18), color=color, linewidth=1.8,
                              label=f"{axis} robust spectrum")
            for mode in modes[:3]:
                frequency = _finite(mode.get("frequency")) if isinstance(mode, Mapping) else None
                if frequency is not None:
                    plots[0].axvline(frequency, color=color, linestyle="--", linewidth=0.9, alpha=0.7)
                    plots[0].annotate(f"{axis} {frequency:.1f} Hz", (frequency, 0.96),
                                      xycoords=("data", "axes fraction"), rotation=90,
                                      va="top", ha="right", color=color, fontsize=8)
        plots[0].set(title="PSD and identified modal evidence", xlabel="Frequency (Hz)",
                     ylabel="Power spectral density (acceleration²/Hz)")
        plots[0].grid(True, which="both", axis="y")
        if spectra:
            ceilings = [item[4] for item in spectra if item[4] is not None]
            if ceilings:
                plots[0].set_xlim(0.0, max(ceilings))
            plots[0].legend(frameon=False, loc="upper right")
        else:
            plots[0].text(0.5, 0.5, "Spectrum not available", ha="center", va="center",
                          transform=plots[0].transAxes, color=SLATE)

        labels, accelerations, colors, hatches = [], [], [], []
        hidden_candidates = 0
        for axis, details in _axes(report).items():
            if not isinstance(details, Mapping):
                continue
            selected = str(details.get("selected", "")).lower()
            candidates = details.get("candidates", [])
            candidates = (
                candidates
                if isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes))
                else []
            )
            visible_candidates = _plot_candidate_subset(details)
            hidden_candidates += max(0, len(candidates) - len(visible_candidates))
            for candidate in visible_candidates:
                name = str(candidate.get("name", ""))
                is_selected = name.lower() == selected
                labels.append(_plot_candidate_label(str(axis), name))
                accelerations.append(float(candidate["max_accel"]))
                colors.append(BLUE if is_selected else "#cbd5e1")
                hatches.append("" if is_selected else "//")
        bars = plots[1].bar(labels or ["No candidates"], accelerations or [0.0], color=colors or ["#cbd5e1"],
                            edgecolor="#64748b", linewidth=0.6)
        for bar, hatch, value in zip(bars, hatches, accelerations):
            bar.set_hatch(hatch)
            plots[1].text(bar.get_x() + bar.get_width() / 2, value, f"{value:,.0f}",
                          ha="center", va="bottom", fontsize=7)
        omitted = (
            f" · {hidden_candidates} additional candidates in candidates.csv"
            if hidden_candidates
            else ""
        )
        plots[1].set(
            title="Selected and highest theoretical-acceleration candidates" + omitted,
            ylabel="Smoothing-derived max accel (mm/s²)",
        )
        plots[1].tick_params(axis="x", rotation=18, labelsize=8)
        plots[1].grid(True, axis="y")

        validation = _validation_axes(report)
        labels, before, after = [], [], []
        for axis, values in validation.items():
            if not isinstance(values, Mapping):
                continue
            base, shaped = _finite(values.get("baseline_energy")), _finite(values.get("shaped_energy"))
            if base is not None and shaped is not None:
                labels.append(str(axis))
                before.append(base)
                after.append(shaped)
        positions = np.arange(len(labels) or 1)
        plots[2].bar(positions - 0.18, before or [0.0], 0.36, label="Held-out reference",
                     color=ORANGE, edgecolor="#9a3412")
        plots[2].bar(positions + 0.18, after or [0.0], 0.36, label="Shaped candidate",
                     color=BLUE, edgecolor="#1e40af", hatch="//")
        plots[2].set_xticks(positions, labels or ["Validation pending"])
        plots[2].set(title="Held-out validation · lower resonant-band energy is better",
                     ylabel="Integrated resonant-band energy (acceleration²)")
        plots[2].grid(True, axis="y")
        if labels:
            plots[2].legend(frameon=False)
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".png", delete=False) as stream:
            temporary = Path(stream.name)
        try:
            figure.savefig(temporary, dpi=170, metadata={"Software": "Klipper Advanced Shaper"})
            plt.close(figure)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def _input_shaper_plots(report: Mapping[str, Any], png_path: Path, svg_path: Path) -> None:
        """Emit a familiar Klipper-style spectrum without inventing unavailable traces."""
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        plt.rcParams.update({
            "font.size": 9,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#cbd5e1",
            "axes.facecolor": "#f8fafc",
            "figure.facecolor": "white",
            "grid.color": "#e2e8f0",
            "text.color": INK,
        })
        axes_data = [
            (str(axis), details)
            for axis, details in _axes(report).items()
            if isinstance(details, Mapping)
        ]
        axis_count = max(1, len(axes_data))
        figure, plots = plt.subplots(
            axis_count * 2,
            1,
            figsize=(12, max(8.5, 7.5 * axis_count)),
            squeeze=False,
            gridspec_kw={"height_ratios": [2.0, 1.1] * axis_count},
        )
        figure.suptitle(
            "INPUT SHAPER CALIBRATION TOOL · ADVANCED SHAPER",
            fontsize=16,
            fontweight="bold",
            x=0.06,
            ha="left",
        )
        trace_spec = (
            ("psd_sum", "X+Y+Z", "purple", "-", 1.8),
            ("psd_x", "X", "#dc2626", "-", 1.25),
            ("psd_y", "Y", "#16a34a", "-", 1.25),
            ("psd_z", "Z", "#2563eb", "-", 1.25),
            ("psd_projected", "Excitation-axis projected", GOLD, "-.", 1.5),
        )
        for row, (axis, details) in enumerate(axes_data or [("—", {})]):
            plot = plots[row * 2][0]
            spectrogram_plot = plots[row * 2 + 1][0]
            spectrum = details.get("native_spectrum", details.get("spectrum", {}))
            spectrum = spectrum if isinstance(spectrum, Mapping) else {}
            frequencies = np.asarray(
                spectrum.get("frequency_hz", spectrum.get("freq_bins", [])), dtype=float
            )
            frequency_ceiling = _plot_frequency_ceiling(details, frequencies)
            traces = 0
            primary_psd = None
            for key, label, color, linestyle, linewidth in trace_spec:
                values = np.asarray(spectrum.get(key, []), dtype=float)
                if len(frequencies) and len(values) == len(frequencies):
                    plot.plot(
                        frequencies,
                        values,
                        label=label,
                        color=color,
                        linestyle=linestyle,
                        linewidth=linewidth,
                    )
                    if key == "psd_sum":
                        primary_psd = values
                    traces += 1
            fallback = np.asarray(spectrum.get("psd", []), dtype=float)
            if traces == 0 and len(frequencies) and len(fallback) == len(frequencies):
                plot.plot(
                    frequencies,
                    fallback,
                    label=f"{axis} projected / aggregate response",
                    color="purple",
                    linewidth=1.8,
                )
                primary_psd = fallback
                traces = 1
            response_plot = None
            response_lines = 0
            response_colors = [BLUE, ORANGE, GOLD, "#7c3aed", SLATE]
            for index, native_candidate in enumerate(details.get("native_candidates", [])):
                if not isinstance(native_candidate, Mapping):
                    continue
                response = native_candidate.get("native_frequency_response", {})
                if not isinstance(response, Mapping):
                    continue
                response_frequency = np.asarray(response.get("frequency_hz", []), dtype=float)
                response_ratio = np.asarray(response.get("response_ratio", []), dtype=float)
                if not len(response_frequency) or len(response_frequency) != len(response_ratio):
                    continue
                if response_plot is None:
                    response_plot = plot.twinx()
                    response_plot.set_ylabel("Shaper vibration reduction (ratio)")
                    response_plot.set_ylim(bottom=0)
                name = str(native_candidate.get("name", ""))
                frequency = native_candidate.get("frequency_hz")
                is_selected = name.lower() == str(details.get("selected", "")).lower()
                metrics = (
                    f"{name.upper()} ({_number(frequency, 1)} Hz, "
                    f"vibr={_percent(native_candidate.get('residual_vibration'), 1)}, "
                    f"sm~={_number(native_candidate.get('smoothing'), 3)}, "
                    f"accel<={_number(native_candidate.get('max_accel'), 0)})"
                )
                response_plot.plot(
                    response_frequency,
                    response_ratio,
                    linestyle=":" if not is_selected else "-.",
                    linewidth=2.0 if is_selected else 1.2,
                    color=response_colors[index % len(response_colors)],
                    label=metrics,
                )
                response_lines += 1
            selected_candidate = _selected_candidate(details)
            selected_name = str(details.get("selected", "No selection"))
            selected_response = _selected_modeled_response(details, frequencies)
            native_names = {
                str(candidate.get("name", "")).lower()
                for candidate in details.get("native_candidates", [])
                if isinstance(candidate, Mapping)
            }
            if selected_response is not None and selected_name.lower() not in native_names:
                if response_plot is None:
                    response_plot = plot.twinx()
                    response_plot.set_ylabel("Shaper vibration reduction (ratio)")
                    response_plot.set_ylim(0.0, 1.0)
                response_plot.plot(
                    frequencies,
                    selected_response,
                    linestyle="-.",
                    linewidth=2.2,
                    color="#06b6d4",
                    label=(
                        f"SELECTED {selected_name} "
                        f"({_number(selected_candidate.get('frequency'), 1)} Hz, "
                        f"vibr={_percent(selected_candidate.get('residual_vibration'), 1)}, "
                        f"sm~={_number(selected_candidate.get('smoothing'), 3)}, "
                        f"accel<={_number(selected_candidate.get('max_accel'), 0)})"
                    ),
                )
                response_lines += 1
            if (
                primary_psd is not None
                and selected_response is not None
                and len(primary_psd) == len(selected_response)
            ):
                plot.plot(
                    frequencies,
                    primary_psd * selected_response,
                    color="#06b6d4",
                    linewidth=1.6,
                    label=f"With {selected_name} applied",
                )
            for index, mode in enumerate(details.get("modes", []), start=1):
                frequency = _finite(mode.get("frequency")) if isinstance(mode, Mapping) else None
                if frequency is None:
                    continue
                peak_height = (
                    float(np.interp(frequency, frequencies, primary_psd))
                    if primary_psd is not None and len(primary_psd) == len(frequencies)
                    else 0.0
                )
                plot.plot(frequency, peak_height, "x", color="black", markersize=7)
                plot.annotate(
                    str(index),
                    (frequency, peak_height),
                    textcoords="offset points",
                    xytext=(7, 5),
                    color="#ef4444",
                    fontsize=11,
                    fontweight="bold",
                )
            metadata = selected_candidate.get("metadata", {})
            metadata = metadata if isinstance(metadata, Mapping) else {}
            primary_mode = next(
                (
                    _finite(mode.get("frequency"))
                    for mode in details.get("modes", [])
                    if isinstance(mode, Mapping) and _finite(mode.get("frequency")) is not None
                ),
                None,
            )
            title_context = (
                f"ω0={_number(primary_mode, 1)} Hz, "
                f"ζ={_number(metadata.get('design_damping_ratio'), 3)}"
            )
            plot.set(
                title=f"Axis {axis} Frequency Profile ({title_context})",
                xlabel="Frequency (Hz)",
                ylabel="Power spectral density",
            )
            plot.set_ylim(bottom=0.0)
            plot.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
            plot.minorticks_on()
            plot.grid(True, which="major", color="#94a3b8", linewidth=0.7)
            plot.grid(True, which="minor", color="#e2e8f0", linewidth=0.5)
            if frequency_ceiling is not None:
                plot.set_xlim(0.0, frequency_ceiling)
            if traces:
                plot.legend(frameon=True, loc="upper left", fontsize=7)
            else:
                plot.text(
                    0.5,
                    0.55,
                    "Measured component spectral response is unavailable in this artifact.",
                    transform=plot.transAxes,
                    ha="center",
                    color=SLATE,
                )
            if response_plot is not None and response_lines:
                if frequency_ceiling is not None:
                    response_plot.set_xlim(0.0, frequency_ceiling)
                response_plot.set_ylim(0.0, 1.0)
                response_plot.legend(frameon=True, loc="upper right", fontsize=7)

            spectrogram = details.get("spectrogram", {})
            spectrogram = spectrogram if isinstance(spectrogram, Mapping) else {}
            times = np.asarray(spectrogram.get("time_s", []), dtype=float)
            spectrogram_frequency = np.asarray(spectrogram.get("frequency_hz", []), dtype=float)
            power = np.asarray(spectrogram.get("power", []), dtype=float)
            if (
                bool(spectrogram.get("available"))
                and power.ndim == 2
                and power.shape == (len(spectrogram_frequency), len(times))
                and power.size
            ):
                decibels = 10.0 * np.log10(np.maximum(power, 1e-18))
                image = spectrogram_plot.pcolormesh(
                    spectrogram_frequency,
                    times,
                    decibels.T,
                    shading="auto",
                    cmap="inferno",
                )
                figure.colorbar(image, ax=spectrogram_plot, pad=0.01, label="Power (dB)")
                for index, mode in enumerate(details.get("modes", []), start=1):
                    frequency = _finite(mode.get("frequency")) if isinstance(mode, Mapping) else None
                    if frequency is None:
                        continue
                    spectrogram_plot.axvline(
                        frequency, color="#22d3ee", linestyle=":", linewidth=0.9
                    )
                    spectrogram_plot.annotate(
                        f"Peak {index} ({frequency:.1f} Hz)",
                        (frequency, times[-1] * 0.92),
                        color="#22d3ee",
                        rotation=90,
                        ha="right",
                        va="top",
                        fontsize=7,
                    )
                if frequency_ceiling is not None:
                    spectrogram_plot.set_xlim(0.0, frequency_ceiling)
                spectrogram_plot.set(
                    title=f"Axis {axis} Time-Frequency Spectrogram",
                    xlabel="Frequency (Hz)",
                    ylabel="Time (s)",
                )
            else:
                reason = spectrogram.get("reason", "not present in this artifact")
                spectrogram_plot.text(
                    0.5,
                    0.54,
                    "Time–frequency evidence unavailable\n" + str(reason),
                    transform=spectrogram_plot.transAxes,
                    ha="center",
                    va="center",
                    color=SLATE,
                )
                spectrogram_plot.set(
                    title=f"Axis {axis} time–frequency evidence",
                    xlabel="Sweep time (s)",
                    ylabel="Frequency (Hz)",
                )
        figure.text(
            0.06,
            0.005,
            "Only measured traces are drawn; unavailable components are not synthesized. "
            "Complete candidate metrics are preserved in candidates.csv.",
            color=SLATE,
            fontsize=8,
        )
        figure.tight_layout(rect=(0, 0.025, 1, 0.95))
        temporary_paths = []
        try:
            for destination, file_format in ((png_path, "png"), (svg_path, "svg")):
                with tempfile.NamedTemporaryFile(
                    dir=destination.parent, suffix=f".{file_format}", delete=False
                ) as stream:
                    temporary = Path(stream.name)
                temporary_paths.append(temporary)
                figure.savefig(
                    temporary,
                    format=file_format,
                    dpi=170 if file_format == "png" else None,
                    metadata=(
                        {"Software": "Klipper Advanced Shaper"}
                        if file_format == "png"
                        else {"Creator": "Klipper Advanced Shaper"}
                    ),
                )
                os.replace(temporary, destination)
                temporary_paths.remove(temporary)
        finally:
            plt.close(figure)
            for temporary in temporary_paths:
                if temporary.exists():
                    temporary.unlink()

    @staticmethod
    def _svg(report: Mapping[str, Any]) -> str:
        title, reason, tone = _status(report)
        tone_color = {"accepted": BLUE, "rejected": ORANGE, "pending": GOLD}[tone]
        cards = []
        for axis, details in _axes(report).items():
            if not isinstance(details, Mapping):
                continue
            candidate = _selected_candidate(details)
            selected = str(details.get("selected", "No selection"))
            frequency = candidate.get("frequency")
            cards.append((str(axis), selected, _number(frequency, 1, "—") + " Hz",
                          _number(candidate.get("max_accel"), 0, "—") + " mm/s²"))
        width, height = 1200, 330 + max(1, len(cards)) * 105
        card_markup = []
        for index, (axis, selected, frequency, accel) in enumerate(cards):
            y = 285 + index * 105
            card_markup.append(
                f'<rect x="48" y="{y}" width="1104" height="82" rx="12" fill="#f8fafc" stroke="#dbe3ee"/>'
                f'<text x="72" y="{y + 33}" class="eyebrow">AXIS {_escape(axis)}</text>'
                f'<text x="72" y="{y + 61}" class="value">{_escape(selected)} · {_escape(frequency)}</text>'
                f'<text x="1125" y="{y + 52}" class="accel" text-anchor="end">{_escape(accel)}</text>'
            )
        if not card_markup:
            card_markup.append('<text x="600" y="325" text-anchor="middle" class="muted">Candidate details are not available.</text>')
        return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
<title id="title">Klipper Advanced Shaper calibration summary</title><desc id="desc">{_escape(title)}. {_escape(reason)}</desc>
<style>.kicker{{font:700 15px system-ui,sans-serif;letter-spacing:1.5px;fill:#64748b}}.status{{font:800 34px system-ui,sans-serif;fill:{tone_color}}}.reason{{font:400 17px system-ui,sans-serif;fill:#334155}}.eyebrow{{font:700 13px system-ui,sans-serif;letter-spacing:1px;fill:#64748b}}.value{{font:750 23px system-ui,sans-serif;fill:{INK}}}.accel{{font:700 20px system-ui,sans-serif;fill:{BLUE}}}.muted{{font:400 18px system-ui,sans-serif;fill:#64748b}}</style>
<rect width="1200" height="{height}" fill="#ffffff"/><rect x="0" y="0" width="14" height="{height}" fill="{tone_color}"/>
<text x="48" y="55" class="kicker">KLIPPER ADVANCED SHAPER · TECHNICAL CALIBRATION</text>
<text x="48" y="105" class="status">{_escape(title)}</text>
<foreignObject x="48" y="128" width="1104" height="105"><div xmlns="http://www.w3.org/1999/xhtml" style="font:17px system-ui,sans-serif;color:#334155;line-height:1.45">{_escape(reason)}</div></foreignObject>
{''.join(card_markup)}</svg>'''

    @staticmethod
    def _html(
        report: Mapping[str, Any],
        svg_name: str,
        png_name: str,
        shaper_png_name: str,
        shaper_svg_name: str,
    ) -> str:
        title, reason, tone = _status(report)
        validation = report.get("validation", {})
        validation = validation if isinstance(validation, Mapping) else {}
        axis_sections = []
        findings = []
        validation_rows = []
        candidate_rows = []
        cards = []
        protocol = report.get("validation_protocol", {})
        protocol = protocol if isinstance(protocol, Mapping) else {}
        if protocol:
            if protocol.get("mode") == "fast_lower_confidence_2_repeat":
                protocol_label = "Lower-confidence fast validation"
            elif protocol.get("lower_confidence"):
                protocol_label = "Lower-confidence validation"
            else:
                protocol_label = "Full-confidence validation"
            motion_seconds = _finite(protocol.get("estimated_motion_seconds_per_axis"))
            motion_minutes = motion_seconds / 60.0 if motion_seconds is not None else None
            cards.append(
                f'''<article class="metric"><span>Validation protocol</span><strong>{_escape(protocol_label)}</strong><small>{_escape(protocol.get("repeats_per_group", "-"))} repeats/group | {_number(motion_minutes, 1)} min estimated motion/axis</small></article>'''
            )
            findings.append(
                f"Validation protocol: <strong>{_escape(protocol_label)}</strong>; "
                "the motion estimate excludes host analysis and artifact time."
            )
        for axis, details in _axes(report).items():
            if not isinstance(details, Mapping):
                continue
            candidate = _selected_candidate(details)
            selected = str(details.get("selected", "No selection"))
            frequency = candidate.get("frequency")
            max_accel = candidate.get("max_accel")
            acceleration_limits = details.get("acceleration_limits", {})
            if not isinstance(acceleration_limits, Mapping):
                acceleration_limits = {}
            resonance_accel = acceleration_limits.get("resonance_validated_mm_s2")
            print_accel = acceleration_limits.get("print_validated_mm_s2")
            modes = [item for item in details.get("modes", []) if isinstance(item, Mapping)]
            mode_text = ", ".join(f"{_number(item.get('frequency'), 1)} Hz" for item in modes[:4]) or "None identified"
            cards.append(f'''<article class="metric"><span>Axis {_escape(axis)} selection</span><strong>{_escape(selected)}</strong><small>{_number(frequency, 1)} Hz · {_number(max_accel, 0)} mm/s²</small></article>''')
            findings.append(f"Axis {_escape(axis)} selected <strong>{_escape(selected)}</strong> at {_number(frequency, 1)} Hz from the allowlisted candidate set.")
            qc_rows = details.get("qc", [])
            qc_pass = bool(qc_rows) and all(bool(row.get("passed")) for row in qc_rows if isinstance(row, Mapping))
            axis_sections.append(f'''<article class="axis-card"><div><span class="eyebrow">AXIS {_escape(axis)}</span><h3>{_escape(selected)} · {_number(frequency, 1)} Hz</h3></div><dl><div><dt>Theoretical smoothing accel</dt><dd>{_number(max_accel, 0)} mm/s²</dd></div><div><dt>Resonance-validated accel</dt><dd>{_number(resonance_accel, 0, "Not validated")}</dd></div><div><dt>Print-validated accel</dt><dd>{_number(print_accel, 0, "Not validated")}</dd></div><div><dt>Residual vibration</dt><dd>{_percent(candidate.get("residual_vibration"), 2)}</dd></div><div><dt>Smoothing</dt><dd>{_number(candidate.get("smoothing"), 4)}</dd></div><div><dt>QC</dt><dd>{"Passed" if qc_pass else "Review details"}</dd></div><div><dt>Modes</dt><dd>{mode_text}</dd></div></dl></article>''')
            pareto = {str(item).lower() for item in details.get("pareto", [])}
            for item in details.get("candidates", []):
                if not isinstance(item, Mapping):
                    continue
                name = str(item.get("name", ""))
                selected_mark = "<span class='tag selected'>Selected</span>" if name.lower() == str(details.get("selected", "")).lower() else ""
                frontier_mark = "<span class='tag'>Pareto</span>" if name.lower() in pareto else ""
                candidate_rows.append(f'''<tr><th scope="row">{_escape(axis)} · {_escape(name)} {selected_mark}{frontier_mark}</th><td>{_number(item.get("frequency"), 1)}</td><td>{_number(item.get("max_accel"), 0)}</td><td>{_percent(item.get("residual_vibration"), 2)}</td><td>{_number(item.get("smoothing"), 4)}</td><td>{_percent(item.get("repeatability"), 2)}</td><td>{_percent(item.get("cross_axis_energy"), 2)}</td><td>{_percent(item.get("sensitivity"), 2)}</td></tr>''')

        for axis, values in _validation_axes(report).items():
            if not isinstance(values, Mapping):
                continue
            ci = values.get("improvement_ci_95", [])
            low = ci[0] if isinstance(ci, Sequence) and not isinstance(ci, (str, bytes)) and len(ci) else None
            high = ci[1] if isinstance(ci, Sequence) and not isinstance(ci, (str, bytes)) and len(ci) > 1 else None
            passed = values.get("passed")
            gate_label = "Passed" if passed is True else "Failed" if passed is False else "Pending"
            gate_class = "accepted" if passed is True else "rejected" if passed is False else "pending"
            validation_rows.append(f'''<tr><th scope="row">{_escape(axis)}</th><td>{_number(values.get("baseline_energy"), 4)}</td><td>{_number(values.get("shaped_energy"), 4)}</td><td>{_percent(low)} to {_percent(high)}</td><td>{_percent(values.get("cross_axis_regression"))}</td><td><span class="tag {gate_class}">{gate_label}</span></td></tr>''')
            cards.append(f'''<article class="metric"><span>Axis {_escape(axis)} attenuation, 95% CI</span><strong>{_percent(low)} to {_percent(high)}</strong><small>Required lower bound: 10.0%</small></article>''')
            cards.append(f'''<article class="metric"><span>Axis {_escape(axis)} cross-axis change</span><strong>{_percent(values.get("cross_axis_regression"))}</strong><small>Maximum permitted regression: 5.0%</small></article>''')

        qc_total = qc_passed = 0
        for details in _axes(report).values():
            if not isinstance(details, Mapping):
                continue
            for row in details.get("qc", []):
                if isinstance(row, Mapping):
                    qc_total += 1
                    qc_passed += int(bool(row.get("passed")))
        cards.append(f'''<article class="metric"><span>Training capture QC</span><strong>{qc_passed}/{qc_total or "—"}</strong><small>Captures passing data-quality checks</small></article>''')
        if tone == "rejected":
            findings.insert(0, f"<strong>Do not apply or stage this attempt.</strong> {_escape(reason)}")
        elif tone == "accepted":
            findings.insert(0, "The statistical acceptance gates passed; the result is ready for operator review.")
        else:
            findings.insert(0, "No final acceptance decision is present; treat all recommendations as pending.")

        audit_items = []
        for label, key in (("Attempt", "attempt_id"), ("Schema", "schema_version"), ("Plugin", "plugin_version"), ("Engine", "engine"), ("Profile", "profile")):
            value = report.get(key)
            if value not in (None, ""):
                audit_items.append(f"<div><dt>{label}</dt><dd>{_escape(value)}</dd></div>")
        if protocol:
            audit_items.append(
                "<div><dt>Validation</dt><dd>%s</dd></div>"
                % _escape(protocol.get("mode", "unknown"))
            )
        payload = html.escape(json.dumps(report, indent=2, sort_keys=True), quote=False)
        validation_body = "".join(validation_rows) or '<tr><td colspan="6" class="empty">Held-out validation is not available for this report.</td></tr>'
        candidate_body = "".join(candidate_rows) or '<tr><td colspan="8" class="empty">Candidate metrics are not available.</td></tr>'
        axis_body = "".join(axis_sections) or '<p class="empty">No axis selection is available.</p>'
        findings_body = "".join(f"<li>{item}</li>" for item in findings)
        cards_body = "".join(cards)
        audit_body = "".join(audit_items) or "<div><dt>Audit data</dt><dd>Not available</dd></div>"
        native_preview = report.get("native_command_preview")
        next_action = (
            "This attempt is retained for diagnosis only. Correct the failing gate or data-quality issue, then run a new matched calibration."
            if tone == "rejected" else
            "Review the held-out evidence and mechanical limits. APPLY changes the runtime only; STAGE still requires a separate SAVE_CONFIG."
            if tone == "accepted" else
            "Complete held-out validation before applying or staging any candidate."
        )
        command_block = f'<pre class="command"><code>{_escape(native_preview)}</code></pre>' if native_preview and tone != "rejected" else ""
        return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{_escape(title)} · Advanced Shaper</title>
<style>
:root{{--bg:#f2f5f9;--surface:#fff;--surface2:#f8fafc;--ink:#172033;--muted:#5d6b7d;--line:#dbe3ee;--blue:{BLUE};--orange:{ORANGE};--gold:{GOLD};--shadow:0 12px 35px rgba(23,32,51,.08)}}
@media(prefers-color-scheme:dark){{:root{{--bg:#0b1220;--surface:#121c2d;--surface2:#182438;--ink:#edf4ff;--muted:#a8b7ca;--line:#2d3d54;--blue:#60a5fa;--orange:#fb923c;--gold:#facc15;--shadow:none}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.58 system-ui,-apple-system,Segoe UI,sans-serif}}main{{max-width:1180px;margin:auto;padding:40px 24px 80px}}h1,h2,h3,p{{margin-top:0}}h1{{font-size:clamp(32px,6vw,60px);line-height:1.02;letter-spacing:-.04em;margin-bottom:18px}}h2{{font-size:25px;letter-spacing:-.02em;margin:52px 0 18px}}h3{{font-size:21px;margin:5px 0}}.eyebrow{{font-size:12px;font-weight:800;letter-spacing:.13em;color:var(--muted)}}.hero{{border-top:8px solid var(--gold);background:var(--surface);padding:38px;border-radius:18px;box-shadow:var(--shadow)}}.hero.accepted{{border-color:var(--blue)}}.hero.rejected{{border-color:var(--orange)}}.status{{display:inline-flex;border:1px solid currentColor;border-radius:99px;padding:5px 11px;font-size:12px;font-weight:850;letter-spacing:.1em;color:var(--gold);margin-bottom:22px}}.hero.accepted .status{{color:var(--blue)}}.hero.rejected .status{{color:var(--orange)}}.lede{{font-size:18px;max-width:850px;color:var(--muted)}}.warning{{margin-top:22px;padding:16px 18px;border-left:4px solid var(--orange);background:var(--surface2);font-weight:650}}.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin-top:26px}}.metric{{background:var(--surface2);border:1px solid var(--line);border-radius:12px;padding:16px;min-height:116px}}.metric span,.metric small{{display:block;color:var(--muted)}}.metric strong{{display:block;font-size:23px;line-height:1.2;margin:8px 0}}.findings{{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:22px 28px;box-shadow:var(--shadow)}}.findings li+li{{margin-top:9px}}.axis-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:15px}}.axis-card{{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:22px;box-shadow:var(--shadow)}}dl{{margin:15px 0 0}}dl div{{display:flex;justify-content:space-between;gap:18px;border-top:1px solid var(--line);padding:8px 0}}dt{{color:var(--muted)}}dd{{margin:0;text-align:right;font-weight:650}}.figure{{background:#fff;border:1px solid var(--line);border-radius:14px;padding:12px;overflow:hidden}}.figure img{{display:block;width:100%;height:auto}}.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:14px;background:var(--surface)}}table{{width:100%;border-collapse:collapse;white-space:nowrap}}th,td{{padding:12px 14px;text-align:right;border-bottom:1px solid var(--line)}}th:first-child,td:first-child{{text-align:left}}thead th{{font-size:12px;letter-spacing:.04em;color:var(--muted);background:var(--surface2)}}tbody tr:last-child>*{{border-bottom:0}}.tag{{display:inline-block;padding:2px 7px;border:1px solid var(--line);border-radius:99px;font-size:10px;text-transform:uppercase;letter-spacing:.06em;margin-left:4px}}.tag.selected,.tag.accepted{{color:var(--blue);border-color:var(--blue)}}.tag.rejected{{color:var(--orange);border-color:var(--orange)}}.tag.pending{{color:var(--gold);border-color:var(--gold)}}.empty{{color:var(--muted);text-align:center!important;padding:30px}}.method{{display:grid;grid-template-columns:1.2fr .8fr;gap:18px}}.panel{{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:24px}}.audit div{{display:grid;grid-template-columns:90px 1fr;gap:12px}}.audit dd{{overflow-wrap:anywhere}}code{{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}}.command{{white-space:pre-wrap;background:var(--surface2);border:1px solid var(--line);border-radius:9px;padding:14px;overflow:auto}}details{{margin-top:20px}}summary{{cursor:pointer;font-weight:700}}details pre{{max-height:420px;overflow:auto;background:#0f172a;color:#dbeafe;padding:16px;border-radius:9px;font-size:12px}}.next{{border-left:5px solid var(--blue)}}footer{{margin-top:50px;color:var(--muted);font-size:13px}}@media(max-width:700px){{main{{padding:20px 12px 50px}}.hero{{padding:24px}}.method{{grid-template-columns:1fr}}}}@media print{{:root{{--bg:#fff;--surface:#fff;--surface2:#fff;--ink:#000;--muted:#444;--line:#bbb}}body{{font-size:11px}}main{{max-width:none;padding:0}}.hero,.findings,.axis-card,.panel{{box-shadow:none;break-inside:avoid}}h2{{break-after:avoid;margin-top:28px}}details{{display:none}}.figure{{break-inside:avoid}}}}
</style></head><body><main>
<header class="hero {tone}"><span class="status">{_escape(title)}</span><div class="eyebrow">KLIPPER ADVANCED SHAPER · TECHNICAL CALIBRATION REPORT</div><h1>Measured shaping evidence,<br>ready for a decision.</h1><p class="lede">{_escape(reason)}</p>{'<div class="warning">This result is rejected and cannot be applied or staged.</div>' if tone == 'rejected' else ''}<div class="metrics">{cards_body}</div></header>
<section aria-labelledby="findings"><h2 id="findings">Technical summary</h2><ol class="findings">{findings_body}</ol></section>
<section aria-labelledby="axis"><h2 id="axis">Selection and modal evidence</h2><div class="axis-grid">{axis_body}</div></section>
<section aria-labelledby="spectrum"><h2 id="spectrum">Input shaper spectrum</h2><p>Klipper-compatible frequency view of the normalized spectral response and evaluated allowlisted candidates. Native component values are unitless normalized or arbitrary response units, not acceleration²/Hz. <a href="{_escape(shaper_svg_name)}">Open the scalable SVG</a>.</p><div class="figure"><img src="{_escape(shaper_png_name)}" alt="Input shaper frequency profile with normalized spectral response traces, detected modal peaks, and allowlisted candidate metrics"></div></section>
<section aria-labelledby="validation"><h2 id="validation">Before / after held-out validation</h2><div class="table-wrap"><table><thead><tr><th>Axis</th><th>Reference energy</th><th>Shaped energy</th><th>Attenuation 95% CI</th><th>Cross-axis change</th><th>Gate</th></tr></thead><tbody>{validation_body}</tbody></table></div></section>
<section aria-labelledby="plots"><h2 id="plots">PSD, candidates, and validation plots</h2><div class="figure"><img src="{_escape(png_name)}" alt="Three technical plots showing modal spectrum, candidate acceleration estimates, and held-out reference versus shaped energy"></div></section>
<section aria-labelledby="candidates"><h2 id="candidates">Candidate and Pareto comparison</h2><div class="table-wrap"><table><thead><tr><th>Candidate</th><th>Hz</th><th>Max accel mm/s²</th><th>Residual</th><th>Smoothing</th><th>Repeatability</th><th>Cross-axis</th><th>Sensitivity</th></tr></thead><tbody>{candidate_body}</tbody></table></div></section>
<section aria-labelledby="method"><h2 id="method">Definitions, method, and uncertainty</h2><div class="method"><article class="panel"><h3>How to read this report</h3><p><strong>Max accel</strong> is Klipper's smoothing-derived theoretical estimate at the captured square-corner velocity; it is not a mechanical safety rating. <strong>Attenuation</strong> compares independent held-out resonant-band energy before and after shaping. The lower bound of its 95% bootstrap confidence interval must reach 10%. <strong>Cross-axis change</strong> must not regress by more than 5%.</p><p>Candidate scores combine residual vibration, smoothing, repeatability, cross-axis energy, and sensitivity. Pareto candidates are non-dominated trade-offs; the selected profile chooses among them.</p></article><article class="panel"><h3>Audit metadata</h3><dl class="audit">{audit_body}</dl></article></div></section>
<section aria-labelledby="limits"><h2 id="limits">Limitations</h2><div class="panel"><p>Results apply only to the tested sensor mount, toolhead mass, belt state, temperature, fan state, excitation, square-corner velocity, and sampling quality. A smoothing-derived acceleration estimate does not prove stepper torque, frame, hotend flow, or print-quality capability.</p><details><summary>Show exact machine-readable report</summary><pre>{payload}</pre></details></div></section>
<section aria-labelledby="action"><h2 id="action">Next action</h2><div class="panel next"><p>{_escape(next_action)}</p>{command_block}</div></section>
<footer>Generated locally and designed for offline review. No external assets, scripts, fonts, or network requests are used. Raw captures remain private and are not exported to CSV.</footer>
</main></body></html>'''
