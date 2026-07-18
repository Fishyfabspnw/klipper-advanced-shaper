"""Welch auto/cross spectra and frequency-response estimates."""

from __future__ import annotations

from typing import Iterator, Sequence, Tuple

import numpy as np

from .models import Spectrum, TransferMetrics


def _segments(x: np.ndarray, nperseg: int, overlap: float) -> Iterator[np.ndarray]:
    step = max(1, int(round(nperseg * (1.0 - overlap))))
    for start in range(0, x.size - nperseg + 1, step):
        yield x[start : start + nperseg]


def _parameters(samples: np.ndarray, sample_rate: float, nperseg: int, overlap: float):
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 1 or x.size < 8 or not np.all(np.isfinite(x)):
        raise ValueError("samples must be a finite 1-D array with at least 8 values")
    if sample_rate <= 0 or not 0 <= overlap < 1:
        raise ValueError("invalid sample rate or overlap")
    size = min(int(nperseg), x.size)
    if size < 8:
        raise ValueError("nperseg must be at least 8")
    return x, size, np.hanning(size)


def welch_psd(
    samples: np.ndarray, sample_rate: float, nperseg: int = 1024, overlap: float = 0.5
) -> Spectrum:
    """One-sided density-scaled Welch PSD with per-segment mean removal."""
    x, size, window = _parameters(samples, sample_rate, nperseg, overlap)
    scale = sample_rate * np.sum(window**2)
    accum = None
    count = 0
    for segment in _segments(x, size, overlap):
        fft = np.fft.rfft((segment - np.mean(segment)) * window)
        power = np.abs(fft) ** 2 / scale
        if size % 2 == 0:
            power[1:-1] *= 2.0
        else:
            power[1:] *= 2.0
        accum = power if accum is None else accum + power
        count += 1
    if count == 0:
        raise ValueError("capture has no complete Welch segments")
    return Spectrum(np.fft.rfftfreq(size, 1.0 / sample_rate), accum / count, sample_rate, count)


def transfer_coherence(
    excitation: np.ndarray,
    response: np.ndarray,
    sample_rate: float,
    nperseg: int = 1024,
    overlap: float = 0.5,
) -> TransferMetrics:
    """Estimate H1 transfer function and magnitude-squared coherence."""
    u, size, window = _parameters(excitation, sample_rate, nperseg, overlap)
    y = np.asarray(response, dtype=np.float64)
    if y.shape != u.shape or not np.all(np.isfinite(y)):
        raise ValueError("response must match excitation")
    suu = np.zeros(size // 2 + 1, dtype=np.float64)
    syy = np.zeros_like(suu)
    syu = np.zeros_like(suu, dtype=np.complex128)
    count = 0
    for useg, yseg in zip(_segments(u, size, overlap), _segments(y, size, overlap)):
        uf = np.fft.rfft((useg - np.mean(useg)) * window)
        yf = np.fft.rfft((yseg - np.mean(yseg)) * window)
        suu += np.abs(uf) ** 2
        syy += np.abs(yf) ** 2
        syu += yf * np.conjugate(uf)
        count += 1
    eps = np.finfo(float).tiny
    transfer = syu / np.maximum(suu, eps)
    coherence = np.abs(syu) ** 2 / np.maximum(suu * syy, eps)
    coherence = np.clip(coherence, 0.0, 1.0)
    return TransferMetrics(
        np.fft.rfftfreq(size, 1.0 / sample_rate), transfer, coherence, sample_rate, count
    )


def project_axes(
    acceleration: np.ndarray, excitation_axis: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Project 3-D acceleration into excitation and aggregate orthogonal response."""
    values = np.asarray(acceleration, dtype=np.float64)
    axis = np.asarray(excitation_axis, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or axis.shape != (3,):
        raise ValueError("acceleration must be N x 3 and axis must have three values")
    norm = np.linalg.norm(axis)
    if norm == 0:
        raise ValueError("excitation axis cannot be zero")
    unit = axis / norm
    along = values @ unit
    orthogonal = values - along[:, None] * unit
    cross = np.sqrt(np.sum(orthogonal**2, axis=1))
    return along, cross


def aggregate_spectra(spectra: Sequence[Spectrum]) -> Tuple[Spectrum, np.ndarray]:
    """Median-aggregate repeat PSDs and return per-bin relative MAD.

    Repeat spectra may use different FFT grids. Each is interpolated onto the
    coarsest grid over their common frequency range, avoiding invented
    high-resolution structure and extrapolation.
    """
    if not spectra:
        raise ValueError("at least one spectrum is required")
    for spectrum in spectra:
        if spectrum.frequencies.ndim != 1 or spectrum.values.shape != spectrum.frequencies.shape:
            raise ValueError("invalid spectrum shape")
        if np.any(np.diff(spectrum.frequencies) <= 0):
            raise ValueError("spectrum frequencies must increase")
    lower = max(float(item.frequencies[0]) for item in spectra)
    upper = min(float(item.frequencies[-1]) for item in spectra)
    reference = min(
        spectra,
        key=lambda item: np.count_nonzero(
            (item.frequencies >= lower) & (item.frequencies <= upper)
        ),
    )
    grid = reference.frequencies[
        (reference.frequencies >= lower) & (reference.frequencies <= upper)
    ]
    if grid.size < 2:
        raise ValueError("repeat spectra have no useful common frequency range")
    matrix = np.vstack([np.interp(grid, item.frequencies, item.values) for item in spectra])
    median = np.median(matrix, axis=0)
    relative_mad = np.median(np.abs(matrix - median), axis=0) / np.maximum(
        np.abs(median), np.finfo(float).eps
    )
    aggregate = Spectrum(
        grid.copy(), median, float(np.median([item.sample_rate for item in spectra])), len(spectra)
    )
    return aggregate, relative_mad


def integrated_band_energy(spectrum: Spectrum, low: float, high: float) -> float:
    """Integrate PSD energy in an inclusive frequency band."""
    if not 0 <= low < high:
        raise ValueError("band bounds must be ordered and non-negative")
    mask = (spectrum.frequencies >= low) & (spectrum.frequencies <= high)
    if np.count_nonzero(mask) < 2:
        raise ValueError("band contains fewer than two frequency bins")
    frequencies = spectrum.frequencies[mask]
    values = spectrum.values[mask]
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(values, frequencies))
    return float(np.trapz(values, frequencies))
