import json
import re

import numpy as np

from klipper_advanced_shaper.artifacts import ArtifactWriter


def test_artifact_writer_emits_private_replay_and_review_files(tmp_path):
    report = {
        "schema_version": "1.0.0-alpha.1",
        "axes": {"X": {"selected": "mzv", "modes": [{"frequency": 74.4}]}},
    }
    raw = {"training": {"X": [{"samples": np.array([[0.0, 1.0, 2.0, 3.0], [0.1, 2.0, 3.0, 4.0]])}]}}
    artifacts = ArtifactWriter(tmp_path).write("result-1", report, raw)
    assert json.loads((tmp_path / "result-1" / "result.json").read_text())["schema_version"]
    assert (tmp_path / "result-1" / "captures.npz").is_file()
    assert (tmp_path / "result-1" / "summary.svg").is_file()
    assert (tmp_path / "result-1" / "summary.png").is_file()
    assert (tmp_path / "result-1" / "report.html").is_file()
    assert (tmp_path / "result-1" / "input_shaper.png").is_file()
    assert (tmp_path / "result-1" / "input_shaper.svg").is_file()
    assert (tmp_path / "result-1" / "manifest.json").is_file()
    assert artifacts["sha256"]


def _complete_report(status="accepted"):
    return {
        "schema_version": "1.0.0-alpha.1",
        "plugin_version": "0.1.0a1",
        "engine": "robust_v1+running_klipper_reference",
        "attempt_id": "attempt-42",
        "status": status,
        "profile": "performance",
        "reason": "<script>alert('report')</script> confidence gate",
        "square_corner_velocity": 7.0,
        "axes": {
            "X": {
                "selected": "mzv",
                "modes": [{"frequency": 74.4}],
                "pareto": ["mzv", "ei"],
                "qc": [{"passed": True}, {"passed": True}, {"passed": True}],
                "spectrum": {
                    "frequency_hz": [10.0, 74.4, 120.0],
                    "psd": [0.01, 3.0, 0.02],
                },
                "native_spectrum": {
                    "available": True,
                    "frequency_hz": [10.0, 74.4, 120.0],
                    "psd_sum": [0.02, 4.0, 0.03],
                    "psd_x": [0.01, 3.0, 0.02],
                    "psd_y": [0.004, 0.7, 0.005],
                    "psd_z": [0.006, 0.3, 0.005],
                },
                "spectrogram": {
                    "available": True,
                    "frequency_hz": [10.0, 74.4, 120.0],
                    "time_s": [0.0, 1.0, 2.0],
                    "power": [[0.1, 0.2, 0.1], [0.2, 4.0, 0.3], [0.1, 0.2, 0.1]],
                },
                "native_candidates": [
                    {
                        "name": "mzv",
                        "frequency_hz": 74.4,
                        "residual_vibration": 0.014,
                        "smoothing": 0.08,
                        "max_accel": 16150.0,
                        "native_frequency_response": {
                            "frequency_hz": [10.0, 74.4, 120.0],
                            "response_ratio": [0.9, 0.014, 0.7],
                        },
                    }
                ],
                "candidates": [
                    {
                        "name": "mzv",
                        "frequency": 74.4,
                        "residual_vibration": 0.014,
                        "smoothing": 0.08,
                        "max_accel": 16150.0,
                        "repeatability": 0.02,
                        "cross_axis_energy": 0.03,
                        "sensitivity": 0.04,
                    },
                    {
                        "name": "ei",
                        "frequency": 71.0,
                        "residual_vibration": 0.009,
                        "smoothing": 0.12,
                        "max_accel": 12000.0,
                        "repeatability": 0.03,
                        "cross_axis_energy": 0.02,
                        "sensitivity": 0.03,
                    },
                ],
            }
        },
        "validation": {
            "passed": status == "accepted",
            "reason": "confidence gate",
            "axes": {
                "X": {
                    "baseline_energy": 12.0,
                    "shaped_energy": 9.0,
                    "improvement_ci_95": [0.12, 0.31],
                    "reference_cross_axis_energy": 2.0,
                    "candidate_cross_axis_energy": 2.04,
                    "cross_axis_regression": 0.02,
                    "passed": status == "accepted",
                }
            },
        },
        "native_command_preview": "SET_INPUT_SHAPER SHAPER_TYPE_X=mzv SHAPER_FREQ_X=74.4",
    }


def test_polished_report_has_decision_metrics_csv_and_no_network_assets(tmp_path):
    report = _complete_report()
    artifacts = ArtifactWriter(tmp_path, keep_raw=False).write("accepted", report)
    output = tmp_path / "accepted"
    rendered = (output / "report.html").read_text(encoding="utf-8")

    assert "ACCEPTED FOR REVIEW" in rendered
    assert "16,150 mm/s²" in rendered
    assert "12.0% to 31.0%" in rendered
    assert "2.0%" in rendered
    assert "not a mechanical safety rating" in rendered
    assert "Input shaper spectrum" in rendered
    assert "unitless normalized or arbitrary response units" in rendered
    input_shaper_svg = (output / "input_shaper.svg").read_text(encoding="utf-8")
    assert "Normalized spectral response" in input_shaper_svg
    assert "&lt;script&gt;alert('report')&lt;/script&gt;" in rendered
    assert "<script" not in rendered.lower()
    assert not re.search(r"(?:src|href)=[\"']https?://", rendered, re.IGNORECASE)
    assert (output / "candidates.csv").is_file()
    assert (output / "validation.csv").is_file()
    assert "raw" not in artifacts
    assert "mzv" in (output / "candidates.csv").read_text(encoding="utf-8")
    assert "0.12" in (output / "validation.csv").read_text(encoding="utf-8")


def test_rejected_report_leads_with_failure_and_cannot_imply_applicability(tmp_path):
    report = _complete_report("rejected")
    ArtifactWriter(tmp_path, keep_raw=False).write("rejected", report)
    rendered = (tmp_path / "rejected" / "report.html").read_text(encoding="utf-8")

    assert "REJECTED" in rendered
    assert "This result is rejected and cannot be applied or staged" in rendered
    assert "Do not apply or stage this attempt" in rendered
    assert "Correct the failing gate" in rendered
    assert '<pre class="command">' not in rendered


def test_sparse_report_generates_all_visuals_with_pending_labels(tmp_path):
    ArtifactWriter(tmp_path, keep_raw=False).write(
        "sparse", {"schema_version": "1", "axes": {}}
    )
    output = tmp_path / "sparse"
    rendered = (output / "report.html").read_text(encoding="utf-8")

    assert "Validation has not produced an acceptance decision" in rendered
    assert "Held-out validation is not available" in rendered
    for name in ("summary.png", "summary.svg", "input_shaper.png", "input_shaper.svg"):
        assert (output / name).stat().st_size > 100
