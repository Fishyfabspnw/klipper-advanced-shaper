import json

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
    assert (tmp_path / "result-1" / "manifest.json").is_file()
    assert artifacts["sha256"]
