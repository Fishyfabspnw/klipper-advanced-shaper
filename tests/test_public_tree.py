from pathlib import Path

from scripts.verify_public_tree import violations


def test_public_tree_rejects_private_artifact_paths(tmp_path: Path):
    capture = tmp_path / "capture.stdata"
    capture.write_text("synthetic", encoding="utf-8")
    assert violations([capture]) == ["forbidden public path: capture.stdata"]


def test_public_tree_rejects_private_keys(tmp_path: Path):
    key = tmp_path / "fixture.txt"
    key.write_text("-----BEGIN OPENSSH PRIVATE" + " KEY-----", encoding="utf-8")
    assert violations([key]) == ["credential-like content in: fixture.txt"]
