from pathlib import Path

import klipper_advanced_shaper

ROOT = Path(__file__).resolve().parents[1]


def test_public_version_matches_alpha_release():
    assert klipper_advanced_shaper.__version__ == "0.1.0a1"


def test_license_and_safety_docs_are_present():
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "GNU GENERAL PUBLIC LICENSE" in license_text
    assert "Version 3, 29 June 2007" in license_text
    assert "GPL-3.0-only" in readme
    assert "Alpha safety notice" in readme


def test_klipper_loader_exposes_load_config():
    namespace = {}
    loader = ROOT / "scripts" / "advanced_input_shaper.py"
    exec(compile(loader.read_text(encoding="utf-8"), str(loader), "exec"), namespace)
    assert callable(namespace["load_config"])


def test_private_artifact_patterns_are_ignored():
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "*.stdata" in ignore
    assert "printer.cfg" in ignore
    assert "secrets/" in ignore
