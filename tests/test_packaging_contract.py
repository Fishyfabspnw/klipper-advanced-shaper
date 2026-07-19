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


def test_installation_guide_covers_full_lifecycle():
    guide = (ROOT / "docs" / "installation.md").read_text(encoding="utf-8")
    for expected in (
        "./scripts/install.sh",
        "./scripts/update.sh",
        "./scripts/uninstall.sh",
        "ADV_SHAPER_STATUS",
        "ACCELEROMETER_QUERY",
        "sudo systemctl restart klipper",
        "KLIPPER_DIR",
        "KLIPPER_VENV",
        "KLIPPER_CONFIG_DIR",
        "[include advanced_shaper_macros.cfg]",
        "enable_experimental_generalized_mzv",
        "ACCEL_PER_HZ",
        "AdvancedShaper_results/<attempt-id>",
    ):
        assert expected in guide


def test_shaketune_style_docs_index_and_macro_reference_are_packaged():
    index = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    calibration = (ROOT / "docs" / "macros" / "advanced_shaper_calibrate.md").read_text(
        encoding="utf-8"
    )
    workflow = (ROOT / "docs" / "macros" / "result_workflow.md").read_text(
        encoding="utf-8"
    )

    assert "Commands and workflows" in index
    assert "| parameter | default value | description |" in calibration
    for parameter in (
        "`AXIS`",
        "`PROFILE`",
        "`REPEATS`",
        "`VALIDATE`",
        "`ACCEL_PER_HZ`",
        "`HZ_PER_SEC`",
        "`SCV`",
        "`FAST_VALIDATION`",
        "`PEAK_LOCK`",
    ):
        assert parameter in calibration
    assert "This is not a preset list" in calibration
    assert "ordinary profiles do not silently add ZVD" in calibration
    assert "SAVE_CONFIG" in workflow


def test_installer_force_replaces_same_version_package_without_dependency_churn():
    installer = (ROOT / "scripts" / "install.sh").read_text(encoding="utf-8")
    normal_install = 'pip install --upgrade "${repo_dir}"'
    forced_install = '--force-reinstall --no-deps "${repo_dir}"'

    assert normal_install in installer
    assert forced_install in installer
    assert installer.index(normal_install) < installer.index(forced_install)
    force_lines = [line for line in installer.splitlines() if "--force-reinstall" in line]
    assert force_lines == ['  --force-reinstall --no-deps "${repo_dir}"']


def test_readme_documents_sweep_timing_and_smoke_test_boundaries():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(readme.split())

    assert "nine full resonance sweeps per axis" in normalized
    assert "ACCEL_PER_HZ` changes excitation intensity, not sweep duration" in normalized
    assert "REPEATS=1 VALIDATE=0" in normalized
    assert "full-confidence default requires `REPEATS>=3`" in normalized
    assert "approximately 5.4 minutes of resonance motion per axis at 2 Hz/s" in normalized
    assert "five commanded sweeps" in normalized
    assert "movement between probe points" in normalized
    assert "not a promise that the complete axis workflow finishes" in normalized
    assert "Validation-rejected attempts retain diagnostic artifacts" in normalized
    assert "never become eligible for apply or stage" in normalized
    assert "unsigned decimal from 20 through 350" in normalized
    assert "does not require Klipper's optional `[respond]` section" in normalized
    assert "fit within 80% of the printer's current" in normalized
    assert "does not guarantee a PSD above `1e-5`" in normalized
    assert "performs 18 total sweeps" in normalized
    assert "free numeric excitation control" in normalized.lower()
    assert "standard macro UI cannot attach a separate tooltip to each input" in normalized
    assert "modify Klipper's motion planner" in normalized
    assert "It does **not** modify" in normalized
    assert "installation alone does not create it" in normalized


def test_readme_has_complete_calibration_parameter_reference():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for parameter in (
        "`AXIS`",
        "`PROFILE`",
        "`REPEATS`",
        "`VALIDATE`",
        "`ACCEL_PER_HZ`",
        "`HZ_PER_SEC`",
        "`SCV`",
        "`FAST_VALIDATION`",
        "`PEAK_LOCK`",
    ):
        assert parameter in readme
    for profile in (
        "quality",
        "balanced",
        "performance",
        "experimental_mzv",
        "adaptive_stock",
    ):
        assert profile in readme


def test_private_artifact_patterns_are_ignored():
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "*.stdata" in ignore
    assert "printer.cfg" in ignore
    assert "secrets/" in ignore
