"""Klipper extra loader installed by ``install.sh``.

This file intentionally stays dependency-free so Klippy can discover the
packaged implementation using its normal extras loader.
"""

from klipper_advanced_shaper.klippy import load_config

__all__ = ["load_config"]
