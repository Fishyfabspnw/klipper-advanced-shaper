"""Klippy integration for Klipper Advanced Shaper.

This package deliberately has no import-time dependency on Klipper so its safety
logic can be tested on a development machine.
"""

from .plugin import AdvancedInputShaper, load_config

__all__ = ["AdvancedInputShaper", "load_config"]
