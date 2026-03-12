#!/usr/bin/env python3
"""
Compatibility launcher.

Canonical pipeline script:
    scripts/hst_uv_photometry_pipeline.py

This file is kept so existing commands that call HST_UV_code13.py still work.
"""

from pathlib import Path
import runpy

if __name__ == "__main__":
    script = Path(__file__).resolve().parent / "scripts" / "hst_uv_photometry_pipeline.py"
    runpy.run_path(str(script), run_name="__main__")
