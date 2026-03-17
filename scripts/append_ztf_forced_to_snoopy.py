#!/usr/bin/env python3
"""
Append converted ZTF forced-photometry blocks to the end of the original
SNooPy `_raw.txt` files.

The standalone `*_ztf_forced_raw.txt` files are kept on disk for provenance,
but after this merge the original files become the main combined products.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from project_paths import light_curves_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--snoopy-dir",
        default=str(light_curves_path("SNooPy")),
        help="Directory containing original and *_ztf_forced_raw.txt files.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    snoopy_dir = Path(args.snoopy_dir).expanduser().resolve()

    merged = 0
    skipped = 0

    for forced_path in sorted(snoopy_dir.glob("*_ztf_forced_raw.txt")):
        base_name = forced_path.name.removesuffix("_ztf_forced_raw.txt")
        original_path = snoopy_dir / f"{base_name}_raw.txt"
        if not original_path.exists():
            print(f"skip missing original: {forced_path.name}")
            skipped += 1
            continue

        original_lines = original_path.read_text(errors="ignore").splitlines()
        forced_lines = forced_path.read_text(errors="ignore").splitlines()
        if not forced_lines:
            print(f"skip empty forced file: {forced_path.name}")
            skipped += 1
            continue

        forced_payload = "\n".join(forced_lines[1:]).strip()
        if not forced_payload:
            print(f"skip no payload: {forced_path.name}")
            skipped += 1
            continue

        original_text = "\n".join(original_lines).strip()
        if forced_payload in original_text:
            print(f"already merged: {original_path.name}")
            skipped += 1
            continue

        merged_text = original_text + "\n" + forced_payload + "\n"
        original_path.write_text(merged_text)
        print(f"merged: {forced_path.name} -> {original_path.name}")
        merged += 1

    print(f"Merged files: {merged}")
    print(f"Skipped files: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
