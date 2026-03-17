#!/usr/bin/env python3
"""
Refresh SNooPy coverage tables for the current active sample.

This rescans the SNooPy directory and matches files to the active sample using
the same name-normalization rules used elsewhere in the project, so newly added
or appended files are reflected immediately in the coverage tables.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from project_paths import DATA_DIR, light_curves_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample-csv", default=str(DATA_DIR / "supernovas_input.csv"))
    p.add_argument("--coverage-csv", default=str(light_curves_path("snoopy_file_coverage.csv")))
    p.add_argument("--matched-csv", default=str(light_curves_path("snoopy_file_matched.csv")))
    p.add_argument("--missing-csv", default=str(light_curves_path("snoopy_file_missing.csv")))
    p.add_argument(
        "--snoopy-dir",
        default=str(light_curves_path("SNooPy")),
        help="Directory scanned for extra *_ztf_forced_raw.txt files to append to existing matches.",
    )
    return p.parse_args()


def norm(name: str) -> str:
    txt = str(name).strip().lower()
    txt = txt.replace("-host", "").replace("_host", "")
    return re.sub(r"[^a-z0-9]+", "", txt)


def variants(name: str) -> set[str]:
    s = str(name).strip().lower()
    out = {s, norm(s)}
    for old, new in [("asassn", "asas"), ("asas", "asassn"), ("iptf", "ptf"), ("ptf", "iptf")]:
        s2 = s.replace(old, new)
        out.add(s2)
        out.add(norm(s2))
    n = norm(s)
    if n:
        out.add(n[2:] if n.startswith("sn") else "sn" + n)
    return {x for x in out if x}


def snoopy_stem_variants(path: Path) -> set[str]:
    name = path.name
    suffixes = ["_ztf_forced_raw.txt", "_raw.txt", ".txt"]
    stem = name
    for suffix in suffixes:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return variants(stem)


def main() -> int:
    args = parse_args()
    sample_csv = Path(args.sample_csv).expanduser().resolve()
    coverage_csv = Path(args.coverage_csv).expanduser().resolve()
    matched_csv = Path(args.matched_csv).expanduser().resolve()
    missing_csv = Path(args.missing_csv).expanduser().resolve()
    snoopy_dir = Path(args.snoopy_dir).expanduser().resolve()

    sample = pd.read_csv(sample_csv)
    sample["sample_sn"] = sample["matched_snname"].astype(str).str.strip()
    active_names = sample["sample_sn"].tolist()

    file_map: dict[str, list[str]] = {}
    for path in sorted(snoopy_dir.glob("*.txt")):
        for key in snoopy_stem_variants(path):
            file_map.setdefault(key, []).append(path.name)

    rows = []
    for sample_sn in active_names:
        matched_files: list[str] = []
        seen = set()
        for key in variants(sample_sn):
            for fname in file_map.get(key, []):
                if fname not in seen:
                    seen.add(fname)
                    matched_files.append(fname)
        rows.append(
            {
                "sample_sn": sample_sn,
                "has_snoopy_txt": bool(matched_files),
                "matched_txt_files": "; ".join(matched_files) if matched_files else pd.NA,
                "n_matched_files": len(matched_files),
            }
        )

    coverage = pd.DataFrame(rows)

    coverage = coverage.sort_values("sample_sn", kind="stable").reset_index(drop=True)
    matched = coverage[coverage["has_snoopy_txt"].fillna(False).astype(bool)].copy()
    missing = coverage[~coverage["has_snoopy_txt"].fillna(False).astype(bool)].copy()

    coverage.to_csv(coverage_csv, index=False)
    matched.to_csv(matched_csv, index=False)
    missing.to_csv(missing_csv, index=False)

    print(f"Active sample rows: {len(coverage)}")
    print(f"Matched rows: {len(matched)}")
    print(f"Missing rows: {len(missing)}")
    print(
        "Rows with appended ZTF forced files: "
        f"{int(coverage['matched_txt_files'].fillna('').str.contains('_ztf_forced_raw.txt').sum())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
