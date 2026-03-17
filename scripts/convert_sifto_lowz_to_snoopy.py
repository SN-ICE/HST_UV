#!/usr/bin/env python3
"""
Build SNooPy-style light-curve TXT files from low-z Ia (SIFTO-like) folders.

Input structure (per SN folder):
  - <sn>.info                (contains Zhel, Ra, Dec)
  - <sn>.out_<filter>        (4 header lines + photometry rows)

Output structure:
  - <output-dir>/<safe_sn>_raw.txt

SNooPy TXT format produced:
  first line: SNname z ra dec
  then repeated blocks:
    filter <filter_name>
    MJD mag err
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

from project_paths import DATA_DIR, light_curves_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert low-z Ia folders into SNooPy TXT files.")
    p.add_argument(
        "--presence-csv",
        default=str(light_curves_path("sifto_lowzIa_found_for_snoopy_missing.csv")),
        help="CSV with sample_sn and lowz_Ia_matches columns.",
    )
    p.add_argument(
        "--sample-csv",
        default=str(DATA_DIR / "supernovas_input.csv"),
        help="Sample table used for fallback z/RA/Dec.",
    )
    p.add_argument(
        "--lowz-root",
        default="/Users/lluisgalbany/sndata/photometry/lowz/Ia",
        help="Root directory with low-z Ia folders.",
    )
    p.add_argument(
        "--output-dir",
        default=str(light_curves_path("SNooPy")),
        help="Destination directory for generated SNooPy TXT files.",
    )
    p.add_argument(
        "--report-csv",
        default=str(light_curves_path("sifto_to_snoopy_conversion_report.csv")),
        help="CSV report of conversion status per SN.",
    )
    return p.parse_args()


def normalize_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def safe_name(s: str) -> str:
    out = re.sub(r"[^A-Za-z0-9._+-]+", "_", str(s).strip())
    out = out.strip("_")
    return out or "SN"


def parse_float_or_none(value) -> float | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"none", "nan", "null"}:
        return None
    try:
        x = float(s)
        return x if np.isfinite(x) else None
    except Exception:
        return None


def parse_ra_deg(value) -> float | None:
    s = str(value).strip()
    if not s:
        return None
    if ":" in s or any(ch in s.lower() for ch in ["h", "m", "s"]):
        try:
            return SkyCoord(ra=s, dec="0d", unit=(u.hourangle, u.deg), frame="icrs").ra.deg
        except Exception:
            return None
    return parse_float_or_none(s)


def parse_dec_deg(value) -> float | None:
    s = str(value).strip()
    if not s:
        return None
    if ":" in s or any(ch in s.lower() for ch in ["d", "m", "s"]):
        try:
            return SkyCoord(ra="0h", dec=s, unit=(u.hourangle, u.deg), frame="icrs").dec.deg
        except Exception:
            return None
    return parse_float_or_none(s)


def read_info_file(info_path: Path) -> dict:
    meta = {}
    for line in info_path.read_text(errors="ignore").splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        meta[k.strip().lower()] = v.strip()
    return meta


def parse_out_file(path: Path) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for line in path.read_text(errors="ignore").splitlines():
        s = line.strip()
        if not s:
            continue
        # Skip headers/metadata.
        if s.lower().startswith(("sn:", "name:", "filter:", "npoints:", "mjd")):
            continue
        parts = s.split()
        if len(parts) < 3:
            continue
        # Keep original tokens if numeric first 3 columns are valid.
        try:
            mjd = float(parts[0])
            mag = float(parts[1])
            err = float(parts[2])
            if not (np.isfinite(mjd) and np.isfinite(mag) and np.isfinite(err)):
                continue
        except Exception:
            continue
        rows.append((parts[0], parts[1], parts[2]))
    return rows


def main() -> int:
    args = parse_args()

    presence_csv = Path(args.presence_csv).expanduser().resolve()
    sample_csv = Path(args.sample_csv).expanduser().resolve()
    lowz_root = Path(args.lowz_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    report_csv = Path(args.report_csv).expanduser().resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    report_csv.parent.mkdir(parents=True, exist_ok=True)

    found_df = pd.read_csv(presence_csv)
    sample_df = pd.read_csv(sample_csv)

    # Build fallback map from project sample table.
    sample_map = {}
    for _, row in sample_df.iterrows():
        sn = str(row.get("matched_snname", "")).strip()
        if not sn:
            continue
        sample_map[normalize_name(sn)] = {
            "z": parse_float_or_none(row.get("redshift", None)),
            "ra": parse_ra_deg(row.get("rasn", "")),
            "dec": parse_dec_deg(row.get("decsn", "")),
        }

    report_rows = []

    for _, row in found_df.iterrows():
        sample_sn = str(row.get("sample_sn", "")).strip()
        match_cell = str(row.get("lowz_Ia_matches", "")).strip()
        if not sample_sn or not match_cell:
            continue

        lowz_matches = [x.strip() for x in match_cell.split(";") if x.strip()]
        lowz_name = lowz_matches[0] if lowz_matches else ""
        sn_dir = lowz_root / lowz_name

        status = "ok"
        message = ""
        n_filters = 0
        n_points = 0
        out_name = f"{safe_name(sample_sn)}_raw.txt"
        out_path = output_dir / out_name

        try:
            if not sn_dir.exists():
                raise FileNotFoundError(f"Directory not found: {sn_dir}")

            info_files = sorted(sn_dir.glob("*.info"))
            if not info_files:
                raise FileNotFoundError("No .info file found")
            info = read_info_file(info_files[0])

            # Metadata from .info, with fallback to sample table.
            z = parse_float_or_none(info.get("zhel"))
            ra = parse_ra_deg(info.get("ra", ""))
            dec = parse_dec_deg(info.get("dec", ""))

            fb = sample_map.get(normalize_name(sample_sn), {})
            if z is None:
                z = fb.get("z", None)
            if ra is None:
                ra = fb.get("ra", None)
            if dec is None:
                dec = fb.get("dec", None)

            if z is None:
                z = 0.0
            if ra is None or dec is None:
                raise ValueError("Could not determine RA/Dec from .info or sample fallback")

            out_files = sorted(sn_dir.glob("*.out_*"))
            if not out_files:
                raise FileNotFoundError("No .out_<filter> files found")

            blocks: list[tuple[str, list[tuple[str, str, str]]]] = []
            for op in out_files:
                # filter name is suffix after ".out_"
                if ".out_" not in op.name:
                    continue
                filt = op.name.split(".out_", 1)[1]
                rows3 = parse_out_file(op)
                if not rows3:
                    continue
                blocks.append((filt, rows3))

            if not blocks:
                raise ValueError("No valid photometry rows parsed from .out files")

            lines = [f"{sample_sn} {z:.6f} {ra:.6f} {dec:.6f}"]
            for filt, rows3 in blocks:
                lines.append(f"filter {filt}")
                for mjd, mag, err in rows3:
                    lines.append(f"{mjd}  {mag}  {err}")
                n_filters += 1
                n_points += len(rows3)

            out_path.write_text("\n".join(lines) + "\n")

            if len(lowz_matches) > 1:
                message = f"Multiple lowz matches ({len(lowz_matches)}), used: {lowz_name}"
        except Exception as e:
            status = "failed"
            message = str(e)

        report_rows.append(
            {
                "sample_sn": sample_sn,
                "lowz_match_used": lowz_name,
                "status": status,
                "message": message,
                "output_file": str(out_path) if status == "ok" else "",
                "n_filters": n_filters if status == "ok" else 0,
                "n_points": n_points if status == "ok" else 0,
            }
        )

    rep = pd.DataFrame(report_rows)
    rep.to_csv(report_csv, index=False)

    ok = int((rep["status"] == "ok").sum()) if len(rep) else 0
    failed = int((rep["status"] != "ok").sum()) if len(rep) else 0
    print(f"Created SNooPy files: {ok}")
    print(f"Failed conversions: {failed}")
    print(f"Report: {report_csv}")
    print(f"Output dir: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
