#!/usr/bin/env python3
"""
Convert downloaded ATLAS forced-photometry files into SNooPy format.

Behavior:
- Reads ATLAS raw files (default: data/light_curves/ATLAS/raw/*_atlas_forcedphot.txt).
- Applies S/N cut in flux space: uJy/duJy >= snr_min.
- Maps ATLAS filters:
    c -> ATgr
    o -> ATri
- Appends ATLAS blocks to existing SNooPy files if missing.
- Creates new SNooPy files if they do not yet exist.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord

from project_paths import DATA_DIR, light_curves_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert ATLAS forced-phot files to SNooPy format.")
    p.add_argument(
        "--atlas-dir",
        default=str(light_curves_path("ATLAS", "raw")),
        help="Directory with *_atlas_forcedphot.txt files.",
    )
    p.add_argument(
        "--sample-csv",
        default=str(DATA_DIR / "supernovas_input.csv"),
        help="Sample table with matched_snname, redshift, rasn, decsn.",
    )
    p.add_argument(
        "--snoopy-dir",
        default=str(light_curves_path("SNooPy")),
        help="Destination folder for SNooPy files.",
    )
    p.add_argument(
        "--report-csv",
        default=str(light_curves_path("atlas_to_snoopy_conversion_report.csv")),
        help="Output report CSV.",
    )
    p.add_argument(
        "--snr-min",
        type=float,
        default=3.0,
        help="Minimum S/N cut in ATLAS flux space (uJy/duJy).",
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


def parse_ra_dec_to_deg(ra_raw: object, dec_raw: object) -> tuple[float, float]:
    ra_txt = str(ra_raw).strip()
    dec_txt = str(dec_raw).strip()

    ra_num = pd.to_numeric(ra_txt, errors="coerce")
    dec_num = pd.to_numeric(dec_txt, errors="coerce")
    if np.isfinite(ra_num) and np.isfinite(dec_num):
        return float(ra_num), float(dec_num)

    if ":" in ra_txt or "h" in ra_txt.lower():
        c = SkyCoord(ra_txt, dec_txt, unit=(u.hourangle, u.deg), frame="icrs")
    else:
        c = SkyCoord(ra_txt, dec_txt, unit=(u.deg, u.deg), frame="icrs")
    return float(c.ra.deg), float(c.dec.deg)


def extract_name_from_atlas_filename(path: Path) -> str:
    s = path.stem
    suffix = "_atlas_forcedphot"
    return s[: -len(suffix)] if s.endswith(suffix) else s


def parse_existing_filters(snoopy_path: Path) -> set[str]:
    if not snoopy_path.exists():
        return set()
    filters = set()
    with snoopy_path.open("r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("filter "):
                parts = line.split()
                if len(parts) >= 2:
                    filters.add(parts[1])
    return filters


def parse_atlas_table(path: Path) -> pd.DataFrame:
    # ATLAS forced-phot text starts with header like "###MJD m dm uJy ...".
    df = pd.read_csv(path, sep=r"\s+", engine="python")
    # Clean header: ###MJD -> MJD
    df.columns = [str(c).strip().lstrip("#") for c in df.columns]
    required = {"MJD", "m", "dm", "uJy", "duJy", "F"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns {required - set(df.columns)}")
    return df


def main() -> int:
    args = parse_args()

    atlas_dir = Path(args.atlas_dir).expanduser().resolve()
    sample_csv = Path(args.sample_csv).expanduser().resolve()
    snoopy_dir = Path(args.snoopy_dir).expanduser().resolve()
    report_csv = Path(args.report_csv).expanduser().resolve()

    snoopy_dir.mkdir(parents=True, exist_ok=True)
    report_csv.parent.mkdir(parents=True, exist_ok=True)

    atlas_files = sorted(atlas_dir.glob("*_atlas_forcedphot.txt"))
    if not atlas_files:
        print(f"No ATLAS files found in {atlas_dir}")
        return 1

    sample = pd.read_csv(sample_csv)
    req = {"matched_snname", "redshift", "rasn", "decsn"}
    if not req.issubset(sample.columns):
        raise ValueError(f"Sample table missing columns: {req - set(sample.columns)}")

    sample["sample_sn"] = sample["matched_snname"].astype(str).str.strip()
    # key -> list of sample rows (indices)
    sample_key_map: dict[str, list[int]] = {}
    for idx, row in sample.iterrows():
        for k in variants(row["sample_sn"]):
            sample_key_map.setdefault(k, []).append(idx)

    filter_map = {"c": "ATgr", "o": "ATri"}
    report_rows = []

    for atlas_path in atlas_files:
        atlas_name = extract_name_from_atlas_filename(atlas_path)
        status = "ok"
        message = ""
        sample_sn = ""
        output_file = ""
        n_points_total = 0
        n_atgr = 0
        n_atri = 0

        try:
            # Match atlas file name to sample SN name.
            cand_idx: set[int] = set()
            for k in variants(atlas_name):
                for idx in sample_key_map.get(k, []):
                    cand_idx.add(idx)
            if len(cand_idx) == 0:
                raise ValueError(f"No sample SN match for ATLAS file name {atlas_name}")
            if len(cand_idx) > 1:
                names = sample.loc[sorted(cand_idx), "sample_sn"].tolist()
                raise ValueError(f"Ambiguous sample SN match for {atlas_name}: {names}")

            srow = sample.loc[next(iter(cand_idx))]
            sample_sn = str(srow["sample_sn"]).strip()
            z = float(pd.to_numeric(srow["redshift"], errors="coerce"))
            ra, dec = parse_ra_dec_to_deg(srow["rasn"], srow["decsn"])
            if not (np.isfinite(z) and np.isfinite(ra) and np.isfinite(dec)):
                raise ValueError(f"Invalid z/ra/dec for {sample_sn} in sample table")

            atlas = parse_atlas_table(atlas_path)
            atlas = atlas.copy()
            for col in ["MJD", "m", "dm", "uJy", "duJy"]:
                atlas[col] = pd.to_numeric(atlas[col], errors="coerce")
            atlas["F"] = atlas["F"].astype(str).str.strip()

            # S/N cut in flux space.
            atlas = atlas[atlas["duJy"] > 0]
            atlas = atlas[np.isfinite(atlas["uJy"] / atlas["duJy"])]
            atlas = atlas[(atlas["uJy"] / atlas["duJy"]) >= args.snr_min]
            atlas = atlas[np.isfinite(atlas["MJD"]) & np.isfinite(atlas["m"]) & np.isfinite(atlas["dm"])].copy()
            atlas = atlas[atlas["dm"] > 0].copy()

            existing_path = snoopy_dir / f"{sample_sn}_raw.txt"
            existing_filters = parse_existing_filters(existing_path)

            blocks = []
            for f_atlas, f_snoopy in filter_map.items():
                if f_snoopy in existing_filters:
                    continue
                sub = atlas[atlas["F"] == f_atlas].sort_values("MJD")
                if len(sub) == 0:
                    continue
                lines = [f"filter {f_snoopy}"]
                for _, rr in sub.iterrows():
                    lines.append(f"{rr['MJD']:.3f} {rr['m']:.3f} {rr['dm']:.3f}")
                blocks.append((f_snoopy, lines, len(sub)))

            if len(blocks) == 0:
                status = "skipped_no_new_filters_or_points"
            else:
                mode = "a" if existing_path.exists() else "w"
                with existing_path.open(mode, encoding="utf-8") as f:
                    if mode == "w":
                        f.write(f"{sample_sn} {z:.6f} {ra:.6f} {dec:.6f}\n")
                    else:
                        # Ensure block starts on a new line.
                        f.seek(0, 2)
                        if f.tell() > 0:
                            f.write("\n")

                    for _, lines, _ in blocks:
                        f.write("\n".join(lines) + "\n")

                output_file = str(existing_path)
                for f_snoopy, _, npts in blocks:
                    if f_snoopy == "ATgr":
                        n_atgr = npts
                    elif f_snoopy == "ATri":
                        n_atri = npts
                n_points_total = n_atgr + n_atri
        except Exception as e:
            status = "failed"
            message = str(e)

        report_rows.append(
            {
                "atlas_file": str(atlas_path),
                "atlas_name": atlas_name,
                "sample_sn": sample_sn,
                "status": status,
                "message": message,
                "output_file": output_file,
                "n_points_total": n_points_total,
                "n_ATgr": n_atgr,
                "n_ATri": n_atri,
            }
        )

    rep = pd.DataFrame(report_rows)
    rep.to_csv(report_csv, index=False)

    ok = int((rep["status"] == "ok").sum()) if len(rep) else 0
    skip = int((rep["status"] == "skipped_no_new_filters_or_points").sum()) if len(rep) else 0
    fail = int((rep["status"] == "failed").sum()) if len(rep) else 0
    print(f"ATLAS files processed: {len(rep)}")
    print(f"Converted/appended: {ok}")
    print(f"Skipped (no new ATgr/ATri): {skip}")
    print(f"Failed: {fail}")
    print(f"Report: {report_csv}")
    print(f"SNooPy dir: {snoopy_dir}")
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
