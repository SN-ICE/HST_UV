#!/usr/bin/env python3
"""
Convert recovered ZTF forced-photometry text files into SNooPy TXT format.

This keeps the converted outputs separate from the main SNooPy files by writing
`*_ztf_forced_raw.txt` files, so we do not overwrite any existing preferred
light-curve products already in the project.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord

from project_paths import DATA_DIR, light_curves_path


FILTER_MAP = {"ZTF_g": "ztfg", "ZTF_r": "ztfr", "ZTF_i": "ztfi"}
FLOOR_FRAC = {"ztfg": 0.025, "ztfr": 0.035, "ztfi": 0.060}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--recovered-report",
        default=str(light_curves_path("ZTF", "ztf_forcedphot_recovered_report.csv")),
        help="Recovered ZTF forced-phot report CSV.",
    )
    p.add_argument(
        "--sample-csv",
        default=str(DATA_DIR / "supernovas_input.csv"),
        help="Current active sample table with matched_snname, redshift, rasn, decsn.",
    )
    p.add_argument(
        "--existing-snoopy-dir",
        default=str(light_curves_path("SNooPy")),
        help="Directory containing existing SNooPy TXT files for metadata fallback.",
    )
    p.add_argument(
        "--output-dir",
        default=str(light_curves_path("SNooPy")),
        help="Destination for converted SNooPy TXT files.",
    )
    p.add_argument(
        "--report-csv",
        default=str(light_curves_path("ZTF", "ztf_forcedphot_to_snoopy_report.csv")),
        help="Output conversion report.",
    )
    p.add_argument(
        "--snr-min",
        type=float,
        default=3.0,
        help="Minimum S/N cut after flux-error floors.",
    )
    return p.parse_args()


def safe_name(text: str) -> str:
    out = re.sub(r"[^A-Za-z0-9._+-]+", "_", str(text).strip()).strip("_")
    return out or "SN"


def parse_ra_dec_to_deg(ra_raw: object, dec_raw: object) -> tuple[float, float]:
    ra_txt = str(ra_raw).strip()
    dec_txt = str(dec_raw).strip()
    ra_num = pd.to_numeric(ra_txt, errors="coerce")
    dec_num = pd.to_numeric(dec_txt, errors="coerce")
    if np.isfinite(ra_num) and np.isfinite(dec_num):
        return float(ra_num), float(dec_num)
    if ":" in ra_txt or "h" in ra_txt.lower():
        coord = SkyCoord(ra_txt, dec_txt, unit=(u.hourangle, u.deg), frame="icrs")
        return float(coord.ra.deg), float(coord.dec.deg)
    coord = SkyCoord(ra_txt, dec_txt, unit=(u.deg, u.deg), frame="icrs")
    return float(coord.ra.deg), float(coord.dec.deg)


def parse_existing_snoopy_header(path: Path) -> tuple[float, float, float] | None:
    if not path.exists():
        return None
    first = path.read_text(errors="ignore").splitlines()[0].strip()
    parts = first.split()
    if len(parts) < 4:
        return None
    try:
        return float(parts[1]), float(parts[2]), float(parts[3])
    except Exception:
        return None


def load_metadata(sample: pd.DataFrame, existing_dir: Path, sn_name: str) -> tuple[float, float, float]:
    hit = sample[sample["matched_snname"].astype(str).str.strip() == sn_name]
    if len(hit):
        row = hit.iloc[0]
        z = float(pd.to_numeric(row["redshift"], errors="coerce"))
        ra, dec = parse_ra_dec_to_deg(row["rasn"], row["decsn"])
        if np.isfinite(z) and np.isfinite(ra) and np.isfinite(dec):
            return z, ra, dec

    existing = parse_existing_snoopy_header(existing_dir / f"{safe_name(sn_name)}_raw.txt")
    if existing is not None:
        return existing

    raise ValueError(f"No metadata found for {sn_name} in active sample or existing SNooPy files")


def read_forcedphot_table(path: Path) -> pd.DataFrame:
    lines = path.read_text(errors="ignore").splitlines()
    header_line = None
    data_start = None
    for i, line in enumerate(lines):
        if line.startswith(" index,") or line.startswith("index,"):
            header_line = line
            data_start = i + 2
            break
    if header_line is None or data_start is None:
        raise ValueError(f"Could not locate forced-phot header in {path}")

    cols = [c.strip() for c in header_line.split(",")]
    rows = []
    for line in lines[data_start:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) != len(cols):
            continue
        rows.append(parts)
    if not rows:
        raise ValueError(f"No data rows found in {path}")

    df = pd.DataFrame(rows, columns=cols)
    numeric_cols = [
        "infobitssci",
        "diffimgstatus",
        "forcediffimflux",
        "forcediffimfluxunc",
        "jd",
        "zpdiff",
        "procstatus",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def apply_cuts(df: pd.DataFrame, snr_min: float) -> pd.DataFrame:
    out = df.copy()
    out = out[out["filter"].isin(FILTER_MAP)].copy()
    procstatus = pd.to_numeric(out["procstatus"], errors="coerce")
    out = out[(out["diffimgstatus"] == 1) & (procstatus.fillna(0) == 0)].copy()
    out = out[(out["forcediffimfluxunc"] > 0) & np.isfinite(out["forcediffimfluxunc"])].copy()

    out["snoopy_filter"] = out["filter"].map(FILTER_MAP)
    out["flux"] = pd.to_numeric(out["forcediffimflux"], errors="coerce")
    out["flux_err"] = pd.to_numeric(out["forcediffimfluxunc"], errors="coerce")

    for filt, frac in FLOOR_FRAC.items():
        mask = out["snoopy_filter"] == filt
        out.loc[mask, "flux_err"] = np.sqrt(
            out.loc[mask, "flux_err"].values ** 2 + (out.loc[mask, "flux"].values * frac) ** 2
        )

    out = out[np.isfinite(out["flux"]) & np.isfinite(out["flux_err"]) & (out["flux"] > 0)].copy()
    out["snr"] = out["flux"] / out["flux_err"]
    out = out[np.isfinite(out["snr"]) & (out["snr"] >= snr_min)].copy()

    out["mag"] = out["zpdiff"] - 2.5 * np.log10(out["flux"])
    out["mag_err"] = 2.5 * np.log10(1.0 + out["flux_err"] / out["flux"])
    out = out[np.isfinite(out["mag"]) & np.isfinite(out["mag_err"])].copy()
    out = out.sort_values(["snoopy_filter", "jd"], kind="stable")
    return out


def main() -> int:
    args = parse_args()

    recovered_report = Path(args.recovered_report).expanduser().resolve()
    sample_csv = Path(args.sample_csv).expanduser().resolve()
    existing_dir = Path(args.existing_snoopy_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    report_csv = Path(args.report_csv).expanduser().resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    report_csv.parent.mkdir(parents=True, exist_ok=True)

    recovered = pd.read_csv(recovered_report)
    sample = pd.read_csv(sample_csv)

    rows = []
    for _, row in recovered.iterrows():
        sn_name = str(row.get("sn_name", "")).strip()
        raw_path = Path(str(row.get("output_file", "")).strip())
        status = "ok"
        message = ""
        output_file = output_dir / f"{safe_name(sn_name)}_ztf_forced_raw.txt"
        n_points_total = n_ztfg = n_ztfr = n_ztfi = 0

        try:
            if str(row.get("status", "")).strip() != "downloaded":
                raise ValueError(f"Recovered status is {row.get('status')}")
            if int(pd.to_numeric(row.get("n_lines"), errors="coerce") or 0) <= 0:
                raise ValueError("Recovered file is empty")
            if not raw_path.exists():
                raise FileNotFoundError(f"Recovered file missing: {raw_path}")

            z, ra, dec = load_metadata(sample, existing_dir, sn_name)
            forced = read_forcedphot_table(raw_path)
            clean = apply_cuts(forced, args.snr_min)
            if clean.empty:
                raise ValueError("No photometry left after cuts")

            lines = [f"{sn_name} {z:.6f} {ra:.6f} {dec:.6f}"]
            for filt in ["ztfg", "ztfr", "ztfi"]:
                sub = clean[clean["snoopy_filter"] == filt]
                if len(sub) == 0:
                    continue
                lines.append(f"filter {filt}")
                for _, rr in sub.iterrows():
                    lines.append(f"{rr['jd']:.3f} {rr['mag']:.3f} {rr['mag_err']:.3f}")
                if filt == "ztfg":
                    n_ztfg = len(sub)
                elif filt == "ztfr":
                    n_ztfr = len(sub)
                elif filt == "ztfi":
                    n_ztfi = len(sub)

            n_points_total = n_ztfg + n_ztfr + n_ztfi
            if n_points_total == 0:
                raise ValueError("No ztfg/ztfr/ztfi points after cuts")

            output_file.write_text("\n".join(lines) + "\n")
        except Exception as e:
            status = "failed"
            message = str(e)

        rows.append(
            {
                "sn_name": sn_name,
                "recovered_file": str(raw_path),
                "status": status,
                "message": message,
                "output_file": str(output_file) if status == "ok" else "",
                "n_points_total": n_points_total if status == "ok" else 0,
                "n_ztfg": n_ztfg if status == "ok" else 0,
                "n_ztfr": n_ztfr if status == "ok" else 0,
                "n_ztfi": n_ztfi if status == "ok" else 0,
            }
        )

    report = pd.DataFrame(rows)
    report.to_csv(report_csv, index=False)
    ok = int((report["status"] == "ok").sum()) if len(report) else 0
    failed = int((report["status"] != "ok").sum()) if len(report) else 0
    print(f"Converted ZTF forced-phot -> SNooPy files: {ok}")
    print(f"Failed conversions: {failed}")
    print(f"Report: {report_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
