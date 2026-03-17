#!/usr/bin/env python3
"""
Convert selected ZTFCOSMO DR3 lightcurves into SNooPy TXT format.

This adapts the user's selection/cut logic:
- quality flags: reject bitmask 1,2,4,8,16
- add filter-dependent floor errors in flux space
- keep phase window around peak (mjd_max_init -20, +60)
- keep only positive flux
- convert to mag/error with ZP=30 convention:
    m  = -2.5*log10(flux) + 30
    em =  2.5*log10(1 + flux_err/flux)
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
    p = argparse.ArgumentParser(description="Convert ZTFCOSMO LC files to SNooPy format.")
    p.add_argument(
        "--matches-csv",
        default=str(light_curves_path("ztfcosmo_missing_coord_matches_within5arcsec.csv")),
        help="CSV with ZTF file names and matched missing sample SN names.",
    )
    p.add_argument(
        "--sample-csv",
        default=str(DATA_DIR / "supernovas_input.csv"),
        help="Sample table with matched_snname, redshift, rasn, decsn.",
    )
    p.add_argument(
        "--ztf-dir",
        default="/Users/lluisgalbany/Desktop/ztfcosmoidr/dr3/lightcurves",
        help="Directory containing *_LC.csv ZTFCOSMO lightcurves.",
    )
    p.add_argument(
        "--output-dir",
        default=str(light_curves_path("SNooPy")),
        help="Destination for generated SNooPy TXT files.",
    )
    p.add_argument(
        "--report-csv",
        default=str(light_curves_path("ztf_to_snoopy_conversion_report.csv")),
        help="Output conversion report.",
    )
    p.add_argument(
        "--snr-min",
        type=float,
        default=3.0,
        help="Minimum flux S/N cut for SNooPy mag-space output (default: 3).",
    )
    return p.parse_args()


def safe_name(s: str) -> str:
    out = re.sub(r"[^A-Za-z0-9._+-]+", "_", str(s).strip()).strip("_")
    return out or "SN"


def parse_mjd_max_init(path: Path) -> float | None:
    pat = re.compile(r"mjd_max_init:\\s*([0-9.+-Ee]+)")
    with path.open("r", errors="ignore") as f:
        for _ in range(80):
            line = f.readline()
            if not line:
                break
            m = pat.search(line)
            if m:
                try:
                    return float(m.group(1))
                except Exception:
                    return None
    return None


def apply_floor_errors(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    floor = {"ztfg": 0.025, "ztfr": 0.035, "ztfi": 0.060}
    for filt, frac in floor.items():
        mask = out["filter"] == filt
        out.loc[mask, "flux_err"] = np.sqrt(
            out.loc[mask, "flux_err"].values ** 2 + (out.loc[mask, "flux"].values * frac) ** 2
        )
    return out


def parse_ra_dec_to_deg(ra_raw: object, dec_raw: object) -> tuple[float, float]:
    """Parse RA/Dec that may be either decimal degrees or sexagesimal strings."""
    ra_txt = str(ra_raw).strip()
    dec_txt = str(dec_raw).strip()

    ra_num = pd.to_numeric(ra_txt, errors="coerce")
    dec_num = pd.to_numeric(dec_txt, errors="coerce")
    if np.isfinite(ra_num) and np.isfinite(dec_num):
        return float(ra_num), float(dec_num)

    # If RA looks like sexagesimal (contains ":"), interpret as hourangle.
    if ":" in ra_txt or "h" in ra_txt.lower():
        c = SkyCoord(ra_txt, dec_txt, unit=(u.hourangle, u.deg), frame="icrs")
    else:
        c = SkyCoord(ra_txt, dec_txt, unit=(u.deg, u.deg), frame="icrs")
    return float(c.ra.deg), float(c.dec.deg)


def main() -> int:
    args = parse_args()

    matches_csv = Path(args.matches_csv).expanduser().resolve()
    sample_csv = Path(args.sample_csv).expanduser().resolve()
    ztf_dir = Path(args.ztf_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    report_csv = Path(args.report_csv).expanduser().resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    report_csv.parent.mkdir(parents=True, exist_ok=True)

    matches = pd.read_csv(matches_csv)
    sample = pd.read_csv(sample_csv)
    sample["sample_sn"] = sample["matched_snname"].astype(str).str.strip()

    report_rows = []

    for _, row in matches.iterrows():
        file_name = str(row.get("file", "")).strip()
        sn_name = str(row.get("nearest_missing_sample", "")).strip()
        sep_arcsec = float(row.get("sep_arcsec", np.nan))
        if not file_name or not sn_name:
            continue

        lc_path = ztf_dir / file_name
        status = "ok"
        message = ""
        n_points_total = 0
        n_ztfg = n_ztfr = n_ztfi = 0
        output_file = output_dir / f"{safe_name(sn_name)}_raw.txt"

        try:
            if not lc_path.exists():
                raise FileNotFoundError(f"LC file not found: {lc_path}")

            srow = sample[sample["sample_sn"] == sn_name]
            if len(srow) == 0:
                raise ValueError(f"SN {sn_name} not found in sample table")

            z = float(pd.to_numeric(srow.iloc[0]["redshift"], errors="coerce"))
            ra, dec = parse_ra_dec_to_deg(srow.iloc[0]["rasn"], srow.iloc[0]["decsn"])
            if not (np.isfinite(z) and np.isfinite(ra) and np.isfinite(dec)):
                raise ValueError(f"Invalid z/ra/dec for {sn_name} in sample table")

            tmx = parse_mjd_max_init(lc_path)
            lc = pd.read_csv(lc_path, sep=r"\s+", comment="#", engine="python")
            req = {"mjd", "filter", "flux", "flux_err", "flag"}
            if not req.issubset(lc.columns):
                raise ValueError(f"Missing required columns in {file_name}: {req - set(lc.columns)}")

            # Recommended cuts from user script.
            lc = lc[(lc["flag"] & 1 == 0) & (lc["flag"] & 2 == 0) & (lc["flag"] & 4 == 0) & (lc["flag"] & 8 == 0) & (lc["flag"] & 16 == 0)]
            lc = apply_floor_errors(lc)

            # For mag-space SNooPy files, keep only reliable detections.
            snr = lc["flux"] / lc["flux_err"]
            lc = lc[(lc["flux_err"] > 0) & np.isfinite(snr) & (snr >= args.snr_min)]

            if tmx is not None and np.isfinite(tmx):
                lc = lc[(lc["mjd"] > tmx - 20.0) & (lc["mjd"] < tmx + 60.0)]

            lc = lc[lc["flux"] > 0].copy()
            if len(lc) == 0:
                raise ValueError("No photometry left after cuts")

            lc["m"] = -2.5 * np.log10(lc["flux"].values) + 30.0
            lc["em"] = 2.5 * np.log10(1.0 + lc["flux_err"].values / lc["flux"].values)
            lc = lc[np.isfinite(lc["m"]) & np.isfinite(lc["em"])].copy()
            if len(lc) == 0:
                raise ValueError("No finite m/em points after conversion")

            lines = [f"{sn_name} {z:.6f} {ra:.6f} {dec:.6f}"]
            for filt in ["ztfg", "ztfr", "ztfi"]:
                sub = lc[lc["filter"] == filt]
                if len(sub) == 0:
                    continue
                lines.append(f"filter {filt}")
                for _, rr in sub.iterrows():
                    lines.append(f"{rr['mjd']:.3f} {rr['m']:.3f} {rr['em']:.3f}")

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

        report_rows.append(
            {
                "sn_name": sn_name,
                "ztf_file": file_name,
                "sep_arcsec": sep_arcsec,
                "status": status,
                "message": message,
                "output_file": str(output_file) if status == "ok" else "",
                "n_points_total": n_points_total if status == "ok" else 0,
                "n_ztfg": n_ztfg if status == "ok" else 0,
                "n_ztfr": n_ztfr if status == "ok" else 0,
                "n_ztfi": n_ztfi if status == "ok" else 0,
            }
        )

    rep = pd.DataFrame(report_rows)
    rep.to_csv(report_csv, index=False)
    ok = int((rep["status"] == "ok").sum()) if len(rep) else 0
    failed = int((rep["status"] != "ok").sum()) if len(rep) else 0
    print(f"Converted ZTF->SNooPy files: {ok}")
    print(f"Failed conversions: {failed}")
    print(f"Report: {report_csv}")
    print(f"Output dir: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
