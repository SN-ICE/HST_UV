#!/usr/bin/env python3
"""Standalone Legacy Survey aperture-photometry test using inverse-variance maps.

This does not modify hostphot or the project pipeline. It measures 1 kpc Legacy
aperture fluxes directly from the local LegacySurvey cutouts and propagates the
uncertainty from the Legacy inverse-variance image. The outputs are intended for
comparison against the current hostphot Legacy measurements and PanSTARRS.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.wcs import WCS
from photutils.aperture import CircularAperture


COSMO = FlatLambdaCDM(H0=70, Om0=0.3)
LEGACY_ZP = 22.5
LEGACY_ZP_ERR_MAG = 3.9e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper-summary",
        default="outputs/sn_mag_summary_1kpc_full.csv",
        help="Summary table containing survey-specific Legacy and PanSTARRS values.",
    )
    parser.add_argument(
        "--images-root",
        default="outputs/images",
        help="Root directory containing per-object survey image folders.",
    )
    parser.add_argument(
        "--output-csv",
        default="outputs/legacy_invvar_test_comparison.csv",
        help="Where to write the comparison table.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit on number of overlap objects to process.",
    )
    return parser.parse_args()


def parse_mag_with_err(text: object) -> tuple[float | None, float | None, bool]:
    if pd.isna(text):
        return None, None, False
    s = str(text).strip()
    if not s or s.lower() == "nan":
        return None, None, False
    if s.startswith("(") and s.endswith(")"):
        try:
            return float(s[1:-1]), None, True
        except ValueError:
            return None, None, False
    if "(" in s and s.endswith(")"):
        mag_text, err_text = s.split("(", 1)
        try:
            return float(mag_text.strip()), float(err_text[:-1].strip()), False
        except ValueError:
            return None, None, False
    try:
        return float(s), None, False
    except ValueError:
        return None, None, False


def calc_aperture_size_arcsec(z: float, ap_radius_kpc: float = 1.0) -> float:
    transv_sep_per_arcmin = COSMO.kpc_proper_per_arcmin(z)
    transv_sep_per_arcsec = transv_sep_per_arcmin.to(u.kpc / u.arcsec)
    return (ap_radius_kpc * u.kpc / transv_sep_per_arcsec).value


def aperture_sum_and_var(
    image: np.ndarray,
    invvar: np.ndarray,
    x: float,
    y: float,
    radius_pix: float,
) -> tuple[float, float]:
    aperture = CircularAperture((x, y), r=radius_pix)
    mask = aperture.to_mask(method="exact")
    weighted_image = mask.cutout(image, fill_value=np.nan)
    weighted_invvar = mask.cutout(invvar, fill_value=0.0)
    weights = mask.data

    if weighted_image is None or weighted_invvar is None:
        raise RuntimeError("aperture cutout failed")

    valid = np.isfinite(weighted_image) & np.isfinite(weights) & (weighted_invvar > 0)
    if not np.any(valid):
        raise RuntimeError("no valid pixels with positive inverse variance in aperture")

    flux = float(np.nansum(weighted_image[valid] * weights[valid]))
    var = float(np.nansum((weights[valid] ** 2) / weighted_invvar[valid]))
    return flux, var


def magnitude_from_flux(flux: float, flux_err: float, zp: float = LEGACY_ZP) -> tuple[float | None, float | None]:
    if not math.isfinite(flux) or flux <= 0:
        return None, None
    mag = -2.5 * math.log10(flux) + zp
    if not math.isfinite(flux_err) or flux_err <= 0:
        return mag, None
    mag_err = abs(2.5 / math.log(10) * flux_err / flux)
    mag_err = math.sqrt(mag_err**2 + LEGACY_ZP_ERR_MAG**2)
    return mag, mag_err


def measure_legacy_invvar(sn_name: str, ra: float, dec: float, z: float, images_root: Path) -> dict[str, object]:
    fits_path = images_root / sn_name / "LegacySurvey" / "LegacySurvey_r.fits"
    if not fits_path.exists():
        return {"legacy_test_status": "missing_fits"}

    with fits.open(fits_path) as hdul:
        image = np.asarray(hdul[0].data, dtype=float)
        if len(hdul) < 2 or hdul[1].data is None:
            return {"legacy_test_status": "missing_invvar"}
        invvar = np.asarray(hdul[1].data, dtype=float)
        wcs = WCS(hdul[0].header, naxis=2)

    if image.ndim != 2 or invvar.ndim != 2:
        return {"legacy_test_status": "invalid_ndim"}

    x, y = wcs.wcs_world2pix(ra, dec, 0)
    radius_arcsec = calc_aperture_size_arcsec(z, 1.0)
    # Legacy cutouts use 0.262 arcsec/pixel in hostphot.
    radius_pix = radius_arcsec / 0.262

    try:
        flux, var = aperture_sum_and_var(image, invvar, x, y, radius_pix)
    except Exception as exc:
        return {"legacy_test_status": f"measure_failed:{exc}"}

    flux_err = math.sqrt(var) if var > 0 else np.nan
    mag, mag_err = magnitude_from_flux(flux, flux_err, LEGACY_ZP)
    snr = flux / flux_err if flux_err and math.isfinite(flux_err) and flux_err > 0 else np.nan

    return {
        "legacy_test_status": "ok",
        "legacy_test_flux": flux,
        "legacy_test_flux_err": flux_err,
        "legacy_test_snr": snr,
        "legacy_test_mag": mag,
        "legacy_test_mag_err": mag_err,
        "legacy_test_radius_pix": radius_pix,
    }


def main() -> int:
    args = parse_args()
    summary = pd.read_csv(args.paper_summary)
    images_root = Path(args.images_root)

    summary["LegacySurvey r mag"] = summary["LegacySurvey r mag"].fillna("").astype(str).str.strip()
    summary["PanSTARRS r mag"] = summary["PanSTARRS r mag"].fillna("").astype(str).str.strip()
    overlap = summary[
        (summary["LegacySurvey r mag"] != "") & (summary["PanSTARRS r mag"] != "")
    ].copy()
    if args.limit and args.limit > 0:
        overlap = overlap.head(args.limit).copy()

    rows: list[dict[str, object]] = []
    for _, row in overlap.iterrows():
        sn = str(row["SN name"])
        legacy_mag, legacy_mag_err, legacy_ul = parse_mag_with_err(row["LegacySurvey r mag"])
        pan_mag, pan_mag_err, pan_ul = parse_mag_with_err(row["PanSTARRS r mag"])

        # Use the local survey photometry files for coordinates/redshift.
        local_phot_path = images_root / sn / "LegacySurvey" / "local_photometry.csv"
        if not local_phot_path.exists():
            rows.append({"SN name": sn, "legacy_test_status": "missing_local_phot"})
            continue
        local_phot = pd.read_csv(local_phot_path).iloc[-1]
        ra = float(local_phot["ra"])
        dec = float(local_phot["dec"])
        z = float(local_phot["redshift"])

        measured = measure_legacy_invvar(sn, ra, dec, z, images_root)
        rows.append(
            {
                "SN name": sn,
                "redshift": z,
                "legacy_hostphot_mag": legacy_mag,
                "legacy_hostphot_mag_err": legacy_mag_err,
                "legacy_hostphot_is_ul": legacy_ul,
                "panstarrs_mag": pan_mag,
                "panstarrs_mag_err": pan_mag_err,
                "panstarrs_is_ul": pan_ul,
                **measured,
            }
        )

    result = pd.DataFrame(rows).sort_values("SN name")
    if "legacy_hostphot_mag_err" in result.columns and "legacy_test_mag_err" in result.columns:
        result["hostphot_over_test_err_ratio"] = result["legacy_hostphot_mag_err"] / result["legacy_test_mag_err"]
    if "legacy_test_mag_err" in result.columns and "panstarrs_mag_err" in result.columns:
        result["legacy_test_over_panstarrs_err_ratio"] = result["legacy_test_mag_err"] / result["panstarrs_mag_err"]
    if "legacy_hostphot_mag_err" in result.columns and "panstarrs_mag_err" in result.columns:
        result["legacy_hostphot_over_panstarrs_err_ratio"] = result["legacy_hostphot_mag_err"] / result["panstarrs_mag_err"]

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)

    ok = result[result["legacy_test_status"] == "ok"].copy()
    print(f"Overlap rows: {len(overlap)}")
    print(f"Legacy test ok: {len(ok)}")
    if not ok.empty:
        med_host = pd.to_numeric(ok["legacy_hostphot_mag_err"], errors="coerce").median()
        med_test = pd.to_numeric(ok["legacy_test_mag_err"], errors="coerce").median()
        med_pan = pd.to_numeric(ok["panstarrs_mag_err"], errors="coerce").median()
        med_ratio_host_test = pd.to_numeric(ok["hostphot_over_test_err_ratio"], errors="coerce").median()
        med_ratio_test_pan = pd.to_numeric(ok["legacy_test_over_panstarrs_err_ratio"], errors="coerce").median()
        med_ratio_host_pan = pd.to_numeric(ok["legacy_hostphot_over_panstarrs_err_ratio"], errors="coerce").median()
        print(f"Median Legacy hostphot mag err: {med_host:.4f}")
        print(f"Median Legacy invvar-test mag err: {med_test:.4f}")
        print(f"Median PanSTARRS mag err: {med_pan:.4f}")
        print(f"Median hostphot/test err ratio: {med_ratio_host_test:.2f}")
        print(f"Median test/PanSTARRS err ratio: {med_ratio_test_pan:.2f}")
        print(f"Median hostphot/PanSTARRS err ratio: {med_ratio_host_pan:.2f}")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
