#!/usr/bin/env python3
"""Standalone Legacy Survey aperture-photometry test with blank-aperture errors.

This script keeps the same 1 kpc aperture flux measurement as the invvar test,
but estimates the uncertainty from the larger of:
1) the inverse-variance aperture error, and
2) the empirical RMS of many blank apertures around the SN position.
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
    parser.add_argument("--paper-summary", default="outputs/sn_mag_summary_1kpc_full.csv")
    parser.add_argument("--images-root", default="outputs/images")
    parser.add_argument("--output-csv", default="outputs/legacy_blank_aperture_test_comparison.csv")
    parser.add_argument(
        "--all-legacy",
        action="store_true",
        help="Run on all objects with local LegacySurvey products, not only the Legacy+PanSTARRS overlap sample.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--n-apertures", type=int, default=80)
    parser.add_argument("--ring-rmin-factor", type=float, default=2.5)
    parser.add_argument("--ring-rmax-factor", type=float, default=6.0)
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


def aperture_weighted_sum(data: np.ndarray, x: float, y: float, r: float) -> tuple[float, np.ndarray, tuple[slice, slice] | None]:
    ap = CircularAperture((x, y), r=r)
    mask = ap.to_mask(method="exact")
    cutout = mask.cutout(data, fill_value=np.nan)
    if cutout is None:
        raise RuntimeError("aperture cutout failed")
    weights = mask.data
    valid = np.isfinite(cutout) & np.isfinite(weights)
    if not np.any(valid):
        raise RuntimeError("no valid pixels in aperture")
    flux = float(np.nansum(cutout[valid] * weights[valid]))
    bbox = mask.bbox
    slices = (slice(bbox.iymin, bbox.iymax), slice(bbox.ixmin, bbox.ixmax))
    return flux, weights, slices


def aperture_invvar_variance(invvar: np.ndarray, weights: np.ndarray, slices: tuple[slice, slice] | None) -> float:
    if slices is None:
        raise RuntimeError("missing aperture slices")
    cutout = invvar[slices]
    valid = np.isfinite(cutout) & np.isfinite(weights) & (cutout > 0)
    if not np.any(valid):
        raise RuntimeError("no valid invvar pixels in aperture")
    return float(np.nansum((weights[valid] ** 2) / cutout[valid]))


def blank_aperture_rms(
    image: np.ndarray,
    x0: float,
    y0: float,
    radius_pix: float,
    n_apertures: int,
    rmin_factor: float,
    rmax_factor: float,
) -> tuple[float | None, int]:
    rng = np.random.default_rng(12345)
    ny, nx = image.shape
    blank_fluxes = []
    max_trials = max(400, n_apertures * 20)
    rmin = rmin_factor * radius_pix
    rmax = rmax_factor * radius_pix

    for _ in range(max_trials):
        if len(blank_fluxes) >= n_apertures:
            break
        theta = rng.uniform(0, 2 * math.pi)
        rho = rng.uniform(rmin, rmax)
        x = x0 + rho * math.cos(theta)
        y = y0 + rho * math.sin(theta)
        if x - radius_pix < 0 or x + radius_pix >= nx or y - radius_pix < 0 or y + radius_pix >= ny:
            continue
        try:
            flux, _, _ = aperture_weighted_sum(image, x, y, radius_pix)
        except Exception:
            continue
        if not math.isfinite(flux):
            continue
        blank_fluxes.append(flux)

    if len(blank_fluxes) < 10:
        return None, len(blank_fluxes)

    arr = np.asarray(blank_fluxes, dtype=float)
    med = np.nanmedian(arr)
    rms = 1.4826 * np.nanmedian(np.abs(arr - med))
    return float(rms), len(blank_fluxes)


def magnitude_from_flux(flux: float, flux_err: float, zp: float = LEGACY_ZP) -> tuple[float | None, float | None]:
    if not math.isfinite(flux) or flux <= 0:
        return None, None
    mag = -2.5 * math.log10(flux) + zp
    if not math.isfinite(flux_err) or flux_err <= 0:
        return mag, None
    mag_err = abs(2.5 / math.log(10) * flux_err / flux)
    mag_err = math.sqrt(mag_err**2 + LEGACY_ZP_ERR_MAG**2)
    return mag, mag_err


def measure_legacy_blank_aperture(
    sn_name: str,
    ra: float,
    dec: float,
    z: float,
    images_root: Path,
    n_apertures: int,
    rmin_factor: float,
    rmax_factor: float,
) -> dict[str, object]:
    fits_path = images_root / sn_name / "LegacySurvey" / "LegacySurvey_r.fits"
    if not fits_path.exists():
        return {"legacy_blank_status": "missing_fits"}

    with fits.open(fits_path) as hdul:
        image = np.asarray(hdul[0].data, dtype=float)
        if len(hdul) < 2 or hdul[1].data is None:
            return {"legacy_blank_status": "missing_invvar"}
        invvar = np.asarray(hdul[1].data, dtype=float)
        wcs = WCS(hdul[0].header, naxis=2)

    x, y = wcs.wcs_world2pix(ra, dec, 0)
    radius_arcsec = calc_aperture_size_arcsec(z, 1.0)
    radius_pix = radius_arcsec / 0.262

    try:
        flux, weights, slices = aperture_weighted_sum(image, x, y, radius_pix)
        invvar_var = aperture_invvar_variance(invvar, weights, slices)
    except Exception as exc:
        return {"legacy_blank_status": f"measure_failed:{exc}"}

    invvar_err = math.sqrt(invvar_var) if invvar_var > 0 else np.nan
    blank_rms, n_used = blank_aperture_rms(
        image=image,
        x0=x,
        y0=y,
        radius_pix=radius_pix,
        n_apertures=n_apertures,
        rmin_factor=rmin_factor,
        rmax_factor=rmax_factor,
    )
    if blank_rms is None:
        total_err = invvar_err
        status = "ok_invvar_only"
    else:
        total_err = max(invvar_err, blank_rms)
        status = "ok"

    mag, mag_err = magnitude_from_flux(flux, total_err, LEGACY_ZP)
    snr = flux / total_err if total_err and math.isfinite(total_err) and total_err > 0 else np.nan

    return {
        "legacy_blank_status": status,
        "legacy_blank_flux": flux,
        "legacy_blank_flux_err": total_err,
        "legacy_blank_snr": snr,
        "legacy_blank_mag": mag,
        "legacy_blank_mag_err": mag_err,
        "legacy_blank_invvar_err": invvar_err,
        "legacy_blank_blank_rms": blank_rms,
        "legacy_blank_nblank": n_used,
        "legacy_blank_radius_pix": radius_pix,
    }


def main() -> int:
    args = parse_args()
    summary = pd.read_csv(args.paper_summary)
    images_root = Path(args.images_root)

    summary["LegacySurvey r mag"] = summary["LegacySurvey r mag"].fillna("").astype(str).str.strip()
    if "PanSTARRS r mag" in summary.columns:
        summary["PanSTARRS r mag"] = summary["PanSTARRS r mag"].fillna("").astype(str).str.strip()
    else:
        summary["PanSTARRS r mag"] = ""

    if args.all_legacy:
        names_with_legacy = {
            p.parent.parent.name
            for p in images_root.glob("*/LegacySurvey/local_photometry.csv")
        }
        overlap = summary[summary["SN name"].isin(names_with_legacy)].copy()
    else:
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
        local_phot_path = images_root / sn / "LegacySurvey" / "local_photometry.csv"
        if not local_phot_path.exists():
            rows.append({"SN name": sn, "legacy_blank_status": "missing_local_phot"})
            continue
        local_phot = pd.read_csv(local_phot_path).iloc[-1]
        ra = float(local_phot["ra"])
        dec = float(local_phot["dec"])
        z = float(local_phot["redshift"])

        measured = measure_legacy_blank_aperture(
            sn_name=sn,
            ra=ra,
            dec=dec,
            z=z,
            images_root=images_root,
            n_apertures=args.n_apertures,
            rmin_factor=args.ring_rmin_factor,
            rmax_factor=args.ring_rmax_factor,
        )
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
    if "legacy_hostphot_mag_err" in result.columns and "legacy_blank_mag_err" in result.columns:
        result["hostphot_over_blank_err_ratio"] = result["legacy_hostphot_mag_err"] / result["legacy_blank_mag_err"]
    if "legacy_blank_mag_err" in result.columns and "panstarrs_mag_err" in result.columns:
        result["blank_over_panstarrs_err_ratio"] = result["legacy_blank_mag_err"] / result["panstarrs_mag_err"]

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)

    ok = result[result["legacy_blank_status"].isin(["ok", "ok_invvar_only"])].copy()
    print(f"Input rows: {len(overlap)}")
    print(f"Legacy blank-aperture test ok: {len(ok)}")
    if not ok.empty:
        med_host = pd.to_numeric(ok["legacy_hostphot_mag_err"], errors="coerce").median()
        med_blank = pd.to_numeric(ok["legacy_blank_mag_err"], errors="coerce").median()
        med_pan = pd.to_numeric(ok["panstarrs_mag_err"], errors="coerce").median()
        med_ratio_host_blank = pd.to_numeric(ok["hostphot_over_blank_err_ratio"], errors="coerce").median()
        med_ratio_blank_pan = pd.to_numeric(ok["blank_over_panstarrs_err_ratio"], errors="coerce").median()
        print(f"Median Legacy hostphot mag err: {med_host:.4f}")
        print(f"Median Legacy blank-aperture mag err: {med_blank:.4f}")
        if pd.notna(med_pan):
            print(f"Median PanSTARRS mag err: {med_pan:.4f}")
        print(f"Median hostphot/blank err ratio: {med_ratio_host_blank:.2f}")
        if pd.notna(med_ratio_blank_pan):
            print(f"Median blank/PanSTARRS err ratio: {med_ratio_blank_pan:.2f}")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
