#!/usr/bin/env python3
"""
Crossmatch HST FITS footprints against the TNS public objects CSV locally.

Workflow:
1) Scan FITS files under the provided roots (recursive).
2) For each FITS, compute WCS center + a "full-footprint" cone radius
   (max separation from center to any corner).
3) Load the TNS public CSV (or CSV inside a .zip/.gz).
4) Cone-search all FITS centers against TNS, then filter by per-FITS radius.

Output: CSV with FITS file, footprint center/radius, and matching TNS objects.
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, search_around_sky
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS

from project_paths import DATA_DIR, hst_program_dir


def _iter_fits_files(roots: Iterable[Path], patterns: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for root in roots:
        for pattern in patterns:
            files.extend(sorted(root.rglob(pattern)))
    # De-duplicate while preserving order
    seen = set()
    unique = []
    for p in files:
        if p not in seen:
            unique.append(p)
            seen.add(p)
    return unique


def _select_celestial_hdu(hdul: fits.HDUList) -> Tuple[Optional[fits.ImageHDU], Optional[WCS]]:
    for hdu in hdul:
        header = hdu.header
        naxis = header.get("NAXIS", 0)
        if naxis < 2:
            continue
        if header.get("NAXIS1") is None or header.get("NAXIS2") is None:
            continue
        try:
            wcs = WCS(header)
        except Exception:
            continue
        if wcs.has_celestial:
            return hdu, wcs
    return None, None


def _fits_center_and_radius(fits_path: Path) -> Optional[Tuple[float, float, float]]:
    """
    Returns (center_ra_deg, center_dec_deg, radius_arcsec) or None if WCS missing.
    """
    with fits.open(fits_path, memmap=False) as hdul:
        hdu, wcs = _select_celestial_hdu(hdul)
        if hdu is None or wcs is None:
            return None

        header = hdu.header
        nx = header.get("NAXIS1")
        ny = header.get("NAXIS2")
        if not nx or not ny:
            return None

        # Pixel coords are 0-based for WCS.pixel_to_world
        cx = (nx - 1) / 2.0
        cy = (ny - 1) / 2.0
        center = wcs.pixel_to_world(cx, cy)

        corners_x = np.array([0, 0, nx - 1, nx - 1], dtype=float)
        corners_y = np.array([0, ny - 1, 0, ny - 1], dtype=float)
        corners = wcs.pixel_to_world(corners_x, corners_y)

        # Max separation from center to corners => full-footprint cone radius
        sep = center.separation(corners)
        radius = sep.max().to(u.arcsec)

        return center.ra.deg, center.dec.deg, float(radius.value)


def _open_tns_csv(tns_path: Path) -> io.IOBase:
    """
    Returns a file-like handle for the TNS CSV.
    Supports .csv, .csv.gz, and .zip (containing a CSV).
    """
    if tns_path.suffix.lower() == ".zip":
        zf = zipfile.ZipFile(tns_path)
        # pick the first .csv inside
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError(f"No .csv found inside {tns_path}")
        return zf.open(csv_names[0], "r")
    return tns_path.open("rb")


def _read_tns_csv(tns_path: Path) -> Tuple[pd.DataFrame, SkyCoord]:
    """
    Reads TNS public objects CSV and returns (df, coords).
    The CSV has a first line with timestamp and a second line with headers,
    so we skip the first line.
    """
    with _open_tns_csv(tns_path) as fh:
        header_df = pd.read_csv(fh, skiprows=1, nrows=0)
        cols = [c.strip().lower() for c in header_df.columns]

    # Column detection based on the TNS public objects CSV format.
    col_map = {c.lower(): c for c in header_df.columns}
    ra_col = col_map.get("ra")
    dec_col = col_map.get("declination")
    name_col = col_map.get("name")
    type_col = col_map.get("type")
    redshift_col = col_map.get("redshift")
    discovery_col = col_map.get("discoverydate")

    if ra_col is None or dec_col is None or name_col is None:
        raise ValueError("Required columns not found in TNS CSV (expected: name, ra, declination).")

    usecols = [c for c in [name_col, ra_col, dec_col, type_col, redshift_col, discovery_col] if c]

    with _open_tns_csv(tns_path) as fh:
        df = pd.read_csv(fh, skiprows=1, usecols=usecols)

    # Clean and parse RA/Dec
    ra_raw = df[ra_col].astype(str).str.strip()
    dec_raw = df[dec_col].astype(str).str.strip()

    ra_num = pd.to_numeric(ra_raw, errors="coerce")
    dec_num = pd.to_numeric(dec_raw, errors="coerce")

    numeric_ok = ra_num.notna().mean() > 0.95 and dec_num.notna().mean() > 0.95
    if numeric_ok:
        coords = SkyCoord(ra=ra_num.values * u.deg, dec=dec_num.values * u.deg, frame="icrs")
    else:
        coords = SkyCoord(ra=ra_raw.values, dec=dec_raw.values, unit=(u.hourangle, u.deg), frame="icrs")

    # Drop rows with invalid coords
    valid = np.isfinite(coords.ra.deg) & np.isfinite(coords.dec.deg)
    df = df.loc[valid].reset_index(drop=True)
    coords = SkyCoord(ra=coords.ra.deg[valid] * u.deg, dec=coords.dec.deg[valid] * u.deg, frame="icrs")

    return df, coords


def main() -> int:
    parser = argparse.ArgumentParser(description="Crossmatch HST FITS footprints against TNS public objects CSV.")
    parser.add_argument(
        "--roots",
        nargs="+",
        default=[str(hst_program_dir("16741")), str(hst_program_dir("17179"))],
        help="Root directories to scan recursively for FITS files.",
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=["*_drc.fits"],
        help="Glob patterns to match FITS files (e.g., *_drc.fits *_drz.fits).",
    )
    parser.add_argument(
        "--tns-csv",
        default=str(DATA_DIR / "tns_public_objects.csv.zip"),
        help="Path to TNS public objects CSV (.csv, .csv.gz, or .zip containing CSV).",
    )
    parser.add_argument(
        "--output",
        default=str(DATA_DIR / "tns_crossmatch.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--only-sne",
        action="store_true",
        help="Keep only rows with TNS type containing 'SN' (case-insensitive).",
    )
    args = parser.parse_args()

    roots = [Path(r).expanduser().resolve() for r in args.roots]
    patterns = args.patterns

    fits_files = _iter_fits_files(roots, patterns)
    if not fits_files:
        print("No FITS files found with the given roots/patterns.")
        return 1

    centers = []
    radii_arcsec = []
    fits_paths = []

    for fp in fits_files:
        res = _fits_center_and_radius(fp)
        if res is None:
            continue
        ra_deg, dec_deg, radius_arcsec = res
        fits_paths.append(fp)
        centers.append((ra_deg, dec_deg))
        radii_arcsec.append(radius_arcsec)

    if not centers:
        print("No FITS files with usable celestial WCS were found.")
        return 1

    centers_coord = SkyCoord(
        ra=np.array([c[0] for c in centers]) * u.deg,
        dec=np.array([c[1] for c in centers]) * u.deg,
        frame="icrs",
    )
    radii = np.array(radii_arcsec) * u.arcsec

    tns_path = Path(args.tns_csv).expanduser().resolve()
    tns_df, tns_coords = _read_tns_csv(tns_path)

    if args.only_sne and "type" in tns_df.columns:
        mask = tns_df["type"].astype(str).str.contains("SN", case=False, na=False)
        tns_df = tns_df.loc[mask].reset_index(drop=True)
        tns_coords = SkyCoord(
            ra=tns_coords.ra.deg[mask.values] * u.deg,
            dec=tns_coords.dec.deg[mask.values] * u.deg,
            frame="icrs",
        )

    max_radius = radii.max()
    idx_center, idx_tns, sep2d, _ = search_around_sky(centers_coord, tns_coords, max_radius)
    within = sep2d <= radii[idx_center]

    idx_center = idx_center[within]
    idx_tns = idx_tns[within]
    sep2d = sep2d[within]

    if len(idx_center) == 0:
        print("No matches found within any FITS footprint radius.")
        return 0

    out = pd.DataFrame(
        {
            "fits_file": [str(fits_paths[i]) for i in idx_center],
            "center_ra_deg": centers_coord.ra.deg[idx_center],
            "center_dec_deg": centers_coord.dec.deg[idx_center],
            "radius_arcsec": radii.to(u.arcsec).value[idx_center],
            "tns_name": tns_df["name"].values[idx_tns],
            "tns_ra_deg": tns_coords.ra.deg[idx_tns],
            "tns_dec_deg": tns_coords.dec.deg[idx_tns],
            "sep_arcsec": sep2d.to(u.arcsec).value,
        }
    )

    # Optional columns if present
    for col in ["type", "redshift", "discoverydate"]:
        if col in tns_df.columns:
            out[f"tns_{col}"] = tns_df[col].values[idx_tns]

    out_path = Path(args.output).expanduser().resolve()
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} matches to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
