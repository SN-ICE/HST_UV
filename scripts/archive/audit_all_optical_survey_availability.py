#!/usr/bin/env python3
"""Audit local and live r-band image availability for all optical surveys."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from astropy.coordinates import SkyCoord
import astropy.units as u
from hostphot.cutouts.des import get_DES_urls
from hostphot.cutouts.panstarrs import get_PS1_urls
from hostphot.cutouts.skymapper import get_SkyMapper_urls


SURVEYS = ("LegacySurvey", "DES", "PanSTARRS", "SkyMapper")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default="data/supernovas_input.csv")
    parser.add_argument("--images-root", default="outputs/images")
    parser.add_argument(
        "--output-csv",
        default="outputs/all_optical_rband_availability_audit.csv",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=20.0)
    return parser.parse_args()


def legacy_available(ra: float, dec: float, timeout: float) -> tuple[bool | None, str]:
    pixel_scale = 0.262
    size_arcsec = 120.0
    size_pixels = int(size_arcsec / pixel_scale)
    url = (
        "https://www.legacysurvey.org/viewer/fits-cutout"
        f"?ra={ra}&dec={dec}&layer=ls-dr10&pixscale={pixel_scale}&bands=r&size={size_pixels}"
    )
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        content_type = response.headers.get("content-type", "")
        ok = response.status_code == 200 and "fits" in content_type.lower()
        message = f"http_{response.status_code}:{content_type or 'unknown'}"
        response.close()
        return ok, message
    except Exception as exc:  # pragma: no cover - network errors are expected in some environments
        return None, f"{type(exc).__name__}:{exc}"


def des_available(ra: float, dec: float) -> tuple[bool | None, str]:
    try:
        urls, _ = get_DES_urls(ra, dec, fov=0.05, filters="r")
        ok = bool(urls)
        return ok, "ok" if ok else "empty"
    except Exception as exc:  # pragma: no cover
        return None, f"{type(exc).__name__}:{exc}"


def panstarrs_available(ra: float, dec: float) -> tuple[bool | None, str]:
    try:
        urls = get_PS1_urls(ra, dec, size=3, filters="r")
        ok = bool(urls)
        return ok, "ok" if ok else "empty"
    except Exception as exc:  # pragma: no cover
        return None, f"{type(exc).__name__}:{exc}"


def skymapper_available(ra: float, dec: float) -> tuple[bool | None, str]:
    try:
        urls = get_SkyMapper_urls(ra, dec, fov=0.05, filters="r")
        ok = bool(urls and urls[0])
        return ok, "ok" if ok else "empty"
    except Exception as exc:  # pragma: no cover
        return None, f"{type(exc).__name__}:{exc}"


def live_check(survey: str, ra: float, dec: float, timeout: float) -> tuple[bool | None, str]:
    if survey == "LegacySurvey":
        return legacy_available(ra, dec, timeout)
    if survey == "DES":
        return des_available(ra, dec)
    if survey == "PanSTARRS":
        return panstarrs_available(ra, dec)
    if survey == "SkyMapper":
        return skymapper_available(ra, dec)
    raise ValueError(f"Unsupported survey: {survey}")


def parse_coords(ra_value: object, dec_value: object) -> tuple[float, float]:
    try:
        return float(ra_value), float(dec_value)
    except Exception:
        coord = SkyCoord(str(ra_value), str(dec_value), unit=(u.hourangle, u.deg))
        return coord.ra.deg, coord.dec.deg


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_csv)
    images_root = Path(args.images_root)
    output_path = Path(args.output_csv)

    sample = pd.read_csv(input_path)
    records: list[dict[str, Any]] = []
    futures = {}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row in sample.itertuples(index=False):
            sn = str(row.matched_snname)
            ra, dec = parse_coords(row.rasn, row.decsn)
            for survey in SURVEYS:
                fits_path = images_root / sn / survey / f"{survey}_r.fits"
                local_fits = fits_path.exists()
                record = {
                    "Supernova": sn,
                    "ra_deg": ra,
                    "dec_deg": dec,
                    "survey": survey,
                    "local_fits": local_fits,
                    "fits_path": str(fits_path) if local_fits else "",
                    "live_checked": False,
                    "live_available": pd.NA,
                    "live_message": "",
                }
                records.append(record)
                if not local_fits:
                    future = pool.submit(live_check, survey, ra, dec, args.timeout)
                    futures[future] = len(records) - 1

        for future in as_completed(futures):
            idx = futures[future]
            ok, message = future.result()
            records[idx]["live_checked"] = True
            records[idx]["live_available"] = ok
            records[idx]["live_message"] = message

    audit = pd.DataFrame(records)
    audit.to_csv(output_path, index=False)

    for survey in SURVEYS:
        subset = audit[audit["survey"] == survey]
        local_count = int(subset["local_fits"].sum())
        live_yes = int((subset["live_available"] == True).sum())  # noqa: E712
        live_no = int((subset["live_available"] == False).sum())  # noqa: E712
        live_unknown = int(subset["live_checked"].sum()) - live_yes - live_no
        print(
            f"{survey}: local={local_count} live_yes={live_yes} live_no={live_no} live_unknown={live_unknown}"
        )
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
