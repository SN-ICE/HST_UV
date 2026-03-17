#!/usr/bin/env python3
"""Regenerate HST+optical diagnostic panels from the current final summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import astropy.units as u
from astropy.cosmology import WMAP9 as cosmo
import pandas as pd

from hst_uv_photometry_pipeline import plot_supernova_panels, sanitize_name_for_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default="outputs/photometry_summary_keep.csv")
    parser.add_argument("--paper-full", default="outputs/sn_mag_summary_1kpc_full.csv")
    parser.add_argument("--plots-dir", default="outputs/plots")
    parser.add_argument("--images-root", default="outputs/images")
    parser.add_argument("--names-file", default="")
    parser.add_argument("--fov-arcsec", type=float, default=30.0)
    return parser.parse_args()


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def main() -> int:
    args = parse_args()
    summary = pd.read_csv(args.summary).drop_duplicates(subset=["Supernova"], keep="last")
    paper = pd.read_csv(args.paper_full).drop_duplicates(subset=["SN name"], keep="last")
    paper = paper.rename(columns={"SN name": "Supernova"})

    merged = summary.merge(
        paper[["Supernova", "UV mag", "Adopted r survey", "Adopted r mag"]],
        on="Supernova",
        how="left",
    )

    plots_dir = Path(args.plots_dir)
    images_root = Path(args.images_root)
    plots_dir.mkdir(parents=True, exist_ok=True)
    selected_names: set[str] | None = None
    if args.names_file:
        names_path = Path(args.names_file)
        selected_names = {
            line.strip()
            for line in names_path.read_text().splitlines()
            if line.strip()
        }

    generated = 0
    skipped = 0

    for _, row in merged.iterrows():
        supernova_name = str(row.get("Supernova"))
        if selected_names is not None and supernova_name not in selected_names:
            continue
        if str(row.get("status", "")).strip() != "ok":
            skipped += 1
            continue

        hst_file = Path(str(row.get("file", ""))).expanduser()
        if not hst_file.exists():
            skipped += 1
            continue

        z = float(row.get("z"))
        if not (z > 0):
            skipped += 1
            continue

        survey = clean_text(row.get("Adopted r survey"))
        uv_text = clean_text(row.get("UV mag"))
        r_text = clean_text(row.get("Adopted r mag"))

        rband_file = ""
        rband_label = ""
        if survey:
            candidate = images_root / supernova_name / survey / f"{survey}_r.fits"
            if candidate.exists():
                rband_file = str(candidate)
                rband_label = f"{survey} r"

        da = cosmo.angular_diameter_distance(z).to(u.kpc)
        aperture_1kpc_deg = [(((1.0 * u.kpc) / da) * u.rad).to(u.deg).value]
        out = plots_dir / f"{sanitize_name_for_file(supernova_name)}_plot.png"

        plot_supernova_panels(
            hst_file=str(hst_file),
            ra=float(row.get("RA(degrees)")),
            dec=float(row.get("DEC(degrees)")),
            supernova_name=supernova_name,
            apertures_deg=aperture_1kpc_deg,
            output_name=str(out),
            rband_file=rband_file,
            rband_label=rband_label,
            hst_text=uv_text,
            rband_text=r_text,
            fov_arcsec=float(args.fov_arcsec),
        )
        generated += 1
        if generated % 50 == 0:
            print(f"Generated {generated} plots...")

    print(f"Generated plots: {generated}")
    print(f"Skipped rows: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
