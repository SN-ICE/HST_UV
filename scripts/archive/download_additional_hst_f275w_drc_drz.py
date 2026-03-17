#!/usr/bin/env python3
"""
Download HST F275W drc/drz FITS products for the 32 CSPII SN shortlist.

Input catalogs are produced by:
  scripts/archive/find_additional_hst_f275w_for_cspii.py

Selection logic:
- Start from all observation-level matches for SN in the 32-object summary list.
- Keep unique matched obs_id values.
- Query MAST for those observations.
- Keep only products whose filename ends with _drc.fits or _drz.fits.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from astroquery.mast import Observations
from astropy.table import vstack

from project_paths import additional_hst_download_dir, light_curves_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download additional HST F275W drc/drz files for CSPII matches.")
    p.add_argument(
        "--matches-csv",
        default=str(light_curves_path("cspii_not_in_sample_hst_f275w_matches.csv")),
        help="Observation-level match table.",
    )
    p.add_argument(
        "--summary-csv",
        default=str(light_curves_path("cspii_not_in_sample_hst_f275w_summary.csv")),
        help="SN-level summary table (32 SN list).",
    )
    p.add_argument(
        "--download-dir",
        default=str(additional_hst_download_dir()),
        help="Directory where MAST products will be downloaded.",
    )
    p.add_argument(
        "--products-csv",
        default=str(light_curves_path("cspii32_hst_f275w_drcdrz_products.csv")),
        help="Output CSV with product metadata selected for download.",
    )
    p.add_argument(
        "--manifest-csv",
        default=str(light_curves_path("cspii32_hst_f275w_drcdrz_download_manifest.csv")),
        help="Output CSV with download manifest.",
    )
    p.add_argument(
        "--obsids-csv",
        default=str(light_curves_path("cspii32_hst_f275w_obsids.csv")),
        help="Output CSV with unique matched obs_id values.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="Number of obs_id values per MAST query batch.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not download; only build product list and size estimate.",
    )
    return p.parse_args()


def chunked(items: list[str], n: int) -> list[list[str]]:
    return [items[i : i + n] for i in range(0, len(items), n)]


def main() -> int:
    args = parse_args()

    matches_csv = Path(args.matches_csv).expanduser().resolve()
    summary_csv = Path(args.summary_csv).expanduser().resolve()
    download_dir = Path(args.download_dir).expanduser().resolve()
    products_csv = Path(args.products_csv).expanduser().resolve()
    manifest_csv = Path(args.manifest_csv).expanduser().resolve()
    obsids_csv = Path(args.obsids_csv).expanduser().resolve()

    products_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    obsids_csv.parent.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        download_dir.mkdir(parents=True, exist_ok=True)

    m = pd.read_csv(matches_csv)
    s = pd.read_csv(summary_csv)
    sn_keep = set(s["sn_name"].astype(str).str.strip())
    m = m[m["sn_name"].astype(str).str.strip().isin(sn_keep)].copy()
    if len(m) == 0:
        print("No rows after filtering matches by summary SN list.")
        return 1

    obs_ids = sorted(set(m["obs_id"].astype(str).str.strip()))
    obs_ids = [x for x in obs_ids if x and x.lower() not in {"nan", "none"}]
    pd.DataFrame({"obs_id": obs_ids}).to_csv(obsids_csv, index=False)

    obs_tables = []
    for batch in chunked(obs_ids, max(args.batch_size, 1)):
        tab = Observations.query_criteria(obs_collection="HST", filters="F275W", obs_id=batch)
        if len(tab):
            obs_tables.append(tab)

    if not obs_tables:
        print("No observation records returned from MAST for matched obs_id list.")
        return 1

    obs = vstack(obs_tables, metadata_conflicts="silent")
    # unique by numeric obsid when present
    obs_df = obs.to_pandas()
    if "obsid" in obs_df.columns:
        obs_df = obs_df.drop_duplicates(subset=["obsid"], keep="first").reset_index(drop=True)
    else:
        obs_df = obs_df.drop_duplicates(subset=["obs_id"], keep="first").reset_index(drop=True)

    obs_for_products = obs.from_pandas(obs_df)
    prod = Observations.get_product_list(obs_for_products)
    p = prod.to_pandas()

    # Select only drc/drz FITS files.
    fn = p["productFilename"].astype(str)
    keep = fn.str.contains(r"(?i)_dr[cz]\.fits$", regex=True)
    p = p[keep].copy()

    # Deduplicate by dataURI (strongest unique ID).
    if "dataURI" in p.columns:
        p = p.drop_duplicates(subset=["dataURI"], keep="first")

    p = p.sort_values(["proposal_id", "obs_id", "productFilename"], kind="stable").reset_index(drop=True)
    p.to_csv(products_csv, index=False)

    total_bytes = pd.to_numeric(p.get("size", 0), errors="coerce").fillna(0).sum()
    total_gb = total_bytes / 1024**3

    print(f"Matched SN (summary list): {len(sn_keep)}")
    print(f"Matched obs_id values: {len(obs_ids)}")
    print(f"MAST observation rows found: {len(obs_df)}")
    print(f"Selected drc/drz FITS products: {len(p)}")
    print(f"Estimated total size: {total_gb:.2f} GB")
    print(f"Obs ID list: {obsids_csv}")
    print(f"Product list: {products_csv}")

    if args.dry_run:
        print("Dry run only: no files downloaded.")
        return 0

    # Convert filtered dataframe back to an astropy table for download_products.
    prod_for_download = prod.from_pandas(p)
    manifest = Observations.download_products(
        prod_for_download,
        download_dir=str(download_dir),
        cache=True,
        mrp_only=False,
    )
    manifest_df = manifest.to_pandas() if hasattr(manifest, "to_pandas") else pd.DataFrame(manifest)
    manifest_df.to_csv(manifest_csv, index=False)

    ok = int((manifest_df["Status"].astype(str) == "COMPLETE").sum()) if "Status" in manifest_df.columns else np.nan
    err = int((manifest_df["Status"].astype(str) != "COMPLETE").sum()) if "Status" in manifest_df.columns else np.nan
    print(f"Downloaded files complete: {ok}")
    print(f"Downloaded files non-complete: {err}")
    print(f"Download directory: {download_dir}")
    print(f"Download manifest: {manifest_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
