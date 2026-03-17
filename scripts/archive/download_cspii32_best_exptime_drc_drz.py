#!/usr/bin/env python3
"""
Download HST F275W drc/drz files for the 32-CSPII list, selecting one
observation per SN from the proposal with the longest exposure time.

Selection rule:
1) For each SN, choose the proposal_id with the highest max(t_exptime).
2) Inside that proposal, choose the obs_id with the highest t_exptime.
3) If tied, prefer canonical obs_id starting with "hst_" and not containing
   "skycell", then smallest sep_arcsec.

Products downloaded for selected obs_id:
- *_drc.fits
- *_drz.fits
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from astroquery.mast import Observations
from astropy.table import vstack

from project_paths import additional_hst_download_dir, light_curves_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download best-exptime HST drc/drz for 32 CSPII SN.")
    p.add_argument(
        "--matches-csv",
        default=str(light_curves_path("cspii_not_in_sample_hst_f275w_matches.csv")),
        help="Observation-level match table.",
    )
    p.add_argument(
        "--summary-csv",
        default=str(light_curves_path("cspii_not_in_sample_hst_f275w_summary.csv")),
        help="SN-level summary table defining the 32 SN list.",
    )
    p.add_argument(
        "--download-dir",
        default=str(additional_hst_download_dir()),
        help="Destination folder for downloaded files.",
    )
    p.add_argument(
        "--selected-csv",
        default=str(light_curves_path("cspii32_bestexptime_selected_obs.csv")),
        help="Output CSV with one selected obs_id per SN.",
    )
    p.add_argument(
        "--products-csv",
        default=str(light_curves_path("cspii32_bestexptime_drcdrz_products.csv")),
        help="Output CSV with selected drc/drz products.",
    )
    p.add_argument(
        "--manifest-csv",
        default=str(light_curves_path("cspii32_bestexptime_download_manifest.csv")),
        help="Output CSV with download manifest.",
    )
    p.add_argument("--batch-size", type=int, default=20, help="obs_id query batch size.")
    p.add_argument("--dry-run", action="store_true", help="Only build lists; do not download.")
    return p.parse_args()


def clean_program_id(val: object) -> str:
    if pd.isna(val):
        return ""
    s = str(val).strip()
    if not s or s.lower() in {"nan", "none"}:
        return ""
    n = pd.to_numeric(s, errors="coerce")
    if np.isfinite(n):
        return str(int(n))
    return s


def is_canonical_obsid(obs_id: str) -> int:
    txt = str(obs_id).lower()
    if txt.startswith("hst_") and "skycell" not in txt:
        return 1
    return 0


def chunked(items: list[str], n: int) -> list[list[str]]:
    return [items[i : i + n] for i in range(0, len(items), n)]


def select_best_obs_per_sn(df: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    for sn, g in df.groupby("sn_name", sort=True):
        gg = g.copy()
        gg["proposal_id_clean"] = gg["proposal_id"].apply(clean_program_id)
        gg["t_exptime_num"] = pd.to_numeric(gg["t_exptime"], errors="coerce").fillna(-np.inf)
        gg["sep_arcsec_num"] = pd.to_numeric(gg["sep_arcsec"], errors="coerce").fillna(np.inf)
        gg["is_canonical"] = gg["obs_id"].map(is_canonical_obsid)

        # Prefer rows with known proposal ID; fallback to unknown-only if needed.
        has_known = (gg["proposal_id_clean"] != "").any()
        if has_known:
            gg = gg[gg["proposal_id_clean"] != ""].copy()

        # Choose proposal with highest max exposure.
        prop_rank = (
            gg.groupby("proposal_id_clean", as_index=False)["t_exptime_num"]
            .max()
            .sort_values(["t_exptime_num", "proposal_id_clean"], ascending=[False, True], kind="stable")
            .reset_index(drop=True)
        )
        best_prop = str(prop_rank.iloc[0]["proposal_id_clean"])
        pp = gg[gg["proposal_id_clean"] == best_prop].copy()

        # Choose one observation in that proposal.
        pp = pp.sort_values(
            ["t_exptime_num", "is_canonical", "sep_arcsec_num", "obs_id"],
            ascending=[False, False, True, True],
            kind="stable",
        ).reset_index(drop=True)
        r = pp.iloc[0]
        out_rows.append(
            {
                "sn_name": sn,
                "proposal_id": best_prop,
                "obs_id": str(r["obs_id"]),
                "t_exptime": float(r["t_exptime_num"]),
                "sep_arcsec": float(r["sep_arcsec_num"]),
                "is_canonical_obsid": int(r["is_canonical"]),
            }
        )
    return pd.DataFrame(out_rows).sort_values("sn_name", kind="stable").reset_index(drop=True)


def main() -> int:
    args = parse_args()

    matches_csv = Path(args.matches_csv).expanduser().resolve()
    summary_csv = Path(args.summary_csv).expanduser().resolve()
    download_dir = Path(args.download_dir).expanduser().resolve()
    selected_csv = Path(args.selected_csv).expanduser().resolve()
    products_csv = Path(args.products_csv).expanduser().resolve()
    manifest_csv = Path(args.manifest_csv).expanduser().resolve()

    selected_csv.parent.mkdir(parents=True, exist_ok=True)
    products_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        download_dir.mkdir(parents=True, exist_ok=True)

    m = pd.read_csv(matches_csv)
    s = pd.read_csv(summary_csv)
    sn_keep = set(s["sn_name"].astype(str).str.strip())
    m = m[m["sn_name"].astype(str).str.strip().isin(sn_keep)].copy()
    if len(m) == 0:
        print("No matches after filtering to summary SN list.")
        return 1

    selected = select_best_obs_per_sn(m)
    selected.to_csv(selected_csv, index=False)
    obs_ids = selected["obs_id"].astype(str).tolist()

    # Query MAST by selected obs_id values.
    obs_tables = []
    for batch in chunked(obs_ids, max(args.batch_size, 1)):
        tab = Observations.query_criteria(obs_collection="HST", filters="F275W", obs_id=batch)
        if len(tab):
            obs_tables.append(tab)
    if not obs_tables:
        print("No observations returned from MAST for selected obs_id list.")
        return 1

    obs = vstack(obs_tables, metadata_conflicts="silent")
    obs_df = obs.to_pandas().drop_duplicates(subset=["obs_id"], keep="first").reset_index(drop=True)
    obs = obs.from_pandas(obs_df)

    prod = Observations.get_product_list(obs)
    p = prod.to_pandas()

    # Keep only drc/drz fits from the selected obs_id rows.
    p = p[p["obs_id"].astype(str).isin(set(obs_ids))].copy()
    keep = p["productFilename"].astype(str).str.contains(r"(?i)_dr[cz]\.fits$", regex=True)
    p = p[keep].copy()
    if "dataURI" in p.columns:
        p = p.drop_duplicates(subset=["dataURI"], keep="first")
    p = p.sort_values(["obs_id", "productFilename"], kind="stable").reset_index(drop=True)
    p.to_csv(products_csv, index=False)

    total_bytes = pd.to_numeric(p.get("size", 0), errors="coerce").fillna(0).sum()
    total_gb = total_bytes / 1024**3
    print(f"SN in shortlist: {len(sn_keep)}")
    print(f"Selected obs_id (1 per SN): {len(selected)}")
    print(f"Selected drc/drz products: {len(p)}")
    print(f"Estimated size: {total_gb:.2f} GB")
    print(f"Selected list: {selected_csv}")
    print(f"Product list: {products_csv}")

    if args.dry_run:
        print("Dry run only: no download performed.")
        return 0

    prod_dl = prod.from_pandas(p)
    manifest = Observations.download_products(
        prod_dl,
        download_dir=str(download_dir),
        cache=True,
        mrp_only=False,
    )
    manifest_df = manifest.to_pandas() if hasattr(manifest, "to_pandas") else pd.DataFrame(manifest)
    manifest_df.to_csv(manifest_csv, index=False)

    if "Status" in manifest_df.columns:
        ok = int((manifest_df["Status"].astype(str) == "COMPLETE").sum())
        bad = int((manifest_df["Status"].astype(str) != "COMPLETE").sum())
    else:
        ok = bad = np.nan

    print(f"Downloaded complete: {ok}")
    print(f"Downloaded non-complete: {bad}")
    print(f"Download directory: {download_dir}")
    print(f"Manifest: {manifest_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
