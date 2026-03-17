#!/usr/bin/env python3
"""
Find HST/F275W observations for CSPII SN positions not in the current HST sample.

Inputs:
- CSPII SNooPy raw files (first line: SN z ra dec)
- Current HST sample table (supernovas_input.csv)

Method:
1) Build CSPII coordinate list from raw files.
2) Remove objects already in supernovas_input.csv (name-variant matching).
3) Query MAST for all HST observations with filter F275W.
4) Crossmatch by sky distance to HST pointing centers.
5) Exclude specified program IDs (default: 16741, 17179).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from astroquery.mast import Observations
from astropy import units as u
from astropy.coordinates import SkyCoord, search_around_sky

from project_paths import DATA_DIR, light_curves_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Find extra HST/F275W matches for CSPII SN coordinates.")
    p.add_argument(
        "--raw-dir",
        default="/Users/lluisgalbany/Downloads/CSPII SNooPy Files/raw",
        help="Directory with CSPII *.txt raw files.",
    )
    p.add_argument(
        "--sample-csv",
        default=str(DATA_DIR / "supernovas_input.csv"),
        help="Current HST sample table.",
    )
    p.add_argument(
        "--exclude-programs",
        nargs="+",
        default=["16741", "17179"],
        help="Proposal IDs to exclude from output.",
    )
    p.add_argument(
        "--max-sep-arcmin",
        type=float,
        default=2.2,
        help="Maximum SN-to-pointing-center separation (arcmin).",
    )
    p.add_argument(
        "--out-matches-csv",
        default=str(light_curves_path("cspii_not_in_sample_hst_f275w_matches.csv")),
        help="Output CSV with all observation-level matches.",
    )
    p.add_argument(
        "--out-summary-csv",
        default=str(light_curves_path("cspii_not_in_sample_hst_f275w_summary.csv")),
        help="Output CSV with one row per SN and program summary.",
    )
    p.add_argument(
        "--cache-f275w-csv",
        default=str(DATA_DIR / "mast_hst_f275w_observations_cache.csv"),
        help="Optional cache for MAST HST/F275W observation table.",
    )
    p.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Force fresh query to MAST instead of using cache CSV.",
    )
    return p.parse_args()


def norm(name: str) -> str:
    txt = str(name).strip().lower().replace("-host", "").replace("_host", "")
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


def load_cspii_raw(raw_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(raw_dir.glob("*.txt")):
        try:
            first = path.read_text(errors="ignore").splitlines()[0].strip().split()
        except Exception:
            continue
        if len(first) < 4:
            continue
        sn_name = first[0].strip()
        try:
            ra = float(first[2])
            dec = float(first[3])
        except Exception:
            continue
        rows.append({"sn_name": sn_name, "ra_deg": ra, "dec_deg": dec, "raw_file": str(path)})
    out = pd.DataFrame(rows)
    out = out.drop_duplicates(subset=["sn_name"], keep="first").reset_index(drop=True)
    return out


def filter_not_in_sample(cspii: pd.DataFrame, sample_csv: Path) -> pd.DataFrame:
    sample = pd.read_csv(sample_csv)
    sample_names = sample["matched_snname"].astype(str).str.strip().tolist()
    sample_keys = set()
    for s in sample_names:
        sample_keys |= variants(s)

    keep = []
    for _, r in cspii.iterrows():
        is_in = len(variants(r["sn_name"]) & sample_keys) > 0
        keep.append(not is_in)
    return cspii[np.array(keep, dtype=bool)].reset_index(drop=True)


def load_or_query_f275w(cache_csv: Path, refresh_cache: bool) -> pd.DataFrame:
    if cache_csv.exists() and not refresh_cache:
        return pd.read_csv(cache_csv)

    tab = Observations.query_criteria(obs_collection="HST", filters="F275W")
    df = tab.to_pandas()
    # Keep only columns used downstream.
    cols = [
        "obs_id",
        "target_name",
        "s_ra",
        "s_dec",
        "instrument_name",
        "filters",
        "proposal_id",
        "proposal_pi",
        "obs_title",
        "t_min",
        "t_max",
        "t_exptime",
        "calib_level",
        "dataproduct_type",
    ]
    use_cols = [c for c in cols if c in df.columns]
    df = df[use_cols].copy()
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_csv, index=False)
    return df


def main() -> int:
    args = parse_args()

    raw_dir = Path(args.raw_dir).expanduser().resolve()
    sample_csv = Path(args.sample_csv).expanduser().resolve()
    out_matches = Path(args.out_matches_csv).expanduser().resolve()
    out_summary = Path(args.out_summary_csv).expanduser().resolve()
    cache_csv = Path(args.cache_f275w_csv).expanduser().resolve()

    out_matches.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    cspii = load_cspii_raw(raw_dir)
    if len(cspii) == 0:
        print("No CSPII raw objects parsed.")
        return 1

    cspii_not_in = filter_not_in_sample(cspii, sample_csv)
    if len(cspii_not_in) == 0:
        print("No CSPII objects outside current sample.")
        return 0

    obs = load_or_query_f275w(cache_csv=cache_csv, refresh_cache=args.refresh_cache)
    if len(obs) == 0:
        print("No HST/F275W observations returned by MAST.")
        return 1

    # Clean/ensure numeric coordinates.
    obs["s_ra"] = pd.to_numeric(obs["s_ra"], errors="coerce")
    obs["s_dec"] = pd.to_numeric(obs["s_dec"], errors="coerce")
    obs = obs[np.isfinite(obs["s_ra"]) & np.isfinite(obs["s_dec"])].copy()
    def clean_program_id(val: object) -> str:
        if pd.isna(val):
            return ""
        s = str(val).strip()
        if not s:
            return ""
        num = pd.to_numeric(s, errors="coerce")
        if np.isfinite(num):
            return str(int(num))
        return s

    obs["proposal_id"] = obs["proposal_id"].apply(clean_program_id)

    sn_coord = SkyCoord(cspii_not_in["ra_deg"].values * u.deg, cspii_not_in["dec_deg"].values * u.deg, frame="icrs")
    obs_coord = SkyCoord(obs["s_ra"].values * u.deg, obs["s_dec"].values * u.deg, frame="icrs")
    max_sep = args.max_sep_arcmin * u.arcmin

    idx_sn, idx_obs, sep2d, _ = search_around_sky(sn_coord, obs_coord, max_sep)
    if len(idx_sn) == 0:
        print("No spatial matches within configured separation.")
        # still save empty outputs
        pd.DataFrame().to_csv(out_matches, index=False)
        pd.DataFrame().to_csv(out_summary, index=False)
        return 0

    m = pd.DataFrame(
        {
            "sn_name": cspii_not_in["sn_name"].values[idx_sn],
            "sn_ra_deg": cspii_not_in["ra_deg"].values[idx_sn],
            "sn_dec_deg": cspii_not_in["dec_deg"].values[idx_sn],
            "sep_arcsec": sep2d.to(u.arcsec).value,
            "obs_id": obs["obs_id"].values[idx_obs],
            "target_name": obs["target_name"].values[idx_obs] if "target_name" in obs.columns else "",
            "instrument_name": obs["instrument_name"].values[idx_obs] if "instrument_name" in obs.columns else "",
            "filters": obs["filters"].values[idx_obs] if "filters" in obs.columns else "",
            "proposal_id": obs["proposal_id"].values[idx_obs],
            "proposal_pi": obs["proposal_pi"].values[idx_obs] if "proposal_pi" in obs.columns else "",
            "obs_title": obs["obs_title"].values[idx_obs] if "obs_title" in obs.columns else "",
            "t_min": obs["t_min"].values[idx_obs] if "t_min" in obs.columns else np.nan,
            "t_max": obs["t_max"].values[idx_obs] if "t_max" in obs.columns else np.nan,
            "t_exptime": obs["t_exptime"].values[idx_obs] if "t_exptime" in obs.columns else np.nan,
            "calib_level": obs["calib_level"].values[idx_obs] if "calib_level" in obs.columns else np.nan,
        }
    )

    exclude_set = {str(x).strip() for x in args.exclude_programs}
    m = m[~m["proposal_id"].isin(exclude_set)].copy()
    m = m.sort_values(["sn_name", "sep_arcsec", "proposal_id", "obs_id"], kind="stable").reset_index(drop=True)

    # Deduplicate observation-level duplicates if present.
    m = m.drop_duplicates(subset=["sn_name", "obs_id", "proposal_id"], keep="first").reset_index(drop=True)
    m.to_csv(out_matches, index=False)

    if len(m) > 0:
        summary = (
            m.groupby("sn_name", as_index=False)
            .agg(
                sn_ra_deg=("sn_ra_deg", "first"),
                sn_dec_deg=("sn_dec_deg", "first"),
                n_obs=("obs_id", "nunique"),
                n_programs=("proposal_id", lambda x: len({v for v in map(str, x) if v})),
                min_sep_arcsec=("sep_arcsec", "min"),
                proposal_ids=("proposal_id", lambda x: ";".join(sorted({v for v in map(str, x) if v}))),
                instruments=("instrument_name", lambda x: ";".join(sorted({v for v in map(str, x) if v and v != 'nan'}))),
            )
            .sort_values(["n_programs", "n_obs", "sn_name"], ascending=[False, False, True], kind="stable")
            .reset_index(drop=True)
        )
    else:
        summary = pd.DataFrame(
            columns=[
                "sn_name",
                "sn_ra_deg",
                "sn_dec_deg",
                "n_obs",
                "n_programs",
                "min_sep_arcsec",
                "proposal_ids",
                "instruments",
            ]
        )
    summary.to_csv(out_summary, index=False)

    print(f"CSPII raw objects parsed: {len(cspii)}")
    print(f"CSPII not in current sample: {len(cspii_not_in)}")
    print(f"HST/F275W observations considered: {len(obs)}")
    print(f"Observation-level matches (excluding programs {sorted(exclude_set)}): {len(m)}")
    print(f"Unique SN with >=1 match: {len(summary)}")
    print(f"Matches CSV: {out_matches}")
    print(f"Summary CSV: {out_summary}")
    print(f"MAST cache CSV: {cache_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
