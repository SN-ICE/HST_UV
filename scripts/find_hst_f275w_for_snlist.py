#!/usr/bin/env python3
"""
Find HST/F275W observations for targets listed in data/list_snlist.txt,
excluding objects already present in the current HST sample.

Outputs:
- observation-level candidate matches for each SN
- one best-match row per SN, chosen by longest exposure time
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from astroquery.mast import Observations
from astropy import units as u
from astropy.coordinates import SkyCoord

from project_paths import DATA_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find additional HST/F275W matches for data/list_snlist.txt.")
    parser.add_argument(
        "--input-list",
        default=str(DATA_DIR / "list_snlist.txt"),
        help="CSV-like target list with columns including snname, cubename, rasn, decsn.",
    )
    parser.add_argument(
        "--sample-csv",
        default=str(DATA_DIR / "supernovas_input.csv"),
        help="Current HST sample table.",
    )
    parser.add_argument(
        "--max-sep-arcmin",
        type=float,
        default=2.2,
        help="Maximum SN-to-pointing-center separation in arcmin.",
    )
    parser.add_argument(
        "--cache-f275w-csv",
        default=str(DATA_DIR / "mast_hst_f275w_observations_cache.csv"),
        help="Cache path for the global MAST HST/F275W observation table.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Force a fresh MAST query instead of reusing the cache CSV.",
    )
    parser.add_argument(
        "--out-matches-csv",
        default=str(DATA_DIR / "list_snlist_hst_f275w_matches.csv"),
        help="Output CSV with all observation-level candidate matches.",
    )
    parser.add_argument(
        "--out-best-csv",
        default=str(DATA_DIR / "list_snlist_hst_f275w_best_exptime.csv"),
        help="Output CSV with the single best-exposure match per SN.",
    )
    return parser.parse_args()


def norm(name: str) -> str:
    txt = str(name).strip().lower().replace("-host", "").replace("_host", "")
    return re.sub(r"[^a-z0-9]+", "", txt)


def variants(name: str) -> set[str]:
    s = str(name).strip().lower()
    if not s:
        return set()

    out = {s, norm(s)}
    swaps = [
        ("asassn", "asas"),
        ("asas", "asassn"),
        ("iptf", "ptf"),
        ("ptf", "iptf"),
    ]
    for old, new in swaps:
        if old in s:
            s2 = s.replace(old, new)
            out.add(s2)
            out.add(norm(s2))

    n = norm(s)
    if n:
        out.add(n[2:] if n.startswith("sn") else "sn" + n)
    return {x for x in out if x}


def load_target_list(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, skipinitialspace=True)
    required = ["snname", "cubename", "rasn", "decsn"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    work = df.copy()
    work["snname"] = work["snname"].astype(str).str.strip()
    work["cubename"] = work["cubename"].astype(str).str.strip()

    ra_vals = work["rasn"].astype(str).str.strip()
    dec_vals = work["decsn"].astype(str).str.strip()
    keep_mask = (
        ra_vals.ne("")
        & dec_vals.ne("")
        & ra_vals.str.lower().ne("nan")
        & dec_vals.str.lower().ne("nan")
    )
    work = work[keep_mask].copy().reset_index(drop=True)

    ra_deg: list[float] = []
    dec_deg: list[float] = []
    valid_rows: list[int] = []
    for idx, (ra_txt, dec_txt) in enumerate(zip(work["rasn"].astype(str).str.strip(), work["decsn"].astype(str).str.strip())):
        try:
            coord = SkyCoord(ra_txt, dec_txt, unit=(u.hourangle, u.deg), frame="icrs")
        except Exception:
            continue
        valid_rows.append(idx)
        ra_deg.append(coord.ra.deg)
        dec_deg.append(coord.dec.deg)

    work = work.iloc[valid_rows].copy().reset_index(drop=True)
    work["ra_deg"] = ra_deg
    work["dec_deg"] = dec_deg
    work = work.drop_duplicates(subset=["snname"], keep="first").reset_index(drop=True)
    return work


def filter_not_in_sample(targets: pd.DataFrame, sample_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample = pd.read_csv(sample_csv)
    sample_keys: set[str] = set()
    for col in ["matched_snname", "cubename", "targname"]:
        if col not in sample.columns:
            continue
        for value in sample[col].astype(str).str.strip():
            sample_keys |= variants(value)

    keep_mask = []
    included_rows = []
    for _, row in targets.iterrows():
        row_keys = variants(row["snname"]) | variants(row["cubename"])
        is_in_sample = len(row_keys & sample_keys) > 0
        keep_mask.append(not is_in_sample)
        if is_in_sample:
            included_rows.append(row.to_dict())

    remaining = targets[np.array(keep_mask, dtype=bool)].reset_index(drop=True)
    included = pd.DataFrame(included_rows)
    return remaining, included


def load_or_query_f275w(cache_csv: Path, refresh_cache: bool) -> pd.DataFrame:
    if cache_csv.exists() and not refresh_cache:
        return pd.read_csv(cache_csv)

    tab = Observations.query_criteria(obs_collection="HST", filters="F275W")
    df = tab.to_pandas()
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
    use_cols = [col for col in cols if col in df.columns]
    df = df[use_cols].copy()
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_csv, index=False)
    return df


def clean_program_id(value: object) -> str:
    if pd.isna(value):
        return ""
    txt = str(value).strip()
    if not txt:
        return ""
    numeric = pd.to_numeric(txt, errors="coerce")
    if np.isfinite(numeric):
        return str(int(numeric))
    return txt


def build_matches(targets: pd.DataFrame, obs: pd.DataFrame, max_sep_arcmin: float) -> pd.DataFrame:
    obs = obs.copy()
    obs["s_ra"] = pd.to_numeric(obs["s_ra"], errors="coerce")
    obs["s_dec"] = pd.to_numeric(obs["s_dec"], errors="coerce")
    obs["t_exptime"] = pd.to_numeric(obs["t_exptime"], errors="coerce")
    obs["calib_level"] = pd.to_numeric(obs["calib_level"], errors="coerce")
    obs["proposal_id"] = obs["proposal_id"].apply(clean_program_id)
    obs = obs[np.isfinite(obs["s_ra"]) & np.isfinite(obs["s_dec"])].copy()

    rows: list[dict] = []
    max_sep = max_sep_arcmin * u.arcmin
    for _, row in targets.iterrows():
        sn_coord = SkyCoord(row["ra_deg"] * u.deg, row["dec_deg"] * u.deg, frame="icrs")
        obs_coord = SkyCoord(obs["s_ra"].values * u.deg, obs["s_dec"].values * u.deg, frame="icrs")
        sep = sn_coord.separation(obs_coord)
        matched = obs[sep <= max_sep].copy()
        if len(matched) == 0:
            continue

        matched["sep_arcsec"] = sep[sep <= max_sep].to(u.arcsec).value
        matched["sn_name"] = row["snname"]
        matched["cubename"] = row["cubename"]
        matched["sn_ra_deg"] = row["ra_deg"]
        matched["sn_dec_deg"] = row["dec_deg"]
        matched["sn_type"] = row.get("sntype", "")
        matched["sem"] = row.get("sem", "")
        matched["notes"] = row.get("notes", "")
        rows.extend(matched.to_dict(orient="records"))

    if not rows:
        return pd.DataFrame()

    matches = pd.DataFrame(rows)
    matches = matches.sort_values(
        ["sn_name", "t_exptime", "sep_arcsec", "calib_level", "obs_id"],
        ascending=[True, False, True, False, True],
        kind="stable",
    ).reset_index(drop=True)
    matches = matches.drop_duplicates(subset=["sn_name", "obs_id", "proposal_id"], keep="first").reset_index(drop=True)
    return matches


def build_best_exptime(matches: pd.DataFrame) -> pd.DataFrame:
    if len(matches) == 0:
        return pd.DataFrame(
            columns=[
                "sn_name",
                "cubename",
                "sn_ra_deg",
                "sn_dec_deg",
                "sn_type",
                "sem",
                "notes",
                "obs_id",
                "target_name",
                "proposal_id",
                "proposal_pi",
                "instrument_name",
                "filters",
                "sep_arcsec",
                "t_exptime",
                "calib_level",
                "t_min",
                "t_max",
                "obs_title",
            ]
        )

    ordered = matches.sort_values(
        ["sn_name", "t_exptime", "sep_arcsec", "calib_level", "obs_id"],
        ascending=[True, False, True, False, True],
        kind="stable",
    )
    best = ordered.groupby("sn_name", as_index=False).head(1).reset_index(drop=True)
    cols = [
        "sn_name",
        "cubename",
        "sn_ra_deg",
        "sn_dec_deg",
        "sn_type",
        "sem",
        "notes",
        "obs_id",
        "target_name",
        "proposal_id",
        "proposal_pi",
        "instrument_name",
        "filters",
        "sep_arcsec",
        "t_exptime",
        "calib_level",
        "t_min",
        "t_max",
        "obs_title",
    ]
    use_cols = [col for col in cols if col in best.columns]
    return best[use_cols].copy()


def main() -> int:
    args = parse_args()

    input_list = Path(args.input_list).expanduser().resolve()
    sample_csv = Path(args.sample_csv).expanduser().resolve()
    cache_csv = Path(args.cache_f275w_csv).expanduser().resolve()
    out_matches = Path(args.out_matches_csv).expanduser().resolve()
    out_best = Path(args.out_best_csv).expanduser().resolve()

    out_matches.parent.mkdir(parents=True, exist_ok=True)
    out_best.parent.mkdir(parents=True, exist_ok=True)

    targets = load_target_list(input_list)
    not_in_sample, already_in_sample = filter_not_in_sample(targets, sample_csv)
    obs = load_or_query_f275w(cache_csv, refresh_cache=args.refresh_cache)
    matches = build_matches(not_in_sample, obs, max_sep_arcmin=args.max_sep_arcmin)
    best = build_best_exptime(matches)

    matches.to_csv(out_matches, index=False)
    best.to_csv(out_best, index=False)

    print(f"Targets in list: {len(targets)}")
    print(f"Targets already in current sample: {len(already_in_sample)}")
    print(f"Targets searched in MAST: {len(not_in_sample)}")
    print(f"HST/F275W observations considered: {len(obs)}")
    print(f"Observation-level candidate matches: {len(matches)}")
    print(f"Unique SN with >=1 match: {len(best)}")
    print(f"All matches CSV: {out_matches}")
    print(f"Best-exposure CSV: {out_best}")
    print(f"MAST cache CSV: {cache_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
