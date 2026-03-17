#!/usr/bin/env python3
"""Merge DES-only optical photometry into existing summary tables.

This keeps the primary Legacy/PanSTARRS/SkyMapper r-band columns untouched and
adds a parallel DES namespace (`des_rband_*`) for any successful DES-only run.
It also updates the 1 kpc paper tables with DES apparent/absolute r-band
columns while preserving the existing final optical sample columns.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--des-summary", required=True)
    parser.add_argument("--des-log", required=True)
    parser.add_argument("--main-summary", required=True)
    parser.add_argument("--main-upper-limits", required=True)
    parser.add_argument("--main-log", required=True)
    parser.add_argument("--paper-full", required=True)
    parser.add_argument("--paper-short", required=True)
    return parser.parse_args()


def is_missing(value: object) -> bool:
    return pd.isna(value) or str(value).strip() == ""


def format_mag_pair(mag: object, err: object, upper_limit: object, upper_limit_mag: object) -> str:
    if pd.notna(upper_limit) and bool(upper_limit) and pd.notna(upper_limit_mag):
        return f"({float(upper_limit_mag):.3f})"
    if pd.notna(mag):
        if pd.notna(err):
            return f"{float(mag):.3f} ({float(err):.3f})"
        return f"{float(mag):.3f}"
    return ""


def format_flux_pair(flux: object, err: object) -> str:
    if pd.notna(flux):
        if pd.notna(err):
            return f"{float(flux):.6g} ({float(err):.6g})"
        return f"{float(flux):.6g}"
    return ""


def build_des_columns(des_summary: pd.DataFrame) -> pd.DataFrame:
    ok = des_summary[des_summary.get("status", "").astype(str) == "rband_only_ok"].copy()
    rename_map: dict[str, str] = {}
    for col in ok.columns:
        if col.startswith("rband_"):
            rename_map[col] = f"des_{col}"
        elif col == "file":
            rename_map[col] = "des_file"
        elif col == "status":
            rename_map[col] = "des_status"
        elif col == "message":
            rename_map[col] = "des_message"
    ok = ok.rename(columns=rename_map)
    keep = ["Supernova", "des_file", "des_status", "des_message"]
    keep.extend(col for col in ok.columns if col.startswith("des_rband_"))
    keep = [col for col in keep if col in ok.columns]
    ok = ok[keep].copy()
    ok = ok.drop_duplicates(subset=["Supernova"], keep="last")
    return ok


def merge_parallel_columns(main_df: pd.DataFrame, des_df: pd.DataFrame) -> pd.DataFrame:
    merged = main_df.copy()
    if "Supernova" not in merged.columns:
        raise ValueError("Main summary is missing the 'Supernova' column.")
    lookup = des_df.set_index("Supernova")
    for col in des_df.columns:
        if col == "Supernova":
            continue
        if col not in merged.columns:
            merged[col] = pd.NA
        for sn, value in lookup[col].items():
            if pd.notna(value):
                merged.loc[merged["Supernova"] == sn, col] = value
    return merged


def merge_logs(main_log: Path, des_log: Path) -> None:
    frames: list[pd.DataFrame] = []
    if main_log.exists():
        frames.append(pd.read_csv(main_log))
    if des_log.exists():
        frames.append(pd.read_csv(des_log))
    if not frames:
        return
    merged = pd.concat(frames, ignore_index=True, sort=False)
    subset = [col for col in ["name", "fits_file", "status", "message"] if col in merged.columns]
    if subset:
        merged = merged.drop_duplicates(subset=subset, keep="last")
    merged.to_csv(main_log, index=False, float_format="%.6g")


def update_paper_table(paper: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    updated = paper.copy()
    for col in ["DES r mag", "DES r abs mag", "DES r flux"]:
        if col not in updated.columns:
            updated[col] = ""

    summary_lookup = summary.set_index("Supernova")
    for idx, sn in updated["SN name"].items():
        if sn not in summary_lookup.index:
            continue
        row = summary_lookup.loc[sn]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        updated.at[idx, "DES r mag"] = format_mag_pair(
            row.get("des_rband_r_1"),
            row.get("des_rband_r_1_err"),
            row.get("des_rband_1_is_upper_limit"),
            row.get("des_rband_1_mag_upper_limit"),
        )
        updated.at[idx, "DES r abs mag"] = format_mag_pair(
            row.get("des_rband_r_1_abs"),
            row.get("des_rband_r_1_err"),
            row.get("des_rband_1_is_upper_limit"),
            row.get("des_rband_1_mag_upper_limit_abs"),
        )
        updated.at[idx, "DES r flux"] = format_flux_pair(
            row.get("des_rband_r_1_flux"),
            row.get("des_rband_r_1_flux_err"),
        )
    return updated


def main() -> int:
    args = parse_args()

    des_summary = pd.read_csv(args.des_summary)
    des_columns = build_des_columns(des_summary)

    main_summary_path = Path(args.main_summary)
    main_ul_path = Path(args.main_upper_limits)
    main_summary = pd.read_csv(main_summary_path)
    main_ul = pd.read_csv(main_ul_path)

    merged_summary = merge_parallel_columns(main_summary, des_columns)
    merged_ul = merge_parallel_columns(main_ul, des_columns)

    merged_summary.to_csv(main_summary_path, index=False, float_format="%.6g")
    merged_ul.to_csv(main_ul_path, index=False, float_format="%.6g")

    merge_logs(Path(args.main_log), Path(args.des_log))

    paper_full_path = Path(args.paper_full)
    paper_short_path = Path(args.paper_short)
    paper_full = pd.read_csv(paper_full_path)
    paper_full = update_paper_table(paper_full, merged_summary)
    paper_full.to_csv(paper_full_path, index=False)
    paper_full.head(18).copy().to_csv(paper_short_path, index=False)

    print(f"Merged DES photometry rows: {len(des_columns)}")
    print(f"Updated {main_summary_path}")
    print(f"Updated {main_ul_path}")
    print(f"Updated {paper_full_path}")
    print(f"Updated {paper_short_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
