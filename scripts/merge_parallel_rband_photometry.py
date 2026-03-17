#!/usr/bin/env python3
"""Merge a parallel optical survey photometry table into the main outputs.

This preserves the existing primary `rband_*` columns and writes the new survey
into a namespaced prefix, e.g. `des_rband_*` or `panstarrs_rband_*`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-summary", required=True)
    parser.add_argument("--source-log", required=True)
    parser.add_argument("--main-summary", required=True)
    parser.add_argument("--main-upper-limits", required=True)
    parser.add_argument("--main-log", required=True)
    parser.add_argument("--prefix", required=True, help="Namespace prefix, e.g. des or panstarrs.")
    parser.add_argument("--paper-full", default="")
    parser.add_argument("--paper-short", default="")
    parser.add_argument("--paper-mag-label", default="", help="Paper-table column label, e.g. DES r mag.")
    parser.add_argument(
        "--paper-abs-mag-label", default="", help="Paper-table absolute-magnitude label, e.g. DES r abs mag."
    )
    return parser.parse_args()


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


def build_prefixed_columns(source_summary: pd.DataFrame, prefix: str) -> pd.DataFrame:
    ok = source_summary[source_summary.get("status", "").astype(str) == "rband_only_ok"].copy()
    rename_map: dict[str, str] = {}
    for col in ok.columns:
        if col.startswith("rband_"):
            rename_map[col] = f"{prefix}_{col}"
        elif col == "file":
            rename_map[col] = f"{prefix}_file"
        elif col == "status":
            rename_map[col] = f"{prefix}_status"
        elif col == "message":
            rename_map[col] = f"{prefix}_message"
    ok = ok.rename(columns=rename_map)
    keep = ["Supernova", f"{prefix}_file", f"{prefix}_status", f"{prefix}_message"]
    keep.extend(col for col in ok.columns if col.startswith(f"{prefix}_rband_"))
    keep = [col for col in keep if col in ok.columns]
    ok = ok[keep].drop_duplicates(subset=["Supernova"], keep="last").copy()
    return ok


def merge_parallel_columns(main_df: pd.DataFrame, prefixed_df: pd.DataFrame) -> pd.DataFrame:
    merged = main_df.copy()
    lookup = prefixed_df.set_index("Supernova")
    for col in prefixed_df.columns:
        if col == "Supernova":
            continue
        if col not in merged.columns:
            merged[col] = pd.NA
        for sn, value in lookup[col].items():
            if pd.notna(value):
                merged.loc[merged["Supernova"] == sn, col] = value
    return merged


def merge_logs(main_log: Path, source_log: Path) -> None:
    frames: list[pd.DataFrame] = []
    if main_log.exists():
        frames.append(pd.read_csv(main_log))
    if source_log.exists():
        frames.append(pd.read_csv(source_log))
    if not frames:
        return
    merged = pd.concat(frames, ignore_index=True, sort=False)
    subset = [col for col in ["name", "fits_file", "status", "message"] if col in merged.columns]
    if subset:
        merged = merged.drop_duplicates(subset=subset, keep="last")
    merged.to_csv(main_log, index=False, float_format="%.6g")


def update_paper_table(
    paper: pd.DataFrame,
    summary: pd.DataFrame,
    prefix: str,
    mag_label: str,
    abs_mag_label: str,
) -> pd.DataFrame:
    updated = paper.copy()
    if not mag_label or not abs_mag_label:
        return updated
    flux_label = mag_label.replace(" mag", " flux")
    for col in [mag_label, abs_mag_label, flux_label]:
        if col not in updated.columns:
            updated[col] = ""
        else:
            updated[col] = updated[col].astype("object")
    lookup = summary.set_index("Supernova")
    for idx, sn in updated["SN name"].items():
        if sn not in lookup.index:
            continue
        row = lookup.loc[sn]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        updated.at[idx, mag_label] = format_mag_pair(
            row.get(f"{prefix}_rband_r_1"),
            row.get(f"{prefix}_rband_r_1_err"),
            row.get(f"{prefix}_rband_1_is_upper_limit"),
            row.get(f"{prefix}_rband_1_mag_upper_limit"),
        )
        updated.at[idx, abs_mag_label] = format_mag_pair(
            row.get(f"{prefix}_rband_r_1_abs"),
            row.get(f"{prefix}_rband_r_1_err"),
            row.get(f"{prefix}_rband_1_is_upper_limit"),
            row.get(f"{prefix}_rband_1_mag_upper_limit_abs"),
        )
        updated.at[idx, flux_label] = format_flux_pair(
            row.get(f"{prefix}_rband_r_1_flux"),
            row.get(f"{prefix}_rband_r_1_flux_err"),
        )
    return updated


def main() -> int:
    args = parse_args()
    prefix = args.prefix.strip()

    source_summary = pd.read_csv(args.source_summary)
    prefixed = build_prefixed_columns(source_summary, prefix)

    main_summary_path = Path(args.main_summary)
    main_ul_path = Path(args.main_upper_limits)
    main_summary = pd.read_csv(main_summary_path)
    main_ul = pd.read_csv(main_ul_path)

    merged_summary = merge_parallel_columns(main_summary, prefixed)
    merged_ul = merge_parallel_columns(main_ul, prefixed)

    merged_summary.to_csv(main_summary_path, index=False, float_format="%.6g")
    merged_ul.to_csv(main_ul_path, index=False, float_format="%.6g")
    merge_logs(Path(args.main_log), Path(args.source_log))

    if args.paper_full and args.paper_short:
        paper_full_path = Path(args.paper_full)
        paper_short_path = Path(args.paper_short)
        paper_full = pd.read_csv(paper_full_path)
        paper_full = update_paper_table(
            paper_full,
            merged_summary,
            prefix=prefix,
            mag_label=args.paper_mag_label,
            abs_mag_label=args.paper_abs_mag_label,
        )
        paper_full.to_csv(paper_full_path, index=False)
        paper_full.head(18).copy().to_csv(paper_short_path, index=False)
        print(f"Updated {paper_full_path}")
        print(f"Updated {paper_short_path}")

    print(f"Merged {prefix} photometry rows: {len(prefixed)}")
    print(f"Updated {main_summary_path}")
    print(f"Updated {main_ul_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
