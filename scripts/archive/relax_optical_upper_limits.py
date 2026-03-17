#!/usr/bin/env python3
"""Recompute optical upper-limit flags using negative flux only.

For optical survey measurements, keep magnitude/error values for low-S/N
positive fluxes. Only report an upper limit when the measured flux is negative.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM


COSMO = FlatLambdaCDM(H0=73, Om0=0.27)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--upper-limits", required=True)
    parser.add_argument("--paper-full", required=True)
    parser.add_argument("--paper-short", required=True)
    parser.add_argument("--snr-threshold", type=float, default=3.0)
    return parser.parse_args()


def distance_modulus_from_z(z: pd.Series) -> pd.Series:
    z = pd.to_numeric(z, errors="coerce")
    out = pd.Series(np.nan, index=z.index, dtype=float)
    valid = np.isfinite(z) & (z > 0)
    if valid.any():
        out.loc[valid] = [COSMO.distmod(float(v)).value for v in z.loc[valid]]
    return out


def update_namespace(df: pd.DataFrame, prefix: str, snr_threshold: float) -> pd.DataFrame:
    zp_col = f"{prefix}rband_r_zeropoint"
    if zp_col not in df.columns:
        return df

    mu = pd.to_numeric(df.get("distance_modulus"), errors="coerce")
    if "z" in df.columns:
        missing_mu = ~np.isfinite(mu)
        if missing_mu.any():
            mu.loc[missing_mu] = distance_modulus_from_z(df.loc[missing_mu, "z"])
            df["distance_modulus"] = mu

    for lab in ("1", "2", "3"):
        flux_col = f"{prefix}rband_r_{lab}_flux"
        flux_err_col = f"{prefix}rband_r_{lab}_flux_err"
        mag_col = f"{prefix}rband_r_{lab}"
        mag_err_col = f"{prefix}rband_r_{lab}_err"
        abs_col = f"{prefix}rband_r_{lab}_abs"
        snr_col = f"{prefix}rband_{lab}_snr"
        is_ul_col = f"{prefix}rband_{lab}_is_upper_limit"
        flux_ul_col = f"{prefix}rband_{lab}_flux_upper_limit"
        snr_thr_col = f"{prefix}rband_{lab}_snr_threshold"
        mag_ul_col = f"{prefix}rband_{lab}_mag_upper_limit"
        mag_ul_abs_col = f"{prefix}rband_{lab}_mag_upper_limit_abs"
        required = [flux_col, flux_err_col, zp_col]
        if not all(col in df.columns for col in required):
            continue

        flux = pd.to_numeric(df[flux_col], errors="coerce")
        flux_err = pd.to_numeric(df[flux_err_col], errors="coerce")
        zp = pd.to_numeric(df[zp_col], errors="coerce")
        valid = np.isfinite(flux) & np.isfinite(flux_err) & (flux_err > 0)

        snr = pd.Series(np.nan, index=df.index, dtype=float)
        snr.loc[valid] = flux.loc[valid] / flux_err.loc[valid]
        is_ul = pd.Series(pd.NA, index=df.index, dtype="boolean")
        is_ul.loc[valid] = flux.loc[valid] < 0
        is_ul_bool = is_ul.fillna(False).astype(bool)

        flux_ul = pd.Series(np.nan, index=df.index, dtype=float)
        flux_ul.loc[is_ul_bool] = snr_threshold * flux_err.loc[is_ul_bool]

        # Keep low-S/N positive fluxes as measurements. Reconstruct missing
        # magnitude columns from flux when necessary.
        positive = valid & (flux > 0)
        mag = pd.to_numeric(df.get(mag_col), errors="coerce") if mag_col in df.columns else pd.Series(np.nan, index=df.index)
        mag_err = (
            pd.to_numeric(df.get(mag_err_col), errors="coerce")
            if mag_err_col in df.columns
            else pd.Series(np.nan, index=df.index)
        )

        need_mag = positive & ~np.isfinite(mag) & np.isfinite(zp)
        mag.loc[need_mag] = -2.5 * np.log10(flux.loc[need_mag]) + zp.loc[need_mag]

        need_mag_err = positive & ~np.isfinite(mag_err)
        mag_err.loc[need_mag_err] = 2.5 / np.log(10) * (flux_err.loc[need_mag_err] / flux.loc[need_mag_err])

        # Upper limits only when flux is negative.
        mag.loc[is_ul_bool] = np.nan
        mag_err.loc[is_ul_bool] = np.nan

        mag_ul = pd.Series(np.nan, index=df.index, dtype=float)
        ul_valid = is_ul_bool & np.isfinite(zp) & np.isfinite(flux_ul) & (flux_ul > 0)
        mag_ul.loc[ul_valid] = -2.5 * np.log10(flux_ul.loc[ul_valid]) + zp.loc[ul_valid]

        df[snr_col] = snr
        df[is_ul_col] = is_ul
        df[flux_ul_col] = flux_ul
        df[snr_thr_col] = snr_threshold
        df[mag_col] = mag
        df[mag_err_col] = mag_err
        df[mag_ul_col] = mag_ul

        if np.isfinite(mu).any():
            df[abs_col] = mag - mu
            df[mag_ul_abs_col] = mag_ul - mu

    return df


def format_flux_pair(flux: object, err: object) -> str:
    if pd.notna(flux):
        if pd.notna(err):
            return f"{float(flux):.6g} ({float(err):.6g})"
        return f"{float(flux):.6g}"
    return ""


def update_paper_table(paper: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "r": ("r mag", "r abs mag", "r flux"),
        "des": ("DES r mag", "DES r abs mag", "DES r flux"),
        "panstarrs": ("PanSTARRS r mag", "PanSTARRS r abs mag", "PanSTARRS r flux"),
        "skymapper": ("SkyMapper r mag", "SkyMapper r abs mag", "SkyMapper r flux"),
    }
    for _, cols in mapping.items():
        for col in cols:
            if col not in paper.columns:
                paper[col] = ""

    lookup = summary.set_index("Supernova")
    for idx, sn in paper["SN name"].items():
        if sn not in lookup.index:
            continue
        row = lookup.loc[sn]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        for prefix, (mag_label, abs_label, flux_label) in mapping.items():
            ns = "" if prefix == "r" else f"{prefix}_"
            mag = row.get(f"{ns}rband_r_1")
            mag_err = row.get(f"{ns}rband_r_1_err")
            flux = row.get(f"{ns}rband_r_1_flux")
            flux_err = row.get(f"{ns}rband_r_1_flux_err")
            is_ul = row.get(f"{ns}rband_1_is_upper_limit")
            mag_ul = row.get(f"{ns}rband_1_mag_upper_limit")
            mag_abs = row.get(f"{ns}rband_r_1_abs")
            mag_ul_abs = row.get(f"{ns}rband_1_mag_upper_limit_abs")
            paper.at[idx, flux_label] = format_flux_pair(flux, flux_err)
            if pd.notna(is_ul) and bool(is_ul) and pd.notna(mag_ul):
                paper.at[idx, mag_label] = f"({float(mag_ul):.3f})"
                if pd.notna(mag_ul_abs):
                    paper.at[idx, abs_label] = f"({float(mag_ul_abs):.3f})"
                else:
                    paper.at[idx, abs_label] = ""
            elif pd.notna(mag):
                if pd.notna(mag_err):
                    paper.at[idx, mag_label] = f"{float(mag):.3f} ({float(mag_err):.3f})"
                else:
                    paper.at[idx, mag_label] = f"{float(mag):.3f}"
                if pd.notna(mag_abs):
                    if pd.notna(mag_err):
                        paper.at[idx, abs_label] = f"{float(mag_abs):.3f} ({float(mag_err):.3f})"
                    else:
                        paper.at[idx, abs_label] = f"{float(mag_abs):.3f}"
                else:
                    paper.at[idx, abs_label] = ""
    return paper


def main() -> int:
    args = parse_args()
    summary_path = Path(args.summary)
    ul_path = Path(args.upper_limits)
    paper_full_path = Path(args.paper_full)
    paper_short_path = Path(args.paper_short)

    summary = pd.read_csv(summary_path)
    ul = pd.read_csv(ul_path)

    for prefix in ("", "des_", "panstarrs_", "skymapper_"):
        summary = update_namespace(summary, prefix=prefix, snr_threshold=args.snr_threshold)
        ul = update_namespace(ul, prefix=prefix, snr_threshold=args.snr_threshold)

    summary.to_csv(summary_path, index=False, float_format="%.6g")
    ul.to_csv(ul_path, index=False, float_format="%.6g")

    paper_full = pd.read_csv(paper_full_path)
    paper_full = update_paper_table(paper_full, summary)
    paper_full.to_csv(paper_full_path, index=False)
    paper_full.head(18).copy().to_csv(paper_short_path, index=False)

    print(f"Updated {summary_path}")
    print(f"Updated {ul_path}")
    print(f"Updated {paper_full_path}")
    print(f"Updated {paper_short_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
