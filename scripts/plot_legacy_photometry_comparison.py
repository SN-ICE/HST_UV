#!/usr/bin/env python3
"""Plot three-panel Legacy/PanSTARRS comparison against the blank-aperture Legacy test."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blank-csv", default="outputs/legacy_blank_aperture_test_comparison.csv")
    parser.add_argument("--invvar-csv", default="outputs/legacy_invvar_test_comparison.csv")
    parser.add_argument("--output", default="outputs/legacy_photometry_threepanel_comparison.png")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    blank = pd.read_csv(args.blank_csv)
    invvar = pd.read_csv(args.invvar_csv)
    df = blank.merge(
        invvar[["SN name", "legacy_test_mag", "legacy_test_mag_err", "legacy_test_status"]],
        on="SN name",
        how="left",
    )

    for col in [
        "legacy_blank_mag",
        "legacy_blank_mag_err",
        "legacy_hostphot_mag",
        "legacy_hostphot_mag_err",
        "legacy_test_mag",
        "legacy_test_mag_err",
        "panstarrs_mag",
        "panstarrs_mag_err",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    good = df[
        df["legacy_blank_status"].isin(["ok", "ok_invvar_only"])
        & (df["legacy_hostphot_is_ul"].fillna(False) == False)
        & (df["panstarrs_is_ul"].fillna(False) == False)
        & np.isfinite(df["legacy_blank_mag"])
    ].copy()

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), sharex=True, sharey=True)
    panels = [
        ("legacy_hostphot_mag", "legacy_hostphot_mag_err", "Current Legacy (hostphot)"),
        ("legacy_test_mag", "legacy_test_mag_err", "Legacy test (invvar only)"),
        ("panstarrs_mag", "panstarrs_mag_err", "PanSTARRS"),
    ]

    x = good["legacy_blank_mag"].to_numpy()
    xerr = good["legacy_blank_mag_err"].to_numpy()
    finite_y = []
    for ycol, _, _ in panels:
        finite_y.append(good[ycol].to_numpy())
    vals = np.concatenate([x] + finite_y)
    vals = vals[np.isfinite(vals)]
    lim_lo = float(np.nanmin(vals) - 0.2)
    lim_hi = float(np.nanmax(vals) + 0.2)

    colors = ["#4477AA", "#228833", "#CC6677"]
    for ax, (ycol, yerr_col, title), color in zip(axes, panels, colors):
        y = good[ycol].to_numpy()
        yerr = good[yerr_col].to_numpy()
        mask = np.isfinite(x) & np.isfinite(y)
        ax.errorbar(
            x[mask],
            y[mask],
            xerr=xerr[mask],
            yerr=yerr[mask],
            fmt="o",
            markersize=4.2,
            alpha=0.7,
            color=color,
            ecolor=color,
            elinewidth=0.7,
            capsize=0,
            rasterized=True,
        )
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], ls="--", lw=1.2, color="0.35")
        ax.set_title(title)
        ax.set_xlim(lim_lo, lim_hi)
        ax.set_ylim(lim_lo, lim_hi)
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.grid(alpha=0.25)
        ax.set_xlabel("Legacy test (blank-aperture) mag")

    axes[0].set_ylabel("Comparison magnitude")
    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out} with {len(good)} points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
