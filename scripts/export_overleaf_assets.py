#!/usr/bin/env python3
"""Rebuild Overleaf-facing tables and aggregate figures from current outputs."""

from __future__ import annotations

import argparse
import math
import re
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D


mpl.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "mathtext.fontset": "stix",
        "axes.unicode_minus": True,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-csv", default="data/supernovas_input.csv")
    parser.add_argument("--summary-csv", default="outputs/sn_mag_summary_1kpc_full.csv")
    parser.add_argument("--photometry-csv", default="outputs/photometry_summary_keep.csv")
    parser.add_argument("--paper-candidates", default="outputs/paper_plot_candidate_names.txt")
    parser.add_argument("--plots-dir", default="outputs/plots")
    parser.add_argument("--overleaf-dir", default="overleaf")
    return parser.parse_args()


def parse_mag_field(value: object) -> tuple[float, float, str]:
    s = str(value).strip()
    if s.lower() in {"nan", ""}:
        return np.nan, np.nan, "nan"

    m_det = re.match(r"^([+-]?\d+(?:\.\d+)?)\s*\(([+-]?\d+(?:\.\d+)?)\)$", s)
    if m_det:
        return float(m_det.group(1)), float(m_det.group(2)), "det"

    m_ul = re.match(r"^\(([+-]?\d+(?:\.\d+)?)\)$", s)
    if m_ul:
        return float(m_ul.group(1)), np.nan, "ul"

    s_clean = s.replace("$", "").replace("?", "").strip()
    m_ul_bound = re.match(r"^[<>]\s*([+-]?\d+(?:\.\d+)?)$", s_clean)
    if m_ul_bound:
        return float(m_ul_bound.group(1)), np.nan, "ul"

    m_plain = re.match(r"^([+-]?\d+(?:\.\d+)?)$", s)
    if m_plain:
        return float(m_plain.group(1)), np.nan, "det"

    return np.nan, np.nan, "nan"


def esc_latex(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    return text.replace("_", r"\_")


def write_short_table(df: pd.DataFrame, out_path: Path) -> None:
    subset = df.head(18).copy()
    with out_path.open("w") as fh:
        fh.write("\\begin{table*}[!htbp]\n")
        fh.write("\\centering\n")
        fh.write("\\scriptsize\n")
        fh.write("\\caption{Summary of 1~kpc aperture photometry for a representative subset of 18 SN from the full sample. Values are reported as magnitude (error). Upper limits are shown in parentheses.}\n")
        fh.write("\\label{tab:mag_summary}\n")
        fh.write("\\resizebox{\\textwidth}{!}{%\n")
        fh.write("\\begin{tabular}{lcccccc}\n")
        fh.write("\\hline\\hline\n")
        fh.write("SN name & UV magnitude & $M_{\\mathrm{UV}}$ & $r$ magnitude & $M_r$ & F275W$-r$ & $r$ survey \\\\\n")
        fh.write("\\hline\n")
        for _, row in subset.iterrows():
            fh.write(
                f"{esc_latex(row['SN name'])} & {row['UV mag']} & {row['UV abs mag']} & "
                f"{row['Adopted r mag']} & {row['Adopted r abs mag']} & {row['F275W-r']} & "
                f"{esc_latex(row['Adopted r survey'])} \\\\\n"
            )
        fh.write("\\hline\n")
        fh.write("\\end{tabular}}\n")
        fh.write("\\tablefoot{Photometry is reported for the 1~kpc aperture. Values in parentheses denote upper limits. Absolute magnitudes are computed from redshift using the WMAP9 cosmology. F275W$-r$ colors are computed from the adopted 1~kpc apparent magnitudes; one-band upper limits are reported as bounds. The current UV-missing cases are SN2005cf\\_B, SN1981D, SN2012Z, and SN2007af; these are excluded from the UV analysis because the aperture falls partly or entirely outside the usable HST footprint. This table is intentionally truncated; the full table is available in the online version of the manuscript.}\n")
        fh.write("\\end{table*}\n")


def write_full_table(df: pd.DataFrame, out_path: Path) -> None:
    with out_path.open("w") as fh:
        fh.write("\\begin{longtable}{lcccccc}\n")
        fh.write("\\caption{Full 1~kpc aperture photometry table for all processed SN. Values are reported as magnitude (error). Upper limits are shown in parentheses.}\\label{tab:mag_summary_full}\\\\\n")
        fh.write("\\hline\\hline\n")
        fh.write("SN name & UV magnitude & $M_{\\mathrm{UV}}$ & $r$ magnitude & $M_r$ & F275W$-r$ & $r$ survey \\\\\n")
        fh.write("\\hline\n")
        fh.write("\\endfirsthead\n")
        fh.write("\\hline\\hline\n")
        fh.write("SN name & UV magnitude & $M_{\\mathrm{UV}}$ & $r$ magnitude & $M_r$ & F275W$-r$ & $r$ survey \\\\\n")
        fh.write("\\hline\n")
        fh.write("\\endhead\n")
        fh.write("\\hline\n")
        fh.write("\\endfoot\n")
        for _, row in df.iterrows():
            fh.write(
                f"{esc_latex(row['SN name'])} & {row['UV mag']} & {row['UV abs mag']} & "
                f"{row['Adopted r mag']} & {row['Adopted r abs mag']} & {row['F275W-r']} & "
                f"{esc_latex(row['Adopted r survey'])} \\\\\n"
            )
        fh.write("\\end{longtable}\n")


def plot_redshift_distribution(sample_df: pd.DataFrame, out_path: Path) -> None:
    z = pd.to_numeric(sample_df["redshift"], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    ax.hist(z, bins=20, color="#4C78A8", edgecolor="white", linewidth=0.5)
    if not z.empty:
        med = float(z.median())
        ax.axvline(med, color="0.25", linestyle="--", linewidth=1.3)
    ax.set_xlabel("Redshift")
    ax.set_ylabel("N")
    ax.grid(axis="y", color="0.85", linewidth=0.6)
    ax.tick_params(direction="in", top=True, right=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_mag_arrays(csv_path: Path, x_col: str, y_col: str):
    df = pd.read_csv(csv_path)
    x_vals, x_errs, x_typ = [], [], []
    y_vals, y_errs, y_typ = [], [], []
    for _, row in df.iterrows():
        yv, ye, yt = parse_mag_field(row[y_col])
        xv, xe, xt = parse_mag_field(row[x_col])
        x_vals.append(xv)
        x_errs.append(xe)
        x_typ.append(xt)
        y_vals.append(yv)
        y_errs.append(ye)
        y_typ.append(yt)
    return (
        np.asarray(x_vals, dtype=float),
        np.asarray(y_vals, dtype=float),
        np.asarray(x_errs, dtype=float),
        np.asarray(y_errs, dtype=float),
        np.asarray(x_typ),
        np.asarray(y_typ),
    )


def load_mag_arrays_preferred_x(csv_path: Path, x_cols: list[str], y_col: str):
    df = pd.read_csv(csv_path)
    x_vals, x_errs, x_typ = [], [], []
    y_vals, y_errs, y_typ = [], [], []
    for _, row in df.iterrows():
        yv, ye, yt = parse_mag_field(row[y_col])
        xv = xe = np.nan
        xt = "nan"
        for x_col in x_cols:
            xv_try, xe_try, xt_try = parse_mag_field(row[x_col])
            if xt_try != "nan":
                xv, xe, xt = xv_try, xe_try, xt_try
                break
        x_vals.append(xv)
        x_errs.append(xe)
        x_typ.append(xt)
        y_vals.append(yv)
        y_errs.append(ye)
        y_typ.append(yt)
    return (
        np.asarray(x_vals, dtype=float),
        np.asarray(y_vals, dtype=float),
        np.asarray(x_errs, dtype=float),
        np.asarray(y_errs, dtype=float),
        np.asarray(x_typ),
        np.asarray(y_typ),
    )


def plot_mag_scatter_marginals(
    csv_path: Path,
    out_main: Path,
    out_alias: Path,
    *,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
) -> None:
    x, y, xerr, yerr, xt, yt = load_mag_arrays(csv_path, x_col=x_col, y_col=y_col)
    valid = np.isfinite(x) & np.isfinite(y)
    det_det = valid & (xt == "det") & (yt == "det")
    r_ul_uv_det = valid & (xt == "ul") & (yt == "det")
    r_det_uv_ul = valid & (xt == "det") & (yt == "ul")
    both_ul = valid & (xt == "ul") & (yt == "ul")

    x_all = x[valid]
    y_all = y[valid]
    x_margin = 0.65
    y_margin = 0.65
    x_left = np.nanmax(x_all) + x_margin
    x_right = np.nanmin(x_all) - x_margin
    y_bottom = np.nanmax(y_all) + y_margin
    y_top = np.nanmin(y_all) - y_margin

    fig = plt.figure(figsize=(8.3, 7.6))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 3.2], width_ratios=[3.2, 1.0], hspace=0.04, wspace=0.04)
    ax = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax)

    col_det = "#2E86B7"
    col_ul = "#D55E00"
    col_histx = "#E3A21A"
    col_histy = "#2FAE8F"

    x_for_hist = x[valid & (xt == "det")]
    y_for_hist = y[valid & (yt == "det")]
    if x_for_hist.size:
        ax_top.hist(x_for_hist, bins=24, color=col_histx, edgecolor="white", linewidth=0.35)
    if y_for_hist.size:
        ax_right.hist(y_for_hist, bins=24, orientation="horizontal", color=col_histy, edgecolor="white", linewidth=0.35)
    if np.any(det_det):
        med_x = np.nanmedian(x[det_det])
        med_y = np.nanmedian(y[det_det])
        ax_top.axvline(med_x, color="0.45", ls="--", lw=1.1, alpha=0.9)
        ax_right.axhline(med_y, color="0.45", ls="--", lw=1.1, alpha=0.9)
    if np.any(det_det):
        ax.errorbar(
            x[det_det], y[det_det], xerr=xerr[det_det], yerr=yerr[det_det],
            fmt="o", ms=5.0, mfc=col_det, mec="white", mew=0.25,
            ecolor=col_det, elinewidth=0.8, alpha=0.9, capsize=0, zorder=3,
        )

    ax.set_xlim(x_left, x_right)
    ax.set_ylim(y_bottom, y_top)
    x_span = abs(x_left - x_right)
    y_span = abs(y_bottom - y_top)
    dx = 0.030 * x_span
    dy = 0.030 * y_span
    padx = 0.015 * x_span
    pady = 0.015 * y_span

    def draw_x_ul(x0: float, y0: float) -> None:
        xl, xr = ax.get_xlim()
        sx = min(max(x0, xr + padx), xl - padx)
        ex = min(sx + dx, xl - padx)
        ax.annotate("", xy=(ex, y0), xytext=(sx, y0), arrowprops=dict(arrowstyle="-|>", color=col_ul, lw=1.0, mutation_scale=8), zorder=6, clip_on=False)

    def draw_y_ul(x0: float, y0: float) -> None:
        yb, yt_lim = ax.get_ylim()
        sy = min(max(y0, yt_lim + pady), yb - pady)
        ey = min(sy + dy, yb - pady)
        ax.annotate("", xy=(x0, ey), xytext=(x0, sy), arrowprops=dict(arrowstyle="-|>", color=col_ul, lw=1.0, mutation_scale=8), zorder=6, clip_on=False)

    def draw_diag_ul(x0: float, y0: float) -> None:
        xl, xr = ax.get_xlim()
        yb, yt_lim = ax.get_ylim()
        sx = min(max(x0, xr + padx), xl - padx)
        sy = min(max(y0, yt_lim + pady), yb - pady)
        ex = min(sx + dx, xl - padx)
        ey = min(sy + dy, yb - pady)
        ax.annotate("", xy=(ex, ey), xytext=(sx, sy), arrowprops=dict(arrowstyle="-|>", color=col_ul, lw=1.0, mutation_scale=8), zorder=6, clip_on=False)

    for idx in np.where(r_ul_uv_det)[0]:
        if np.isfinite(yerr[idx]):
            ax.errorbar(x[idx], y[idx], yerr=yerr[idx], fmt="none", ecolor=col_ul, elinewidth=0.85, alpha=0.85, zorder=4)
        draw_x_ul(float(x[idx]), float(y[idx]))
    for idx in np.where(r_det_uv_ul)[0]:
        if np.isfinite(xerr[idx]):
            ax.errorbar(x[idx], y[idx], xerr=xerr[idx], fmt="none", ecolor=col_ul, elinewidth=0.85, alpha=0.85, zorder=4)
        draw_y_ul(float(x[idx]), float(y[idx]))
    for idx in np.where(both_ul)[0]:
        draw_diag_ul(float(x[idx]), float(y[idx]))

    xmin, xmax = sorted(ax.get_xlim())
    ymin, ymax = sorted(ax.get_ylim())
    d0, d1 = max(xmin, ymin), min(xmax, ymax)
    if d1 > d0:
        t = np.linspace(d0, d1, 200)
        ax.plot(t, t, ls="--", lw=1.2, color="0.35", alpha=0.85, zorder=2)

    handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=6, markerfacecolor=col_det, markeredgecolor="white", markeredgewidth=0.3, label="Detections"),
        Line2D([0], [0], linestyle="-", color=col_ul, lw=1.0, marker=">", markersize=5, markerfacecolor=col_ul, markeredgecolor=col_ul, label="Upper limits"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=True, framealpha=0.92, facecolor="white", edgecolor="0.8", fontsize=11)
    ax.set_xlabel(x_label, fontsize=17)
    ax.set_ylabel(y_label, fontsize=17)
    ax.grid(color="0.80", alpha=0.5, linewidth=0.6)
    ax_top.set_ylabel("N", fontsize=17)
    ax_right.set_xlabel("N", fontsize=17)
    plt.setp(ax_top.get_xticklabels(), visible=False)
    plt.setp(ax_right.get_yticklabels(), visible=False)
    for target_ax in (ax, ax_top, ax_right):
        target_ax.tick_params(direction="in", top=True, right=True, labelsize=10)
        for spine in target_ax.spines.values():
            spine.set_linewidth(1.1)

    fig.savefig(out_main, dpi=320, bbox_inches="tight")
    fig.savefig(out_alias, dpi=320, bbox_inches="tight")
    plt.close(fig)


def plot_mag_scatter_marginals_preferred_x(
    csv_path: Path,
    out_main: Path,
    out_alias: Path,
    *,
    x_cols: list[str],
    y_col: str,
    x_label: str,
    y_label: str,
) -> None:
    x, y, xerr, yerr, xt, yt = load_mag_arrays_preferred_x(csv_path, x_cols=x_cols, y_col=y_col)
    valid = np.isfinite(x) & np.isfinite(y)
    det_det = valid & (xt == "det") & (yt == "det")
    r_ul_uv_det = valid & (xt == "ul") & (yt == "det")
    r_det_uv_ul = valid & (xt == "det") & (yt == "ul")
    both_ul = valid & (xt == "ul") & (yt == "ul")

    x_all = x[valid]
    y_all = y[valid]
    x_margin = 0.65
    y_margin = 0.65
    x_left = np.nanmax(x_all) + x_margin
    x_right = np.nanmin(x_all) - x_margin
    y_bottom = np.nanmax(y_all) + y_margin
    y_top = np.nanmin(y_all) - y_margin

    fig = plt.figure(figsize=(8.3, 7.6))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 3.2], width_ratios=[3.2, 1.0], hspace=0.04, wspace=0.04)
    ax = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax)

    col_det = "#2E86B7"
    col_ul = "#D55E00"
    col_histx = "#E3A21A"
    col_histy = "#2FAE8F"

    x_for_hist = x[valid & (xt == "det")]
    y_for_hist = y[valid & (yt == "det")]
    if x_for_hist.size:
        ax_top.hist(x_for_hist, bins=24, color=col_histx, edgecolor="white", linewidth=0.35)
    if y_for_hist.size:
        ax_right.hist(y_for_hist, bins=24, orientation="horizontal", color=col_histy, edgecolor="white", linewidth=0.35)
    if np.any(det_det):
        med_x = np.nanmedian(x[det_det])
        med_y = np.nanmedian(y[det_det])
        ax_top.axvline(med_x, color="0.45", ls="--", lw=1.1, alpha=0.9)
        ax_right.axhline(med_y, color="0.45", ls="--", lw=1.1, alpha=0.9)
    if np.any(det_det):
        ax.errorbar(
            x[det_det], y[det_det], xerr=xerr[det_det], yerr=yerr[det_det],
            fmt="o", ms=5.0, mfc=col_det, mec="white", mew=0.25,
            ecolor=col_det, elinewidth=0.8, alpha=0.9, capsize=0, zorder=3,
        )

    ax.set_xlim(x_left, x_right)
    ax.set_ylim(y_bottom, y_top)
    x_span = abs(x_left - x_right)
    y_span = abs(y_bottom - y_top)
    dx = 0.030 * x_span
    dy = 0.030 * y_span
    padx = 0.015 * x_span
    pady = 0.015 * y_span

    def draw_x_ul(x0: float, y0: float) -> None:
        xl, xr = ax.get_xlim()
        sx = min(max(x0, xr + padx), xl - padx)
        ex = min(sx + dx, xl - padx)
        ax.annotate("", xy=(ex, y0), xytext=(sx, y0), arrowprops=dict(arrowstyle="-|>", color=col_ul, lw=1.0, mutation_scale=8), zorder=6, clip_on=False)

    def draw_y_ul(x0: float, y0: float) -> None:
        yb, yt_lim = ax.get_ylim()
        sy = min(max(y0, yt_lim + pady), yb - pady)
        ey = min(sy + dy, yb - pady)
        ax.annotate("", xy=(x0, ey), xytext=(x0, sy), arrowprops=dict(arrowstyle="-|>", color=col_ul, lw=1.0, mutation_scale=8), zorder=6, clip_on=False)

    def draw_diag_ul(x0: float, y0: float) -> None:
        xl, xr = ax.get_xlim()
        yb, yt_lim = ax.get_ylim()
        sx = min(max(x0, xr + padx), xl - padx)
        sy = min(max(y0, yt_lim + pady), yb - pady)
        ex = min(sx + dx, xl - padx)
        ey = min(sy + dy, yb - pady)
        ax.annotate("", xy=(ex, ey), xytext=(sx, sy), arrowprops=dict(arrowstyle="-|>", color=col_ul, lw=1.0, mutation_scale=8), zorder=6, clip_on=False)

    for idx in np.where(r_ul_uv_det)[0]:
        if np.isfinite(yerr[idx]):
            ax.errorbar(x[idx], y[idx], yerr=yerr[idx], fmt="none", ecolor=col_ul, elinewidth=0.85, alpha=0.85, zorder=4)
        draw_x_ul(float(x[idx]), float(y[idx]))
    for idx in np.where(r_det_uv_ul)[0]:
        if np.isfinite(xerr[idx]):
            ax.errorbar(x[idx], y[idx], xerr=xerr[idx], fmt="none", ecolor=col_ul, elinewidth=0.85, alpha=0.85, zorder=4)
        draw_y_ul(float(x[idx]), float(y[idx]))
    for idx in np.where(both_ul)[0]:
        draw_diag_ul(float(x[idx]), float(y[idx]))

    xmin, xmax = sorted(ax.get_xlim())
    ymin, ymax = sorted(ax.get_ylim())
    d0, d1 = max(xmin, ymin), min(xmax, ymax)
    if d1 > d0:
        t = np.linspace(d0, d1, 200)
        ax.plot(t, t, ls="--", lw=1.2, color="0.35", alpha=0.85, zorder=2)

    handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=6, markerfacecolor=col_det, markeredgecolor="white", markeredgewidth=0.3, label="Detections"),
        Line2D([0], [0], linestyle="-", color=col_ul, lw=1.0, marker=">", markersize=5, markerfacecolor=col_ul, markeredgecolor=col_ul, label="Upper limits"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=True, framealpha=0.92, facecolor="white", edgecolor="0.8", fontsize=11)
    ax.set_xlabel(x_label, fontsize=17)
    ax.set_ylabel(y_label, fontsize=17)
    ax.grid(color="0.80", alpha=0.5, linewidth=0.6)
    ax_top.set_ylabel("N", fontsize=17)
    ax_right.set_xlabel("N", fontsize=17)
    plt.setp(ax_top.get_xticklabels(), visible=False)
    plt.setp(ax_right.get_yticklabels(), visible=False)
    for target_ax in (ax, ax_top, ax_right):
        target_ax.tick_params(direction="in", top=True, right=True, labelsize=10)
        for spine in target_ax.spines.values():
            spine.set_linewidth(1.1)

    fig.savefig(out_main, dpi=320, bbox_inches="tight")
    fig.savefig(out_alias, dpi=320, bbox_inches="tight")
    plt.close(fig)


def plot_absmag_scatter_marginals(csv_path: Path, out_main: Path, out_alias: Path) -> None:
    plot_mag_scatter_marginals(
        csv_path,
        out_main,
        out_alias,
        x_col="Adopted r abs mag",
        y_col="UV abs mag",
        x_label=r"$M_r$ (mag)",
        y_label=r"$M_{F275W}$ (mag)",
    )


def plot_appmag_scatter_marginals(csv_path: Path, out_main: Path, out_alias: Path) -> None:
    plot_mag_scatter_marginals(
        csv_path,
        out_main,
        out_alias,
        x_col="Adopted r mag",
        y_col="UV mag",
        x_label=r"$r$ (mag)",
        y_label=r"$F275W$ (mag)",
    )


def plot_appmag_scatter_marginals_panstarrs_pref(csv_path: Path, out_main: Path, out_alias: Path) -> None:
    plot_mag_scatter_marginals_preferred_x(
        csv_path,
        out_main,
        out_alias,
        x_cols=["PanSTARRS r mag", "Adopted r mag"],
        y_col="UV mag",
        x_label=r"$r$ (mag; Pan-STARRS preferred)",
        y_label=r"$F275W$ (mag)",
    )


def plot_appmag_scatter_marginals_adopted_best(csv_path: Path, out_main: Path, out_alias: Path) -> None:
    plot_mag_scatter_marginals(
        csv_path,
        out_main,
        out_alias,
        x_col="Adopted Best",
        y_col="UV mag",
        x_label=r"$r$ (mag; Adopted Best)",
        y_label=r"$F275W$ (mag)",
    )


def plot_legacy_vs_panstarrs_r_scatter_marginals(csv_path: Path, out_main: Path, out_alias: Path) -> None:
    plot_mag_scatter_marginals(
        csv_path,
        out_main,
        out_alias,
        x_col="PanSTARRS r mag",
        y_col="LegacySurvey invvar+bkg r mag",
        x_label=r"Pan-STARRS $r$ (mag)",
        y_label=r"Legacy Survey $r$ (invvar+bkg; mag)",
    )


def load_detected_colors(photometry_csv: Path) -> np.ndarray:
    df = pd.read_csv(photometry_csv)
    uv_det = df["hst_1_is_upper_limit"].fillna(True).eq(False)
    r_det = df["rband_1_is_upper_limit"].fillna(True).eq(False)
    uv_mag = pd.to_numeric(df["WFC3_UVIS_F275W_1"], errors="coerce")
    r_mag = pd.to_numeric(df["rband_r_1"], errors="coerce")
    color = uv_mag - r_mag
    mask = uv_det & r_det & np.isfinite(color)
    return color[mask].to_numpy(dtype=float)


def plot_color_distribution(photometry_csv: Path, out_main: Path, out_alias: Path) -> None:
    color_vals = load_detected_colors(photometry_csv)
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.hist(color_vals, bins=18, color="#56B4E9", edgecolor="white", linewidth=0.45)
    if color_vals.size:
        med = float(np.nanmedian(color_vals))
        ax.axvline(med, color="0.35", ls="--", lw=1.4)
        ax.text(0.98, 0.96, f"median={med:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=11)
        ax.text(0.98, 0.88, f"N={len(color_vals)}", transform=ax.transAxes, ha="right", va="top", fontsize=11)
    ax.set_xlabel("F275W-r (mag)", fontsize=15)
    ax.set_ylabel("N", fontsize=15)
    ax.grid(axis="y", color="0.86", linewidth=0.6)
    ax.tick_params(direction="in", top=True, right=True, labelsize=10)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
    fig.savefig(out_main, dpi=320, bbox_inches="tight")
    fig.savefig(out_alias, dpi=320, bbox_inches="tight")
    plt.close(fig)


def parse_color_field(value: object) -> tuple[float, str]:
    s = str(value).strip()
    if s.lower() in {"nan", ""}:
        return np.nan, "nan"
    s = s.replace("$", "").strip()
    if s.startswith("?"):
        return np.nan, "amb"
    if s.startswith("<"):
        try:
            return float(s[1:].strip()), "lt"
        except ValueError:
            return np.nan, "nan"
    if s.startswith(">"):
        try:
            return float(s[1:].strip()), "gt"
        except ValueError:
            return np.nan, "nan"
    try:
        return float(s), "det"
    except ValueError:
        return np.nan, "nan"


def plot_uvr_adopted_vs_best(csv_path: Path, out_main: Path, out_alias: Path) -> None:
    df = pd.read_csv(csv_path)
    x_vals, x_typ = [], []
    y_vals, y_typ = [], []
    for _, row in df.iterrows():
        xv, xt = parse_color_field(row["F275W-r"])
        yv, yt = parse_color_field(row["F275W-r Best"])
        x_vals.append(xv)
        x_typ.append(xt)
        y_vals.append(yv)
        y_typ.append(yt)

    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)
    xt = np.asarray(x_typ)
    yt = np.asarray(y_typ)

    valid = np.isfinite(x) & np.isfinite(y)
    det_det = valid & (xt == "det") & (yt == "det")
    bound_any = valid & (xt != "det") | valid & (yt != "det")

    finite_all = np.concatenate([x[valid], y[valid]])
    lo = float(np.nanmin(finite_all) - 0.18)
    hi = float(np.nanmax(finite_all) + 0.18)

    fig = plt.figure(figsize=(8.1, 7.4))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 3.2], width_ratios=[3.2, 1.0], hspace=0.04, wspace=0.04)
    ax = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax)

    col_det = "#2E86B7"
    col_bound = "#D55E00"
    col_histx = "#E3A21A"
    col_histy = "#2FAE8F"

    x_det = x[valid & (xt == "det")]
    y_det = y[valid & (yt == "det")]
    if x_det.size:
        ax_top.hist(x_det, bins=20, color=col_histx, edgecolor="white", linewidth=0.35)
    if y_det.size:
        ax_right.hist(y_det, bins=20, orientation="horizontal", color=col_histy, edgecolor="white", linewidth=0.35)

    if np.any(det_det):
        ax.scatter(x[det_det], y[det_det], s=28, c=col_det, edgecolors="white", linewidths=0.35, alpha=0.92, zorder=3)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    span = hi - lo
    d = 0.035 * span

    def draw_arrow(x0: float, y0: float, dx: float, dy: float) -> None:
        ax.annotate(
            "",
            xy=(x0 + dx, y0 + dy),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>", color=col_bound, lw=0.95, mutation_scale=8),
            zorder=6,
            clip_on=False,
        )

    for idx in np.where(bound_any)[0]:
        xv, yv, xk, yk = float(x[idx]), float(y[idx]), xt[idx], yt[idx]
        dx = 0.0
        dy = 0.0
        if xk == "lt":
            dx = -d
        elif xk == "gt":
            dx = d
        if yk == "lt":
            dy = -d
        elif yk == "gt":
            dy = d
        if dx != 0.0 or dy != 0.0:
            draw_arrow(xv, yv, dx, dy)

    t = np.linspace(lo, hi, 200)
    ax.plot(t, t, ls="--", lw=1.2, color="0.35", alpha=0.85, zorder=2)

    handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=6, markerfacecolor=col_det, markeredgecolor="white", markeredgewidth=0.3, label="Detections"),
        Line2D([0], [0], linestyle="-", color=col_bound, lw=1.0, marker=">", markersize=5, markerfacecolor=col_bound, markeredgecolor=col_bound, label="Bounds"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=True, framealpha=0.92, facecolor="white", edgecolor="0.8", fontsize=11)
    ax.set_xlabel(r"F275W$-r$ (Adopted)", fontsize=16)
    ax.set_ylabel(r"F275W$-r$ (Adopted Best)", fontsize=16)
    ax.grid(color="0.80", alpha=0.5, linewidth=0.6)
    ax_top.set_ylabel("N", fontsize=16)
    ax_right.set_xlabel("N", fontsize=16)
    plt.setp(ax_top.get_xticklabels(), visible=False)
    plt.setp(ax_right.get_yticklabels(), visible=False)
    for target_ax in (ax, ax_top, ax_right):
        target_ax.tick_params(direction="in", top=True, right=True, labelsize=10)
        for spine in target_ax.spines.values():
            spine.set_linewidth(1.1)

    fig.savefig(out_main, dpi=320, bbox_inches="tight")
    fig.savefig(out_alias, dpi=320, bbox_inches="tight")
    plt.close(fig)


def sync_candidate_panels(candidate_names: list[str], plots_dir: Path, overleaf_img: Path) -> None:
    for name in candidate_names:
        src = plots_dir / f"{name}_plot.png"
        dst = overleaf_img / f"{name}_plot.png"
        if src.exists():
            shutil.copy2(src, dst)


def main() -> int:
    args = parse_args()
    sample_csv = Path(args.sample_csv)
    summary_csv = Path(args.summary_csv)
    photometry_csv = Path(args.photometry_csv)
    paper_candidates = Path(args.paper_candidates)
    plots_dir = Path(args.plots_dir)
    overleaf_dir = Path(args.overleaf_dir)
    overleaf_img = overleaf_dir / "img"
    overleaf_img.mkdir(parents=True, exist_ok=True)

    sample_df = pd.read_csv(sample_csv)
    summary_df = pd.read_csv(summary_csv)

    write_short_table(summary_df, overleaf_dir / "sn_mag_summary_table.tex")
    write_full_table(summary_df, overleaf_dir / "sn_mag_summary_table_full.tex")
    plot_redshift_distribution(sample_df, overleaf_img / "redshift_distribution.png")
    plot_absmag_scatter_marginals(
        summary_csv,
        overleaf_img / "absmag_scatter_marginals_snr3.png",
        overleaf_img / "absmag_scatter_marginals.png",
    )
    plot_appmag_scatter_marginals(
        summary_csv,
        overleaf_img / "appmag_scatter_marginals_snr3.png",
        overleaf_img / "appmag_scatter_marginals.png",
    )
    plot_appmag_scatter_marginals_panstarrs_pref(
        summary_csv,
        overleaf_img / "appmag_scatter_marginals_panstarrs_pref_snr3.png",
        overleaf_img / "appmag_scatter_marginals_panstarrs_pref.png",
    )
    plot_appmag_scatter_marginals_adopted_best(
        summary_csv,
        overleaf_img / "appmag_scatter_marginals_adopted_best_snr3.png",
        overleaf_img / "appmag_scatter_marginals_adopted_best.png",
    )
    plot_legacy_vs_panstarrs_r_scatter_marginals(
        summary_csv,
        overleaf_img / "legacy_panstarrs_r_scatter_marginals_snr3.png",
        overleaf_img / "legacy_panstarrs_r_scatter_marginals.png",
    )
    plot_color_distribution(
        photometry_csv,
        overleaf_img / "f275w_r_color_distribution_snr3.png",
        overleaf_img / "f275w_r_color_distribution.png",
    )
    plot_uvr_adopted_vs_best(
        summary_csv,
        overleaf_img / "uvr_adopted_vs_best_snr3.png",
        overleaf_img / "uvr_adopted_vs_best.png",
    )
    candidate_names = [ln.strip() for ln in paper_candidates.read_text().splitlines() if ln.strip()]
    sync_candidate_panels(candidate_names, plots_dir, overleaf_img)
    print(f"Exported Overleaf assets for {len(summary_df)} rows and {len(candidate_names)} paper candidates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
