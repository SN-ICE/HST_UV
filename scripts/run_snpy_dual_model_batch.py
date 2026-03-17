#!/usr/bin/env python3
"""
Run SNooPy fits over local SNooPy light-curve files and write a summary CSV.

The script is resumable: if the output CSV already contains a row for a given SN name,
that file is skipped unless --overwrite-existing is passed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from project_paths import PROJECT_ROOT, light_curves_path


SUMMARY_BANDS = ["B", "g", "r"]
# Galactic extinction coefficients used for corrected peak magnitudes.
# Matches the values you were using for g/r and extends the same treatment to B.
EXTINCTION_COEFF = {
    "B": 4.10,
    "g": 3.83,
    "r": 2.58,
}
UV_FILTER_TOKENS = ("uvm2", "uvw1", "uvw2", "fuv", "nuv")
OPTICAL_RESTBANDS = {"u", "B", "V", "g", "r", "i", "Y", "J", "H", "K", "Rs", "Is"}


def canonical_band_family(band: str) -> str | None:
    band_lower = str(band).strip().lower()
    if not band_lower:
        return None

    if band_lower.startswith("b"):
        return "B"

    if (
        band_lower.startswith("g")
        or band_lower in {"ztfg", "ltg", "atgr"}
        or band_lower.endswith("_g")
        or band_lower.endswith("g")
    ):
        return "g"

    if (
        band_lower.startswith("r")
        or band_lower in {"ztfr", "ltr", "rkait", "rs", "atri"}
        or band_lower.endswith("_r")
        or band_lower.endswith("r")
    ):
        return "r"

    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-run SNooPy fits over local light curves.")
    parser.add_argument(
        "--snoopy-dir",
        default=str(light_curves_path("SNooPy")),
        help="Directory containing local SNooPy TXT files.",
    )
    parser.add_argument(
        "--input-csv",
        default=str(PROJECT_ROOT / "data" / "supernovas_input.csv"),
        help="Main sample table used for metadata fallback.",
    )
    parser.add_argument(
        "--output-csv",
        default=str(PROJECT_ROOT / "outputs" / "snpy_fit_summary.csv"),
        help="Summary CSV written incrementally after each processed SN.",
    )
    parser.add_argument(
        "--plots-dir",
        default=str(PROJECT_ROOT / "outputs" / "snpy_plots"),
        help="Directory for SNooPy fit plots.",
    )
    parser.add_argument(
        "--raw-plots-dir",
        default=str(PROJECT_ROOT / "outputs" / "snpy_raw_plots"),
        help="Directory for raw photometry plots saved before any fit.",
    )
    parser.add_argument(
        "--fits-dir",
        default=str(PROJECT_ROOT / "outputs" / "snpy_fits"),
        help="Directory for serialized fitted SNooPy objects.",
    )
    parser.add_argument("--model", default="max_model", help="Primary SNooPy model name.")
    parser.add_argument("--ebv-model", default="EBV_model2", help="Secondary SNooPy model used to fit EBVhost.")
    parser.add_argument("--stype", default="st", help="SNooPy stretch/color parameterization.")
    parser.add_argument("--gen", type=int, default=3, help="Generation passed to s.fit(..., gen=...).")
    parser.add_argument(
        "--names",
        nargs="*",
        default=None,
        help="Optional list of SN names to process; default is all *_raw.txt files.",
    )
    parser.add_argument(
        "--names-file",
        default=None,
        help="Optional text file with one SN name per line to process.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of files to process.")
    parser.add_argument("--no-plots", action="store_true", help="Disable SNooPy PDF plot output.")
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Re-fit objects already present in the summary CSV.",
    )
    parser.add_argument(
        "--ebvgal-default",
        type=float,
        default=0.0,
        help="Fallback Galactic E(B-V) if SNooPy cannot fetch IRSA extinction.",
    )
    return parser.parse_args()


def normalize_name(name: str) -> str:
    return str(name).strip().lower()


def load_sample_metadata(input_csv: Path) -> dict[str, dict]:
    if not input_csv.exists():
        return {}
    df = pd.read_csv(input_csv)
    out: dict[str, dict] = {}
    for _, row in df.iterrows():
        name = str(row.get("matched_snname", "")).strip()
        if not name:
            continue
        out[normalize_name(name)] = row.to_dict()
    return out


def discover_files(snoopy_dir: Path, names: Iterable[str] | None) -> list[Path]:
    wanted = None if names is None else {normalize_name(name) for name in names}
    files = []
    for path in sorted(snoopy_dir.glob("*_raw.txt")):
        sn_name = path.stem.removesuffix("_raw")
        if wanted is not None and normalize_name(sn_name) not in wanted:
            continue
        files.append(path)
    return files


def load_names_file(path: str | None) -> list[str] | None:
    if not path:
        return None
    names_path = Path(path).expanduser().resolve()
    if not names_path.exists():
        raise FileNotFoundError(f"Names file not found: {names_path}")
    names = []
    for line in names_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        names.append(line)
    return names


def load_existing(output_csv: Path) -> pd.DataFrame:
    if not output_csv.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(output_csv)
    except Exception:
        return pd.DataFrame()


def existing_done_names(existing: pd.DataFrame) -> set[str]:
    if len(existing) == 0 or "sn_name" not in existing.columns:
        return set()
    return {normalize_name(name) for name in existing["sn_name"].astype(str) if str(name).strip()}


def canonical_restband(restband_map: dict, band: str) -> str:
    try:
        rb_val = restband_map[band]
    except Exception:
        rb_val = ""
    rb = str(rb_val).strip()
    family = canonical_band_family(rb)
    if family is not None:
        return family
    family = canonical_band_family(band)
    if family is not None:
        return family
    if rb in OPTICAL_RESTBANDS or rb in {"UVM2", "UVW1", "UVW2"}:
        return rb
    return rb or band


def has_bgr_family(s) -> bool:
    for band in s.data.keys():
        canonical = canonical_restband(getattr(s, "restbands", {}), band)
        if canonical in {"B", "g", "r", "Rs"}:
            return True
    return False


def apply_atlas_restband_fallback(s) -> None:
    for band in s.data.keys():
        family = canonical_band_family(band)
        if family is not None:
            s.restbands[band] = family

    bands = set(s.data.keys())
    if has_bgr_family(s):
        return
    if "ATgr" in bands:
        s.restbands["ATgr"] = "g"
    if "ATri" in bands:
        s.restbands["ATri"] = "r"


def candidate_band_lists(s) -> list[list[str]]:
    apply_atlas_restband_fallback(s)
    all_bands = list(s.data.keys())
    restbands = getattr(s, "restbands", {})
    candidates: list[list[str]] = []

    def add_candidate(seq: list[str]) -> None:
        cleaned = [band for band in seq if band in all_bands]
        if len(cleaned) < 2:
            return
        if cleaned not in candidates:
            candidates.append(cleaned)

    add_candidate(all_bands)
    add_candidate([b for b in all_bands if not any(tok in b.lower() for tok in UV_FILTER_TOKENS)])
    add_candidate([b for b in all_bands if canonical_restband(restbands, b) in OPTICAL_RESTBANDS])
    add_candidate(
        [
            b
            for b in all_bands
            if canonical_restband(restbands, b) in {"u", "B", "V", "g", "r", "i", "Rs", "Is"}
        ]
    )
    return candidates if candidates else [all_bands]


def finite_or_nan(value) -> float:
    try:
        x = float(value)
    except Exception:
        return np.nan
    return x if np.isfinite(x) else np.nan


def compute_kcorr(s, band: str, tmax: float | None = None) -> float:
    try:
        ks = s.ks[band]
        band_obj = s.data[band]
        t_eval = finite_or_nan(tmax)
        if not np.isfinite(t_eval):
            t_eval = finite_or_nan(getattr(band_obj, "Tmax", np.nan))
        if not np.isfinite(t_eval):
            return np.nan
        return float(np.interp(float(t_eval), np.asarray(band_obj.MJD), np.asarray(ks)))
    except Exception:
        return np.nan


def get_peak_metrics(s, band: str) -> dict[str, float]:
    try:
        tmax, mmax, e_mmax, _ = s.get_max([band], use_model=1)
        tmax = finite_or_nan(np.asarray(tmax).reshape(-1)[0])
        mmax = finite_or_nan(np.asarray(mmax).reshape(-1)[0])
        e_mmax = finite_or_nan(np.asarray(e_mmax).reshape(-1)[0])
        return {
            "Tmax": tmax,
            "e_Tmax": np.nan,
            "Mmax": mmax,
            "e_Mmax": e_mmax,
        }
    except Exception:
        band_obj = s.data.get(band)
        if band_obj is None:
            return {"Tmax": np.nan, "e_Tmax": np.nan, "Mmax": np.nan, "e_Mmax": np.nan}
        return {
            "Tmax": finite_or_nan(getattr(band_obj, "Tmax", np.nan)),
            "e_Tmax": finite_or_nan(getattr(band_obj, "e_Tmax", np.nan)),
            "Mmax": finite_or_nan(getattr(band_obj, "Mmax", np.nan)),
            "e_Mmax": finite_or_nan(getattr(band_obj, "e_Mmax", np.nan)),
        }


def select_best_band_entries(s, bands_used: list[str], canonical_targets: list[str]) -> dict[str, tuple[str, object]]:
    restbands = getattr(s, "restbands", {})
    best_by_canonical: dict[str, tuple[str, object, float, float]] = {}

    for band in bands_used:
        band_obj = s.data.get(band)
        if band_obj is None:
            continue
        canonical = canonical_restband(restbands, band)
        if canonical not in canonical_targets:
            continue
        peak = get_peak_metrics(s, band)
        e_mmax = peak["e_Mmax"]
        mjd = getattr(band_obj, "MJD", [])
        npts = len(mjd) if mjd is not None else 0
        score = (np.isfinite(e_mmax), -e_mmax if np.isfinite(e_mmax) else -np.inf, npts)
        prev = best_by_canonical.get(canonical)
        prev_npts = 0
        if prev is not None:
            prev_mjd = getattr(prev[1], "MJD", [])
            prev_npts = len(prev_mjd) if prev_mjd is not None else 0
        if prev is None or score > (prev[2], prev[3], prev_npts):
            best_by_canonical[canonical] = (band, band_obj, score[0], score[1])

    out: dict[str, tuple[str, object]] = {}
    for canonical, value in best_by_canonical.items():
        out[canonical] = (value[0], value[1])
    return out


def extract_band_metrics(s, bands_used: list[str]) -> dict[str, float | str]:
    best_by_canonical = select_best_band_entries(s, bands_used, SUMMARY_BANDS)
    out: dict[str, float | str] = {}
    ebvgal = finite_or_nan(getattr(s, "EBVgal", np.nan))
    for canonical in SUMMARY_BANDS:
        if canonical not in best_by_canonical:
            continue
        band, band_obj = best_by_canonical[canonical]
        peak = get_peak_metrics(s, band)
        kcorr = compute_kcorr(s, band, peak["Tmax"])
        mmax = peak["Mmax"]
        coeff = EXTINCTION_COEFF.get(canonical)
        mcorr = mmax - coeff * ebvgal - kcorr if np.isfinite(mmax) and np.isfinite(kcorr) and coeff is not None else np.nan
        out[f"{canonical}_source"] = band
        out[f"{canonical}_Tmax"] = peak["Tmax"]
        out[f"{canonical}_e_Tmax"] = peak["e_Tmax"]
        out[f"{canonical}_Mmax"] = mmax
        out[f"{canonical}_e_Mmax"] = peak["e_Mmax"]
        out[f"{canonical}_kcor"] = kcorr
        out[f"{canonical}_Mcorr"] = mcorr
    return out


def compute_sbv(s, bands_used: list[str]) -> tuple[float, float]:
    best = select_best_band_entries(s, bands_used, ["B", "V"])
    if "B" not in best or "V" not in best:
        return np.nan, np.nan
    try:
        sbv, e_sbv = s.color_stretch(best["B"][0], best["V"][0])
        return finite_or_nan(sbv), finite_or_nan(e_sbv)
    except Exception:
        return np.nan, np.nan


SUMMARY_COLUMNS = [
    "sn_name",
    "sample_name",
    "source_file",
    "status",
    "fit_message",
    "max_status",
    "max_fit_message",
    "ebv2_status",
    "ebv2_fit_message",
    "z",
    "sample_redshift",
    "sample_rasn",
    "sample_decsn",
    "sample_sntype",
    "sample_redshift_source",
    "max_Tmax",
    "max_e_Tmax",
    "max_sBV",
    "max_e_sBV",
    "max_st",
    "max_e_st",
    "ebv2_Tmax",
    "ebv2_e_Tmax",
    "ebv2_sBV",
    "ebv2_e_sBV",
    "ebv2_st",
    "ebv2_e_st",
    "ebv2_EBVhost",
    "ebv2_e_EBVhost",
    "B_Tmax",
    "B_e_Tmax",
    "g_Tmax",
    "g_e_Tmax",
    "r_Tmax",
    "r_e_Tmax",
    "B_Mmax",
    "B_e_Mmax",
    "g_Mmax",
    "g_e_Mmax",
    "r_Mmax",
    "r_e_Mmax",
    "B_Mcorr",
    "B_kcor",
    "g_Mcorr",
    "g_kcor",
    "r_Mcorr",
    "r_kcor",
    "EBVgal",
    "max_rchisq",
    "ebv2_rchisq",
    "bands_available",
    "max_bands_used",
    "max_fit_attempts",
    "ebv2_bands_used",
    "ebv2_fit_attempts",
    "k_version",
    "stype",
    "raw_plot_file",
    "max_plot_file",
    "ebv2_plot_file",
    "max_fit_file",
    "ebv2_fit_file",
    "B_source",
    "g_source",
    "r_source",
]


def fit_single_model(
    snpy_module,
    path: Path,
    model: str,
    stype: str,
    gen: int,
    ebvgal_default: float,
):
    s = snpy_module.get_sn(str(path))
    apply_atlas_restband_fallback(s)
    ebvgal = finite_or_nan(getattr(s, "EBVgal", np.nan))
    if not np.isfinite(ebvgal):
        s.EBVgal = float(ebvgal_default)
        ebvgal = float(ebvgal_default)

    attempts = []
    last_error = ""
    fit_ok = False
    bands_used: list[str] = []
    for bands in candidate_band_lists(s):
        attempts.append(",".join(bands))
        try:
            s.choose_model(model, stype=stype)
            s.k_version = "H3"
            s.fit(bands, gen=gen)
            bands_used = list(bands)
            fit_ok = True
            break
        except Exception as exc:
            last_error = str(exc)
            s = snpy_module.get_sn(str(path))
            apply_atlas_restband_fallback(s)
            if not np.isfinite(finite_or_nan(getattr(s, "EBVgal", np.nan))):
                s.EBVgal = float(ebvgal_default)

    return {
        "sn_name": str(getattr(s, "name", path.stem.removesuffix("_raw"))),
        "source_file": str(path),
        "status": "ok" if fit_ok else "failed",
        "fit_message": "" if fit_ok else last_error,
        "bands_available": ",".join(list(s.data.keys())),
        "bands_used": ",".join(bands_used),
        "fit_attempts": " | ".join(attempts),
        "z": finite_or_nan(getattr(s, "z", np.nan)),
        "Tmax": finite_or_nan(getattr(s, "Tmax", np.nan)),
        "e_Tmax": finite_or_nan(getattr(s, "e_Tmax", np.nan)),
        "st": finite_or_nan(getattr(s, "st", np.nan)),
        "e_st": finite_or_nan(getattr(s, "e_st", np.nan)),
        "EBVgal": ebvgal,
        "EBVhost": finite_or_nan(getattr(s, "EBVhost", np.nan)),
        "e_EBVhost": finite_or_nan(getattr(s, "e_EBVhost", np.nan)),
        "rchisq": finite_or_nan(getattr(s, "rchisq", np.nan)),
        "k_version": str(getattr(s, "k_version", "")),
        "stype": stype,
        "s": s,
        "bands_used_list": bands_used,
    }


def fit_one(
    snpy_module,
    path: Path,
    raw_plot_path: Path | None,
    max_plot_path: Path | None,
    ebv_plot_path: Path | None,
    fits_dir: Path,
    max_model: str,
    ebv_model: str,
    stype: str,
    gen: int,
    ebvgal_default: float,
) -> dict:
    raw_plot_file = ""
    if raw_plot_path is not None:
        try:
            raw_s = snpy_module.get_sn(str(path))
            apply_atlas_restband_fallback(raw_s)
            raw_plot_path.parent.mkdir(parents=True, exist_ok=True)
            raw_s.plot(outfile=str(raw_plot_path))
            raw_plot_file = str(raw_plot_path)
        except Exception:
            raw_plot_file = ""

    max_fit = fit_single_model(
        snpy_module=snpy_module,
        path=path,
        model=max_model,
        stype=stype,
        gen=gen,
        ebvgal_default=ebvgal_default,
    )
    ebv_fit = fit_single_model(
        snpy_module=snpy_module,
        path=path,
        model=ebv_model,
        stype=stype,
        gen=gen,
        ebvgal_default=ebvgal_default,
    )

    max_s, max_bands = max_fit["s"], max_fit["bands_used_list"]
    ebv_s, ebv_bands = ebv_fit["s"], ebv_fit["bands_used_list"]
    max_sbv, max_e_sbv = compute_sbv(max_s, max_bands) if max_fit["status"] == "ok" else (np.nan, np.nan)
    ebv_sbv, ebv_e_sbv = compute_sbv(ebv_s, ebv_bands) if ebv_fit["status"] == "ok" else (np.nan, np.nan)

    max_ok = max_fit["status"] == "ok"
    ebv_ok = ebv_fit["status"] == "ok"
    if max_ok and ebv_ok:
        status = "ok"
    elif max_ok or ebv_ok:
        status = "partial"
    else:
        status = "failed"

    messages = [msg for msg in [max_fit["fit_message"], ebv_fit["fit_message"]] if msg]
    max_fit_file = fits_dir / f"{max_fit['sn_name']}_maxmodel.snpy"
    ebv_fit_file = fits_dir / f"{max_fit['sn_name']}_ebvmodel2.snpy"
    row = {
        "sn_name": max_fit["sn_name"],
        "source_file": max_fit["source_file"],
        "status": status,
        "fit_message": " | ".join(messages),
        "max_status": max_fit["status"],
        "max_fit_message": max_fit["fit_message"],
        "ebv2_status": ebv_fit["status"],
        "ebv2_fit_message": ebv_fit["fit_message"],
        "bands_available": max_fit["bands_available"],
        "max_bands_used": max_fit["bands_used"],
        "max_fit_attempts": max_fit["fit_attempts"],
        "ebv2_bands_used": ebv_fit["bands_used"],
        "ebv2_fit_attempts": ebv_fit["fit_attempts"],
        "z": max_fit["z"],
        "EBVgal": max_fit["EBVgal"],
        "k_version": max_fit["k_version"],
        "stype": stype,
        "raw_plot_file": raw_plot_file,
        "max_Tmax": max_fit["Tmax"],
        "max_e_Tmax": max_fit["e_Tmax"],
        "max_st": max_fit["st"],
        "max_e_st": max_fit["e_st"],
        "max_sBV": max_sbv,
        "max_e_sBV": max_e_sbv,
        "max_rchisq": max_fit["rchisq"],
        "ebv2_Tmax": ebv_fit["Tmax"],
        "ebv2_e_Tmax": ebv_fit["e_Tmax"],
        "ebv2_st": ebv_fit["st"],
        "ebv2_e_st": ebv_fit["e_st"],
        "ebv2_sBV": ebv_sbv,
        "ebv2_e_sBV": ebv_e_sbv,
        "ebv2_EBVhost": ebv_fit["EBVhost"],
        "ebv2_e_EBVhost": ebv_fit["e_EBVhost"],
        "ebv2_rchisq": ebv_fit["rchisq"],
        "max_plot_file": "",
        "ebv2_plot_file": "",
        "max_fit_file": "",
        "ebv2_fit_file": "",
    }

    if max_ok:
        row.update(extract_band_metrics(max_s, max_bands))
        if max_plot_path is not None:
            max_plot_path.parent.mkdir(parents=True, exist_ok=True)
            max_s.plot(outfile=str(max_plot_path))
            row["max_plot_file"] = str(max_plot_path)
        fits_dir.mkdir(parents=True, exist_ok=True)
        max_s.save(str(max_fit_file))
        row["max_fit_file"] = str(max_fit_file)

    if ebv_ok:
        if ebv_plot_path is not None:
            ebv_plot_path.parent.mkdir(parents=True, exist_ok=True)
            ebv_s.plot(outfile=str(ebv_plot_path))
            row["ebv2_plot_file"] = str(ebv_plot_path)
        fits_dir.mkdir(parents=True, exist_ok=True)
        ebv_s.save(str(ebv_fit_file))
        row["ebv2_fit_file"] = str(ebv_fit_file)

    return row


def merge_metadata(row: dict, metadata: dict[str, dict]) -> dict:
    meta = metadata.get(normalize_name(row["sn_name"]), {})
    for source, target in [
        ("matched_snname", "sample_name"),
        ("redshift", "sample_redshift"),
        ("rasn", "sample_rasn"),
        ("decsn", "sample_decsn"),
        ("sntype", "sample_sntype"),
        ("redshift_source", "sample_redshift_source"),
    ]:
        row[target] = meta.get(source, "")
    return row


def write_rows(rows: list[dict], output_csv: Path) -> None:
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["sn_name", "source_file"], kind="stable").reset_index(drop=True)
        preferred = [col for col in SUMMARY_COLUMNS if col in df.columns]
        extra = [col for col in df.columns if col not in preferred]
        df = df[preferred + extra]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)


def main() -> int:
    args = parse_args()

    import snpy as sn

    snoopy_dir = Path(args.snoopy_dir).expanduser().resolve()
    input_csv = Path(args.input_csv).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    plots_dir = Path(args.plots_dir).expanduser().resolve()
    raw_plots_dir = Path(args.raw_plots_dir).expanduser().resolve()
    fits_dir = Path(args.fits_dir).expanduser().resolve()

    selected_names = []
    if args.names:
        selected_names.extend(args.names)
    names_from_file = load_names_file(args.names_file)
    if names_from_file:
        selected_names.extend(names_from_file)
    names_arg = selected_names if selected_names else None

    files = discover_files(snoopy_dir, names_arg)
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        print("No matching SNooPy light-curve files found.")
        return 1

    metadata = load_sample_metadata(input_csv)
    existing = load_existing(output_csv)
    done = set() if args.overwrite_existing else existing_done_names(existing)
    rows = existing.to_dict("records") if (len(existing) and not args.overwrite_existing) else []

    processed = 0
    skipped = 0
    for index, path in enumerate(files, start=1):
        sn_name = path.stem.removesuffix("_raw")
        if normalize_name(sn_name) in done:
            skipped += 1
            continue

        print(f"[{index}/{len(files)}] Fitting {sn_name}")
        raw_plot_path = raw_plots_dir / f"{sn_name}_raw_plot.jpg"
        max_plot_path = None if args.no_plots else plots_dir / f"{sn_name}_maxmodel_plot.jpg"
        ebv_plot_path = None if args.no_plots else plots_dir / f"{sn_name}_ebvmodel2_plot.jpg"
        try:
            row = fit_one(
                snpy_module=sn,
                path=path,
                raw_plot_path=raw_plot_path,
                max_plot_path=max_plot_path,
                ebv_plot_path=ebv_plot_path,
                fits_dir=fits_dir,
                max_model=args.model,
                ebv_model=args.ebv_model,
                stype=args.stype,
                gen=args.gen,
                ebvgal_default=args.ebvgal_default,
            )
        except Exception as exc:
            row = {
                "sn_name": sn_name,
                "source_file": str(path),
                "status": "failed",
                "fit_message": str(exc),
                "max_status": "import_failed",
                "max_fit_message": str(exc),
                "ebv2_status": "import_failed",
                "ebv2_fit_message": str(exc),
                "bands_available": "",
                "max_bands_used": "",
                "max_fit_attempts": "",
                "ebv2_bands_used": "",
                "ebv2_fit_attempts": "",
                "z": np.nan,
                "EBVgal": np.nan,
                "k_version": "",
                "stype": args.stype,
                "raw_plot_file": "",
                "max_plot_file": "",
                "ebv2_plot_file": "",
                "max_fit_file": "",
                "ebv2_fit_file": "",
            }
        row = merge_metadata(row, metadata)
        rows.append(row)
        write_rows(rows, output_csv)
        processed += 1

    print(f"Processed now: {processed}")
    print(f"Skipped existing: {skipped}")
    print(f"Summary CSV: {output_csv}")
    if not args.no_plots:
        print(f"Plots dir: {plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
