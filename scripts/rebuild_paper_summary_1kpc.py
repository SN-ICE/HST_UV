#!/usr/bin/env python3
"""Rebuild the paper-style 1 kpc summary table from the main photometry summary."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
from astropy.cosmology import WMAP9


UV_MAG = "WFC3_UVIS_F275W_1"
UV_MAG_ERR = "WFC3_UVIS_F275W_1_err"
UV_MAG_ABS = "WFC3_UVIS_F275W_1_abs"
UV_IS_UL = "hst_1_is_upper_limit"
UV_MAG_UL = "hst_1_mag_upper_limit"
UV_MAG_UL_ABS = "hst_1_mag_upper_limit_abs"
SURVEY_ORDER = ("LegacySurvey", "DES", "PanSTARRS", "SkyMapper")
SURVEY_PREFIX = {
    "DES": "des_rband",
    "PanSTARRS": "panstarrs_rband",
    "SkyMapper": "skymapper_rband",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--paper-full", required=True)
    parser.add_argument("--paper-short", required=True)
    parser.add_argument("--images-root", default="outputs/images")
    parser.add_argument("--adopted-overrides", default="data/adopted_r_survey_overrides.csv")
    parser.add_argument(
        "--legacy-alt-csv",
        default="outputs/legacy_blank_aperture_test_comparison.csv",
        help="Optional standalone Legacy comparison table (invvar+background) to add as parallel columns and adopt when LegacySurvey is selected.",
    )
    return parser.parse_args()


def to_float(value: object) -> float | None:
    series = pd.to_numeric(pd.Series([value]), errors="coerce")
    if series.notna().iloc[0]:
        numeric = float(series.iloc[0])
        if math.isfinite(numeric):
            return numeric
    return None


def to_bool(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes"}:
            return True
        if text in {"false", "0", "no", ""}:
            return False
    return bool(value)


def format_mag_value(mag: float | None, err: float | None, is_ul: bool, ul_mag: float | None) -> str:
    if is_ul and ul_mag is not None:
        return f"({ul_mag:.3f})"
    if mag is not None:
        if err is not None:
            return f"{mag:.3f} ({err:.3f})"
        return f"{mag:.3f}"
    return ""


def infer_distance_modulus(row: pd.Series) -> float | None:
    candidate_pairs = [
        ("rband_r_1", "rband_r_1_abs"),
        ("des_rband_r_1", "des_rband_r_1_abs"),
        ("panstarrs_rband_r_1", "panstarrs_rband_r_1_abs"),
        ("skymapper_rband_r_1", "skymapper_rband_r_1_abs"),
    ]
    for mag_col, abs_col in candidate_pairs:
        mag = to_float(row.get(mag_col))
        abs_mag = to_float(row.get(abs_col))
        if mag is not None and abs_mag is not None:
            return mag - abs_mag
    redshift = to_float(row.get("redshift"))
    if redshift is None:
        return None
    if redshift <= 0:
        return None
    return float(WMAP9.distmod(redshift).value)
    return None


def load_local_photometry(images_root: Path, sn_name: str, survey: str) -> dict[str, float | str] | None:
    phot_csv = images_root / sn_name / survey / "local_photometry.csv"
    if not phot_csv.exists():
        return None
    phot = pd.read_csv(phot_csv)
    if phot.empty:
        return None
    last = phot.iloc[-1]
    return {
        "survey": str(last.get("survey", survey)),
        "r_1": to_float(last.get("r_1")),
        "r_1_err": to_float(last.get("r_1_err")),
        "r_1_flux": to_float(last.get("r_1_flux")),
        "r_1_flux_err": to_float(last.get("r_1_flux_err")),
        "r_zeropoint": to_float(last.get("r_zeropoint")),
    }


def get_upper_limit_info(row: pd.Series, survey: str) -> tuple[bool, float | None, float | None]:
    if survey == "LegacySurvey":
        if str(row.get("rband_survey", "")).strip() == "LegacySurvey":
            return (
                to_bool(row.get("rband_1_is_upper_limit")),
                to_float(row.get("rband_1_mag_upper_limit")),
                to_float(row.get("rband_1_mag_upper_limit_abs")),
            )
        return False, None, None

    prefix = SURVEY_PREFIX.get(survey)
    if not prefix:
        return False, None, None
    return (
        to_bool(row.get(f"{prefix}_1_is_upper_limit")),
        to_float(row.get(f"{prefix}_1_mag_upper_limit")),
        to_float(row.get(f"{prefix}_1_mag_upper_limit_abs")),
    )


def build_survey_measurement(
    row: pd.Series,
    images_root: Path,
    survey: str,
    distance_modulus: float | None,
) -> dict[str, object]:
    phot = load_local_photometry(images_root, str(row["Supernova"]), survey)
    flux = None
    flux_err = None
    snr = None
    mag = None
    mag_err = None
    abs_mag = None
    is_ul = False
    ul_mag = None
    ul_abs_mag = None

    if phot:
        flux = phot["r_1_flux"]
        flux_err = phot["r_1_flux_err"]
        if flux is not None and flux_err not in (None, 0):
            snr = flux / flux_err
        mag = phot["r_1"]
        mag_err = phot["r_1_err"]
        is_ul, ul_mag, ul_abs_mag = get_upper_limit_info(row, survey)
        if distance_modulus is not None and mag is not None:
            abs_mag = mag - distance_modulus
        if is_ul and ul_abs_mag is None and ul_mag is not None and distance_modulus is not None:
            ul_abs_mag = ul_mag - distance_modulus

    return {
        "survey": survey,
        "mag": mag,
        "mag_err": mag_err,
        "is_upper_limit": is_ul,
        "upper_limit_mag": ul_mag,
        "mag_text": format_mag_value(mag, mag_err, is_ul, ul_mag),
        "abs_mag_text": format_mag_value(abs_mag, mag_err, is_ul, ul_abs_mag),
        "flux": flux,
        "flux_err": flux_err,
        "snr": snr,
        "zeropoint": phot["r_zeropoint"] if phot else None,
    }


def infer_adopted_survey_name(row: pd.Series) -> str:
    raw_adopted = row.get("rband_survey", "")
    if pd.isna(raw_adopted):
        return ""
    else:
        return str(raw_adopted).strip()


def load_adopted_overrides(path_str: str) -> dict[str, str]:
    path = Path(path_str)
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "Supernova" not in df.columns or "Adopted r survey override" not in df.columns:
        return {}
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        sn = str(row.get("Supernova", "")).strip()
        survey = str(row.get("Adopted r survey override", "")).strip()
        if sn and survey:
            out[sn] = survey
    return out


def load_legacy_alt_measurements(path_str: str) -> dict[str, dict[str, object]]:
    path = Path(path_str)
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    required = {"SN name", "legacy_blank_status", "legacy_blank_mag", "legacy_blank_mag_err", "legacy_blank_flux", "legacy_blank_flux_err"}
    if not required.issubset(df.columns):
        return {}
    out: dict[str, dict[str, object]] = {}
    for _, row in df.iterrows():
        sn = str(row.get("SN name", "")).strip()
        if not sn:
            continue
        status = str(row.get("legacy_blank_status", "")).strip()
        if status not in {"ok", "ok_invvar_only"}:
            continue
        mag = to_float(row.get("legacy_blank_mag"))
        mag_err = to_float(row.get("legacy_blank_mag_err"))
        flux = to_float(row.get("legacy_blank_flux"))
        flux_err = to_float(row.get("legacy_blank_flux_err"))
        snr = None
        if flux is not None and flux_err not in (None, 0):
            snr = flux / flux_err
        out[sn] = {
            "survey": "LegacySurvey",
            "mag": mag,
            "mag_err": mag_err,
            "flux": flux,
            "flux_err": flux_err,
            "snr": snr,
            "zeropoint": 22.5,
        }
    return out


def survey_has_measurement(measurement: dict[str, object]) -> bool:
    if measurement.get("mag_text") not in ("", None):
        return True
    if measurement.get("abs_mag_text") not in ("", None):
        return True
    flux = measurement.get("flux")
    flux_err = measurement.get("flux_err")
    if isinstance(flux, (int, float)) and flux != 0.0:
        return True
    if isinstance(flux_err, (int, float)) and flux_err != 0.0:
        return True
    return False


def survey_meets_adoption_rule(survey: str, measurement: dict[str, object]) -> bool:
    if not survey_has_measurement(measurement):
        return False
    snr = measurement.get("snr")
    if isinstance(snr, (int, float)) and math.isfinite(float(snr)):
        return float(snr) >= 3.0
    return False


def with_upper_limit_fallback(measurement: dict[str, object], distance_modulus: float | None) -> dict[str, object] | None:
    if not survey_has_measurement(measurement):
        return None
    if measurement.get("is_upper_limit") and measurement.get("upper_limit_mag") is not None:
        return measurement

    snr = measurement.get("snr")
    flux_err = measurement.get("flux_err")
    zeropoint = measurement.get("zeropoint")
    if (
        isinstance(snr, (int, float))
        and math.isfinite(float(snr))
        and float(snr) < 3.0
        and isinstance(flux_err, (int, float))
        and math.isfinite(float(flux_err))
        and float(flux_err) > 0.0
        and isinstance(zeropoint, (int, float))
        and math.isfinite(float(zeropoint))
    ):
        upper_flux = 3.0 * float(flux_err)
        upper_mag = float(zeropoint) - 2.5 * math.log10(upper_flux)
        upper_abs_mag = upper_mag - distance_modulus if distance_modulus is not None else None
        updated = dict(measurement)
        updated["is_upper_limit"] = True
        updated["upper_limit_mag"] = upper_mag
        updated["mag_text"] = format_mag_value(None, None, True, upper_mag)
        updated["abs_mag_text"] = format_mag_value(None, None, True, upper_abs_mag)
        return updated
    return None


def select_measurement_for_adoption(
    survey: str,
    measurement: dict[str, object],
    distance_modulus: float | None,
) -> dict[str, object] | None:
    if not survey_has_measurement(measurement):
        return None
    if survey_meets_adoption_rule(survey, measurement):
        return measurement
    return with_upper_limit_fallback(measurement, distance_modulus)


def format_color(row: pd.Series, adopted: dict[str, object]) -> str:
    uv = to_float(row.get(UV_MAG))
    r = adopted.get("mag")
    uv_ul = to_bool(row.get(UV_IS_UL))
    r_ul = bool(adopted.get("is_upper_limit"))
    uv_ul_mag = to_float(row.get(UV_MAG_UL))
    r_ul_mag = adopted.get("upper_limit_mag")

    if uv is not None and r is not None and not uv_ul and not r_ul:
        return f"{uv - r:.3f}"
    if uv_ul_mag is not None and r is not None and uv_ul and not r_ul:
        return f"$<{uv_ul_mag - r:.3f}$"
    if uv is not None and r_ul_mag is not None and not uv_ul and r_ul:
        return f"$>{uv - r_ul_mag:.3f}$"
    if uv_ul_mag is not None and r_ul_mag is not None and uv_ul and r_ul:
        return f"$?{uv_ul_mag - r_ul_mag:.3f}$"
    return ""


def choose_adopted_best(adopted: dict[str, object], panstarrs: dict[str, object]) -> dict[str, object]:
    adopted_survey = str(adopted.get("survey") or "").strip()
    if adopted_survey not in {"LegacySurvey", "DES"}:
        return adopted
    if not survey_has_measurement(panstarrs):
        return adopted

    adopted_is_ul = bool(adopted.get("is_upper_limit"))
    pan_is_ul = bool(panstarrs.get("is_upper_limit"))

    # If the current adopted value is only an upper limit but Pan-STARRS has
    # a detection, prefer the Pan-STARRS measurement.
    if adopted_is_ul and not pan_is_ul:
        return panstarrs

    adopted_err = adopted.get("mag_err")
    pan_err = panstarrs.get("mag_err")
    if (
        not adopted_is_ul
        and not pan_is_ul
        and isinstance(adopted_err, (int, float))
        and math.isfinite(float(adopted_err))
        and isinstance(pan_err, (int, float))
        and math.isfinite(float(pan_err))
        and float(pan_err) < float(adopted_err)
    ):
        return panstarrs

    return adopted


def main() -> int:
    args = parse_args()
    summary = pd.read_csv(args.summary).drop_duplicates(subset=["Supernova"], keep="last")
    images_root = Path(args.images_root)
    adopted_overrides = load_adopted_overrides(args.adopted_overrides)
    legacy_alt_measurements = load_legacy_alt_measurements(args.legacy_alt_csv)

    rows = []
    for _, row in summary.iterrows():
        dm = infer_distance_modulus(row)
        survey_measurements = {
            survey: build_survey_measurement(row, images_root, survey, dm)
            for survey in SURVEY_ORDER
        }
        legacy_alt = legacy_alt_measurements.get(str(row["Supernova"]), None)
        legacy_alt_measurement = {
            "survey": "LegacySurvey",
            "mag": None,
            "mag_err": None,
            "is_upper_limit": False,
            "upper_limit_mag": None,
            "mag_text": "",
            "abs_mag_text": "",
            "flux": None,
            "flux_err": None,
            "snr": None,
            "zeropoint": 22.5,
        }
        if legacy_alt is not None:
            abs_mag = legacy_alt["mag"] - dm if (dm is not None and legacy_alt.get("mag") is not None) else None
            legacy_alt_measurement = {
                "survey": "LegacySurvey",
                "mag": legacy_alt.get("mag"),
                "mag_err": legacy_alt.get("mag_err"),
                "is_upper_limit": False,
                "upper_limit_mag": None,
                "mag_text": format_mag_value(legacy_alt.get("mag"), legacy_alt.get("mag_err"), False, None),
                "abs_mag_text": format_mag_value(abs_mag, legacy_alt.get("mag_err"), False, None),
                "flux": legacy_alt.get("flux"),
                "flux_err": legacy_alt.get("flux_err"),
                "snr": legacy_alt.get("snr"),
                "zeropoint": legacy_alt.get("zeropoint"),
            }
        legacy_alt_display = legacy_alt_measurement
        if legacy_alt is not None and not legacy_alt_measurement["mag_text"]:
            legacy_alt_fallback = with_upper_limit_fallback(legacy_alt_measurement, dm)
            if legacy_alt_fallback is not None:
                legacy_alt_display = legacy_alt_fallback

        adoption_measurements = dict(survey_measurements)
        if survey_has_measurement(legacy_alt_measurement):
            adoption_measurements["LegacySurvey"] = legacy_alt_measurement

        override_survey = adopted_overrides.get(str(row["Supernova"]), "")
        preferred_survey = infer_adopted_survey_name(row)
        adopted = {"survey": "", "mag_text": "", "abs_mag_text": "", "flux": None, "flux_err": None}
        if override_survey in adoption_measurements:
            override_measurement = select_measurement_for_adoption(
                override_survey,
                adoption_measurements[override_survey],
                dm,
            )
            if override_measurement is not None:
                adopted = override_measurement
        elif preferred_survey in adoption_measurements and survey_meets_adoption_rule(preferred_survey, adoption_measurements[preferred_survey]):
            adopted = adoption_measurements[preferred_survey]
        else:
            for survey in SURVEY_ORDER:
                measurement = adoption_measurements[survey]
                if survey_meets_adoption_rule(survey, measurement):
                    adopted = measurement
                    break
        if not survey_has_measurement(adopted):
            for survey in SURVEY_ORDER:
                measurement = adoption_measurements[survey]
                fallback = with_upper_limit_fallback(measurement, dm)
                if fallback is not None:
                    adopted = fallback
                    break
        adopted_survey_name = str(adopted.get("survey") or "").strip()
        if not adopted_survey_name and override_survey:
            adopted_survey_name = override_survey
        adopted_best = choose_adopted_best(adopted, survey_measurements["PanSTARRS"])
        adopted_best_survey_name = str(adopted_best.get("survey") or "").strip()

        rows.append(
            {
                "SN name": row["Supernova"],
                "UV mag": format_mag_value(
                    to_float(row.get(UV_MAG)),
                    to_float(row.get(UV_MAG_ERR)),
                    to_bool(row.get(UV_IS_UL)),
                    to_float(row.get(UV_MAG_UL)),
                ),
                "UV abs mag": format_mag_value(
                    to_float(row.get(UV_MAG_ABS)),
                    to_float(row.get(UV_MAG_ERR)),
                    to_bool(row.get(UV_IS_UL)),
                    to_float(row.get(UV_MAG_UL_ABS)),
                ),
                "F275W-r": format_color(row, adopted),
                "Adopted r survey": adopted_survey_name,
                "Adopted r mag": adopted["mag_text"],
                "Adopted r abs mag": adopted["abs_mag_text"],
                "Adopted r flux": adopted["flux"],
                "Adopted r flux err": adopted["flux_err"],
                "F275W-r Best": format_color(row, adopted_best),
                "Adopted Best survey": adopted_best_survey_name,
                "Adopted Best": adopted_best["mag_text"],
                "Adopted Best abs mag": adopted_best["abs_mag_text"],
                "Adopted Best flux": adopted_best["flux"],
                "Adopted Best flux err": adopted_best["flux_err"],
                "LegacySurvey r mag": survey_measurements["LegacySurvey"]["mag_text"],
                "LegacySurvey r abs mag": survey_measurements["LegacySurvey"]["abs_mag_text"],
                "LegacySurvey r flux": survey_measurements["LegacySurvey"]["flux"],
                "LegacySurvey r flux err": survey_measurements["LegacySurvey"]["flux_err"],
                "LegacySurvey invvar+bkg r mag": legacy_alt_display["mag_text"],
                "LegacySurvey invvar+bkg r abs mag": legacy_alt_display["abs_mag_text"],
                "LegacySurvey invvar+bkg r flux": legacy_alt_display["flux"],
                "LegacySurvey invvar+bkg r flux err": legacy_alt_display["flux_err"],
                "DES r mag": survey_measurements["DES"]["mag_text"],
                "DES r abs mag": survey_measurements["DES"]["abs_mag_text"],
                "DES r flux": survey_measurements["DES"]["flux"],
                "DES r flux err": survey_measurements["DES"]["flux_err"],
                "PanSTARRS r mag": survey_measurements["PanSTARRS"]["mag_text"],
                "PanSTARRS r abs mag": survey_measurements["PanSTARRS"]["abs_mag_text"],
                "PanSTARRS r flux": survey_measurements["PanSTARRS"]["flux"],
                "PanSTARRS r flux err": survey_measurements["PanSTARRS"]["flux_err"],
                "SkyMapper r mag": survey_measurements["SkyMapper"]["mag_text"],
                "SkyMapper r abs mag": survey_measurements["SkyMapper"]["abs_mag_text"],
                "SkyMapper r flux": survey_measurements["SkyMapper"]["flux"],
                "SkyMapper r flux err": survey_measurements["SkyMapper"]["flux_err"],
            }
        )

    paper = pd.DataFrame(rows)
    paper.to_csv(args.paper_full, index=False)
    paper.head(18).copy().to_csv(args.paper_short, index=False)
    print(f"Updated {args.paper_full}")
    print(f"Updated {args.paper_short}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
