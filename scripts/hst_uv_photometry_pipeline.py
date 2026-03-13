import argparse
import os
import pickle
import shutil
from pathlib import Path
import re
import tempfile

import aplpy
import hostphot
import hostphot.photometry.local_photometry as lp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.cosmology import WMAP9 as cosmo
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
import astropy.units as u
from hostphot.cutouts import set_HST_image, download_images
from hostphot.surveys_utils import get_survey_filters
from reproject import reproject_interp

fits.conf.use_memmap = False

# ------------------------------------------------------------------
# Project-level paths and constants
# ------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
WORK_IMAGES_DIR = PROJECT_ROOT / "images"
OUTPUT_IMAGES_DIR = PROJECT_ROOT / "outputs" / "images"
SURVEY = "HST"
FILTER = "WFC3_UVIS_F275W"
RBAND_DEFAULT_ORDER = ["DES", "LegacySurvey", "PanSTARRS", "SkyMapper"]
PLOT_FONT_FAMILY = "STIXGeneral"
CBF_BLUE = "#0072B2"       # Okabe-Ito blue
CBF_ORANGE = "#E69F00"     # Okabe-Ito orange
CBF_VERMILION = "#D55E00"  # Okabe-Ito vermilion

plt.rcParams["font.family"] = PLOT_FONT_FAMILY


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse CLI options for data locations and runtime behavior."""
    parser = argparse.ArgumentParser(description="Run HST UV host photometry from a prepared input table.")

    parser.add_argument("--input-csv", default=str(PROJECT_ROOT / "data" / "supernovas_input.csv"))
    parser.add_argument("--mapping-csv", default=str(PROJECT_ROOT / "data" / "fits_to_sn_mapping.csv"))

    parser.add_argument("--resume-pickle", default=str(PROJECT_ROOT / "outputs" / "results.pkl"))
    parser.add_argument("--save-pickle", default=str(PROJECT_ROOT / "outputs" / "results.pkl"))

    parser.add_argument("--output-summary", default=str(PROJECT_ROOT / "outputs" / "photometry_summary_keep.csv"))
    parser.add_argument(
        "--output-upper-limits",
        default=str(PROJECT_ROOT / "outputs" / "sn_flux_and_mag_upperlimits.csv"),
    )
    parser.add_argument("--processing-log", default=str(PROJECT_ROOT / "outputs" / "processing_log.csv"))

    parser.add_argument("--max-rows", type=int, default=None, help="Process only first N rows for testing.")
    parser.add_argument("--no-plots", action="store_true", help="Disable custom plot generation.")
    parser.add_argument("--no-resume", action="store_true", help="Do not preload previous results.pkl.")

    parser.add_argument(
        "--ra-format",
        choices=["auto", "deg", "hourangle"],
        default="auto",
        help="Interpretation for numeric RA input values.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate input rows, FITS presence, and WCS only (no photometry).",
    )

    parser.add_argument(
        "--ap-radii-kpc",
        nargs="+",
        type=float,
        default=[1.0, 2.0, 3.0],
        help="Local photometry apertures in kpc.",
    )

    parser.add_argument(
        "--snr-threshold",
        type=float,
        default=3.0,
        help="Threshold below which we report upper limits (S/N <= threshold).",
    )

    parser.add_argument(
        "--rband-surveys",
        nargs="+",
        default=RBAND_DEFAULT_ORDER,
        help="Priority order for selecting exactly one r-band survey.",
    )
    parser.add_argument("--disable-rband", action="store_true", help="Skip r-band photometry attempts.")

    return parser.parse_args()


# ------------------------------------------------------------------
# Generic helpers
# ------------------------------------------------------------------
def normalize_name(name: str) -> str:
    """Standardize object names for robust dictionary lookups."""
    return str(name).strip().lower()


def normalize_apertures(ap_radii_kpc: list[float]) -> list:
    """Convert near-integer aperture radii to int so hostphot keys are stable (_1, _2, _3)."""
    out = []
    for a in ap_radii_kpc:
        if np.isfinite(a) and abs(a - round(a)) < 1e-9:
            out.append(int(round(a)))
        else:
            out.append(float(a))
    return out


def ap_label(ap) -> str:
    """String label used in hostphot output column names."""
    if isinstance(ap, (int, np.integer)):
        return str(int(ap))
    if isinstance(ap, float) and abs(ap - round(ap)) < 1e-9:
        return str(int(round(ap)))
    return str(ap)


def parse_ra_to_deg(value, ra_format: str = "auto") -> float:
    """Parse RA from degrees or hourangle-like strings into decimal degrees."""
    s = str(value).strip()
    sexa_hint = (":" in s) or any(ch in s.lower() for ch in ["h", "m", "s"])

    if ra_format == "hourangle" or (ra_format == "auto" and sexa_hint):
        return SkyCoord(ra=s, dec="0d", unit=(u.hourangle, u.deg), frame="icrs").ra.deg

    # Default behavior for numeric RA in this project is decimal degrees.
    return float(s)


def parse_dec_to_deg(value) -> float:
    """Parse Dec from degrees or sexagesimal-like strings into decimal degrees."""
    s = str(value).strip()
    sexa_hint = (":" in s) or any(ch in s.lower() for ch in ["d", "m", "s"])
    if sexa_hint:
        return SkyCoord(ra="0h", dec=s, unit=(u.hourangle, u.deg), frame="icrs").dec.deg
    try:
        return float(s)
    except Exception:
        return SkyCoord(ra="0h", dec=s, unit=(u.hourangle, u.deg), frame="icrs").dec.deg


def result_to_dict(res) -> dict:
    """Normalize hostphot return types to a flat dictionary."""
    if isinstance(res, dict):
        return dict(res)
    if isinstance(res, pd.Series):
        return res.to_dict()
    if isinstance(res, pd.DataFrame):
        if len(res) == 0:
            return {}
        return res.iloc[0].to_dict()
    return {}


def sanitize_name_for_file(name: str) -> str:
    """Create filesystem-safe filenames for plot outputs."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name).strip())
    return s.strip("_") or "object"


def _finite_float(value) -> float:
    """Convert values to finite float, else NaN."""
    try:
        x = float(value)
    except Exception:
        return np.nan
    return x if np.isfinite(x) else np.nan


def format_1kpc_mag_text(
    row_dict: dict,
    mag_col: str,
    mag_err_col: str,
    flux_col: str,
    flux_err_col: str,
    zp_col: str,
    snr_threshold: float,
) -> str:
    """
    Build compact annotation text for the 1 kpc aperture:
    - detection: m=...+/-...
    - upper limit: m_lim(...sigma)=...
    """
    mag = _finite_float(row_dict.get(mag_col, np.nan))
    mag_err = _finite_float(row_dict.get(mag_err_col, np.nan))
    flux = _finite_float(row_dict.get(flux_col, np.nan))
    flux_err = _finite_float(row_dict.get(flux_err_col, np.nan))
    zp = _finite_float(row_dict.get(zp_col, np.nan))

    if np.isfinite(flux) and np.isfinite(flux_err) and flux_err > 0:
        snr = flux / flux_err
        if np.isfinite(snr) and snr <= snr_threshold:
            flux_ul = snr_threshold * flux_err
            if np.isfinite(zp) and np.isfinite(flux_ul) and flux_ul > 0:
                mag_ul = -2.5 * np.log10(flux_ul) + zp
                return f"1 kpc: m_lim({snr_threshold:g}sigma)={mag_ul:.2f}"
            return f"1 kpc: upper limit ({snr_threshold:g}sigma)"

    if np.isfinite(mag) and np.isfinite(mag_err):
        return f"1 kpc: m={mag:.2f}+/-{mag_err:.2f}"
    if np.isfinite(mag):
        return f"1 kpc: m={mag:.2f}"
    return "1 kpc: n/a"


# ------------------------------------------------------------------
# Input + mapping
# ------------------------------------------------------------------
def load_mapping(mapping_csv: Path) -> dict:
    """
    Load the SN->FITS mapping table generated by tns_crossmatch.py.
    Returns: {normalized_sn_name: [absolute_fits_paths, ...]}
    """
    if not mapping_csv.exists():
        return {}
    df = pd.read_csv(mapping_csv)
    if not {"fits_file", "tns_name"}.issubset(df.columns):
        return {}

    out = {}
    for _, row in df.iterrows():
        name = normalize_name(row["tns_name"])
        fp = str(row["fits_file"]).strip()
        if not fp:
            continue
        p = Path(fp)
        if not p.is_absolute():
            p = (PROJECT_ROOT / p).resolve()
        out.setdefault(name, []).append(str(p))
    return out


def prepare_input_table(input_csv: Path, ra_format: str = "auto") -> pd.DataFrame:
    """
    Normalize supported input schemas into one working table with columns:
    Supernova, RA(DEGREES), DEC(DEGREES), Redshift z, fits_file.
    """
    df = pd.read_csv(input_csv)
    cols = set(df.columns)

    # New schema prepared in this project.
    if {"matched_snname", "rasn", "decsn", "redshift"}.issubset(cols):
        work = pd.DataFrame()

        def pick_name(r):
            for c in ["matched_snname", "cubename", "targname"]:
                if c in cols:
                    v = str(r.get(c, "")).strip()
                    if v:
                        return v
            return ""

        work["Supernova"] = df.apply(pick_name, axis=1)
        work["RA(DEGREES)"] = df["rasn"].apply(lambda v: parse_ra_to_deg(v, ra_format=ra_format))
        work["DEC(DEGREES)"] = df["decsn"].apply(parse_dec_to_deg)
        work["Redshift z"] = pd.to_numeric(df["redshift"], errors="coerce")

        if "fits_relpath" in cols:
            work["fits_file"] = df["fits_relpath"].astype(str).map(
                lambda p: str((PROJECT_ROOT / p).resolve()) if p else ""
            )
        elif "fits_subfolder" in cols and "fits_filename" in cols:
            work["fits_file"] = (
                df["fits_subfolder"].astype(str).str.strip()
                + "/"
                + df["fits_filename"].astype(str).str.strip()
            ).map(lambda p: str((PROJECT_ROOT / p).resolve()))
        else:
            work["fits_file"] = ""

        return work

    # Legacy schema fallback.
    required = {"Supernova", "RA(H:M:S)", "DEC(D:M:S)", "Redshift z"}
    if required.issubset(cols):
        work = pd.DataFrame()
        work["Supernova"] = df["Supernova"].astype(str).str.strip()
        coords = SkyCoord(
            ra=df["RA(H:M:S)"].values,
            dec=df["DEC(D:M:S)"].values,
            unit=(u.hourangle, u.deg),
            frame="icrs",
        )
        work["RA(DEGREES)"] = coords.ra.degree
        work["DEC(DEGREES)"] = coords.dec.degree
        work["Redshift z"] = pd.to_numeric(df["Redshift z"], errors="coerce")
        work["fits_file"] = ""
        return work

    raise ValueError(
        "Input CSV schema not recognized. "
        "Expected either new columns [matched_snname, rasn, decsn, redshift] "
        "or legacy [Supernova, RA(H:M:S), DEC(D:M:S), Redshift z]."
    )


def choose_image_files(row: pd.Series, mapping: dict) -> list:
    """Resolve FITS files for one row, preferring explicit input path over mapping fallback."""
    explicit = str(row.get("fits_file", "")).strip()
    if explicit and Path(explicit).exists():
        return [explicit]

    key = normalize_name(row["Supernova"])
    candidates = mapping.get(key, [])
    drc = [p for p in candidates if p.lower().endswith("_drc.fits")]
    return drc if drc else candidates


# ------------------------------------------------------------------
# FITS/WCS validation and zeropoint utilities
# ------------------------------------------------------------------
def validate_fits_wcs(fits_file: str, ra: float, dec: float) -> tuple[bool, str]:
    """Check that FITS has image data, celestial WCS, and contains the target coordinates."""
    try:
        with fits.open(fits_file) as hdul:
            header = None
            data = None
            for hdu in hdul:
                if hdu.data is not None:
                    data = hdu.data
                    header = hdu.header
                    break
            if header is None or data is None:
                return False, "no_image_hdu"

        wcs = WCS(header)
        if not wcs.is_celestial:
            return False, "non_celestial_wcs"

        x, y = wcs.world_to_pixel_values(ra, dec)
        ny, nx = data.shape
        if not (0 <= x < nx and 0 <= y < ny):
            return False, "coord_out_of_bounds"
        return True, "ok"
    except Exception as e:
        return False, f"wcs_error:{e}"


def get_zeropoint_from_fits(fits_file: str) -> float | None:
    """Fallback zeropoint extraction from FITS headers (ext1 then ext0)."""
    hdr = None
    for ext in (1, 0):
        try:
            hdr = fits.getheader(fits_file, ext)
            break
        except Exception:
            pass
    if hdr is None:
        return None

    if "PHOTZPT" in hdr:
        return float(hdr["PHOTZPT"])
    if "MAGZERO" in hdr:
        return float(hdr["MAGZERO"])
    if "PHOTFLAM" in hdr and "PHOTPLAM" in hdr:
        photflam = float(hdr["PHOTFLAM"])
        photplam = float(hdr["PHOTPLAM"])
        return -2.5 * np.log10(photflam) - 5 * np.log10(photplam) - 2.408
    return None


# ------------------------------------------------------------------
# R-band selection (choose exactly one survey in priority order)
# ------------------------------------------------------------------
def survey_has_r(survey: str) -> bool:
    """Return True if survey supports an r-band filter in hostphot."""
    try:
        filts = get_survey_filters(survey)
    except Exception:
        return False
    if filts is None:
        return False
    if isinstance(filts, str):
        filts = [filts]
    return "r" in [str(f).lower() for f in filts]


def find_rband_fits(name: str, survey: str) -> str:
    """Best-effort discovery of downloaded r-band FITS file for plotting."""
    name = str(name).strip()
    search_dirs = [
        OUTPUT_IMAGES_DIR / name / survey,
        WORK_IMAGES_DIR / name / survey,
    ]
    for survey_dir in search_dirs:
        if not survey_dir.exists():
            continue
        fits_files = sorted(survey_dir.glob("*.fits"))
        if not fits_files:
            continue

        # Prefer files containing '_r' or 'r_' hints.
        preferred = [p for p in fits_files if re.search(r"(^|[_-])r([_-]|\.)", p.stem.lower())]
        target = preferred[0] if preferred else fits_files[0]
        return str(target.resolve())
    return ""


def try_rband_local_photometry(
    name: str,
    ra: float,
    dec: float,
    z: float,
    ap_radii: list,
    save_plots: bool,
    survey_priority: list[str],
) -> tuple[str, str, dict, str, str]:
    """
    Try r-band local photometry in priority order.
    Returns (survey, filter, result_dict, rband_fits_path, error_message).
    If no survey succeeds, returns empty survey/filter/result/path + error message.
    """
    errors = []

    for survey in survey_priority:
        if not survey_has_r(survey):
            errors.append(f"{survey}:no_r_filter")
            continue

        try:
            # Download only r-band for this survey.
            download_images(
                name=name,
                ra=ra,
                dec=dec,
                survey=survey,
                filters=["r"],
                overwrite=True,
                save_input=False,
            )
        except Exception as e:
            errors.append(f"{survey}:download_failed:{e}")
            continue

        try:
            rb = lp.multi_band_phot(
                name,
                ra,
                dec,
                z,
                survey=survey,
                filters=["r"],
                ap_radii=ap_radii,
                ap_units="kpc",
                correct_extinction=True,
                save_plots=save_plots,
                raise_exception=True,
                use_mask=False,
                save_input=False,
            )
            rb_dict = result_to_dict(rb)
            rb_file = find_rband_fits(name, survey)
            return survey, "r", rb_dict, rb_file, ""
        except Exception as e:
            errors.append(f"{survey}:phot_failed:{e}")
            continue

    return "", "", {}, "", " | ".join(errors)[:2000]


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------
def _draw_panel(
    fits_file: str,
    fig,
    subplot,
    ra: float,
    dec: float,
    apertures_deg: list[float],
    panel_title: str,
    panel_text: str = "",
    fov_arcsec: float = 30.0,
):
    """Draw one APLpy panel with circles and consistent styling."""
    with fits.open(fits_file) as hdul:
        data = None
        header = None
        hdu_index = None
        for idx, hdu in enumerate(hdul):
            if hdu.data is not None and np.ndim(hdu.data) >= 2:
                arr = np.squeeze(hdu.data)
                if arr.ndim != 2:
                    continue
                data = arr
                header = hdu.header
                hdu_index = idx
                break
        if data is None or header is None or hdu_index is None:
            raise RuntimeError(f"No image data in {fits_file}")

    wcs = WCS(header)
    if not wcs.is_celestial:
        raise RuntimeError(f"No celestial WCS in {fits_file}")

    x, y = wcs.world_to_pixel_values(ra, dec)
    ny, nx = data.shape
    if not (0 <= x < nx and 0 <= y < ny):
        raise RuntimeError(f"Coordinates out of bounds in {fits_file}")

    f = aplpy.FITSFigure(fits_file, hdu=hdu_index, north=True, figure=fig, subplot=subplot)
    finite = data[np.isfinite(data)]
    vmin = None
    vmax = None
    if finite.size > 0:
        vmin, vmax = np.nanpercentile(finite, [1.0, 99.5])
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
            vmin, vmax = None, None
    f.show_colorscale(cmap="gray_r", stretch="arcsinh", vmin=vmin, vmax=vmax)
    f.axis_labels.hide()
    f.tick_labels.hide()
    f.ticks.hide()

    dx_deg, dy_deg = proj_plane_pixel_scales(wcs)
    pixscale_deg = (dx_deg + dy_deg) / 2.0
    max_radius_deg = (min(nx, ny) * pixscale_deg) / 2.0

    for rdeg in apertures_deg:
        f.show_circles(
            ra,
            dec,
            radius=min(rdeg, max_radius_deg),
            edgecolor=CBF_VERMILION,
            lw=2.8,
        )

    f.recenter(ra, dec, width=fov_arcsec / 3600.0, height=fov_arcsec / 3600.0)
    f.add_scalebar(1 / 3600)
    f.scalebar.set_color(CBF_BLUE)
    f.scalebar.set_linewidth(3.0)
    f.scalebar.set_corner("bottom right")
    f.scalebar.set_label("")
    f.ax.text(
        0.5,
        0.965,
        panel_title,
        transform=f.ax.transAxes,
        color="white",
        fontsize=12.0,
        ha="center",
        va="top",
        fontweight="bold",
        fontfamily=PLOT_FONT_FAMILY,
        bbox={"boxstyle": "round,pad=0.16", "fc": "black", "ec": "none", "alpha": 0.5},
    )
    if panel_text:
        f.ax.text(
            0.03,
            0.88,
            panel_text,
            transform=f.ax.transAxes,
            color="white",
            fontsize=10.8,
            ha="left",
            va="top",
            fontfamily=PLOT_FONT_FAMILY,
            bbox={"boxstyle": "round,pad=0.18", "fc": "black", "ec": "none", "alpha": 0.6},
        )
    f.ax.text(
        0.965,
        0.07,
        "1 arcsec",
        transform=f.ax.transAxes,
        color=CBF_BLUE,
        fontsize=10.6,
        ha="right",
        va="top",
        fontfamily=PLOT_FONT_FAMILY,
        fontweight="semibold",
    )

    return f


def _load_first_2d_hdu(fits_file: str) -> tuple[np.ndarray, fits.Header]:
    """Load first 2D image plane and header from a FITS file."""
    with fits.open(fits_file) as hdul:
        for hdu in hdul:
            if hdu.data is None:
                continue
            arr = np.squeeze(hdu.data)
            if np.ndim(arr) != 2:
                continue
            return np.asarray(arr, dtype=float), hdu.header
    raise RuntimeError(f"No 2D image data found in {fits_file}")


def _write_temp_fits(data: np.ndarray, header: fits.Header) -> str:
    """Write temporary FITS file and return path."""
    with tempfile.NamedTemporaryFile(prefix="plot_align_", suffix=".fits", delete=False) as tf:
        tmp_path = tf.name
    fits.writeto(tmp_path, data, header, overwrite=True)
    return tmp_path


def _make_wcs_aligned_plot_files(
    hst_file: str,
    rband_file: str,
    ra: float,
    dec: float,
    fov_arcsec: float,
) -> tuple[str, str, list[str]]:
    """
    Build temporary FITS files aligned on a common WCS grid.
    Reference grid is a sky cutout from HST centered on (ra, dec).
    """
    temp_files: list[str] = []
    try:
        hst_data, hst_header = _load_first_2d_hdu(hst_file)
        hst_wcs = WCS(hst_header)
        if not hst_wcs.is_celestial:
            raise RuntimeError(f"Non-celestial WCS in {hst_file}")

        center = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
        size = (fov_arcsec * u.arcsec, fov_arcsec * u.arcsec)
        hst_cut = Cutout2D(hst_data, position=center, size=size, wcs=hst_wcs, mode="partial", fill_value=np.nan)
        ref_header = hst_cut.wcs.to_header()
        if "BUNIT" in hst_header:
            ref_header["BUNIT"] = hst_header["BUNIT"]
        hst_tmp = _write_temp_fits(hst_cut.data, ref_header)
        temp_files.append(hst_tmp)

        rb_tmp = ""
        if rband_file and Path(rband_file).exists():
            rb_data, rb_header = _load_first_2d_hdu(rband_file)
            rb_wcs = WCS(rb_header)
            if rb_wcs.is_celestial:
                rb_reproj, _ = reproject_interp(
                    (rb_data, rb_wcs),
                    hst_cut.wcs,
                    shape_out=hst_cut.data.shape,
                )
                rb_header_out = hst_cut.wcs.to_header()
                if "BUNIT" in rb_header:
                    rb_header_out["BUNIT"] = rb_header["BUNIT"]
                rb_tmp = _write_temp_fits(rb_reproj, rb_header_out)
                temp_files.append(rb_tmp)

        return hst_tmp, rb_tmp, temp_files
    except Exception as e:
        print(f"Warning: WCS alignment fallback for plotting ({e})")
        return hst_file, rband_file, temp_files


def plot_supernova_panels(
    hst_file: str,
    ra: float,
    dec: float,
    supernova_name: str,
    apertures_deg: list[float],
    output_name: str,
    rband_file: str = "",
    rband_label: str = "",
    hst_text: str = "",
    rband_text: str = "",
    fov_arcsec: float = 30.0,
):
    """
    Save one-panel (HST only) or two-panel (HST + r-band) diagnostic plot.
    Right panel is added only when a valid r-band FITS exists.
    """
    f1 = None
    f2 = None
    temp_files: list[str] = []
    try:
        out = Path(output_name)
        out.parent.mkdir(parents=True, exist_ok=True)

        plot_hst_file, plot_rband_file, temp_files = _make_wcs_aligned_plot_files(
            hst_file=hst_file,
            rband_file=rband_file,
            ra=ra,
            dec=dec,
            fov_arcsec=fov_arcsec,
        )

        has_r = bool(plot_rband_file) and Path(plot_rband_file).exists()

        if has_r:
            fig = plt.figure(figsize=(9.6, 4.8))
            f1 = _draw_panel(
                fits_file=plot_hst_file,
                fig=fig,
                subplot=[0.05, 0.1, 0.42, 0.82],
                ra=ra,
                dec=dec,
                apertures_deg=apertures_deg,
                panel_title="HST F275W",
                panel_text=hst_text,
                fov_arcsec=fov_arcsec,
            )
            f2 = _draw_panel(
                fits_file=plot_rband_file,
                fig=fig,
                subplot=[0.53, 0.1, 0.42, 0.82],
                ra=ra,
                dec=dec,
                apertures_deg=apertures_deg,
                panel_title=rband_label or "r-band",
                panel_text=rband_text,
                fov_arcsec=fov_arcsec,
            )
            plt.suptitle(supernova_name, fontsize=18, fontfamily=PLOT_FONT_FAMILY, fontweight="semibold")
        else:
            fig = plt.figure(figsize=(4.8, 4.8))
            f1 = _draw_panel(
                fits_file=plot_hst_file,
                fig=fig,
                subplot=[0.08, 0.08, 0.84, 0.84],
                ra=ra,
                dec=dec,
                apertures_deg=apertures_deg,
                panel_title="HST F275W",
                panel_text=hst_text,
                fov_arcsec=fov_arcsec,
            )
            plt.title(supernova_name, fontsize=18, fontfamily=PLOT_FONT_FAMILY, fontweight="semibold")

        plt.savefig(out, dpi=400, bbox_inches="tight")
    except Exception as e:
        print(f"Error plotting {supernova_name}: {e}")
    finally:
        try:
            if f1 is not None:
                f1.close()
        except Exception:
            pass
        try:
            if f2 is not None:
                f2.close()
        except Exception:
            pass
        plt.close("all")
        for tf in temp_files:
            try:
                Path(tf).unlink(missing_ok=True)
            except Exception:
                pass


# ------------------------------------------------------------------
# Output math: S/N and upper limits
# ------------------------------------------------------------------
def _add_band_limits(
    df: pd.DataFrame,
    flux_col: str,
    flux_err_col: str,
    zp_cols: list[str],
    prefix: str,
    snr_threshold: float,
    fits_col: str | None = None,
    mag_col: str | None = None,
    mag_err_col: str | None = None,
    blank_detected_mag_on_ul: bool = True,
) -> pd.DataFrame:
    """
    Add S/N and upper-limit columns for one band/aperture.

    Definitions used:
    - S/N = flux / flux_err
    - Upper-limit condition: flux_err > 0 and S/N <= snr_threshold
    - Flux upper limit: snr_threshold * flux_err
    - Magnitude upper limit: -2.5 log10(flux_upper_limit) + zeropoint

    Zeropoint priority:
    1) First finite value found in zp_cols
    2) Fallback to FITS header zeropoint if fits_col provided
    """
    if flux_col not in df.columns or flux_err_col not in df.columns:
        return df

    flux = pd.to_numeric(df[flux_col], errors="coerce")
    flux_err = pd.to_numeric(df[flux_err_col], errors="coerce")

    with np.errstate(divide="ignore", invalid="ignore"):
        snr = flux / flux_err

    valid_snr = np.isfinite(flux) & np.isfinite(flux_err) & (flux_err > 0)
    snr = snr.where(valid_snr, np.nan)

    is_ul = pd.Series(pd.NA, index=df.index, dtype="boolean")
    is_ul.loc[valid_snr] = snr.loc[valid_snr] <= snr_threshold

    is_ul_bool = is_ul.fillna(False).astype(bool)
    flux_ul = pd.Series(np.nan, index=df.index, dtype=float)
    flux_ul.loc[is_ul_bool] = snr_threshold * flux_err.loc[is_ul_bool]

    df[f"{prefix}_snr"] = snr
    df[f"{prefix}_is_upper_limit"] = is_ul
    df[f"{prefix}_flux_upper_limit"] = flux_ul
    df[f"{prefix}_snr_threshold"] = snr_threshold

    # Clean invalid magnitude values from hostphot (e.g., inf from log10(0)).
    if mag_col and (mag_col in df.columns):
        df[mag_col] = pd.to_numeric(df[mag_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if mag_err_col and (mag_err_col in df.columns):
        df[mag_err_col] = pd.to_numeric(df[mag_err_col], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )

    # If a point is classified as upper limit, suppress the detection magnitude columns.
    if blank_detected_mag_on_ul and mag_col and (mag_col in df.columns):
        df.loc[is_ul_bool, mag_col] = np.nan
        if mag_err_col and (mag_err_col in df.columns):
            df.loc[is_ul_bool, mag_err_col] = np.nan

    mag_ul = []
    for i in range(len(df)):
        if not bool(is_ul_bool.iloc[i]):
            mag_ul.append(np.nan)
            continue

        zp = None
        for zc in zp_cols:
            if zc in df.columns:
                zv = pd.to_numeric(pd.Series([df.iloc[i][zc]]), errors="coerce").iloc[0]
                if np.isfinite(zv):
                    zp = float(zv)
                    break

        if zp is None and fits_col and (fits_col in df.columns):
            fp = str(df.iloc[i][fits_col]).strip()
            if fp:
                zp = get_zeropoint_from_fits(fp)

        ful = pd.to_numeric(pd.Series([flux_ul[i]]), errors="coerce").iloc[0]
        if zp is None or (not np.isfinite(ful)) or ful <= 0:
            mag_ul.append(np.nan)
        else:
            mag_ul.append(-2.5 * np.log10(float(ful)) + float(zp))

    df[f"{prefix}_mag_upper_limit"] = mag_ul
    return df


def add_absolute_magnitude_columns(df: pd.DataFrame, z_col: str = "z") -> pd.DataFrame:
    """
    Convert apparent magnitudes to absolute magnitudes using distance modulus:
        M = m - mu
    where mu is derived from redshift with the configured cosmology.

    Notes:
    - No K-correction is applied.
    - Redshift must be finite and > 0.
    """
    if z_col not in df.columns:
        return df

    z = pd.to_numeric(df[z_col], errors="coerce")
    mu = pd.Series(np.nan, index=df.index, dtype=float)
    valid = np.isfinite(z) & (z > 0)
    if valid.any():
        mu.loc[valid] = [cosmo.distmod(float(zz)).value for zz in z.loc[valid]]

    df["distance_modulus"] = mu

    # Base apparent magnitude columns from hostphot outputs.
    mag_cols = [
        c
        for c in df.columns
        if re.match(rf"^{re.escape(FILTER)}_[^_]+$", c) and (not c.endswith("_zeropoint"))
    ]
    mag_cols += [
        c
        for c in df.columns
        if re.match(r"^rband_r_[^_]+$", c) and (not c.endswith("_zeropoint"))
    ]

    # Upper-limit magnitude columns created in this pipeline.
    mag_cols += [c for c in df.columns if c.endswith("_mag_upper_limit")]

    for col in sorted(set(mag_cols)):
        m = pd.to_numeric(df[col], errors="coerce")
        df[f"{col}_abs"] = m - mu

    return df


def relocate_hostphot_images(name: str) -> tuple[Path, Path] | None:
    """
    Move hostphot-generated per-object folders from <project>/images/<name>
    into <project>/outputs/images/<name> to keep the project root tidy.
    """
    src = WORK_IMAGES_DIR / str(name).strip()
    dst_root = OUTPUT_IMAGES_DIR
    dst_root.mkdir(parents=True, exist_ok=True)
    dst = dst_root / str(name).strip()
    if not src.exists():
        return None
    src_base = src.resolve()
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    shutil.move(str(src), str(dst))
    return src_base, dst.resolve()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main() -> int:
    args = parse_args()

    print(hostphot.__version__)
    print(f"Current Working Directory: {os.getcwd()}")

    input_csv = Path(args.input_csv).expanduser().resolve()
    mapping_csv = Path(args.mapping_csv).expanduser().resolve()
    resume_pickle = Path(args.resume_pickle).expanduser().resolve()
    save_pickle = Path(args.save_pickle).expanduser().resolve()
    output_summary = Path(args.output_summary).expanduser().resolve()
    output_ul = Path(args.output_upper_limits).expanduser().resolve()
    processing_log = Path(args.processing_log).expanduser().resolve()

    save_pickle.parent.mkdir(parents=True, exist_ok=True)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_ul.parent.mkdir(parents=True, exist_ok=True)
    processing_log.parent.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "outputs" / "plots").mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    ap_radii = normalize_apertures(args.ap_radii_kpc)

    data = prepare_input_table(input_csv, ra_format=args.ra_format)
    if args.max_rows is not None:
        data = data.head(args.max_rows).copy()

    mapping = load_mapping(mapping_csv)
    print(f"Loaded {len(data)} input rows")
    print(f"Loaded mapping keys: {len(mapping)}")

    results_dict = {}
    nan_summary_rows = []

    if (not args.no_resume) and (not args.dry_run) and resume_pickle.exists():
        try:
            with open(resume_pickle, "rb") as f:
                previous = pickle.load(f)
            if isinstance(previous, dict):
                results_dict.update(previous)
            print(f"Loaded previous results from {resume_pickle} ({len(results_dict)} entries)")
        except Exception as e:
            print(f"Could not load resume pickle {resume_pickle}: {e}")

    log_rows = []

    for _, row in data.iterrows():
        name = str(row["Supernova"]).strip()
        ra = pd.to_numeric(row["RA(DEGREES)"], errors="coerce")
        dec = pd.to_numeric(row["DEC(DEGREES)"], errors="coerce")
        z = pd.to_numeric(row.get("Redshift z"), errors="coerce")

        if not np.isfinite(ra) or not np.isfinite(dec):
            log_rows.append(
                {
                    "name": name,
                    "fits_file": "",
                    "status": "skipped_bad_coords",
                    "message": "Invalid RA/Dec",
                    "ra_deg": ra,
                    "dec_deg": dec,
                    "redshift": z,
                }
            )
            continue

        ra = float(ra)
        dec = float(dec)

        image_files = choose_image_files(row, mapping)
        if not image_files:
            print(f"No FITS found for {name}")
            log_rows.append(
                {
                    "name": name,
                    "fits_file": "",
                    "status": "no_fits",
                    "message": "No FITS resolved from input row or mapping",
                    "ra_deg": ra,
                    "dec_deg": dec,
                    "redshift": z,
                }
            )
            continue

        for file in image_files:
            if args.dry_run:
                ok, msg = validate_fits_wcs(file, ra, dec)
                log_rows.append(
                    {
                        "name": name,
                        "fits_file": file,
                        "status": "dry_run_ok" if ok else "dry_run_fail",
                        "message": msg,
                        "ra_deg": ra,
                        "dec_deg": dec,
                        "redshift": z,
                    }
                )
                continue

            ok, msg = validate_fits_wcs(file, ra, dec)
            if not ok:
                status = "outside_fov_nan" if msg == "coord_out_of_bounds" else "fits_invalid_nan"
                log_rows.append(
                    {
                        "name": name,
                        "fits_file": file,
                        "status": status,
                        "message": msg,
                        "ra_deg": ra,
                        "dec_deg": dec,
                        "redshift": z,
                    }
                )

                base_nan = {
                    "file": file,
                    "Supernova": name,
                    "RA(degrees)": ra,
                    "DEC(degrees)": dec,
                    "z": z,
                    "status": status,
                    "message": msg,
                    "rband_survey": "",
                    "rband_filter": "",
                    "rband_fits_file": "",
                }
                for ap in ap_radii:
                    lab = ap_label(ap)
                    base_nan[f"{FILTER}_{lab}"] = np.nan
                    base_nan[f"{FILTER}_{lab}_err"] = np.nan
                    base_nan[f"{FILTER}_{lab}_flux"] = np.nan
                    base_nan[f"{FILTER}_{lab}_flux_err"] = np.nan
                nan_summary_rows.append(base_nan)
                continue

            try:
                # HST local photometry in 1,2,3 kpc apertures.
                set_HST_image(file, FILTER, name)
                hst_res = lp.multi_band_phot(
                    name,
                    ra,
                    dec,
                    z,
                    survey=SURVEY,
                    filters=[FILTER],
                    ap_radii=ap_radii,
                    ap_units="kpc",
                    correct_extinction=True,
                    save_plots=not args.no_plots,
                    raise_exception=True,
                    use_mask=False,
                    save_input=False,
                )
                hst_dict = result_to_dict(hst_res)

                combined = dict(hst_dict)
                combined["status"] = "ok"
                combined["message"] = ""
                combined["rband_survey"] = ""
                combined["rband_filter"] = ""
                combined["rband_fits_file"] = ""

                # One r-band survey only, using priority order.
                rband_file = ""
                rband_label = ""
                if not args.disable_rband:
                    rb_survey, rb_filter, rb_dict, rb_fits, rb_err = try_rband_local_photometry(
                        name=name,
                        ra=ra,
                        dec=dec,
                        z=z,
                        ap_radii=ap_radii,
                        save_plots=not args.no_plots,
                        survey_priority=args.rband_surveys,
                    )
                    if rb_dict:
                        combined["rband_survey"] = rb_survey
                        combined["rband_filter"] = rb_filter
                        combined["rband_fits_file"] = rb_fits
                        for k, v in rb_dict.items():
                            combined[f"rband_{k}"] = v
                        rband_file = rb_fits
                        rband_label = f"{rb_survey} r"
                    else:
                        combined["rband_error"] = rb_err

                # Compact 1 kpc annotations for publication-friendly panels.
                hst_text = format_1kpc_mag_text(
                    row_dict=combined,
                    mag_col=f"{FILTER}_1",
                    mag_err_col=f"{FILTER}_1_err",
                    flux_col=f"{FILTER}_1_flux",
                    flux_err_col=f"{FILTER}_1_flux_err",
                    zp_col=f"{FILTER}_zeropoint",
                    snr_threshold=float(args.snr_threshold),
                )
                rband_text = format_1kpc_mag_text(
                    row_dict=combined,
                    mag_col="rband_r_1",
                    mag_err_col="rband_r_1_err",
                    flux_col="rband_r_1_flux",
                    flux_err_col="rband_r_1_flux_err",
                    zp_col="rband_r_zeropoint",
                    snr_threshold=float(args.snr_threshold),
                )

                results_dict[file] = combined

                # Diagnostic figure: left HST panel + right r-band panel (if available).
                if (not args.no_plots) and np.isfinite(z) and z > 0:
                    da = cosmo.angular_diameter_distance(float(z)).to(u.kpc)
                    aperture_1kpc_deg = [(((1.0 * u.kpc) / da) * u.rad).to(u.deg).value]
                    safe_name = sanitize_name_for_file(name)
                    plot_supernova_panels(
                        hst_file=file,
                        ra=ra,
                        dec=dec,
                        supernova_name=name,
                        apertures_deg=aperture_1kpc_deg,
                        output_name=str(PROJECT_ROOT / "outputs" / "plots" / f"{safe_name}_plot.png"),
                        rband_file=rband_file,
                        rband_label=rband_label,
                        hst_text=hst_text,
                        rband_text=rband_text,
                    )

                log_rows.append(
                    {
                        "name": name,
                        "fits_file": file,
                        "status": "ok",
                        "message": "",
                        "ra_deg": ra,
                        "dec_deg": dec,
                        "redshift": z,
                        "rband_survey": combined.get("rband_survey", ""),
                    }
                )
            except Exception as e:
                print(f"Error processing {file} for {name}: {e}")
                log_rows.append(
                    {
                        "name": name,
                        "fits_file": file,
                        "status": "failed",
                        "message": str(e),
                        "ra_deg": ra,
                        "dec_deg": dec,
                        "redshift": z,
                    }
                )

                base_nan = {
                    "file": file,
                    "Supernova": name,
                    "RA(degrees)": ra,
                    "DEC(degrees)": dec,
                    "z": z,
                    "status": "failed_nan",
                    "message": str(e),
                    "rband_survey": "",
                    "rband_filter": "",
                    "rband_fits_file": "",
                }
                for ap in ap_radii:
                    lab = ap_label(ap)
                    base_nan[f"{FILTER}_{lab}"] = np.nan
                    base_nan[f"{FILTER}_{lab}_err"] = np.nan
                    base_nan[f"{FILTER}_{lab}_flux"] = np.nan
                    base_nan[f"{FILTER}_{lab}_flux_err"] = np.nan
                nan_summary_rows.append(base_nan)
            finally:
                moved = relocate_hostphot_images(name)
                # Keep stored r-band paths consistent after moving hostphot folders.
                if moved and (file in results_dict):
                    src_base, dst_base = moved
                    rb_file = str(results_dict[file].get("rband_fits_file", "")).strip()
                    if rb_file and rb_file.startswith(str(src_base)):
                        results_dict[file]["rband_fits_file"] = rb_file.replace(
                            str(src_base), str(dst_base), 1
                        )

    pd.DataFrame(log_rows).to_csv(processing_log, index=False)
    print(f"Wrote processing log: {processing_log} ({len(log_rows)} rows)")

    if args.dry_run:
        print("Dry-run completed. No photometry outputs were generated.")
        return 0

    with open(save_pickle, "wb") as f:
        pickle.dump(results_dict, f)
    print(f"Saved results pickle: {save_pickle} ({len(results_dict)} entries)")

    # Build table from successful photometry results.
    rows = []
    for fits_file, res in results_dict.items():
        row = {"file": fits_file}
        row.update(result_to_dict(res))
        rows.append(row)

    if (not rows) and (not nan_summary_rows):
        print("No photometry results produced.")
        return 0

    frames = []
    if rows:
        frames.append(pd.DataFrame(rows))
    if nan_summary_rows:
        frames.append(pd.DataFrame(nan_summary_rows))

    df_all = pd.concat(frames, ignore_index=True)
    # Prefer successful rows if duplicate FITS keys exist.
    df_all = df_all.drop_duplicates(subset=["file"], keep="first")

    # Normalize base columns.
    if "Supernova" not in df_all.columns and "name" in df_all.columns:
        df_all["Supernova"] = df_all["name"]
    if "RA(degrees)" not in df_all.columns and "ra" in df_all.columns:
        df_all["RA(degrees)"] = df_all["ra"]
    if "DEC(degrees)" not in df_all.columns and "dec" in df_all.columns:
        df_all["DEC(degrees)"] = df_all["dec"]
    if "z" not in df_all.columns and "redshift" in df_all.columns:
        df_all["z"] = df_all["redshift"]
    if "status" not in df_all.columns:
        df_all["status"] = "ok"
    if "message" not in df_all.columns:
        df_all["message"] = ""

    # Keep human-friendly first columns, then all measurement columns.
    first_cols = [
        "file",
        "Supernova",
        "RA(degrees)",
        "DEC(degrees)",
        "z",
        "status",
        "message",
        "rband_survey",
        "rband_filter",
        "rband_fits_file",
    ]
    first_cols = [c for c in first_cols if c in df_all.columns]
    other_cols = [c for c in df_all.columns if c not in first_cols]
    df_summary = df_all[first_cols + other_cols].copy()

    # Add explicit S/N and upper-limit columns for HST and selected r-band.
    for ap in ap_radii:
        lab = ap_label(ap)

        # HST band limits.
        df_summary = _add_band_limits(
            df=df_summary,
            flux_col=f"{FILTER}_{lab}_flux",
            flux_err_col=f"{FILTER}_{lab}_flux_err",
            zp_cols=[f"{FILTER}_zeropoint"],
            prefix=f"hst_{lab}",
            snr_threshold=float(args.snr_threshold),
            fits_col="file",
            mag_col=f"{FILTER}_{lab}",
            mag_err_col=f"{FILTER}_{lab}_err",
        )

        # r-band limits (if r-band columns exist).
        df_summary = _add_band_limits(
            df=df_summary,
            flux_col=f"rband_r_{lab}_flux",
            flux_err_col=f"rband_r_{lab}_flux_err",
            zp_cols=["rband_r_zeropoint"],
            prefix=f"rband_{lab}",
            snr_threshold=float(args.snr_threshold),
            fits_col=None,
            mag_col=f"rband_r_{lab}",
            mag_err_col=f"rband_r_{lab}_err",
        )

    # Keep types readable.
    for c in ["RA(degrees)", "DEC(degrees)", "z"]:
        if c in df_summary.columns:
            df_summary[c] = pd.to_numeric(df_summary[c], errors="coerce")

    # Add absolute magnitudes from apparent magnitudes and redshift distance modulus.
    df_summary = add_absolute_magnitude_columns(df_summary, z_col="z")

    df_summary.to_csv(output_summary, index=False, float_format="%.6g")
    print(f"Wrote summary: {output_summary} ({len(df_summary)} rows)")

    # Upper-limits file includes the same rows with explicit UL columns.
    df_summary.to_csv(output_ul, index=False, float_format="%.6g")
    print(f"Wrote upper-limits table: {output_ul} ({len(df_summary)} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
