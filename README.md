# HST UV Host Photometry Pipeline

![Status](https://img.shields.io/badge/status-active%20research-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-informational)
![Band](https://img.shields.io/badge/primary%20band-HST%20F275W-orange)
![Data policy](https://img.shields.io/badge/FITS-local%20only-lightgrey)

End-to-end local pipeline for host-galaxy photometry at SN positions, based on HST/WFC3 UVIS F275W with optional companion r-band photometry from ground-based surveys.

## Table of Contents

- [Highlights](#highlights)
- [Quick Start](#quick-start)
- [Workflow](#workflow)
- [Repository Layout](#repository-layout)
- [Input Tables](#input-tables)
- [Commands](#commands)
- [Outputs](#outputs)
- [Main Scripts](#main-scripts)
- [Citation](#citation)
- [Notes](#notes)

## Highlights

- Validates FITS + celestial WCS coverage for each target position.
- Measures local-aperture photometry at 1, 2, and 3 kpc.
- Selects exactly one r-band survey using a priority list (`DES`, `LegacySurvey`, `PanSTARRS`, `SkyMapper` by default).
- Computes S/N and explicit upper-limit columns from a configurable threshold.
- Produces publication-friendly diagnostic panels and structured CSV outputs.
- Supports resume mode via `outputs/results.pkl`.

## Quick Start

```bash
python -m pip install numpy pandas astropy matplotlib aplpy reproject hostphot
python scripts/hst_uv_photometry_pipeline.py \
  --input-csv data/supernovas_input.csv \
  --mapping-csv data/fits_to_sn_mapping.csv
```

Example plot output:

![Example diagnostic panel](outputs/plots/SN2013da_plot.png)

## Workflow

```mermaid
flowchart TD
    A["Input table (SN, RA, Dec, z)"] --> B["Resolve FITS path<br/>(explicit path or mapping)"]
    B --> C["Validate FITS + WCS + in-frame coordinates"]
    C --> D["HST F275W photometry<br/>(1,2,3 kpc)"]
    D --> E["Optional r-band photometry<br/>(priority survey list)"]
    E --> F["Compute SNR and upper limits"]
    F --> G["Write summary CSV + UL CSV + processing log + pickle"]
    G --> H["Generate HST / HST+r-band diagnostic plots"]
```

## Repository Layout

```text
HST_UV/
├── scripts/
│   ├── hst_uv_photometry_pipeline.py
│   └── tns_crossmatch.py
├── data/
│   ├── supernovas_input.csv
│   ├── fits_to_sn_mapping.csv
│   ├── tns_crossmatch.csv
│   └── tns_public_objects.csv.zip
├── 16741/                       # HST products (Program 16741)
├── 17176/                       # HST products (Program 17176)
├── images/                      # temporary/working image area during runs
├── outputs/
│   ├── images/                  # per-object survey products (moved from images/)
│   ├── plots/                   # diagnostic panel figures
│   ├── photometry_summary_keep.csv
│   ├── sn_flux_and_mag_upperlimits.csv
│   ├── processing_log.csv
│   └── results.pkl
```

## Input Tables

`scripts/hst_uv_photometry_pipeline.py` supports two schemas:

1. Preferred schema in `data/supernovas_input.csv`
- `matched_snname`, `rasn`, `decsn`, `redshift`
- optional FITS columns: `fits_relpath` or `fits_subfolder` + `fits_filename`

2. Legacy schema
- `Supernova`, `RA(H:M:S)`, `DEC(D:M:S)`, `Redshift z`

If a row does not contain an explicit FITS path, the pipeline falls back to `data/fits_to_sn_mapping.csv`.

## Commands

### Optional: rebuild TNS/FITS crossmatch

```bash
python scripts/tns_crossmatch.py \
  --roots 16741 17176 \
  --patterns "*_drc.fits" \
  --tns-csv data/tns_public_objects.csv.zip \
  --output data/tns_crossmatch.csv \
  --only-sne
```

### Main photometry run

```bash
python scripts/hst_uv_photometry_pipeline.py \
  --input-csv data/supernovas_input.csv \
  --mapping-csv data/fits_to_sn_mapping.csv
```

### Useful flags

| Flag | Purpose |
|---|---|
| `--dry-run` | Validate rows/FITS/WCS only; skip photometry outputs |
| `--max-rows N` | Process first `N` rows for tests |
| `--no-plots` | Disable diagnostic plot generation |
| `--disable-rband` | Skip r-band photometry attempts |
| `--rband-surveys ...` | Set r-band survey priority order |
| `--snr-threshold 3.0` | Threshold for upper-limit reporting |

## Outputs

| File | Description |
|---|---|
| `outputs/photometry_summary_keep.csv` | Main result table with measurements, status, S/N, and upper-limit columns |
| `outputs/sn_flux_and_mag_upperlimits.csv` | Upper-limit deliverable table |
| `outputs/processing_log.csv` | Per-target/FITS processing statuses and failure reasons |
| `outputs/results.pkl` | Resume cache of per-FITS result dictionaries |
| `outputs/plots/*.png` | HST-only or HST+r-band diagnostic panels |
| `outputs/images/<object>/<survey>/...` | Per-object photometry artifacts |

## Main Scripts

- `scripts/hst_uv_photometry_pipeline.py`: production pipeline for validation, photometry, S/N and UL derivation, and plotting.
- `scripts/tns_crossmatch.py`: local cone-search crossmatch between FITS footprints and TNS public objects.

## Citation

If this workflow is used in scientific results, cite:

- the repository itself (URL + commit hash used in analysis),
- `hostphot`,
- relevant HST data products and survey data sources used for r-band comparison.

## Notes

- FITS files are intentionally excluded from version control (`*.fits`/`*.fit` in `.gitignore`).
- Keep raw FITS locally under `16741/`, `17176/`, or absolute paths referenced in input/mapping tables.
