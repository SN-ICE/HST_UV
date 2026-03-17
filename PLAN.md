# HST_UV Working Plan

This file tracks the current agreed project state and the remaining analysis work.

## Current State

- [x] Final active sample defined: `340` SN with IFS coverage (`list_snlist.txt` crossmatch applied).
- [x] HST + optical photometry products consolidated for the active sample.
- [x] Final paper-style summary tables rebuilt from the active sample:
  - `outputs/sn_mag_summary_1kpc_full.csv`
  - `outputs/sn_mag_summary_1kpc.csv`
- [x] Parallel optical-survey measurements retained in the summary table:
  - `LegacySurvey`
  - `DES`
  - `PanSTARRS`
  - `SkyMapper`
- [x] Legacy Survey alternative photometry/error test completed:
  - original `hostphot` Legacy values kept
  - new `LegacySurvey invvar+bkg` columns added to the paper summary table
- [x] Adopted `r`-band values rebuilt using the current survey-priority / SNR logic.
- [x] `Adopted Best` columns rebuilt comparing Pan-STARRS against the new Legacy `invvar+bkg` values.
- [x] Diagnostic HST-left / optical-right plots regenerated and reviewed.
- [x] Paper candidate set defined and reduced to `20` objects.
- [x] Overleaf tables and figures synced to the current local state.

## Known Caveats

- [ ] Keep track of the `19` Legacy standalone `invvar` failures (`no valid invvar pixels in aperture`) in case they matter for later interpretation.
- [ ] Keep track of the UV-missing / unusable footprint cases already noted in the manuscript text.
- [ ] The Overleaf repo still has unrelated local leftovers that were intentionally not pushed:
  - untracked `.DS_Store`
  - unstaged deletion of `figure.eps`

## Next Phase

- [ ] Run the SNooPy dual-model fitting batch for the active sample.
- [ ] Review SNooPy fit success/failure rates and build the cleaned fit summary table.
- [ ] Convert the SNooPy light curves / fit products into the format needed for `sncosmo`.
- [ ] Run the `sncosmo` fits and build the SALT-parameter table.
- [ ] Combine photometry, SNooPy, and `sncosmo` outputs into one master analysis table.
- [ ] Compute distance moduli and Hubble residuals.
- [ ] Start the paper-plot / notebook analysis for the final science figures.

## Later Phase

- [ ] Obtain Halpha fluxes and SFR from the MUSE IFS Halpha maps.

Constraint:
- Do not start the Halpha / MUSE step until the light-curve fitting and combined-table work are complete.
