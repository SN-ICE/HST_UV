#!/usr/bin/env python3
"""
Request and download ZTF forced photometry for the 14 SN objects that were
successfully populated with ATLAS photometry into SNooPy files.

This script adapts the workflow in:
  https://github.com/joanalnu/ZTF_api

In particular:
- Request step via:
    https://ztfweb.ipac.caltech.edu/cgi-bin/requestForcedPhotometry.cgi
- Download step from lightcurve paths listed in the "All recent jobs" table.

Because ZTF processing is asynchronous, the recommended flow is:
1) run with --mode submit
2) once jobs finish on the ZTF website, export/copy the jobs table to a txt file
3) run with --mode download --jobs-table <that_file>
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from project_paths import DATA_DIR, light_curves_path


REQUEST_BASE = "https://ztfweb.ipac.caltech.edu/cgi-bin/requestForcedPhotometry.cgi"
REQUESTS_TABLE_URL = "https://ztfweb.ipac.caltech.edu/cgi-bin/getForcedPhotometryRequests.cgi"
DOWNLOAD_BASE = "https://ztfweb.ipac.caltech.edu"
DOWNLOAD_AUTH_USER = "ztffps"
DOWNLOAD_AUTH_PASS = "dontgocrazy!"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ZTF forced photometry helper for ATLAS-converted 14 SN objects.")
    p.add_argument("--mode", choices=["submit", "download"], required=True, help="Submit new requests or download finished jobs.")
    p.add_argument(
        "--atlas-report",
        default=str(light_curves_path("atlas_to_snoopy_conversion_report.csv")),
        help="Report used to select the 14 objects (status == ok).",
    )
    p.add_argument(
        "--sample-csv",
        default=str(DATA_DIR / "supernovas_input.csv"),
        help="Sample table with matched_snname, rasn, decsn.",
    )
    p.add_argument(
        "--out-dir",
        default=str(light_curves_path("ZTF", "raw")),
        help="Directory to save downloaded ZTF forced-phot files.",
    )
    p.add_argument(
        "--report-csv",
        default=str(light_curves_path("ZTF", "ztf_forcedphot_report.csv")),
        help="Output report CSV.",
    )
    p.add_argument(
        "--target-list-txt",
        default=str(light_curves_path("ZTF", "ztf_forcedphot_targets_atlas14.txt")),
        help="Output text file with Name JD RA Dec for traceability.",
    )
    p.add_argument("--email", default=None, help="ZTF request email (required for --mode submit).")
    p.add_argument("--userpass", default=None, help="ZTF request user password (required for --mode submit).")
    p.add_argument(
        "--jd-start",
        type=float,
        default=Time("2018-01-01", format="iso", scale="utc").jd,
        help="Start JD for forced-phot requests.",
    )
    p.add_argument(
        "--jd-end",
        type=float,
        default=Time.now().jd + 30.0,
        help="End JD for forced-phot requests.",
    )
    p.add_argument("--sleep-seconds", type=float, default=1.0, help="Delay between request submissions.")
    p.add_argument(
        "--jobs-table",
        default=None,
        help="Path to txt copy of 'All recent jobs' table from ZTF website (required for --mode download).",
    )
    p.add_argument(
        "--coord-match-arcsec",
        type=float,
        default=1.0,
        help="Max separation (arcsec) when matching job rows to target SN by coordinates.",
    )
    p.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout seconds.")
    return p.parse_args()


def parse_ra_dec_to_deg(ra_raw: object, dec_raw: object) -> tuple[float, float]:
    ra_txt = str(ra_raw).strip()
    dec_txt = str(dec_raw).strip()

    ra_num = pd.to_numeric(ra_txt, errors="coerce")
    dec_num = pd.to_numeric(dec_txt, errors="coerce")
    if np.isfinite(ra_num) and np.isfinite(dec_num):
        return float(ra_num), float(dec_num)

    if ":" in ra_txt or "h" in ra_txt.lower():
        c = SkyCoord(ra_txt, dec_txt, unit=(u.hourangle, u.deg), frame="icrs")
    else:
        c = SkyCoord(ra_txt, dec_txt, unit=(u.deg, u.deg), frame="icrs")
    return float(c.ra.deg), float(c.dec.deg)


def build_targets(atlas_report: Path, sample_csv: Path) -> pd.DataFrame:
    rep = pd.read_csv(atlas_report)
    ok_names = rep.loc[rep["status"] == "ok", "sample_sn"].astype(str).str.strip().tolist()
    ok_set = set(ok_names)

    sample = pd.read_csv(sample_csv)
    sample["sample_sn"] = sample["matched_snname"].astype(str).str.strip()
    sample = sample[sample["sample_sn"].isin(ok_set)].copy()
    if len(sample) == 0:
        raise ValueError("No targets selected from atlas report with status==ok")

    sample["ra_deg"], sample["dec_deg"] = zip(*sample.apply(lambda r: parse_ra_dec_to_deg(r["rasn"], r["decsn"]), axis=1))
    sample = sample[["sample_sn", "ra_deg", "dec_deg"]].drop_duplicates("sample_sn")
    sample = sample.sort_values("sample_sn", kind="stable").reset_index(drop=True)
    return sample


def write_target_list_txt(targets: pd.DataFrame, out_txt: Path, jd_ref: float) -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with out_txt.open("w", encoding="utf-8") as f:
        f.write("Name DateObs RA Dec\n")
        for _, r in targets.iterrows():
            f.write(f"{r['sample_sn']} {jd_ref:.5f} {r['ra_deg']:.8f} {r['dec_deg']:.8f}\n")


def submit_requests(args: argparse.Namespace, targets: pd.DataFrame) -> pd.DataFrame:
    if not args.email or not args.userpass:
        raise RuntimeError("--email and --userpass are required for --mode submit")

    rows = []
    with requests.Session() as s:
        for _, r in targets.iterrows():
            sn = r["sample_sn"]
            params = {
                "ra": f"{r['ra_deg']:.8f}",
                "dec": f"{r['dec_deg']:.8f}",
                "jdstart": f"{args.jd_start:.5f}",
                "jdend": f"{args.jd_end:.5f}",
                "email": args.email,
                "userpass": args.userpass,
            }
            url = f"{REQUEST_BASE}?{urlencode(params)}"

            status = "ok"
            message = ""
            http_status = np.nan
            response_text = ""
            try:
                # ZTF request endpoint requires both query credentials (email/userpass)
                # and HTTP basic auth with the service credentials.
                resp = s.get(
                    REQUEST_BASE,
                    params=params,
                    auth=(DOWNLOAD_AUTH_USER, DOWNLOAD_AUTH_PASS),
                    timeout=args.timeout,
                )
                http_status = resp.status_code
                resp.raise_for_status()
                response_text = (resp.text or "").strip().replace("\n", " ")
            except Exception as e:
                status = "failed"
                message = str(e)

            rows.append(
                {
                    "mode": "submit",
                    "sn_name": sn,
                    "ra_deg": r["ra_deg"],
                    "dec_deg": r["dec_deg"],
                    "status": status,
                    "http_status": http_status,
                    "request_url": url,
                    "response_excerpt": response_text[:500],
                    "message": message,
                }
            )

            if args.sleep_seconds > 0:
                import time

                time.sleep(args.sleep_seconds)

    return pd.DataFrame(rows)


def parse_jobs_table(path: Path) -> pd.DataFrame:
    """
    Parse a copy/paste text table from:
    https://ztfweb.ipac.caltech.edu/cgi-bin/getForcedPhotometryRequests.cgi

    Expected columns include:
      reqId ra dec startJD endJD created started ended exitcode lightcurve
    with lightcurve path as last column.
    """
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            if line.lower().startswith("reqid"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 11:
                parts = re.split(r"\s+", line.strip())
            if len(parts) < 11:
                continue
            try:
                rows.append(
                    {
                        "reqid": parts[0],
                        "ra": float(parts[1]),
                        "dec": float(parts[2]),
                        "startjd": float(parts[3]),
                        "endjd": float(parts[4]),
                        "created": parts[5],
                        "started": parts[6],
                        "ended": parts[7],
                        "exitcode": parts[8],
                        "lightcurve_path": parts[10],
                    }
                )
            except Exception:
                continue
    return pd.DataFrame(rows)


def download_results(args: argparse.Namespace, targets: pd.DataFrame) -> pd.DataFrame:
    if not args.jobs_table:
        raise RuntimeError("--jobs-table is required for --mode download")

    jobs = parse_jobs_table(Path(args.jobs_table).expanduser().resolve())
    if len(jobs) == 0:
        raise ValueError(f"No parseable job rows in {args.jobs_table}")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tcoord = SkyCoord(targets["ra_deg"].values * u.deg, targets["dec_deg"].values * u.deg, frame="icrs")
    jcoord = SkyCoord(jobs["ra"].values * u.deg, jobs["dec"].values * u.deg, frame="icrs")

    rows = []
    with requests.Session() as s:
        for jidx, jrow in jobs.iterrows():
            sep = jcoord[jidx].separation(tcoord).arcsec
            k = int(np.argmin(sep))
            best_sep = float(sep[k])
            sn = targets.iloc[k]["sample_sn"]

            status = "ok"
            message = ""
            output_file = ""
            n_lines = 0
            download_url = f"{DOWNLOAD_BASE}{jrow['lightcurve_path']}"

            if best_sep > args.coord_match_arcsec:
                status = "unmatched_coord"
                message = f"nearest target at {best_sep:.3f} arcsec"
            elif not str(jrow["lightcurve_path"]).startswith("/"):
                status = "invalid_lightcurve_path"
                message = f"path={jrow['lightcurve_path']}"
            else:
                try:
                    resp = s.get(download_url, auth=(DOWNLOAD_AUTH_USER, DOWNLOAD_AUTH_PASS), timeout=args.timeout)
                    resp.raise_for_status()
                    text = resp.text
                    out_path = out_dir / f"ZTF_{sn}_forcedphot.txt"
                    out_path.write_text(text)
                    output_file = str(out_path)
                    n_lines = sum(1 for ln in text.splitlines() if ln.strip())
                except Exception as e:
                    status = "failed_download"
                    message = str(e)

            rows.append(
                {
                    "mode": "download",
                    "sn_name": sn,
                    "reqid": jrow["reqid"],
                    "ra_job_deg": jrow["ra"],
                    "dec_job_deg": jrow["dec"],
                    "sep_to_target_arcsec": best_sep,
                    "exitcode": jrow["exitcode"],
                    "lightcurve_path": jrow["lightcurve_path"],
                    "download_url": download_url,
                    "status": status,
                    "message": message,
                    "output_file": output_file,
                    "n_lines": n_lines,
                }
            )

    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()

    atlas_report = Path(args.atlas_report).expanduser().resolve()
    sample_csv = Path(args.sample_csv).expanduser().resolve()
    report_csv = Path(args.report_csv).expanduser().resolve()
    target_list_txt = Path(args.target_list_txt).expanduser().resolve()
    report_csv.parent.mkdir(parents=True, exist_ok=True)

    targets = build_targets(atlas_report, sample_csv)
    write_target_list_txt(targets, target_list_txt, jd_ref=(args.jd_start + args.jd_end) / 2.0)

    if args.mode == "submit":
        out = submit_requests(args, targets)
    else:
        out = download_results(args, targets)

    out.to_csv(report_csv, index=False)

    print(f"Targets in batch: {len(targets)}")
    print(f"Mode: {args.mode}")
    print(f"Requests table URL (for manual status check): {REQUESTS_TABLE_URL}")
    print(f"Target list file: {target_list_txt}")
    print(f"Report: {report_csv}")
    if "status" in out.columns and len(out):
        print("Status counts:")
        print(out["status"].value_counts(dropna=False).to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
