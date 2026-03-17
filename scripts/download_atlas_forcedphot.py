#!/usr/bin/env python3
"""
Download ATLAS forced-photometry light curves for SN targets.

API flow follows the official ATLAS forced-phot guide:
https://fallingstar-data.com/forcedphot/apiguide/

Typical usage:
  export ATLAS_USERNAME="your_user"
  export ATLAS_PASSWORD="your_password"
  python scripts/download_atlas_forcedphot.py

Or with an existing token:
  export ATLAS_TOKEN="your_token"
  python scripts/download_atlas_forcedphot.py --token "$ATLAS_TOKEN"
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
from astropy import units as u
from astropy.coordinates import SkyCoord

from project_paths import DATA_DIR, light_curves_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download ATLAS forced photometry for SN targets.")
    parser.add_argument(
        "--sample-csv",
        default=str(DATA_DIR / "supernovas_input.csv"),
        help="Input sample table containing matched_snname, rasn, decsn.",
    )
    parser.add_argument(
        "--missing-csv",
        default=str(light_curves_path("snoopy_file_missing.csv")),
        help="Missing-table path; used when --selection missing.",
    )
    parser.add_argument(
        "--selection",
        choices=["missing", "all"],
        default="missing",
        help="Use only currently missing SNs or all sample SNs.",
    )
    parser.add_argument("--start-year", type=int, default=2014, help="Minimum SN name year (inclusive).")
    parser.add_argument("--end-year", type=int, default=2026, help="Maximum SN name year (inclusive).")
    parser.add_argument(
        "--out-dir",
        default=str(light_curves_path("ATLAS", "raw")),
        help="Output directory for downloaded ATLAS files.",
    )
    parser.add_argument(
        "--report-csv",
        default=str(light_curves_path("ATLAS", "atlas_download_report.csv")),
        help="Output report CSV.",
    )
    parser.add_argument(
        "--base-url",
        default="https://fallingstar-data.com/forcedphot",
        help="ATLAS forced-phot base URL.",
    )
    parser.add_argument("--mjd-min", type=float, default=None, help="Optional minimum MJD for query.")
    parser.add_argument("--mjd-max", type=float, default=None, help="Optional maximum MJD for query.")
    parser.add_argument(
        "--send-email",
        action="store_true",
        help="Request ATLAS email notification from the server.",
    )
    parser.add_argument("--token", default=None, help="ATLAS API token. If absent, use username/password.")
    parser.add_argument("--username", default=None, help="ATLAS username (fallback: ATLAS_USERNAME env var).")
    parser.add_argument("--password", default=None, help="ATLAS password (fallback: ATLAS_PASSWORD env var).")
    parser.add_argument("--poll-seconds", type=float, default=10.0, help="Polling cadence for queued jobs.")
    parser.add_argument("--request-timeout", type=float, default=120.0, help="HTTP timeout in seconds.")
    parser.add_argument("--max-retries", type=int, default=8, help="Retries for transient HTTP errors.")
    parser.add_argument("--resume", action="store_true", help="Skip targets with existing output files.")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N filtered targets.")
    return parser.parse_args()


def extract_year(name: str) -> Optional[int]:
    """Extract discovery-style year from common SN naming patterns."""
    s = str(name).strip()

    # SN2018abc, AT2021xyz, ZTF21..., etc. with explicit 4-digit year.
    m = re.search(r"(19|20)\d{2}", s)
    if m:
        return int(m.group(0))

    # ASASSN-14xx / ASAS14xx / PTF13xx / iPTF14xx style.
    m = re.search(r"(?i)\b(?:ASASSN|ASAS|IPTF|PTF)[-_]?(\d{2})[a-z]", s)
    if m:
        yy = int(m.group(1))
        return 2000 + yy

    return None


def parse_ra_dec_to_deg(ra_raw: object, dec_raw: object) -> tuple[float, float]:
    """Parse RA/Dec in decimal degrees or sexagesimal."""
    ra_txt = str(ra_raw).strip()
    dec_txt = str(dec_raw).strip()

    ra_num = pd.to_numeric(ra_txt, errors="coerce")
    dec_num = pd.to_numeric(dec_txt, errors="coerce")
    if np.isfinite(ra_num) and np.isfinite(dec_num):
        return float(ra_num), float(dec_num)

    if ":" in ra_txt or "h" in ra_txt.lower():
        coord = SkyCoord(ra_txt, dec_txt, unit=(u.hourangle, u.deg), frame="icrs")
    else:
        coord = SkyCoord(ra_txt, dec_txt, unit=(u.deg, u.deg), frame="icrs")
    return float(coord.ra.deg), float(coord.dec.deg)


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._+-]+", "_", str(name).strip()).strip("_") or "SN"


def get_token(base_url: str, username: str, password: str, timeout: float) -> str:
    auth_url = f"{base_url.rstrip('/')}/api-token-auth/"
    resp = requests.post(auth_url, data={"username": username, "password": password}, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    token = data.get("token")
    if not token:
        raise RuntimeError(f"No token returned by {auth_url}")
    return token


def backoff_sleep(attempt: int) -> None:
    delay = min(2 ** attempt, 60)
    time.sleep(delay)


def submit_job(
    session: requests.Session,
    queue_url: str,
    payload: dict,
    timeout: float,
    max_retries: int,
) -> str:
    for attempt in range(max_retries + 1):
        resp = session.post(queue_url, data=payload, timeout=timeout)
        if resp.status_code == 201:
            data = resp.json()
            task_url = data.get("url")
            if not task_url:
                raise RuntimeError(f"Queue response missing task URL: {data}")
            return task_url

        if resp.status_code == 429:
            wait_seconds = 30
            retry_after = resp.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                wait_seconds = int(retry_after)
            else:
                m = re.search(r"(\d+)\s*seconds", resp.text or "", flags=re.IGNORECASE)
                if m:
                    wait_seconds = int(m.group(1))
            time.sleep(max(wait_seconds, 1))
            continue

        if resp.status_code in (500, 502, 503, 504):
            if attempt < max_retries:
                backoff_sleep(attempt)
                continue
        resp.raise_for_status()

    raise RuntimeError("Exceeded maximum retries while submitting job.")


def poll_until_complete(
    session: requests.Session,
    task_url: str,
    timeout: float,
    poll_seconds: float,
    max_retries: int,
) -> dict:
    transient_errors = 0
    while True:
        try:
            resp = session.get(task_url, timeout=timeout)
            resp.raise_for_status()
            task = resp.json()
        except requests.RequestException:
            transient_errors += 1
            if transient_errors > max_retries:
                raise
            backoff_sleep(transient_errors)
            continue

        transient_errors = 0
        fin = task.get("finishtimestamp")
        if fin:
            return task
        time.sleep(max(poll_seconds, 1.0))


def build_target_table(sample_csv: Path, missing_csv: Path, selection: str) -> pd.DataFrame:
    sample = pd.read_csv(sample_csv)
    required = {"matched_snname", "rasn", "decsn"}
    if not required.issubset(sample.columns):
        raise ValueError(f"{sample_csv} must include columns: {sorted(required)}")
    sample["matched_snname"] = sample["matched_snname"].astype(str).str.strip()

    if selection == "all":
        return sample.copy()

    missing = pd.read_csv(missing_csv)
    if "sample_sn" not in missing.columns:
        raise ValueError(f"{missing_csv} must include column: sample_sn")
    missing_names = set(missing["sample_sn"].astype(str).str.strip())
    return sample[sample["matched_snname"].isin(missing_names)].copy()


def main() -> int:
    args = parse_args()

    sample_csv = Path(args.sample_csv).expanduser().resolve()
    missing_csv = Path(args.missing_csv).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    report_csv = Path(args.report_csv).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_csv.parent.mkdir(parents=True, exist_ok=True)

    targets = build_target_table(sample_csv, missing_csv, args.selection)
    targets["year_from_name"] = targets["matched_snname"].apply(extract_year)
    targets = targets[
        targets["year_from_name"].notna()
        & (targets["year_from_name"] >= args.start_year)
        & (targets["year_from_name"] <= args.end_year)
    ].copy()
    targets = targets.sort_values("matched_snname", kind="stable").reset_index(drop=True)
    if args.limit is not None:
        targets = targets.head(args.limit).copy()

    if len(targets) == 0:
        print("No targets matched the requested year range/selection.")
        return 0

    username = args.username or os.getenv("ATLAS_USERNAME")
    password = args.password or os.getenv("ATLAS_PASSWORD")

    token = args.token
    if token is None:
        token = os.getenv("ATLAS_TOKEN")

    if not token:
        if not username or not password:
            raise RuntimeError(
                "Provide --token (or ATLAS_TOKEN), or --username/--password (or ATLAS_USERNAME/ATLAS_PASSWORD)."
            )
        token = get_token(args.base_url, username, password, timeout=args.request_timeout)

    session = requests.Session()
    session.headers.update({"Authorization": f"Token {token}", "Accept": "application/json"})
    queue_url = f"{args.base_url.rstrip('/')}/queue/"

    report_rows = []

    for _, row in targets.iterrows():
        sn = str(row["matched_snname"]).strip()
        output_file = out_dir / f"{safe_name(sn)}_atlas_forcedphot.txt"

        status = "ok"
        message = ""
        task_url = ""
        result_url = ""
        n_data_lines = 0

        try:
            if args.resume and output_file.exists():
                status = "skipped_existing"
                report_rows.append(
                    {
                        "sn_name": sn,
                        "year_from_name": int(row["year_from_name"]),
                        "ra_deg": np.nan,
                        "dec_deg": np.nan,
                        "status": status,
                        "message": "",
                        "task_url": "",
                        "result_url": "",
                        "output_file": str(output_file),
                        "n_data_lines": np.nan,
                    }
                )
                continue

            ra_deg, dec_deg = parse_ra_dec_to_deg(row["rasn"], row["decsn"])
            payload = {
                "ra": f"{ra_deg:.8f}",
                "dec": f"{dec_deg:.8f}",
                "send_email": bool(args.send_email),
            }
            if args.mjd_min is not None:
                payload["mjd_min"] = f"{args.mjd_min:.5f}"
            if args.mjd_max is not None:
                payload["mjd_max"] = f"{args.mjd_max:.5f}"

            task_url = submit_job(
                session=session,
                queue_url=queue_url,
                payload=payload,
                timeout=args.request_timeout,
                max_retries=args.max_retries,
            )
            task = poll_until_complete(
                session=session,
                task_url=task_url,
                timeout=args.request_timeout,
                poll_seconds=args.poll_seconds,
                max_retries=args.max_retries,
            )

            result_url = task.get("result_url", "")
            if not result_url:
                result_url = task_url.rstrip("/") + "/result"
            if result_url.startswith("/"):
                result_url = urljoin(args.base_url.rstrip("/") + "/", result_url.lstrip("/"))

            result_resp = session.get(result_url, timeout=args.request_timeout)
            result_resp.raise_for_status()
            text = result_resp.text
            output_file.write_text(text)
            n_data_lines = sum(1 for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#"))

            report_rows.append(
                {
                    "sn_name": sn,
                    "year_from_name": int(row["year_from_name"]),
                    "ra_deg": ra_deg,
                    "dec_deg": dec_deg,
                    "status": status,
                    "message": message,
                    "task_url": task_url,
                    "result_url": result_url,
                    "output_file": str(output_file),
                    "n_data_lines": n_data_lines,
                }
            )
        except Exception as exc:
            status = "failed"
            message = str(exc)
            report_rows.append(
                {
                    "sn_name": sn,
                    "year_from_name": int(row["year_from_name"]) if pd.notna(row["year_from_name"]) else np.nan,
                    "ra_deg": np.nan,
                    "dec_deg": np.nan,
                    "status": status,
                    "message": message,
                    "task_url": task_url,
                    "result_url": result_url,
                    "output_file": "",
                    "n_data_lines": np.nan,
                }
            )

    report = pd.DataFrame(report_rows)
    report.to_csv(report_csv, index=False)

    ok = int((report["status"] == "ok").sum()) if len(report) else 0
    skipped = int((report["status"] == "skipped_existing").sum()) if len(report) else 0
    failed = int((report["status"] == "failed").sum()) if len(report) else 0
    print(f"Targets considered: {len(targets)}")
    print(f"Downloaded: {ok}")
    print(f"Skipped existing: {skipped}")
    print(f"Failed: {failed}")
    print(f"Report: {report_csv}")
    print(f"Output dir: {out_dir}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
