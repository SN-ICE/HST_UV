#!/usr/bin/env python3
"""
Robust resumable downloader for MAST HST products listed in a CSV.

Expected columns in products CSV:
- dataURI
- obs_id
- productFilename
- size (bytes; optional but recommended)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

from project_paths import additional_hst_download_path, light_curves_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download MAST HST products from a CSV list.")
    p.add_argument(
        "--products-csv",
        default=str(light_curves_path("cspii32_bestexptime_drcdrz_products.csv")),
        help="CSV with product rows (must include dataURI, obs_id, productFilename).",
    )
    p.add_argument(
        "--download-root",
        default=str(additional_hst_download_path("mastDownload", "HST")),
        help="Root folder where files will be stored as <obs_id>/<productFilename>.",
    )
    p.add_argument(
        "--manifest-csv",
        default=str(light_curves_path("cspii32_bestexptime_download_manifest.csv")),
        help="Output manifest CSV.",
    )
    p.add_argument("--timeout", type=float, default=180.0, help="HTTP timeout seconds.")
    p.add_argument("--chunk-bytes", type=int, default=1024 * 1024, help="Streaming chunk size.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    products_csv = Path(args.products_csv).expanduser().resolve()
    download_root = Path(args.download_root).expanduser().resolve()
    manifest_csv = Path(args.manifest_csv).expanduser().resolve()

    download_root.mkdir(parents=True, exist_ok=True)
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(products_csv)
    required = {"dataURI", "obs_id", "productFilename"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required - set(df.columns)}")

    rows = []
    n_total = len(df)
    n_done = 0

    with requests.Session() as s:
        for i, r in df.iterrows():
            data_uri = str(r["dataURI"]).strip()
            obs_id = str(r["obs_id"]).strip()
            filename = str(r["productFilename"]).strip()
            expected_size = pd.to_numeric(r.get("size"), errors="coerce")
            expected_size = int(expected_size) if pd.notna(expected_size) else None

            out_dir = download_root / obs_id
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / filename

            status = "ok"
            message = ""
            n_bytes = 0

            # Skip if already complete.
            if out_file.exists() and expected_size is not None and out_file.stat().st_size == expected_size:
                status = "skipped_existing"
                n_bytes = out_file.stat().st_size
                rows.append(
                    {
                        "obs_id": obs_id,
                        "productFilename": filename,
                        "dataURI": data_uri,
                        "status": status,
                        "message": "",
                        "local_path": str(out_file),
                        "bytes": n_bytes,
                    }
                )
                n_done += 1
                continue

            url = f"https://mast.stsci.edu/api/v0.1/Download/file?uri={quote(data_uri, safe=':/')}"

            try:
                with s.get(url, stream=True, timeout=args.timeout) as resp:
                    resp.raise_for_status()
                    tmp_path = out_file.with_suffix(out_file.suffix + ".part")
                    with tmp_path.open("wb") as f:
                        for chunk in resp.iter_content(chunk_size=max(args.chunk_bytes, 8192)):
                            if chunk:
                                f.write(chunk)
                    tmp_size = tmp_path.stat().st_size
                    if expected_size is not None and tmp_size != expected_size:
                        raise RuntimeError(f"size mismatch: got {tmp_size}, expected {expected_size}")
                    tmp_path.replace(out_file)
                    n_bytes = out_file.stat().st_size
            except Exception as e:
                status = "failed"
                message = str(e)

            rows.append(
                {
                    "obs_id": obs_id,
                    "productFilename": filename,
                    "dataURI": data_uri,
                    "status": status,
                    "message": message,
                    "local_path": str(out_file) if status != "failed" else "",
                    "bytes": n_bytes,
                }
            )
            n_done += 1
            print(f"[{n_done}/{n_total}] {status} - {filename}")

    manifest = pd.DataFrame(rows)
    manifest.to_csv(manifest_csv, index=False)
    ok = int((manifest["status"] == "ok").sum()) if len(manifest) else 0
    skipped = int((manifest["status"] == "skipped_existing").sum()) if len(manifest) else 0
    failed = int((manifest["status"] == "failed").sum()) if len(manifest) else 0
    print(f"Total rows: {len(manifest)}")
    print(f"Downloaded now: {ok}")
    print(f"Already present: {skipped}")
    print(f"Failed: {failed}")
    print(f"Manifest: {manifest_csv}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
