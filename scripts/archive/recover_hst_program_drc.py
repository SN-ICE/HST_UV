#!/usr/bin/env python3
"""
Recover missing HST proposal DRC FITS into their canonical data/hst_programs paths.

This script derives the required product filenames directly from supernovas_input.csv,
so it can rebuild the local proposal archive even if prior manifests were lost.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

from project_paths import PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover missing HST proposal DRC FITS from MAST.")
    parser.add_argument(
        "--input-csv",
        default=str(PROJECT_ROOT / "data" / "supernovas_input.csv"),
        help="CSV containing fits_relpath entries under data/hst_programs.",
    )
    parser.add_argument(
        "--program-ids",
        nargs="+",
        default=["16741", "17179"],
        help="Proposal/program IDs to recover.",
    )
    parser.add_argument(
        "--products-csv",
        default=str(PROJECT_ROOT / "data" / "hst_programs" / "proposal_drc_products.csv"),
        help="Output CSV listing unique DRC products implied by the input table.",
    )
    parser.add_argument(
        "--manifest-csv",
        default=str(PROJECT_ROOT / "data" / "hst_programs" / "proposal_drc_recovery_manifest.csv"),
        help="Output CSV describing recovery status for each product.",
    )
    parser.add_argument("--download", action="store_true", help="Download missing files from MAST.")
    parser.add_argument("--timeout", type=float, default=180.0, help="HTTP timeout in seconds.")
    parser.add_argument("--chunk-bytes", type=int, default=1024 * 1024, help="Streaming chunk size.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel download workers.")
    return parser.parse_args()


def build_products(input_csv: Path, program_ids: set[str]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    with input_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            fits_relpath = str(row.get("fits_relpath", "")).strip()
            if not fits_relpath.startswith("data/hst_programs/"):
                continue
            parts = Path(fits_relpath).parts
            if len(parts) < 4:
                continue
            program_id = parts[2]
            if program_id not in program_ids:
                continue
            product_filename = Path(fits_relpath).name
            if not product_filename.lower().endswith("_drc.fits"):
                continue
            obs_id = product_filename[: -len("_drc.fits")]
            rows.append(
                {
                    "program_id": program_id,
                    "obs_id": obs_id,
                    "productFilename": product_filename,
                    "dataURI": f"mast:HST/product/{product_filename}",
                    "fits_relpath": fits_relpath,
                    "local_path": str((PROJECT_ROOT / fits_relpath).resolve()),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=["program_id", "obs_id", "productFilename", "dataURI", "fits_relpath", "local_path"]
        )

    df = pd.DataFrame(rows).drop_duplicates(subset=["program_id", "productFilename"], keep="first")
    return df.sort_values(["program_id", "productFilename"], kind="stable").reset_index(drop=True)


def recover_one(record: dict[str, object], timeout: float, chunk_bytes: int) -> dict[str, str | int]:
    local_path = Path(str(record["local_path"]))
    data_uri = str(record["dataURI"]).strip()
    status = "ok"
    message = ""
    n_bytes = 0

    if local_path.exists() and local_path.stat().st_size > 0:
        status = "skipped_existing"
        n_bytes = local_path.stat().st_size
    else:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://mast.stsci.edu/api/v0.1/Download/file?uri={quote(data_uri, safe=':/')}"
        tmp_path = local_path.with_suffix(local_path.suffix + ".part")
        try:
            with requests.Session() as session:
                with session.get(url, stream=True, timeout=timeout) as response:
                    response.raise_for_status()
                    with tmp_path.open("wb") as handle:
                        for chunk in response.iter_content(chunk_size=max(chunk_bytes, 8192)):
                            if chunk:
                                handle.write(chunk)
            tmp_path.replace(local_path)
            n_bytes = local_path.stat().st_size
            if n_bytes <= 0:
                raise RuntimeError("empty file downloaded")
        except Exception as exc:
            status = "failed"
            message = str(exc)
            if tmp_path.exists():
                tmp_path.unlink()

    return {
        "program_id": str(record["program_id"]),
        "obs_id": str(record["obs_id"]),
        "productFilename": str(record["productFilename"]),
        "dataURI": data_uri,
        "fits_relpath": str(record["fits_relpath"]),
        "local_path": str(local_path),
        "status": status,
        "message": message,
        "bytes": n_bytes,
    }


def download_products(df: pd.DataFrame, manifest_csv: Path, timeout: float, chunk_bytes: int, workers: int) -> int:
    records = df.to_dict("records")
    rows: list[dict[str, str | int]] = []
    failures = 0

    with ThreadPoolExecutor(max_workers=max(workers, 1)) as executor:
        future_map = {
            executor.submit(recover_one, record, timeout, chunk_bytes): index
            for index, record in enumerate(records, start=1)
        }
        for future in as_completed(future_map):
            index = future_map[future]
            row = future.result()
            if row["status"] == "failed":
                failures += 1
            rows.append(row)
            print(f"[{index}/{len(records)}] {row['status']} - {row['productFilename']}")

    manifest = pd.DataFrame(rows).sort_values(["program_id", "productFilename"], kind="stable").reset_index(drop=True)
    manifest.to_csv(manifest_csv, index=False)
    return failures


def main() -> int:
    args = parse_args()

    input_csv = Path(args.input_csv).expanduser().resolve()
    products_csv = Path(args.products_csv).expanduser().resolve()
    manifest_csv = Path(args.manifest_csv).expanduser().resolve()
    program_ids = {str(value).strip() for value in args.program_ids if str(value).strip()}

    products_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)

    products = build_products(input_csv, program_ids)
    products.to_csv(products_csv, index=False)

    print(f"Unique DRC products: {len(products)}")
    print(f"Products CSV: {products_csv}")

    if not args.download:
        print("Dry run only: no download performed.")
        return 0

    failures = download_products(products, manifest_csv, args.timeout, args.chunk_bytes, args.workers)
    print(f"Manifest: {manifest_csv}")
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
