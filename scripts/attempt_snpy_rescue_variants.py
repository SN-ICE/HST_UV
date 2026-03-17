from __future__ import annotations

import argparse
import csv
import itertools
import re
import shutil
import subprocess
from pathlib import Path


FILTER_RE = re.compile(r"filter ([A-Za-z0-9_:+-]+)")
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Try minimal band-drop rescue variants for non-ok SNooPy files.")
    p.add_argument("--summary-csv", required=True, help="Input SNooPy summary CSV with status/fit messages.")
    p.add_argument(
        "--snoopy-dir",
        default=str(PROJECT_ROOT / "data" / "light_curves" / "SNooPy"),
        help="Directory with *_raw.txt files.",
    )
    p.add_argument(
        "--work-dir",
        default="/tmp/snpy_rescue_search",
        help="Temporary working directory for rescue runs.",
    )
    p.add_argument(
        "--output-csv",
        default="/tmp/snpy_rescue_search_report.csv",
        help="Where to write the rescue report.",
    )
    p.add_argument("--limit", type=int, default=0, help="Optional limit on number of non-ok objects to test.")
    return p.parse_args()


def read_blocks(path: Path) -> tuple[str, list[dict[str, object]]]:
    lines = path.read_text(errors="replace").splitlines()
    header = lines[0]
    blocks: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    for line in lines[1:]:
        if line.startswith("filter "):
            if current is not None:
                blocks.append(current)
            current = {"band": line.split(None, 1)[1], "lines": [line]}
        else:
            assert current is not None
            current["lines"].append(line)
    if current is not None:
        blocks.append(current)
    return header, blocks


def extract_failed_bands(row: dict[str, str]) -> list[str]:
    text = " | ".join(
        [
            row.get("fit_message", ""),
            row.get("max_fit_message", ""),
            row.get("ebv2_fit_message", ""),
        ]
    )
    seen: list[str] = []
    for band in FILTER_RE.findall(text):
        if band not in seen:
            seen.append(band)
    return seen


def candidate_drop_sets(available_bands: list[str], failed_bands: list[str]) -> list[tuple[str, set[str]]]:
    available = set(available_bands)
    candidates: list[tuple[str, set[str]]] = []

    def add(label: str, bands: set[str]) -> None:
        bands = set(bands) & available
        if not bands:
            return
        key = ",".join(sorted(bands))
        if any(",".join(sorted(existing)) == key for _, existing in candidates):
            return
        candidates.append((label, bands))

    # First use the bands explicitly named in failure messages.
    for band in failed_bands:
        add(f"drop_{band}", {band})
    for n in range(2, min(4, len(failed_bands)) + 1):
        for combo in itertools.combinations(failed_bands, n):
            add("drop_" + "_".join(combo), set(combo))

    # Then try the common troublemakers we have seen in this project.
    common_sets = [
        {"u"},
        {"B"},
        {"g"},
        {"r"},
        {"i"},
        {"Bk"},
        {"u", "B"},
        {"g", "u"},
        {"B", "g"},
        {"B", "u"},
        {"B", "i"},
        {"B", "g", "i"},
        {"B", "g", "i", "u"},
        {"B", "V"},
        {"B", "V", "u"},
        {"B", "V", "g"},
        {"g", "u", "i"},
        {"B", "r"},
        {"B", "r", "i"},
    ]
    for bands in common_sets:
        add("drop_" + "_".join(sorted(bands)), bands)

    return candidates


def run_variant(
    run_script: Path,
    src_file: Path,
    sn_name: str,
    drop_bands: set[str],
    work_dir: Path,
) -> dict[str, str]:
    if work_dir.exists():
        shutil.rmtree(work_dir)
    snoopy_dir = work_dir / "snoopy"
    plots_dir = work_dir / "plots"
    fits_dir = work_dir / "fits"
    snoopy_dir.mkdir(parents=True)
    plots_dir.mkdir()
    fits_dir.mkdir()

    header, blocks = read_blocks(src_file)
    kept_lines = [header]
    for block in blocks:
        band = str(block["band"])
        if band in drop_bands:
            continue
        kept_lines.extend(block["lines"])  # type: ignore[arg-type]
    out_file = snoopy_dir / src_file.name
    out_file.write_text("\n".join(kept_lines) + "\n")

    summary_csv = work_dir / "summary.csv"
    cmd = [
        "python",
        str(run_script),
        "--snoopy-dir",
        str(snoopy_dir),
        "--names",
        sn_name,
        "--output-csv",
        str(summary_csv),
        "--plots-dir",
        str(plots_dir),
        "--fits-dir",
        str(fits_dir),
        "--overwrite-existing",
        "--no-plots",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    result: dict[str, str] = {
        "returncode": str(proc.returncode),
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-8:]),
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-8:]),
    }
    if summary_csv.exists():
        with summary_csv.open() as f:
            rows = list(csv.DictReader(f))
        if rows:
            result.update(rows[0])
    return result


def locate_raw_file(snoopy_dir: Path, sn_name: str) -> Path | None:
    candidates = [
        snoopy_dir / f"{sn_name}_raw.txt",
        snoopy_dir / f"{sn_name.replace(':', '_')}_raw.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary_csv)
    snoopy_dir = Path(args.snoopy_dir)
    work_root = Path(args.work_dir)
    output_csv = Path(args.output_csv)
    run_script = PROJECT_ROOT / "scripts" / "run_snpy_dual_model_batch.py"

    with summary_path.open() as f:
        rows = list(csv.DictReader(f))

    non_ok_rows = [row for row in rows if row.get("status") != "ok"]
    if args.limit:
        non_ok_rows = non_ok_rows[: args.limit]

    report_rows: list[dict[str, str]] = []

    for row in non_ok_rows:
        sn_name = row["sn_name"]
        raw_file = locate_raw_file(snoopy_dir, sn_name)
        base = {
            "sn_name": sn_name,
            "source_file": str(raw_file or ""),
            "initial_status": row.get("status", ""),
            "initial_fit_message": row.get("fit_message", ""),
            "rescue_status": "",
            "drop_bands": "",
            "variant_label": "",
            "best_status": "",
            "best_fit_message": "",
        }
        if raw_file is None:
            base["rescue_status"] = "missing_raw_file"
            report_rows.append(base)
            continue

        _, blocks = read_blocks(raw_file)
        available_bands = [str(block["band"]) for block in blocks]
        base["available_bands"] = ",".join(available_bands)

        if len(available_bands) <= 1:
            base["rescue_status"] = "not_attempted_single_filter"
            report_rows.append(base)
            continue

        failed_bands = extract_failed_bands(row)
        best_ok: dict[str, str] | None = None
        best_drop: set[str] | None = None
        best_label = ""
        for idx, (label, drop_bands) in enumerate(candidate_drop_sets(available_bands, failed_bands), 1):
            variant_work = work_root / sn_name.replace(":", "_") / f"{idx:02d}_{label}"
            result = run_variant(run_script, raw_file, sn_name, drop_bands, variant_work)
            status = result.get("status", "")
            if status == "ok":
                best_ok = result
                best_drop = drop_bands
                best_label = label
                break

        if best_ok is not None and best_drop is not None:
            base["rescue_status"] = "ok"
            base["drop_bands"] = ",".join(sorted(best_drop))
            base["variant_label"] = best_label
            base["best_status"] = best_ok.get("status", "")
            base["best_fit_message"] = best_ok.get("fit_message", "")
            base["best_max_status"] = best_ok.get("max_status", "")
            base["best_ebv2_status"] = best_ok.get("ebv2_status", "")
        else:
            base["rescue_status"] = "no_ok_variant_found"
        report_rows.append(base)

    fieldnames: list[str] = []
    for row in report_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    print(output_csv)
    print(f"processed={len(report_rows)}")
    print(f"rescued={sum(1 for row in report_rows if row.get('rescue_status') == 'ok')}")


if __name__ == "__main__":
    main()
