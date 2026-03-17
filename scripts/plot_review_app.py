#!/usr/bin/env python3
"""Small Flask app to review diagnostic plots one by one."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path

import pandas as pd
from flask import Flask, redirect, render_template, request, send_from_directory, url_for


def sanitize_name_for_file(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name).strip())
    return s.strip("_") or "object"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plots-dir", default="outputs/plots")
    parser.add_argument("--summary", default="outputs/photometry_summary_keep.csv")
    parser.add_argument("--paper-table", default="outputs/sn_mag_summary_1kpc_full.csv")
    parser.add_argument("--review-csv", default="outputs/plot_review_flags.csv")
    parser.add_argument("--names-file", default="")
    parser.add_argument("--title", default="Plot Review")
    parser.add_argument("--paper-only-mode", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


@dataclass
class PlotRecord:
    supernova: str
    filename: str
    status: str
    adopted_r_survey: str
    uv_mag: str
    adopted_r_mag: str


def load_plot_records(
    plots_dir: Path,
    summary_path: Path,
    paper_path: Path,
    names_file: Path | None = None,
) -> list[PlotRecord]:
    summary = pd.read_csv(summary_path).drop_duplicates(subset=["Supernova"], keep="last")
    paper = pd.read_csv(paper_path).drop_duplicates(subset=["SN name"], keep="last")
    paper = paper.rename(columns={"SN name": "Supernova"})
    merged = summary.merge(
        paper[["Supernova", "UV mag", "Adopted r survey", "Adopted r mag"]],
        on="Supernova",
        how="left",
    )
    selected_names: set[str] | None = None
    if names_file and names_file.exists():
        selected_names = {
            line.strip()
            for line in names_file.read_text().splitlines()
            if line.strip()
        }

    records: list[PlotRecord] = []
    for _, row in merged.iterrows():
        supernova = str(row["Supernova"])
        if selected_names is not None and supernova not in selected_names:
            continue
        filename = f"{sanitize_name_for_file(row['Supernova'])}_plot.png"
        if not (plots_dir / filename).exists():
            continue
        records.append(
            PlotRecord(
                supernova=supernova,
                filename=filename,
                status=str(row.get("status", "")),
                adopted_r_survey="" if pd.isna(row.get("Adopted r survey")) else str(row.get("Adopted r survey")),
                uv_mag="" if pd.isna(row.get("UV mag")) else str(row.get("UV mag")),
                adopted_r_mag="" if pd.isna(row.get("Adopted r mag")) else str(row.get("Adopted r mag")),
            )
        )
    records.sort(key=lambda rec: rec.supernova.lower())
    return records


def load_review_flags(review_csv: Path) -> dict[str, dict[str, str]]:
    if not review_csv.exists():
        return {}
    rows: dict[str, dict[str, str]] = {}
    with review_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows[row["filename"]] = row
    return rows


def save_review_flags(review_csv: Path, flags: dict[str, dict[str, str]]) -> None:
    review_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "filename",
        "supernova",
        "needs_revision",
        "paper_candidate",
        "reviewed",
        "updated_at_utc",
    ]
    with review_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for filename in sorted(flags):
            writer.writerow(flags[filename])


def create_app(args: argparse.Namespace) -> Flask:
    plots_dir = Path(args.plots_dir).resolve()
    summary_path = Path(args.summary).resolve()
    paper_path = Path(args.paper_table).resolve()
    review_csv = Path(args.review_csv).resolve()
    names_file = Path(args.names_file).resolve() if args.names_file else None

    app = Flask(
        __name__,
        template_folder=str(Path(__file__).resolve().parent / "templates"),
    )
    app.config["PLOTS_DIR"] = plots_dir
    app.config["REVIEW_CSV"] = review_csv
    app.config["APP_TITLE"] = args.title
    app.config["PAPER_ONLY_MODE"] = args.paper_only_mode
    app.config["PLOT_RECORDS"] = load_plot_records(plots_dir, summary_path, paper_path, names_file)

    def get_flags() -> dict[str, dict[str, str]]:
        return load_review_flags(review_csv)

    def record_for_index(index: int) -> PlotRecord | None:
        records: list[PlotRecord] = app.config["PLOT_RECORDS"]
        if 0 <= index < len(records):
            return records[index]
        return None

    def first_unreviewed_index() -> int:
        flags = get_flags()
        records: list[PlotRecord] = app.config["PLOT_RECORDS"]
        for i, record in enumerate(records):
            if flags.get(record.filename, {}).get("reviewed") != "true":
                return i
        return 0

    @app.get("/")
    def index():
        return redirect(url_for("review", idx=first_unreviewed_index()))

    @app.get("/plots/<path:filename>")
    def serve_plot(filename: str):
        return send_from_directory(plots_dir, filename, max_age=0)

    @app.get("/review/<int:idx>")
    def review(idx: int):
        record = record_for_index(idx)
        if record is None:
            return redirect(url_for("review", idx=max(0, len(app.config["PLOT_RECORDS"]) - 1)))

        flags = get_flags()
        current = flags.get(record.filename, {})
        records: list[PlotRecord] = app.config["PLOT_RECORDS"]
        reviewed_count = sum(1 for r in records if flags.get(r.filename, {}).get("reviewed") == "true")
        revision_count = sum(1 for row in flags.values() if row.get("needs_revision") == "true")
        paper_count = sum(1 for row in flags.values() if row.get("paper_candidate") == "true")

        return render_template(
            "plot_review.html",
            app_title=app.config["APP_TITLE"],
            paper_only_mode=app.config["PAPER_ONLY_MODE"],
            record=record,
            plot_mtime=int((plots_dir / record.filename).stat().st_mtime),
            idx=idx,
            total=len(records),
            reviewed_count=reviewed_count,
            revision_count=revision_count,
            paper_count=paper_count,
            needs_revision=current.get("needs_revision") == "true",
            paper_candidate=current.get("paper_candidate") == "true",
            prev_idx=max(0, idx - 1),
            next_idx=min(len(records) - 1, idx + 1),
            first_unreviewed_idx=first_unreviewed_index(),
        )

    @app.post("/review/<int:idx>")
    def save_review(idx: int):
        record = record_for_index(idx)
        if record is None:
            return redirect(url_for("index"))

        flags = get_flags()
        needs_revision = "true" in request.form.getlist("needs_revision")
        paper_candidate = "true" in request.form.getlist("paper_candidate")
        reviewed = request.form.get("reviewed") == "true" or needs_revision or paper_candidate
        action = request.form.get("action", "save")

        flags[record.filename] = {
            "filename": record.filename,
            "supernova": record.supernova,
            "needs_revision": "true" if needs_revision else "false",
            "paper_candidate": "true" if paper_candidate else "false",
            "reviewed": "true" if reviewed else "false",
            "updated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        save_review_flags(review_csv, flags)

        records: list[PlotRecord] = app.config["PLOT_RECORDS"]
        if action == "next":
            return redirect(url_for("review", idx=min(len(records) - 1, idx + 1)))
        if action == "prev":
            return redirect(url_for("review", idx=max(0, idx - 1)))
        if action == "next-unreviewed":
            return redirect(url_for("review", idx=first_unreviewed_index()))
        return redirect(url_for("review", idx=idx))

    return app


def main() -> int:
    args = parse_args()
    app = create_app(args)
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
