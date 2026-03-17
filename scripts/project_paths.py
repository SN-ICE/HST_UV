from __future__ import annotations

from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
LIGHT_CURVES_DIR = DATA_DIR / "light_curves"
HST_DOWNLOADS_DIR = DATA_DIR / "hst_downloads"
HST_PROGRAMS_DIR = DATA_DIR / "hst_programs"


def light_curves_dir() -> Path:
    return LIGHT_CURVES_DIR


def light_curves_path(*parts: str) -> Path:
    return light_curves_dir().joinpath(*parts)


def additional_hst_download_dir() -> Path:
    return HST_DOWNLOADS_DIR / "additional_f275w_bestexptime"


def additional_hst_download_path(*parts: str) -> Path:
    return additional_hst_download_dir().joinpath(*parts)


def hst_program_dir(program_id: str) -> Path:
    return HST_PROGRAMS_DIR / str(program_id)


def normalize_hst_relpath(path_str: str) -> str:
    return str(path_str).strip()


def resolve_project_path(path_str: str) -> Path:
    raw = str(path_str).strip()
    if not raw:
        return PROJECT_ROOT

    raw_path = Path(raw).expanduser()
    if raw_path.is_absolute():
        return raw_path.resolve()

    normalized = normalize_hst_relpath(raw)
    preferred = (PROJECT_ROOT / normalized).resolve()
    if preferred.exists():
        return preferred

    original = (PROJECT_ROOT / raw).resolve()
    if original.exists():
        return original

    return preferred
