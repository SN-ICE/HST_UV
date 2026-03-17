#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_NAME="$(basename "$PROJECT_ROOT")"
BACKUP_NAME="HST_UV_local_env"
SCRIPT_NAME="$(basename "$0")"
MANIFEST_DIR_NAME="_backup_manifests"

HEAVY_DIRS=(
  "data/hst_programs"
  "data/hst_downloads"
  "outputs/images"
)

RSYNC_EXCLUDES=(
  "--exclude=.DS_Store"
  "--exclude=._*"
  "--exclude=__pycache__/"
  "--exclude=*.pyc"
  "--exclude=.pytest_cache/"
)

usage() {
  cat <<EOF
Usage:
  $SCRIPT_NAME DESTINATION_ROOT [--dry-run] [--prune-heavy-local]

Description:
  Copies the full project directory to DESTINATION_ROOT/$BACKUP_NAME using rsync.
  If DESTINATION_ROOT already ends with $BACKUP_NAME, that path is used directly.
  If --prune-heavy-local is provided, the large local folders are removed only
  after a successful backup and only if the corresponding destination folders
  pass a file-count and byte-size verification check.

Examples:
  $SCRIPT_NAME /Volumes/MIRO/Work_done
  $SCRIPT_NAME /Volumes/MIRO/Work_done --prune-heavy-local
EOF
}

log() {
  printf '[backup] %s\n' "$*"
}

fail() {
  printf '[backup] ERROR: %s\n' "$*" >&2
  exit 1
}

resolve_path() {
  python3 - "$1" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).expanduser().resolve())
PY
}

sum_tree() {
  python3 - "$1" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
files = 0
bytes_total = 0
if root.exists():
    for path in root.rglob("*"):
        if path.is_file():
            files += 1
            bytes_total += path.stat().st_size
print(f"{files} {bytes_total}")
PY
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

DESTINATION_ROOT=""
DRY_RUN=0
PRUNE_HEAVY_LOCAL=0

for arg in "$@"; do
  case "$arg" in
    --dry-run)
      DRY_RUN=1
      ;;
    --prune-heavy-local)
      PRUNE_HEAVY_LOCAL=1
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    -*)
      fail "Unknown option: $arg"
      ;;
    *)
      if [[ -n "$DESTINATION_ROOT" ]]; then
        fail "Only one destination root can be provided."
      fi
      DESTINATION_ROOT="$arg"
      ;;
  esac
done

[[ -n "$DESTINATION_ROOT" ]] || fail "Destination root is required."

if [[ ! -d "$DESTINATION_ROOT" ]]; then
  fail "Destination root does not exist: $DESTINATION_ROOT"
fi

DESTINATION_ROOT="$(resolve_path "$DESTINATION_ROOT")"
if [[ "$(basename "$DESTINATION_ROOT")" == "$BACKUP_NAME" ]]; then
  DESTINATION_PROJECT="$DESTINATION_ROOT"
else
  DESTINATION_PROJECT="$DESTINATION_ROOT/$BACKUP_NAME"
fi
MANIFEST_DIR="$DESTINATION_PROJECT/$MANIFEST_DIR_NAME"

log "Project root: $PROJECT_ROOT"
log "Backup destination: $DESTINATION_PROJECT"

log "Large local folders targeted for optional pruning:"
for rel_dir in "${HEAVY_DIRS[@]}"; do
  src_dir="$PROJECT_ROOT/$rel_dir"
  if [[ -e "$src_dir" ]]; then
    du -sh "$src_dir"
  else
    log "Missing locally, skipping size report: $rel_dir"
  fi
done

mkdir -p "$DESTINATION_PROJECT"
mkdir -p "$MANIFEST_DIR"

RSYNC_ARGS=(
  -a
  --human-readable
  --progress
  --partial
  "${RSYNC_EXCLUDES[@]}"
  "$PROJECT_ROOT/"
  "$DESTINATION_PROJECT/"
)

if [[ $DRY_RUN -eq 1 ]]; then
  RSYNC_ARGS=( -a -n --human-readable --progress "${RSYNC_EXCLUDES[@]}" "$PROJECT_ROOT/" "$DESTINATION_PROJECT/" )
  log "Running rsync in dry-run mode."
else
  log "Starting full project backup with rsync."
fi

rsync "${RSYNC_ARGS[@]}"

if [[ $DRY_RUN -eq 1 ]]; then
  log "Dry run completed. No files were copied or deleted."
  exit 0
fi

log "Backup completed successfully."

MANIFEST_PATH="$MANIFEST_DIR/$(date +%Y%m%d_%H%M%S)_backup_manifest.txt"
{
  printf 'project_root=%s\n' "$PROJECT_ROOT"
  printf 'backup_destination=%s\n' "$DESTINATION_PROJECT"
  printf 'created_utc=%s\n' "$(env TZ=UTC date +%Y-%m-%dT%H:%M:%SZ)"
  for rel_dir in "${HEAVY_DIRS[@]}"; do
    src_dir="$PROJECT_ROOT/$rel_dir"
    dst_dir="$DESTINATION_PROJECT/$rel_dir"
    src_stats="$(sum_tree "$src_dir")"
    dst_stats="$(sum_tree "$dst_dir")"
    printf '%s source_files=%s source_bytes=%s backup_files=%s backup_bytes=%s\n' \
      "$rel_dir" \
      "${src_stats%% *}" "${src_stats##* }" \
      "${dst_stats%% *}" "${dst_stats##* }"
  done
} > "$MANIFEST_PATH"
log "Wrote backup manifest: $MANIFEST_PATH"

if [[ $PRUNE_HEAVY_LOCAL -eq 0 ]]; then
  log "Local files were left untouched. Re-run with --prune-heavy-local to remove the heavy local folders."
  exit 0
fi

log "Verifying backup copies of the heavy folders before local removal."
for rel_dir in "${HEAVY_DIRS[@]}"; do
  src_dir="$PROJECT_ROOT/$rel_dir"
  dst_dir="$DESTINATION_PROJECT/$rel_dir"

  if [[ ! -e "$src_dir" ]]; then
    log "Local folder already absent, skipping: $rel_dir"
    continue
  fi

  [[ -e "$dst_dir" ]] || fail "Backup copy missing, refusing to delete local folder: $dst_dir"

  src_stats="$(sum_tree "$src_dir")"
  dst_stats="$(sum_tree "$dst_dir")"
  src_files="${src_stats%% *}"
  src_bytes="${src_stats##* }"
  dst_files="${dst_stats%% *}"
  dst_bytes="${dst_stats##* }"

  if [[ "$src_files" != "$dst_files" || "$src_bytes" != "$dst_bytes" ]]; then
    fail "Verification failed for $rel_dir (source files/bytes: $src_files/$src_bytes, backup files/bytes: $dst_files/$dst_bytes)"
  fi

  log "Verified $rel_dir (files=$src_files bytes=$src_bytes)"
done

log "Removing heavy local folders after successful backup verification."
for rel_dir in "${HEAVY_DIRS[@]}"; do
  src_dir="$PROJECT_ROOT/$rel_dir"
  if [[ -e "$src_dir" ]]; then
    rm -rf "$src_dir"
    log "Removed local folder: $src_dir"
  fi
done

log "Local pruning completed."
