#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEST_BASE="${1:-/Volumes/MIRO/Work_done}"
DEST_ROOT="$DEST_BASE/$(basename "$SRC_ROOT")"

typeset -a SOURCE_PATHS=()
typeset -a DEST_PATHS=()

add_mapping() {
  local src_rel="$1"
  local dest_rel="$2"
  SOURCE_PATHS+=("$src_rel")
  DEST_PATHS+=("$dest_rel")
}

add_mapping_if_dir() {
  local src_rel="$1"
  local dest_rel="$2"
  if [[ -d "$SRC_ROOT/$src_rel" ]]; then
    add_mapping "$src_rel" "$dest_rel"
  fi
}

sync_dir() {
  local src_dir="$1"
  local dest_dir="$2"
  mkdir -p "$dest_dir"
  rsync -aH --partial --append-verify --info=stats2 "$src_dir/" "$dest_dir/"
}

verify_dir() {
  local src_dir="$1"
  local dest_dir="$2"
  local verify_output
  verify_output="$(rsync -aH --delete --dry-run --itemize-changes "$src_dir/" "$dest_dir/")"
  if [[ -n "$verify_output" ]]; then
    echo "Verification failed for $src_dir -> $dest_dir"
    echo "$verify_output"
    return 1
  fi
}

migrate_backup_dir() {
  local src_rel="$1"
  local dest_rel="$2"
  local src_dir="$DEST_ROOT/$src_rel"
  local dest_dir="$DEST_ROOT/$dest_rel"

  if [[ ! -d "$src_dir" ]]; then
    return 0
  fi

  echo "Migrating backup layout: $src_rel -> $dest_rel"
  sync_dir "$src_dir" "$dest_dir"

  local verify_output
  verify_output="$(rsync -aH --dry-run --itemize-changes "$src_dir/" "$dest_dir/")"
  if [[ -n "$verify_output" ]]; then
    echo "Migration verification failed for $src_rel"
    echo "$verify_output"
    exit 1
  fi

  rm -rf "$src_dir"
}

if [[ ! -d "$DEST_BASE" ]]; then
  echo "Destination not found: $DEST_BASE"
  exit 1
fi

mkdir -p "$DEST_ROOT"

add_mapping_if_dir "data/hst_programs" "data/hst_programs"
add_mapping_if_dir "data/hst_downloads" "data/hst_downloads_v2"
add_mapping_if_dir "outputs/images" "outputs/images_v2"

echo "Source root:      $SRC_ROOT"
echo "Backup root:      $DEST_ROOT"
echo "Directories:"
for (( i = 1; i <= ${#SOURCE_PATHS}; ++i )); do
  echo "  - ${SOURCE_PATHS[$i]} -> ${DEST_PATHS[$i]}"
done

echo
echo "Migrating legacy backup layout if needed..."
migrate_backup_dir "16741" "data/hst_programs/16741"
migrate_backup_dir "17179" "data/hst_programs/17179"
migrate_backup_dir "images" "outputs/images_v2"

echo
echo "Starting rsync..."
for (( i = 1; i <= ${#SOURCE_PATHS}; ++i )); do
  sync_dir "$SRC_ROOT/${SOURCE_PATHS[$i]}" "$DEST_ROOT/${DEST_PATHS[$i]}"
done

echo
echo "Verifying destination copies with rsync dry-run..."
for (( i = 1; i <= ${#SOURCE_PATHS}; ++i )); do
  verify_dir "$SRC_ROOT/${SOURCE_PATHS[$i]}" "$DEST_ROOT/${DEST_PATHS[$i]}" || {
    echo "Aborting before any deletion."
    exit 1
  }
done

echo
echo "---- Local ----"
for path in "${SOURCE_PATHS[@]}"; do
  du -sh "$SRC_ROOT/$path"
done
echo "---- Copied ----"
for path in "${DEST_PATHS[@]}"; do
  du -sh "$DEST_ROOT/$path"
done

echo
read -r "CONFIRM?Type YES to delete the verified local copies: "
if [[ "$CONFIRM" != "YES" ]]; then
  echo "Aborted, nothing deleted."
  exit 1
fi

for path in "${SOURCE_PATHS[@]}"; do
  rm -rf "$SRC_ROOT/$path"
done

mkdir -p "$SRC_ROOT/data/hst_programs" "$SRC_ROOT/data/hst_downloads" "$SRC_ROOT/outputs/images"

echo "Backup verified, backup layout migrated, and local heavy folders removed."
