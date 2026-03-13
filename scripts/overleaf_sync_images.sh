#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/overleaf_sync_images.sh [options]

Options:
  --clone-dir <path>         Override local Overleaf clone dir
  --source-dir <path>        Override source image directory
  --target-dir <path>        Override destination directory in Overleaf repo
  --branch <name>            Override branch
  --commit-message <text>    Commit message for sync commit
  --delete                   Delete files in destination that are not in source
  --include-fits             Include FITS files (excluded by default)
  --help                     Show this help
EOF
}

repo_root="$(git rev-parse --show-toplevel)"
clone_dir="$(git config --local --get overleaf.cloneDir || true)"
source_dir="$(git config --local --get overleaf.sourceDir || true)"
target_dir="$(git config --local --get overleaf.targetDir || true)"
branch="$(git config --local --get overleaf.branch || true)"
delete_flag=false
include_fits=false
commit_message="Sync images from HST_UV ($(date +%Y-%m-%d\ %H:%M:%S))"

clone_dir="${clone_dir:-.overleaf}"
source_dir="${source_dir:-images}"
target_dir="${target_dir:-images}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clone-dir)
      clone_dir="${2:-}"
      shift 2
      ;;
    --source-dir)
      source_dir="${2:-}"
      shift 2
      ;;
    --target-dir)
      target_dir="${2:-}"
      shift 2
      ;;
    --branch)
      branch="${2:-}"
      shift 2
      ;;
    --commit-message)
      commit_message="${2:-}"
      shift 2
      ;;
    --delete)
      delete_flag=true
      shift
      ;;
    --include-fits)
      include_fits=true
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

abs_clone_dir="$repo_root/$clone_dir"
abs_source_dir="$repo_root/$source_dir"

if [[ ! -d "$abs_clone_dir/.git" ]]; then
  echo "Overleaf clone not found at: $abs_clone_dir" >&2
  echo "Run scripts/overleaf_connect.sh first." >&2
  exit 1
fi

if [[ ! -d "$abs_source_dir" ]]; then
  echo "Source directory not found: $abs_source_dir" >&2
  exit 1
fi

if [[ -n "$branch" ]]; then
  git -C "$abs_clone_dir" checkout "$branch" 2>/dev/null || git -C "$abs_clone_dir" checkout -b "$branch"
else
  branch="$(git -C "$abs_clone_dir" rev-parse --abbrev-ref HEAD)"
fi

git -C "$abs_clone_dir" pull --rebase origin "$branch" || true
mkdir -p "$abs_clone_dir/$target_dir"

rsync_args=(-av)
if [[ "$delete_flag" == "true" ]]; then
  rsync_args+=(--delete)
fi
if [[ "$include_fits" != "true" ]]; then
  rsync_args+=(--exclude "*.fits" --exclude "*.fit" --exclude "*.FITS" --exclude "*.FIT")
fi
rsync "${rsync_args[@]}" "$abs_source_dir"/ "$abs_clone_dir/$target_dir"/

git -C "$abs_clone_dir" add "$target_dir"
if git -C "$abs_clone_dir" diff --cached --quiet; then
  echo "No image changes to push."
  exit 0
fi

git -C "$abs_clone_dir" commit -m "$commit_message"
git -C "$abs_clone_dir" push origin "$branch"

echo "Overleaf sync complete."
