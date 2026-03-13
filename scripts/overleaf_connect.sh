#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/overleaf_connect.sh --git-url <overleaf_git_url> [options]

Options:
  --clone-dir <path>      Local clone folder (default: .overleaf)
  --source-dir <path>     Source images folder in this repo (default: images)
  --target-dir <path>     Destination folder in Overleaf repo (default: images)
  --branch <name>         Overleaf branch to use (default: detected clone branch)
  --help                  Show this help
EOF
}

git_url=""
clone_dir=".overleaf"
source_dir="images"
target_dir="images"
branch=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --git-url)
      git_url="${2:-}"
      shift 2
      ;;
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

if [[ -z "$git_url" ]]; then
  echo "Missing required argument: --git-url" >&2
  usage >&2
  exit 1
fi

repo_root="$(git rev-parse --show-toplevel)"
abs_clone_dir="$repo_root/$clone_dir"

if [[ -d "$abs_clone_dir/.git" ]]; then
  git -C "$abs_clone_dir" remote set-url origin "$git_url"
  git -C "$abs_clone_dir" fetch --all --prune
else
  git clone "$git_url" "$abs_clone_dir"
fi

if [[ -n "$branch" ]]; then
  git -C "$abs_clone_dir" checkout "$branch" 2>/dev/null || git -C "$abs_clone_dir" checkout -b "$branch"
else
  branch="$(git -C "$abs_clone_dir" rev-parse --abbrev-ref HEAD)"
fi

git config --local overleaf.cloneDir "$clone_dir"
git config --local overleaf.sourceDir "$source_dir"
git config --local overleaf.targetDir "$target_dir"
git config --local overleaf.branch "$branch"
git config --local --unset-all overleaf.url 2>/dev/null || true

cat <<EOF
Overleaf connection configured.
  Clone dir:   $clone_dir
  Source dir:  $source_dir
  Target dir:  $target_dir
  Branch:      $branch

Next sync command:
  scripts/overleaf_sync_images.sh
EOF
