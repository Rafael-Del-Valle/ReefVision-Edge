#!/usr/bin/env bash
set -euo pipefail

repo_root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)
output=${1:-"$repo_root/.artifacts/reefvision-edge-offline"}
artifact_root="$repo_root/.artifacts"
mkdir -p "$artifact_root"
artifact_root=$(CDPATH='' cd -- "$artifact_root" && pwd -P)
case "$output" in
  /*) bundle=$output ;;
  *) bundle="$repo_root/$output" ;;
esac
bundle=$(realpath -m -- "$bundle")
case "$bundle" in
  "$artifact_root"/*) ;;
  *) echo "output must be a child of $artifact_root" >&2; exit 2 ;;
esac
[ ! -e "$bundle" ] || { echo "refusing to overwrite existing bundle: $bundle" >&2; exit 2; }

mkdir -p "$bundle/app" "$bundle/wheels"
cd "$repo_root"
npm --prefix web ci
npm --prefix web run build
python3 -m build --wheel --outdir "$bundle/app"
python3 -m pip download --dest "$bundle/wheels" .
git archive --format=tar.gz --output="$bundle/reefvision-edge-source.tar.gz" HEAD
(
  cd "$bundle"
  find . -type f ! -name SHA256SUMS.txt -print0 |
    sort -z |
    xargs -0 sha256sum > SHA256SUMS.txt
)
printf 'Offline bundle created at %s\n' "$bundle"
