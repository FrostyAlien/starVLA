#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   export DEST=/path/to/dir && bash examples/LIBERO/data_preparation.sh
# or
#   bash examples/LIBERO/data_preparation.sh /path/to/dir
#
# Optional env:
#   export HF_TOKEN=hf_...              # strongly recommended to avoid rate limits
#   export HF_MAX_WORKERS=4             # lower = fewer parallel requests
#   export HF_RETRY_MAX=8               # retry attempts
#   export HF_BACKOFF_BASE=5            # seconds (exponential backoff base)
#   export HF_BACKOFF_CAP=300           # max sleep seconds between retries
#   export HF_XET_HIGH_PERFORMANCE=1    # (optional) maximize transfer performance (uses more CPU/bandwidth)

DEST="${DEST:-${1:-}}"
if [[ -z "${DEST}" ]]; then
  echo "ERROR: DEST is not set."
  echo "  export DEST=/path/to/dir && bash examples/LIBERO/data_preparation.sh"
  echo "  or: bash examples/LIBERO/data_preparation.sh /path/to/dir"
  exit 1
fi

CUR="$(pwd)"
mkdir -p "$DEST"
DEST="$(cd "$DEST" && pwd -P)"

# Keep HF caches inside DEST by default (helps on clusters / shared machines).
# You can override any of these via environment variables.
export HF_HOME="${HF_HOME:-$DEST/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_XET_CACHE="${HF_XET_CACHE:-$HF_HOME/xet}"

HF_MAX_WORKERS="${HF_MAX_WORKERS:-4}"
HF_RETRY_MAX="${HF_RETRY_MAX:-8}"
HF_BACKOFF_BASE="${HF_BACKOFF_BASE:-5}"
HF_BACKOFF_CAP="${HF_BACKOFF_CAP:-300}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARN: HF_TOKEN is not set. Anonymous downloads are more likely to hit rate limits."
  echo "      Consider: export HF_TOKEN=hf_...  (or run: hf auth login)"
fi

# Pick CLI:
# - Prefer `hf download` (newer huggingface_hub CLI)
# - Fallback to legacy `huggingface-cli download`
HF_FLAVOR=""
if command -v hf >/dev/null 2>&1; then
  HF_FLAVOR="hf"
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_FLAVOR="huggingface-cli"
else
  echo "ERROR: Neither 'hf' nor 'huggingface-cli' was found in PATH."
  exit 1
fi

retry() {
  local attempt=1
  while true; do
    if "$@"; then
      return 0
    fi
    local rc=$?
    if (( attempt >= HF_RETRY_MAX )); then
      echo "ERROR: command failed after ${HF_RETRY_MAX} attempts (exit $rc): $*" >&2
      return "$rc"
    fi

    local sleep_for=$(( HF_BACKOFF_BASE * (2 ** (attempt - 1)) ))
    if (( sleep_for > HF_BACKOFF_CAP )); then
      sleep_for="$HF_BACKOFF_CAP"
    fi

    echo "WARN: command failed (exit $rc). Retrying in ${sleep_for}s: $*" >&2
    sleep "${sleep_for}"
    attempt=$((attempt + 1))
  done
}

download_repo() {
  local repo="$1"
  local repo_type="$2"
  local out_dir="$3"

  mkdir -p "$out_dir"

  if [[ "$HF_FLAVOR" == "hf" ]]; then
    # `hf download` supports max-workers and token.
    local args=(download "$repo" --repo-type "$repo_type" --local-dir "$out_dir" --max-workers "$HF_MAX_WORKERS")
    if [[ -n "${HF_TOKEN:-}" ]]; then
      args+=(--token "$HF_TOKEN")
    fi
    hf "${args[@]}"
  else
    # Legacy CLI: use resume + avoid symlink mode in local-dir.
    local args=(download "$repo" --repo-type "$repo_type" --local-dir "$out_dir" --resume-download --local-dir-use-symlinks False)
    if [[ -n "${HF_TOKEN:-}" ]]; then
      args+=(--token "$HF_TOKEN")
    fi
    huggingface-cli "${args[@]}"
  fi
}

link_force() {
  local target="$1"
  local link_path="$2"
  rm -rf -- "$link_path"
  ln -s -- "$target" "$link_path"
}

# ---- Downloads ----

LIBERO_ROOT="$DEST/libero"
mkdir -p "$LIBERO_ROOT"

for repo in \
  IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot \
  IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot \
  IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot \
  IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot
do
  retry download_repo "$repo" dataset "$LIBERO_ROOT/${repo##*/}"
done

COCO_DIR="$DEST/LLaVA-OneVision-COCO"
retry download_repo "StarVLA/LLaVA-OneVision-COCO" dataset "$COCO_DIR"

ZIP_PATH="$COCO_DIR/sharegpt4v_coco.zip"
if [[ -f "$ZIP_PATH" ]]; then
  if command -v unzip >/dev/null 2>&1; then
    unzip -o -- "$ZIP_PATH" -d "$COCO_DIR/"
  else
    echo "ERROR: unzip not found, but $ZIP_PATH exists. Please install unzip and re-run." >&2
    exit 1
  fi
else
  echo "WARN: Expected zip not found at: $ZIP_PATH (maybe the dataset structure changed?)" >&2
fi

# ---- Symlinks ----

mkdir -p "$CUR/playground/Datasets"
link_force "$LIBERO_ROOT" "$CUR/playground/Datasets/LEROBOT_LIBERO_DATA"
link_force "$COCO_DIR"    "$CUR/playground/Datasets/LLaVA-OneVision-COCO"

# ---- Copy modality.json ----

MOD_SRC="$CUR/examples/LIBERO/train_files/modality.json"
if [[ ! -f "$MOD_SRC" ]]; then
  echo "ERROR: modality.json not found at: $MOD_SRC" >&2
  exit 1
fi

for ds in \
  libero_10_no_noops_1.0.0_lerobot \
  libero_goal_no_noops_1.0.0_lerobot \
  libero_object_no_noops_1.0.0_lerobot \
  libero_spatial_no_noops_1.0.0_lerobot
do
  mkdir -p "$CUR/playground/Datasets/LEROBOT_LIBERO_DATA/$ds/meta"
  cp -f -- "$MOD_SRC" "$CUR/playground/Datasets/LEROBOT_LIBERO_DATA/$ds/meta/"
done
