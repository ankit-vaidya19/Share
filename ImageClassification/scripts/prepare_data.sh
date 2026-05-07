#!/usr/bin/env bash
# =============================================================================
# prepare_data.sh — Set up a unified data directory for Share ImageClassification
# =============================================================================
#
# Creates a <DATA_DIR>/ folder with symlinks pointing at the actual on-disk
# locations so every training script can use a single --data_root value.
#
# Expected on-disk layout (Backlog):
#   /mnt/d/Backlog/imagenet-r/        ← imagenet-r source (train/ test/ inside)
#   /mnt/d/Backlog/ina/imagenet-a/    ← imagenet-a source (train/ test/ inside)
#   <domainnet_src>/                  ← DomainNet (must be provided by the user)
#
# Usage:
#   ./scripts/prepare_data.sh [DATA_DIR] [IMAGENET_R_SRC] [IMAGENET_A_SRC] [DOMAINNET_SRC]
#
# Defaults:
#   DATA_DIR      = ./data
#   IMAGENET_R_SRC = /mnt/d/Backlog/imagenet-r
#   IMAGENET_A_SRC = /mnt/d/Backlog/ina/imagenet-a
#   DOMAINNET_SRC  = (empty — must be supplied to enable DomainNet)
#
# After running this script, point training scripts at DATA_DIR:
#   ./scripts/run_share_imagenet_r.sh ./outputs <DATA_DIR>
#   ./scripts/run_share_imagenet_a.sh ./outputs <DATA_DIR>
#   ./scripts/run_share_domainnet.sh  ./outputs <DATA_DIR>
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

DATA_DIR="${1:-./data}"
IMAGENET_R_SRC="${2:-/mnt/d/Backlog/imagenet-r}"
IMAGENET_A_SRC="${3:-/mnt/d/Backlog/ina/imagenet-a}"
DOMAINNET_SRC="${4:-}"

mkdir -p "${DATA_DIR}"
DATA_DIR="$(realpath "${DATA_DIR}")"

echo "=== Share ImageClassification — Data Preparation ==="
echo "Target data directory: ${DATA_DIR}"
echo ""

# ── ImageNet-R ────────────────────────────────────────────────────────────────
echo "[1/3] ImageNet-R"
if [ -d "${IMAGENET_R_SRC}" ]; then
    # Verify expected sub-dirs
    if [ ! -d "${IMAGENET_R_SRC}/train" ] || [ ! -d "${IMAGENET_R_SRC}/test" ]; then
        echo "  WARNING: ${IMAGENET_R_SRC} exists but is missing train/ or test/ sub-directory."
    fi
    ln -sfn "$(realpath "${IMAGENET_R_SRC}")" "${DATA_DIR}/imagenet-r"
    TRAIN_CLS=$(ls "${IMAGENET_R_SRC}/train" | wc -l)
    TEST_CLS=$(ls  "${IMAGENET_R_SRC}/test"  | wc -l)
    echo "  Linked: ${IMAGENET_R_SRC} → ${DATA_DIR}/imagenet-r"
    echo "  Classes — train: ${TRAIN_CLS}, test: ${TEST_CLS}"
else
    echo "  SKIPPED: source not found at ${IMAGENET_R_SRC}"
    echo "  Set IMAGENET_R_SRC or pass it as \$2 to point at the correct location."
fi
echo ""

# ── ImageNet-A ────────────────────────────────────────────────────────────────
echo "[2/3] ImageNet-A"
if [ -d "${IMAGENET_A_SRC}" ]; then
    if [ ! -d "${IMAGENET_A_SRC}/train" ] || [ ! -d "${IMAGENET_A_SRC}/test" ]; then
        echo "  WARNING: ${IMAGENET_A_SRC} exists but is missing train/ or test/ sub-directory."
    fi
    ln -sfn "$(realpath "${IMAGENET_A_SRC}")" "${DATA_DIR}/imagenet-a"
    TRAIN_CLS=$(ls "${IMAGENET_A_SRC}/train" | wc -l)
    TEST_CLS=$(ls  "${IMAGENET_A_SRC}/test"  | wc -l)
    echo "  Linked: ${IMAGENET_A_SRC} → ${DATA_DIR}/imagenet-a"
    echo "  Classes — train: ${TRAIN_CLS}, test: ${TEST_CLS}"
else
    echo "  SKIPPED: source not found at ${IMAGENET_A_SRC}"
    echo "  Set IMAGENET_A_SRC or pass it as \$3 to point at the correct location."
fi
echo ""

# ── DomainNet ─────────────────────────────────────────────────────────────────
echo "[3/3] DomainNet"
if [ -n "${DOMAINNET_SRC}" ] && [ -d "${DOMAINNET_SRC}" ]; then
    if [ ! -d "${DOMAINNET_SRC}/train" ] || [ ! -d "${DOMAINNET_SRC}/test" ]; then
        echo "  WARNING: ${DOMAINNET_SRC} exists but is missing train/ or test/ sub-directory."
        echo "  Run split_domainnet.py first to create the train/test split."
    fi
    ln -sfn "$(realpath "${DOMAINNET_SRC}")" "${DATA_DIR}/domainnet"
    echo "  Linked: ${DOMAINNET_SRC} → ${DATA_DIR}/domainnet"
else
    echo "  SKIPPED: DomainNet source not provided."
    echo "  Download from http://ai.bu.edu/M3SDA/ (clipart/infograph/painting/quickdraw/real/sketch),"
    echo "  then run split_domainnet.py to create the ImageFolder-style train/test split:"
    echo "    python scripts/split_domainnet.py --src_dir <raw_domainnet_dir> --out_dir ${DATA_DIR}/domainnet"
fi
echo ""

# ── Summary ───────────────────────────────────────────────────────────────────
echo "=== Done. Use --data_root ${DATA_DIR} in all training scripts. ==="
echo ""
echo "Example:"
echo "  ./scripts/run_share_imagenet_r.sh ./outputs/imagenet_r  ${DATA_DIR}"
echo "  ./scripts/run_share_imagenet_a.sh ./outputs/imagenet_a  ${DATA_DIR}"
echo "  ./scripts/run_share_domainnet.sh  ./outputs/domainnet   ${DATA_DIR}"
