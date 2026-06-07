#!/bin/bash
# DELPHI quick-start demo
set -e

echo "=== DELPHI Demo ==="

# 1. Download model if not present
if [ ! -f checkpoints/delphi_her2st.pt ]; then
    echo "[1/3] Downloading model..."
    bash download.sh
else
    echo "[1/3] Model found, skipping download."
fi

# 2. Preprocess demo data
echo "[2/3] Preprocessing demo data..."
python scripts/preprocessing/preprocess_her2st.py \
  --data_dir data/demo

# 3. Run inference
echo "[3/3] Running inference..."
python scripts/inference/infer.py \
  --dataset her2st \
  --patient A \
  --ckpt checkpoints/delphi_her2st.pt

echo "=== Demo complete ==="
