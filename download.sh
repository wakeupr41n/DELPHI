#!/bin/bash
# Download pre-trained DELPHI model checkpoint from Hugging Face
set -e

echo "Downloading DELPHI model checkpoint..."
pip install -q huggingface-hub

mkdir -p checkpoints
huggingface-cli download wakeupR41n/delphi \
  --local-dir checkpoints/ \
  --local-dir-use-symlinks False

echo "Done. Model saved to checkpoints/"
