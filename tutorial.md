# DELPHI Tutorial

## 1. Setup

```bash
git clone https://github.com/wakeupr41n/DELPHI.git
cd DELPHI
pip install -r requirements.txt
bash download.sh
```

## 2. Preprocess Data

```bash
python scripts/preprocessing/preprocess_her2st.py --data_dir data/her2st
```

This reads raw H&E images and spatial transcriptomics counts,
extracts UNI2-h features, and saves PyG Data objects.

## 3. Train

```bash
python scripts/training/train.py --dataset her2st --mode loocv --patient A
```

Leave-one-out cross-validation. The model is evaluated on held-out
patient A after training on patients B-H.

## 4. Inference

```bash
python scripts/inference/infer.py \
  --dataset her2st \
  --patient A \
  --ckpt checkpoints/delphi_her2st.pt
```

Outputs per-spot predictions with epistemic and aleatoric uncertainty.

## 5. Fine-tune on a New Dataset

```bash
python scripts/training/finetune.py --dataset cscc
python scripts/inference/infer_cscc_zeroshot.py
```
