"""Preprocess HER2ST raw data → PyG Data objects with UNI2-h features.

Reads HER2ST raw counts, spot coordinates, and H&E patches; extracts UNI2-h
(1536-dim) features per spot; builds KNN spatial graph; saves torch_geometric
Data objects to data/processed/processed_data_her2st_uni_fixed/.

Output: 36 .pt files (A1.pt .. H6.pt), each a torch_geometric Data with:
  x [N, 1536]  UNI2-h embeddings
  pos [N, 2]    pixel coordinates
  y [N, 785]    log1p-CPM-normalized expression (HER2 785-gene panel)
  edge_index [2, E]  KNN spatial graph

Usage:
  python scripts/preprocess_her2st.py
    --raw_root data/raw/her2st/data
    --hvg_path data/raw/her2st/her_hvg_cut_1000.npy
    --save_dir data/processed/processed_data_her2st_uni_fixed
    [--hf_token $HF_TOKEN]   # needed to download UNI2-h from HuggingFace

Or set env var HF_TOKEN (recommended for security).
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from torchvision import transforms
from tqdm import tqdm

# ── UNI2-h model ID (timm hub) ──
MODEL_ID = "hf-hub:MahmoodLab/UNI2-h"


class Preprocessor:
    def __init__(self, raw_root, save_dir, hvg_path, hf_token=None):
        self.raw_root = Path(raw_root)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Load HVG gene list (785 genes for HER2ST)
        self.target_genes = np.load(hvg_path, allow_pickle=True)
        if self.target_genes.dtype.kind in {"S", "U"}:
            self.target_genes = [str(g) for g in self.target_genes]

        # 2. Load UNI2-h
        print(f">>> Loading UNI2-h from {MODEL_ID}")
        if hf_token:
            from huggingface_hub import login

            login(token=hf_token)

        import timm
        import timm.layers

        timm_kwargs = dict(
            img_size=224,
            patch_size=14,
            depth=24,
            num_heads=24,
            init_values=1e-5,
            embed_dim=1536,
            mlp_ratio=5.33334,
            num_classes=0,
            no_embed_class=True,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
            reg_tokens=8,
            dynamic_img_size=True,
        )
        self.model = timm.create_model(MODEL_ID, pretrained=True, **timm_kwargs).to(self.device)
        self.model.eval()

        dummy = torch.randn(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            out = self.model(dummy)
        print(f"  UNI2-h loaded. Feature dim: {out.shape[1]}")

        self.transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def process_image_patches(self, img_path, coords):
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        patches = []
        crop_size, half = 224, 112
        for x, y in coords:
            x, y = int(x), int(y)
            left, upper = x - half, y - half
            if left < 0 or upper < 0 or x + half > w or y + half > h:
                padded = Image.new("RGB", (crop_size, crop_size), (255, 255, 255))
                c_left, c_upper = max(0, left), max(0, upper)
                c_right, c_lower = min(w, x + half), min(h, y + half)
                crop = image.crop((c_left, c_upper, c_right, c_lower))
                paste_x = max(0, -left)
                paste_y = max(0, -upper)
                padded.paste(crop, (paste_x, paste_y))
                patches.append(self.transform(padded))
            else:
                patches.append(self.transform(image.crop((left, upper, x + half, y + half))))
        return torch.stack(patches)

    def build_graph(self, pos_norm, k=8):
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(pos_norm)
        _, indices = nbrs.kneighbors(pos_norm)
        src, tgt = [], []
        for i in range(len(pos_norm)):
            neighbors = indices[i, 1:]
            src.extend([i] * len(neighbors))
            tgt.extend(neighbors)
        return torch.tensor([src, tgt], dtype=torch.long)

    def run(self):
        cnt_files = sorted(glob.glob(os.path.join(self.raw_root, "ST-cnts", "*.tsv")))
        sample_ids = [os.path.basename(f).replace(".tsv", "") for f in cnt_files]
        print(f"[Preprocessor] Found {len(sample_ids)} samples.")
        for sid in tqdm(sample_ids):
            cnts_path = os.path.join(self.raw_root, "ST-cnts", f"{sid}.tsv")
            meta_path = os.path.join(self.raw_root, "ST-spotfiles", f"{sid}_selection.tsv")
            img_path = os.path.join(self.raw_root, "ST-imgs", sid[0], sid, f"{sid}.jpg")
            if not os.path.exists(meta_path):
                meta_path = os.path.join(self.raw_root, "ST-spotfiles", f"{sid}_labeled.tsv")
            if not (
                os.path.exists(cnts_path) and os.path.exists(meta_path) and os.path.exists(img_path)
            ):
                print(f"  Skip {sid}: files missing")
                continue
            try:
                expr_df = pd.read_csv(cnts_path, sep="\t", index_col=0)
                meta_df = pd.read_csv(meta_path, sep="\t")
                if "x" in meta_df.columns and "y" in meta_df.columns:
                    meta_df.index = (
                        meta_df["x"].astype(int).astype(str)
                        + "x"
                        + meta_df["y"].astype(int).astype(str)
                    )
                else:
                    print(f"  {sid}: meta missing x/y")
                    continue
                common_ids = expr_df.index.intersection(meta_df.index)
                if len(common_ids) == 0:
                    print(f"  {sid}: no spot intersection")
                    continue
                expr_df = expr_df.loc[common_ids]
                meta_df = meta_df.loc[common_ids]
                # CPM + log1p
                expr_df = expr_df.reindex(columns=self.target_genes).fillna(0.0)
                adata = sc.AnnData(expr_df)
                sc.pp.normalize_total(adata, target_sum=1e6)
                sc.pp.log1p(adata)
                y = torch.tensor(adata.X, dtype=torch.float)
                # UNI2-h features
                pixel_coords = meta_df[["pixel_x", "pixel_y"]].values
                patches = self.process_image_patches(img_path, pixel_coords).to(self.device)
                features = []
                with torch.no_grad():
                    for i in range(0, len(patches), 32):
                        features.append(self.model(patches[i : i + 32]).cpu())
                x = torch.cat(features, dim=0)
                # KNN graph
                pos = torch.tensor(pixel_coords, dtype=torch.float)
                pos_norm = (pos - pos.min(0)[0]) / (pos.max(0)[0] - pos.min(0)[0] + 1e-6)
                edge_index = self.build_graph(pos_norm.numpy())
                # save
                data = Data(x=x, pos=pos, y=y, edge_index=edge_index, sid=sid)
                torch.save(data, self.save_dir / f"{sid}.pt")
            except Exception as e:
                print(f"  Error {sid}: {e}")


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[2]  # repo root (from scripts/preprocessing/)
    ap = argparse.ArgumentParser(description="HER2ST preprocessing → UNI2-h PyG Data")
    ap.add_argument("--raw_root", default=str(ROOT / "data/raw/her2st/data"))
    ap.add_argument("--hvg_path", default=str(ROOT / "data/raw/her2st/her_hvg_cut_1000.npy"))
    ap.add_argument(
        "--save_dir", default=str(ROOT / "data/processed/processed_data_her2st_uni_fixed")
    )
    ap.add_argument("--hf_token", default=os.environ.get("HF_TOKEN", ""))
    args = ap.parse_args()
    print(f"HER2ST preprocessing: {args.raw_root} → {args.save_dir}")
    preprocessor = Preprocessor(
        raw_root=args.raw_root,
        save_dir=args.save_dir,
        hvg_path=args.hvg_path,
        hf_token=args.hf_token or None,
    )
    preprocessor.run()
