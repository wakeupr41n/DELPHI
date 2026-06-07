"""HEST-1k PRAD preprocessing for cross-organ zero-shot eval.

Phase 3 of the v3-canon reset plan.

Pipeline:
  1. Pull HEST-1k metadata (HEST_v1_1_0.csv) from HuggingFace MahmoodLab/hest.
  2. Filter to oncotree_code == 'PRAD' (Erickson 2022, 23 FF Visium samples,
     ~10 GB total).
  3. Download matching .h5ad (st/) + WSI (wsis/) via snapshot_download with
     allow_patterns=[f"*{id}[_.]**"].
  4. For each sample:
       a. Load AnnData; obs has spatial pixel coords; X is raw counts.
       b. Intersect var_names with the 785-HVG panel from HER2ST.
       c. Pad missing genes with zero -> 785-dim y.
       d. Log-normalize counts: y = log1p(y * 1e4 / y.sum(1, keepdims=True)).
       e. Open WSI; crop 1.5x avg-spacing patches at each spot center;
          resize 224x224; UNI2-h forward -> 1536-dim features.
       f. Build k=8 KNN spatial graph on pixel coords.
       g. Save Data(x, pos, y, edge_index) to data/processed/HEST_PRAD/<id>.pt.
  5. Write summary JSON listing per-sample (#spots, gene_coverage_785).

Usage:
    # Default endpoint (huggingface.co) -- requires direct access:
    python scripts/preprocess_hest_prad.py [--device cuda] [--max-samples N]

    # If huggingface.co is blocked (e.g. China), use the hf-mirror endpoint:
    HF_ENDPOINT=https://hf-mirror.com python scripts/preprocess_hest_prad.py
    # ...or set --hf-mirror to flip it inside the script.

Notes:
  - Requires huggingface_hub login if rate-limited; usually works anonymously.
  - WSI files are pyramidal Generic TIFF; openslide is preferred but tifffile
    is the lighter dep already in the environment.
  - The 785 panel was selected on HER2ST so coverage on PRAD is lower than the
    HER2ST/cSCC ~95%; expect ~70-85%.
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "hest_prad"
SAVE_DIR = PROJECT_ROOT / "data" / "processed" / "HEST_PRAD"
HVG_PATH = PROJECT_ROOT / "data" / "processed" / "gene_names_785.npy"
UNI2H_DIR = PROJECT_ROOT / "data" / "raw" / "uni2-h"
META_FILE = "HEST_v1_1_0.csv"
HEST_REPO = "MahmoodLab/hest"
K_NEIGHBORS = 8

RAW_DIR.mkdir(parents=True, exist_ok=True)
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Metadata + selective download
# ---------------------------------------------------------------------------


def fetch_metadata():
    """Download HEST metadata CSV (small, ~1MB).

    Tries the latest known version names; HF_TOKEN env var must be set so
    that the gated repo grants access.
    """
    from huggingface_hub import hf_hub_download
    # newest first; v1_3_0 added Visium HD samples in 2026-01
    candidates = ["HEST_v1_3_0.csv", "HEST_v1_2_0.csv", "HEST_v1_1_0.csv"]
    for fname in candidates:
        try:
            p = hf_hub_download(repo_id=HEST_REPO, filename=fname,
                                repo_type="dataset", local_dir=str(RAW_DIR))
            logger.info(f"Metadata downloaded: {p}")
            return pd.read_csv(p)
        except Exception as e:
            logger.debug(f"  {fname} not found: {e}")
            continue
    raise RuntimeError("HEST metadata CSV not found on hub. "
                       "Check HF_TOKEN and access to MahmoodLab/hest.")


def select_prad_ids(meta: pd.DataFrame) -> list:
    """Return prostate-cancer FF Visium sample IDs.

    HEST-1k labels Erickson-2022 / Mendeley prostate samples with
    oncotree_code == NaN; the reliable filter is:
        organ == 'Prostate'  AND  disease_state == 'Cancer'
        AND  preservation_method == 'Fresh Frozen'
        AND  st_technology starts with 'Visium'.

    On HEST v1_3_0 this returns ~12 samples spanning MEND139-145, MEND59-62,
    and NCBI793 (the latter is the only one with oncotree_code='PRAD').
    """
    sub = meta[
        meta["organ"].astype(str).str.contains("rostate", case=False, na=False) &
        (meta["disease_state"].astype(str).str.lower() == "cancer") &
        (meta["preservation_method"].astype(str).str.lower() == "fresh frozen") &
        meta["st_technology"].astype(str).str.startswith("Visium")
    ]
    ids = sorted(sub["id"].astype(str).tolist())
    logger.info(f"PRAD FF-Visium samples: {len(ids)} -> {ids}")
    return ids


def download_subset(ids: list):
    """snapshot_download with allow_patterns to grab only the chosen IDs."""
    from huggingface_hub import snapshot_download
    patterns = []
    for sid in ids:
        # st/<id>.h5ad and wsis/<id>.tif (and any related extension)
        patterns.extend([f"st/{sid}*", f"wsis/{sid}*"])
    logger.info(f"Downloading {len(ids)} PRAD samples ({len(patterns)} patterns)")
    snapshot_download(
        repo_id=HEST_REPO, repo_type="dataset",
        allow_patterns=patterns, local_dir=str(RAW_DIR),
    )
    logger.info("Download complete")


# ---------------------------------------------------------------------------
# 2. UNI2-h feature extractor
# ---------------------------------------------------------------------------


def load_uni2h(device):
    import timm
    try:
        from timm.layers import SwiGLUPacked
    except ImportError:
        from timm.models.layers import SwiGLUPacked
    weights = UNI2H_DIR / "pytorch_model.bin"
    if not weights.exists():
        raise FileNotFoundError(f"UNI2-h weights missing: {weights}")
    kwargs = dict(model_name="vit_giant_patch14_224", img_size=224, patch_size=14,
                  depth=24, num_heads=24, init_values=1e-5, embed_dim=1536,
                  mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True,
                  mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU,
                  reg_tokens=8, dynamic_img_size=True)
    model = timm.create_model(pretrained=False, **kwargs)
    sd = torch.load(str(weights), map_location="cpu")
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    norm = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225))
    return model, norm


def extract_features(model, normalize, slide, coords, device, batch_size=32):
    """Extract UNI2-h features for the given pixel coords using OpenSlide.
    `slide` is an openslide.OpenSlide handle.
    """
    nbrs = NearestNeighbors(n_neighbors=min(5, len(coords))).fit(coords)
    distances, _ = nbrs.kneighbors(coords)
    avg_spacing = np.median(distances[:, 1]) if distances.shape[1] > 1 else 100
    crop = max(16, int(avg_spacing * 1.5))
    logger.info(f"    avg_spacing={avg_spacing:.1f}px crop={crop}px")

    resize = transforms.Compose([
        transforms.Resize((224, 224),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])

    features = []
    half = crop // 2
    feats_buf = []
    with torch.no_grad():
        for i, (x, y) in enumerate(coords):
            x0, y0 = max(0, int(x) - half), max(0, int(y) - half)
            tile = slide.read_region((x0, y0), 0, (crop, crop)).convert("RGB")
            t = normalize(resize(tile)).unsqueeze(0)
            feats_buf.append(t)
            if len(feats_buf) >= batch_size or i == len(coords) - 1:
                batch = torch.cat(feats_buf, dim=0).to(device)
                feats = model(batch)
                features.append(feats.cpu())
                feats_buf = []
    return torch.cat(features, dim=0)  # [N, 1536]


# ---------------------------------------------------------------------------
# 3. Per-slide pipeline
# ---------------------------------------------------------------------------


def process_one(sample_id: str, hvg_panel: np.ndarray,
                model, normalize, device) -> dict | None:
    import anndata as ad
    import openslide
    from torch_geometric.data import Data

    h5ad_path = RAW_DIR / "st" / f"{sample_id}.h5ad"
    if not h5ad_path.exists():
        logger.warning(f"  {sample_id}: .h5ad missing")
        return None

    # WSI candidate names -- HEST puts them under wsis/
    wsi_dir = RAW_DIR / "wsis"
    wsi_candidates = sorted(wsi_dir.glob(f"{sample_id}.*"))
    if not wsi_candidates:
        logger.warning(f"  {sample_id}: WSI missing")
        return None
    wsi_path = wsi_candidates[0]

    logger.info(f"  loading {sample_id}")
    adata = ad.read_h5ad(h5ad_path)

    # Pixel coordinates: HEST stores in obsm['spatial'] = [N, 2] in WSI pixels
    if "spatial" not in adata.obsm:
        logger.warning(f"  {sample_id}: no obsm['spatial']")
        return None
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)

    # Genes -- intersect with 785 HVG
    hvg_idx = []
    var_names = list(adata.var_names.astype(str))
    var_lookup = {g: i for i, g in enumerate(var_names)}
    miss = 0
    for g in hvg_panel:
        if g in var_lookup:
            hvg_idx.append(var_lookup[g])
        else:
            hvg_idx.append(-1)
            miss += 1
    coverage = (len(hvg_panel) - miss) / len(hvg_panel)
    logger.info(f"    spots={len(coords)} gene_coverage={coverage:.2%}")

    # Build expression matrix [N, 785]; pad missing with 0
    X = adata.X  # noqa: N806
    if hasattr(X, "toarray"):
        X = X.toarray()  # noqa: N806
    X = np.asarray(X, dtype=np.float32)  # noqa: N806

    expr = np.zeros((X.shape[0], len(hvg_panel)), dtype=np.float32)
    valid = [i for i, idx in enumerate(hvg_idx) if idx >= 0]
    src = [hvg_idx[i] for i in valid]
    expr[:, valid] = X[:, src]

    # log-normalize using the original per-spot total (not the truncated 785)
    total = X.sum(axis=1, keepdims=True) + 1e-8
    expr = np.log1p(expr * 1e4 / total)
    y = torch.from_numpy(expr.astype(np.float32))

    # KNN edge_index
    nbrs = NearestNeighbors(n_neighbors=K_NEIGHBORS + 1).fit(coords)
    _, neigh = nbrs.kneighbors(coords)
    src_idx, dst_idx = [], []
    for i, ne in enumerate(neigh):
        for j in ne[1:]:  # skip self
            src_idx.append(i)
            dst_idx.append(j)
    edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)

    # UNI2-h features from WSI
    slide = openslide.OpenSlide(str(wsi_path))
    feats = extract_features(model, normalize, slide, coords, device)
    slide.close()

    pos = torch.from_numpy(coords.astype(np.float32))
    data = Data(x=feats, pos=pos, y=y, edge_index=edge_index)

    save_path = SAVE_DIR / f"{sample_id}.pt"
    torch.save(data, save_path)
    logger.info(f"    saved {save_path.name}  spots={len(coords)} "
                f"feat={tuple(feats.shape)} cov={coverage:.2%}")

    return {"id": sample_id, "n_spots": int(len(coords)),
            "gene_coverage": float(coverage),
            "feat_shape": list(feats.shape)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="cap for smoke testing")
    parser.add_argument("--skip-download", action="store_true",
                        help="re-use already-downloaded raw files")
    parser.add_argument("--hf-mirror", action="store_true",
                        help="route HuggingFace traffic through hf-mirror.com "
                             "(workaround when huggingface.co is blocked)")
    args = parser.parse_args()

    if args.hf_mirror and not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        logger.info("HF_ENDPOINT -> https://hf-mirror.com")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    logger.info(f"device={device}")
    hvg = np.load(HVG_PATH, allow_pickle=True).astype(str)
    logger.info(f"HVG panel: {len(hvg)} genes")

    if not args.skip_download:
        meta = fetch_metadata()
        ids = select_prad_ids(meta)
        if args.max_samples:
            ids = ids[: args.max_samples]
        download_subset(ids)
    else:
        ids = sorted(p.stem for p in (RAW_DIR / "st").glob("*.h5ad"))
        if args.max_samples:
            ids = ids[: args.max_samples]
        logger.info(f"reuse mode: {len(ids)} ids on disk")

    model, normalize = load_uni2h(device)

    summary = []
    for sid in ids:
        try:
            rec = process_one(sid, hvg, model, normalize, device)
            if rec is not None:
                summary.append(rec)
        except Exception as e:
            logger.exception(f"  {sid} FAILED: {e}")

    out = SAVE_DIR / "_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    logger.info(f"wrote {out}  ({len(summary)} samples)")


if __name__ == "__main__":
    main()
