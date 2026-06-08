"""
P1b: cSCC Data Preprocessing (Old ST Format)
==============================================
Process cSCC (cutaneous squamous cell carcinoma) ST data
from Ji et al. Cell 2020, GEO GSE144239.

Data format: Old ST (TSV count matrices + JPG H&E + spot selection files)
  - *_stdata.tsv: rows=spots (AxB format), columns=genes, values=raw counts
  - *_spot_data-selection-*.tsv: pixel coordinates for selected spots
  - *.jpg: H&E images

4 patients (P2/P5/P9/P10), 3 replicates each = 12 slides

Usage: python scripts/preprocess_cscc.py
"""

import glob
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from torchvision import transforms

Image.MAX_IMAGE_PIXELS = None  # cSCC H&E images are very large WSIs (~250M pixels)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "cSCC"
SAVE_DIR = PROJECT_ROOT / "data" / "processed" / "cSCC"
HVG_PATH = PROJECT_ROOT / "data" / "raw" / "her2st" / "her_hvg_cut_1000.npy"
UNI2H_DIR = PROJECT_ROOT / "data" / "raw" / "uni2-h"
K_NEIGHBORS = 8

SAVE_DIR.mkdir(parents=True, exist_ok=True)


def load_uni2h_model(device):
    """Load UNI2-h vision foundation model."""
    import timm

    try:
        from timm.layers import SwiGLUPacked
    except ImportError:
        from timm.models.layers import SwiGLUPacked

    weights_path = UNI2H_DIR / "pytorch_model.bin"
    if not weights_path.exists():
        raise FileNotFoundError(f"UNI2-h weights not found at {weights_path}")

    timm_kwargs = {
        "model_name": "vit_giant_patch14_224",
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": SwiGLUPacked,
        "act_layer": torch.nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }
    model = timm.create_model(pretrained=False, **timm_kwargs)
    state_dict = torch.load(str(weights_path), map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()

    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return model, normalize


def extract_features(model, normalize, image, coords, device, batch_size=32):
    """Extract UNI2-h features for spots given pixel coordinates."""
    nbrs = NearestNeighbors(n_neighbors=min(5, len(coords))).fit(coords)
    distances, _ = nbrs.kneighbors(coords)
    avg_spacing = np.median(distances[:, 1]) if distances.shape[1] > 1 else 100
    crop_size = max(16, int(avg_spacing * 1.5))
    logger.info(f"    Dynamic zoom: avg_spacing={avg_spacing:.1f}px, crop_size={crop_size}px")

    resize_transform = transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ]
    )

    patches = []
    half = crop_size // 2
    w, h = image.size

    for x, y in coords:
        x, y = int(x), int(y)
        left, upper = x - half, y - half
        right, lower = x + half, y + half

        if left < 0 or upper < 0 or right > w or lower > h:
            padded = Image.new("RGB", (crop_size, crop_size), (255, 255, 255))
            c_left, c_upper = max(0, left), max(0, upper)
            c_right, c_lower = min(w, right), min(h, lower)
            crop = image.crop((c_left, c_upper, c_right, c_lower))
            paste_x, paste_y = max(0, -left), max(0, -upper)
            padded.paste(crop, (paste_x, paste_y))
            patches.append(resize_transform(padded))
        else:
            patches.append(resize_transform(image.crop((left, upper, right, lower))))

    patches_tensor = torch.stack(patches).to(device)
    features = []
    with torch.no_grad():
        for i in range(0, len(patches_tensor), batch_size):
            batch = patches_tensor[i : i + batch_size]
            features.append(model(batch).cpu())

    return torch.cat(features, dim=0)


def process_sample(stdata_path, spots_path, img_path, target_genes, model, normalize, device):
    """Process a single cSCC ST sample."""
    sample_name = Path(stdata_path).stem.replace("_stdata", "")
    # Extract patient and rep info from filename (e.g., GSM4284316_P2_ST_rep1)
    parts = sample_name.split("_")
    patient_id = None
    for p in parts:
        if p.startswith("P") and p[1:].isdigit():
            patient_id = p
            break
    sid = "_".join([p for p in parts if p.startswith("P") or p.startswith("rep") or p == "ST"])
    logger.info(f"\n  Processing {sid} (patient={patient_id})...")

    # 1. Load count data
    stdata = pd.read_csv(stdata_path, sep="\t", index_col=0)
    logger.info(f"    Raw data: {stdata.shape[0]} spots x {stdata.shape[1]} genes")

    # 2. Load spot coordinates
    spots = pd.read_csv(spots_path, sep="\t")
    # spots has columns: x, y, new_x, new_y, pixel_x, pixel_y
    # Index format: AxB array coordinates → need to map to stdata index
    # Create spot ID from x,y (array coords match stdata index format "AxB")
    spot_ids = [f"{int(row['x'])}x{int(row['y'])}" for _, row in spots.iterrows()]
    spots["spot_id"] = spot_ids
    pixel_coords = spots[["pixel_x", "pixel_y"]].values

    # 3. Match spots between stdata and spot selection
    common = set(stdata.index) & set(spot_ids)
    if len(common) < 50:
        logger.warning(f"    Too few common spots ({len(common)}), skipping")
        return None
    logger.info(f"    Common spots: {len(common)}")

    # Filter and align
    spots_filtered = spots[spots["spot_id"].isin(common)].set_index("spot_id")
    common_ordered = [s for s in stdata.index if s in common]
    stdata_filtered = stdata.loc[common_ordered]
    pixel_coords = spots_filtered.loc[common_ordered, ["pixel_x", "pixel_y"]].values

    # 4. Gene intersection with HER2ST HVG
    overlap_genes = sorted(set(stdata_filtered.columns) & set(target_genes))
    logger.info(f"    Gene overlap: {len(overlap_genes)}/{len(target_genes)}")
    if len(overlap_genes) < 200:
        logger.warning("    Gene overlap too small, skipping")
        return None

    expr_raw = stdata_filtered[overlap_genes].values.astype(np.float32)

    # 5. Normalize: CPM + ln(x+1)
    lib_size = expr_raw.sum(axis=1, keepdims=True)
    lib_size[lib_size == 0] = 1
    expr_cpm = expr_raw / lib_size * 1e6
    expr = np.log1p(expr_cpm).astype(np.float32)
    logger.info(f"    After normalization: {expr.shape}")

    # 6. Extract UNI2-h features
    image = Image.open(img_path).convert("RGB")
    logger.info(f"    Image size: {image.size}")
    features = extract_features(model, normalize, image, pixel_coords, device)

    # 7. Build kNN graph
    pos_norm = pixel_coords.copy().astype(np.float32)
    for dim in range(2):
        rng = pos_norm[:, dim].max() - pos_norm[:, dim].min()
        if rng > 0:
            pos_norm[:, dim] = (pos_norm[:, dim] - pos_norm[:, dim].min()) / rng

    nbrs = NearestNeighbors(n_neighbors=K_NEIGHBORS + 1).fit(pos_norm)
    _, indices = nbrs.kneighbors(pos_norm)
    src, dst = [], []
    for i in range(len(pos_norm)):
        for j in indices[i, 1:]:
            src.extend([i, j])
            dst.extend([j, i])
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # 8. Build PyG Data
    data = Data(
        x=features,
        y=torch.tensor(expr, dtype=torch.float32),
        pos=torch.tensor(pixel_coords.astype(np.float32), dtype=torch.float32),
        edge_index=edge_index,
    )
    data.barcodes = common_ordered
    data.gene_symbols = overlap_genes
    data.sid = sid
    data.patient_id = patient_id

    return data


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load target genes
    target_genes = list(np.load(str(HVG_PATH), allow_pickle=True))
    logger.info(f"Target HVG: {len(target_genes)} genes")

    # Load UNI2-h
    logger.info("Loading UNI2-h model...")
    model, normalize = load_uni2h_model(device)

    # Find all samples
    stdata_files = sorted(glob.glob(str(RAW_DIR / "*_stdata.tsv")))
    logger.info(f"Found {len(stdata_files)} samples")

    processed = 0
    for stdata_path in stdata_files:
        base = Path(stdata_path).stem.replace("_stdata", "")
        # Find matching files
        spots_candidates = glob.glob(
            str(RAW_DIR / f"*spot_data-selection-{base.split('_', 1)[1]}*")
        )
        img_candidates = glob.glob(str(RAW_DIR / f"{base}.jpg"))

        if not spots_candidates:
            # Try alternative naming
            parts = base.split("_")
            sample_part = "_".join(parts[1:])  # Remove GSM prefix
            spots_candidates = glob.glob(str(RAW_DIR / f"*spot*{sample_part}*"))
        if not img_candidates:
            img_candidates = glob.glob(str(RAW_DIR / f"{base}*.jpg"))

        if not spots_candidates or not img_candidates:
            logger.warning(
                f"  Missing files for {base}: spots={len(spots_candidates)}, img={len(img_candidates)}"
            )
            continue

        data = process_sample(
            stdata_path,
            spots_candidates[0],
            img_candidates[0],
            target_genes,
            model,
            normalize,
            device,
        )
        if data is not None:
            save_path = SAVE_DIR / f"{data.sid}.pt"
            torch.save(data, str(save_path))
            logger.info(
                f"    Saved: {save_path} ({data.x.shape[0]} spots, {data.y.shape[1]} genes)"
            )
            processed += 1

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Processed {processed}/{len(stdata_files)} samples -> {SAVE_DIR}")

    # Summary
    pt_files = sorted(SAVE_DIR.glob("*.pt"))
    if pt_files:
        sample = torch.load(str(pt_files[0]), map_location="cpu", weights_only=False)
        logger.info(f"Gene count: {sample.y.shape[1]}")
        patients = set()
        for f in pt_files:
            d = torch.load(str(f), map_location="cpu", weights_only=False)
            patients.add(d.patient_id)
        logger.info(f"Patients: {sorted(patients)} ({len(patients)} total)")
        logger.info(f"Slides: {len(pt_files)} total")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
