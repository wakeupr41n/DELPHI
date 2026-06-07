"""
Preprocess TCGA-BRCA H&E slides for DELPHI Section 6.

For each SVS file:
  1. Load with OpenSlide
  2. Detect tissue regions (Otsu threshold on gray thumbnail)
  3. Grid tile at 256x256 (non-overlapping, skip background)
  4. Extract UNI2-h features (frozen, GPU)
  5. Build k=8 KNN spatial graph
  6. Save as .pt (x, pos, edge_index) — no y (no ground truth)

Usage: python scripts/preprocess_tcga_wsi.py [--device cuda] [--tile-size 256]
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from torchvision import transforms

Image.MAX_IMAGE_PIXELS = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_SLIDE_DIR = PROJECT_ROOT / "data" / "external" / "tcga_brca" / "slides"
DEFAULT_SAVE_DIR  = PROJECT_ROOT / "data" / "processed" / "TCGA_BRCA"
UNI2H_DIR = PROJECT_ROOT / "data" / "raw" / "uni2-h"
K_NEIGHBORS = 8
TISSUE_THRESH = 0.15  # fraction of non-white pixels to count as tissue


def load_uni2h_model(device):
    """Load UNI2-h ViT-giant model (same as preprocess_crc.py)."""
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
        "img_size": 224, "patch_size": 14,
        "depth": 24, "num_heads": 24, "init_values": 1e-5, "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2, "num_classes": 0, "no_embed_class": True,
        "mlp_layer": SwiGLUPacked, "act_layer": torch.nn.SiLU,
        "reg_tokens": 8, "dynamic_img_size": True,
    }
    model = timm.create_model(pretrained=False, **timm_kwargs)
    state_dict = torch.load(str(weights_path), map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return model, normalize


def detect_tissue_tiles(slide, tile_size, thumb_scale=32):
    """Detect tissue-containing tiles using Otsu on thumbnail."""
    w, h = slide.dimensions
    thumb_w, thumb_h = w // thumb_scale, h // thumb_scale
    thumb = slide.get_thumbnail((thumb_w, thumb_h)).convert("L")
    thumb_arr = np.array(thumb)

    # Otsu threshold
    from skimage.filters import threshold_otsu
    try:
        otsu = threshold_otsu(thumb_arr)
    except ValueError:
        otsu = 200
    tissue_mask = thumb_arr < otsu

    # Grid tiles
    tiles = []
    n_x = w // tile_size
    n_y = h // tile_size
    for ix in range(n_x):
        for iy in range(n_y):
            # Check thumbnail region
            tx0 = ix * tile_size // thumb_scale
            ty0 = iy * tile_size // thumb_scale
            tx1 = min(tx0 + tile_size // thumb_scale, thumb_w)
            ty1 = min(ty0 + tile_size // thumb_scale, thumb_h)
            region = tissue_mask[ty0:ty1, tx0:tx1]
            if region.mean() > TISSUE_THRESH:
                cx = ix * tile_size + tile_size // 2
                cy = iy * tile_size + tile_size // 2
                tiles.append((ix * tile_size, iy * tile_size, cx, cy))

    return tiles


def extract_features_from_tiles(model, normalize, slide_path, tiles, tile_size,
                                 device, batch_size=256, n_workers=8):
    """Extract UNI2-h features with multi-threaded I/O + fp16 inference."""
    from concurrent.futures import ThreadPoolExecutor

    import openslide

    resize_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])

    # Per-worker slide handle for thread-safe reads
    thread_local = __import__("threading").local()
    def _get_slide():
        if not hasattr(thread_local, "slide"):
            thread_local.slide = openslide.OpenSlide(slide_path)
        return thread_local.slide

    def _read_tile(args):
        x0, y0, cx, cy = args
        sl = _get_slide()
        region = sl.read_region((x0, y0), 0, (tile_size, tile_size)).convert("RGB")
        return resize_transform(region), (cx, cy)

    all_features = []
    coords = []
    use_amp = device.type == "cuda"

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for batch_start in range(0, len(tiles), batch_size):
            batch_tiles = tiles[batch_start:batch_start + batch_size]
            results = list(ex.map(_read_tile, batch_tiles))
            patches = [r[0] for r in results]
            coords.extend([r[1] for r in results])

            batch_tensor = torch.stack(patches).to(device)
            with torch.no_grad():
                if use_amp:
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        feats = model(batch_tensor).float().cpu()
                else:
                    feats = model(batch_tensor).cpu()
            all_features.append(feats)

            if batch_start % (batch_size * 5) == 0:
                log.info(f"    {batch_start + len(batch_tiles)}/{len(tiles)} tiles processed")

    features = torch.cat(all_features, dim=0)
    coords = np.array(coords, dtype=np.float32)
    return features, coords


def process_slide(svs_path, model, normalize, device, tile_size):
    """Process one WSI into a .pt file."""
    import openslide

    name = svs_path.stem
    log.info(f"\n{'='*50}")
    log.info(f"Processing: {name}")

    slide = openslide.OpenSlide(str(svs_path))
    w, h = slide.dimensions
    log.info(f"  Dimensions: {w} x {h}")

    # Detect tissue tiles
    tiles = detect_tissue_tiles(slide, tile_size)
    log.info(f"  Tissue tiles: {len(tiles)} (tile_size={tile_size})")

    if len(tiles) < 10:
        log.warning(f"  Too few tiles ({len(tiles)}), skipping")
        slide.close()
        return None

    # GHIST-faithful: use ALL tissue tiles (no cap). Pseudobulk = mean over spots.
    log.info(f"  Using all {len(tiles)} tiles (GHIST-faithful, no cap)")

    # Extract features
    log.info("  Extracting UNI2-h features...")
    slide.close()  # close main handle; workers create their own
    features, coords = extract_features_from_tiles(
        model, normalize, str(svs_path), tiles, tile_size, device
    )
    log.info(f"  Features: {features.shape}")

    # Normalize coordinates to [0, 1]
    pos_norm = coords.copy()
    for dim in range(2):
        rng = pos_norm[:, dim].max() - pos_norm[:, dim].min()
        if rng > 0:
            pos_norm[:, dim] = (pos_norm[:, dim] - pos_norm[:, dim].min()) / rng

    # Build KNN graph
    nbrs = NearestNeighbors(n_neighbors=K_NEIGHBORS + 1).fit(pos_norm)
    _, indices = nbrs.kneighbors(pos_norm)
    src, dst = [], []
    for i in range(len(pos_norm)):
        for j in indices[i, 1:]:
            src.extend([i, j])
            dst.extend([j, i])
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # Build PyG Data (no y — no ground truth)
    data = Data(
        x=features,
        pos=torch.tensor(pos_norm, dtype=torch.float32),
        edge_index=edge_index,
    )
    # Store raw pixel coordinates for visualization
    data.pixel_coords = torch.tensor(coords, dtype=torch.float32)
    data.slide_dims = torch.tensor([w, h], dtype=torch.long)

    log.info(f"  Output: x={data.x.shape}, pos={data.pos.shape}, "
             f"edges={data.edge_index.shape[1]}")
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--slide-dir", default=str(DEFAULT_SLIDE_DIR),
                        help="Directory containing SVS files (default: tcga_brca/slides)")
    parser.add_argument("--save-dir", default=str(DEFAULT_SAVE_DIR),
                        help="Output directory for .pt files (default: processed/TCGA_BRCA)")
    args = parser.parse_args()

    slide_dir = Path(args.slide_dir)
    save_dir  = Path(args.save_dir)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    log.info(f"Slide dir: {slide_dir}")
    log.info(f"Save dir:  {save_dir}")

    svs_files = sorted(slide_dir.glob("*.svs"))
    log.info(f"Found {len(svs_files)} SVS files")
    if not svs_files:
        log.error(f"No SVS files found in {slide_dir}. "
                  "Run download_tcga_her2pos.py first.")
        return

    # Load UNI2-h
    log.info("Loading UNI2-h model...")
    model, normalize = load_uni2h_model(device)

    # Process
    save_dir.mkdir(parents=True, exist_ok=True)
    processed = 0
    for svs_path in svs_files:
        save_path = save_dir / f"{svs_path.stem}.pt"
        if save_path.exists():
            log.info(f"Already processed: {save_path.name}")
            processed += 1
            continue

        data = process_slide(svs_path, model, normalize, device, args.tile_size)
        if data is not None:
            torch.save(data, str(save_path))
            log.info(f"  Saved: {save_path.name}")
            processed += 1

    # Summary
    log.info(f"\n{'='*60}")
    log.info(f"Processed {processed}/{len(svs_files)} slides → {save_dir}")
    for pt in sorted(save_dir.glob("*.pt")):
        d = torch.load(str(pt), map_location="cpu", weights_only=False)
        log.info(f"  {pt.stem}: {d.x.shape[0]} tiles, x={d.x.shape}")


if __name__ == "__main__":
    main()
