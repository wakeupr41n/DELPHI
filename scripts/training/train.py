"""Generic DELPHI training for any LOOCV fold or FULL training.

Usage:
  # HER2ST LOOCV fold A
  python scripts/train.py --dataset her2st --patient A --seed 42 \
      --save_dir checkpoints/npd_bll_h/loocv_A_s42

  # HER2ST FULL training
  python scripts/train.py --dataset her2st --mode full --seed 42 \
      --save_dir checkpoints/npd_bll_h/full_s42

  # cSCC LOOCV fold P2 (skin HVG, 171 genes)
  python scripts/train.py --dataset cscc_skinhvg --patient P2 --seed 42 \
      --save_dir checkpoints/npd_bll_cscc/loocv_P2_s42
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root (from scripts/training/)
sys.path.insert(0, str(ROOT))

from src.dataset import Her2stDataset  # noqa: E402
from src.model import DELPHI  # noqa: E402
from src.trainer import Trainer  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


DATASET_DIRS = {
    "her2st": "data/processed/processed_data_her2st_uni_fixed",
    "cscc": "data/processed/cSCC",
    "cscc756": "data/processed/cSCC_756",  # 756-gene benchmark-aligned panel (LOOCV)
    "cscc_skinhvg": "data/processed/cSCC_skinHVG",
    "hest_prad": "data/processed/HEST_PRAD",
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="her2st", choices=list(DATASET_DIRS.keys()))
    ap.add_argument("--mode", default="loocv", choices=["loocv", "full"])
    ap.add_argument(
        "--patient", default=None, help="(loocv only) held-out patient ID, e.g. A or P2"
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument(
        "--pretrain_ckpt",
        default=None,
        help="Warm-start from a pretrained npd checkpoint (e.g. HER2ST FULL); "
        "transfers shape-matching tensors, keeps gene-dependent heads fresh.",
    )
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lambda_pcc", type=float, default=1.0)
    ap.add_argument("--lambda_kl_max", type=float, default=5e-6)
    ap.add_argument("--kl_warmup_epochs", type=int, default=10)
    ap.add_argument("--hidden_dim", type=int, default=384)
    ap.add_argument("--gh", type=int, default=12)
    ap.add_argument("--gw", type=int, default=12)
    ap.add_argument("--knn_k", type=int, default=8)
    ap.add_argument("--n_swin_blocks", type=int, default=4)
    ap.add_argument(
        "--no_bll",
        action="store_true",
        help="Ablation: replace BLL mu head with deterministic Linear",
    )
    ap.add_argument(
        "--no_hetero",
        action="store_true",
        help="Ablation: homoscedastic noise (per-gene constant log_phi)",
    )
    ap.add_argument(
        "--no_pi", action="store_true", help="Ablation: drop zero-inflation -> pure Gaussian NLL"
    )
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device={device} seed={args.seed} dataset={args.dataset} mode={args.mode}")

    data_dir = ROOT / DATASET_DIRS[args.dataset]
    ds = Her2stDataset(root_dir=str(data_dir))  # Her2stDataset works for all (handles list/Data)
    pmap = ds.get_patient_indices()
    logger.info(f"slides: {len(ds)}  patients: {sorted(pmap.keys())}")

    if args.mode == "loocv":
        assert args.patient is not None, "--patient required in loocv mode"
        assert args.patient in pmap, f"unknown patient {args.patient}; have {sorted(pmap.keys())}"
        val_idx = pmap[args.patient]
        train_idx = [i for i in range(len(ds)) if i not in val_idx]
        train_loader = DataLoader(Subset(ds, train_idx), batch_size=1, shuffle=True)
        val_loader = DataLoader(Subset(ds, val_idx), batch_size=1, shuffle=False)
        fold_id = args.patient
        logger.info(
            f"LOOCV held-out {args.patient}: train={len(train_idx)} val={len(val_idx)} slides"
        )
    else:  # full
        train_loader = DataLoader(ds, batch_size=1, shuffle=True)
        val_loader = None
        fold_id = "FULL"

    sample = ds[0]
    num_genes = sample.y.shape[1]
    uni_dim = sample.x.shape[1]
    logger.info(f"num_genes={num_genes}  uni_dim={uni_dim}")

    model = DELPHI(
        uni2h_dim=uni_dim,
        hidden_dim=args.hidden_dim,
        num_genes=num_genes,
        gh=args.gh,
        gw=args.gw,
        knn_k=args.knn_k,
        n_swin_blocks=args.n_swin_blocks,
        use_bll=not args.no_bll,
        use_hetero=not args.no_hetero,
        use_pi=not args.no_pi,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"model params: {n_params / 1e6:.2f} M")

    # Optional warm-start from a pretrained (e.g. HER2ST FULL) npd checkpoint:
    # transfer every parameter whose shape matches (backbone + head bodies);
    # gene-dependent final layers (different num_genes) stay freshly initialised.
    # Mirrors train_v6_cscc.py's HER2ST->cSCC fine-tuning protocol.
    if args.pretrain_ckpt:
        pre = torch.load(args.pretrain_ckpt, map_location="cpu")
        if isinstance(pre, dict) and "model" in pre and "projector.0.weight" not in pre:
            pre = pre["model"]
        msd = model.state_dict()
        transferred, skipped = 0, 0
        for k, v in msd.items():
            if k in pre and pre[k].shape == v.shape:
                msd[k] = pre[k]
                transferred += 1
            else:
                skipped += 1
        model.load_state_dict(msd)
        logger.info(
            f"warm-start from {args.pretrain_ckpt}: transferred {transferred} "
            f"tensors, {skipped} kept fresh (gene-dependent heads)"
        )

    config = {
        "experiment": {"save_dir": args.save_dir},
        "train": {
            "lr": args.lr,
            "weight_decay": 1e-4,
            "epochs": args.epochs,
            "warmup_epochs": 5,
            "grad_clip": 1.0,
            "val_every": 1 if val_loader else 999,
        },
        "loss": {
            "lambda_pcc": args.lambda_pcc,
            "lambda_kl_max": args.lambda_kl_max,
            "kl_warmup_epochs": args.kl_warmup_epochs,
        },
    }
    trainer = Trainer(model, config, device)
    best = trainer.run(train_loader, val_loader=val_loader, fold_id=fold_id)
    logger.info(f"=== {args.dataset} {args.mode} {fold_id} done. best val PCC = {best:.4f} ===")


if __name__ == "__main__":
    main()
