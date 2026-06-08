"""DELPHI-NPD cross-organ fine-tune -- per-patient leave-one-slide-out (spec 0514).

Continue training the HER2ST FULL NPD-BLL ckpt on each held-out slide's
sister slides (within-patient), predict on the held-out slide, save in
HEtoSGEBench format.

Splits (per spec 实验设置.txt L16):
  cscc: 4 patients × 3 reps = 12 slides. For each (patient, rep), FT on the
        other 2 reps of the SAME patient, eval on the held-out rep. → 12 runs.
  prad: 28 slides, each its own patient (HEST 1-slide-per-patient). LOSO
        across the cohort: FT on 27 slides, eval on the held-out 1. → 28 runs.

Output: results/hetosgebench/predictions/{cscc_ft,prad_ft}/delphi_<slide>.pt
        Schema matches infer_baseline_crossorgan_zs.py / infer_cscc_zeroshot.py.

Usage:
  python scripts/finetune.py --dataset cscc                 # 12 runs
  python scripts/finetune.py --dataset cscc --only P2_ST_rep1   # single slide
  python scripts/finetune.py --dataset prad                 # 28 runs
  python scripts/finetune.py --dataset prad --only MEND140
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
from src.loss import (  # noqa: E402
    hurdle_gaussian_mean,
)
from src.model import DELPHI  # noqa: E402
from src.utils import align_y_to_panel, per_gene_pcc  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

NUM_GENES = 785
HER2_PANEL = ROOT / "data" / "processed" / "gene_names_785.npy"
CKPT_FULL = ROOT / "checkpoints" / "npd_bll_h" / "full_s42_e22" / "final_model_FULL.pt"

DATASET_DIRS = {
    "cscc": ROOT / "data" / "processed" / "cSCC",
    "prad": ROOT / "data" / "processed" / "HEST_PRAD",
}
OUT_DIRS = {
    "cscc": ROOT / "results" / "hetosgebench" / "predictions" / "cscc_ft",
    "prad": ROOT / "results" / "hetosgebench" / "predictions" / "prad_ft",
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def enumerate_loso_folds(ds, dataset):
    """Return list of (test_slide_id, train_idx_list, test_idx) tuples.

    cscc: within-patient leave-1-slide-out. For each (patient, rep),
          train_idx = other 2 reps of SAME patient, test_idx = held-out rep.
    prad: cohort-wide leave-1-slide-out (each slide is its own patient
          in HEST, so within-patient LOSO is undefined; we use cohort LOSO
          as the spec-compatible adaptation).
    """
    folds = []
    if dataset == "cscc":
        pmap = ds.get_patient_indices()
        for patient_id in sorted(pmap.keys()):
            sib_idx = pmap[patient_id]  # 3 reps of this patient
            for held in sib_idx:
                test_slide_id = Path(ds.file_paths[held]).stem
                train_idx = [i for i in sib_idx if i != held]
                if not train_idx:
                    log.warning(f"  {test_slide_id}: only 1 rep -- skip")
                    continue
                folds.append((test_slide_id, train_idx, [held]))
    elif dataset == "prad":
        all_idx = sorted(range(len(ds)), key=lambda i: ds.file_paths[i])
        for held in all_idx:
            test_slide_id = Path(ds.file_paths[held]).stem
            train_idx = [i for i in all_idx if i != held]
            folds.append((test_slide_id, train_idx, [held]))
    else:
        raise ValueError(f"unknown dataset {dataset}")
    return folds


def _ft_one_fold(ds, gene_names, train_idx, test_idx, test_slide_id, device, args, out_dir):
    """Run a single LOSO fold: reload FULL ckpt → FT on train_idx → eval on test_idx."""
    set_seed(args.seed)
    model = DELPHI(
        uni2h_dim=1536,
        hidden_dim=384,
        num_genes=NUM_GENES,
        gh=12,
        gw=12,
        knn_k=8,
        n_swin_blocks=4,
    ).to(device)
    state = torch.load(str(CKPT_FULL), map_location=device)
    model.load_state_dict(state)

    if args.freeze_backbone:
        for n, p in model.named_parameters():
            if not any(k in n for k in ("mu_head", "phi_head", "pi_head", "bll", "head")):
                p.requires_grad = False
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4
    )
    # Loss functions are used directly (hurdle_gaussian_nll, pcc_loss_log1p) —
    # no HurdleGaussianLoss wrapper needed here; the loss is composed inline below.
    from src.loss import hurdle_gaussian_nll, pcc_loss_log1p  # noqa: E402

    ft_loader = DataLoader(Subset(ds, train_idx), batch_size=1, shuffle=True)
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for batch in ft_loader:
            batch = batch.to(device)
            opt.zero_grad()
            out = model(
                batch.x, batch.pos, batch.edge_index, batch_idx=getattr(batch, "batch", None)
            )
            if len(out) == 5:
                mu, log_phi, pi, _, _ = out
            else:
                mu, log_phi, pi, _ = out
            y_native = batch.y
            gs = getattr(batch, "gene_symbols", None)
            if isinstance(gs, list) and len(gs) == 1 and isinstance(gs[0], (list, tuple)):
                gs = gs[0]
            if gs is not None and y_native.shape[1] != NUM_GENES:
                y_np = align_y_to_panel(y_native.cpu().numpy(), list(gs), gene_names)
                y_aligned = torch.from_numpy(y_np).to(device)
            else:
                y_aligned = y_native
            mask = torch.isfinite(y_aligned)
            y_masked = torch.where(mask, y_aligned, torch.zeros_like(y_aligned))
            valid = mask.float()
            if valid.sum() < 1:
                continue
            kl_term = model.kl_divergence() if hasattr(model, "kl_divergence") else None
            nll_full = hurdle_gaussian_nll(y_masked, mu, log_phi, pi)
            loss_nll = (nll_full * valid).sum() / valid.sum().clamp(min=1.0)
            loss_pcc = pcc_loss_log1p(mu * valid, pi, y_masked)
            loss = loss_nll + 1.0 * loss_pcc
            if kl_term is not None:
                loss = loss + 5e-6 * kl_term
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += float(loss.item())
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            log.info(
                f"    [{test_slide_id}] epoch {epoch + 1}/{args.epochs}  "
                f"loss={epoch_loss / max(len(ft_loader), 1):.4f}"
            )

    # Eval on held-out slide
    model.eval()
    with torch.no_grad():
        idx = test_idx[0]
        loader = DataLoader(Subset(ds, [idx]), batch_size=1, shuffle=False)
        preds, trues, poss = [], [], []
        for batch in loader:
            batch = batch.to(device)
            out = model(
                batch.x, batch.pos, batch.edge_index, batch_idx=getattr(batch, "batch", None)
            )
            if len(out) == 5:
                mu, log_phi, pi, _, _ = out
            else:
                mu, log_phi, pi, _ = out
            mean_pred = hurdle_gaussian_mean(mu, pi)
            preds.append(mean_pred.cpu().numpy())
            poss.append(batch.pos.cpu().numpy())
            data = ds[idx]
            y_native = (
                data.y.cpu().numpy().astype(np.float32)
                if getattr(data, "y", None) is not None
                else None
            )
            gs = getattr(data, "gene_symbols", None)
            if y_native is not None and gs is not None and y_native.shape[1] != NUM_GENES:
                y_aligned = align_y_to_panel(y_native, gs, gene_names)
            elif y_native is not None and y_native.shape[1] == NUM_GENES:
                y_aligned = y_native
            else:
                y_aligned = np.full((mean_pred.shape[0], NUM_GENES), np.nan, dtype=np.float32)
            trues.append(y_aligned)
    pred = np.concatenate(preds, 0)
    true = np.concatenate(trues, 0)
    pos = np.concatenate(poss, 0)
    pcc_g = per_gene_pcc(pred, true)
    pcc_mean = float(np.nanmean(pcc_g))
    log.info(f"  [{test_slide_id}] FT-LOSO eval: n={pred.shape[0]}  PCC={pcc_mean:.4f}")
    record = {
        "pred": torch.from_numpy(pred.astype(np.float32)),
        "true": torch.from_numpy(true.astype(np.float32)),
        "pos": torch.from_numpy(pos.astype(np.float32)),
        "gene_names": gene_names,
        "pcc": pcc_mean,
        "n_spots": int(pred.shape[0]),
        "method": "delphi_ft_loso",
        "test_slide_id": test_slide_id,
        "n_train_slides": int(len(train_idx)),
    }
    torch.save(record, str(out_dir / f"delphi_{test_slide_id}.pt"))
    return pcc_mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASET_DIRS.keys()))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="freeze G-TNP backbone, only FT heads (more stable on small targets)",
    )
    ap.add_argument(
        "--only",
        default=None,
        help="run only the fold whose held-out slide_id matches this "
        "(useful for parallelizing across multiple GPUs)",
    )
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log.info(f"device={device} dataset={args.dataset}")

    gene_names = list(np.load(str(HER2_PANEL), allow_pickle=True))
    gene_names = [str(g) for g in gene_names]
    assert len(gene_names) == NUM_GENES

    ds = Her2stDataset(root_dir=str(DATASET_DIRS[args.dataset]))
    folds = enumerate_loso_folds(ds, args.dataset)
    if args.only:
        folds = [f for f in folds if f[0] == args.only]
        if not folds:
            log.error(f"no fold matches --only {args.only}")
            return
    log.info(f"LOSO folds: {len(folds)} ({args.dataset})")

    out_dir = OUT_DIRS[args.dataset]
    out_dir.mkdir(parents=True, exist_ok=True)

    pcc_log = []
    for i, (test_slide_id, train_idx, test_idx) in enumerate(folds, 1):
        out_path = out_dir / f"delphi_{test_slide_id}.pt"
        if out_path.exists():
            log.info(
                f"\n=== fold {i}/{len(folds)}: hold out {test_slide_id} (skip -- output exists) ==="
            )
            try:
                d = torch.load(str(out_path), map_location="cpu", weights_only=False)
                pcc_log.append((test_slide_id, float(d.get("pcc", float("nan")))))
            except Exception:
                pass
            continue
        log.info(
            f"\n=== fold {i}/{len(folds)}: hold out {test_slide_id} "
            f"(FT on {len(train_idx)} sister slide(s)) ==="
        )
        pcc = _ft_one_fold(
            ds, gene_names, train_idx, test_idx, test_slide_id, device, args, out_dir
        )
        pcc_log.append((test_slide_id, pcc))

    log.info("\n=== LOSO FT summary ===")
    for sid, pcc in pcc_log:
        log.info(f"  {sid}: PCC={pcc:.4f}")
    if pcc_log:
        log.info(f"  mean PCC = {np.mean([p for _, p in pcc_log]):.4f}")


if __name__ == "__main__":
    main()
