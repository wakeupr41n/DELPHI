"""DELPHI-NPD-BLL cross-organ ZS inference on cSCC.

Uses HER2ST FULL ckpts (s42 + s3047 ensemble), predicts in HER2-785 panel,
aligns cSCC native (773 genes) ground truth to HER2-785 with NaN padding.
Output schema matches `infer_baseline_crossorgan_zs.py` so HEtoSGEBench eval
script (`eval_hetosgebench_format.py --dataset cscc_zs`) treats it identically.

Output: results/hetosgebench/predictions_full/cscc_zs/delphi_{P2,P5,P9,P10}.pt

Usage:
  python scripts/infer_cscc_zeroshot.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root (from scripts/inference/)
sys.path.insert(0, str(ROOT))

from src.dataset import Her2stDataset  # noqa: E402
from src.loss import hurdle_gaussian_mean, hurdle_gaussian_variance  # noqa: E402
from src.model import DELPHI  # noqa: E402
from src.utils import align_y_to_panel, per_gene_pcc  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

CSCC_DIR = ROOT / "data" / "processed" / "cSCC"
HER2_PANEL = ROOT / "data" / "processed" / "gene_names_785.npy"
OUT_DIR = ROOT / "results" / "hetosgebench" / "predictions_full" / "cscc_zs"
CKPTS = [
    ROOT / "checkpoints" / "npd_bll_h" / "full_s42_e22"   / "final_model_FULL.pt",
    ROOT / "checkpoints" / "npd_bll_h" / "full_s3047_e22" / "final_model_FULL.pt",
]
NUM_GENES = 785


def load_ensemble(device):
    models = []
    for ckpt in CKPTS:
        if not ckpt.exists():
            log.warning(f"missing ckpt: {ckpt}")
            continue
        model = DELPHI(
            uni2h_dim=1536, hidden_dim=384, num_genes=NUM_GENES,
            gh=12, gw=12, knn_k=8, n_swin_blocks=4,
        ).to(device)
        state = torch.load(str(ckpt), map_location=device)
        model.load_state_dict(state)
        model.eval()
        models.append(model)
        log.info(f"  loaded {ckpt.name}")
    if not models:
        raise RuntimeError("no ckpts loaded")
    return models


@torch.no_grad()
def forward_ensemble(models, batch):
    """Mean-ensemble forward returning (mu_pred, log_phi, pi, sigma_total)."""
    accum = None
    for m in models:
        out = m(batch.x, batch.pos, batch.edge_index,
                batch_idx=getattr(batch, "batch", None))
        if len(out) == 5:
            mu, log_phi, pi, _, epist_var = out
        else:
            mu, log_phi, pi, _ = out
            epist_var = torch.zeros_like(mu)
        mean_pred = hurdle_gaussian_mean(mu, pi)
        sigma_al = torch.sqrt(hurdle_gaussian_variance(mu, log_phi, pi) + 1e-8)
        sigma_ep = torch.sqrt(epist_var + 1e-8)
        sigma_tot = torch.sqrt(sigma_al ** 2 + sigma_ep ** 2 + 1e-8)
        if accum is None:
            accum = {
                "pred": mean_pred.cpu().numpy(),
                "mu": mu.cpu().numpy(),
                "log_phi": log_phi.cpu().numpy(),
                "pi": pi.cpu().numpy(),
                "sigma_al": sigma_al.cpu().numpy(),
                "sigma_ep": sigma_ep.cpu().numpy(),
                "sigma_total": sigma_tot.cpu().numpy(),
            }
        else:
            accum["pred"] += mean_pred.cpu().numpy()
            accum["mu"] += mu.cpu().numpy()
            accum["log_phi"] += log_phi.cpu().numpy()
            accum["pi"] += pi.cpu().numpy()
            accum["sigma_al"] += sigma_al.cpu().numpy()
            accum["sigma_ep"] += sigma_ep.cpu().numpy()
            accum["sigma_total"] += sigma_tot.cpu().numpy()
    n_models = len(models)
    for k in accum:
        accum[k] = accum[k] / float(n_models)
    return accum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    gene_names = list(np.load(str(HER2_PANEL), allow_pickle=True))
    gene_names = [str(g) for g in gene_names]
    assert len(gene_names) == NUM_GENES

    ds = Her2stDataset(root_dir=str(CSCC_DIR))
    pmap = ds.get_patient_indices()
    log.info(f"cSCC patients: {sorted(pmap.keys())}  ({len(ds)} slides)")

    models = load_ensemble(device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for pid in sorted(pmap.keys()):
        out_path = OUT_DIR / f"delphi_{pid}.pt"
        log.info(f"--- {pid} ---")
        idx_list = pmap[pid]
        loader = DataLoader(Subset(ds, idx_list), batch_size=1, shuffle=False)
        preds, trues, poss = [], [], []
        mus, lps, pis, sas, ses, sts = [], [], [], [], [], []
        for batch in loader:
            batch = batch.to(device)
            data = ds[idx_list[len(preds)]]   # local file for gene_symbols
            acc = forward_ensemble(models, batch)
            preds.append(acc["pred"])
            mus.append(acc["mu"])
            lps.append(acc["log_phi"])
            pis.append(acc["pi"])
            sas.append(acc["sigma_al"])
            ses.append(acc["sigma_ep"])
            sts.append(acc["sigma_total"])
            y_native = data.y.cpu().numpy().astype(np.float32) \
                if getattr(data, "y", None) is not None else None
            gs = getattr(data, "gene_symbols", None)
            if y_native is not None and gs is not None:
                y_aligned = align_y_to_panel(y_native, gs, gene_names)
            else:
                y_aligned = np.full((acc["pred"].shape[0], NUM_GENES),
                                     np.nan, dtype=np.float32)
            trues.append(y_aligned)
            poss.append(batch.pos.cpu().numpy())

        pred = np.concatenate(preds, axis=0)
        true = np.concatenate(trues, axis=0)
        pos = np.concatenate(poss, axis=0)
        pcc_g = per_gene_pcc(pred, true)
        pcc_mean = float(np.nanmean(pcc_g))
        log.info(f"  n_spots={pred.shape[0]}  PCC={pcc_mean:.4f}")

        record = {
            "pred": torch.from_numpy(pred.astype(np.float32)),
            "true": torch.from_numpy(true.astype(np.float32)),
            "pos": torch.from_numpy(pos.astype(np.float32)),
            "gene_names": gene_names,
            "mu": torch.from_numpy(np.concatenate(mus, 0).astype(np.float32)),
            "log_phi": torch.from_numpy(np.concatenate(lps, 0).astype(np.float32)),
            "pi": torch.from_numpy(np.concatenate(pis, 0).astype(np.float32)),
            "sigma_aleatoric": torch.from_numpy(np.concatenate(sas, 0).astype(np.float32)),
            "sigma_epistemic": torch.from_numpy(np.concatenate(ses, 0).astype(np.float32)),
            "sigma_total": torch.from_numpy(np.concatenate(sts, 0).astype(np.float32)),
            "pcc": pcc_mean,
            "n_spots": int(pred.shape[0]),
            "method": "delphi",
            "patient_id": pid,
        }
        torch.save(record, str(out_path))
        log.info(f"  saved -> {out_path.name}")


if __name__ == "__main__":
    main()
