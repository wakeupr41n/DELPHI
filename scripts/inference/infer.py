"""Generic DELPHI inference: load ckpt, run on a target dataset, save dict.

Usage:
  # Infer on HER2ST patient B (uses LOOCV-B ckpt)
  python scripts/infer.py --dataset her2st --patient B \
      --ckpt checkpoints/npd_bll_h/loocv_B_s42/best_model_B.pt \
      --out  results/hetosgebench/predictions_npd_bll/her2st/delphi_B.pt

  # Cross-organ ZS: use HER2ST FULL ckpt to predict cSCC slides
  python scripts/infer.py --dataset cscc --patient ALL \
      --ckpt checkpoints/npd_bll_h/full_s42/final_model_FULL.pt \
      --out  results/hetosgebench/predictions_npd_zs/cscc/delphi_ALL.pt
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

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


DATASET_DIRS = {
    "her2st": "data/processed/processed_data_her2st_uni_fixed",
    "cscc": "data/processed/cSCC",
    "cscc756": "data/processed/cSCC_756",
    "cscc_skinhvg": "data/processed/cSCC_skinHVG",
    "hest_prad": "data/processed/HEST_PRAD",
    "tcga_her2pos": "data/processed/TCGA_HER2POS",
}

GENE_FILES = {
    "her2st": "data/raw/her2st/her_hvg_cut_1000.npy",
    "cscc": "data/raw/her2st/her_hvg_cut_1000.npy",  # 773 of 785 -- pad-to-785 fine
    "cscc756": None,  # use gene_symbols from .pt (756 benchmark panel)
    "cscc_skinhvg": None,  # use gene_symbols from .pt
    "hest_prad": "data/raw/her2st/her_hvg_cut_1000.npy",
    "tcga_her2pos": "data/raw/her2st/her_hvg_cut_1000.npy",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASET_DIRS.keys()))
    ap.add_argument(
        "--patient", default="ALL", help="Patient ID to infer; 'ALL' = all patients in dataset"
    )
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--per_patient",
        action="store_true",
        help="Save one .pt per patient instead of one merged file (use for ZS / TCGA)",
    )
    ap.add_argument("--hidden_dim", type=int, default=384)
    ap.add_argument("--gh", type=int, default=12)
    ap.add_argument("--gw", type=int, default=12)
    ap.add_argument("--knn_k", type=int, default=8)
    ap.add_argument("--n_swin_blocks", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = ROOT / DATASET_DIRS[args.dataset]
    ds = Her2stDataset(root_dir=str(data_dir))
    pmap = ds.get_patient_indices()

    if args.patient == "ALL":
        target_ids = sorted(pmap.keys())
    else:
        assert args.patient in pmap, f"unknown patient {args.patient}; have {sorted(pmap.keys())}"
        target_ids = [args.patient]

    sample = ds[0]
    num_genes = sample.y.shape[1] if hasattr(sample, "y") and sample.y is not None else 785
    uni_dim = sample.x.shape[1]
    logger.info(f"dataset={args.dataset} patients={target_ids} num_genes={num_genes}")

    model = DELPHI(
        uni2h_dim=uni_dim,
        hidden_dim=args.hidden_dim,
        num_genes=num_genes,
        gh=args.gh,
        gw=args.gw,
        knn_k=args.knn_k,
        n_swin_blocks=args.n_swin_blocks,
    ).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()
    logger.info(f"loaded ckpt: {args.ckpt}")

    # gene names
    gene_names = None
    gf = GENE_FILES.get(args.dataset)
    if gf is not None and (ROOT / gf).exists():
        gene_names = list(np.load(str(ROOT / gf), allow_pickle=True))
    if gene_names is None and hasattr(sample, "gene_symbols"):
        gene_names = list(sample.gene_symbols)
    if gene_names is None:
        gene_names = [f"gene_{i}" for i in range(num_genes)]
    if len(gene_names) != num_genes:
        logger.warning(
            f"gene_names len {len(gene_names)} != num_genes {num_genes}; padding/truncating"
        )
        gene_names = (list(gene_names) + [f"gene_{i}" for i in range(num_genes)])[:num_genes]

    out_root = Path(args.out)
    if args.per_patient:
        out_root.mkdir(parents=True, exist_ok=True)
    else:
        out_root.parent.mkdir(parents=True, exist_ok=True)

    # accumulators (only used when not per_patient)
    all_preds, all_trues, all_poss = [], [], []
    all_mu, all_log_phi, all_pi = [], [], []
    all_sa, all_se, all_st = [], [], []

    with torch.no_grad():
        for pid in target_ids:
            idx = pmap[pid]
            loader = DataLoader(Subset(ds, idx), batch_size=1, shuffle=False)
            preds, trues, poss = [], [], []
            mus, log_phis, pis, sas, ses, sts = [], [], [], [], [], []
            for batch in loader:
                batch = batch.to(device)
                out = model(
                    batch.x, batch.pos, batch.edge_index, batch_idx=getattr(batch, "batch", None)
                )
                if len(out) == 5:
                    mu, log_phi, pi, _, epist_var = out
                else:
                    mu, log_phi, pi, _ = out
                    epist_var = torch.zeros_like(mu)
                mean_pred = hurdle_gaussian_mean(mu, pi)
                sigma_al = torch.sqrt(hurdle_gaussian_variance(mu, log_phi, pi) + 1e-8)
                sigma_ep = torch.sqrt(epist_var + 1e-8)
                sigma_tot = torch.sqrt(sigma_al**2 + sigma_ep**2 + 1e-8)
                preds.append(mean_pred.cpu().numpy())
                trues.append(
                    batch.y.cpu().numpy()
                    if getattr(batch, "y", None) is not None
                    else np.zeros_like(mean_pred.cpu().numpy())
                )
                poss.append(batch.pos.cpu().numpy())
                mus.append(mu.cpu().numpy())
                log_phis.append(log_phi.cpu().numpy())
                pis.append(pi.cpu().numpy())
                sas.append(sigma_al.cpu().numpy())
                ses.append(sigma_ep.cpu().numpy())
                sts.append(sigma_tot.cpu().numpy())

            pred = np.concatenate(preds, axis=0)
            true = np.concatenate(trues, axis=0)
            pos = np.concatenate(poss, axis=0)
            mu_a = np.concatenate(mus, axis=0)
            lp_a = np.concatenate(log_phis, axis=0)
            pi_a = np.concatenate(pis, axis=0)
            sa_a = np.concatenate(sas, axis=0)
            se_a = np.concatenate(ses, axis=0)
            st_a = np.concatenate(sts, axis=0)

            pm = pred - pred.mean(0, keepdims=True)
            tm = true - true.mean(0, keepdims=True)
            denom = np.sqrt((pm**2).sum(0) * (tm**2).sum(0))
            mask = denom > 1e-9
            pcc = np.full(pred.shape[1], np.nan)
            pcc[mask] = (pm * tm).sum(0)[mask] / denom[mask]
            pcc_mean = float(np.nanmean(pcc))
            logger.info(f"  patient {pid}: n_spots={pred.shape[0]}  PCC={pcc_mean:.4f}")

            d_save = {
                "pred": torch.from_numpy(pred).float(),
                "true": torch.from_numpy(true).float(),
                "pos": torch.from_numpy(pos).float(),
                "gene_names": gene_names,
                "mu": torch.from_numpy(mu_a).float(),
                "log_phi": torch.from_numpy(lp_a).float(),
                "pi": torch.from_numpy(pi_a).float(),
                "sigma_aleatoric": torch.from_numpy(sa_a).float(),
                "sigma_epistemic": torch.from_numpy(se_a).float(),
                "sigma_total": torch.from_numpy(st_a).float(),
                "pcc": pcc_mean,
                "n_spots": pred.shape[0],
            }
            if args.per_patient:
                pp = out_root / f"delphi_{pid}.pt"
                torch.save(d_save, pp)
                logger.info(f"  saved → {pp}")
            else:
                all_preds.append(pred)
                all_trues.append(true)
                all_poss.append(pos)
                all_mu.append(mu_a)
                all_log_phi.append(lp_a)
                all_pi.append(pi_a)
                all_sa.append(sa_a)
                all_se.append(se_a)
                all_st.append(st_a)

    if not args.per_patient:
        # only target_ids[0] expected; concat everything as one
        d_merge = {
            "pred": torch.from_numpy(np.concatenate(all_preds, 0)).float(),
            "true": torch.from_numpy(np.concatenate(all_trues, 0)).float(),
            "pos": torch.from_numpy(np.concatenate(all_poss, 0)).float(),
            "gene_names": gene_names,
            "mu": torch.from_numpy(np.concatenate(all_mu, 0)).float(),
            "log_phi": torch.from_numpy(np.concatenate(all_log_phi, 0)).float(),
            "pi": torch.from_numpy(np.concatenate(all_pi, 0)).float(),
            "sigma_aleatoric": torch.from_numpy(np.concatenate(all_sa, 0)).float(),
            "sigma_epistemic": torch.from_numpy(np.concatenate(all_se, 0)).float(),
            "sigma_total": torch.from_numpy(np.concatenate(all_st, 0)).float(),
        }
        torch.save(d_merge, out_root)
        logger.info(f"saved merged → {out_root}")


if __name__ == "__main__":
    main()
