"""Stripped trainer for DELPHI.

No phases, no CASD, no EMA, no sigma-gating. Single AdamW + cosine LR with
warmup. Val-best ckpt saved by mean per-gene Pearson PCC across the validation
slides (concatenated for the final metric -- matches HEtoSGEBench protocol).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.loss import HurdleGaussianLoss, hurdle_gaussian_mean
from src.utils import compute_slide_pcc, cosine_with_warmup

logger = logging.getLogger(__name__)


class Trainer:
    """DELPHI training loop with cosine LR schedule, BLL KL annealing,
    and HEtoSGEBench-compliant logging.
    """

    def __init__(
        self,
        model: nn.Module,
        config: dict[str, Any],
        device: torch.device,
    ):
        self.model = model.to(device)
        self.device = device

        td = config.get("train", {})
        ld = config.get("loss", {})
        self.criterion = HurdleGaussianLoss(
            lambda_pcc=ld.get("lambda_pcc", 0.5),
        )
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=td.get("lr", 1e-4),
            weight_decay=td.get("weight_decay", 1e-4),
        )
        self.epochs = td.get("epochs", 50)
        self.warmup_epochs = td.get("warmup_epochs", 5)
        self.grad_clip = td.get("grad_clip", 1.0)
        self.val_every = td.get("val_every", 1)
        self.lr_max = self.optimizer.param_groups[0]["lr"]
        self.save_dir = config["experiment"]["save_dir"]
        os.makedirs(self.save_dir, exist_ok=True)

        # BLL KL annealing: ramp lambda_kl from 0 -> kl_max linearly over
        # kl_warmup_epochs, then hold at kl_max.  kl_max is normalised by the
        # number of train slides so the per-batch KL contribution is on the
        # order of NLL.
        self.kl_max = float(ld.get("lambda_kl_max", 1e-4))
        self.kl_warmup_epochs = int(ld.get("kl_warmup_epochs", 10))
        self.has_kl = hasattr(self.model, "kl_divergence")

    def _set_lr(self, epoch: int) -> None:
        mult = cosine_with_warmup(epoch, self.epochs, self.warmup_epochs)
        for g in self.optimizer.param_groups:
            g["lr"] = self.lr_max * mult

    def _kl_lambda(self, epoch: int) -> float:
        if not self.has_kl or self.kl_max <= 0.0:
            return 0.0
        if epoch <= self.kl_warmup_epochs:
            return self.kl_max * float(epoch) / max(1, self.kl_warmup_epochs)
        return self.kl_max

    def train_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        """Run one training epoch. Returns (avg_loss, metrics_dict)."""
        self.model.train()
        self._set_lr(epoch)
        kl_lambda = self._kl_lambda(epoch)
        agg = {
            "total": 0.0,
            "nll": 0.0,
            "pcc": 0.0,
            "kl": 0.0,
            "mu_mean": 0.0,
            "phi_mean": 0.0,
            "pi_mean": 0.0,
        }
        n = 0
        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(
                batch.x, batch.pos, batch.edge_index, batch_idx=getattr(batch, "batch", None)
            )
            # model returns (mu, log_phi, pi, h_final) or (mu, log_phi, pi, h_final, epist_var)
            if len(out) == 5:
                mu, log_phi, pi, _, _ = out
            else:
                mu, log_phi, pi, _ = out
            kl = self.model.kl_divergence() if self.has_kl else None
            total, comp = self.criterion(batch.y, mu, log_phi, pi, kl_term=kl, lambda_kl=kl_lambda)
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            for k in agg:
                agg[k] += float(comp[k])
            n += 1
        for k in agg:
            agg[k] /= max(1, n)
        agg["lr"] = self.optimizer.param_groups[0]["lr"]
        agg["kl_lambda"] = kl_lambda
        return agg

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> dict[str, float]:
        """Concat all val slides into one big spot-set and compute per-gene PCC.

        This matches the HEtoSGEBench-style metric (PCC over merged B1..B6).
        """
        self.model.eval()
        preds, trues = [], []
        for batch in loader:
            batch = batch.to(self.device)
            out = self.model(
                batch.x, batch.pos, batch.edge_index, batch_idx=getattr(batch, "batch", None)
            )
            if len(out) == 5:
                mu, log_phi, pi, _, _ = out
            else:
                mu, log_phi, pi, _ = out
            mean_pred = hurdle_gaussian_mean(mu, pi)  # log1p domain
            preds.append(mean_pred.cpu().numpy())
            trues.append(batch.y.cpu().numpy())
        if not preds:
            return {"pcc": 0.0}
        pred_all = np.concatenate(preds, axis=0)
        true_all = np.concatenate(trues, axis=0)
        per_gene_pcc = compute_slide_pcc(pred_all, true_all)
        # compute_slide_pcc returns vector [G]; take nanmean
        per_gene_pcc = np.asarray(per_gene_pcc, dtype=float)
        pcc_all = float(np.nanmean(per_gene_pcc))
        return {"pcc": pcc_all, "n_genes_valid": int(np.isfinite(per_gene_pcc).sum())}

    def run(self, train_loader, val_loader=None, fold_id: str = "B") -> float:
        """Execute full training loop. Saves best model by validation PCC.
        Returns path to best checkpoint."""
        best = -1.0
        best_epoch = 0
        for epoch in range(1, self.epochs + 1):
            m = self.train_epoch(train_loader, epoch)
            log_msg = (
                f"[{fold_id}] Ep {epoch:03d} lr {m['lr']:.2e} kl_w {m.get('kl_lambda', 0):.1e} | "
                f"total {m['total']:.3f} nll {m['nll']:.3f} pcc {m['pcc']:.3f} kl {m.get('kl', 0):.0f} | "
                f"mu_avg {m['mu_mean']:.2f} phi_avg {m['phi_mean']:.2f} pi_avg {m['pi_mean']:.2f}"
            )
            if val_loader is not None and epoch % self.val_every == 0:
                v = self.validate(val_loader)
                log_msg += f" | val PCC {v['pcc']:.4f}"
                if v["pcc"] > best:
                    best = v["pcc"]
                    best_epoch = epoch
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(self.save_dir, f"best_model_{fold_id}.pt"),
                    )
            logger.info(log_msg)

        torch.save(
            self.model.state_dict(), os.path.join(self.save_dir, f"final_model_{fold_id}.pt")
        )
        logger.info(f"[{fold_id}] best val PCC = {best:.4f} at epoch {best_epoch}")
        return best
