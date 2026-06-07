"""Shared utilities: positional encoding, learning rate schedule, logging, random seed."""

import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn


def seed_everything(seed: int = 42):
    """
    Sets the seed for generating random numbers for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_slide_pcc(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    Computes Gene-wise Pearson Correlation Coefficient (PCC) for a single slide.

    Args:
        preds: (N_spots, N_genes)
        targets: (N_spots, N_genes)

    Returns:
        pcc_vec: (N_genes,)
    """
    preds_m = preds - preds.mean(axis=0, keepdims=True)
    targets_m = targets - targets.mean(axis=0, keepdims=True)

    cov = (preds_m * targets_m).sum(axis=0)
    preds_var = (preds_m ** 2).sum(axis=0)
    targets_var = (targets_m ** 2).sum(axis=0)

    denom = np.sqrt(preds_var * targets_var)

    # Handle division by zero
    mask = denom > 1e-9
    pcc_vec = np.empty(preds.shape[1])
    pcc_vec[:] = np.nan

    pcc_vec[mask] = cov[mask] / denom[mask]
    return pcc_vec


import math as _math  # noqa: E402

# ── Positional encoding helpers (originally from model_v6.py, shared by model.py) ──


class SinusoidalPE(nn.Module):
    """2D sinusoidal positional encoding on per-slide normalized pos."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        div_term = torch.exp(
            torch.arange(0, d_model // 2, 2).float()
            * (-_math.log(10000.0) / (d_model // 2))
        )
        self.register_buffer("div_term", div_term)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        pos_min = torch.amin(pos, dim=0)
        pos_max = torch.amax(pos, dim=0)
        pos = (pos - pos_min) / (pos_max - pos_min + 1e-6)
        x, y = pos[:, 0:1], pos[:, 1:2]
        pe_x = torch.zeros(pos.size(0), self.d_model // 2, device=pos.device)
        pe_y = torch.zeros(pos.size(0), self.d_model // 2, device=pos.device)
        pe_x[:, 0::2] = torch.sin(x * self.div_term)
        pe_x[:, 1::2] = torch.cos(x * self.div_term)
        pe_y[:, 0::2] = torch.sin(y * self.div_term)
        pe_y[:, 1::2] = torch.cos(y * self.div_term)
        return torch.cat([pe_x, pe_y], dim=-1)


class HybridPE(nn.Module):
    """Sinusoidal PE + small learned correction on normalised pos."""

    def __init__(self, d_model: int, correction_scale: float = 0.1):
        super().__init__()
        self.sin = SinusoidalPE(d_model)
        self.correction = nn.Sequential(
            nn.Linear(2, d_model // 4), nn.GELU(),
            nn.Linear(d_model // 4, d_model),
        )
        self.scale = correction_scale

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        pos_min = torch.amin(pos, dim=0)
        pos_max = torch.amax(pos, dim=0)
        pos_n = (pos - pos_min) / (pos_max - pos_min + 1e-6)
        return self.sin(pos) + self.scale * self.correction(pos_n)


def cosine_with_warmup(step: int, total_steps: int, warmup_steps: int,
                       min_ratio: float = 0.05) -> float:
    """LR multiplier: linear warmup then cosine decay to min_ratio * lr_max."""
    if step < warmup_steps:
        return float(step) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + _math.cos(_math.pi * progress))


# ── original utils below ──


def get_logger(name: str, log_dir: str = None) -> logging.Logger:
    """Configures and returns a logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if not logger.handlers:
        # Stream Handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File Handler (if dir provided)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            fh = logging.FileHandler(os.path.join(log_dir, 'run.log'))
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger


# ── Cross-script shared utilities (gene-panel alignment + per-gene PCC) ──
# Originally defined in infer_cscc_zeroshot.py; relocated here so both
# infer_cscc_zeroshot.py and finetune.py can import them without
# cross-directory path hacks.


def align_y_to_panel(
    y_native: np.ndarray,
    gene_symbols_native: list[str],
    gene_names_target: list[str],
) -> np.ndarray:
    """Align a native gene-expression matrix into a target gene panel.

    Spots x native_genes -> Spots x target_genes.
    Genes not present in the native panel become NaN columns.
    """
    n_spots = y_native.shape[0]
    g_target = len(gene_names_target)
    out = np.full((n_spots, g_target), np.nan, dtype=np.float32)
    native_idx = {str(g): i for i, g in enumerate(gene_symbols_native)}
    for j, gn in enumerate(gene_names_target):
        i_src = native_idx.get(str(gn))
        if i_src is not None:
            out[:, j] = y_native[:, i_src]
    return out


def per_gene_pcc(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Compute per-gene Pearson correlation (col-wise).

    Returns a (n_genes,) float64 array; genes with < 3 finite true values
    or near-zero variance produce NaN entries.
    """
    pcc = np.full(pred.shape[1], np.nan, dtype=np.float64)
    for g in range(pred.shape[1]):
        t = true[:, g]
        m = np.isfinite(t)
        if m.sum() < 3:
            continue
        p = pred[m, g]
        t = t[m]
        if p.std() < 1e-9 or t.std() < 1e-9:
            continue
        pcc[g] = float(np.corrcoef(p, t)[0, 1])
    return pcc
