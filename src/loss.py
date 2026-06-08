"""DDPN-inspired Hurdle-Gaussian loss for log1p-normalized spatial transcriptomics.

Why Hurdle-Gaussian instead of pure DDPN Double-Poisson:
  Existing HER2ST processed data is log1p of normalized counts, with per-spot
  scaling that is not closed-form invertible. Re-extracting raw counts with a
  matching normalization is a high-risk silent bug. Instead we adapt DDPN's
  *core idea* -- fully heteroscedastic free per-output dispersion -- to continuous
  log1p data. Zero-inflation is added because log1p(0)=0 is exact and abundant
  (~70% of entries are exactly 0), violating Gaussian assumptions.

Key contributions vs the Hurdle-Gamma in v6 (loss_v6.hurdle_gamma_nll):
  - Per-(spot, gene) free dispersion phi_g(spot) -- DDPN-spirit, replaces v6's
    constrained Gamma rate beta_g that was tied to mean via fixed clamps.
  - Direct on log1p domain (no expensive expm1 round-trip in eval).
  - PCC consistency loss in same space as data (no cross-domain fight that
    plagued v6's PCC-vs-NLL trade-off).

Output head contract (per spot, per gene g):
  mu_g     = softplus(W_mu  · h)         in (0, +inf)   -- Gaussian mean
  log_phi  = (W_phi · h)                 in R           -- log Gaussian variance
  pi_g     = sigmoid(W_pi · h)           in (0, 1)      -- zero-inflation prob

NLL per (spot, gene):
  if y == 0:
    L = -log( pi + (1-pi) * N(0 ; mu, sigma^2) )      [logsumexp]
  else:
    L = -log(1-pi) - log N(y ; mu, sigma^2)
  where sigma^2 = exp(log_phi).

PCC loss: 1 - mean_g Pearson( (1-pi)*mu , y )         in log1p domain.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

__all__ = [
    "HurdleGaussianLoss",
    "hurdle_gaussian_mean",
    "hurdle_gaussian_nll",
    "hurdle_gaussian_variance",
    "pcc_loss_log1p",
]

_EPS = 1e-6
_LOG_2PI = math.log(2.0 * math.pi)


def hurdle_gaussian_nll(
    y: torch.Tensor,  # [N, G] log1p target, y >= 0, exact 0s allowed
    mu: torch.Tensor,  # [N, G] Gaussian mean, > 0
    log_phi: torch.Tensor,  # [N, G] log variance, free (DDPN-spirit)
    pi: torch.Tensor,  # [N, G] zero-inflation prob, in (0,1)
    zero_threshold: float = 1e-8,
) -> torch.Tensor:
    """Per-(spot, gene) NLL under Hurdle-Gaussian. Returns [N, G]."""
    log_var = log_phi
    var = log_var.exp().clamp(min=_EPS)
    log_pi = torch.log(pi.clamp(min=_EPS, max=1.0 - _EPS))
    log_1mpi = torch.log((1.0 - pi).clamp(min=_EPS, max=1.0 - _EPS))

    # log N(y; mu, var) = -0.5*[(y-mu)^2 / var + log(2*pi*var)]
    log_norm = -0.5 * ((y - mu) ** 2 / var + log_var + _LOG_2PI)

    # zero branch: -log( pi + (1-pi) * N(0; mu, var) )
    log_norm_at_0 = -0.5 * (mu**2 / var + log_var + _LOG_2PI)
    # logsumexp(log_pi, log_1mpi + log_norm_at_0)
    log_zero_term = torch.logsumexp(torch.stack([log_pi, log_1mpi + log_norm_at_0], dim=0), dim=0)

    # nonzero branch: -log(1-pi) - log_norm
    log_pos_term = log_1mpi + log_norm

    is_zero = (y < zero_threshold).float()
    return -(is_zero * log_zero_term + (1.0 - is_zero) * log_pos_term)


def hurdle_gaussian_mean(mu: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
    """E[Y] under hurdle: (1-pi)*mu (mass at 0 contributes nothing)."""
    return (1.0 - pi) * mu


def hurdle_gaussian_variance(
    mu: torch.Tensor,
    log_phi: torch.Tensor,
    pi: torch.Tensor,
) -> torch.Tensor:
    """Var[Y] = (1-pi)*var + pi*(1-pi)*mu^2 (law of total variance)."""
    var = log_phi.exp()
    return (1.0 - pi) * var + pi * (1.0 - pi) * mu**2


def pcc_loss_log1p(
    mu: torch.Tensor,
    pi: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """1 - mean per-gene Pearson r between (1-pi)*mu and y (both log1p)."""
    pred = (1.0 - pi) * mu
    p = pred - pred.mean(dim=0, keepdim=True)
    t = y - y.mean(dim=0, keepdim=True)
    num = (p * t).sum(dim=0)
    denom = torch.sqrt((p**2).sum(dim=0) * (t**2).sum(dim=0) + eps)
    pcc_per_gene = num / (denom + eps)
    valid = (denom > eps).float()
    pcc_avg = (pcc_per_gene * valid).sum() / (valid.sum() + eps)
    return 1.0 - pcc_avg


class HurdleGaussianLoss(nn.Module):
    """Composite Hurdle-Gaussian + BLL ELBO loss.

      L = nll.mean()
        + lambda_pcc * (1 - mean_g PCC)
        + lambda_kl  * KL_BLL                         (BLL posterior penalty)

    `kl_term` is provided externally (model.kl_divergence()) so the loss is
    decoupled from the architecture.  `lambda_kl` should be set per-step by
    the trainer (annealed during warmup).
    """

    def __init__(self, lambda_pcc: float = 0.5):
        super().__init__()
        self.lambda_pcc = lambda_pcc

    def forward(
        self,
        y: torch.Tensor,
        mu: torch.Tensor,
        log_phi: torch.Tensor,
        pi: torch.Tensor,
        kl_term: torch.Tensor | None = None,
        lambda_kl: float = 0.0,
    ) -> tuple[torch.Tensor, dict]:
        nll = hurdle_gaussian_nll(y, mu, log_phi, pi)
        L_nll = nll.mean()  # noqa: N806
        L_pcc = pcc_loss_log1p(mu, pi, y)  # noqa: N806
        total = L_nll + self.lambda_pcc * L_pcc
        kl_val = torch.zeros((), device=L_nll.device)
        if kl_term is not None and lambda_kl > 0.0:
            total = total + lambda_kl * kl_term
            kl_val = kl_term.detach()
        return total, {
            "total": total.detach(),
            "nll": L_nll.detach(),
            "pcc": L_pcc.detach(),
            "kl": kl_val,
            "mu_mean": mu.mean().detach(),
            "phi_mean": log_phi.exp().mean().detach(),
            "pi_mean": pi.mean().detach(),
        }
