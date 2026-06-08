"""DELPHI: Gridded Transformer Neural Process backbone + DDPN-inspired heads.

Backbone (Gridded Transformer Neural Process, Ashman et al., ICML 2025 spotlight):
  1. Project per-spot UNI features to D-dim, add 2D HybridPE.
  2. Grid encoder: bin spots into G_h x G_w cells over normalised position;
     mean-pool spots inside each cell -> [G_h*G_w, D] cell tokens.
     Empty cells get a learned default token + are masked in attention.
  3. Grid processor (Swin-style): two windowed self-attention layers over the
     cell grid, window=3x3, second layer uses a 1-cell shift.
  4. k-NN cross-attention decoder: for each spot, attend to the K=4 nearest cell
     tokens by spatial distance to the cell centre.

Head (DDPN-inspired Hurdle-Gaussian, see src/loss.py):
  Per spot, per gene g:
    mu_g    = softplus(W_mu  . h)         in (0, +inf)
    log_phi = (W_phi . h)                 in R   (free dispersion, DDPN-spirit)
    pi_g    = sigmoid(W_pi . h)           in (0, 1)

Reuses HybridPE from src/model_v6.py via import. No spectral norm, no sigma-K/V
gating, no CASD: those are deliberately removed.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from src.utils import HybridPE

__all__ = [
    "DELPHI",
    "GridEncoder",
    "WindowSelfAttn",
    "SwinBlock",
    "KNNDecoder",
    "BayesianLinear",
    "BLLMuHead",
    "_DetMuHead",
    "_HomoscedasticHead",
    "_ZeroPiHead",
]


# ---------------------------------------------------------------------------
# Grid encoder: bin spots into a G_h x G_w grid by normalised position
# ---------------------------------------------------------------------------


def _bin_pos(pos: torch.Tensor, gh: int, gw: int) -> torch.Tensor:
    """Map positions to flat cell indices in [0, gh*gw).

    pos: [N, 2] absolute coords (any range; normalised internally per slide).
    Returns: [N] long cell index = row * gw + col.
    """
    pmin = pos.amin(dim=0)
    pmax = pos.amax(dim=0)
    pn = (pos - pmin) / (pmax - pmin + 1e-6)  # [N, 2] in [0, 1]
    col = (pn[:, 0] * gw).clamp(0, gw - 1).long()
    row = (pn[:, 1] * gh).clamp(0, gh - 1).long()
    return row * gw + col  # [N]


class GridEncoder(nn.Module):
    """Mean-pool spot embeddings into G_h x G_w cell tokens, with a learned
    default token for empty cells.
    """

    def __init__(self, d_model: int, gh: int = 6, gw: int = 6):
        super().__init__()
        self.gh, self.gw = gh, gw
        self.n_cells = gh * gw
        self.empty_token = nn.Parameter(torch.zeros(1, d_model))
        nn.init.normal_(self.empty_token, std=0.02)

    def forward(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """h: [N, D]. pos: [N, 2].

        Returns:
          cell_tokens : [n_cells, D]  mean-pooled per cell, empty=learned token
          empty_mask  : [n_cells]      True where cell had no spots (mask in attn)
          cell_idx    : [N]             which cell each spot landed in
        """
        N, D = h.shape  # noqa: N806
        cell_idx = _bin_pos(pos, self.gh, self.gw)  # [N]
        sums = h.new_zeros(self.n_cells, D)
        counts = h.new_zeros(self.n_cells)
        sums.index_add_(0, cell_idx, h)
        counts.index_add_(0, cell_idx, h.new_ones(N))
        nonempty = counts > 0
        means = sums / counts.clamp(min=1.0).unsqueeze(-1)
        empty_token = self.empty_token.expand(self.n_cells, D)
        cell_tokens = torch.where(nonempty.unsqueeze(-1), means, empty_token)
        empty_mask = ~nonempty  # True for empty
        return cell_tokens, empty_mask, cell_idx


# ---------------------------------------------------------------------------
# Swin-style window self-attention over the cell grid
# ---------------------------------------------------------------------------


class WindowSelfAttn(nn.Module):
    """3x3 windowed self-attention over a (gh, gw) cell grid.

    For shift=0 we partition the grid into (ceil(gh/3), ceil(gw/3)) non-overlapping
    3x3 blocks. For shift=1 we cyclic-shift the grid by (1, 1) before partitioning,
    then unshift after attention. This matches Swin v1 (Liu et al., 2021).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        gh: int = 6,
        gw: int = 6,
        window: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gh, self.gw = gh, gw
        self.window = window
        self.n_heads = n_heads
        self.d_h = d_model // n_heads
        assert gh % window == 0 and gw % window == 0, "grid must divide window"
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

    def _windows(self, x: torch.Tensor) -> torch.Tensor:
        """[gh*gw, D] -> [n_windows, window*window, D]."""
        D = x.shape[-1]  # noqa: N806
        x = x.view(self.gh, self.gw, D)
        nh, nw = self.gh // self.window, self.gw // self.window
        x = x.view(nh, self.window, nw, self.window, D)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # [nh, nw, w, w, D]
        return x.view(nh * nw, self.window * self.window, D)

    def _unwindows(self, x: torch.Tensor) -> torch.Tensor:
        """[n_windows, window*window, D] -> [gh*gw, D]."""
        D = x.shape[-1]  # noqa: N806
        nh, nw = self.gh // self.window, self.gw // self.window
        x = x.view(nh, nw, self.window, self.window, D)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x.view(self.gh * self.gw, D)

    def forward(
        self,
        x: torch.Tensor,
        empty_mask: torch.Tensor,
        shift: int = 0,
    ) -> torch.Tensor:
        """x: [gh*gw, D]. empty_mask: [gh*gw] True where empty."""
        D = x.shape[-1]  # noqa: N806
        # cyclic shift
        if shift > 0:
            x_2d = x.view(self.gh, self.gw, D)
            x_2d = torch.roll(x_2d, shifts=(-shift, -shift), dims=(0, 1))
            x = x_2d.view(self.gh * self.gw, D)
            mask_2d = empty_mask.view(self.gh, self.gw)
            mask_2d = torch.roll(mask_2d, shifts=(-shift, -shift), dims=(0, 1))
            empty_mask = mask_2d.view(self.gh * self.gw)

        # Q, K, V
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # window
        Wn = self.window * self.window  # noqa: N806
        q_w = self._windows(q)  # [n_w, Wn, D]
        k_w = self._windows(k)
        v_w = self._windows(v)
        m_w = empty_mask.view(self.gh, self.gw)
        nh, nw = self.gh // self.window, self.gw // self.window
        m_w = m_w.view(nh, self.window, nw, self.window).permute(0, 2, 1, 3).contiguous()
        m_w = m_w.view(nh * nw, Wn)  # [n_w, Wn], True = empty (mask)

        n_w = q_w.shape[0]
        # multi-head reshape
        q_w = q_w.view(n_w, Wn, self.n_heads, self.d_h).transpose(1, 2)  # [n_w, H, Wn, d_h]
        k_w = k_w.view(n_w, Wn, self.n_heads, self.d_h).transpose(1, 2)
        v_w = v_w.view(n_w, Wn, self.n_heads, self.d_h).transpose(1, 2)

        scores = torch.matmul(q_w, k_w.transpose(-2, -1)) / math.sqrt(self.d_h)  # [n_w,H,Wn,Wn]
        # mask out empty keys (broadcast over H and over query dim)
        if m_w.any():
            scores = scores.masked_fill(
                m_w.unsqueeze(1).unsqueeze(2),  # [n_w, 1, 1, Wn]
                float("-inf"),
            )
        attn = F.softmax(scores, dim=-1)
        # if a query has all -inf keys (no non-empty in window), softmax → NaN.
        # Replace NaN rows with uniform 0 contribution.
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v_w)  # [n_w,H,Wn,d_h]
        out = out.transpose(1, 2).contiguous().view(n_w, Wn, D)
        out = self._unwindows(out)  # [gh*gw, D]
        out = self.W_o(out)

        # un-shift
        if shift > 0:
            out_2d = out.view(self.gh, self.gw, D)
            out_2d = torch.roll(out_2d, shifts=(shift, shift), dims=(0, 1))
            out = out_2d.view(self.gh * self.gw, D)
        return out


class SwinBlock(nn.Module):
    """Pre-norm Swin block: WindowSelfAttn + FFN."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        gh: int,
        gw: int,
        dim_ff: int,
        dropout: float = 0.1,
        shift: int = 0,
    ):
        super().__init__()
        self.shift = shift
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = WindowSelfAttn(d_model, n_heads, gh, gw, window=3, dropout=dropout)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, empty_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.drop1(self.attn(self.norm1(x), empty_mask, shift=self.shift))
        x = x + self.drop2(self.ff(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# k-NN cross-attention decoder
# ---------------------------------------------------------------------------


class KNNDecoder(nn.Module):
    """For each query spot: attend to K nearest cell tokens by 2D distance.

    Cell centres are inferred from the bin layout: cell c at row r, col k has
    centre ((k+0.5)/gw, (r+0.5)/gh) in normalised coords.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        K: int = 4,  # noqa: N803
        gh: int = 6,
        gw: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.K = K
        self.gh = gh
        self.gw = gw
        self.n_heads = n_heads
        self.d_h = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

        rows = torch.arange(gh).float() + 0.5
        cols = torch.arange(gw).float() + 0.5
        cy = rows / gh
        cx = cols / gw
        grid_y, grid_x = torch.meshgrid(cy, cx, indexing="ij")
        centres = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # [n_cells, 2]
        self.register_buffer("centres", centres)

    def forward(
        self,
        h_q: torch.Tensor,
        pos: torch.Tensor,
        cell_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """h_q: [N, D] query features.
        pos: [N, 2] absolute spot coords (for norm).
        cell_tokens: [n_cells, D].
        Returns: [N, D] decoded features.
        """
        N, D = h_q.shape  # noqa: N806
        # normalised query positions
        pmin = pos.amin(dim=0)
        pmax = pos.amax(dim=0)
        pn = (pos - pmin) / (pmax - pmin + 1e-6)  # [N, 2]
        # distance to each cell centre
        d2 = ((pn.unsqueeze(1) - self.centres.unsqueeze(0)) ** 2).sum(dim=-1)  # [N, n_cells]
        knn_idx = d2.topk(self.K, dim=-1, largest=False).indices  # [N, K]

        # gather K cell tokens per query
        kv = cell_tokens[knn_idx]  # [N, K, D]
        q = self.W_q(h_q).view(N, self.n_heads, self.d_h)  # [N, H, d_h]
        k = self.W_k(kv).view(N, self.K, self.n_heads, self.d_h).transpose(1, 2)  # [N, H, K, d_h]
        v = self.W_v(kv).view(N, self.K, self.n_heads, self.d_h).transpose(1, 2)

        scores = (q.unsqueeze(2) * k).sum(dim=-1) / math.sqrt(self.d_h)  # [N, H, K]
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        out = (attn.unsqueeze(-1) * v).sum(dim=2)  # [N, H, d_h]
        out = out.contiguous().view(N, D)
        return self.W_o(out)


# ---------------------------------------------------------------------------
# Bayesian Last Layer (multivariate, diagonal Gaussian posterior)
# ---------------------------------------------------------------------------


class BayesianLinear(nn.Module):
    """Multivariate Bayesian last layer with diagonal Gaussian posterior.

    Reference: Wang et al., "Multivariate Bayesian Last Layer for Regression
    with Uncertainty Quantification and Decomposition", arXiv:2405.01761 (v2,
    Jan 2026), and VBLL (Harrison et al., ICLR 2024, arXiv:2404.11599).

    Prior:     p(W_ij) = N(0, prior_std^2),       p(b_j) = N(0, prior_std^2)
    Posterior: q(W_ij) = N(M_ij, exp(log_S_ij)),  q(b_j) = N(M_b_j, exp(log_S_b_j))

    Forward (analytical, single-pass):
      mean = M @ h^T + M_b
      var per output j = sum_i  h_i^2 * exp(log_S_ij) + exp(log_S_b_j)

    The variance corresponds to the *epistemic* uncertainty contribution from
    the last layer's weight posterior.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_std: float = 1.0,
        init_log_var: float = -6.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std
        self.M = nn.Parameter(torch.empty(out_features, in_features))
        self.M_b = nn.Parameter(torch.empty(out_features))
        self.log_S = nn.Parameter(torch.full((out_features, in_features), init_log_var))
        self.log_S_b = nn.Parameter(torch.full((out_features,), init_log_var))
        nn.init.kaiming_normal_(self.M, nonlinearity="relu")
        nn.init.zeros_(self.M_b)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """h: [N, in_features].

        Returns:
          out_mean : [N, out_features]    posterior mean of W h + b
          out_var  : [N, out_features]    posterior variance per output
        """
        out_mean = F.linear(h, self.M, self.M_b)
        var_W = self.log_S.exp()  # noqa: N806  [out, in]
        var_b = self.log_S_b.exp()  # [out]
        h_sq = h**2  # [N, in]
        out_var = F.linear(h_sq, var_W) + var_b  # [N, out]
        return out_mean, out_var

    def kl_divergence(self) -> torch.Tensor:
        """KL( q(W,b) || N(0, prior_std^2 * I) ), summed over all parameters."""
        prior_var = self.prior_std**2
        log_prior_var = math.log(prior_var)
        var_q = self.log_S.exp()
        kl_W = (  # noqa: N806
            0.5
            * ((self.M**2) / prior_var + var_q / prior_var - 1.0 - self.log_S + log_prior_var).sum()
        )
        var_qb = self.log_S_b.exp()
        kl_b = (
            0.5
            * (
                (self.M_b**2) / prior_var + var_qb / prior_var - 1.0 - self.log_S_b + log_prior_var
            ).sum()
        )
        return kl_W + kl_b


class BLLMuHead(nn.Module):
    """Body MLP + BayesianLinear last layer + softplus activation.

    Returns:
      mu        : [N, num_genes]   softplus(out_mean)
      epist_var : [N, num_genes]   delta-method variance: sigmoid(out_mean)^2 * out_var
                                     ≈ Var(softplus(z)) for small Var(z).
    """

    def __init__(
        self, hidden_dim: int, num_genes: int, dropout: float = 0.1, prior_std: float = 1.0
    ):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.bll = BayesianLinear(hidden_dim // 2, num_genes, prior_std=prior_std)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.body(h)
        out_mean, out_var = self.bll(z)
        mu = F.softplus(out_mean)
        # Delta method: Var(softplus(z)) ≈ (sigmoid(z))^2 * Var(z)
        epist_var = torch.sigmoid(out_mean) ** 2 * out_var
        return mu, epist_var

    def kl_divergence(self) -> torch.Tensor:
        return self.bll.kl_divergence()


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


def _make_head(d_model: int, num_genes: int, activation: str, dropout: float = 0.1) -> nn.Module:
    if activation == "softplus":
        out = nn.Softplus()
    elif activation == "sigmoid":
        out = nn.Sigmoid()
    elif activation in ("identity", "linear"):
        out = nn.Identity()
    else:
        raise ValueError(activation)
    return nn.Sequential(
        nn.Linear(d_model, d_model // 2),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(d_model // 2, num_genes),
        out,
    )


# ---------------------------------------------------------------------------
# Ablation heads (used only when the corresponding component is disabled; the
# default DELPHI config keeps all three enabled so production is unchanged).
# ---------------------------------------------------------------------------


class _DetMuHead(nn.Module):
    """Deterministic mu head (ablation: w/o BLL).  Mirrors BLLMuHead's
    (mu, epist_var) interface but the last layer is a plain Linear, so the
    epistemic variance is identically zero and the KL term vanishes."""

    def __init__(self, hidden_dim: int, num_genes: int, dropout: float = 0.1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out = nn.Linear(hidden_dim // 2, num_genes)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu = F.softplus(self.out(self.body(h)))
        return mu, torch.zeros_like(mu)

    def kl_divergence(self) -> torch.Tensor:
        return torch.zeros((), device=self.out.weight.device)


class _HomoscedasticHead(nn.Module):
    """Per-gene constant log-variance (ablation: w/o heteroscedastic noise).
    Independent of the input, so sigma_al is homoscedastic across spots."""

    def __init__(self, num_genes: int):
        super().__init__()
        self.log_phi = nn.Parameter(torch.zeros(num_genes))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.log_phi.unsqueeze(0).expand(h.shape[0], -1)


class _ZeroPiHead(nn.Module):
    """Zero zero-inflation probability (ablation: w/o pi -> pure Gaussian NLL)."""

    def __init__(self, num_genes: int):
        super().__init__()
        self.num_genes = num_genes

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.zeros(h.shape[0], self.num_genes, device=h.device, dtype=h.dtype)


class DELPHI(nn.Module):
    """G-TNP backbone + Hurdle-Gaussian (DDPN-spirit) heads.

    Forward returns (mu, log_phi, pi, h_final).
    """

    def __init__(
        self,
        uni2h_dim: int = 1536,
        hidden_dim: int = 256,
        num_genes: int = 785,
        n_heads: int = 4,
        dropout: float = 0.1,
        gh: int = 6,
        gw: int = 6,
        knn_k: int = 4,
        n_swin_blocks: int = 2,
        use_bll: bool = True,
        use_hetero: bool = True,
        use_pi: bool = True,
    ):
        super().__init__()
        self.use_bll = use_bll
        self.use_hetero = use_hetero
        self.use_pi = use_pi
        # 1. Per-spot encoder: project + PE
        self.projector = nn.Sequential(
            nn.Linear(uni2h_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pe = HybridPE(hidden_dim)

        # 2. Grid encoder
        self.grid_encoder = GridEncoder(hidden_dim, gh=gh, gw=gw)

        # 3. Swin processor (alternating shift)
        self.swin_blocks = nn.ModuleList(
            [
                SwinBlock(
                    hidden_dim,
                    n_heads,
                    gh,
                    gw,
                    dim_ff=hidden_dim * 4,
                    dropout=dropout,
                    shift=(i % 2),
                )
                for i in range(n_swin_blocks)
            ]
        )

        # 4. k-NN decoder
        self.decoder = KNNDecoder(
            hidden_dim, n_heads=n_heads, K=knn_k, gh=gh, gw=gw, dropout=dropout
        )
        self.post_norm = nn.LayerNorm(hidden_dim)

        # 5. DDPN-inspired Hurdle-Gaussian heads
        # mu head uses Bayesian Last Layer (multivariate Gaussian posterior on
        # the final linear weights) so we get an epistemic uncertainty
        # estimate per (spot, gene) in a single forward pass.
        self.head_mu = (
            BLLMuHead(hidden_dim, num_genes, dropout=dropout, prior_std=1.0)
            if use_bll
            else _DetMuHead(hidden_dim, num_genes, dropout=dropout)
        )
        self.head_log_phi = (
            _make_head(hidden_dim, num_genes, "identity", dropout)
            if use_hetero
            else _HomoscedasticHead(num_genes)
        )
        self.head_pi = (
            _make_head(hidden_dim, num_genes, "sigmoid", dropout)
            if use_pi
            else _ZeroPiHead(num_genes)
        )

    def forward(
        self,
        x: torch.Tensor,  # [N, uni2h_dim]
        pos: torch.Tensor,  # [N, 2]
        edge_index: torch.Tensor | None = None,  # unused, API-compat
        batch_idx: torch.Tensor | None = None,  # unused (batch_size=1)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. project + PE
        h = self.projector(x)
        h = h + self.pe(pos)

        # 2. grid encoder
        cell_tokens, empty_mask, _cell_idx = self.grid_encoder(h, pos)

        # 3. Swin processor
        for blk in self.swin_blocks:
            cell_tokens = blk(cell_tokens, empty_mask)

        # 4. k-NN decoder
        h_dec = self.decoder(h, pos, cell_tokens)
        h_final = self.post_norm(h + h_dec)

        # 5. heads
        mu_raw, epist_var = self.head_mu(h_final)
        mu = mu_raw.clamp(min=1e-3, max=20.0)
        # epistemic var clamp (avoid extreme values; convert to std for output)
        epist_var = epist_var.clamp(min=1e-8, max=100.0)
        log_phi = self.head_log_phi(h_final).clamp(min=-6.0, max=4.0)  # sigma_al in [0.05, 7.4]
        pi = self.head_pi(h_final).clamp(min=1e-3, max=1.0 - 1e-3)
        return mu, log_phi, pi, h_final, epist_var

    def kl_divergence(self) -> torch.Tensor:
        """Total KL divergence from BLL-equipped heads.  Used by ELBO loss."""
        return self.head_mu.kl_divergence()
