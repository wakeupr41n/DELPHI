"""DELPHI: Deep Epistemic Learning for Pathology-Histology Inference.

Core library providing the NPD architecture (Swin-grid Transformer + BLL +
Hurdle-Gaussian head) for uncertainty-aware spatial gene expression
prediction from H&E histology images.
"""

from src.loss import (  # noqa: F401
    HurdleGaussianLoss,
    hurdle_gaussian_mean,
    hurdle_gaussian_variance,
)
from src.model import (  # noqa: F401
    DELPHI,
    BayesianLinear,
    BLLMuHead,
    GridEncoder,
    KNNDecoder,
    SwinBlock,
    WindowSelfAttn,
)
from src.trainer import Trainer  # noqa: F401
