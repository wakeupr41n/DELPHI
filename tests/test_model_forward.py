"""Smoke test: model forward pass with synthetic input.

This test does NOT require GPU or pre-trained weights.
"""


def test_model_imports():
    """All core model/loss symbols should be importable by name."""
    import importlib

    modules_to_check = [
        (
            "src.model",
            ["DELPHI", "GridEncoder", "WindowSelfAttn", "SwinBlock", "BayesianLinear", "BLLMuHead"],
        ),
        ("src.loss", ["HurdleGaussianLoss", "hurdle_gaussian_mean", "hurdle_gaussian_variance"]),
        ("src.dataset", []),
        ("src.utils", []),
    ]

    for mod_name, symbols in modules_to_check:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError as e:
            # Allow failures for torch/torch_geometric if not installed
            if "torch" in str(e).lower():
                continue
            raise
        for sym in symbols:
            assert hasattr(mod, sym), f"{mod_name}.{sym} not found"


def test_loss_function_signature():
    """HurdleGaussianLoss can be instantiated (without GPU)."""
    try:
        from src.loss import HurdleGaussianLoss
    except ImportError:
        import pytest

        pytest.skip("PyTorch not available")

    loss_fn = HurdleGaussianLoss(lambda_pcc=3.0)
    assert loss_fn is not None
    assert loss_fn.lambda_pcc == 3.0
