"""Verify key data files exist and have expected structure.

These are smoke tests — they don't load large .pt files,
just check that critical pipeline inputs are present.
"""
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent


def test_checkpoints_present():
    """All 60 LOOCV checkpoints exist."""
    ckpt_dir = REPO / "checkpoints" / "npd_bll_h"
    her2st_folds = sum(
        1 for _ in ckpt_dir.glob("loocv_?_s*/best_model_?.pt"))
    cscc_dir = REPO / "checkpoints" / "npd_cscc756_final"
    cscc_folds = sum(
        1 for _ in cscc_dir.glob("loocv_*/best_model_*.pt"))
    assert her2st_folds >= 40, f"Expected >=40 HER2ST ckpts, found {her2st_folds}"
    assert cscc_folds >= 20, f"Expected >=20 cSCC ckpts, found {cscc_folds}"


def test_full_checkpoints_present():
    """All 5 FULL checkpoints for Fig 3/5 exist."""
    ckpt_dir = REPO / "checkpoints" / "npd_bll_h"
    full_ckpts = list(ckpt_dir.glob("full_s*_e22/final_model_FULL.pt"))
    assert len(full_ckpts) >= 5, f"Expected >=5 FULL ckpts, found {len(full_ckpts)}"


def test_external_rds_present():
    """Benchmark RDS files for Fig 2/3 exist."""
    het = REPO / "_external" / "HEtoSGEBench" / "benchmark pipeline" / "data" / "processed"
    rds_files = [
        het / "her2st" / "her2st_pred_feat_cor_11.rds",
        het / "cscc" / "cscc_pred_feat_cor_11.rds",
        het / "cscc_zs" / "cscc_zs_pred_feat_cor_11.rds",
        het / "prad_zs" / "prad_zs_pred_feat_cor_11.rds",
    ]
    for f in rds_files:
        assert f.exists(), f"Missing RDS: {f}"


def test_fig5_cache_present():
    """Fig 5 downstream data exists (can render figure from cache)."""
    sec6 = REPO / "results" / "tcga_brca"
    required = [
        sec6 / "spatial_features.csv",
        sec6 / "univariate_cox_by_pam50.csv",
        sec6 / "multivariate_cox.json",
        sec6 / "inference" / "TCGA-BH-A1FN.pt",
    ]
    for f in required:
        assert f.exists(), f"Missing Fig5 data: {f}"


def test_figure_scripts_exist():
    """All figure generation scripts are present."""
    fig_dir = REPO / "scripts" / "figures"
    main_figs = ["plot_fig2_main.R", "plot_fig3_main.R",
                 "plot_fig4_main.R", "plot_fig5_main.py"]
    for f in main_figs:
        assert (fig_dir / f).exists(), f"Missing figure script: {f}"

    supp_count = len(list(fig_dir.glob("plot_supp_*.R")))
    assert supp_count >= 15, f"Expected >=15 supp R scripts, found {supp_count}"


def test_preprocessing_scripts_exist():
    """All preprocessing entry points are present."""
    prep_dir = REPO / "scripts" / "preprocessing"
    for f in ["preprocess_her2st.py", "preprocess_cscc.py", "preprocess_hest_prad.py"]:
        assert (prep_dir / f).exists(), f"Missing preprocessing script: {f}"
