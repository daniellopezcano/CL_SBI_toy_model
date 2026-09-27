"""
Single configuration file for the whole pipeline.

Every notebook does `import config as cfg` and passes `cfg` to the package functions.
Edit the values here; nothing else needs to change.

Quick mode (tiny datasets, few epochs) is meant for smoke tests. Enable it either by
setting QUICK = True below or by exporting the environment variable CLMINI_QUICK=1
before starting Jupyter / nbconvert. Quick and full runs write to separate folders.
"""
import os
from pathlib import Path

# ----------------------------------------------------------------------------- #
# Global switches
# ----------------------------------------------------------------------------- #
QUICK = os.environ.get("CLMINI_QUICK", "0") == "1"   # or set to True by hand
DEVICE = "cpu"                                        # "cpu" or "cuda"
SEED = 0                                              # seed for CL and SBI training
N_THREADS = 4                                         # torch CPU threads

# Parameters inferred by the SBI posteriors (any subset of ["A", "B", "C"], in this order).
# The prior, data loading, evaluation and comparison plots all follow this setting.
SBI_PARAMS = ["A", "B", "C"]

# ----------------------------------------------------------------------------- #
# Paths (relative to this file)
# ----------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
RUN_NAME = "quick" if QUICK else "full"
DATA_DIR = ROOT / "data" / RUN_NAME       # generated spectra and CL latents
OUT_DIR = ROOT / "outputs" / RUN_NAME     # models, figures (PNG), metrics (JSON), posterior samples

# ----------------------------------------------------------------------------- #
# Toy model (identical to the original CL_mini, ks=1.0, kcut=1.0)
# ----------------------------------------------------------------------------- #
PARAM_NAMES = ["A", "B", "C"]              # A, B: "cosmology";  C: "baryonic" nuisance
PRIORS = {"A": [-0.9, -0.1], "B": [1.6, 2.0], "C": [-2.0, 2.0]}
C_RANGES = {"v1": [-2.0, -1.7], "v2": [-0.3, 0.0]}   # the two baryonic models used for training
TOY = dict(kbaryons=0.5, ks=1.0, kmin=0.0, kmax=1.0, n_k=48, kcut=1.0)
NOISE_SIGMA = (0.05, 0.0001)               # Gaussian noise sigma, linear in k from low-k to high-k

# Dataset sizes and Latin-hypercube seeds (seeds as in the original notebook)
TRAIN = dict(n_cosmo=2048, n_aug=2, n_noise=1, seed=137)
VAL = dict(n_cosmo=1024, n_aug=2, n_noise=1, seed=1)
NORM = dict(n_cosmo=1024, n_aug=3, n_noise=2, seed=2)            # C in full prior, only for normalisation
TEST_GRID = dict(n_cosmo=256, n_c=41, n_noise=1, seed=2)         # C on a regular grid over its prior

# ----------------------------------------------------------------------------- #
# Contrastive learning (hand-tuned; the wandb run "easy-sweep-5" selected by the original code used encoder [48],
# projector [24, 24], 50 x 128 batches of 256, lr 4.42e-3, weight decay 1e-4, clip 0.571)
# ----------------------------------------------------------------------------- #
CL = dict(
    hidden_encoder=[48, 32, 24], dim_latent=2,
    hidden_projector=[24, 24, 32], dim_projection=32,
    n_epochs=10, n_batches_per_epoch=64, batch_size=1024,
    lr=0.005, weight_decay=1e-8, clip_grad_norm=10.0,
    val_batch_fraction=1 / 6,              # original: validation batch = N_val_cosmo / 6
    loss=dict(delta_pull=0.5, delta_push=1.5, c_pull=1.0, c_push=1.0, c_reg=0.001),
)

# ----------------------------------------------------------------------------- #
# SBI: NPE with a MAF density estimator and an MLP embedding net
# (values from configs_SBI/ks_1.0/kcut_1.0/SBI_*_single_run.yaml)
# ----------------------------------------------------------------------------- #
SBI = dict(
    hidden_features=64, num_transforms=4, num_blocks=3,
    embedding_layers=[32, 16, 8, 2],
    batch_size=128, lr=1e-3, validation_fraction=0.2,
    max_num_epochs=2**31 - 1, stop_after_epochs=20,
)

# ----------------------------------------------------------------------------- #
# Posterior evaluation (python -m clmini.sample_posteriors, notebooks 07-09)
# ----------------------------------------------------------------------------- #
# Measured on a 16-core laptop CPU with these defaults: pk ~7-8 min (val ~1 min, test ~6.5 min),
# latents ~4 min. Cost scales with n_samples x (number of observations).
EVAL = dict(
    n_samples=1000,        # posterior samples per observation
    n_test_cosmo=None,     # test-grid cosmologies used (None = all generated, TEST_GRID["n_cosmo"])
    n_c=None,              # C grid values used, evenly spaced incl. both ends (None = all, TEST_GRID["n_c"])
    n_noise=1,             # noise realisations per (cosmology, C) used for val and test
    chunk_size=64,         # observations per chunk: bounds memory (measured peak ~0.5 GB) and is the resume unit
    min_acceptance=0.01,   # below this fraction of samples inside the prior -> fall back to unrejected samples
)

# ----------------------------------------------------------------------------- #
# Quick-mode overrides
# ----------------------------------------------------------------------------- #
if QUICK:
    TRAIN.update(n_cosmo=256)
    VAL.update(n_cosmo=128)
    NORM.update(n_cosmo=128)
    TEST_GRID.update(n_cosmo=16, n_c=9)
    CL.update(n_epochs=3, n_batches_per_epoch=8, batch_size=64)
    SBI.update(max_num_epochs=3)
    EVAL.update(n_samples=200)
