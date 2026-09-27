"""
Toy model and dataset generation.

The toy "power spectrum" is a smoothed broken line in k:
    k << kbaryons :  f(k) = A k + B                         (A: large-scale slope, B: amplitude)
    k >> kbaryons :  f(k) = B + A kbaryons + C (k - kbaryons) (C: small-scale "baryonic" slope)
with a tanh transition whose sharpness is set by 10**ks.

Stored dataset format (one .npz per set, same shapes as the original CL_mini):
    cosmos : (n_cosmo, 2)                       -> A, B
    aug    : (n_cosmo, n_aug, 1)                -> C
    xx     : (n_noise, n_cosmo, n_aug, n_k)     -> normalised noisy spectra
"""
import numpy as np
from scipy.stats import qmc


def k_grid(kmin=0.0, kmax=1.0, n_k=48, kcut=1.0, **_):
    kk = np.linspace(kmin, kmax, n_k)
    return kk[kk <= kcut]


def toy_model(A, B, C, kbaryons=0.5, ks=1.0, kmin=0.0, kmax=1.0, n_k=48, kcut=1.0):
    """A, B: (n_cosmo,), C: (n_cosmo, n_aug) -> f: (n_cosmo, n_aug, n_k), k: (n_k,)."""
    kk = k_grid(kmin, kmax, n_k, kcut)
    k = kk[None, None, :]
    A, B, C = A[:, None, None], B[:, None, None], C[..., None]
    ff = 0.5 * (
        np.tanh(10**ks * (k - kbaryons)) * (k * (C - A) - kbaryons * (C - A))
        + k * (C + A) - kbaryons * (C - A) + 2 * B
    )
    return ff, kk


def sample_lhs(bounds, n, seed):
    """Latin hypercube samples, bounds = [[lo, hi], ...] -> (n, len(bounds)). Same as the original."""
    bounds = np.asarray(bounds, dtype=float)
    unit = qmc.LatinHypercube(d=len(bounds), seed=seed).random(n=n)
    return qmc.scale(unit, bounds[:, 0], bounds[:, 1])


def noise_sigma(cfg):
    kk_all = np.linspace(cfg.TOY["kmin"], cfg.TOY["kmax"], cfg.TOY["n_k"])
    return np.linspace(*cfg.NOISE_SIGMA, len(kk_all))[kk_all <= cfg.TOY["kcut"]]


def make_dataset(cfg, cosmos, C, n_noise, noise_seed, norm=None):
    """Evaluate the toy model on (cosmos, C), add Gaussian noise and (optionally) normalise."""
    ff, _ = toy_model(cosmos[:, 0], cosmos[:, 1], C, **cfg.TOY)
    rng = np.random.default_rng(noise_seed)
    xx = rng.normal(loc=ff, scale=noise_sigma(cfg), size=(n_noise,) + ff.shape)
    if norm is not None:
        xx = (xx - norm["mean"]) / norm["std"]
    return dict(cosmos=cosmos, aug=C[..., None], xx=xx)


def lhs_dataset(cfg, n_cosmo, n_aug, n_noise, seed, c_range, noise_offset, norm=None):
    """Cosmologies and C both from Latin hypercubes with the same seed (as in the original code).
    Using the same seed for v1 and v2 gives both models *identical* cosmologies at every index,
    which is what pairs the v1/v2 views of one cosmology for contrastive learning."""
    cosmos = sample_lhs([cfg.PRIORS["A"], cfg.PRIORS["B"]], n_cosmo, seed)
    C = sample_lhs([c_range], n_cosmo * n_aug, seed).reshape(n_cosmo, n_aug)
    return make_dataset(cfg, cosmos, C, n_noise, noise_seed=1000 * seed + noise_offset, norm=norm)


def grid_dataset(cfg, n_cosmo, n_c, n_noise, seed, norm=None):
    """Test set: LHS cosmologies, each evaluated on the same regular grid of C over its full prior."""
    cosmos = sample_lhs([cfg.PRIORS["A"], cfg.PRIORS["B"]], n_cosmo, seed)
    C = np.tile(np.linspace(*cfg.PRIORS["C"], n_c), (n_cosmo, 1))
    return make_dataset(cfg, cosmos, C, n_noise, noise_seed=1000 * seed + 99, norm=norm)
