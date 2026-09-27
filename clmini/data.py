"""
Generating, saving and loading datasets.

Files written to cfg.DATA_DIR:
    norm.npz                                   per-k mean/std used to normalise all spectra
    pk/{train,val}_{v1,v2}.npz, pk/test_grid.npz   normalised spectra (see toy_model.py for the format)
    latents/<same names>.npz                   CL latents, same format with n_k -> dim_latent
"""
import numpy as np

from . import toy_model as tm

SETS = ["train_v1", "train_v2", "val_v1", "val_v2", "test_grid"]


def rel(cfg, path):
    """Path relative to the project folder (for printing, so outputs contain no absolute paths)."""
    return path.relative_to(cfg.ROOT)


def generate_all(cfg):
    """Generate the normalisation, train/val (v1, v2) and C-grid test sets and save them."""
    # 1) Normalisation from a set with C over its whole prior (as the original "Modelall" TEST set)
    c = cfg.NORM
    raw = tm.lhs_dataset(cfg, c["n_cosmo"], c["n_aug"], c["n_noise"], c["seed"], cfg.PRIORS["C"], noise_offset=0)
    flat = raw["xx"].reshape(-1, raw["xx"].shape[-1])
    norm = dict(mean=flat.mean(0), std=flat.std(0))
    (cfg.DATA_DIR / "pk").mkdir(parents=True, exist_ok=True)
    np.savez(cfg.DATA_DIR / "norm.npz", **norm)

    # 2) Train / val sets for the two baryonic models
    out = {}
    for split, c in [("train", cfg.TRAIN), ("val", cfg.VAL)]:
        for ii, model in enumerate(["v1", "v2"]):
            out[f"{split}_{model}"] = tm.lhs_dataset(
                cfg, c["n_cosmo"], c["n_aug"], c["n_noise"], c["seed"], cfg.C_RANGES[model],
                noise_offset=ii + 1, norm=norm)

    # 3) Test set with C on a regular grid
    c = cfg.TEST_GRID
    out["test_grid"] = tm.grid_dataset(cfg, c["n_cosmo"], c["n_c"], c["n_noise"], c["seed"], norm=norm)

    for name, d in out.items():
        save_set(cfg, name, d, kind="pk")
    return out


def save_set(cfg, name, d, kind="pk"):
    path = cfg.DATA_DIR / kind / f"{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **d)


def load_set(cfg, name, kind="pk"):
    """kind = 'pk' (normalised spectra) or 'latents' (CL latents)."""
    with np.load(cfg.DATA_DIR / kind / f"{name}.npz") as f:
        return {k: f[k] for k in f.files}


def flatten(d):
    """(n_noise, n_cosmo, n_aug, dim) -> x: (N, dim) and theta: (N, 3) = (A, B, C), cosmology-major order."""
    n_noise, n_cosmo, n_aug, dim = d["xx"].shape
    x = d["xx"].transpose(1, 0, 2, 3).reshape(-1, dim)
    AB = np.broadcast_to(d["cosmos"][:, None, None, :], (n_cosmo, n_noise, n_aug, 2))
    C = np.broadcast_to(d["aug"][:, None, :, :], (n_cosmo, n_noise, n_aug, 1))
    theta = np.concatenate([AB, C], axis=-1).reshape(-1, 3)
    return x, theta


def load_flat(cfg, names, kind="pk"):
    """Load and concatenate several sets. Returns x (N, dim) and theta_all (N, 3) = (A, B, C)."""
    xs, ths = zip(*(flatten(load_set(cfg, n, kind)) for n in names))
    return np.concatenate(xs), np.concatenate(ths)


def select_params(cfg, theta_all):
    """Pick the columns of cfg.SBI_PARAMS from theta_all = (A, B, C)."""
    return theta_all[:, [cfg.PARAM_NAMES.index(p) for p in cfg.SBI_PARAMS]]


def cl_views(d_v1, d_v2):
    """Group every view (noise x C x model) of each cosmology: -> (n_cosmo, n_views, n_k).
    Views of the same cosmology are the positive pairs of the contrastive loss."""
    def per_cosmo(d):
        n_noise, n_cosmo, n_aug, dim = d["xx"].shape
        return d["xx"].transpose(1, 0, 2, 3).reshape(n_cosmo, n_noise * n_aug, dim)
    assert np.array_equal(d_v1["cosmos"], d_v2["cosmos"]), "v1 and v2 must share cosmologies"
    return np.concatenate([per_cosmo(d_v1), per_cosmo(d_v2)], axis=1)
