"""
Metrics for the CL latents and for the SBI posteriors, plus small I/O helpers.

Posterior metrics (same definitions as the original comparison notebooks), per parameter p:
    err   = |median - true| / prior_width
    width = 2 * std / prior_width
    bias  = (median - true) / std
    rank  = fraction of posterior samples below the true value (uniform if calibrated)
"""
import json
import numpy as np

from . import data


# ----------------------------------------------------------------------------- #
# I/O
# ----------------------------------------------------------------------------- #
def save_json(cfg, obj, relpath):
    path = cfg.OUT_DIR / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))
    return data.rel(cfg, path)


def save_fig(cfg, fig, relpath):
    """Save a figure as PNG under cfg.OUT_DIR (the notebook still shows it inline)."""
    path = (cfg.OUT_DIR / relpath).with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    return data.rel(cfg, path)


# ----------------------------------------------------------------------------- #
# CL invariance
# ----------------------------------------------------------------------------- #
def invariance_ratio(xx):
    """xx: (n_noise, n_cosmo, n_views, dim). Ratio of the scatter *within* a cosmology (across noise
    and C) to the scatter *between* cosmologies (of the per-cosmology means), summed over dims:
        sqrt( E_cosmo[ Var_views ] / Var_cosmo[ E_views ] )
    0 = perfectly invariant to C and noise; ~1 or more = C dominates the representation."""
    z = xx.transpose(1, 0, 2, 3).reshape(xx.shape[1], -1, xx.shape[-1])   # (n_cosmo, views, dim)
    within = z.var(axis=1).mean(axis=0).sum()
    between = z.mean(axis=1).var(axis=0).sum()
    return float(np.sqrt(within / between))


def in_training_c(cfg, C):
    return np.any([(C >= lo - 1e-9) & (C <= hi + 1e-9) for lo, hi in cfg.C_RANGES.values()], axis=0)


def invariance_vs_c(cfg, d):
    """For the C-grid set: displacement of the representation at each C from the per-cosmology mean
    over the C values inside the training ranges (v1, v2), in units of the between-cosmology scatter.
    Returns C values and the RMS displacement over cosmologies (and noise)."""
    xx = d["xx"]                                            # (n_noise, n_cosmo, n_c, dim)
    C = d["aug"][0, :, 0]
    ref_mask = in_training_c(cfg, C)
    if not ref_mask.any():
        ref_mask[:] = True
    ref = xx[:, :, ref_mask].mean(axis=(0, 2), keepdims=True)                 # (1, n_cosmo, 1, dim)
    scale = np.sqrt(ref[0, :, 0].var(axis=0).sum())
    disp = np.sqrt(((xx - ref) ** 2).sum(-1)) / scale                          # (n_noise, n_cosmo, n_c)
    return C, np.sqrt((disp ** 2).mean(axis=(0, 1)))


def cl_invariance_metrics(cfg):
    """Invariance ratio of the CL latents and, as a baseline, of the normalised spectra."""
    out = {}
    for kind in ["pk", "latents"]:
        v1, v2 = data.load_set(cfg, "val_v1", kind), data.load_set(cfg, "val_v2", kind)
        test = data.load_set(cfg, "test_grid", kind)
        C = test["aug"][0, :, 0]
        m = in_training_c(cfg, C)
        out[kind] = dict(
            val_v1_v2=invariance_ratio(np.concatenate([v1["xx"], v2["xx"]], axis=2)),
            test_grid_training_C=invariance_ratio(test["xx"][:, :, m]) if m.sum() > 1 else None,
            test_grid_all_C=invariance_ratio(test["xx"]),
        )
    return out


# ----------------------------------------------------------------------------- #
# Posterior metrics
# ----------------------------------------------------------------------------- #
def posterior_stats(cfg, res):
    s, t = res["samples"], res["theta_true"]                   # (N, n_samples, P), (N, P)
    widths = np.array([np.diff(cfg.PRIORS[p])[0] for p in res["params"]])
    med, std = np.median(s, axis=1), s.std(axis=1)
    return dict(
        med=med, std=std,
        err=np.abs(med - t) / widths,
        width=2 * std / widths,
        bias=(med - t) / std,
        rank=(s < t[:, None, :]).mean(axis=1),
    )


def coverage_curve(rank, levels=np.linspace(0, 1, 101)):
    """Empirical CDF of the ranks: fraction of true values below each credibility level."""
    return levels, np.array([(rank <= q).mean() for q in levels])


def posterior_metrics(cfg, res):
    """Scalar summary per parameter (JSON friendly)."""
    st = posterior_stats(cfg, res)
    fb = res["fallback"].astype(bool)
    out = {"n_obs": int(len(fb)), "n_obs_fallback": int(fb.sum()),
           "mean_acceptance": float(res["acceptance"].mean())}
    C = res["theta_all"][:, 2]
    if len(np.unique(C)) <= 101:                      # C-grid set: fallback count per C value
        out["fallback_per_C"] = {f"{c:+.2f}": int(fb[C == c].sum()) for c in np.unique(C)}
    for i, p in enumerate(res["params"]):
        lv, cov = coverage_curve(st["rank"][:, i])
        out[p] = dict(
            median_err_over_prior=np.median(st["err"][:, i]),
            median_2sigma_over_prior=np.median(st["width"][:, i]),
            mean_bias=np.mean(st["bias"][:, i]),
            frac_abs_bias_lt_1=np.mean(np.abs(st["bias"][:, i]) < 1),
            frac_abs_bias_lt_2=np.mean(np.abs(st["bias"][:, i]) < 2),
            coverage_68=np.mean(np.abs(st["rank"][:, i] - 0.5) < 0.34),
            max_calibration_error=np.max(np.abs(cov - lv)),
        )
    return out


def metrics_vs_c(cfg, res):
    """Posterior metrics as a function of the true C (for the C-grid test set)."""
    st = posterior_stats(cfg, res)
    C = res["theta_all"][:, 2]
    c_vals = np.unique(C)
    groups = [C == c for c in c_vals]
    agg = lambda f, key: np.array([[f(st[key][g, i]) for i in range(len(res["params"]))] for g in groups])
    return dict(
        C=c_vals,
        fallback_frac=np.array([res["fallback"][g].mean() for g in groups]),
        mean_acceptance=np.array([res["acceptance"][g].mean() for g in groups]),
        median_err=agg(np.median, "err"),
        median_width=agg(np.median, "width"),
        mean_bias=agg(np.mean, "bias"),
        coverage_68=np.array([[np.mean(np.abs(st["rank"][g, i] - 0.5) < 0.34)
                               for i in range(len(res["params"]))] for g in groups]),
    )


def metrics_table(cfg, results):
    """{label: samples dict} -> (pandas DataFrame for display, nested dict for JSON)."""
    import pandas as pd
    summary = {label: posterior_metrics(cfg, res) for label, res in results.items()}
    rows = [dict(set=label, param=p, **m[p]) for label, m in summary.items() for p in cfg.SBI_PARAMS]
    return pd.DataFrame(rows).set_index(["set", "param"]).round(3), summary
