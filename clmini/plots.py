"""
All figures of the pipeline. Every function returns a matplotlib Figure; notebooks show it inline
and save it with metrics.save_fig.

Colour conventions: categorical series in fixed order (blue, orange, aqua); continuous parameters
(A, B, C) on a single-hue blue ramp; the C ranges seen in training (v1, v2) as grey bands.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from . import data, metrics, toy_model as tm

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SEQ = mpl.colors.LinearSegmentedColormap.from_list(
    "blue_ramp", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
TRAIN_BAND = dict(color="0.85", zorder=0)
LABELS = {"A": "$A$", "B": "$B$", "C": "$C$"}

plt.rcParams.update({
    "axes.grid": True, "grid.color": "0.9", "grid.linewidth": 0.6, "axes.edgecolor": "0.4",
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "legend.frameon": False, "figure.dpi": 100, "font.size": 11,
})


def _shade_training_c(cfg, ax, axis="x"):
    for lo, hi in cfg.C_RANGES.values():
        (ax.axvspan if axis == "x" else ax.axhspan)(lo, hi, **TRAIN_BAND)


def _denorm(cfg, xx):
    with np.load(cfg.DATA_DIR / "norm.npz") as n:
        return xx * n["std"] + n["mean"]


# ----------------------------------------------------------------------------- #
# 02 - data inspection
# ----------------------------------------------------------------------------- #
def parameter_distributions(cfg):
    """Sampled (A, B) and C for every stored set."""
    sets = {n: data.load_set(cfg, n) for n in data.SETS}
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for i, split in enumerate(["train", "val", "test_grid"]):
        d = sets[f"{split}_v1"] if split != "test_grid" else sets["test_grid"]
        axs[0].scatter(*d["cosmos"].T, s=6, color=SERIES[i], label=split, alpha=0.7)
    axs[0].set(xlabel="$A$", ylabel="$B$", title="Cosmologies (v1 and v2 share them)")
    axs[0].legend(markerscale=2, loc="upper left", bbox_to_anchor=(1, 1))
    bins = np.linspace(*cfg.PRIORS["C"], 81)
    for i, name in enumerate(["train_v1", "train_v2", "test_grid"]):
        axs[1].hist(sets[name]["aug"].ravel(), bins=bins, color=SERIES[i], alpha=0.8, label=name, density=True)
    axs[1].set(xlabel="$C$", ylabel="density", title="C values: v1, v2 (training) and the C-grid test set")
    axs[1].legend(loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    return fig


def spectra_examples(cfg, n_cosmo=3):
    """A few training cosmologies: all their v1 and v2 views (noisy) and the noise-free model."""
    v1, v2 = data.load_set(cfg, "train_v1"), data.load_set(cfg, "train_v2")
    kk = tm.k_grid(**cfg.TOY)
    fig, axs = plt.subplots(1, n_cosmo, figsize=(4.2 * n_cosmo, 3.8), sharey=True)
    for j, ax in enumerate(np.atleast_1d(axs)):
        A, B = v1["cosmos"][j]
        for i, (name, d) in enumerate([("v1", v1), ("v2", v2)]):
            ff, _ = tm.toy_model(d["cosmos"][j:j + 1, 0], d["cosmos"][j:j + 1, 1], d["aug"][j:j + 1, :, 0], **cfg.TOY)
            for a in range(d["xx"].shape[2]):
                ax.plot(kk, _denorm(cfg, d["xx"][0, j, a]), color=SERIES[i], lw=1, alpha=0.6)
                ax.plot(kk, ff[0, a], color=SERIES[i], lw=2, label=name if a == 0 else None)
        ax.axvline(cfg.TOY["kbaryons"], color="0.5", lw=1, ls=":")
        ax.set(title=f"A={A:.2f}, B={B:.2f}", xlabel="$k$")
    np.atleast_1d(axs)[0].set_ylabel("$f(k)$  (thin: noisy, thick: noise-free)")
    np.atleast_1d(axs)[0].legend()
    fig.tight_layout()
    return fig


def c_grid_spectra(cfg, i_cosmo=0):
    """One test cosmology evaluated on the whole C grid (colour = C), noisy and normalised."""
    d = data.load_set(cfg, "test_grid")
    kk, C = tm.k_grid(**cfg.TOY), d["aug"][i_cosmo, :, 0]
    norm = mpl.colors.Normalize(*cfg.PRIORS["C"])
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for c, a in enumerate(C):
        axs[0].plot(kk, _denorm(cfg, d["xx"][0, i_cosmo, c]), color=SEQ(norm(a)), lw=1)
        axs[1].plot(kk, d["xx"][0, i_cosmo, c], color=SEQ(norm(a)), lw=1)
    axs[0].set(xlabel="$k$", ylabel="$f(k)$", title=f"C grid, A={d['cosmos'][i_cosmo, 0]:.2f}, B={d['cosmos'][i_cosmo, 1]:.2f}")
    axs[1].set(xlabel="$k$", ylabel="normalised input", title="Same, as fed to the networks")
    fig.colorbar(mpl.cm.ScalarMappable(norm, SEQ), ax=axs, label="$C$")
    return fig


def noise_level(cfg):
    kk = tm.k_grid(**cfg.TOY)
    fig, ax = plt.subplots(figsize=(5, 3.3))
    ax.plot(kk, tm.noise_sigma(cfg), color=SERIES[0])
    ax.set(xlabel="$k$", ylabel="noise $\\sigma$", title="Gaussian noise level")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- #
# 03 / 04 - contrastive learning
# ----------------------------------------------------------------------------- #
def loss_history(history, title="CL loss", keys=("train", "val")):
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for i, k in enumerate(keys):
        ax.plot(np.arange(1, len(history[k]) + 1), history[k], color=SERIES[i], label=k)
    ax.set(xlabel="epoch", ylabel="loss", title=title)
    ax.legend()
    fig.tight_layout()
    return fig


def latent_clusters(cfg, n_cosmo=4):
    """Validation latents: grey = all, coloured = a few cosmologies with all their views
    (o = v1, x = v2). A good CL model gives tight, well separated clusters."""
    v1, v2 = data.load_set(cfg, "val_v1", "latents"), data.load_set(cfg, "val_v2", "latents")
    colors = SERIES[:n_cosmo]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for d in (v1, v2):
        z = d["xx"].reshape(-1, d["xx"].shape[-1])
        ax.scatter(z[:, 0], z[:, 1], s=3, color="0.8", zorder=1)
    for j in range(n_cosmo):
        for d, m in [(v1, "o"), (v2, "x")]:
            z = d["xx"][:, j].reshape(-1, d["xx"].shape[-1])
            ax.scatter(z[:, 0], z[:, 1], s=40, marker=m, color=colors[j], zorder=2)
    ax.scatter([], [], marker="o", color="k", label="v1 views")
    ax.scatter([], [], marker="x", color="k", label="v2 views")
    ax.legend(loc="best")
    ax.set(xlabel="$z_1$", ylabel="$z_2$", title="Validation latents (colour = cosmology)")
    fig.tight_layout()
    return fig


def latent_plane_by_param(cfg, split="test_grid"):
    """Latent plane coloured by A, B and C. Invariance to C shows as no colour gradient in the C panel."""
    d = data.load_set(cfg, split, "latents")
    x, theta = data.flatten(d)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    for i, (ax, p) in enumerate(zip(axs, cfg.PARAM_NAMES)):
        sc = ax.scatter(x[:, 0], x[:, 1], c=theta[:, i], cmap=SEQ, s=4, vmin=cfg.PRIORS[p][0], vmax=cfg.PRIORS[p][1])
        fig.colorbar(sc, ax=ax, label=LABELS[p])
        ax.set(xlabel="$z_1$", ylabel="$z_2$", title=f"{split}: colour = {p}")
    fig.tight_layout()
    return fig


def latents_vs_params(cfg, split="test_grid"):
    """Each latent component against A, B and C (C-grid test set)."""
    d = data.load_set(cfg, split, "latents")
    x, theta = data.flatten(d)
    D = x.shape[1]
    fig, axs = plt.subplots(D, 3, figsize=(13, 3.2 * D), squeeze=False)
    for r in range(D):
        for i, p in enumerate(cfg.PARAM_NAMES):
            ax = axs[r, i]
            if p == "C":
                _shade_training_c(cfg, ax)
            ax.scatter(theta[:, i], x[:, r], s=2, color=SERIES[0], alpha=0.3)
            ax.set(xlabel=LABELS[p], ylabel=f"$z_{r + 1}$")
    fig.suptitle("Latents vs parameters (grey band = C seen in training)")
    fig.tight_layout()
    return fig


def invariance_vs_c(cfg):
    """Displacement of the representation with C, relative to the training C values (0 = invariant)."""
    fig, ax = plt.subplots(figsize=(6, 3.8))
    _shade_training_c(cfg, ax)
    for i, (kind, label) in enumerate([("pk", "input spectra"), ("latents", "CL latents")]):
        C, disp = metrics.invariance_vs_c(cfg, data.load_set(cfg, "test_grid", kind))
        ax.plot(C, disp, color=SERIES[i], marker="o", ms=4, label=label)
    ax.set(xlabel="$C$", ylabel="RMS shift / cosmology scatter",
           title="Sensitivity to C (grey = training C ranges)")
    ax.legend()
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- #
# 05 - 09 - SBI posteriors.  `results` = {label: samples dict from sbi_tools.load_samples}
# ----------------------------------------------------------------------------- #
def _styles(labels):
    """Line style per result label: 'Pk ...' blue, 'CL ...' orange (else categorical order);
    labels containing 'test' are dashed, so model = colour and data set = line style."""
    out = {}
    for i, lab in enumerate(labels):
        color = SERIES[0] if lab.startswith("Pk") else SERIES[1] if lab.startswith("CL") else SERIES[i % len(SERIES)]
        out[lab] = dict(color=color, ls="--" if "test" in lab else "-")
    return out


def true_vs_pred(cfg, results, max_points=500):
    """Posterior median +- std against the true value, one column per result set."""
    params = next(iter(results.values()))["params"]
    fig, axs = plt.subplots(len(params), len(results), figsize=(3.8 * len(results), 3.5 * len(params)),
                            squeeze=False)
    rng = np.random.default_rng(0)
    for c, (label, res) in enumerate(results.items()):
        st = metrics.posterior_stats(cfg, res)
        idx = rng.choice(len(res["samples"]), min(max_points, len(res["samples"])), replace=False)
        col = _styles(results)[label]["color"]
        for r, p in enumerate(params):
            ax = axs[r, c]
            lo, hi = cfg.PRIORS[p]
            ax.errorbar(res["theta_true"][idx, r], st["med"][idx, r], yerr=st["std"][idx, r],
                        fmt="o", ms=2, color=col, alpha=0.25, elinewidth=0.8)
            ax.plot([lo, hi], [lo, hi], color="k", lw=1)
            ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel=f"true {p}", ylabel=f"predicted {p}" if c == 0 else None)
            if r == 0:
                ax.set_title(label)
    fig.tight_layout()
    return fig


def error_distributions(cfg, results):
    """Per parameter: |median-true|/prior, 2 sigma/prior and bias=(median-true)/sigma distributions.
    Values outside the x range are clipped, so the first/last bins also count the overflow."""
    params = next(iter(results.values()))["params"]
    styles = _styles(results)
    cols = [("err", r"$|\mathrm{median}-\mathrm{true}|/\Delta_\mathrm{prior}$", (0, 0.4)),
            ("width", r"$2\sigma/\Delta_\mathrm{prior}$", (0, 0.6)),
            ("bias", r"bias $(\mathrm{median}-\mathrm{true})/\sigma$", (-5, 5))]
    fig, axs = plt.subplots(len(params), 3, figsize=(14, 3.2 * len(params)), squeeze=False)
    for label, res in results.items():
        st = metrics.posterior_stats(cfg, res)
        for r, p in enumerate(params):
            for c, (key, xlabel, rng) in enumerate(cols):
                v = np.clip(st[key][:, r], *rng)          # out-of-range values pile up in the edge bins
                h, e = np.histogram(v, bins=40, range=rng)
                axs[r, c].plot(0.5 * (e[1:] + e[:-1]), h / len(v), label=label, **styles[label])
                axs[r, c].set(xlabel=xlabel if r == len(params) - 1 else None, ylabel=f"{p}: fraction")
    axs[0, 0].legend()
    fig.tight_layout()
    return fig


def coverage(cfg, results):
    """Calibration: fraction of true values below each credibility level (diagonal = calibrated)."""
    params = next(iter(results.values()))["params"]
    styles = _styles(results)
    fig, axs = plt.subplots(1, len(params), figsize=(4.2 * len(params), 4), squeeze=False)
    for label, res in results.items():
        st = metrics.posterior_stats(cfg, res)
        for r, p in enumerate(params):
            lv, cov = metrics.coverage_curve(st["rank"][:, r])
            axs[0, r].plot(lv, cov, label=label, **styles[label])
    for r, p in enumerate(params):
        axs[0, r].plot([0, 1], [0, 1], color="k", lw=1)
        axs[0, r].set(xlabel="credibility level", ylabel="empirical CDF of rank" if r == 0 else None, title=p,
                      aspect="equal", xlim=(0, 1), ylim=(0, 1))
    axs[0, 0].legend()
    fig.tight_layout()
    return fig


def metrics_vs_c(cfg, results):
    """Posterior metrics against the true C on the C-grid test set (grey = C seen in training)."""
    params = next(iter(results.values()))["params"]
    styles = _styles(results)
    cols = [("median_err", r"median $|\mathrm{err}|/\Delta_\mathrm{prior}$"),
            ("median_width", r"median $2\sigma/\Delta_\mathrm{prior}$"),
            ("mean_bias", "mean bias"),
            ("coverage_68", "68% interval coverage")]
    fig, axs = plt.subplots(len(params), len(cols), figsize=(16, 3.2 * len(params)), squeeze=False)
    for label, res in results.items():
        m = metrics.metrics_vs_c(cfg, res)
        for r, p in enumerate(params):
            for c, (key, _) in enumerate(cols):
                axs[r, c].plot(m["C"], m[key][:, r], marker="o", ms=3, label=label, **styles[label])
    for r, p in enumerate(params):
        for c, (key, ylabel) in enumerate(cols):
            ax = axs[r, c]
            _shade_training_c(cfg, ax)
            if key == "mean_bias":
                ax.axhline(0, color="k", lw=1)
            if key == "coverage_68":
                ax.axhline(0.68, color="k", lw=1)
            ax.set(xlabel="true $C$" if r == len(params) - 1 else None, ylabel=f"{p}: {ylabel}")
    axs[0, 0].legend()
    fig.tight_layout()
    return fig


def fallback_vs_c(cfg, results):
    """Mean rejection acceptance and fraction of observations that fell back to unrejected samples,
    against the true C (C-grid test set; grey = C ranges seen in training)."""
    styles = _styles(results)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    for label, res in results.items():
        m = metrics.metrics_vs_c(cfg, res)
        axs[0].plot(m["C"], m["mean_acceptance"], marker="o", ms=3, label=label, **styles[label])
        axs[1].plot(m["C"], m["fallback_frac"], marker="o", ms=3, label=label, **styles[label])
    for ax, ylabel in zip(axs, ["mean acceptance", "fraction with fallback"]):
        _shade_training_c(cfg, ax)
        ax.set(xlabel="true $C$", ylabel=ylabel, ylim=(-0.03, 1.03))
    axs[0].legend()
    fig.tight_layout()
    return fig

