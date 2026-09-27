"""
Neural posterior estimation (NPE with a MAF density estimator) with the `sbi` package.

kind = "pk"      -> Model A: posterior trained on the normalised spectra
kind = "latents" -> Model B: posterior trained on the CL latents

Files written to cfg.OUT_DIR / f"sbi_{kind}":
    posterior.pkl            trained posterior + the parameter names it infers
    history.json             sbi training / validation loss per epoch
    samples_{val,test}.npz   posterior samples for the evaluation sets (written by sample_posteriors.py)
"""
import json
import pickle
import numpy as np
import torch
from torch import nn
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn
from sbi.utils import BoxUniform

from . import data

TRAIN_SETS = ["train_v1", "train_v2"]


def out_dir(cfg, kind):
    d = cfg.OUT_DIR / f"sbi_{kind}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def make_prior(cfg):
    low = torch.tensor([cfg.PRIORS[p][0] for p in cfg.SBI_PARAMS], dtype=torch.float32)
    high = torch.tensor([cfg.PRIORS[p][1] for p in cfg.SBI_PARAMS], dtype=torch.float32)
    return BoxUniform(low=low, high=high, device=cfg.DEVICE)


def embedding_net(dim_in, layers):
    """MLP summary network (ReLU between layers, linear output), as SummaryNet in the original code."""
    mods, d = [], dim_in
    for h in layers:
        mods += [nn.Linear(d, h), nn.ReLU()]
        d = h
    return nn.Sequential(*mods[:-1])


def train_npe(cfg, kind):
    """Train an NPE posterior for cfg.SBI_PARAMS on the v1+v2 train set of `kind` and save it."""
    torch.manual_seed(cfg.SEED)
    torch.set_num_threads(cfg.N_THREADS)
    x, theta_all = data.load_flat(cfg, TRAIN_SETS, kind)
    theta = data.select_params(cfg, theta_all)
    print(f"[{kind}] training NPE on x {x.shape} -> theta {cfg.SBI_PARAMS} {theta.shape}")

    s = cfg.SBI
    density_estimator = posterior_nn(
        model="maf", hidden_features=s["hidden_features"], num_transforms=s["num_transforms"],
        num_blocks=s["num_blocks"], embedding_net=embedding_net(x.shape[1], s["embedding_layers"]))
    inference = NPE(prior=make_prior(cfg), density_estimator=density_estimator, device=cfg.DEVICE)
    to_t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=cfg.DEVICE)
    estimator = inference.append_simulations(to_t(theta), to_t(x)).train(
        training_batch_size=s["batch_size"], learning_rate=s["lr"],
        validation_fraction=s["validation_fraction"], max_num_epochs=s["max_num_epochs"],
        stop_after_epochs=s["stop_after_epochs"], show_train_summary=True)
    posterior = inference.build_posterior(estimator)

    d = out_dir(cfg, kind)
    with open(d / "posterior.pkl", "wb") as f:
        pickle.dump(dict(posterior=posterior, params=list(cfg.SBI_PARAMS)), f)
    history = {k: [float(v) for v in inference.summary[k]] for k in ["training_loss", "validation_loss"]}
    (d / "history.json").write_text(json.dumps(history, indent=1))
    return posterior, history


def load_posterior(cfg, kind):
    with open(cfg.OUT_DIR / f"sbi_{kind}" / "posterior.pkl", "rb") as f:
        saved = pickle.load(f)
    if saved["params"] != list(cfg.SBI_PARAMS):
        raise ValueError(f"Posterior for '{kind}' was trained on {saved['params']} but config.SBI_PARAMS = "
                         f"{cfg.SBI_PARAMS}. Re-run the SBI training notebook.")
    return saved["posterior"]


def load_history(cfg, kind):
    return json.loads((cfg.OUT_DIR / f"sbi_{kind}" / "history.json").read_text())


def load_samples(cfg, kind, split):
    """Posterior samples written by sample_posteriors.py: samples (N, n, P), theta_true (N, P),
    theta_all (N, 3) = (A, B, C), acceptance (N,), fallback (N,) and the params list."""
    with np.load(cfg.OUT_DIR / f"sbi_{kind}" / f"samples_{split}.npz") as f:
        res = {k: f[k] for k in f.files if k != "meta"}
    res["params"] = [str(p) for p in res["params"]]
    if res["params"] != list(cfg.SBI_PARAMS):
        raise ValueError(f"Samples for '{kind}' were drawn for {res['params']}, config has {cfg.SBI_PARAMS}. "
                         "Re-run the SBI training and posterior sampling.")
    return res
