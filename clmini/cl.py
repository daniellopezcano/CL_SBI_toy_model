"""
Contrastive learning: encoder/projector MLPs, Weinberger loss, training loop and latent export.

Files written to cfg.OUT_DIR / "cl":
    model.pt       best encoder + projector state dicts (lowest validation loss) and architecture
    history.json   train/val loss and learning rate per epoch
Latents are written to cfg.DATA_DIR / "latents" (see data.py).
"""
import json
import numpy as np
import torch
from torch import nn

from . import data


def set_seed(seed, n_threads=None):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_threads:
        torch.set_num_threads(n_threads)


def mlp(dim_in, hidden, dim_out, bn=True):
    """Linear -> BatchNorm -> ReLU for each hidden layer, then a bias-free linear output layer.
    `hidden` must be a flat list of ints, e.g. [24, 24] (not [[24, 24]])."""
    assert all(isinstance(h, int) for h in hidden), f"hidden layers must be a flat list of ints, got {hidden}"
    layers, d = [], dim_in
    for h in hidden:
        layers += [nn.Linear(d, h)] + ([nn.BatchNorm1d(h)] if bn else []) + [nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, dim_out, bias=False))
    return nn.Sequential(*layers)


def build_models(cfg, dim_in):
    c = cfg.CL
    encoder = mlp(dim_in, c["hidden_encoder"], c["dim_latent"])
    projector = mlp(c["dim_latent"], c["hidden_projector"], c["dim_projection"])
    return encoder.to(cfg.DEVICE), projector.to(cfg.DEVICE)


def weinberger_loss(z, delta_pull=0.5, delta_push=1.5, c_pull=1.0, c_push=1.0, c_reg=0.001, eps=1e-8):
    """Weinberger (discriminative) loss, https://arxiv.org/abs/1708.02551.
    z: (batch, n_views, dim). Views of one cosmology form a cluster:
      pull: views closer than delta_pull to their cluster centre are free, farther ones are pulled in;
      push: cluster centres closer than 2*delta_push are pushed apart;
      reg : keeps cluster centres near the origin.
    Normalised by batch * n_views, as in the original code."""
    B, V, _ = z.shape
    centres = z.mean(dim=1, keepdim=True)                                         # (B, 1, D)
    d_pull = torch.sqrt(((z - centres) ** 2).sum(-1) + eps)                       # (B, V)
    L_pull = torch.clamp(d_pull - delta_pull, min=0).pow(2).sum()
    c = centres[:, 0]
    d_push = torch.sqrt(((c[:, None] - c[None]) ** 2).sum(-1) + eps)              # (B, B)
    L_push = torch.triu(torch.clamp(2 * delta_push - d_push, min=0).pow(2), diagonal=1).sum()
    L_reg = torch.sqrt((c ** 2).sum(-1) + eps).sum() / B
    return (c_pull * L_pull + c_push * L_push + c_reg * L_reg) / (B * V)


def batch_loss(cfg, encoder, projector, xb):
    B, V, dim = xb.shape
    z = projector(encoder(xb.reshape(B * V, dim))).reshape(B, V, -1)
    return weinberger_loss(z, **cfg.CL["loss"])


def train_cl(cfg):
    """Train encoder + projector on the v1/v2 train set; keep the model with the lowest val loss."""
    set_seed(cfg.SEED, cfg.N_THREADS)
    c = cfg.CL
    X_train = data.cl_views(data.load_set(cfg, "train_v1"), data.load_set(cfg, "train_v2"))
    X_val = data.cl_views(data.load_set(cfg, "val_v1"), data.load_set(cfg, "val_v2"))
    to_t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=cfg.DEVICE)

    encoder, projector = build_models(cfg, X_train.shape[-1])
    params = [*encoder.parameters(), *projector.parameters()]
    opt = torch.optim.AdamW(params, lr=c["lr"], weight_decay=c["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", patience=20, threshold=0.01, threshold_mode="abs", factor=0.3, min_lr=1e-8)

    rng = np.random.default_rng(cfg.SEED)
    # Fixed monitoring batches (the original evaluated a fixed-seed batch of N_val/6 cosmologies)
    n_mon = max(2, int(len(X_val) * c["val_batch_fraction"]))
    mon_train = to_t(X_train[rng.choice(len(X_train), n_mon, replace=False)])
    mon_val = to_t(X_val[rng.choice(len(X_val), n_mon, replace=False)])

    history, best = {"train": [], "val": [], "lr": []}, None
    for epoch in range(c["n_epochs"]):
        encoder.train(); projector.train()
        for _ in range(c["n_batches_per_epoch"]):
            idx = rng.choice(len(X_train), min(c["batch_size"], len(X_train)), replace=False)
            loss = batch_loss(cfg, encoder, projector, to_t(X_train[idx]))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=c["clip_grad_norm"])
            opt.step()

        encoder.eval(); projector.eval()
        with torch.no_grad():
            l_train = batch_loss(cfg, encoder, projector, mon_train).item()
            l_val = batch_loss(cfg, encoder, projector, mon_val).item()
        sched.step(l_val)
        history["train"].append(l_train); history["val"].append(l_val)
        history["lr"].append(opt.param_groups[0]["lr"])
        print(f"epoch {epoch + 1:3d}/{c['n_epochs']}  train {l_train:.4f}  val {l_val:.4f}")
        if best is None or l_val < best:
            best = l_val
            save_models(cfg, encoder, projector, X_train.shape[-1])

    out = cfg.OUT_DIR / "cl"
    (out / "history.json").write_text(json.dumps(history, indent=1))
    return history


def save_models(cfg, encoder, projector, dim_in):
    out = cfg.OUT_DIR / "cl"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(dict(encoder=encoder.state_dict(), projector=projector.state_dict(), dim_in=dim_in, cl=cfg.CL),
               out / "model.pt")


def load_models(cfg):
    ckpt = torch.load(cfg.OUT_DIR / "cl" / "model.pt", map_location=cfg.DEVICE, weights_only=False)
    encoder, projector = build_models(cfg, ckpt["dim_in"])
    encoder.load_state_dict(ckpt["encoder"]); projector.load_state_dict(ckpt["projector"])
    return encoder.eval(), projector.eval()


def load_history(cfg):
    return json.loads((cfg.OUT_DIR / "cl" / "history.json").read_text())


@torch.no_grad()
def encode(cfg, encoder, xx):
    """Apply the encoder to an array (..., n_k) -> (..., dim_latent)."""
    x = torch.as_tensor(xx.reshape(-1, xx.shape[-1]), dtype=torch.float32, device=cfg.DEVICE)
    return encoder(x).cpu().numpy().reshape(xx.shape[:-1] + (-1,))


def export_latents(cfg):
    """Encode every dataset with the best encoder, standardise the latents with the train (v1+v2)
    mean/std, and save them in the same format as the spectra (cfg.DATA_DIR / 'latents')."""
    encoder, _ = load_models(cfg)
    raw = {name: encode(cfg, encoder, data.load_set(cfg, name)["xx"]) for name in data.SETS}
    train = np.concatenate([raw[n].reshape(-1, raw[n].shape[-1]) for n in ["train_v1", "train_v2"]])
    mean, std = train.mean(0), train.std(0)
    np.savez(cfg.OUT_DIR / "cl" / "latent_norm.npz", mean=mean, std=std)
    for name, h in raw.items():
        d = data.load_set(cfg, name)
        data.save_set(cfg, name, dict(cosmos=d["cosmos"], aug=d["aug"], xx=(h - mean) / std), kind="latents")
    print("latents saved to", data.rel(cfg, cfg.DATA_DIR / "latents"))
