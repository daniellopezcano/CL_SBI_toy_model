"""
Posterior sampling for evaluation: the heavy step, runnable from a terminal.

    python -m clmini.sample_posteriors --kind pk            # Model A (spectra)
    python -m clmini.sample_posteriors --kind latents       # Model B (CL latents)
    nohup python -m clmini.sample_posteriors --kind pk > sample_pk.log 2>&1 &

Options: --split {val,test,all}  --quick  --overwrite  --max-chunks N (stop early, e.g. for timing).

How it works
* Observations are processed in chunks of cfg.EVAL["chunk_size"]; each chunk is written to
  outputs/<run>/sbi_<kind>/chunks_<split>/chunk_XXXXX.npz as soon as it is done, so memory stays bounded
  and an interrupted run resumes from the last finished chunk. When all chunks exist they are merged into
  samples_<split>.npz and the chunk folder is removed.
* Chunks are only reused if they were made with the same posterior file and evaluation settings
  (stored in meta.json); otherwise the run stops and asks for --overwrite.
* Rejection sampling (keep only samples inside the prior = the NPE posterior truncated to the prior) is done
  per observation: new proposals are drawn only for observations that still need samples, with at most
  chunk_size * n_samples proposals per round (the same memory as the first round).
* Fallback: an observation whose acceptance is below cfg.EVAL["min_acceptance"] (or that exhausts its
  budget of n_samples / min_acceptance proposals) keeps the raw, unrejected estimator samples. These are not
  clipped: clipping would pile mass on the prior boundary and create artificial medians/widths. The
  observation is flagged (`fallback`) and counted per C in metrics.json.
"""
import argparse
import hashlib
import json
import os
import shutil
import time
import warnings

import numpy as np
import torch
from tqdm.auto import tqdm

from . import data, sbi_tools

SPLITS = {"val": ["val_v1", "val_v2"], "test": ["test_grid"]}

# We sample without sbi's rejection on purpose and do the rejection/acceptance bookkeeping ourselves.
warnings.filterwarnings("ignore", message=".*lie outside the prior support.*")
warnings.filterwarnings("ignore", message="Capping max_sampling_batch_size")


# ----------------------------------------------------------------------------- #
# Evaluation data (with the subsetting of cfg.EVAL)
# ----------------------------------------------------------------------------- #
def subset(cfg, d, split):
    e = cfg.EVAL
    xx, cosmos, aug = d["xx"][: e["n_noise"]], d["cosmos"], d["aug"]
    if split == "test":
        n_cosmo = e["n_test_cosmo"] or len(cosmos)
        n_c_all = aug.shape[1]
        c_idx = np.unique(np.round(np.linspace(0, n_c_all - 1, e["n_c"] or n_c_all)).astype(int))
        xx, cosmos, aug = xx[:, :n_cosmo][:, :, c_idx], cosmos[:n_cosmo], aug[:n_cosmo][:, c_idx]
    return dict(xx=xx, cosmos=cosmos, aug=aug)


def eval_data(cfg, kind, split):
    """x (N, dim) and theta_all (N, 3) for the evaluation set `split` ('val' or 'test')."""
    xs, ths = zip(*(data.flatten(subset(cfg, data.load_set(cfg, n, kind), split)) for n in SPLITS[split]))
    return np.concatenate(xs), np.concatenate(ths)


# ----------------------------------------------------------------------------- #
# Sampling
# ----------------------------------------------------------------------------- #
@torch.no_grad()
def sample_chunk(cfg, posterior, x, low, high):
    """Rejection-sample cfg.EVAL['n_samples'] per observation. Returns samples (n_obs, n, P),
    acceptance (n_obs,) and fallback flags (n_obs,)."""
    e = cfg.EVAL
    n, n_obs = e["n_samples"], len(x)
    max_per_round = e["chunk_size"] * n                       # proposals per round: bounds memory
    budget = int(np.ceil(n / e["min_acceptance"]))            # proposals per observation
    x = torch.as_tensor(x, dtype=torch.float32, device=cfg.DEVICE)
    kept = [[] for _ in range(n_obs)]
    n_acc, n_inside, n_prop = np.zeros(n_obs, int), np.zeros(n_obs, int), np.zeros(n_obs, int)
    raw = None
    todo, m = np.arange(n_obs), n
    while len(todo):
        s = posterior.sample_batched((m,), x=x[todo], reject_outside_prior=False, show_progress_bars=False)
        s = s.permute(1, 0, 2)                                           # (len(todo), m, P)
        inside = ((s >= low) & (s <= high)).all(-1).cpu().numpy()
        s = s.cpu().numpy()
        if raw is None:
            raw = s[:, :n].copy()                                        # first round: all observations
        for j, i in enumerate(todo):
            good = s[j][inside[j]][: n - n_acc[i]]
            kept[i].append(good)
            n_acc[i] += len(good)
            n_inside[i] += inside[j].sum()
            n_prop[i] += m
        rate = n_inside / np.maximum(n_prop, 1)                          # acceptance = inside / proposed
        todo = np.where((n_acc < n) & (n_prop < budget) & (rate >= e["min_acceptance"]))[0]
        if len(todo):
            need = (n - n_acc[todo]) / np.maximum(rate[todo], e["min_acceptance"])
            m = int(np.clip(np.ceil(1.2 * need.max()), 100, max_per_round // len(todo)))
    fallback = n_acc < n
    samples = np.stack([raw[i] if fallback[i] else np.concatenate(kept[i]) for i in range(n_obs)])
    return samples.astype(np.float32), rate.astype(np.float32), fallback


def signature(cfg, kind, n_obs):
    """Everything that must match for saved chunks/samples to be reused."""
    h = hashlib.sha1((cfg.OUT_DIR / f"sbi_{kind}" / "posterior.pkl").read_bytes()).hexdigest()
    return dict(posterior_sha1=h, params=list(cfg.SBI_PARAMS), n_obs=int(n_obs), seed=cfg.SEED,
                **{k: v for k, v in cfg.EVAL.items()})


def sample_split(cfg, kind, split, overwrite=False, max_chunks=None):
    """Sample one evaluation set chunk by chunk (resumable). Returns the path of samples_<split>.npz,
    or None if stopped early by max_chunks."""
    out = sbi_tools.out_dir(cfg, kind)
    final, chunk_dir = out / f"samples_{split}.npz", out / f"chunks_{split}"
    x, theta_all = eval_data(cfg, kind, split)
    sig = signature(cfg, kind, len(x))

    if final.exists() and not overwrite:
        if stored_signature(final) == sig:
            print(f"[{kind}/{split}] {final.name} already exists and matches the settings -> skipped")
            return final
        raise RuntimeError(f"{data.rel(cfg, final)} was made with different settings/posterior. Re-run with --overwrite.")
    if overwrite:
        final.unlink(missing_ok=True)
        shutil.rmtree(chunk_dir, ignore_errors=True)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    meta = chunk_dir / "meta.json"
    if meta.exists() and json.loads(meta.read_text()) != sig:
        raise RuntimeError(f"Chunks in {data.rel(cfg, chunk_dir)} were made with different settings/posterior. "
                           "Re-run with --overwrite.")
    meta.write_text(json.dumps(sig, indent=1))

    posterior = sbi_tools.load_posterior(cfg, kind)
    prior = sbi_tools.make_prior(cfg)
    low, high = prior.base_dist.low, prior.base_dist.high
    cs = cfg.EVAL["chunk_size"]
    starts = list(range(0, len(x), cs))
    done = [i for i in range(len(starts)) if (chunk_dir / f"chunk_{i:05d}.npz").exists()]
    print(f"[{kind}/{split}] {len(x)} observations x {cfg.EVAL['n_samples']} samples, {len(starts)} chunks "
          f"of {cs} ({len(done)} already done)")
    todo = [i for i in range(len(starts)) if i not in done][:max_chunks]
    t_start = time.time()
    for i in tqdm(todo, desc=f"{kind}/{split}", unit="chunk"):
        torch.manual_seed(cfg.SEED + 100_000 * (split == "test") + i)   # same result whether resumed or not
        t0 = time.time()
        s, acc, fb = sample_chunk(cfg, posterior, x[starts[i]:starts[i] + cs], low, high)
        np.savez(chunk_dir / f"chunk_{i:05d}.npz", samples=s, acceptance=acc, fallback=fb)
        tqdm.write(f"  chunk {i:4d}: {time.time() - t0:5.1f} s | acceptance mean {acc.mean():.3f} "
                   f"min {acc.min():.3f} | fallback {int(fb.sum())}/{len(fb)}")
    print(f"[{kind}/{split}] {len(todo)} chunks in {time.time() - t_start:.0f} s")

    if len(done) + len(todo) < len(starts):
        print(f"[{kind}/{split}] stopped early: {len(done) + len(todo)}/{len(starts)} chunks on disk "
              "(run again to resume)")
        return None
    parts = [np.load(chunk_dir / f"chunk_{i:05d}.npz") for i in range(len(starts))]
    cat = lambda key: np.concatenate([p[key] for p in parts])
    np.savez(final, samples=cat("samples"), acceptance=cat("acceptance"), fallback=cat("fallback"),
             theta_true=data.select_params(cfg, theta_all), theta_all=theta_all,
             params=np.array(cfg.SBI_PARAMS), meta=json.dumps(sig))
    shutil.rmtree(chunk_dir)
    fb = cat("fallback")
    print(f"[{kind}/{split}] saved {data.rel(cfg, final)} | fallback for {int(fb.sum())}/{len(fb)} observations")
    return final


def stored_signature(path):
    with np.load(path) as f:
        return json.loads(str(f["meta"])) if "meta" in f.files else None


def samples_ready(cfg, kind, split):
    path = cfg.OUT_DIR / f"sbi_{kind}" / f"samples_{split}.npz"
    return path.exists() and stored_signature(path) == signature(cfg, kind, len(eval_data(cfg, kind, split)[0]))


def load_or_sample(cfg, kind):
    """For notebooks 07/08: load saved samples if they match the current settings, otherwise compute them."""
    results = {}
    for split in SPLITS:
        if samples_ready(cfg, kind, split):
            print(f"[{kind}/{split}] loading saved posterior samples")
        else:
            print(f"[{kind}/{split}] no up-to-date samples found -> computing them now. This is the slow step; "
                  f"for full runs prefer a terminal:  python -m clmini.sample_posteriors --kind {kind}")
            try:
                sample_split(cfg, kind, split)
            except RuntimeError as err:                 # stale samples/chunks from other settings
                print(f"  {err}\n  -> discarding them and recomputing")
                sample_split(cfg, kind, split, overwrite=True)
        results[split] = sbi_tools.load_samples(cfg, kind, split)
    return results


def main():
    p = argparse.ArgumentParser(description="Sample the SBI posteriors on the evaluation sets.")
    p.add_argument("--kind", required=True, choices=["pk", "latents"])
    p.add_argument("--split", default="all", choices=["val", "test", "all"])
    p.add_argument("--quick", action="store_true", help="use the quick-mode configuration")
    p.add_argument("--overwrite", action="store_true", help="discard existing samples/chunks")
    p.add_argument("--max-chunks", type=int, default=None, help="stop after N new chunks per split")
    a = p.parse_args()
    if a.quick:
        os.environ["CLMINI_QUICK"] = "1"
    import config as cfg                                # imported after --quick is applied
    torch.set_num_threads(cfg.N_THREADS)
    print(f"run = {cfg.RUN_NAME} | params = {cfg.SBI_PARAMS} | EVAL = {cfg.EVAL}")
    for split in (SPLITS if a.split == "all" else [a.split]):
        sample_split(cfg, a.kind, split, overwrite=a.overwrite, max_chunks=a.max_chunks)


if __name__ == "__main__":
    main()
