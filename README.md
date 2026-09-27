# CL_mini_simple

A minimal, readable version of the `CL_mini` toy-model pipeline. It asks whether **contrastive learning (CL)**
can build a summary of a "power spectrum" that is insensitive to a nuisance ("baryonic") parameter. It then
checks whether **simulation-based inference (SBI)** on that summary generalises better to baryonic models
never seen in training than SBI on the raw spectrum.

## Results in this repository

`data/full/` and `outputs/full/` contain the results of a **full-size run**, and the nine notebooks are stored
**with their outputs**. You can look at the figures and metrics directly on GitHub, with nothing to re-run:
start with `09_compare_SBI.ipynb`, or browse `outputs/full/compare/` (PNG + `metrics.json`).

* **Not included:** the posterior samples (`outputs/full/sbi_*/samples_{val,test}.npz`, up to 126 MB each, over
  GitHub's file-size limit). To re-run notebooks 07–09, regenerate them from the committed posteriors (about
  12 min on a laptop CPU, see [Posterior sampling](#posterior-sampling)):
  `python -m clmini.sample_posteriors --kind pk` and `... --kind latents`.
* **Version-dependent files:** the saved posteriors (`outputs/full/sbi_*/posterior.pkl`) and the CL model
  (`outputs/full/cl/model.pt`) are pickled torch/sbi objects. Load them with the torch, sbi and nflows versions
  pinned in `requirements.txt` (torch 2.5.1, sbi 0.27.0, nflows 0.14, Python 3.10).
* The stored notebook outputs show absolute paths from the machine that ran them; they have no effect on
  re-running.

## The toy model

Each observation is a noisy 48-point "spectrum" `f(k)` on `k ∈ [0, 1]`. It is a smoothed broken line with a break at `k_baryons = 0.5`:

| regime | shape | parameter meaning |
|---|---|---|
| `k < 0.5` | `f = A·k + B` | **A** ∈ [−0.9, −0.1]: large-scale slope; **B** ∈ [1.6, 2.0]: amplitude ("cosmology") |
| `k > 0.5` | `f = B + 0.5·A + C·(k − 0.5)` | **C** ∈ [−2, 2]: small-scale slope ("baryons", the nuisance) |

Gaussian noise (σ from 0.05 at low k down to 1e‑4 at high k) is added, and every spectrum is standardised per k.

Two **baryonic models** are used for training. They are the same function and differ only in the range of C:
**v1**: C ∈ [−2.0, −1.7], **v2**: C ∈ [−0.3, 0.0]. Both use the *same* cosmologies (A, B), so the v1 and v2
spectra of one cosmology are the positive pairs for CL. The final test set evaluates every test cosmology on a
**regular grid of C over its whole prior**, including C values never seen in training.

## Installation

Using an existing conda env (e.g. `CL_inference`):

```bash
conda activate CL_inference
pip install -r requirements.txt   # pinned versions of the published run
```

Or a fresh environment:

```bash
python -m venv .venv && source .venv/bin/activate      # or: conda create -n clmini python=3.10
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu   # CPU wheels (instead of `conda -c pytorch`)
pip install -r requirements.txt
```

## Running the pipeline

Open the notebooks in Jupyter from this folder and run them **in order**. Each notebook is a short walkthrough
that calls functions from the `clmini/` package, shows the figures inline, and saves them as PNG (and metrics as
JSON) under `outputs/<run>/`. `<run>` is `full`, or `quick` in quick mode.

| # | Notebook | What it does | Reads | Writes |
|---|---|---|---|---|
| 01 | `01_generate_data.ipynb` | Generates v1/v2 train and validation sets and the C-grid test set; computes the normalisation | – | `data/<run>/pk/*.npz`, `data/<run>/norm.npz` |
| 02 | `02_inspect_data.ipynb` | Plots parameter coverage, example spectra (v1 vs v2), one cosmology across the C grid, and the noise level | `data/<run>/pk/` | `outputs/<run>/data/*.png` |
| 03 | `03_train_CL.ipynb` | Trains encoder + projector with the Weinberger loss (best validation model kept), then encodes every set | `data/<run>/pk/` | `outputs/<run>/cl/model.pt`, `history.json`, `loss.png`; `data/<run>/latents/*.npz` |
| 04 | `04_evaluate_CL.ipynb` | Latent clusters, latent plane coloured by A/B/C, latents vs parameters, invariance metric (with the raw spectra as a baseline) | `outputs/<run>/cl/`, `data/<run>/{pk,latents}/` | `outputs/<run>/cl/*.png`, `invariance.json` |
| 05 | `05_train_SBI_Pk.ipynb` | **Model A**: NPE (MAF + MLP embedding) trained on the v1+v2 spectra | `data/<run>/pk/` | `outputs/<run>/sbi_pk/posterior.pkl`, `history.json`, `loss.png` |
| 06 | `06_train_SBI_latents.ipynb` | **Model B**: the same NPE trained on the CL latents | `data/<run>/latents/` | `outputs/<run>/sbi_latents/…` (same files) |
| – | `python -m clmini.sample_posteriors --kind pk` and `--kind latents` (terminal) | **Heavy step:** draws the posterior samples for validation (v1+v2) and the C-grid test set, chunk by chunk (resumable, bounded memory). See [Posterior sampling](#posterior-sampling). | `outputs/<run>/sbi_<kind>/posterior.pkl`, `data/<run>/<kind>/` | `outputs/<run>/sbi_<kind>/samples_{val,test}.npz` |
| 07 | `07_evaluate_SBI_Pk.ipynb` | Loads the Model A samples (computes them if missing); shows acceptance/fallback vs C, true vs predicted, error/width/bias distributions, coverage and metrics vs C | `outputs/<run>/sbi_pk/` | `outputs/<run>/sbi_pk/metrics.json`, `*.png` |
| 08 | `08_evaluate_SBI_latents.ipynb` | The same evaluation for Model B | `outputs/<run>/sbi_latents/` | `outputs/<run>/sbi_latents/metrics.json`, `*.png` |
| 09 | `09_compare_SBI.ipynb` | Model A vs Model B side by side (colour = model, dashed = C-grid test), including acceptance/fallback and every metric as a function of the true C | samples of both models | `outputs/<run>/compare/metrics.json`, `*.png` |

To run everything non-interactively (notebooks 01–06, the sampling step, then 07–09; executed copies go to
`outputs/<run>/executed_notebooks/`):

```bash
./run_pipeline.sh            # full run
./run_pipeline.sh --quick    # smoke test, about 1 minute on a laptop CPU
```

## Posterior sampling

Drawing posterior samples for every evaluation observation is the expensive part: 4096 validation and
10496 test observations × 1000 samples at full size. Run it from a terminal after notebooks 05 and 06, so it
does not depend on Jupyter/VS Code:

```bash
cd CL_mini_simple
nohup python -m clmini.sample_posteriors --kind pk      > sample_pk.log 2>&1 &
nohup python -m clmini.sample_posteriors --kind latents > sample_latents.log 2>&1 &   # can run in parallel
tail -f sample_pk.log                                    # progress bar + one line per chunk
```

Options: `--split val|test|all`, `--quick` (quick-mode config), `--overwrite` (discard existing samples),
`--max-chunks N` (stop after N new chunks, e.g. to time a subset).

* **Bounded memory.** Observations are processed in chunks of `EVAL["chunk_size"]` (64). Each chunk is
  written to `outputs/<run>/sbi_<kind>/chunks_<split>/` as soon as it is done. The measured peak is about
  0.5 GB.
* **Resumable.** Re-running the same command skips finished chunks. When all chunks exist, they are merged
  into `samples_<split>.npz` and the chunk folder is removed. Chunks and samples are only reused if the
  posterior file and the `SBI_PARAMS`/`EVAL` settings are unchanged; otherwise you are asked for `--overwrite`.
  Notebooks 07/08 recompute automatically in that case, with a message.
* **Rejection and fallback.** Samples outside the prior are rejected, i.e. the posterior is the NPE estimate
  truncated to the prior. Far outside the training C ranges, the Pk posterior puts almost no mass inside the
  prior (acceptance ≈ 0 at C = 2, mostly because A leaves its prior range). An observation with acceptance
  below `EVAL["min_acceptance"]` (1 %) keeps the **raw, unrejected, unclipped** samples and is flagged as a
  *fallback*. These samples are not clipped, because clipping to the prior would put spikes on the boundary and produce
  artificial medians, widths and coverage. Raw samples show what the estimator actually predicts; the flag
  marks those posteriors as unreliable. The fallback counts per C value are in `metrics.json`
  (`n_obs_fallback`, `fallback_per_C`) and are plotted in notebooks 07–09.
* **Runtime** (16-core laptop CPU, full defaults): `pk` ≈ 7–8 min (val ≈ 1 min, test ≈ 6.5 min; about 27 % of
  test observations fall back); `latents` ≈ 4 min. The time grows with `n_samples` × number of
  observations.

## Where things end up

```
data/<run>/        norm.npz, pk/*.npz (spectra), latents/*.npz (CL latents)
outputs/<run>/     data/ (inspection plots), cl/ (CL model + evaluation),
                   sbi_pk/, sbi_latents/ (posteriors, samples, metrics, plots), compare/
```

Git ignores the posterior samples (`samples_*.npz`, too large), the sampling chunks, and the quick-mode
folders `data/quick/` and `outputs/quick/`.

Each dataset `.npz` holds `cosmos (n_cosmo, 2)` = (A, B), `aug (n_cosmo, n_aug, 1)` = C and
`xx (n_noise, n_cosmo, n_aug, dim)`. These shapes match the original `CL_mini`.

## Changing the settings

All settings live in **`config.py`** (a plain Python module, so there are no YAML string/None/float pitfalls).

| Setting | How |
|---|---|
| SBI target parameters | `SBI_PARAMS = ["A", "B", "C"]` (default) or `["A", "B"]`. The prior, data loading, metrics and every plot follow it. After changing it, re-run notebooks 05–09. A stale posterior raises a clear error. |
| Quick mode | `export CLMINI_QUICK=1` before starting Jupyter, or set `QUICK = True`. Quick runs use small datasets and few epochs, and write to `data/quick`, `outputs/quick`. |
| Device | `DEVICE = "cpu"` (default) or `"cuda"` |
| Seeds | `SEED` (CL and SBI training). Dataset seeds live in `TRAIN/VAL/NORM/TEST_GRID["seed"]` (the original values 137 / 1 / 2 / 2). Noise, batches and network initialisation are all seeded. |
| Evaluation cost | `EVAL`: `n_samples` (posterior samples per observation), `n_test_cosmo` and `n_c` (use a subset of the generated test grid; `None` = all), `n_noise` (noise realisations used), `chunk_size`, `min_acceptance` |
| Sizes / hyperparameters | `TRAIN`, `VAL`, `TEST_GRID` (what is *generated*, e.g. `n_c` = number of C grid points), `CL`, `SBI` |

## Package layout

```
config.py            all settings
clmini/toy_model.py  toy model, Latin-hypercube sampling, dataset generation
clmini/data.py       generate/save/load datasets, flattening for SBI, CL views
clmini/cl.py         MLPs, Weinberger loss, CL training, latent export
clmini/sbi_tools.py  NPE training (sbi.inference.NPE), loading posteriors and samples
clmini/sample_posteriors.py  chunked, resumable posterior sampling (also the command-line tool)
clmini/metrics.py    invariance and posterior metrics, save_fig / save_json
clmini/plots.py      every figure
```

## Differences from the original `CL_mini`

* **Kept, identical:** toy model, priors, v1/v2 C ranges, k grid, noise model, Latin-hypercube seeds (the
  generated cosmologies and C values match the original files exactly), data shapes, the Weinberger loss (same
  normalisation and hyperparameters), and the SBI settings of the `SBI_*_single_run.yaml` configs.
* **CL architecture and optimiser:** hand-tuned (`config.CL`: encoder [48, 32, 24] → 2, projector
  [24, 24, 32] → 32, 10 × 64 batches of 1024, lr 5e-3). The original code used wandb run *easy-sweep-5*
  (encoder [48], projector [24, 24], lr 4.4e-3, 50 × 128 batches of 256).
* **Now reproducible:** the noise, CL batches, network initialisation and SBI training are seeded. The
  original drew the noise without a seed and seeded batches from the clock.
* **Test set:** a regular C grid over [−2, 2] (256 cosmologies × 41 C values) replaces the random-C "Modelall"
  set. In-distribution evaluation uses the v1+v2 validation sets (the original used the TEST v1/v2 sets).
* **Latent normalisation:** uses the mean/std of the *training* latents (the original used the test set).
* **Also saved:** the projector (alongside the encoder), and the loss histories.
* **Single posterior:** one posterior per model (the original evaluated an ensemble of the best wandb runs).
  Sampling uses per-observation rejection with an acceptance threshold and a raw-sample fallback (see
  [Posterior sampling](#posterior-sampling)). The original used a 60 s per-observation timeout and dropped
  those observations.
* **New:** an invariance metric for the latents, and all posterior metrics as a function of C.
* **Dropped:** BACCO emulator, wandb and sweeps, VICReg/SimCLR losses, other training modes, other ks/kcut
  values, and the non-linear toy-model variant.
