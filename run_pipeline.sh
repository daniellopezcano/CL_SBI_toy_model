#!/usr/bin/env bash
# Execute the notebooks 01-09 in order, non-interactively.
#   ./run_pipeline.sh           full run
#   ./run_pipeline.sh --quick   smoke test (small data, few epochs), written to data/quick and outputs/quick
# Executed copies (with outputs) go to outputs/<run>/executed_notebooks/; the source notebooks are untouched.
set -euo pipefail
cd "$(dirname "$0")"
RUN=full
if [[ "${1:-}" == "--quick" ]]; then export CLMINI_QUICK=1; RUN=quick; fi
OUT="outputs/$RUN/executed_notebooks"
mkdir -p "$OUT"
run_nb() {
    echo "=== $1"
    jupyter nbconvert --to notebook --execute "$1" --output-dir "$OUT" \
        --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3
}
for nb in 0[1-6]_*.ipynb; do run_nb "$nb"; done
# Heavy step outside Jupyter (resumable; notebooks 07-08 then just load the samples)
for kind in pk latents; do
    echo "=== sampling posteriors: $kind"
    python -m clmini.sample_posteriors --kind "$kind"
done
for nb in 0[7-9]_*.ipynb; do run_nb "$nb"; done
echo "done: outputs in outputs/$RUN"
