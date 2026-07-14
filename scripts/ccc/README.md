# CCC job scripts — Granite DSpark proof-of-concept

Scripts to run the Granite-4.1-8B DSpark speculator pipeline on the IBM CCC
(LSF) cluster on **2× A100-80GB**, sized for a **~1 TB** target cache.

All paths assume the repo and cache live on GPFS (home is only 25 GB):

```
REPO=/dccstor/galbloch/DeepSpec
CACHE=/dccstor/galbloch/granite_cache/granite_4_1_8b_target_cache
```

## One-time setup on CCC

```bash
# from an ssh session on any reachable ccc-login node
cd /dccstor/galbloch
git clone -b add-granite-dspark https://github.com/Gal-bloch/DeepSpec.git
cd DeepSpec
source $(conda info --base)/etc/profile.d/conda.sh
conda create -y -n granite python=3.13 && conda activate granite
pip install -r requirements.txt
pip install "sglang[all]"          # serving engine for the regen step
```

## Pipeline (run in order)

Each stage is a `bsub` batch job. Submit with the wrapper:

```bash
bash scripts/ccc/submit.sh data     # steps 1-3: download subset, regen, build cache
bash scripts/ccc/submit.sh train    # step 4: train the draft on 2x A100
```

Or submit the individual job scripts directly (see `submit.sh` for the exact
`bsub` flags). Watch progress with `bjobs` and tail the `%J.stdout` files in
`$HOME`. Each job prints `===JOB_COMPLETE===` as its final line — poll for that
marker before reading results (LSF output is GPFS-buffered and lags DONE).

## Sizing notes

- `--sample-size 11000` → ~10k train samples after the 5% eval split. This is
  the lever that keeps the cache under 1 TB (per-token cache ~48 KB for this
  target; verify actual size after a small dry run before trusting the estimate).
- `max_length=2048` and `target_layer_ids=[2,11,20,29,38]` are set in
  `config/dspark/dspark_granite_4_1_8b.py`. Do NOT drop layers to save space —
  shrink `--sample-size` instead.
- Regen uses greedy decoding (`temperature 0`) for clean, deterministic target
  answers. Adjust in `01_regen.sh` if you want sampled targets.
