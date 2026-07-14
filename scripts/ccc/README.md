# CCC job scripts — Granite DSpark proof-of-concept

Scripts to run the Granite-4.1-8B DSpark speculator pipeline on the IBM CCC
(LSF) cluster on **2× A100-80GB**. Proof-of-concept sized for a **~100 GB**
target cache (the `/dccstor/knewedge` fileset is shared and often near-full).

All paths live on GPFS (home quota is small and usually full):

```
REPO=/dccstor/knewedge/galbloch/DeepSpec
CACHE=/dccstor/knewedge/galbloch/granite_cache/granite_4_1_8b_target_cache
```

## One-time setup on CCC

```bash
# from an ssh session on a reachable ccc-login node
cd /dccstor/knewedge/galbloch
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

- `SAMPLE_SIZE=2200` → ~2k train samples after the 5% eval split, targeting a
  ~100 GB cache (per-token cache ~48 KB for this target; `00_data.sh` prints the
  real size and refuses to build if free space < `MIN_FREE_MB`, default 40 GB).
- `max_length=2048` and `target_layer_ids=[2,11,20,29,38]` (all 5 layers kept)
  are set in `config/dspark/dspark_granite_4_1_8b.py`, `num_train_epochs=3`.
  This is a PoC to confirm a nonzero acceptance rate, not a quality checkpoint.
  To save space, shrink `SAMPLE_SIZE` — do NOT drop target layers.
- Regen uses greedy decoding (`temperature 0`) for clean, deterministic target
  answers. Adjust in `00_data.sh` if you want sampled targets.
