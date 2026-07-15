"""Minimal acceptance probe for the Granite DSpark drafter.

Bypasses eval.py's torch.multiprocessing.spawn path (which hung on CCC before
producing any result) by initializing a single-process distributed group and
driving the evaluator directly over a few gsm8k samples. Prints the acceptance
length — the PoC's nonzero-acceptance signal.
"""
import os
import argparse
import torch
import torch.distributed as dist
from types import SimpleNamespace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--task", default="gsm8k")
    ap.add_argument("--max-samples", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    a = ap.parse_args()

    # Single-process distributed group so the evaluator's dist.* calls work
    # without the multiprocessing spawn that deadlocked.
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29555")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)

    from deepspec.eval.dspark import GraniteDSparkEvaluator

    args = SimpleNamespace(
        target_name_or_path=a.target,
        draft_name_or_path=a.draft,
        max_new_tokens=a.max_new_tokens,
        temperature=1.0,
        confidence_threshold=0.0,
        tensorboard_dir=None,
        step=None,
        seed=980406,
        tasks=[(a.task, a.max_samples)],
    )
    evaluator = GraniteDSparkEvaluator(0, args)

    print(f"[probe] running {a.task} on {a.max_samples} samples...", flush=True)
    responses = evaluator.run_dataset(dataset_name=a.task, max_samples=a.max_samples)
    summary = evaluator.allreduce_response_metrics(responses)
    row = evaluator.build_metrics_row(dataset_name=a.task, metric_summary=summary)

    print("\n==== PROBE RESULT ====", flush=True)
    print(f"task                 : {row['dataset']}", flush=True)
    print(f"num_samples          : {row['num_samples']}", flush=True)
    print(f"draft_tokens/proposal: {row['draft_tokens_per_proposal']:.2f}", flush=True)
    print(f"acceptance_length    : {row['acceptance_length']:.3f}", flush=True)
    print(f"verify_rate          : {row['verify_rate']:.4f}", flush=True)
    print(f"accept_rates_by_pos  : {row['accept_rates_by_position']}", flush=True)
    nonzero = row["acceptance_length"] > 0.0
    print(f"\nNONZERO_ACCEPTANCE: {nonzero}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
