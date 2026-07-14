"""Stream a small subset of open-perfectblend and write DeepSpec conversations
JSONL, without downloading the full multi-GB dataset.

The repo's scripts/data/download_and_split.py does a non-streaming full download
(then .select()s), which is wasteful when we only want ~2k rows for a PoC.
This streams row-by-row and stops after --max-rows, writing:
  - <train-out>: {"id", "conversations":[{"role","content"},...]} per line
  - <eval-out> : {"turns":[user strings...]} per line (held-out tail)
matching the formats consumed by prepare_target_cache.py and the eval harness.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROLE_MAPPING = {
    "human": "user",
    "gpt": "assistant",
    "chatgpt": "assistant",
    "bing": "assistant",
    "bard": "assistant",
}


def normalize(row, idx):
    conv = []
    for m in row["conversations"]:
        if m["from"] not in ROLE_MAPPING:
            continue
        conv.append({"role": ROLE_MAPPING[m["from"]], "content": m["value"]})
    return {"id": idx, "conversations": conv}


def valid(conv):
    c = conv["conversations"]
    if not c or c[0]["role"] != "user":
        return False
    for m in c:
        if m["role"] not in {"user", "assistant"}:
            return False
        if not isinstance(m["content"], str) or not m["content"]:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-name", default="mlabonne/open-perfectblend")
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-rows", type=int, required=True,
                    help="total rows to pull via streaming (train + eval)")
    ap.add_argument("--eval-frac", type=float, default=0.05)
    ap.add_argument("--train-out", type=Path, required=True)
    ap.add_argument("--eval-out", type=Path, required=True)
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset(args.dataset_name, split=args.split, streaming=True)
    rows = []
    for i, row in enumerate(ds):
        if len(rows) >= args.max_rows:
            break
        conv = normalize(row, i)
        if valid(conv):
            rows.append(conv)

    n_eval = max(1, int(len(rows) * args.eval_frac))
    train_rows = rows[:-n_eval]
    eval_rows = rows[-n_eval:]

    args.train_out.parent.mkdir(parents=True, exist_ok=True)
    args.eval_out.parent.mkdir(parents=True, exist_ok=True)
    with args.train_out.open("w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with args.eval_out.open("w", encoding="utf-8") as f:
        for r in eval_rows:
            turns = [m["content"] for m in r["conversations"] if m["role"] == "user"]
            f.write(json.dumps({"turns": turns}, ensure_ascii=False) + "\n")

    print(f"streamed {len(rows)} rows -> train={len(train_rows)} eval={len(eval_rows)}")
    print(f"train: {args.train_out}")
    print(f"eval : {args.eval_out}")


if __name__ == "__main__":
    main()
