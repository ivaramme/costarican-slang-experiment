#!/usr/bin/env python3
"""
Split a JSONL file into train/valid/test with deterministic shuffling.

Defaults:
  input: data/source_augmented.jsonl
  outputs: data/train.jsonl, data/valid.jsonl, data/test.jsonl
Also writes data/eval.jsonl mirroring 'valid' by default to match the 
commands described in README.md.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Dict, Any


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    Read a JSONL file into a list of dicts, skipping malformed lines with a warning.
    """
    items = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: skipping malformed JSON on line {i}: {e}", file=sys.stderr)
    return items


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write a list of dicts to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")


def split(items: List[Dict[str, Any]], train_ratio: float, valid_ratio: float, seed: int) -> tuple[list, list, list]:
    """
    Split items into train/valid/test lists based on ratios with a fixed random seed.
    """
    n = len(items)
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)

    train_n = int(n * train_ratio)
    valid_n = int(n * valid_ratio)
    train_idx = idx[:train_n]
    valid_idx = idx[train_n: train_n + valid_n]
    test_idx = idx[train_n + valid_n:]

    to_rows = lambda id_list: [items[i] for i in id_list]
    return to_rows(train_idx), to_rows(valid_idx), to_rows(test_idx)


def main(argv: list[str]) -> int:
    """Parse arguments and run."""
    p = argparse.ArgumentParser(description="Split JSONL into train/valid/test")
    p.add_argument("--input", type=Path, default=Path("data/source_augmented.jsonl"))
    p.add_argument("--train", type=Path, default=Path("data/train.jsonl"))
    p.add_argument("--valid", type=Path, default=Path("data/valid.jsonl"))
    p.add_argument("--test", type=Path, default=Path("data/test.jsonl"))
    p.add_argument("--eval", type=Path, default=Path("data/eval.jsonl"), help="Extra copy of valid for training script")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--valid-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    items = read_jsonl(args.input)
    if not items:
        print("No records found in input.", file=sys.stderr)
        return 1

    train_rows, valid_rows, test_rows = split(items, args.train_ratio, args.valid_ratio, args.seed)

    write_jsonl(args.train, train_rows)
    write_jsonl(args.valid, valid_rows)
    write_jsonl(args.test, test_rows)
    # Also write eval.jsonl with a subset of training data for verification during training
    if args.eval:
        # Use a small subset of training data for eval (e.g., first 10% of train)
        eval_size = max(1, len(train_rows) // 10)
        write_jsonl(args.eval, train_rows[:eval_size])

    print(
        f"Split {len(items)} -> train={len(train_rows)}, valid={len(valid_rows)}, test={len(test_rows)}.\n"
        f"Wrote: {args.train}, {args.valid}, {args.test}, and {args.eval}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
