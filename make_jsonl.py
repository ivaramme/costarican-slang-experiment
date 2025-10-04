#!/usr/bin/env python3
"""
Convert 'phrase::explanation' text into JSONL with the requested conversation template.

Defaults:
  input:  data/source_augmented.txt
  output: data/source_augmented.jsonl

Output schema matches train.py: {"instruction": str, "output": str}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

TEMPLATE = """{{ "messages": [ {{"role": "system", "content": "Eres un experto en la cultura de Costa Rica y puedes explicar el significado de un costarriqueñismo que el usuario ingrese."}}, {{"role": "user", "content": "{instruction}"}}, {{"role": "assistant", "content": "El significado de este costarriqueñismo es: {output}"}}] }}"""

def normalize_explanation(text: str) -> str:
    """Ensure the explanation ends with proper punctuation."""
    t = text.strip()
    if not t.endswith((".", "!", "?", "…")):
        t += "."
    return t


def convert_line(line: str) -> str | None:
    """
    Convert a single line of 'phrase::explanation' into the JSONL template, or None if invalid.
    """
    line = line.strip()
    if not line or "::" not in line:
        return None
    parts = line.split("::")
    definition = parts[0].strip()
    explanation = "::".join(parts[1:]).strip()
    if not definition or not explanation:
        return None

    template = TEMPLATE.format(instruction=definition, output=normalize_explanation(explanation))
    return template


def run(input_path: Path, output_path: Path) -> int:
    """Process the input file and write to output file."""
    count_in, count_out = 0, 0
    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for raw in fin:
            count_in += 1
            instruction = convert_line(raw) or ""
            fout.write(instruction + "\n")
            count_out += 1
    print(f"Processed {count_in} lines, wrote {count_out} JSONL records to {output_path}")
    return 0


def main(argv: list[str]) -> int:
    """Parse arguments and run."""
    parser = argparse.ArgumentParser(description="Make JSONL from 'phrase::explanation' txt")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/source_augmented.txt"),
        help="Path to input txt (default: data/source_augmented.txt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/source_augmented.jsonl"),
        help="Path to output jsonl (default: data/source_augmented.jsonl)",
    )
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    return run(args.input, args.output)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
