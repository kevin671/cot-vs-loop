#!/usr/bin/env python3
# cvp_dataset.py
"""
Generate a Circuit Value Problem (CVP) dataset compatible with the
format in 'Chain of Thought Empowers ...' (Figure 4).

Each gate is encoded by four tokens:
    [gate_type, input1_id, input2_id, gate_id]
and the full sequence is terminated with '='.

The final CSV has two columns:
    sequence,label
where `sequence` is a single space-separated string of exactly
`input_length` tokens, and `label` is 0/1 (value of gate m).
"""

import argparse
import csv
import os
import random
from pathlib import Path
from typing import List, Tuple

GATE_TYPES = ["AND", "OR", "NOT"]
CONST_TYPES = ["TRUE", "FALSE"]
PAD_TOKEN = "NA"  # used for padding


def _gate_value(gtype: str, a: int, b: int) -> int:
    if gtype == "TRUE":
        return 1
    if gtype == "FALSE":
        return 0
    if gtype == "NOT":
        return 1 - a
    if gtype == "AND":
        return a & b
    if gtype == "OR":
        return a | b
    raise ValueError(f"Unknown gate type {gtype}")


def generate_circuit(
    m: int,
    p_const: float = 0.2,
    p_not: float = 0.2,
    rng: random.Random | None = None,
) -> Tuple[List[str], int]:
    """
    Generate a random CVP instance with m gates in topological order.

    Returns
    -------
    tokens : list[str]
        Token sequence (including trailing '=').
    label : int
        Boolean value (0/1) of the final gate m.
    """
    if rng is None:
        rng = random  # global RNG

    gate_vals: List[int] = [None] * (m + 1)  # 1-indexed
    tokens: List[str] = []

    for gid in range(1, m + 1):
        # choose gate type
        if gid <= 2 or rng.random() < p_const:
            gtype = rng.choice(CONST_TYPES)
        else:
            gtype = "NOT" if rng.random() < p_not else rng.choice(["AND", "OR"])

        # choose inputs and compute value
        if gtype in CONST_TYPES:
            in1_id = in2_id = PAD_TOKEN
            val = _gate_value(gtype, 0, 0)
        elif gtype == "NOT":
            src = rng.randrange(1, gid)
            in1_id, in2_id = src, PAD_TOKEN
            val = _gate_value(gtype, gate_vals[src], 0)
        else:  # AND / OR
            a = rng.randrange(1, gid)
            b = rng.randrange(1, gid)
            in1_id, in2_id = a, b
            val = _gate_value(gtype, gate_vals[a], gate_vals[b])

        gate_vals[gid] = val
        tokens.extend([gtype, str(in1_id), str(in2_id), str(gid)])

    tokens.append("=")
    return tokens, gate_vals[m]


def pad_or_truncate(tokens: List[str], length: int) -> List[str]:
    """Ensure the sequence has exactly `length` tokens."""
    if len(tokens) < length:
        return tokens + [PAD_TOKEN] * (length - len(tokens))
    return tokens[:length]


def write_split(
    n_samples: int,
    m_min: int,
    m_max: int,
    out_file: Path,
    input_length: int,
    rng: random.Random,
) -> None:
    """Write one CSV split with header 'sequence,label'."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence", "label"])
        for _ in range(n_samples):
            m = rng.randint(m_min, m_max)
            seq, label = generate_circuit(m, rng=rng)
            seq = pad_or_truncate(seq, input_length)
            writer.writerow([" ".join(seq), label])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_length", type=int, default=64, help="Fixed token length after padding/truncation")
    parser.add_argument("--train_size", type=int, default=int(1e6), help="Number of training samples")
    parser.add_argument("--test_size", type=int, default=int(1e4), help="Number of test samples")
    parser.add_argument("--data_dir", type=str, default="data/cvp", help="Output directory for CSV files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    data_dir = Path(args.data_dir)
    write_split(
        n_samples=args.train_size,
        m_min=4,
        m_max=32,
        out_file=data_dir / "train.csv",
        input_length=args.input_length,
        rng=rng,
    )
    write_split(
        n_samples=args.test_size,
        m_min=4,
        m_max=32,
        out_file=data_dir / "test.csv",
        input_length=args.input_length,
        rng=rng,
    )
    print(f"Datasets saved to {data_dir}/train.csv and {data_dir}/test.csv")


if __name__ == "__main__":
    main()
