import argparse
import csv
import random
from pathlib import Path
from typing import List, Tuple

GATE_TYPES = ["AND", "OR", "NOT"]
CONST_TYPES = ["TRUE", "FALSE"]
PAD_TOKEN = "NA"  # placeholder in the *output* side as well


def gate_value(gtype: str, a: int, b: int) -> int:
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
    raise ValueError(f"unknown gate type {gtype}")


def generate_circuit(
    m: int,
    p_const: float,
    p_not: float,
    rng: random.Random,
) -> Tuple[List[str], List[int]]:
    """
    Return the input-side tokens and a list of gate values.
    """
    values: List[int] = [None] * (m + 1)  # 1-indexed
    tokens: List[str] = []

    for gid in range(1, m + 1):
        # choose gate type
        if gid <= 2 or rng.random() < p_const:
            gtype = rng.choice(CONST_TYPES)
        else:
            gtype = "NOT" if rng.random() < p_not else rng.choice(["AND", "OR"])

        # choose inputs
        if gtype in CONST_TYPES:
            in1, in2 = PAD_TOKEN, PAD_TOKEN
            val = gate_value(gtype, 0, 0)
        elif gtype == "NOT":
            src = rng.randrange(1, gid)
            in1, in2 = src, PAD_TOKEN
            val = gate_value(gtype, values[src], 0)
        else:  # AND / OR
            a = rng.randrange(1, gid)
            b = rng.randrange(1, gid)
            in1, in2 = a, b
            val = gate_value(gtype, values[a], values[b])

        values[gid] = val
        tokens.extend([gtype, str(in1), str(in2), str(gid)])

    return tokens, values[1:]  # drop dummy index 0


def make_output_tokens(values: List[int]) -> List[str]:
    """
    Convert gate values [v1, …, vm] to
    [NA NA NA token(v1), …, NA NA NA token(vm), '=']
    """
    out: List[str] = []
    for v in values:
        token = "TRUE" if v else "FALSE"
        out.extend([PAD_TOKEN, PAD_TOKEN, PAD_TOKEN, token])
    return out


def write_split(
    n_samples: int,
    m: int,
    out_file: Path,
    rng: random.Random,
    p_const: float = 0.2,
    p_not: float = 0.2,
) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["input", "output"])
        for _ in range(n_samples):
            inp_tokens, gate_vals = generate_circuit(m, p_const, p_not, rng)
            out_tokens = make_output_tokens(gate_vals)
            w.writerow([" ".join(inp_tokens), " ".join(out_tokens)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_nodes", type=int, default=32, help="Fixed number of gates per circuit")
    parser.add_argument("--train_size", type=int, default=int(1e6))
    parser.add_argument("--test_size", type=int, default=int(1e4))
    parser.add_argument("--data_dir", type=str, default="data/cvp")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    data_dir = Path(args.data_dir)
    num_nodes = args.num_nodes

    write_split(args.train_size, num_nodes, data_dir / "train.csv", rng)
    write_split(args.test_size, num_nodes, data_dir / "test.csv", rng)

    print(f"Done. Seq-to-seq datasets with {num_nodes} nodes saved to " f"{data_dir}/train.csv and {data_dir}/test.csv")


if __name__ == "__main__":
    main()
