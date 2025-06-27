import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from tasks.task import GeneralizationTask


class WordProblemDataset(Dataset):
    def __init__(self, config, split="train"):
        csv_path = config["csv_path"]
        if split == "test":
            csv_path = csv_path.replace(".csv", "_test.csv")
        self.samples = []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                inp = list(map(int, row["input"].split()))
                tgt = list(map(int, row["target"].split()))
                self.samples.append((inp, tgt))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class WordProblemTask(GeneralizationTask):
    """
    A task where the goal is to compute the *prefix product* over a sequence of group elements.

    ------------------------------------------------------------
    Task Description
    ----------------
    Let G be a finite group with elements encoded as integer IDs.
    Given an input sequence of group elements
        (g₁, g₂, ..., gₖ) ∈ Gᵏ,
    the task is to compute the sequence of prefix products:
        (g₁,
         g₁ * g₂,
         g₁ * g₂ * g₃,
         ...,
         g₁ * g₂ * ... * gₖ),
    where * denotes group multiplication (typically left-to-right).

    Both input and output sequences are represented as lists of integer IDs
    corresponding to elements of G.

    ------------------------------------------------------------
    Example
    -------
    Suppose the underlying group is the symmetric group S₃, and the group
    elements are assigned the following IDs::

        ID : permutation
         0 : (1 2 3)   # identity
         1 : (2 1 3)
         2 : (1 3 2)
         3 : (3 2 1)
         4 : (2 3 1)
         5 : (3 1 2)

    Then for the input sequence:

        Input : 4 2 1

    The corresponding prefix product sequence is:

        Output: 4 5 0
    """

    config = {
        "name": "word_problem",
        "description": "Compute prefix products over a sequence of group elements.",
        "csv_path": "data/word_problem/S5_512.csv",  # Path to the CSV file with input-output pairs
        "vocab_size": 22,  # Number of unique group elements (IDs)
        "block_size": 512,  # Maximum sequence length
    }

    def pointwise_loss_fn(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the pointwise cross-entropy loss between logits and integer target IDs.

        Args:
            output: [B, L, V] - logits over vocab/group elements
            target: [B, L]    - integer target indices
        Returns:
            pointwise loss: [B, L] - cross-entropy at each position
        """
        loss = F.cross_entropy(output.view(-1, output.size(-1)), target.view(-1), ignore_index=-1)
        return loss

    def accuracy_fn(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Returns mean accuracy over all tokens.

        Returns:
            accuracy: Scalar tensor (mean accuracy).
        """
        pred = output.argmax(dim=-1)  # (B, T)
        return (pred == target).float().mean()

    @staticmethod
    def collate_fn(batch):
        return batch


# ============================
#  Data Generation for Word Problems
#  - Based on: https://github.com/jopetty/word-problem/blob/main/src/generate_data.py
# ============================

import argparse
import csv
import itertools as it
import os
import random
import sys
from functools import reduce
from itertools import product
from pathlib import Path

# https://github.com/alreich/abstract_algebra/blob/main/src/finite_algebras.py


class Perm:
    def __init__(self, permutation):
        self.perm = permutation
        self.base = min(self.perm)  # lowest value in perm
        self.size = len(self.perm) + self.base
        self.mapping = {i: self.perm[i - self.base] for i in range(self.base, self.size)}

    def __eq__(self, other):
        return self.perm == other.perm

    def __hash__(self):
        return hash(tuple(self.perm))

    def __repr__(self):
        return f"Perm({self.perm})"

    def __len__(self):
        return len(self.perm)

    def __mul__(self, other):
        if self.base == other.base:
            if len(self) == len(other):
                return Perm(tuple([self.mapping[other.mapping[i]] for i in range(self.base, self.size)]))
            else:
                raise Exception(f"Mixed lengths: {len(self)} != {len(other)}")
        else:
            raise Exception(f"Mixed bases: {self.base} != {other.base}")


def index_table_from_name_table(elements, name_table):
    """Converts a table (list of lists) of strings into a table (list of lists) of ints."""
    return [[elements.index(elem_name) for elem_name in row] for row in name_table]


class SimpleGroup:
    def __init__(self, name, elements, table, identity_idx):
        self.name = name
        self.elements = elements
        self.table = table  # list of list of indices
        self.identity = elements[identity_idx]

    def op(self, a, b):
        i = self.elements.index(a)
        j = self.elements.index(b)
        return self.elements[self.table[i][j]]


def make_finite_algebra(name, desc, elements, table):
    identity_idx = None
    for i, e in enumerate(elements):
        if all(
            elements[table[i][j]] == elements[j] and elements[table[j][i]] == elements[j] for j in range(len(elements))
        ):
            identity_idx = i
            break

    if identity_idx is None:
        raise ValueError("No identity found — not a valid group")

    return SimpleGroup(name, elements, table, identity_idx)


def generate_symmetric_group(n, name=None, description=None, base=1):
    """Generates a symmetric group on n elements. The 'base' is a non-negative integer
    (typically, 0 or 1), so permutations will be tuples, like (1, 2, 3, ..., n), where
    n is the order, or (0, 1, 2, ..., n-1).
    """
    if name:
        nm = name
    else:
        nm = "S" + str(n)
    if description:
        desc = description
    else:
        desc = f"Autogenerated symmetric Group on {n} elements"
    ident = tuple(range(base, n + base))
    perms = list(it.permutations(ident))
    elem_dict = {str(p): Perm(p) for p in perms}
    rev_elem_dict = {val: key for key, val in elem_dict.items()}
    mul_tbl = [[rev_elem_dict[elem_dict[a] * elem_dict[b]] for b in elem_dict] for a in elem_dict]
    index_table = index_table_from_name_table(list(elem_dict.keys()), mul_tbl)
    return make_finite_algebra(nm, desc, list(elem_dict.keys()), index_table)


def generate_cyclic_group(order, elem_name="", name=None, description=None, zfill=False):
    """Generates a cyclic group with the given order. If zfill is True, then left fill element
    names with zeros."""
    if name:
        nm = name
    else:
        nm = "Z" + str(order)
    if description:
        desc = description
    else:
        desc = f"Autogenerated cyclic Group of order {order}"
    if zfill:
        nfill = len(str(order - 1))  # Number of zeros to left-fill integers in element names
        elements = [elem_name + str(i).zfill(nfill) for i in range(order)]
    else:
        elements = [elem_name + str(i) for i in range(order)]
    table = [[((a + b) % order) for b in range(order)] for a in range(order)]
    return make_finite_algebra(nm, desc, elements, table)


def group_reduce(lhs, rhs, G):
    if isinstance(lhs, str):
        prod = G.op(lhs, G.elements[rhs])
    else:
        prod = G.op(G.elements[lhs], G.elements[rhs])
    return G.elements.index(prod)


def generate_group(g):
    if g[0] == "S":
        return generate_symmetric_group(g[1])
    elif g[0] == "Z":
        return generate_cyclic_group(g[1])
    elif g[0] == "A":
        raise NotImplementedError("Commutator subalgebras for A_n are not implemented yet.")
        s_n = generate_symmetric_group(g[1])
        a_n = s_n.commutator_subalgebra()
        a_n.name = f"A{g[1]}"
        return a_n
    else:
        raise ValueError("Group must be one of S, Z, or A")


if __name__ == "__main__":
    # python word.py --group=S5 --k=512 --data_dir=data/word_problem --samples=1000000 --overwrite
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", required=True, help="Group identifier, e.g., S3 or A5_x_Z2")
    parser.add_argument("--k", type=int, default=10, help="Sequence length")
    parser.add_argument("--samples", type=int, default=None, help="Number of samples")
    # parser.add_argument("--train_size", type=int, default=1e6)
    # parser.add_argument("--test_size", type=int, default=1e4)
    parser.add_argument("--data_dir", default="data", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing file")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_path = data_dir / f"{args.group}={args.k}.csv"
    if data_path.exists() and not args.overwrite:
        print(f"Data already exists at {data_path}. Use --overwrite to regenerate.")
        sys.exit(0)

    random.seed(args.seed)

    group_ids = [(g[0], int(g[1:])) for g in args.group.split("_x_")]
    group_list = [generate_group(g) for g in group_ids]
    group_prod = reduce(lambda x, y: x * y, group_list)

    num_elements = len(group_prod.elements)
    num_unique_sequences = num_elements**args.k

    if args.samples is None:
        print(f"Generating all {num_unique_sequences} sequences.")
        sequences = product(range(num_elements), repeat=args.k)
    else:
        if args.samples > num_unique_sequences:
            print(f"Warning: {args.samples} > {num_unique_sequences}. Clipping.")
            args.samples = num_unique_sequences
        sequences = set()
        while len(sequences) < args.samples:
            sequences.add(tuple(random.choices(range(num_elements), k=args.k)))
        sequences = list(sequences)

    examples = []
    for seq in sequences:
        acc = 0
        scanned = [acc := group_reduce(acc, x, group_prod) for x in seq]
        examples.append({"input": " ".join(map(str, seq)), "target": " ".join(map(str, scanned))})

    random.shuffle(examples)
    split_idx = int(0.9 * len(examples))
    train_examples = examples[:split_idx]
    test_examples = examples[split_idx:]

    train_path = data_dir / f"{args.group}_{args.k}.csv"
    test_path = data_dir / f"{args.group}_{args.k}_test.csv"

    # Write training data
    os.makedirs(data_dir, exist_ok=True)
    # with open(data_path, "w", newline="") as f:
    #    writer = csv.DictWriter(f, fieldnames=["input", "target"])
    #    writer.writeheader()
    #    writer.writerows(examples)
    # print(f"Data written to {data_path}")

    with open(train_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["input", "target"])
        writer.writeheader()
        writer.writerows(train_examples)
    print(f"Training data written to {train_path}")

    with open(test_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["input", "target"])
        writer.writeheader()
        writer.writerows(test_examples)
    print(f"Test data written to {test_path}")
