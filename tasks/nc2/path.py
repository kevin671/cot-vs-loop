# file: tasks/reachability.py
import csv
from collections import defaultdict

# tasks/reachability.py
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from tasks.task import CurriculumDataset, GeneralizationTask


class ReachabilityDataset(CurriculumDataset):
    """
    One-line format (tab separated):

        vertices                     edges                   query  label
        v0 v1 v2 v3 v4 v5 v6 v7      0,1 0,3 0,5 1,4 … 4,5   1,4    1

    Returned input sequence:
        [query_token] + vertex_tokens + edge_tokens
    Returned label:
        0 if t is NOT reachable from s, otherwise 1.
    """

    def __init__(self, config, split="train") -> None:
        super().__init__()
        self.config = config

        def build_fixed_vocab(max_node: int) -> Dict[str, int]:
            tok2id: Dict[str, int] = {"<pad>": 0, "TRUE": 1, "FALSE": 2}
            # vertices
            for i in range(max_node):
                tok2id[f"v{i}"] = len(tok2id)
            # ordered pairs
            for i in range(max_node):
                for j in range(max_node):
                    if i != j:
                        tok2id[f"{i},{j}"] = len(tok2id)
            return tok2id

        max_node = config["max_node"]
        self.token2id = build_fixed_vocab(max_node)
        self.samples: List[Tuple[List[int], int]] = []

        file_path = Path(config["data_dir"]) / f"{split}.txt"
        self.samples: List[Tuple[List[int], int]] = []

        with file_path.open() as f:
            for line in f:
                verts, edges, query, label = line.rstrip().split("\t")
                tokens = [query] + verts.split() + edges.split()
                ids = [self.token2id[tok] for tok in tokens]
                self.samples.append((ids, int(label) + 1))  # label: 1 for TRUE, 2 for FALSE

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        ids, label = self.samples[idx]
        return (
            torch.tensor(ids, dtype=torch.long),  # 1-D token id tensor
            torch.tensor(label, dtype=torch.long),  # scalar label tensor
        )


class ReachabilityTask(GeneralizationTask):
    """
    Binary classification: 1 iff t is reachable from s in the serialized graph.
    """

    max_n = 8
    config = {
        "name": "reachability",
        "description": "Predict s→t reachability from a tokenised graph.",
        "data_dir": "data/path",
        "max_node": max_n,
        "vocab_size": max_n + max_n * (max_n - 1) + 1 + 2,  # nodes + edges + <pad> + labels
        # "max_length": max_n + max_n * (max_n - 1) + 2,  # nodes + edges + <pad> + labels
    }

    def pointwise_loss_fn(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        last_logits = output[:, 0, :]  # (batch_size, vocab_size)
        return F.cross_entropy(last_logits, target)

    def accuracy_fn(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = output[:, 0, :].argmax(dim=-1)  # (batch_size,)
        return (pred == target).float().mean()


if __name__ == "__main__":
    task = ReachabilityTask()
    dataset = ReachabilityDataset(task.config, split="test")
    for i in range(5):
        inp, tgt = dataset[i]
        print(f"Input: {inp.tolist()}, Target: {tgt.item()}")
        # Input: [21, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 21, 31, 37, 43], Target: 2
