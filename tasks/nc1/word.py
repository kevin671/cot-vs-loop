import csv

import torch
import torch.nn.functional as F

from tasks.task import GeneralizationTask, CurriculumDataset


class WordProblemDataset(CurriculumDataset):
    def __init__(self, config, split="train"):
        super().__init__()
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
        # TODO: =を追加する
        inp, tgt = self.samples[idx]
        length = self.curriculum.sample_sequence_length() if self.curriculum else len(inp)
        if length < len(inp):
            inp = inp[:length]
            tgt = tgt[:length]
        # self.curriculum.step() if self.curriculum else None
        return torch.tensor(inp, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


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

    Input : 4 2 1
    Output: 4 5 0
    """

    config = {
        "name": "word_problem",
        "description": "Compute prefix products over a sequence of group elements.",
        "csv_path": "data/word_problem/S5_512.csv",  # "data/word_problem/S5_512.csv",  # Path to the CSV file with input-output pairs
        "vocab_size": 120,  # 120,  # Number of unique group elements (IDs)
        "max_length": 512,  # 512,  # Maximum sequence length
    }


if __name__ == "__main__":
    # Example usage
    task = WordProblemTask()
    dataset = WordProblemDataset(task.config, split="test")
    for inp, tgt in dataset:
        print("Input:", inp.tolist())
        print("Target:", tgt.tolist())
        break  # Just show the first sample
