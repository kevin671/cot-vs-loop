import csv

import numpy as np
import torch
import torch.nn.functional as F

from tasks.task import CurriculumDataset, GeneralizationTask, GeneralizationTaskChain


class WordProblemDataset(CurriculumDataset):
    def __init__(self, config, split="train", chain: bool = False):
        super().__init__()
        self.config = config
        self.chain = chain
        csv_path = config["csv_path"]
        if split == "test":
            csv_path = csv_path.replace(".csv", "_test.csv")
        self.samples = []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                inp = list(map(int, row["input"].split()))
                tgt = list(map(int, row["target"].split()))

                if chain:
                    if split == "train":
                        eos_token_id = config["eos_token_id"]
                        cot_len = config["cot_length"]
                        cot, ans = tgt[:-1], tgt[-1]
                        total_len = len(cot)

                        if cot_len is None:
                            sampled_cot = cot
                        else:
                            indices = np.linspace(0, total_len - 1, cot_len, dtype=int).tolist()
                            sampled_cot = [cot[i] for i in indices]
                        input_ids = inp + sampled_cot + [ans]
                        label_ids = torch.tensor(input_ids[1:] + [eos_token_id], dtype=torch.long)
                        label_ids[: len(inp) - 1] = self.config["ignore_index"]
                        self.samples.append((input_ids, label_ids))
                    else:
                        label_id = tgt[-1]
                        self.samples.append((inp, label_id))
                else:
                    self.samples.append((inp, tgt))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.chain:
            inp, tgt = self.samples[idx]
            return torch.tensor(inp, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)
        else:
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
        "csv_path": "data/word_problem/S5_256.csv",  # "data/word_problem/S5_512.csv",  # Path to the CSV file with input-output pairs
        "vocab_size": 120,
        "max_length": 256,
        "ignore_index": -100,
    }

    def pointwise_loss_fn(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = F.cross_entropy(
            output.view(-1, output.size(-1)), target.view(-1), ignore_index=self.config["ignore_index"]
        )
        return loss

    def accuracy_fn(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = output.argmax(dim=-1)  # (B, T)
        return (pred == target).float().mean(dim=0)


class WordProblemTaskChain(GeneralizationTaskChain):
    def __init__(self, max_input_size: int = 64, cot_length: int = None) -> None:
        config = {
            "name": "word_problem",
            "csv_path": f"data/word_problem/S5_{max_input_size}.csv",
            "max_length": max_input_size * 2 + 1,
            "vocab_size": 121,
            "ignore_index": -100,
            "eos_token_id": 120,
        }
        config["cot_length"] = cot_length
        self.config = config


if __name__ == "__main__":
    # Example usage
    task = WordProblemTaskChain(max_input_size=256, cot_length=None)
    dataset = WordProblemDataset(task.config, split="train", chain=True)
    for inp, tgt in dataset:
        print("Input:", inp.tolist())
        print("Target:", tgt.tolist())
        break
