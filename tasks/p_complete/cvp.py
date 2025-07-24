import csv

import torch
import torch.nn.functional as F

from tasks.task import CurriculumDataset, GeneralizationTask

GATE_TYPES = ["AND", "OR", "NOT"]
CONST_TYPES = ["TRUE", "FALSE"]
PAD_TOKEN = "NA"


class CircuitValueProblemDataset(CurriculumDataset):
    def __init__(self, config, split="train"):
        super().__init__()
        self.config = config
        csv_path = config["csv_path"]
        if split == "test":
            csv_path = csv_path.replace("train", "test")

        BASE_TOKENS = ["PAD", "NA", "AND", "OR", "NOT", "TRUE", "FALSE"]  # "PAD" ifor causal models
        self.token2id_inp: dict[str, int] = {tok: i for i, tok in enumerate(BASE_TOKENS)}
        next_id = len(BASE_TOKENS)
        for n in range(1, self.config["input_size"] + 1):
            self.token2id_inp[str(n)] = next_id
            next_id += 1
        self.vocab_size = next_id

        self.label2id = {
            "NA": self.config["ignore_index"],
            "FALSE": BASE_TOKENS.index("FALSE"),
            "TRUE": BASE_TOKENS.index("TRUE"),
        }

        self.samples = []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                inp_tokens = row["input"].strip().split()
                tgt_tokens = row["output"].strip().split()

                inp_ids = [self.token2id_inp[tok] for tok in inp_tokens]
                tgt_ids = [self.label2id[tok] for tok in tgt_tokens]

                self.samples.append(
                    (
                        torch.tensor(inp_ids, dtype=torch.long),
                        torch.tensor(tgt_ids, dtype=torch.long),
                    )
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inp, tgt = self.samples[idx]
        length = self.curriculum.sample_sequence_length() if self.curriculum else len(inp)
        if length < len(inp):
            inp = inp[: length * 4]
            tgt = tgt[: length * 4]
        return torch.tensor(inp, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


class CircuitValueProblemTask(GeneralizationTask):
    def __init__(self, max_input_size: int = 64) -> None:
        config = {
            "name": "circuit_value_problem",
            "csv_path": "data/cvp/train.csv",
            "input_size": max_input_size,
            "vocab_size": max_input_size + 8,
            "max_length": max_input_size * 4,
            "min_input_size": 8,
            "ignore_index": -100,
        }
        self.config = config

    def pointwise_loss_fn(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = F.cross_entropy(
            output.view(-1, output.size(-1)), target.view(-1), ignore_index=self.config["ignore_index"]
        )
        return loss

    def accuracy_fn(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = output.argmax(dim=-1)  # (B, T)
        return (pred[:, -1] == target[:, -1]).float().mean()


if __name__ == "__main__":
    # Example usage
    task = CircuitValueProblemTask()
    dataset = CircuitValueProblemDataset(task.config, split="test")
    for i, (inp, tgt) in enumerate(dataset):
        print("Input:", inp.tolist())
        print("Target:", tgt.tolist())
        if i >= 5:
            break
