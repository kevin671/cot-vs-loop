import os
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

from tasks.task import CurriculumDataset, GeneralizationTask


class BayesNetDataset(CurriculumDataset):
    def __init__(self, config: Dict[str, Any], split: str = "train"):
        super().__init__()
        self.config = config
        self.split = split
        self.max_len = config["max_length"]
        self._build_vocab()

        path = os.path.join(config["data_dir"], f"{split}.txt")
        with open(path, encoding="utf-8") as f:
            self.lines: List[str] = [ln.strip() for ln in f if ln.strip()]

    def _build_vocab(self) -> None:
        v = self.config["node_size"]
        self.tok2id: Dict[str, int] = {}

        def add(tok: str):
            self.tok2id[tok] = len(self.tok2id)

        add("<pad>")  # 0
        add("|")  # 1
        add("0")  # 2
        add("1")  # 3
        for i in range(v):  # "0=" .. "14="
            add(f"{i}=")

        self.pad_id = self.tok2id["<pad>"]
        self.id0 = self.tok2id["0"]
        self.id1 = self.tok2id["1"]

    def __len__(self):
        return len(self.lines)

    def _encode(self, toks: List[str]) -> List[int]:
        return [self.tok2id[t] for t in toks]

    def __getitem__(self, idx: int):
        line = self.lines[idx]
        toks = line.split()

        input_ids = self._encode(toks[:-1])
        label_raw = toks[-1]

        if self.split == "train":
            labels = torch.full((self.max_len,), self.config["ignore_index"], dtype=torch.long)
            ans = self.tok2id[label_raw]
            labels[len(input_ids) - 1] = ans
            # padding
            pad_len = self.max_len - len(input_ids)
            if pad_len < 0:
                raise RuntimeError(f"Input length {len(input_ids)} > max_len {self.max_len}")
            input_ids += [self.pad_id] * pad_len
        else:
            labels = torch.tensor(float(label_raw), dtype=torch.float)
            # for test, we assume input_ids is all the same length

        return torch.tensor(input_ids, dtype=torch.long), labels


class BayesNetTask(GeneralizationTask):
    # Warning: Use causal mask for this task
    config = {
        "name": "bayes_net",
        "description": "Ancestor sampling in Bayesian networks.",
        "data_dir": "data/bayes_net",
        "node_size": 15,
        "max_length": 15 * 2 + 2,
        "ignore_index": -100,
    }
    config["vocab_size"] = config["node_size"] + 4

    def pointwise_loss_fn(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            output.view(-1, output.size(-1)),
            target.view(-1),
            ignore_index=self.config["ignore_index"],
        )

    def _select_logits01(self, last_logits: torch.Tensor) -> torch.Tensor:
        ds = getattr(self, "_cached_ds", None)
        if ds is None:
            self._cached_ds = BayesNetDataset(self.config, split="test")
            ds = self._cached_ds
        logits01 = torch.stack((last_logits[:, ds.id0], last_logits[:, ds.id1]), dim=-1)  # (B,2)
        return logits01

    def accuracy_fn(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # we assume the test data has the same length, so we can use the last time step
        last_logits = output[:, -1, :]
        logits01 = self._select_logits01(last_logits)
        prob1 = F.softmax(logits01, dim=-1)[:, 1]
        rmse = torch.sqrt(F.mse_loss(prob1, target.float()))
        return rmse


if __name__ == "__main__":
    # Example usage
    task = BayesNetTask()
    dataset = BayesNetDataset(task.config, split="test")
    print(f"Number of samples in {dataset.split} set: {len(dataset)}")
    for i in range(5):
        input_ids, label = dataset[i]
        print(f"Input IDs: {input_ids}, Label: {label}")
