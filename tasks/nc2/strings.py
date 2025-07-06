from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from tasks.task import CurriculumDataset, GeneralizationTask


class PairwiseAlignmentDataset(CurriculumDataset):
    def __init__(self, config, split="train") -> None:
        super().__init__()
        self.config = config
        self.max_input_size = config["max_input_size"]

        def build_fixed_vocab(config):
            tok2id: Dict[str, int] = {"<pad>": 0, "<cls>": 1, "<sep>": 2, "<eos>": 3, "|": 4, ",": 5}
            alphabet = "abcdefghijklmnopqrstuvwxyz"
            for i in range(26):
                tok2id[alphabet[i]] = i + 6
            for i in range(3 * config["max_input_size"] + 1):
                tok2id[str(i)] = i + 32
            return tok2id

        dictionary = build_fixed_vocab(config)

        raw = {}
        d = config["min_input_size"]
        data_dir = config["data_dir"]
        while d <= self.max_input_size:
            path = f"{data_dir}/{d}/decoder/{split}_data.txt"
            with open(path) as f:
                lines = [line.split() for line in f.read().splitlines()]
            raw[d] = lines
            d *= 2

        self.X, self.Y = {}, {}
        for length, lines in raw.items():
            xs, ys = [], []
            for tokens in lines:
                eq_pos = tokens.index("<sep>")
                seq = tokens[:eq_pos]
                seq = ["<cls>"] + seq
                xs.append(torch.tensor([dictionary[t] for t in seq], dtype=torch.long))
                ans_tok = tokens[eq_pos + 1]
                ys.append(torch.tensor(dictionary[ans_tok], dtype=torch.long))
            self.X[length] = xs
            self.Y[length] = ys

    def __len__(self) -> int:
        return len(self.X[self.max_input_size])

    def __getitem__(self, idx: int):
        length = self.curriculum.sample_sequence_length() if self.curriculum else self.max_input_size
        inp = self.X[length][idx]
        tgt = self.Y[length][idx]
        return inp, tgt


class PairwiseAlignmentTask(GeneralizationTask):
    def pointwise_loss_fn(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        last_logits = output[:, 0, :]  # (batch_size, vocab_size)
        return F.cross_entropy(last_logits, target)

    def accuracy_fn(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = output[:, 0, :].argmax(dim=-1)  # (batch_size,)
        return (pred == target).float().mean()

    @staticmethod
    def collate_fn(batch):
        PAD_ID = 0
        seqs, labels = zip(*batch)
        padded = pad_sequence(seqs, batch_first=True, padding_value=PAD_ID)  # (B, L_max)
        labels = torch.stack(labels)
        return padded, labels


class EditDistanceTask(PairwiseAlignmentTask):
    max_input_size = 64
    config = {
        "name": "edit_distance",
        "data_dir": "data/ed",
        "max_input_size": max_input_size,
        "min_input_size": 16,
    }
    config["max_value"] = 3 * (config["max_input_size"] + 2)
    config["vocab_size"] = config["max_value"] + 11 + 5
    config["max_length"] = config["max_input_size"] * 2 + 7


class LongestCommonSubsequenceTask(PairwiseAlignmentTask):
    max_input_size = 64
    config = {
        "name": "longest_common_subsequence",
        "data_dir": "data/lcs",
        "max_input_size": max_input_size,
        "min_input_size": 16,
    }
    config["max_value"] = config["max_input_size"] + 2
    config["vocab_size"] = config["max_value"] + 11 + 5
    config["max_length"] = config["max_input_size"] * 2 + 7


if __name__ == "__main__":
    task = EditDistanceTask()  # LongestCommonSubsequenceTask()
    dataset = PairwiseAlignmentDataset(task.config, split="test")
    for i in range(5):
        inp, tgt = dataset[i]
        print(f"Input: {inp}, Target: {tgt}")
