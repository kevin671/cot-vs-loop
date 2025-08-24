import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from tasks.task import CurriculumDataset, GeneralizationTask, GeneralizationTaskChain


class ArithmeticExpressionDataset(CurriculumDataset):
    def __init__(self, config, split="train", chain: bool = False):
        super().__init__()
        self.chain = chain
        signs = ["+", "-", "*", "/", "(", ")", "="]
        num_range = config["num_range"]
        dictionary = {"<pad>": 0, "<cls>": 1, "<eos>": 2}
        data_dir = config["data_dir"]

        self.max_input_size = config["max_input_size"]
        for i in range(len(signs)):
            dictionary[signs[i]] = i + 3
        for i in range(num_range):
            dictionary[str(i)] = i + 10

        raw = {}
        d = config["min_input_size"]
        while d <= self.max_input_size:
            path = f"{data_dir}/{d}/chain/{split}_data.txt" if chain else f"{data_dir}/{d}/decoder/{split}_data.txt"
            if os.path.exists(path):
                with open(path) as f:
                    lines = [line.split() for line in f.read().splitlines()]
                raw[d] = lines
            # d *= 2
            d += 4

        max_examples = 1000000

        self.X, self.Y = {}, {}
        for length, lines in raw.items():
            xs, ys = [], []
            # for tokens in lines:
            for tokens in lines[:max_examples]:
                if chain:
                    eq_pos = tokens.index("=")
                    inp = tokens[: eq_pos + 1]
                    if split == "train":
                        cot, ans = tokens[eq_pos + 1 : -1], tokens[-1]
                        cot_len = config["cot_length"]
                        total_len = len(cot)
                        if cot_len is None:
                            sampled_cot = cot
                        else:
                            indices = np.linspace(0, total_len - 1, cot_len, dtype=int).tolist()
                            sampled_cot = [cot[i] for i in indices]
                        input_ids = inp + sampled_cot + [ans]
                        input_ids = torch.tensor([dictionary[t] for t in input_ids], dtype=torch.long)
                        label_ids = torch.tensor(input_ids[1:].tolist() + [dictionary["<eos>"]], dtype=torch.long)
                        label_ids[:eq_pos] = config["ignore_index"]
                        xs.append(input_ids)
                        ys.append(label_ids)

                    else:
                        label_id = tokens[-1]
                        input_ids = inp
                        xs.append(torch.tensor([dictionary[t] for t in input_ids], dtype=torch.long))
                        ys.append(torch.tensor(dictionary[label_id], dtype=torch.long))
                else:
                    eq_pos = tokens.index("=")
                    seq = tokens[:eq_pos]
                    seq = ["<cls>"] + seq
                    xs.append(torch.tensor([dictionary[t] for t in seq], dtype=torch.long))
                    ans_tok = tokens[eq_pos + 1]
                    ys.append(torch.tensor(dictionary[ans_tok], dtype=torch.long))
            self.X[length] = xs
            self.Y[length] = ys

    def __len__(self):
        return len(self.X[self.max_input_size])  # assume all lengths have the same number of samples

    def __getitem__(self, idx):
        if self.chain:
            length = self.max_input_size
        else:
            length = self.curriculum.sample_sequence_length() if self.curriculum else self.max_input_size
        inp = self.X[length][idx]
        tgt = self.Y[length][idx]
        return inp, tgt


num_range = 11


class ArithmeticExpressionTask(GeneralizationTask):
    def __init__(self, max_input_size=64):
        self.config = {
            "name": "arithmetic_expression",
            "data_dir": "data/arithmetic",
            "vocab_size": num_range + 10,
            "min_input_size": 8,  # 4,
            "num_range": num_range,
            "increase_amount": 8,
        }
        self.config["max_input_size"] = max_input_size
        self.config["max_length"] = max_input_size * 4 + 1

    def pointwise_loss_fn(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        cls_logits = output[:, 0, :]  # (batch_size, vocab_size)
        return F.cross_entropy(cls_logits, target)

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


class ArithmeticExpressionTaskChain(GeneralizationTaskChain):
    def __init__(self, max_input_size: int = 64, cot_length: int = None) -> None:
        config = {
            "name": "arithmetic_expression",
            "data_dir": "data/arithmetic",
            "vocab_size": num_range + 10,
            "min_input_size": max_input_size,
            "max_input_size": max_input_size,
            "num_range": num_range,
            "ignore_index": -100,
            "eos_token_id": 2,
        }
        config["max_length"] = max_input_size * (max_input_size + 1) // 2 * 4 + 10
        config["cot_length"] = cot_length
        self.config = config


if __name__ == "__main__":
    task = ArithmeticExpressionTaskChain(max_input_size=4)  # , cot_length=4)
    dataset = ArithmeticExpressionDataset(task.config, split="train", chain=True)
    for i in range(len(dataset)):
        inp, tgt = dataset[i]
        print(f"Input: {inp.tolist()}, Target: {tgt.tolist()}")
        break
