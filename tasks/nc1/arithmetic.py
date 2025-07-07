import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from tasks.task import CurriculumDataset, GeneralizationTask


class ArithmeticExpressionDataset(CurriculumDataset):
    def __init__(self, config, split="train"):
        super().__init__()
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
            path = f"{data_dir}/{d}/decoder/{split}_data.txt"
            with open(path) as f:
                lines = [line.split() for line in f.read().splitlines()]
            raw[d] = lines
            d *= 2

        self.X, self.Y = {}, {}
        for length, lines in raw.items():
            xs, ys = [], []
            for tokens in lines:
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
        length = self.curriculum.sample_sequence_length() if self.curriculum else self.max_input_size
        inp = self.X[length][idx]
        tgt = self.Y[length][idx]
        return inp, tgt


class ArithmeticExpressionTask(GeneralizationTask):
    config = {
        "name": "arithmetic_expression",
        "description": "Evaluate an arithmetic expression, which can be viewed as a binary syntax tree.",
        "data_dir": "data/arithmetic",
        "vocab_size": 21,
        "max_input_size": 64,
        "max_length": 64 * 4 + 1,  # 3,
        "num_range": 11,
        "min_input_size": 4,
    }

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


if __name__ == "__main__":
    dataset = ArithmeticExpressionDataset(ArithmeticExpressionTask.config, split="train")
    for i in range(len(dataset)):
        inp, tgt = dataset[i]
        print(f"Input: {inp.tolist()}, Target: {tgt.item()}")
        if i >= 10:
            break
