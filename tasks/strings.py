from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from tasks.task import CurriculumDataset, GeneralizationTask


class PairwiseAlignmentDataset(CurriculumDataset):
    def __init__(self, config, split="train", chain: bool = False) -> None:
        super().__init__()
        self.config = config
        self.max_input_size = config["max_input_size"]

        def build_fixed_vocab(config):
            tok2id: Dict[str, int] = {"<pad>": 0, "<cls>": 1, "<sep>": 2, "<eos>": 3, "|": 4, ",": 5}
            alphabet = "abcdefghijklmnopqrstuvwxyz"
            for i in range(26):
                tok2id[alphabet[i]] = i + 6
            for i in range(config["max_value"]):
                tok2id[str(i)] = i + 32
            return tok2id

        dictionary = build_fixed_vocab(config)

        max_examples = 1000000  # 1005000

        if chain:
            path = Path(config["data_dir"]) / f"{config['max_input_size']}/chain/{split}_data.txt"
            with open(path) as f:
                lines = [line.split() for line in f.read().splitlines()]
            raw = {config["max_input_size"]: lines}
        else:
            raw = {}
            d = config["min_input_size"]
            data_dir = config["data_dir"]
            while d <= self.max_input_size:
                path = f"{data_dir}/{d}/decoder/{split}_data.txt"
                with open(path) as f:
                    lines = [line.split() for line in f.read().splitlines()]
                if len(lines) > max_examples:
                    lines = lines[:max_examples]

                raw[d] = lines
                d *= 2

        self.X, self.Y = {}, {}
        if chain:
            cot_len = config["cot_length"] or config["max_length"] + 1
            for length, lines in raw.items():
                xs, ys = [], []
                for tokens in lines:
                    sep_positions = [i for i, t in enumerate(tokens) if t == "<sep>"]
                    assert len(sep_positions) == 2
                    sep1, sep2 = sep_positions

                    if split == "train":
                        inp = tokens[: sep1 + 1]  # +1 to include <sep>
                        cot = tokens[sep1 + 1 : sep2]
                        ans = tokens[sep2:]  # includes second <sep> and targets

                        # systematic sampling
                        total_len = len(cot)
                        if total_len <= cot_len:
                            sampled_cot = cot
                        else:
                            indices = [int(i * total_len / cot_len) for i in range(cot_len)]
                            sampled_cot = [cot[i] for i in indices]
                            assert len(sampled_cot) == cot_len

                        new_tokens = inp + sampled_cot + ans

                        # eq_pos = tokens.index("<sep>")
                        token_ids = [dictionary[t] for t in new_tokens]

                        input_ids = torch.tensor(token_ids, dtype=torch.long)
                        label_ids = torch.tensor(token_ids, dtype=torch.long)
                        # shift
                        label_ids = torch.cat([label_ids[1:], torch.tensor([dictionary["<eos>"]], dtype=torch.long)])
                        # ignore the labels before the <sep>
                        label_ids[:sep1] = config["ignore_index"]
                        xs.append(input_ids)
                        ys.append(label_ids)
                    else:
                        inp = tokens[: sep1 + 1]  # <sep> まで含む
                        ans = tokens[sep2 + 1]  # 次のトークン（解答）
                        token_ids = [dictionary[t] for t in inp]
                        input_ids = torch.tensor(token_ids, dtype=torch.long)
                        label_id = torch.tensor(dictionary[ans], dtype=torch.long)
                        xs.append(input_ids)
                        ys.append(label_id)

                self.X[length] = xs
                self.Y[length] = ys

        else:
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


class EditDistanceTask(PairwiseAlignmentTask):
    def __init__(self, max_input_size: int = 64) -> None:
        config = {
            "name": "edit_distance",
            "data_dir": "data/ed",
            "max_input_size": max_input_size,
            "min_input_size": max_input_size,  # 16,
        }
        config["max_value"] = 3 * (config["max_input_size"] + 2)
        config["vocab_size"] = config["max_value"] + 32
        config["max_length"] = config["max_input_size"] * 2 + 7
        self.config = config


class LongestCommonSubsequenceTask(PairwiseAlignmentTask):
    def __init__(self, max_input_size: int = 64) -> None:
        config = {
            "name": "longest_common_subsequence",
            "data_dir": "data/lcs",
            "max_input_size": max_input_size,
            "min_input_size": max_input_size,  # 16,
        }
        config["max_value"] = config["max_input_size"] + 5
        config["vocab_size"] = config["max_value"] + 32
        config["max_length"] = config["max_input_size"] * 2 + 7
        self.config = config


class EditDistanceTaskChain(GeneralizationTask):
    def __init__(self, max_input_size: int = 64, cot_length: int = None) -> None:
        config = {
            "name": "edit_distance",
            "data_dir": "data/ed",
            "max_input_size": max_input_size,
            "min_input_size": max_input_size,  # 16,
            "ignore_index": -100,  # for causal language modeling
        }
        config["max_value"] = 3 * (config["max_input_size"] + 3)  # + 10 for the DP table values from chain
        config["vocab_size"] = config["max_value"] + 32
        config["max_length"] = config["max_input_size"] * 2 + 7 + (config["max_input_size"] + 3) ** 2
        config["cot_length"] = cot_length
        self.config = config

    def pointwise_loss_fn(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = F.cross_entropy(
            output.view(-1, output.size(-1)), target.view(-1), ignore_index=self.config["ignore_index"]
        )
        return loss

    def accuracy_fn(self, output_idx: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        eos_token_id = 3
        batch_size, _ = output_idx.shape

        eos_mask = output_idx == eos_token_id
        first_eos_idx = eos_mask.float().cumsum(dim=1).eq(1).float().argmax(dim=1)
        pred_idx = (first_eos_idx - 1).clamp(min=0)

        final_preds = output_idx[torch.arange(batch_size), pred_idx]
        final_tgts = target

        accuracy = (final_preds == final_tgts).float().mean()
        return accuracy

    @staticmethod
    def collate_fn(batch):
        PAD_ID = 0
        IGNORE_ID = -100
        seqs, labels = zip(*batch)
        padded_inp = pad_sequence(seqs, batch_first=True, padding_value=PAD_ID)  # (B, L_max)

        if isinstance(labels[0], torch.Tensor) and labels[0].dim() == 0:
            padded_tgt = torch.stack(labels)
        else:
            padded_tgt = pad_sequence(labels, batch_first=True, padding_value=IGNORE_ID)

        return padded_inp, padded_tgt


if __name__ == "__main__":
    task = EditDistanceTask()
    dataset = PairwiseAlignmentDataset(task.config, split="test")
    for i in range(5):
        inp, tgt = dataset[i]
        print(f"Input: {inp}, Target: {tgt}")
