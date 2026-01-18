from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from tasks.task import CurriculumDataset, GeneralizationTask, GeneralizationTaskChain


class ReachabilityDataset(CurriculumDataset):
    def __init__(self, config, split="train", chain: bool = False, is_coconut: bool = False, lambda_distribution=None):
        super().__init__()
        self.config = config
        self.split = split
        self.chain = chain
        self.is_coconut = is_coconut

        if chain:

            def build_fixed_vocab(max_input_size: int) -> Dict[str, int]:
                tok2id: Dict[str, int] = {"<pad>": 0, "TRUE": 1, "FALSE": 2, "N": 3, "<eos>": 4, "|": 5}
                # vertices
                for i in range(max_input_size):
                    tok2id[i] = len(tok2id)
                return tok2id

        else:

            def build_fixed_vocab(max_input_size: int) -> Dict[str, int]:
                tok2id: Dict[str, int] = {"<pad>": 0, "TRUE": 1, "FALSE": 2}

                # vertices
                for i in range(max_input_size):
                    tok2id[f"v{i}"] = len(tok2id)
                # ordered pairs
                for i in range(max_input_size):
                    for j in range(max_input_size):
                        if i != j:
                            tok2id[f"{i},{j}"] = len(tok2id)
                return tok2id

        max_input_size = config["max_input_size"]
        self.max_input_size = max_input_size
        self.token2id = build_fixed_vocab(max_input_size)

        # self.samples: List[Tuple[List[int], int]] = []
        self.samples = {}
        d = config["min_input_size"]
        while d <= max_input_size:
            file_path = (
                Path(config["data_dir"]) / str(d) / f"{split}.txt"
                if not chain
                else Path(config["data_dir"]) / str(d) / "chain" / f"{split}.txt"
            )
            self.samples[d] = []
            with file_path.open() as f:
                for line in f:
                    if chain:
                        _, edges, query, cot, label = line.rstrip().split("\t")
                        # edges = [int(n) for pair in edges.split() for n in pair.split(",")]
                        query = [int(n) for n in query.split(",")]
                        edges = edges.replace(" ", ",|,").split(",")  # if edges else []
                        edges = [t for t in edges if t != ""]  # hot fix
                        edges = [int(t) if t != "|" else "|" for t in edges]

                        if split == "train":
                            cot_len = config["cot_length"]
                            random_removal_offset = (
                                torch.multinomial(lambda_distribution, 1).item()
                                if lambda_distribution is not None
                                else 0
                            )
                            cot_len = cot_len - random_removal_offset if cot_len is not None else None
                            cot_tokens = [int(n) if n.isdigit() else n for pair in cot.split() for n in pair.split(",")]

                            total_len = len(cot_tokens)
                            if cot_len is None:
                                sampled_cot = cot_tokens
                            elif total_len <= cot_len:
                                sampled_cot = cot_tokens
                            else:
                                # indices = np.linspace(0, total_len - 1, cot_len, dtype=int).tolist()
                                # sampled_cot = [cot_tokens[i] for i in indices]
                                sampled_cot = cot_tokens[-cot_len:]  # TODO

                            inp = edges + ["|"] + query
                            if is_coconut:
                                # store precomputed token ids for coconut mode
                                inp_ids = [self.token2id[tok] for tok in inp]
                                cot_ids = [self.token2id[tok] for tok in sampled_cot] + [int(label) + 1]
                                self.samples[d].append((inp_ids, cot_ids, None))

                            else:
                                tokens = inp + sampled_cot
                                # label: 1 for TRUE, 2 for FALSE
                                input_ids = [self.token2id[tok] for tok in tokens] + [int(label) + 1]
                                label_ids = torch.tensor(input_ids[1:] + [self.token2id["<eos>"]], dtype=torch.long)
                                label_ids[: len(inp) - 1] = config["ignore_index"]
                                self.samples[d].append((input_ids, label_ids))

                        else:
                            inp = edges + ["|"] + query
                            inp_ids = [self.token2id[tok] for tok in inp]
                            if is_coconut:
                                self.samples[d].append((inp_ids, int(label) + 1))
                            else:
                                self.samples[d].append((inp_ids, int(label) + 1))
                    else:
                        verts, edges, query, label = line.rstrip().split("\t")
                        tokens = [query] + verts.split() + edges.split()
                        ids = [self.token2id[tok] for tok in tokens]
                        self.samples[d].append((ids, int(label) + 1))  # label: 1 for TRUE, 2 for FALSE
            # d *= 2
            d += 4

    def __len__(self) -> int:
        return len(self.samples[self.max_input_size])

    def __getitem__(self, idx: int):
        length = self.curriculum.sample_sequence_length() if self.curriculum else max(self.samples.keys())
        if self.chain:
            if self.is_coconut:
                if self.split == "train":
                    inp_ids, cot_ids, _ = self.samples[length][idx]
                    inp_tensor = torch.tensor(inp_ids, dtype=torch.long)
                    cot_tensor = torch.tensor(cot_ids, dtype=torch.long)
                    # tgt_tensor = tgt if isinstance(tgt, torch.Tensor) else torch.tensor(tgt, dtype=torch.long)
                    return inp_tensor, cot_tensor, None  # tgt_tensor
                else:
                    ids, label = self.samples[length][idx]
                    ids_tensor = ids if isinstance(ids, torch.Tensor) else torch.tensor(ids, dtype=torch.long)
                    label_tensor = label if isinstance(label, torch.Tensor) else torch.tensor(label, dtype=torch.long)
                    return ids_tensor, label_tensor
            else:
                ids, label = self.samples[length][idx]
                ids_tensor = ids if isinstance(ids, torch.Tensor) else torch.tensor(ids, dtype=torch.long)
                label_tensor = label if isinstance(label, torch.Tensor) else torch.tensor(label, dtype=torch.long)
                return ids_tensor, label_tensor
        else:
            ids, label = self.samples[length][idx]
            return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


class ReachabilityTask(GeneralizationTask):

    def __init__(self, max_input_size: int = 16) -> None:
        max_n = max_input_size
        self.config = {
            "name": "reachability",
            "data_dir": "data/path",
            "max_input_size": max_n,
            "min_input_size": 8,
            "vocab_size": max_n + max_n * (max_n - 1) + 1 + 2,  # nodes + edges + <pad> + labels
            "max_length": max_n + max_n * (max_n - 1) + 2,  # nodes + edges + <pad> + labels
        }

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


class ReachabilityTaskChain(GeneralizationTaskChain):
    def __init__(self, max_input_size: int = 16, cot_length: int = None) -> None:
        max_n = max_input_size
        config = {
            "name": "reachability",
            "data_dir": "data/path",
            "max_input_size": max_n,
            "min_input_size": max_n,
            "vocab_size": max_n + 5 + 2,  # nodes +  <pad> / N / <eos> / "|" + labels
            "eos_token_id": 4,  # <eos>
            "ignore_index": -100,
        }
        config["max_length"] = max_n * max_n  # not strictly but practically sufficient
        config["cot_length"] = cot_length
        self.config = config


if __name__ == "__main__":
    task = ReachabilityTaskChain(max_input_size=4, cot_length=20)
    dataset = ReachabilityDataset(task.config, split="train", chain=True)
    for i in range(5):
        inp, tgt = dataset[i]
        print(f"Input: {inp.tolist()}, Target: {tgt.tolist()}")
        # print(len(inp), len(tgt))
        # Input: [21, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 21, 31, 37, 43], Target: 2
