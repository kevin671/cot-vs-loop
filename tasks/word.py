import csv

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from tasks.task import CurriculumDataset, GeneralizationTask, GeneralizationTaskChain


class WordProblemDataset(CurriculumDataset):
    def __init__(self, config, split="train", chain: bool = False, is_coconut: bool = False, lambda_distribution=None):
        super().__init__()
        self.config = config
        self.split = split
        self.chain = chain
        self.is_coconut = is_coconut
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
                        random_removal_offset = (
                            torch.multinomial(lambda_distribution, 1).item() if lambda_distribution is not None else 0
                        )
                        cot_len = cot_len - random_removal_offset if cot_len is not None else None

                        cot, ans = tgt[:-1], tgt[-1]
                        total_len = len(cot)

                        if cot_len is None:
                            sampled_cot = cot
                        else:
                            if cot_len >= total_len:
                                sampled_cot = cot
                            else:
                                indices = np.linspace(0, total_len - 1, cot_len, dtype=int).tolist()
                                sampled_cot = [cot[i] for i in indices]
                                # sampled_cot = cot[-cot_len:]  # This is ineffective !!
                        input_ids = inp + sampled_cot + [ans]
                        label_ids = torch.tensor(input_ids[1:] + [eos_token_id], dtype=torch.long)
                        label_ids[: len(inp) - 1] = self.config["ignore_index"]
                        if is_coconut:
                            # store original input, CoT and labels separately for coconut mode
                            self.samples.append((inp, sampled_cot + [ans], label_ids))
                        else:
                            self.samples.append((input_ids, label_ids))
                    else:
                        label_id = tgt[-1]
                        if is_coconut:
                            # no CoT available for eval split; store empty cot
                            self.samples.append((inp, label_id))
                        else:
                            self.samples.append((inp, label_id))
                else:
                    self.samples.append((inp, tgt))

        # Note: padding not required — all input lengths are guaranteed equal

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.chain:
            if self.is_coconut and self.split == "train":
                inp, cot, tgt = self.samples[idx]
                inp_tensor = torch.tensor(inp, dtype=torch.long)
                cot_tensor = torch.tensor(cot, dtype=torch.long)
                tgt_tensor = tgt if isinstance(tgt, torch.Tensor) else torch.tensor(tgt, dtype=torch.long)
                return inp_tensor, cot_tensor, tgt_tensor
            else:
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

    def __init__(self):
        super().__init__()
        config = {
            "name": "word_problem",
            "description": "Compute prefix products over a sequence of group elements.",
            "csv_path": "data/word_problem/S5_64.csv",
            "vocab_size": 120,
            "max_length": 64,
            "ignore_index": -100,
            "min_input_size": 4,
        }
        self.config = config

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

    @staticmethod
    def collate_fn(batch):
        PAD_ID = 0
        IGNORE_ID = -100
        EOS_ID = 120

        if len(batch[0]) == 2:
            seqs, labels = zip(*batch)
            padded_inp = pad_sequence(seqs, batch_first=True, padding_value=PAD_ID)  # (B, L_max)

            if isinstance(labels[0], torch.Tensor) and labels[0].dim() == 0:
                padded_tgt = torch.stack(labels)
            else:
                padded_tgt = pad_sequence(labels, batch_first=True, padding_value=IGNORE_ID)

            return padded_inp, padded_tgt
        else:
            inps, cots, _ = zip(*batch)

            inps = [t if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.long) for t in inps]
            cots = [t if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.long) for t in cots]

            padded_inps = pad_sequence(
                [s.flip(0) for s in inps],
                batch_first=True,
                padding_value=PAD_ID,
            ).flip(1)
            # pad_sequence(inps, batch_first=True, padding_value=PAD_ID)  # (B, Linp_max)
            padded_cots = pad_sequence(cots, batch_first=True, padding_value=PAD_ID)  # (B, Lcot_max)

            B, Linp_max = padded_inps.shape
            _, Lcot_max = padded_cots.shape
            L = Linp_max + Lcot_max  # full input length (inp+cot)

            # [inp tokens] [inp padding] [cot tokens] [cot padding]
            full_inp = torch.cat([padded_inps, padded_cots], dim=1)  # (B, L)

            cot_lens = torch.tensor([len(x) for x in cots], dtype=torch.long)

            labels = torch.full((B, L), IGNORE_ID, dtype=torch.long)  # (B, L)
            labels[:, :-1] = full_inp[:, 1:]  # next token prediction

            for i in range(B):

                labels[i, : Linp_max - 1] = IGNORE_ID  # inp part
                labels[i, Linp_max + cot_lens[i] - 1] = EOS_ID
                labels[i, Linp_max + cot_lens[i] :] = IGNORE_ID  # cot padding

            return padded_inps, padded_cots, labels


if __name__ == "__main__":
    # Example usage
    task = WordProblemTaskChain(max_input_size=256, cot_length=None)
    dataset = WordProblemDataset(task.config, split="train", chain=True)
    for inp, tgt in dataset:
        print("Input:", inp.tolist())
        print("Target:", tgt.tolist())
        break
