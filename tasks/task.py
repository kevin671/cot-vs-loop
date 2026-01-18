import abc

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from experiments.curriculum import Curriculum


class GeneralizationTask(abc.ABC):
    def pointwise_loss_fn(self, output: torch.Tensor, target: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
        loss = F.cross_entropy(output.view(-1, output.size(-1)), target.view(-1), ignore_index=ignore_index)
        return loss

    def accuracy_fn(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = output.argmax(dim=-1)  # (B, T)
        return (pred == target).float().mean()


class CurriculumDataset(Dataset):
    def __init__(self):
        self.curriculum = None

    def set_curriculum(self, curriculum: Curriculum | None):
        self.curriculum = curriculum


class GeneralizationTaskChain(GeneralizationTask):
    def pointwise_loss_fn(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = F.cross_entropy(
            output.view(-1, output.size(-1)), target.view(-1), ignore_index=self.config["ignore_index"]
        )
        return loss

    def accuracy_fn(self, output_idx: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        eos_token_id = self.config["eos_token_id"]
        batch_size, seq_len = output_idx.shape
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
        EOS_ID = 4

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
