import abc

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from experiments.curriculum import Curriculum


class GeneralizationTask(abc.ABC):
    def pointwise_loss_fn(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = F.cross_entropy(output.view(-1, output.size(-1)), target.view(-1), ignore_index=-1)
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
