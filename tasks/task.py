import abc
import torch
from torch.utils.data import Dataset
from experiments.curriculum import Curriculum

import torch.nn.functional as F


class GeneralizationTask(abc.ABC):
    def pointwise_loss_fn(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # last_logits = output[:, -1, :]
        # last_target = target[:, -1]
        # loss = F.cross_entropy(last_logits, last_target, ignore_index=-1)
        # return loss
        loss = F.cross_entropy(output.view(-1, output.size(-1)), target.view(-1), ignore_index=-1)
        return loss

    def accuracy_fn(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # last_logits = output[:, -1, :]  # (B, V)
        # last_target = target[:, -1]  # (B,)
        # pred = last_logits.argmax(dim=-1)  # (B,)
        # return (pred == last_target).float().mean()  # scalar accuracy
        pred = output.argmax(dim=-1)  # (B, T)
        return (pred == target).float().mean()


class CurriculumDataset(Dataset):
    def __init__(self):
        self.curriculum = None

    def set_curriculum(self, curriculum: Curriculum | None):
        self.curriculum = curriculum
