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
