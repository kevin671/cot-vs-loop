import abc
import torch


class GeneralizationTask(abc.ABC):
    def pointwise_loss_fn(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Pointwise loss function to be implemented by subclasses."""
        raise NotImplementedError("Subclasses should implement this method.")

    def accuracy_fn(self, output, target):
        """Accuracy function to be implemented by subclasses."""
        raise NotImplementedError("Subclasses should implement this method.")
