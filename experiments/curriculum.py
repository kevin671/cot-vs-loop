import abc
import numpy as np
from typing import Optional


class Curriculum(abc.ABC):
    """Curriculum to sample sequence lengths."""

    def __init__(self):
        self.global_step: int = 0

    def step(self) -> None:
        """Advances the global step counter."""
        self.global_step += 1

    @abc.abstractmethod
    def sample_sequence_length(self) -> int:
        """Samples a sequence length based on the current curriculum state."""
        pass


class RegularIncreaseCurriculum(Curriculum):
    """Regularly increases the maximum sequence length."""

    def __init__(
        self,
        initial_sequence_length: int,
        increase_frequency: int,
        increase_amount: int,
        sample_all_length: bool = False,
        max_sequence_length: Optional[int] = None,
        warmup_steps: int = 0,
    ):
        super().__init__()
        assert increase_frequency > 0, "increase_frequency must be ≥ 1"
        self.initial_sequence_length = initial_sequence_length
        self.increase_frequency = increase_frequency
        self.increase_amount = increase_amount
        self.sample_all_length = sample_all_length
        self.max_sequence_length = max_sequence_length
        self.warmup_steps = warmup_steps

    def current_max_length(self) -> int:
        if self.global_step < self.warmup_steps:
            return self.initial_sequence_length

        effective_step = self.global_step - self.warmup_steps
        inc_steps = effective_step // self.increase_frequency
        length = self.initial_sequence_length + self.increase_amount * inc_steps
        if self.max_sequence_length is not None:
            length = min(length, self.max_sequence_length)
        return length

    def sample_sequence_length(self) -> int:
        max_len = self.current_max_length()
        if self.sample_all_length:
            length = int(np.random.randint(self.initial_sequence_length, max_len + 1))
        else:
            length = max_len
        return length


class GeometricIncreaseCurriculum(Curriculum):
    """Geometrically increases the max sequence length: 2, 4, 8, …, 512."""

    def __init__(
        self,
        initial_sequence_length: int,
        increase_frequency: int,
        increase_factor: int = 2,
        sample_all_length: bool = False,
        max_sequence_length: Optional[int] = None,
        warmup_steps: int = 0,
    ):
        super().__init__()
        assert increase_frequency > 0
        assert increase_factor == 2
        self.initial_sequence_length = initial_sequence_length
        self.increase_frequency = increase_frequency
        self.increase_factor = increase_factor
        self.sample_all_length = sample_all_length
        self.max_sequence_length = max_sequence_length
        self.warmup_steps = warmup_steps

    def current_max_length(self) -> int:
        if self.global_step < self.warmup_steps:
            return self.initial_sequence_length

        effective_step = self.global_step - self.warmup_steps
        inc_steps = effective_step // self.increase_frequency

        length = self.initial_sequence_length * (self.increase_factor**inc_steps)

        if self.max_sequence_length is not None:
            length = min(length, self.max_sequence_length)

        return int(length)

    def sample_sequence_length(self) -> int:
        max_len = self.current_max_length()
        if self.sample_all_length:
            return int(np.random.randint(self.initial_sequence_length, max_len + 1))
        else:
            return max_len
