import abc
import math
from typing import Optional

import numpy as np


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
    """
    Geometrically increases the maximum sequence length: L₀, 2 L₀, 4 L₀, …

    For a given length L, the curriculum allocates
        S(L) = S₀ · log₂(L / L₀)
    training steps (with a minimum of S₀).
    """

    def __init__(
        self,
        initial_sequence_length: int,  # L₀
        base_steps: int,  # S₀
        increase_factor: int = 2,
        sample_all_length: bool = False,
        max_sequence_length: Optional[int] = None,
        warmup_steps: int = 0,
    ):
        super().__init__()
        self.L0 = initial_sequence_length
        self.S0 = base_steps
        self.r = increase_factor
        self.sample_all_length = sample_all_length
        self.L_max = max_sequence_length
        self.warmup_steps = warmup_steps

    def _steps_for_length(self, length: int) -> int:
        ratio = max(1, length // self.L0)           # L / L0
        k = max(1, int(math.log2(ratio)))           # log2(L/L0) (最小1)
        return self.S0 * k

    def current_max_length(self) -> int:
        if self.global_step < self.warmup_steps:
            return self.L0

        remaining = self.global_step - self.warmup_steps
        length = self.L0

        while True:
            steps_here = self._steps_for_length(length)
            if remaining < steps_here:
                break

            remaining -= steps_here
            length *= self.r
            if self.L_max and length >= self.L_max:
                length = self.L_max
                break

        return length

    def sample_sequence_length(self) -> int:
        L = self.current_max_length()
        if self.sample_all_length:
            return int(np.random.randint(self.L0, L + 1))
        return L


class FixedLengthCurriculum(Curriculum):
    """Always returns a fixed sequence length."""

    def __init__(self, sequence_length: int):
        super().__init__()
        self.sequence_length = sequence_length

    def sample_sequence_length(self) -> int:
        return self.sequence_length
