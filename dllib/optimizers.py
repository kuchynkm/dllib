from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Tuple

from dllib.tensors import Tensor


class Optimizer(ABC):
    @abstractmethod
    def apply_gradients(self, vars_and_grads: Iterator[Tuple[Tensor, Tensor]]) -> None:
        ...


class SGD(Optimizer):
    """Stochastic gradient descent optimizer."""

    def __init__(self, learning_rate: float = 1e-1):
        self.learning_rate = learning_rate

    def apply_gradients(self, vars_and_grads: Iterator[Tuple[Tensor, Tensor]]) -> None:
        for var, grad in vars_and_grads:
            var -= self.learning_rate * grad
