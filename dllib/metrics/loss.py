"""
Abstrac loss function class.
"""

from abc import ABC, abstractmethod

from dllib.tensors import Tensor


class Loss(ABC):
    """Abstract loss function class."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def call(self, y_true: Tensor, y_pred: Tensor) -> float:
        ...

    @abstractmethod
    def grad(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        ...

    def __call__(self, y_true: Tensor, y_pred: Tensor) -> float:
        return self.call(y_true, y_pred)
