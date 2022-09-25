"""Layers for neural network."""
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, Callable, Optional, Tuple

import numpy as np

from dllib.tensors import Tensor
from dllib.utils import math

# Tensor function type
TensorFunction = Callable[[Tensor], Tensor]


class Layer(ABC):
    """Abstract layer class."""

    def __init__(self) -> None:
        self._weights: dict[str, Tensor] = {}
        self._grads: dict[str, Tensor] = {}
        self.input_tensor: Optional[Tensor] = None

    @abstractmethod
    def forward_propagation(self, input_tensor: Tensor) -> Tensor:
        """Propagates of input tensor through layer."""

    @abstractmethod
    def backward_propagation(self, grad: Tensor) -> Tensor:
        """Back-propagates gradient tensor of previous layer through layer."""

    def get_weights_and_grads(
        self,
    ) -> Iterator[Tuple[Tensor, Tensor]]:
        """Returns iterator over layer (weights, gradients) tuples."""
        for name, weights in self._weights.items():
            yield weights, self._grads[name]

    def __call__(self, input_tensor: Tensor, **kwargs: Optional[Any]) -> Tensor:
        return self.forward_propagation(input_tensor, **kwargs)


class Linear(Layer):
    """Linear layer class."""

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self._weights["kernel"] = np.random.randn(input_size, output_size)
        self._weights["bias"] = np.random.randn(output_size)

    def forward_propagation(
        self, input_tensor: Tensor, keep_input: bool = False
    ) -> Tensor:

        if keep_input:
            # needed during back-propagation
            self.input_tensor = input_tensor

        return input_tensor @ self._weights["kernel"] + self._weights["bias"]

    def backward_propagation(self, grad: Tensor) -> Tensor:
        if self.input_tensor is None:
            raise ValueError("Forward propagation must be called first.")

        self._grads["bias"] = np.sum(grad, axis=0)
        self._grads["kernel"] = self.input_tensor.T @ grad
        return grad @ self._weights["kernel"].T


class Activation(Layer):
    """Activation layer class."""

    def __init__(self, fn: TensorFunction, fn_prime: TensorFunction) -> None:
        super().__init__()
        self.fn = fn
        self.fn_prime = fn_prime

    def forward_propagation(
        self, input_tensor: Tensor, keep_input: bool = False
    ) -> Tensor:

        if keep_input:
            # needed during back-propagation
            self.input_tensor = input_tensor

        return self.fn(input_tensor)

    def backward_propagation(self, grad: Tensor) -> Tensor:
        return self.fn_prime(self.input_tensor) * grad


class ReLu(Activation):
    """ReLu activation layer."""

    def __init__(self) -> None:
        super().__init__(fn=math.relu, fn_prime=math.relu_prime)


class Sigmoid(Activation):
    """ReLu activation layer."""

    def __init__(self) -> None:
        super().__init__(fn=math.sigmoid, fn_prime=math.sigmoid_prime)
