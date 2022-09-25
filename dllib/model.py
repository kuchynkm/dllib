"""
Neural network built upon layer objects.
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Optional, Tuple

from dllib.layers import Layer
from dllib.metrics.loss import Loss
from dllib.optimizers import Optimizer
from dllib.tensors import Tensor
from dllib.utils.data import BatchGenerator


class NeuralNetwork:
    """Sequential neural network class."""

    def __init__(self, layers: list[Layer], loss: Loss, optimizer: Optimizer) -> None:
        self.layers = layers
        self.loss = loss
        self.optimizer = optimizer

    def forward_propagation(
        self, input_tensor: Tensor, **kwargs: Optional[Any]
    ) -> Tensor:
        """Propagates input tensor through model layers."""

        output_tensor = input_tensor.copy()

        for layer in self.layers:
            output_tensor = layer.forward_propagation(output_tensor, **kwargs)

        return output_tensor

    def backward_propagation(self, grad: Tensor) -> Tensor:
        """Back-propagates loss gradient through model layers."""

        for layer in reversed(self.layers):
            grad = layer.backward_propagation(grad)

        return grad

    def get_weights_and_grads(self) -> Iterator[Tuple[Tensor, Tensor]]:
        """Returns iterator over (weights, gradients) tuple."""
        for layer in self.layers:
            for weights, grads in layer.get_weights_and_grads():
                yield weights, grads

    def predict(self, X: Tensor, **kwargs: Any) -> Tensor:
        """Returns model prediction."""
        return self.forward_propagation(X, **kwargs)

    def fit(
        self,
        X: Tensor,
        y: Tensor,
        epochs: int = 1,
        batch_size: int = 1,
        shuffle: bool = True,
        verbose: bool = True,
        **_: Optional[Any],
    ) -> NeuralNetwork:
        """Fits the model on data and returns trained model instance."""
        batch_generator = BatchGenerator(batch_size=batch_size, shuffle=shuffle)

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0

            for X_batch, y_batch in batch_generator(X, y):
                batch_loss = self._train_step(X_batch, y_batch)
                epoch_loss += batch_loss

            if verbose:
                print(f"Epoch {epoch} / {epochs}: loss = {epoch_loss}")

        return self

    def _train_step(self, X: Tensor, y: Tensor) -> float:
        """Performs single training step and returns loss."""

        # calculate batch of predictions
        y_pred = self.predict(X, keep_input=True)
        # calculate batch loss
        loss = self.loss(y_true=y, y_pred=y_pred)
        # calculate batch loss gradient
        loss_grad = self.loss.grad(y_true=y, y_pred=y_pred)
        # calculate layer weight gradients
        self.backward_propagation(loss_grad)
        # update model weights with calculated gradient
        self.optimizer.apply_gradients(self.get_weights_and_grads())

        return loss
