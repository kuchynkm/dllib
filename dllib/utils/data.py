from collections.abc import Iterator
from typing import Tuple

import numpy as np

from dllib.tensors import Tensor


class BatchGenerator:
    def __init__(self, batch_size: int = 16, shuffle: bool = False) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, X: Tensor, y: Tensor) -> Iterator[Tuple[Tensor, Tensor]]:
        start_range = np.arange(0, len(X), self.batch_size)

        if self.shuffle:
            np.random.shuffle(start_range)

        for start in start_range:
            end = start + self.batch_size
            X_batch = X[start:end]
            y_batch = y[start:end]
            yield (X_batch, y_batch)
