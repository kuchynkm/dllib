import numpy as np

from dllib.tensors import Tensor


def relu(input_tensor: Tensor) -> Tensor:
    return np.maximum(0, input_tensor)


def relu_prime(input_tensor: Tensor) -> Tensor:
    output_tensor = input_tensor.copy()
    output_tensor[output_tensor < 0] = 0
    output_tensor[output_tensor > 0] = 1
    return output_tensor
