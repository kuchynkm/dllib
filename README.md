# DLlib: deep learning library from scratch

This library was built as a result of my attempt to write simple deep learning models from scratch using only numpy.

It contains only a limited number of building blocks (more may be added in the future).

### Layers:
* linear layer
* relu activation layer
* sigmoid activation layer

### Metrics:
* mean squared error
* binary cross-entropy

### Optimizers:
* Stochastic gradient descent

Only sequential models can be built using these building blocks.

## Examples
Usage is illustrated on binary classification and regression problems in jupyter notebooks [here](https://github.com/kuchynkm/dllib/tree/master/examples).


## Installation
```
pip install git+https://github.com/kuchynkm/dllib.git
```


## Usage
### Classification:
```
from dllib.metrics.classification import BinaryCrossEntropy
from dllib.layers import Linear, Sigmoid, ReLu
from dllib.model import SequentialNN
from dllib.optimizers import SGD


model = SequentialNN(
    layers=[
        Linear(input_size=X.shape[1], output_size=16),
        ReLu(),
        Linear(input_size=16, output_size=1),
        Sigmoid(),
    ],
    loss=BinaryCrossEntropy(),
    optimizer=SGD(learning_rate=1e-2)
)

model.fit(
    X, y,
    epochs=2000,
    batch_size=16,
    verbose=True,
)
```

### Regression:
```
from dllib.metrics.regression import MeanSquaredError
from dllib.layers import Linear, Sigmoid, ReLu
from dllib.model import SequentialNN
from dllib.optimizers import SGD


model = SequentialNN(
    layers=[
        Linear(input_size=X.shape[1], output_size=16),
        ReLu(),
        Linear(input_size=16, output_size=1),
    ],
    loss=MeanSquaredError(),
    optimizer=SGD(learning_rate=1e-3)
)

model.fit(
    X, y,
    epochs=2000,
    batch_size=16,
    verbose=True,
)
```
