# Custom Neural Network Components with NumPy

This repository contains custom implementations of neural network components using NumPy. These modules are designed to provide foundational components for building neural networks from scratch, focusing on clarity and educational value.

## Modules Included

### Sequential Container
- **Sequential**: A sequential container to stack layers sequentially.

### Layers
1. **Linear Layer**: Implementation of a linear (fully connected) layer with support for forward and backward passes.
2. **SoftMax Layer**: Implementation of the SoftMax activation function.
3. **LogSoftMax Layer**: Implementation of the LogSoftMax activation function.
4. **Batch Normalization**: Implementation of batch normalization with support for training and evaluation modes.
5. **Dropout Layer**: Implementation of dropout regularization.

### Activation Functions
6. **Leaky ReLU**: Implementation of Leaky ReLU activation function.
7. **ELU (Exponential Linear Unit)**: Implementation of ELU activation function.
8. **SoftPlus**: Implementation of the SoftPlus activation function.

### Loss Functions
9. **Negative LogLikelihood Criterion (Unstable)**: An unstable version of Negative Log-Likelihood loss (NLLLoss) for classification tasks.
10. **Negative LogLikelihood Criterion (Stable)**: A stable version of NLLLoss, designed for numerical stability.

### Optimizers
11. **SGD Optimizer with Momentum**: Implementation of Stochastic Gradient Descent (SGD) optimizer with momentum.
12. **Adam Optimizer**: Implementation of the Adam optimizer.

### Other Criteria
- **MSE Criterion**: Mean Squared Error (MSE) criterion for regression tasks.
- **Abs Criterion**: Absolute loss criterion for regression tasks.

## Usage

Each module (`Linear`, `SoftMax`, etc.) can be imported and used independently. They are designed to mimic the behavior of similar modules found in deep learning frameworks like PyTorch, but implemented using NumPy for educational purposes.

## Usage

Each module (`Linear`, `SoftMax`, etc.) can be imported and used independently. They are designed to replicate the functionality of similar modules found in deep learning frameworks like PyTorch, but implemented using NumPy for educational purposes.

Example usage in a Jupyter Notebook:

```markdown
import numpy as np
from linear import Linear
from softmax import SoftMax
from logsoftmax import LogSoftMax

# Create instances of layers
linear_layer = Linear(10, 5)
softmax_layer = SoftMax()
logsoftmax_layer = LogSoftMax()

# Forward pass example
input_data = np.random.randn(10, 10)
output = linear_layer.forward(input_data)
softmax_output = softmax_layer.forward(output)
logsoftmax_output = logsoftmax_layer.forward(output)

print("Linear layer output shape:", output.shape)
print("SoftMax layer output shape:", softmax_output.shape)
print("LogSoftMax layer output shape:", logsoftmax_output.shape)


# Create instances of layers
linear_layer = Linear(10, 5)
softmax_layer = SoftMax()
logsoftmax_layer = LogSoftMax()

# Forward pass example
input_data = np.random.randn(10, 10)
output = linear_layer.forward(input_data)
softmax_output = softmax_layer.forward(output)

