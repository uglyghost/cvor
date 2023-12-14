# CVor

CVor is a Python package designed for advanced loss computation in neural networks, particularly with PyTorch. 
It offers a unique approach to calculating loss using three different methods: AO, NN, and LOO. 
CVor provides low-variance, unbiased gradient estimation based on control variates.
It uniquely focuses on transforming the gradient mapping process and demonstrating superior performance in various machine learning benchmarks, including variational autoencoder training and reinforcement learning tasks.

## Features

- **Multiple Loss Computation Methods**: 'AO' (Average Optimization), 'NN' (Neural Network based), and 'LOO' (Leave-One-Out) methods.
- **Flexible Alpha Parameter**: Allows setting the alpha parameter within the range [0, 1] for CVor loss adjustment.

## Installation

You can install CVor using pip:

```bash
pip install cvor
```

## Usage

Here's a quick example of how to use CVor:

```python
import torch
from CVor import CVor_loss_PyTorch

# Sample loss tensor
loss_input = torch.tensor([1.0, 2.0, 3.0], device='cuda')

# Compute CVor loss
loss = CVor_loss_PyTorch(loss_input, method='NN', alpha=0.1)
print(loss)
```

### Parameters

- `loss_input (Tensor)`: The input tensor representing loss.
- `method (str)`: The method for computing CVor loss. Options are 'AO', 'NN', and 'LOO'.
- `alpha (float)`: The adjustment coefficient. Must be between 0 and 1.


## F_value Calculation

CVor now supports calculation of the `F_value`, which is a key component in computing the CVor loss. 
This feature allows users to inject their own logic into the loss calculation process, offering greater flexibility and adaptability to specific needs.

### Using F_calculator

To use this feature, define a function that takes `loss_input` and `alpha` as parameters and returns the calculated `F_value`. 
This function can then be passed to `CVor_loss_PyTorch` as the `F_calculator` argument.

#### Example

Here's an example of how to define and use a F_value calculation function:

```python
import torch
from CVor import CVor_loss_PyTorch

# F_value calculation function
def F_calculator(loss_input, alpha):
    mean_value = loss_input.mean()
    F_value = alpha * mean_value / loss_input.sum()
    return F_value

# Sample loss tensor
loss_input = torch.tensor([1.0, 2.0, 3.0], device='cuda')

# Compute CVor loss using F_value calculator
loss = CVor_loss_PyTorch(loss_input, F_calculator=F_calculator)
print(loss)
```

In this example, `F_calculator` computes `F_value` based on the mean of the input tensor and the `alpha` value. You can define the logic of `F_value` calculation as per your requirements.

### Note

When using `F_calculator`, the `method` parameter in `CVor_loss_PyTorch` is ignored, and the provided function is used instead for the `F_value` calculation.


## Requirements

- Python 3.8+
- PyTorch

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to check [issues page](https://github.com/yourusername/cvor/issues) if you want to contribute.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

CHEN XINGYAN - xychen@swufe.edu.cn

Project Link: [https://github.com/uglyghost/cvor.git](https://github.com/uglyghost/cvor)