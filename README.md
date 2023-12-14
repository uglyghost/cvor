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