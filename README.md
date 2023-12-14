# CVor

CVor is a Python package designed for advanced loss computation in neural networks, particularly with PyTorch. It offers a unique approach to calculating loss using three different methods: AO, NN, and LOO. This flexibility allows for a more customized and efficient optimization process in machine learning models.

## Features

- **Multiple Loss Computation Methods**: Supports 'AO' (Average Optimization), 'NN' (Neural Network based), and 'LOO' (Leave-One-Out) methods.
- **Flexible Alpha Parameter**: Allows setting the alpha parameter within the range [0, 1] for loss computation adjustment.
- **CUDA Support**: Automatically detects and utilizes CUDA if available, for enhanced performance.

## Installation

You can install CVor using pip:

```bash
pip install cvor
```

## Usage

Here's a quick example of how to use CVor:

```python
import torch
from cvor import compute_cvor_loss

# Sample loss tensor
loss_input = torch.tensor([1.0, 2.0, 3.0], device='cuda')

# Compute CVor loss
loss = compute_cvor_loss(loss_input, method='NN', alpha=0.1)
print(loss)
```

### Parameters

- `loss_input (Tensor)`: The input tensor representing loss.
- `method (str)`: The method for computing CVor loss. Options are 'AO', 'NN', and 'LOO'.
- `alpha (float)`: The adjustment coefficient. Must be between 0 and 1.

## Requirements

- Python 3.8
- PyTorch

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to check [issues page](https://github.com/yourusername/cvor/issues) if you want to contribute.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

CHEN XINGYAN - xychen@swufe.edu.cn

Project Link: [https://github.com/uglyghost/cvor.git](https://github.com/uglyghost/cvor)