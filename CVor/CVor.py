import torch
import torch.nn as nn


def calculate_F_value_AO(F_value, alpha):
    return alpha * F_value / F_value.sum()


def calculate_F_value_NN(F_value, alpha, device):
    F1 = nn.Sequential(
        nn.Linear(len(F_value), len(F_value) * 2),
        nn.Tanh(),
        nn.Linear(len(F_value) * 2, len(F_value))
    ).to(device)
    F_value_nn = F1(F_value)
    return alpha * F_value_nn / F_value_nn.sum()


def calculate_F_value_LOO(F_value, alpha):
    F_value_sum = F_value.sum()
    loo_F_value = (F_value_sum - F_value) / (len(F_value) - 1)
    return alpha * loo_F_value / loo_F_value.sum()


def CVor_loss_PyTorch(loss_input, F_value, method='AO', alpha=0.1):
    """
    Calculate the CVor loss.

    Parameters:
    loss_input (Tensor): The input loss tensor.
    F_value (Tensor): The F value tensor.
    method (str): The calculation method, can be 'AO', 'NN', or 'LOO'.
    alpha (float): The adjustment coefficient, ranging from 0 to 1.

    Returns:
    Tensor: The calculated CVor loss.

    Raises:
    ValueError: If 'alpha' is not in the range [0, 1] or if an invalid method is specified.
    """

    if not 0 <= alpha <= 1:
        raise ValueError("'alpha' value must be between 0 and 1.")

    device = 'cuda' if loss_input.is_cuda else 'cpu'
    loss_input = loss_input.to(device)
    F_value = F_value.to(device)

    if method == 'AO':
        F_value = calculate_F_value_AO(F_value, alpha)
    elif method == 'NN':
        F_value = calculate_F_value_NN(F_value, alpha, device)
    elif method == 'LOO':
        F_value = calculate_F_value_LOO(F_value, alpha)
    else:
        raise ValueError("Invalid method. Must be 'AO', 'NN', or 'LOO'.")

    tilde_F_value = torch.exp(F_value - F_value.detach()).mean()
    CVor = torch.exp(tilde_F_value - tilde_F_value.detach() - F_value + F_value.detach())
    final_loss = CVor.mean() * loss_input

    return final_loss
