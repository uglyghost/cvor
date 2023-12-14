import torch
import torch.nn as nn


def CVor_loss_PyTorch(loss_input, method='AO', alpha=0.1):
    """
    计算CVor损失。

    参数:
    loss_input (Tensor): 输入的损失张量。
    method (str): 计算方法，可以是 'AO'、'NN' 或 'LOO'。
    alpha (float): 调节系数，取值范围为 0 到 1。

    返回:
    Tensor: 计算出的CVor损失，或错误信息。
    """
    # 检查alpha的有效性
    if not 0 <= alpha <= 1:
        return "Error: 'alpha' value must be between 0 and 1."

    # 检查输入的loss是否在CUDA上
    device = 'cuda' if loss_input.is_cuda else 'cpu'

    # 确保输入也在相同的设备
    loss_input = loss_input.to(device)

    # 根据不同的方法计算F_value
    if method == 'AO':
        F_value = alpha * loss_input / loss_input.sum()
    elif method == 'NN':
        # 定义神经网络
        F1 = nn.Sequential(
            nn.Linear(len(loss_input), len(loss_input) * 2),
            nn.Tanh(),
            nn.Linear(len(loss_input) * 2, len(loss_input))
        ).to(device)
        F_value = (F1(loss_input) + alpha * loss_input / loss_input.sum())
    elif method == 'LOO':
        # 计算去一法的loss
        loss_sum = loss_input.sum()
        loo_loss = (loss_sum - loss_input) / (len(loss_input) - 1)
        F_value = alpha * loo_loss / loo_loss.sum()
    else:
        return "Error: Invalid method. Must be 'AO', 'NN', or 'LOO'."

    # 计算tilde F值
    tilde_F_value = torch.exp(F_value - F_value.detach()).mean()

    # 计算CVor
    CVor = torch.exp((torch.exp(tilde_F_value - tilde_F_value.detach()) - torch.exp(F_value - F_value.detach())))

    # 计算最终的损失
    final_loss = (CVor * loss_input).mean()

    return final_loss
