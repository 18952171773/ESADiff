import torch
import torch.nn.functional as F

def adaptive_directional_smoothing_loss(image, mask, edge_weight=1.0, smooth_weight=0.5):
    """
    自适应方向平滑损失，用于保持分割图像中目标的边缘清晰和线性结构的连贯性。

    参数:
    image (torch.Tensor): 输入图像，形状为 (B, C, H, W)
    mask (torch.Tensor): 分割标签，形状为 (B, C, H, W)
    edge_weight (float): 用于加强边缘的权重
    smooth_weight (float): 控制平滑的权重

    返回:
    torch.Tensor: 自适应方向平滑损失值
    """

    # 计算x方向和y方向的梯度
    grad_x = image[:, :, :, 1:] - image[:, :, :, :-1]
    grad_y = image[:, :, 1:, :] - image[:, :, :-1, :]

    # 计算标签的边缘 (例如通过Sobel算子或Canny边缘检测)
    mask_grad_x = mask[:, :, :, 1:] - mask[:, :, :, :-1]
    mask_grad_y = mask[:, :, 1:, :] - mask[:, :, :-1, :]

    # 计算边缘的权重：对mask梯度的绝对值进行加权
    edge_mask_x = torch.exp(-torch.abs(mask_grad_x) * edge_weight)
    edge_mask_y = torch.exp(-torch.abs(mask_grad_y) * edge_weight)

    # 计算梯度差异（x方向和y方向）
    diff_x = torch.abs(grad_x[:, :, :, 1:] - grad_x[:, :, :, :-1]) * edge_mask_x
    diff_y = torch.abs(grad_y[:, :, 1:, :] - grad_y[:, :, :-1, :]) * edge_mask_y

    # 总损失是x方向和y方向梯度差异的加权和
    loss = smooth_weight * (torch.mean(diff_x) + torch.mean(diff_y))
    
    return loss

