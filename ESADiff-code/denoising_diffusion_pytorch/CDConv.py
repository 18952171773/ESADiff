# -*- coding: utf-8 -*-
import os
import torch
from torch import nn
import einops
import torch.nn.functional as F
import numpy as np



class CDCM(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 9,
        extend_scope: float = 1.0,
        morph: int = 0,
        if_offset: bool = True,
        device:  str="cuda:3"
    ):
        """
        A Dynamic Snake Convolution Implementation

        Based on:

            TODO

        Args:
            in_ch: number of input channels. Defaults to 1.
            out_ch: number of output channels. Defaults to 1.
            kernel_size: the size of kernel. Defaults to 9.
            extend_scope: the range to expand. Defaults to 1 for this method.
            morph: the morphology of the convolution kernel is mainly divided into two types along the x-axis (0) and the y-axis (1) (see the paper for details).
            if_offset: whether deformation is required,  if it is False, it is the standard convolution kernel. Defaults to True.

        """

        super().__init__()

        if morph not in (0, 1):
            raise ValueError("morph should be 0 or 1.")

        self.kernel_size = kernel_size
        self.extend_scope = extend_scope
        self.morph = morph
        self.if_offset = if_offset
        self.device = torch.device(device)
        self.to(device)

        # self.bn = nn.BatchNorm2d(2 * kernel_size)
        self.gn_offset = nn.GroupNorm(kernel_size, 2 * kernel_size)
        self.gn = nn.GroupNorm(out_channels // 4, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size, 3, padding=1)

        self.dsc_conv_x = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            stride=(kernel_size, 1),
            padding=0,
        )
        self.dsc_conv_y = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            stride=(1, kernel_size),
            padding=0,
        )
    def HessianAndEig2D(self, I, Sigma=1):
        #with torch.no_grad():  # 禁用梯度计算
        # 计算卷积核坐标
        range_val = round(3 * Sigma)  # 使用标量值计算范围
        X, Y = torch.meshgrid(torch.arange(-range_val, range_val + 1),
                            torch.arange(-range_val, range_val + 1))

        # 构建高斯二阶导数滤波器
        DGaussxx = 1 / (2 * np.pi * Sigma**4) * (X**2 / Sigma**2 - 1) * torch.exp(-(X**2 + Y**2) / (2 * Sigma**2))
        DGaussxy = 1 / (2 * np.pi * Sigma**6) * (X * Y) * torch.exp(-(X**2 + Y**2) / (2 * Sigma**2))
        DGaussyy = DGaussxx.T

        # 对每个通道单独应用卷积操作
        DGaussxx = DGaussxx.unsqueeze(0).unsqueeze(0).to(I.device)
        DGaussxy = DGaussxy.unsqueeze(0).unsqueeze(0).to(I.device)
        DGaussyy = DGaussyy.unsqueeze(0).unsqueeze(0).to(I.device)
        
        Dxx = F.conv2d(I, DGaussxx.expand(I.size(1), -1, -1, -1), groups=I.size(1), padding='same').detach()
        Dxy = F.conv2d(I, DGaussxy.expand(I.size(1), -1, -1, -1), groups=I.size(1), padding='same').detach()
        Dyy = F.conv2d(I, DGaussyy.expand(I.size(1), -1, -1, -1), groups=I.size(1), padding='same').detach()

        # 计算特征
        tmp = torch.sqrt((Dxx - Dyy)**2 + 4 * Dxy**2)
        v2x = 2 * Dxy
        v2y = Dyy - Dxx + tmp

        mag = torch.sqrt(v2x**2 + v2y**2)
        i = (mag != 0)
        v2x[i] /= mag[i]
        v2y[i] /= mag[i]

        v1x = -v2y
        v1y = v2x

        mu1 = 0.5 * (Dxx + Dyy + tmp)
        mu2 = 0.5 * (Dxx + Dyy - tmp)

        check = torch.abs(mu1) > torch.abs(mu2)
        Lambda1 = mu1.clone()
        Lambda1[check] = mu2[check]
        Lambda2 = mu2.clone()
        Lambda2[check] = mu1[check]

        Ix = v1x.clone()
        Ix[check] = v2x[check]
        Iy = v1y.clone()
        Iy[check] = v2y[check]

        

        # Compute some similarity measures
        near_zeros = torch.isclose(Lambda2, torch.zeros_like(Lambda1))
        #Lambda1[near_zeros] = 2**(-52)
        Lambda2 = torch.where(near_zeros, torch.tensor(2**(-52), device=Lambda1.device), Lambda1)

        # 计算方向角（使用特征向量Ix, Iy）
        anglesl = torch.atan2(Ix, Iy)
        # Ix_gradient = np.gradient(anglesl, axis=1)  
        # Iy_gradient = np.gradient(anglesl, axis=0) 
        # gradient_indicator = np.sqrt(Ix_gradient**2 + Iy_gradient**2)
        Ix_gradient = anglesl[:, :, :, 1:] - anglesl[:, :, :, :-1]
        Iy_gradient = anglesl[:, :, 1:, :] - anglesl[:, :, :-1, :]
        Ix_gradient = torch.cat((Ix_gradient, torch.zeros(Ix_gradient.size(0), Ix_gradient.size(1), Ix_gradient.size(2), 1, device=Ix_gradient.device)), dim=3)
        Iy_gradient = torch.cat((Iy_gradient, torch.zeros(Iy_gradient.size(0), Iy_gradient.size(1), 1, Iy_gradient.size(3), device=Iy_gradient.device)), dim=2)
        #gradient_indicator = torch.sqrt(Ix_gradient**2 + Iy_gradient**2)
        gradient_indicator = torch.sqrt(Ix_gradient**2 + Iy_gradient**2)
        #print(f"GRAD: {gradient_indicator.mean()}")
        Rb = (Lambda1 / (Lambda2 + 1e-10))**2
        Rb.to(I.device)
        S2 = Lambda1**2 + Lambda2**2
        S2.to(I.device)
        #beta = 2*0.5**2
        #c = 2*15**2
        # Compute the output image
        # a1=torch.exp(-Rb/(2*0.5**2))
        # a2=(torch.ones(I.shape, device=Lambda1.device)-torch.exp(-S2/(2*0.5**2)))
        # a3=(torch.exp(-gradient_indicator/(2*0.5**2)))
        # a4=(torch.ones(I.shape, device=Lambda1.device)-torch.exp(-gradient_indicator/(2*0.5**2)))
        # print(f"a1: {a1.mean()}")
        # print(f"a2: {a2.mean()}")
        # print(f"a3: {a3.mean()}")
        # print(f"a4: {a4.mean()}")
        #change
        #Ifiltered = torch.exp(-Rb/(2*0.5**2))*(torch.ones(I.shape, device=Lambda1.device)-torch.exp(-S2/(2*0.5**2)))*(torch.exp(-gradient_indicator/(2*0.5**2)))
        #Ifiltered = torch.exp(-Rb/(2*0.5**2))*(torch.ones(I.shape, device=Lambda1.device)-torch.exp(-S2/(2*0.5**2)))
        #Ifiltered = (torch.ones(Lambda1.shape, device=Lambda1.device)-torch.exp(-S2/(2*0.5**2)))*(torch.exp(-gradient_indicator/(2*0.5**2)))
        #Ifiltered = torch.exp(-Rb/(2*0.5**2))
        #Ifiltered = (torch.ones(Lambda1.shape, device=Lambda1.device)-torch.exp(-S2/(2*0.5**2)))
        Ifiltered = (torch.exp(-gradient_indicator/(2*0.5**2)))
        #Ifiltered = torch.exp(-Rb/(2*0.5**2))*(torch.exp(-gradient_indicator/(2*0.5**2)))
        #print(f"IM: {Ifiltered.mean()}")

        #gradient_indicator+=Ifiltered
        # Rb = (Lambda2/Lambda1)**2
        # Rb.to(Lambda1.device)
        # S2 = Lambda1**2 + Lambda2**2
        # S2.to(Lambda1.device)
        #beta = 2*0.5**2
        #c = 2*15**2
        # # Compute the output image
        # gradient_indicator = torch.exp(-Rb/(2*0.3**2))*(torch.ones(Lambda1.shape, device=Lambda1.device)-torch.exp(-S2/(2*20**2)))
        # #      # 计算方向角（使用特征向量Ix, Iy）
        # anglesl = torch.atan2(Ix, Iy)
        # #print(anglesl.size())
        # #Ix_gradient = torch.gradient(anglesl, dim=1)  # 对列进行梯度计算
        # #Iy_gradient = torch.gradient(anglesl, dim=0) # 对行进行梯度计算
        # Ix_gradient = anglesl[:, :, :, 1:] - anglesl[:, :, :, :-1]
        # Iy_gradient = anglesl[:, :, 1:, :] - anglesl[:, :, :-1, :]
        # Ix_gradient = torch.cat((Ix_gradient, torch.zeros(Ix_gradient.size(0), Ix_gradient.size(1), Ix_gradient.size(2), 1, device=Ix_gradient.device)), dim=3)
        # Iy_gradient = torch.cat((Iy_gradient, torch.zeros(Iy_gradient.size(0), Iy_gradient.size(1), 1, Iy_gradient.size(3), device=Iy_gradient.device)), dim=2)

        # Ifiltered = torch.sqrt(Ix_gradient**2 + Iy_gradient**2)
        # gradient_min = torch.min(gradient_indicator)
        # gradient_max = torch.max(gradient_indicator)

        # # 归一化 gradient
        # normalized_gradient = (gradient_indicator - gradient_min) / (gradient_max - gradient_min)

        # # 处理可能出现的 NaN 值，确保 Ifiltered 的相应位置不受影响
        # # 将 NaN 值替换为 0
        # Ifiltered = torch.nan_to_num(Ifiltered, nan=0.0)

        # 结合两者，选择相乘或加和
        #Rb = Ifiltered * (1 + normalized_gradient) 

        return Ifiltered


    def forward(self, input: torch.Tensor):
        # Predict offset map between [-1, 1]
        # offset = self.offset_conv(input)
        offset = self.offset_conv(input)
        # print(offset)
        offset= self.HessianAndEig2D(offset)
        
        # offset = self.bn(offset)
        offset = self.gn_offset(offset)

        offset = self.tanh(offset)
        #print(offset)
        # Run deformative conv
        y_coordinate_map, x_coordinate_map = get_coordinate_map_2D(
            offset=offset,
            morph=self.morph,
            extend_scope=self.extend_scope,
            device=self.device,
        )
        deformed_feature = get_interpolated_feature(
            input,
            y_coordinate_map,
            x_coordinate_map,
        )

        if self.morph == 0:
            output = self.dsc_conv_x(deformed_feature)
        elif self.morph == 1:
            output = self.dsc_conv_y(deformed_feature)

        # Groupnorm & ReLU
        output = self.gn(output)
        output = self.relu(output)

        return output


def get_coordinate_map_2D(
    offset: torch.Tensor,
    morph: int,
    extend_scope: float = 1.0,
    device:  str="cuda:3"
):
    """Computing 2D coordinate map of DSCNet based on: TODO

    Args:
        offset: offset predict by network with shape [B, 2*K, W, H]. Here K refers to kernel size.
        morph: the morphology of the convolution kernel is mainly divided into two types along the x-axis (0) and the y-axis (1) (see the paper for details).
        extend_scope: the range to expand. Defaults to 1 for this method.
        device: location of data. Defaults to 'cuda'.

    Return:
        y_coordinate_map: coordinate map along y-axis with shape [B, K_H * H, K_W * W]
        x_coordinate_map: coordinate map along x-axis with shape [B, K_H * H, K_W * W]
    """

    if morph not in (0, 1):
        raise ValueError("morph should be 0 or 1.")

    batch_size, _, width, height = offset.shape
    kernel_size = offset.shape[1] // 2
    center = kernel_size // 2
    device = torch.device(device)

    y_offset_, x_offset_ = torch.split(offset, kernel_size, dim=1)

    y_center_ = torch.arange(0, width, dtype=torch.float32, device=device)
    y_center_ = einops.repeat(y_center_, "w -> k w h", k=kernel_size, h=height)

    x_center_ = torch.arange(0, height, dtype=torch.float32, device=device)
    x_center_ = einops.repeat(x_center_, "h -> k w h", k=kernel_size, w=width)

    if morph == 0:
        """
        Initialize the kernel and flatten the kernel
            y: only need 0
            x: -num_points//2 ~ num_points//2 (Determined by the kernel size)
        """
        y_spread_ = torch.zeros([kernel_size], device=device)
        x_spread_ = torch.linspace(-center, center, kernel_size, device=device)

        y_grid_ = einops.repeat(y_spread_, "k -> k w h", w=width, h=height)
        x_grid_ = einops.repeat(x_spread_, "k -> k w h", w=width, h=height)

        y_new_ = y_center_ + y_grid_
        x_new_ = x_center_ + x_grid_

        y_new_ = einops.repeat(y_new_, "k w h -> b k w h", b=batch_size)
        x_new_ = einops.repeat(x_new_, "k w h -> b k w h", b=batch_size)

        y_offset_ = einops.rearrange(y_offset_, "b k w h -> k b w h")
        y_offset_new_ = y_offset_.detach().clone()

        # The center position remains unchanged and the rest of the positions begin to swing
        # This part is quite simple. The main idea is that "offset is an iterative process"

        y_offset_new_[center] = 0

        for index in range(1, center + 1):
            y_offset_new_[center + index] = (
                y_offset_new_[center + index - 1] + y_offset_[center + index]
            )
            y_offset_new_[center - index] = (
                y_offset_new_[center - index + 1] + y_offset_[center - index]
            )

        y_offset_new_ = einops.rearrange(y_offset_new_, "k b w h -> b k w h")

        y_new_ = y_new_.add(y_offset_new_.mul(extend_scope))

        y_coordinate_map = einops.rearrange(y_new_, "b k w h -> b (w k) h")
        x_coordinate_map = einops.rearrange(x_new_, "b k w h -> b (w k) h")

    elif morph == 1:
        """
        Initialize the kernel and flatten the kernel
            y: -num_points//2 ~ num_points//2 (Determined by the kernel size)
            x: only need 0
        """
        y_spread_ = torch.linspace(-center, center, kernel_size, device=device)
        x_spread_ = torch.zeros([kernel_size], device=device)

        y_grid_ = einops.repeat(y_spread_, "k -> k w h", w=width, h=height)
        x_grid_ = einops.repeat(x_spread_, "k -> k w h", w=width, h=height)

        y_new_ = y_center_ + y_grid_
        x_new_ = x_center_ + x_grid_

        y_new_ = einops.repeat(y_new_, "k w h -> b k w h", b=batch_size)
        x_new_ = einops.repeat(x_new_, "k w h -> b k w h", b=batch_size)

        x_offset_ = einops.rearrange(x_offset_, "b k w h -> k b w h")
        x_offset_new_ = x_offset_.detach().clone()

        # The center position remains unchanged and the rest of the positions begin to swing
        # This part is quite simple. The main idea is that "offset is an iterative process"

        x_offset_new_[center] = 0

        for index in range(1, center + 1):
            x_offset_new_[center + index] = (
                x_offset_new_[center + index - 1] + x_offset_[center + index]
            )
            x_offset_new_[center - index] = (
                x_offset_new_[center - index + 1] + x_offset_[center - index]
            )

        x_offset_new_ = einops.rearrange(x_offset_new_, "k b w h -> b k w h")

        x_new_ = x_new_.add(x_offset_new_.mul(extend_scope))

        y_coordinate_map = einops.rearrange(y_new_, "b k w h -> b w (h k)")
        x_coordinate_map = einops.rearrange(x_new_, "b k w h -> b w (h k)")

    return y_coordinate_map, x_coordinate_map


def get_interpolated_feature(
    input_feature: torch.Tensor,
    y_coordinate_map: torch.Tensor,
    x_coordinate_map: torch.Tensor,
    interpolate_mode: str = "bilinear",
):
    """From coordinate map interpolate feature of DSCNet based on: TODO

    Args:
        input_feature: feature that to be interpolated with shape [B, C, H, W]
        y_coordinate_map: coordinate map along y-axis with shape [B, K_H * H, K_W * W]
        x_coordinate_map: coordinate map along x-axis with shape [B, K_H * H, K_W * W]
        interpolate_mode: the arg 'mode' of nn.functional.grid_sample, can be 'bilinear' or 'bicubic' . Defaults to 'bilinear'.

    Return:
        interpolated_feature: interpolated feature with shape [B, C, K_H * H, K_W * W]
    """

    if interpolate_mode not in ("bilinear", "bicubic"):
        raise ValueError("interpolate_mode should be 'bilinear' or 'bicubic'.")

    y_max = input_feature.shape[-2] - 1
    x_max = input_feature.shape[-1] - 1

    y_coordinate_map_ = _coordinate_map_scaling(y_coordinate_map, origin=[0, y_max])
    x_coordinate_map_ = _coordinate_map_scaling(x_coordinate_map, origin=[0, x_max])

    y_coordinate_map_ = torch.unsqueeze(y_coordinate_map_, dim=-1)
    x_coordinate_map_ = torch.unsqueeze(x_coordinate_map_, dim=-1)

    # Note here grid with shape [B, H, W, 2]
    # Where [:, :, :, 2] refers to [x ,y]
    grid = torch.cat([x_coordinate_map_, y_coordinate_map_], dim=-1)

    interpolated_feature = nn.functional.grid_sample(
        input=input_feature,
        grid=grid,
        mode=interpolate_mode,
        padding_mode="zeros",
        align_corners=True,
    )

    return interpolated_feature


def _coordinate_map_scaling(
    coordinate_map: torch.Tensor,
    origin: list,
    target: list = [-1, 1],
):
    """Map the value of coordinate_map from origin=[min, max] to target=[a,b] for DSCNet based on: TODO

    Args:
        coordinate_map: the coordinate map to be scaled
        origin: original value range of coordinate map, e.g. [coordinate_map.min(), coordinate_map.max()]
        target: target value range of coordinate map,Defaults to [-1, 1]

    Return:
        coordinate_map_scaled: the coordinate map after scaling
    """
    min, max = origin
    a, b = target

    coordinate_map_scaled = torch.clamp(coordinate_map, min, max)

    scale_factor = (b - a) / (max - min)
    coordinate_map_scaled = a + scale_factor * (coordinate_map_scaled - min)

    return coordinate_map_scaled
