from torchvision.ops import DeformConv2d
from torch import nn
import torch

class DCNConv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True,device:  str="cuda:0"):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv1 = nn.Conv2d(c1, c2, 3, 1, 1, groups=g, bias=False)
        deformable_groups = 1
        offset_channels = 18
        self.conv2_offset = nn.Conv2d(c2, deformable_groups * offset_channels, kernel_size=3, padding=1)
        self.conv2 = DeformConv2d(c2, c2, kernel_size=3, padding=1, bias=False)
        
        # self.conv2 = DeformableConv2d(c2, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn1 = nn.BatchNorm2d(c2)
        self.act1 = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.bn2 = nn.BatchNorm2d(c2)
        self.act2 = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.device = torch.device(device)
        self.to(device)
 
    def forward(self, x):
        # print(x.shape)
        # print('-'*50)
        x = self.act1(self.bn1(self.conv1(x)))
        # print(x.shape)
        offset = self.conv2_offset(x)
        x = self.act2(self.bn2(self.conv2(x,offset)))
        # print('-'*50)
        # print(x.shape)
        return x
def test_dcn_conv():
    # 创建一个 DCNConv 实例
    dcn = DCNConv(c1=64, c2=128, k=3, s=2, g=1, act=True)
    
    # 打印模型结构
    print(dcn)
    
    # 创建一个随机输入张量 (batch_size=2, channels=64, height=32, width=32)
    x = torch.randn(2, 64, 32, 32)
    print("Input shape:", x.shape)
    
    # 前向传播
    out = dcn(x)
    print("Output shape:", out.shape)
    
    # 检查输出尺寸是否符合预期
    # 输入32x32，经过stride=2的卷积，输出应为16x16
    assert out.shape == (2, 128, 16, 16), f"Expected (2, 128, 16, 16), got {out.shape}"
    print("Test passed!")
    
    # 测试分组卷积
    dcn_group = DCNConv(c1=64, c2=128, k=3, s=2, g=2, act=True)
    out_group = dcn_group(x)
    assert out_group.shape == (2, 128, 16, 16), f"Expected (2, 128, 16, 16), got {out_group.shape}"
    print("Group convolution test passed!")

if __name__ == "__main__":
    test_dcn_conv()