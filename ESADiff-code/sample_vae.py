import torch
from denoising_diffusion_pytorch.encoder_decoder import AutoencoderKL
import torchvision as tv
from denoising_diffusion_pytorch.encoder_decoder import AutoencoderKL
from denoising_diffusion_pytorch.data import *
def load_model(cfg):
    """加载训练好的AutoencoderKL模型"""
    model_cfg = cfg['model']
    model = AutoencoderKL(
        ddconfig=model_cfg['ddconfig'],
        lossconfig=model_cfg['lossconfig'],
        embed_dim=model_cfg['embed_dim'],
        ckpt_path=None,  # 不用预训练权重
    )
    # 加载已训练权重
    checkpoint_path = "/mnt/no1/miaochenlong/DiffusionEdge/results/image-crisp-more/model-15.pt"
    checkpoint = torch.load(checkpoint_path, map_location='cuda:0')
    model.load_state_dict(checkpoint['model'])
    model.eval()  # 设置为推理模式
    return model

def encode_images(model, images):
    """将输入图片编码到潜在空间"""
    images = images.cuda()  # 确保图片在GPU上
    with torch.no_grad():
        latent_representation = model.encoder(images)
    return latent_representation

# 示例主函数
if __name__ == "__main__":
    import yaml
    from torchvision.transforms import Compose, ToTensor, Resize
    import torchvision.utils as vutils


    # 加载配置文件
    config_path = "/mnt/no1/miaochenlong/DiffusionEdge/configs/first_stage_d4.yaml"
    with open(config_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # 加载模型
    model = load_model(cfg).cuda()
    # 准备输入图片
    image_path = "/mnt/no1/miaochenlong/DiffusionEdge/result/contour/02_AJZTHDP1-1.png"
    image = tv.io.read_image(image_path).float() / 255.0  # 读取图片并归一化
    image = tv.transforms.Resize((256, 256))(image)  # 调整尺寸
    image = image.unsqueeze(0).cuda()  # 添加 batch 维度并移动到 GPU

    # 获取潜在空间表示
    # 假设 latent 是你的潜在表示，形状为 [batch_size, channels, height, width]
    latent = encode_images(model, image)  # 获取潜在表示

    # 选取 batch 的第一张图片，取某一通道（如第一个通道）
    latent_image = latent[0, 0, :, :].detach().cpu()  # [height, width]

    # 将潜在表示归一化到 0-1
    latent_min = latent_image.min()
    latent_max = latent_image.max()
    latent_image = (latent_image - latent_min) / (latent_max - latent_min)

    # 保存图片到文件
    vutils.save_image(latent_image, "latent_representation.png")
