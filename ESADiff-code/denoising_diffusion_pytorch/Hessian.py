import numpy as np
import torch
import torch.nn.functional as F

def Hessian2D(I, Sigma=1):
    # Make kernel coordinates
    X, Y = torch.meshgrid(torch.arange(-round(3*Sigma), round(3*Sigma)+1),
                          torch.arange(-round(3*Sigma), round(3*Sigma)+1))

    # Build the gaussian 2nd derivatives filters
    DGaussxx = 1/(2*np.pi*Sigma**4) * (X**2/Sigma**2 - 1) * torch.exp(-(X**2 + Y**2)/(2*Sigma**2))
    DGaussxy = 1/(2*np.pi*Sigma**6) * (X * Y) * torch.exp(-(X**2 + Y**2)/(2*Sigma**2))
    DGaussyy = DGaussxx.T

    Dxx = F.conv2d(I.unsqueeze(0).unsqueeze(0), DGaussxx.unsqueeze(0).unsqueeze(0), padding='same').squeeze(0).squeeze(0)
    Dxy = F.conv2d(I.unsqueeze(0).unsqueeze(0), DGaussxy.unsqueeze(0).unsqueeze(0), padding='same').squeeze(0).squeeze(0)
    Dyy = F.conv2d(I.unsqueeze(0).unsqueeze(0), DGaussyy.unsqueeze(0).unsqueeze(0), padding='same').squeeze(0).squeeze(0)

    return Dxx, Dxy, Dyy

def eig2image(Dxx, Dxy, Dyy):
    # Compute the eigenvectors of J, v1 and v2
    tmp = torch.sqrt((Dxx - Dyy)**2 + 4*Dxy**2)
    v2x = 2*Dxy
    v2y = Dyy - Dxx + tmp

    # Normalize
    mag = torch.sqrt(v2x**2 + v2y**2)
    i = (mag != 0)
    v2x[i] /= mag[i]
    v2y[i] /= mag[i]

    # The eigenvectors are orthogonal
    v1x = -v2y
    v1y = v2x

    # Compute the eigenvalues
    mu1 = 0.5*(Dxx + Dyy + tmp)
    mu2 = 0.5*(Dxx + Dyy - tmp)

    # Sort eigenvalues by absolute value abs(Lambda1) < abs(Lambda2)
    check = torch.abs(mu1) > torch.abs(mu2)
    Lambda1 = mu1.clone()
    Lambda1[check] = mu2[check]
    Lambda2 = mu2.clone()
    Lambda2[check] = mu1[check]

    Ix = v1x.clone()
    Ix[check] = v2x[check]
    Iy = v1y.clone()
    Iy[check] = v2y[check]

    return Lambda1, Lambda2, Ix, Iy


# image_path = '/mnt/no1/miaochenlong/DiffusionEdge/data-image/test/02_AJZTHDP7-2.png'

# image = cv2.imread(image_path)
# image = rgb2gray(image)

# # 对图像进行高斯模糊
# image_blurred = gaussian(image, sigma=2)

# # 计算 Hessian 矩阵的三个分量
# Dxx, Dxy, Dyy = Hessian2D(image_blurred)
# # 解析原始文件名
# file_name = os.path.basename(image_path)
# file_name_without_extension = os.path.splitext(file_name)[0]
# file_name2 = file_name_without_extension+"-0.67.png"
# file_name3 = file_name_without_extension+"-3.png"
# Lambda1, Lambda2, Ix, Iy=eig2image(Dxx, Dxy, Dyy)

# Lambda2 = cv2.normalize(Lambda2, None, 0, 255, cv2.NORM_MINMAX)
# Lambda2 = np.uint8(Lambda2)

# Lambda2[Lambda2 < 0.67*255] = 0
# Lambda2[Lambda2 >= 0.67*255] = 255
# Lambda2 = cv2.bitwise_not(Lambda2)
# plt.imsave(file_name2,Lambda2,cmap='gray')


