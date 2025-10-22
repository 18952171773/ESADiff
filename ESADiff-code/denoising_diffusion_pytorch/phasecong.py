import math
import time
import numpy as np
import cv2 # Faster Fourier transforms than NumPy and Scikit-Image
import matplotlib.pyplot as plt

def phase(InputImage, NumberScales, NumberAngles):
    '''
    计算图像相位
    '''    
    minWaveLength = 3
    mult = 2.1
    sigmaOnf = 0.55
    k = 2.0
    cutOff = 0.5
    g = 10
    noiseMethod = -1
    epsilon = .0001 # Used to prevent division by zero.
    f_cv = cv2.dft(np.float32(InputImage),flags=cv2.DFT_COMPLEX_OUTPUT)
    #------------------------------
    nrows, ncols = InputImage.shape
    zero = np.zeros((nrows,ncols))
    EO = np.zeros((nrows,ncols,NumberScales,NumberAngles),dtype=complex)
    PC = np.zeros((nrows,ncols,NumberAngles))
    covx2 = np.zeros((nrows,ncols))
    covy2 = np.zeros((nrows,ncols))
    covxy = np.zeros((nrows,ncols))
    EnergyV = np.zeros((nrows,ncols,3))
    pcSum = np.zeros((nrows,ncols))
    # Matrix of radii 矩阵的半径
    cy = math.floor(nrows/2)
    cx = math.floor(ncols/2)
    y, x = np.mgrid[0:nrows, 0:ncols]
    y = (y-cy)/nrows
    x = (x-cx)/ncols
    radius = np.sqrt(x**2 + y**2)
    radius[cy, cx] = 1
    theta = np.arctan2(-y, x)
    sintheta = np.sin(theta)
    costheta = np.cos(theta)

    # 初始化一组环形带通滤波器
    # 这里我使用了我用来生成的代码中的比例选择方法
    # 刺激，用于我最新的实验（空间特征缩放）。
    #NumberScales = 3 # should be odd
    annularBandpassFilters = np.empty((nrows,ncols,NumberScales))
    #p = np.arange(NumberScales) - math.floor(NumberScales/2)
    #fSetCpo = CriticalBandCyclesPerObject*mult**p
    #fSetCpi = fSetCpo * ObjectsPerImage

    # 过滤器方向的数量
    #NumberAngles = 6
    """ 滤波器方向的角度间隔和用于在频率平面上构建滤波器的角度高斯函数的标准偏差之比。
    """
    dThetaOnSigma = 1.3
    filterOrient = np.arange(start=0, stop=math.pi - math.pi / NumberAngles, step = math.pi / NumberAngles)

    # 角度高斯函数的标准偏差，用于在频率面构建滤波器
    thetaSigma = math.pi / NumberAngles / dThetaOnSigma

    BandpassFilters = np.empty((nrows,ncols,NumberScales,NumberAngles))
    evenWavelets = np.empty((nrows,ncols,NumberScales,NumberAngles))
    oddWavelets  = np.empty((nrows,ncols,NumberScales,NumberAngles))

    # 以下是实现对数Gabor传递函数的方法
    # sigmaOnf = 0.74  # approximately 1 octave
    # sigmaOnf = 0.55  # approximately 2 octaves
    filterorder = 15  # filter 'sharpness'
    cutoff = .45
    normradius = radius / (abs(x).max()*2)
    lowpassbutterworth = 1.0 / (1.0 + (normradius / cutoff)**(2*filterorder))
    #
    # Note: lowpassbutterworth目前是以DC为中心。
    #
    #
    #annularBandpassFilters[:,:,i] = logGabor * lowpassbutterworth
    #logGabor = np.empty((nrows,ncols,NumberScales)) --> same as annularBandpassFilters
    for s in np.arange(NumberScales):
        wavelength = minWaveLength*mult**s
        fo = 1.0/wavelength                  # Centre frequency of filter.
        logGabor = np.exp((-(np.log(radius/fo))**2) / (2 * math.log(sigmaOnf)**2))
        annularBandpassFilters[:,:,s] = logGabor*lowpassbutterworth  # Apply low-pass filter
        annularBandpassFilters[cy,cx,s] = 0         
    # 主循环
    for o in np.arange(NumberAngles):
        # 构建角度滤波扩散函数
        angl = o*math.pi/NumberAngles # Filter angle.
        # 对于过滤器矩阵中的每个点，计算其与指定过滤器方向的角度距离。指定的过滤器方向的角度距离。为了克服角度环绕的问题
        # 问题，首先计算正弦差值和余弦差值,然后用atan2函数来确定角距离。
        ds = sintheta * math.cos(angl) - costheta * math.sin(angl)      # Difference in sine.
        dc = costheta * math.cos(angl) + sintheta * math.sin(angl)      # Difference in cosine.
        dtheta = np.abs(np.arctan2(ds,dc))                              # Absolute angular distance.
        # 缩放theta，使余弦扩散函数具有合适的波长，并对pi进行钳制
        dtheta = np.minimum(dtheta*NumberAngles/2, math.pi)
        #spread = np.exp((-dtheta**2) / (2 * thetaSigma**2));  # Calculate the angular
                                                              # filter component.
        # The spread function is cos(dtheta) between -pi and pi.  We add 1,
        #   and then divide by 2 so that the value ranges 0-1
        spread = (np.cos(dtheta)+1)/2

        sumE_ThisOrient   = np.zeros((nrows,ncols))  # Initialize accumulator matrices.
        sumO_ThisOrient   = np.zeros((nrows,ncols))
        sumAn_ThisOrient  = np.zeros((nrows,ncols))
        Energy            = np.zeros((nrows,ncols))

        maxAn = []
        for s in np.arange(NumberScales):
            filter = annularBandpassFilters[:,:,s] * spread # Multiply radial and angular
                                                            # components to get the filter.

            criticalfiltershift = np.fft.ifftshift( filter )
            criticalfiltershift_cv = np.empty((nrows, ncols, 2))
            for ip in range(2):
                criticalfiltershift_cv[:,:,ip] = criticalfiltershift

            # Convolve image with even and odd filters returning the result in EO
            MatrixEO = cv2.idft( criticalfiltershift_cv * f_cv )
            EO[:,:,s,o] = MatrixEO[:,:,1] + 1j*MatrixEO[:,:,0]

            An = cv2.magnitude(MatrixEO[:,:,0], MatrixEO[:,:,1])    # Amplitude of even & odd filter response.

            sumAn_ThisOrient = sumAn_ThisOrient + An             # Sum of amplitude responses.
            sumE_ThisOrient = sumE_ThisOrient + MatrixEO[:,:,1] # Sum of even filter convolution results.
            sumO_ThisOrient = sumO_ThisOrient + MatrixEO[:,:,0] # Sum of odd filter convolution results.

            # 在最小范围内，从存储在sumAn中的滤波器振幅响应的分布中估计噪声特性。
            if s == 0:
            #     if noiseMethod == -1     # Use median to estimate noise statistics
                tau = np.median(sumAn_ThisOrient) / math.sqrt(math.log(4))
            #     elseif noiseMethod == -2 # Use mode to estimate noise statistics
            #         tau = rayleighmode(sumAn_ThisOrient(:));
            #     end
                maxAn = An
            else:
                # Record maximum amplitude of components across scales.  This is needed
                # to determine the frequency spread weighting.
                maxAn = np.maximum(maxAn,An)
            # end
        EnergyV[:,:,0] = EnergyV[:,:,0] + sumE_ThisOrient
        EnergyV[:,:,1] = EnergyV[:,:,1] + math.cos(angl)*sumO_ThisOrient
        EnergyV[:,:,2] = EnergyV[:,:,2] + math.sin(angl)*sumO_ThisOrient

        # Get weighted mean filter response vector, this gives the weighted mean
        # phase angle.
        XEnergy = np.sqrt(sumE_ThisOrient**2 + sumO_ThisOrient**2) + epsilon
        MeanE = sumE_ThisOrient / XEnergy
        MeanO = sumO_ThisOrient / XEnergy

        # Now calculate An(cos(phase_deviation) - | sin(phase_deviation)) | by
        # using dot and cross products between the weighted mean filter response
        # vector and the individual filter response vectors at each scale.  This
        # quantity is phase congruency multiplied by An, which we call energy.

        for s in np.arange(NumberScales):
            # Extract even and odd convolution results.
            E = EO[:,:,s,o].real
            O = EO[:,:,s,o].imag

            Energy = Energy + E*MeanE + O*MeanO - np.abs(E*MeanO - O*MeanE)

        # if noiseMethod >= 0:     % We are using a fixed noise threshold
        #     T = noiseMethod;    % use supplied noiseMethod value as the threshold
        # else:
        # Estimate the effect of noise on the sum of the filter responses as
        # the sum of estimated individual responses (this is a simplistic
        # overestimate). As the estimated noise response at succesive scales
        # is scaled inversely proportional to bandwidth we have a simple
        # geometric sum.
        totalTau = tau * (1 - (1/mult)**NumberScales)/(1-(1/mult))

        # 利用这些参数与tau之间的固定关系，从tau计算出平均值和std dev。
        EstNoiseEnergyMean = totalTau*math.sqrt(math.pi/2)        # Expected mean and std
        EstNoiseEnergySigma = totalTau*math.sqrt((4-math.pi)/2)   # values of noise energy

        T =  EstNoiseEnergyMean + k*EstNoiseEnergySigma # Noise threshold
        # end

        # 应用噪声阈值，这实际上是通过小波去噪。软阈值处理。
        Energy = np.maximum(Energy - T, 0)

        # 形成加权，去除那些特别窄的频率分布。计算频率的分数 "宽度"，方法是取滤波器响应振幅的总和并除以图像上每个点的最大振幅。  如果只有一个非零分量，宽度的值为0，如果所有分量都相等，宽度为1。

        width = (sumAn_ThisOrient/(maxAn + epsilon) - 1) / (NumberScales-1)

        # 现在计算这个方向的sigmoidal加权函数
        weight = 1.0 / (1 + np.exp( (cutOff - width)*g))

        # 对能量进行加权，然后计算出相位一致性
        PC[:,:,o] = weight*Energy/sumAn_ThisOrient   # 该方向的相位一致性

        pcSum = pcSum + PC[:,:,o]

        # 为每一个点建立协方差数据
        covx = PC[:,:,o]*math.cos(angl)
        covy = PC[:,:,o]*math.sin(angl)
        covx2 = covx2 + covx**2
        covy2 = covy2 + covy**2
        covxy = covxy + covx*covy
        # above everyting within orientaiton loop
    # ------------------------------------------------------------------------
    # 边缘和角点的计算
    # 以下是优化后的代码，用于计算相位一致性协方差数据的主向量，并计算最小和最大矩--这些对应于奇异值。

    # 首先通过方向数/2将协方差值归一化
    covx2 = covx2/(NumberAngles/2)
    covy2 = covy2/(NumberAngles/2)
    covxy = 4*covxy/NumberAngles   # This gives us 2*covxy/(norient/2)
    denom = np.sqrt(covxy**2 + (covx2-covy2)**2)+epsilon
    M = (covy2+covx2 + denom)/2          # 最大矩
    m = (covy2+covx2 - denom)/2          # 最小矩

    # 方向和特征相位/类型计算
    ORM = np.arctan2(EnergyV[:,:,2], EnergyV[:,:,1])
    ORM[ORM<0] = ORM[ORM<0]+math.pi       # Wrap angles -pi..0 to 0..pi
    ORM = np.round(ORM*180/math.pi)        # 方向，在0到180度之间

    OddV = np.sqrt(EnergyV[:,:,1]**2 + EnergyV[:,:,2]**2)
    featType = np.arctan2(EnergyV[:,:,0], OddV)  # 特征相位  pi/2 <-> white line,0 <-> step, -pi/2 <-> black line
    #

    # return M, m, ORM, EO, T, featType, annularBandpassFilters, lowpassbutterworth
    return M

if __name__ == "__main__":
    starttime=time.time()    #开始时间
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))    #输出程序运行时间    
    path='/mnt/no1/miaochenlong/DiffusionEdge/data-image/test/02_AJZTHDP1-1.png'
    img=cv2.imread(path,0)
    M=phase(img,4,6)
    out=M>0.3
    print(M)
    filename='out3.png'
    plt.imsave(filename,out,cmap='gray')
    
