import os
import numpy as np
import cv2

try:
    import skimage
except:
    pass


def MSE(y_true, y_pred):
    delta = y_true.astype(np.float) - y_pred.astype(np.float)
    delta = np.multiply(delta, delta)
    delta = delta.sum()
    nData = np.prod(y_true.shape)
    return delta.item() / nData

# matlab version: peaksnr = 10*log10(peakval.^2/err)
def PSNR(y_true,y_pred):
    err = MSE(y_true,y_pred)
    peakval = 255
    x = 10* np.log10(peakval*peakval / err)
    return x
 
 
def SSIM(y_true, y_pred):
    u_true = np.mean(y_true)
    u_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    std_true = np.sqrt(var_true)
    std_pred = np.sqrt(var_pred)
    c1 = np.square(0.01*7)
    c2 = np.square(0.03*7)
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    x = ssim / denom
    
    return x



def getGaborFilters(minKernelSize=7, nKernelSize=6, nTheta=4):
    filters = []
    ksize = list(np.linspace(minKernelSize, minKernelSize+(nKernelSize-1)*2, nKernelSize, dtype=int))
    lamda = np.pi/2.0 
    for theta in np.arange(0, np.pi, np.pi / nTheta): 
        for K in range(nKernelSize): 
            kern = cv2.getGaborKernel((ksize[K], ksize[K]), 1.0, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
            kern /= 1.5*kern.sum()
            filters.append(kern)
    return filters 
    
def filterImgWithGabor(img, gaborFilters=None):
    if gaborFilters is None:
        gaborFilters = getGaborFilters()
        
    accum = np.zeros_like(img)
    for kern in gaborFilters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum

def filterImgWithSobel(img):
    sobelx = cv2.Sobel(img, cv2.CV_64F,1,0,ksize=3) #横向边缘提取
    sobely = cv2.Sobel(img, cv2.CV_64F,0,1,ksize=3) #竖向边缘提取
    sobelx = cv2.convertScaleAbs(sobelx) # 负值取正，图像展示只能有正值
    sobely = cv2.convertScaleAbs(sobely)
    sobelxy = cv2.addWeighted(sobelx,0.5,sobely,0.5,0) #图像融合
    return sobelxy

def filterImgWithScharr(img):
    scharrx = cv2.Scharr(img,cv2.CV_64F,1,0)
    scharry = cv2.Scharr(img,cv2.CV_64F,0,1)
    scharrx = cv2.convertScaleAbs(scharrx) # 负值取正
    scharry = cv2.convertScaleAbs(scharry)
    scharrxy = cv2.addWeighted(scharrx,0.5,scharry,0.5,0) #图像融合
    return scharrxy

def filterImgWithLaplacian(img):
    img1 = cv2.Laplacian(img, cv2.CV_64F)
    img1 = cv2.convertScaleAbs(img1)
    return img1
        
        
