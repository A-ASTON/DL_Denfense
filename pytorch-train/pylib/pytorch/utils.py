import torch
import PIL.Image as Image
import numpy as np
import copy
import cv2
from torchvision.transforms import transforms as T

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        assert(tensor.ndim == 3)
        for k in range(tensor.shape[0]):
            m = self.mean[k]
            s = self.std[k]
            tensor[k].mul_(s).add_(m)
        return tensor
        
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        assert(tensor.ndim == 3)
        for k in range(tensor.shape[0]):
            m = self.mean[k]
            s = self.std[k]
            tensor[k].sub_(m).div_(s)
        return tensor
        
imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)
cifar10_mean = (0.4914, 0.4822, 0.4465), 
cifar10_std =  (0.2023, 0.1994, 0.2010)

imagenet_unnormalizer = UnNormalize(imagenet_mean, imagenet_std)
imagenet_normalizer   = Normalize(imagenet_mean, imagenet_std)

        
def findModule(model, moduleName):
    for name, module in model.named_modules():
        if name == moduleName:
            return module
    return None

## PIL <--> Numpy
def PILImgToNumpy(img_pil):
    array = np.array(img_pil)	
    return array


# tested
def PILImgToOpencv(img_pil):
    img = cv2.cvtColor(np.asarray(img_pil),cv2.COLOR_RGB2BGR)
    return img

def numpyToPILImg(np_array):
    tmparray = copy.deepcopy(np_array)
    if np_array.max() <= 1:
        tmparray *= 255
    img_pil = Image.fromarray(tmparray.astype('uint8'))
    img_pil.convert('RGB')       # this is problematic
    return img_pil

## Numpy <--> Tensor
def numpyToTensor(np_array):
    tensor_data = torch.from_numpy(np_array)   
    return tensor_data
# Dr.He:
# def tensorToNumpy(tensor_data):
#     return tensor_data.numpy()

# My
def tensorToNumpy(tensor_data):
    return tensor_data.cpu().numpy()

## Tensor <--> PIL
def tensorToPIlImg(tensor_data):
    if tensor_data.max() <= 1:
        tdata = copy.deepcopy(tensor_data) * 255
        img_pil = T.ToPILImage()(tdata)
    else:
        img_pil = T.ToPILImage()(tensor_data)
    return img_pil

# this does not normalize data, unlike transform.ToTensor()
def PILImgToTensor(img_pil):
    np_data = np.array(img_pil)
    if np_data.ndim == 3:
        tensor_data = torch.from_numpy(np.transpose(np_data, (2, 0, 1)))    
    else:
        tensor_data = torch.from_numpy(np_data)
    return tensor_data

def opencvToTensor(img_cv):
    # assuming HWC and BRG, one channel data for gray image
    if img_cv.ndim == 3:
        img_cv2 = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        img_cv2 = np.transpose(img_cv2, (2, 0, 1))
        tensor_data = torch.from_numpy(img_cv2)
    else:
        tensor_data = torch.from_numpy(img_cv)

    return tensor_data

# torch tensor: CHW, opencv: HWC, [0, 255]
# make sure input tensor is within [0, 1]
def tensorToOpencv(intensor_data):
    # CHW    
    tensor_data = copy.deepcopy(intensor_data)    
    np_data = tensor_data.numpy() 
    
    np_data = np_data * 255
    np_data = np_data.astype(np.int8).astype(np.uint8)
    assert(np_data.ndim == 2 or np_data.ndim == 3)

    if np_data.ndim == 2:
        return np_data

    if np_data.ndim == 3:
        img_cv = np.transpose(np_data, (1,2,0))   
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        return img_cv

    return np_data

def addGaussianNoise(img_pil, std=1):
    img = PILImgToNumpy(img_pil)    
    noise = np.random.normal(size=img.shape, loc=0, scale=std)
    img = img + noise
    
    # clamp 
    
    img_pil2 = numpyToPILImg(img)
    return img_pil2
    
# put before totensor()    
class AddGaussianNoise(object):
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def __call__(self, img_pil):
        img = PILImgToNumpy(img_pil)
        noise = np.random.normal(size=img.shape, loc=self.mean, scale=self.std)
        img2 = img + noise

        xx = np.where(img2 < 0)
        img2[xx] = img[xx]

        xx = np.where(img2 > 255)
        img2[xx] = img[xx]

        img_pil2 = numpyToPILImg(img2)
        return img_pil2        
  
        