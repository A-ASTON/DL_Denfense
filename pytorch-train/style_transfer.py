import torch
import net
import torch.nn as nn
from function import adaptive_instance_normalization, coral
from PIL import Image
import numpy
from torchvision.utils import save_image
import cv2

import os
from torch.utils.data import DataLoader, Dataset
from pylib.pytorch.data import FolderDataset as FolderDataset
from torchvision.transforms import transforms as T
from pylib.pytorch import utils as thutils
import pylib.pytorch.utils as tchutils

basic_transform = T.Compose([
    T.ToTensor(),
    T.Resize(224),
    thutils.imagenet_normalizer
])

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1).cpu()
        out = self.sigmoid(self.conv2d(out))
        return out


def inverse_normalize(x) :
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    y = x.clone().to(x.device)
    y[:,0,:,:] = y[:,0,:,:] * std[0] + mean[0]
    y[:,1,:,:] = y[:,1,:,:] * std[1] + mean[1]
    y[:,2,:,:] = y[:,2,:,:] * std[2] + mean[2]
    return y

def normalize_(x) :
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    y = x.clone().to(x.device)
    y[:,0,:,:] = (y[:,0,:,:] - mean[0]) / std[0]
    y[:,1,:,:] = (y[:,1,:,:] - mean[1]) / std[1]
    y[:,2,:,:] = (y[:,2,:,:] - mean[2]) / std[2]    
    return y

def vggmodel(vgg, input, threshold):
    vgg1 = nn.Sequential(*list(vgg.children())[:18]).cuda()
    vgg2 = nn.Sequential(*list(vgg.children())[18:31]).cuda()    

    input = vgg1(input)
    zero1 = torch.zeros_like(input[0]).cuda()
    input = torch.where(input[0] < threshold, zero1, input)
    
    input = vgg2(input)
    zero2 = torch.zeros_like(input[0]).cuda()
    input = torch.where(input[0] < 1.6, zero2, input)
    return input

# attention = SpatialAttentionModule()


def style_transfer(vgg, decoder, content, style, threshold, alpha=0.6,interpolation_weights=None):
# def style_transfer(vgg, decoder, content, style, alpha=0.6,interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    useCuda = True
    device = torch.device("cuda" if (useCuda and torch.cuda.is_available()) else "cpu")
    
    # content_f = vggmodel(vgg, content, threshold)
    content_f = vgg(content)
    style_f = vgg(style)
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)
# def pic_transfer(imgTensor, styleTensor):

def pic_transfer(imgTensor,styleTensor,vgg,decoder, threshold):
# def pic_transfer(imgTensor,styleTensor):
    device = torch.device("cuda")   

    decoder = net.decoder
    vgg = net.vgg
    vgg = nn.Sequential(*list(vgg.children())[:31])
    
    decoder.eval()
    vgg.eval()
    
    vgg.to(device)
    decoder.to(device)

    imgTensor = imgTensor.to(device)
    styleTensor = inverse_normalize(styleTensor)
    styleTensor = styleTensor.to(device).requires_grad_()
    # styleTensor = coral(styleTensor, imgTensor)
    # output = style_transfer(vgg, decoder, imgTensor, styleTensor, 0.6)
    output = style_transfer(vgg, decoder, imgTensor, styleTensor, threshold, 0.6)

    # output_name = r'D:\aaaaaaaaaaaaaaaaa\project\pytorch-train\output.jpg'
    # save_image(output, str(output_name))
    # img_name = r'D:\aaaaaaaaaaaaaaaaa\project\pytorch-train\input.jpg'
    # save_image(imgTensor, str(img_name))
    # style_name = r'D:\aaaaaaaaaaaaaaaaa\project\pytorch-train\style.jpg'
    # save_image(styleTensor, str(style_name))

    return output
