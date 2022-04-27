import os
from pathlib import Path
import numpy as np
import torch
from torch._C import device
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import pylib.utils as libutil
import net
from function import adaptive_instance_normalization, coral, calc_mean_std
from torchvision.transforms import transforms as T
from pylib.pytorch import utils as thutils
from pylib.pytorch.data import FolderDataset as FolderDataset
from torch.utils.data import DataLoader



def getStyleTFVector(vgg, style):
    style_f = vgg(style)
    style_mean, style_std = calc_mean_std(style_f)
    return style_mean, style_std

def rebuilt_content(vgg, decoder, content):
    content_f = vgg(content)
    return decoder(content_f)

def get_model(vggModelFile, decoderModelFile):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    decoder = net.decoder
    vgg = net.vgg
    decoder.eval()
    vgg.eval()

    decoder.load_state_dict(torch.load(decoderModelFile))
    
    vgg.load_state_dict(torch.load(vggModelFile))
    vgg = nn.Sequential(*list(vgg.children())[:31])
    
    vgg.to(device)
    decoder.to(device)
    return vgg, decoder

def getStyleImg(imgFile, imgSize):
    style_tf = transforms.Compose([        
        transforms.Resize(imgSize),
        transforms.ToTensor()
    ])

    styleImg = Image.open(imgFile)
    styleImg = style_tf(styleImg) 
    styleImg = styleImg.unsqueeze(0)
    return styleImg   


class max_vote(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output, x = torch.max(input, 0, keepdim=True)
        ctx.input_shape = input.shape
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x = []
        for k in range(ctx.input_shape[0]):
            x.append(grad_output)
        xx = torch.cat(x, 0)
        return xx



class StyleDefendNet(nn.Module):
    def __init__(self, encoder, decoder):
        super(StyleDefendNet, self).__init__()
        self.targetModel = None        # classify model
        self.votenet = None            # vote net
        self.vgg = encoder
        self.decoder = decoder
        self.styleNetwork = None
        self.alpha = 0.6
        

    def init(self, device, styleloader):
        self.device = device
        self.styleNetwork = net.Net(self.vgg, self.decoder)
        self.style_loader = styleloader
        # self.vgg.to(device).eval()
        # self.decoder.to(device).eval()
        # self.targetModel.to(device).eval()


    def set_eval(self):
        self.targetModel.to(self.device).eval
        # self.votenet.to(self.device).eval
        self.vgg.to(self.device).eval
        self.decoder.to(self.device).eval
    
    def forward(self, input):
        model_output_temp = []
        for batch_data in self.style_loader:
            styleTensor = batch_data[0].to(self.device)
            styledImg, loss_c, loss_s = self.styleNetwork(input, styleTensor, self.alpha)    
            model_output_temp = self.targetModel(styledImg).view(1,300)
        output = torch.tensor(self.vote_net.predict_proba(model_output_temp.cpu().detach().numpy()), requires_grad = True).cuda()
        return output