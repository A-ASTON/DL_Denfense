#coding=utf-8

import os, platform
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import PIL.Image as Image
import argparse 
import numpy as np
import cv2
import copy
import random
import time
import tempfile
import net
import torch
import torch.nn as nn
from torch import optim
import torchvision as tv
from torchvision.transforms import transforms as T
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.autograd import Variable

  


from pylib.pytorch.data import FolderDataset as FolderDataset
from pylib.pytorch.traintester import TrainTester
from pylib.pytorch import utils as thutils
from pylib import utils as libutils
import style_transfer as img_styler
import project_utils as prjutils

def get_style_loader(batchsize, dbhome):
    imgroot = os.path.join(dbhome, 'image')
    imgfile = os.path.join(dbhome, 'label.txt')

    style_dataset = FolderDataset(imgListFile=imgfile, imgRoot=imgroot, shuffle=False,
                                  xTransform=basic_transform)
    style_loader = DataLoader(style_dataset, batch_size=batchsize, shuffle=False)
    return style_loader

basic_transform = T.Compose([
    T.Resize(224),
    T.ToTensor(),
    thutils.imagenet_normalizer
])

def defend(img_tensor, classify_model, threshold, flag=None):
    device = torch.device('cuda')
    decoder_model_file = r'/home/ubuntu/LXJ/pytorch-train/models/decoder.pth'
    vgg_model_file = r'/home/ubuntu/LXJ/pytorch-train/models/vgg_normalised.pth'

    decoder = net.decoder
    vgg = net.vgg
    decoder.eval()
    vgg.eval()

    decoder.load_state_dict(torch.load(decoder_model_file))
    vgg.load_state_dict(torch.load(vgg_model_file))
    vgg = nn.Sequential(*list(vgg.children())[:31])

    vgg.to(device)
    decoder.to(device)
    classify_model.eval()

    dbhome = r'/home/ubuntu/LXJ/pytorch-train/style_img'
    style_loader = get_style_loader(1, dbhome)

    if flag is None:
        model_output_temp = {}
        img_output_temp = {}
        possibility = []
        i = 0
        for batch_data in style_loader:
            styleTensor = batch_data[0]
            img_with_style = img_styler.pic_transfer(img_tensor,styleTensor,vgg,decoder,threshold)
            img_output_temp[i] = img_with_style
            model_output_temp[i] = classify_model(img_with_style)
            possibility.append(model_output_temp[i].data.max(1, keepdim=True)[0])
            i = i + 1

        idx = possibility.index(max(possibility))
        model_output = model_output_temp[idx]
    else:
        output_temp = classify_model(img_tensor)
        model_output = torch.zeros_like(output_temp)
        for imgk in range(img_tensor.shape[0]):
            img = img_tensor[imgk].unsqueeze(0)
            model_output_temp = {}
            img_output_temp = {}
            possibility = []
            i = 0
            for batch_data in style_loader:
                styleTensor = batch_data[0]
                img_with_style = img_styler.pic_transfer(img,styleTensor,vgg,decoder,threshold)
                img_output_temp[i] = img_with_style
                model_output_temp[i] = classify_model(img_with_style)
                possibility.append(model_output_temp[i].data.max(1, keepdim=True)[0])
                i = i + 1
            idx = possibility.index(max(possibility))
            temp = model_output_temp[idx]
            model_output[0:imgk+1] = temp

    return model_output


def test(dataLoader):        
    model.cuda()
    model.eval()    

    correct = 0     
    for batch_data in enumerate(dataLoader):   
        batch_idx = batch_data[0]
        if len(batch_data[1]) == 2:
            data,target = batch_data[1]
        else:
            data,target,imgfilename = batch_data[1]             

        data, target = data.cuda(), target.cuda()

        with torch.no_grad():
            data, target = Variable(data), Variable(target)

        # output = model(data)
        output = defend(data, model ,threshold)
        if isinstance(output, tuple):
            output = output[0]

        pred = output.data.max(1, keepdim=True)[1]  
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    acc = int(100. * correct / len(dataLoader.dataset))
    print('Test: nImages=%5d nCorrect=%5d avgAcc=%d' % (len(dataLoader.dataset), correct, acc))
    return acc
   


if __name__ == "__main__":    
    useCuda = True
    device = torch.device("cuda" if (useCuda and torch.cuda.is_available()) else "cpu")
    
    ###################  TRAIN models #############################
    # get the model
    model_name = 'resnet34'
    model, modififed = prjutils.get_resnet_model(model_name)
    model.to(device)    

    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-4) 
    criterion = nn.CrossEntropyLoss() 
    # get data loader.
    dbhome = '../dataset'
    train_loader, test_loader = prjutils.get_mini_imagenet(trainBS=64, testBS=1, dbhome=dbhome)
    prjutils.train_model(criterion, model, model_name, train_loader,test_loader)
