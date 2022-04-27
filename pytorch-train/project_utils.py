# coding:gb18030

import os, platform
import PIL.Image as Image
import argparse 
import numpy as np
import cv2
import copy
import random
import time
import tempfile

import torch as th
import torch.nn as nn
from torch import optim
import torchvision as tv
from torchvision.transforms import transforms as T
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from pylib.pytorch.data import FolderDataset as FolderDataset
from pylib.pytorch.traintester import TrainTester
from pylib.pytorch import utils as thutils
from pylib import utils as utils


basic_transform = T.Compose([
        T.ToTensor(),
        # T.Resize(224),
        # thutils.imagenet_normalizer
    ])
        
def get_alexnet_model(num_classes=100, modelFile=None, use_pretrained=True):
    from pytorchModels import alexnet as alexnet

    model = alexnet.alexnet(num_classes=num_classes)
    if use_pretrained == True:
        model = alexnet.alexnet()
        modelFile = os.path.join(homedir,'pytorchModels','pretrained', 'alexnet.pth')

    modififed = False         
    if modelFile is not None:
        print('Load target alexnet model from %s' % modelFile)
        model.load_state_dict(th.load(modelFile))
      
    fcModule = thutils.findModule(model,'classifier.6')
    featureSize = fcModule.in_features
    nClasses = fcModule.out_features
    if not nClasses == num_classes:
        model.classifier[6] = th.nn.Linear(featureSize, num_classes)
        modififed = True
        
    return model, modififed
# def get_resnet_model(model_name, num_classes=100, model_file=None, use_pretrained=True):
def get_resnet_model(model_name, num_classes=20, model_file=None, use_pretrained=True):
    from pytorchModels import resnet as resnet

    if model_name == 'resnet34':
        model = resnet.resnet34(num_classes=num_classes)
        if use_pretrained == True:
            model = resnet.resnet34()
            model_file = os.path.join('./pytorchModels','pretrained', model_name+'.pth')

    modififed = False         
    if model_file is not None:
        print('Load target resnet model from %s' % model_file)
        model.load_state_dict({k.replace('module.',''): v for k,v in th.load(model_file).items()})
      
    fcModule = thutils.findModule(model,'fc')
    featureSize = fcModule.in_features
    nClasses = fcModule.out_features
    if not nClasses == num_classes:
        model.fc = th.nn.Linear(featureSize, num_classes)
        modififed = True    
        
    return model, modififed

# very basic train    
def train_model(loss_func, model, modelName, trainLoader,testLoader, classNames=''):
    
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)   
    run = TrainTester(model=model,nEpochs=200, optimizer=optimizer, criterion=loss_func,
            train_loader=trainLoader, test_loader=testLoader)

    run.setModelType(0)
    run.setDefaultCfg()
    run.schedulerType = 1
    run.scheduler_step_size = 10
    run.scheduler_curr_lr = 0.001

    prefix = modelName
    if len(classNames) > 0:
        prefix = modelName+"_" + '_'.join(classNames)
    
    infofile = prefix + ".info"
    run.saveTrainCfg(infofile)    
    run.train(prefix, None)

def get_mini_imagenet(trainBS=32,testBS=32, dbhome=None):
    imgroot = os.path.join(dbhome, 'images')    
    trainListFile = os.path.join(dbhome,'train.list')
    testListFile = os.path.join(dbhome,'test.list')

    assert(os.path.exists(trainListFile) and os.path.exists(testListFile))
    
    train_dataset = FolderDataset(imgListFile=trainListFile, imgRoot=imgroot, shuffle=True,
                            xTransform=basic_transform)
                            
    test_dataset = FolderDataset(imgListFile=testListFile, imgRoot=imgroot, shuffle=False, 
                            xTransform=basic_transform)    

    train_loader = DataLoader(train_dataset, batch_size=trainBS, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=testBS, shuffle=False)
    print("Data loaded from %s: train %d test %d" % (dbhome, len(train_loader.dataset), len(test_loader.dataset)))
    
    return train_loader, test_loader
    
def get_mini_imagenet_from_list_file(listfile, batch_size=32, dbhome=None, size=None, shuffle=False):  
    imgroot = os.path.join(dbhome, 'images')
    test_dataset = FolderDataset(imgListFile=listfile, imgRoot=imgroot, shuffle=shuffle, xTransform=basic_transform)
    if size is not None:
        idxList = list(range(0,size))    
        if shuffle == True:
            np.random.shuffle(idxList)
        test_dataset = FolderDataset.filterByIndex(test_dataset, idxList)
        
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)   
    return test_loader, test_dataset

def get_mini_imagenet_list(dbhome=None):
    allListFile = os.path.join(dbhome,'all.list')
    con = utils.readAFile(allListFile)
    nameId = {}
    namePath = {}
    for line in con:
        xx = line.split()
        label = int(xx[0])
        tmppath = os.path.join(dbhome, 'images', xx[1])
        tmppath = utils.correctFilePath(tmppath)
        _,imgname,_ = utils.fileparts(tmppath)
        nameId[imgname] = label
        namePath[imgname] = tmppath
    return nameId, namePath

def get_mini_imagenet_from_name_list(listfile, testBS=32, dbhome=None, shuffle=True):
    con = utils.readAFile(listfile)

    nameId, namePath = get_mini_imagenet_list(dbhome=dbhome)
    tmpfd, tempfilename = tempfile.mkstemp()
    tempfilename += '.txt'
    fid = open(tempfilename,'w')
    for line in con:
        xx = line.split()
        imgname = xx[0]

        id = nameId[imgname]
        tmppath = namePath[imgname]
        fid.write('%d %s\n' % (id, tmppath))
    fid.close()

    test_dataset = FolderDataset(imgListFile=tempfilename, shuffle=shuffle, xTransform=basic_transform)
    test_loader  = DataLoader(test_dataset, batch_size=testBS, shuffle=shuffle)    
    return test_loader    



    
    
    
    