#coding=utf-8

import os
import PIL.Image as Image
import numpy as np

import torch
from torch.utils.data import Dataset
import pylib.utils as utils
from pylib.pytorch import utils as thutil

class FolderDataset(Dataset):
    def __init__(self, 
            imgRoot=None, imgListFile=None,     # way 1
            imgDir=None, fextList=None,         # way 2
            yId=-1,
            shuffle=True, maxCount=-1, xTransform=None, yTransform=None, cacheImg=False, getitemFun=None):
        self.imgPathList = []
        self.targetList = []
        self.imgList = []

        self.cacheImgData = cacheImg
        self.x_transform = xTransform
        self.y_transform = yTransform
        self.getitemFun = getitemFun

        if imgListFile is not None:
            fid = open(imgListFile,'r')
            lines = fid.readlines()
            fid.close()

            if shuffle == True:
                np.random.shuffle(lines)

            for line in lines:
                x = line.split()
                if len(x) == 2:
                    imgfile = x[1]
                else:
                    imgfile = x[0]
                
                if imgRoot is not None:
                    imgfile = os.path.abspath(os.path.join(imgRoot, imgfile))
                imgfile = utils.correctFilePath(imgfile)
                if not os.path.exists(imgfile):
                    continue
                self.imgPathList.append(imgfile)

                if len(x) == 2:
                    self.targetList.append(int(x[0]))
                elif yId is not None:
                    self.targetList.append(yId)
                    
                if(maxCount > 0 and len(self.imgPathList) >= maxCount):
                    break
        elif imgDir is not None:
            fileList = os.listdir(os.path.join(imgDir))
            if shuffle == True:
                np.random.shuffle(fileList)
            
            for fname in fileList:
                _, _, tmpext = utils.fileparts(fname)
                imgfile = os.path.join(imgDir, fname)
                if (fextList is not None) and  (tmpext not in fextList):
                    continue
                    
                self.imgPathList.append(imgfile)
                self.targetList.append(yId)
                if(maxCount > 0 and len(self.imgPathList) >= maxCount):
                    break                

        if self.cacheImgData == True:
            for imgfile in self.imgPathList:
                img = Image.read(imgfile)
                img = thutil.PILImgToNumpy(img)
                self.imgList.append(img)

    def __getitem__(self,index):
        if(self.getitemFun is not None):
            return self.getitemFun(self, index)

        if self.cacheImgData == False:
            x = Image.open(self.imgPathList[index]) 
        else:
            x = thutil.NumpyToPILImg(self.imgList[index])

        if self.x_transform is not None:
            x = self.x_transform(x)

        y = self.targetList[index]
        if self.y_transform is not None:
            y = self.y_transform(y)    

        return x,y,self.imgPathList[index]
    
    def __len__(self):
        return len(self.targetList)

    def showImg(self, index):
        x = Image.open(self.imgPathList[index])
        x.show()

    def size(self):
        return len(self.targetList)

    def setTarget(self, index, y):
        if index < 0:
            self.targetList = [y]  * len(self.targetList)
        else:
            self.targetList[index] = y

    def extend(self, datasetB):
        self.imgPathList.extend(datasetB.imgPathList)
        self.targetList.extend(datasetB.targetList)
        self.imgList.extend(datasetB.imgList)
        return self
        
    def getTargetHist(self, numOfClasses):
        freq = [0] * numOfClasses
        for v in self.targetList:
            freq[v] += 1
        return freq, len(self.targetList)
        
    def saveImgList2File(self, ofile):
        fid = open(ofile, 'w')        
        for k in range(0, len(self.imgPathList)):
            tmpstr = ''
            if(len(self.targetList) == len(self.imgPathList)):
                if(type(self.targetList[k]) == int):
                    tmpstr = '%3d ' % self.targetList[k]
                if(type(self.targetList[k]) == float):
                    tmpstr = '%.3f ' % self.targetList[k]
            tmpstr += self.imgPathList[k] + "\n"        
            fid.write(tmpstr)
        fid.close()    
        
    def splitTrainTest(self, trainRatio=0.7):
        train = FolderDataset()
        test = FolderDataset()

        train.x_transform = self.x_transform
        train.y_transform = self.y_transform
        test.x_transform = self.x_transform
        test.y_transform = self.y_transform

        targetList = np.array(self.targetList)
        nClasses = max(targetList) + 1
        for target in range(0, nClasses):
            idx = np.where(targetList == target)[0]
            randList = np.random.rand(1,len(idx))[0]
            for k in range(0, len(idx)):                
                kk = idx[k]
                if(randList[k] > 1-trainRatio):
                    train.targetList.append(self.targetList[kk])
                    train.imgPathList.append(self.imgPathList[kk])
                else:
                    test.targetList.append(self.targetList[kk])
                    test.imgPathList.append(self.imgPathList[kk])                    
        return train, test
    
    @staticmethod
    def filterByIndex(inSet, idxList):
        one = FolderDataset()
        one.cacheImgData = inSet.cacheImgData
        one.x_transform = inSet.x_transform
        one.y_transform = inSet.y_transform
        one.getitemFun = inSet.getitemFun

        for k in idxList:
            one.imgPathList.append(inSet.imgPathList[k])
            one.targetList.append(inSet.targetList[k])
            if len(inSet.imgList) > 0:
                one.imgList.append(inSet.imgList[k])
        return one
    
    @staticmethod
    def filterByImgName(inSet, nameList):
        one = FolderDataset()
        one.cacheImgData = inSet.cacheImgData
        one.x_transform = inSet.x_transform
        one.y_transform = inSet.y_transform
        one.getitemFun = inSet.getitemFun

        for k in range(0, len(inSet.imgPathList)):
            _, tmpname, _ = utils.fileparts(inSet.imgPathList[k])
            if tmpname in nameList:
                one.imgPathList.append(inSet.imgPathList[k])
                one.targetList.append(inSet.targetList[k])
                if len(inSet.imgList) > 0:
                    one.imgList.append(inSet.imgList[k])
        return one

    @staticmethod
    def filterByIdList(inSet,idList):
        one = FolderDataset()
        one.cacheImgData = inSet.cacheImgData
        one.x_transform = inSet.x_transform
        one.y_transform = inSet.y_transform
        one.getitemFun = inSet.getitemFun

        assert(len(inSet.imgPathList) == len(inSet.targetList))
        for k in range(0, len(inSet.targetList)):
            if inSet.targetList[k] in idList:
                one.imgPathList.append(inSet.imgPathList[k])
                idx = idList.index(inSet.targetList[k])
                one.targetList.append(idx)
        return one
    
    @staticmethod
    def split(inSet, sizeList):
        # split the dataset as a whole. not split within each class.
        outList = []
        start = 0
        for S in sizeList:
            end = min(start + S, inSet.getLen())
            if S < 0:
                end = None

            one = FolderDataset()
            one.cacheImgData = inSet.cacheImgData
            one.x_transform = inSet.x_transform
            one.y_transform = inSet.y_transform
            one.getitemFun = inSet.getitemFun

            one.imgPathList = inSet.imgPathList[start:end]
            one.targetList  = inSet.targetList[start:end]
            one.imgList     = inSet.imgList[start:end]            
            outList.append(one)
            start += S            
        return outList
                
    @staticmethod
    def saveToFile(ofile, obj):
        utils.saveClassObjToFile(ofile, obj)

    @staticmethod
    def loadFromFile(infile):
        obj = utils.loadClassObjFromFile(infile, FolderDataset)    
        return obj
 




    
