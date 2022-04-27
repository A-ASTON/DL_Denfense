
import os
import torch
from torch.autograd import Variable
from torch import optim
from pylib.pytorch.torchsummary import summary as torchsummary

class TrainTester():
    def __init__(self, model=None, modelType=-1, train_loader=None, test_loader=None, 
            optimizer=None,criterion=None, scheduler=None, nEpochs=0):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.nEpochs = nEpochs
        self.useCuda = torch.cuda.is_available() #如果有显卡则使用显卡训练
        self.saveModelInterval = 5

        self.isClassify = False
        self.isBinarySeg = False
        self.segThreshold = 0.5         # segmentation threshold

        # scheduler
        self.schedulerType = -1         # -1: disabled, 0: stepLR, 1: piecewise
        self.scheduler_step_size = 10
        self.scheduler_gamma = 0.01     # for StepLR 
        self.scheduler_down_ratio = 0.1 # ratio to multiply
        self.scheduler_curr_lr = 0.01

        self.defaultSGD = optim.SGD(self.model.parameters(), lr=0.01, weight_decay=0.0005, momentum=0.9)   
        self.defaultAdam = optim.Adam(self.model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0005)

        if modelType >= 0:
            self.setModelType(modelType)

    def setModelType(self, intype):
        self.isClassify = (intype == 0)
        self.isBinarySeg = (intype == 1)        # binary segmentation

    def setUseCuda(self, useCuda):
        if useCuda == False:
            self.useCuda = False
        else:
            self.useCuda = torch.cuda.is_available()

    def setDefaultCfg(self):
        if self.model is not None:
            if self.optimizer is None:
                self.optimizer = self.defaultSGD     

            if self.scheduler is None:    
                self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.01)

        if (self.criterion is None) and (self.isClassify == True):    
            self.criterion = torch.nn.CrossEntropyLoss()    
            
    def _setLR_(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def showModelSummary(self, datashape):
        torchsummary(self.model, datashape)

    def loadModelState(self, modelStateDictFile):
        assert(os.path.exists(modelStateDictFile))
        self.model.load_state_dict(torch.load(modelStateDictFile))

    def train(self, modelPrefix, pretrainedModelFile=None, deviceIds=None):
        assert(self.model is not None)
        assert(self.optimizer is not None)

        print("Optimizer: %s" % type(self.optimizer))
        print("Scheduler type: %d" % self.schedulerType)

        if pretrainedModelFile is not None and os.path.exists(pretrainedModelFile):
            self.model.load_state_dict(torch.load(pretrainedModelFile))
            self.test()

        if self.useCuda:
            self.model.cuda()       
        
        for epoch in range(1, self.nEpochs+1):
            if self.schedulerType == 1:
                self._setLR_(self.scheduler_curr_lr)

            print(">> Train Epoch: %d" % epoch)
            self._train_model(epoch) 
            acc = self.test(dbName='test')

            if(epoch % self.saveModelInterval == 0):
                ofile = "./resnet_model/%s_%d_%d.pth" % (modelPrefix,epoch,acc)
                torch.save(self.model, ofile)
                print("Model saved to %s" % ofile)            

            #schedulerType = 0 # -1: disabled, 0: stepLR, 1: piecewise; stepLR学习率调整;piecewise分段
            #学习率调整相关
            if self.schedulerType == 0 and (self.scheduler is not None):
                self.scheduler.step()
                print("\nLr = %.10f" % (self.scheduler.get_lr()[0]))
            elif self.schedulerType == 1:
                if(epoch >= self.scheduler_step_size and epoch % self.scheduler_step_size == 0):
                    self.scheduler_curr_lr  = self.scheduler_curr_lr * self.scheduler_down_ratio
                    if self.scheduler_curr_lr < 0.000001:
                        self.scheduler_curr_lr = 0.000001                    
                    self._setLR_(self.scheduler_curr_lr)
                print("\nLr = %.10f" % (self.scheduler_curr_lr))
            
    def test(self, inTestLoader=None, dbName=None):        
        if self.useCuda:
            self.model.cuda()
        self.model.eval()    

        correct = 0     
        totalRecall = 0
        totalPrecision = 0
        dataLoader = self.test_loader
        if inTestLoader is not None:
            dataLoader = inTestLoader

        for batch_data in enumerate(dataLoader):   
            batch_idx = batch_data[0]
            if len(batch_data[1]) == 2:
                data,target = batch_data[1]
            else:
                data,target,imgfilename = batch_data[1]             

            if self.useCuda:
                data, target = data.cuda(), target.cuda()

            with torch.no_grad():
                data, target = Variable(data), Variable(target)

            output = self.model(data)
            if isinstance(output, tuple):
                output = output[0]

            if self.isClassify == True:
                pred = output.data.max(1, keepdim=True)[1]  
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()
                continue
            
            if self.isBinarySeg == True:
                output = output.detach().cpu().squeeze()
                target = target.squeeze()
                output[output >= self.segThreshold] = 1
                output[output <  self.segThreshold] = 0                
                tmp = output[target == 1]
                totalRecall += (1.0*tmp.sum() / target.sum())  
                totalPrecision += (1.0*tmp.sum() / (output.sum()+0.00001))  
    
        if self.isClassify == True:
            acc = int(100. * correct / len(dataLoader.dataset))
            if dbName is not None:
                print('Test on %s: nImages=%5d nCorrect=%5d avgAcc=%d' % (dbName, len(dataLoader.dataset), correct, acc))
            else:
                print('Test: nImages=%5d nCorrect=%5d avgAcc=%d' % (len(dataLoader.dataset), correct, acc))
            return acc

        if self.isBinarySeg == True:
            nImg = len(dataLoader.dataset)
            avgRecall = int(100 * totalRecall / nImg)
            avgPrecision = int(100 * totalPrecision / nImg)
            if dbName is not None:
                print("Test on %s: nImages=%d avgRecall=%.3f avgPrecision=%.3f" % (dbName, nImg,avgRecall,avgPrecision))   
            else:
                print("Test: nImages=%d avgRecall=%.3f avgPrecision=%.3f" % (nImg,avgRecall,avgPrecision))   
            return (avgRecall, avgPrecision)            

    def _train_model(self, epoch):
        self.model.train() 
        for batch_data in enumerate(self.train_loader):   
            batch_idx = batch_data[0]
            if len(batch_data[1]) == 2:
                data,target = batch_data[1]
            else:
                data,target,imgfilename = batch_data[1] 

            if self.useCuda:                                       
                data, target = data.cuda(), target.cuda()
            
            data, target = Variable(data), Variable(target)   
            self.optimizer.zero_grad()                             
            output = self.model(data)
            if isinstance(output, tuple):
                output = output[0]
            loss = self.criterion(output, target)
            loss.backward()                                   
            self.optimizer.step()                                  
            if batch_idx % 100 == 0:     
                print("%5d/%5d Loss: %.6f" % (batch_idx * len(data), len(self.train_loader.dataset), loss.data.item()))   

    def saveTrainCfg(self, ofile):
        fid = open(ofile, 'a')
                
        tmpstr = 'Model type: '
        if self.isClassify == True:
            tmpstr += 'classification'
        if self.isBinarySeg == True:
            tmpstr += "binary segmentation"            
        fid.write(tmpstr+"\n")
        
        tmpstr = "Criterion: " + str(self.criterion)
        fid.write(tmpstr+"\n")
        
        tmpstr = "Optimizer: " + str(self.optimizer)
        fid.write(tmpstr+"\n")
        
        fid.close()