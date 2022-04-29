from torch.autograd import Variable
import project_utils as prjutils
import os, platform
import numpy

import sys
from pylib.pytorch.data import FolderDataset as FolderDataset
import argparse
from pathlib import Path
from torchvision.transforms import transforms as T
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
import net
from function import adaptive_instance_normalization, coral
from pylib.pytorch import utils as thutils
import pylib.utils as libutils
from torchvision import transforms
from PIL import Image
import pylib.pytorch.utils as tchutils
from torchvision import utils as vutils
import style_transfer as img_styler
from tqdm import tqdm

basic_transform = T.Compose([
    T.Resize(224),
    T.ToTensor(),
    thutils.imagenet_normalizer
])

def get_style_loader(batchsize, dbhome):
    imgroot = os.path.join(dbhome, 'image')
    imgfile = os.path.join(dbhome, 'label.txt')

    style_dataset = FolderDataset(imgListFile=imgfile, imgRoot=imgroot, shuffle=False,
                                  xTransform=basic_transform)
    style_loader = DataLoader(style_dataset, batch_size=batchsize, shuffle=False)
    return style_loader

class StyleDefenseNet(nn.Module):
    def __init__(self, model, alpha):
        super(StyleDefenseNet, self).__init__()
        self.model = model
        self.alpha = alpha
        print('=====> init StyleDefenseNet alpha: %.1f' % alpha)

    def forward(self, input):
        x = StyleTransfer(input, self.model, self.alpha)
        
        return self.model(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
decoder_model_file = r'./models/decoder.pth'
vgg_model_file = r'./models/vgg_normalised.pth'

decoder = net.decoder
vgg = net.vgg
decoder.eval()
vgg.eval()

decoder.load_state_dict(torch.load(decoder_model_file, map_location=device))
vgg.load_state_dict(torch.load(vgg_model_file, map_location=device))


vgg = nn.Sequential(*list(vgg.children())[:31])

vgg.to(device)
decoder.to(device)

resnet_model_path = './resnet_model/resnet34_20_94.pth'
styleimg_path = './style_img'

resnet_model = torch.load(resnet_model_path,  map_location=device)

resnet_model.eval()
resnet_model.to(device)
style_loader = get_style_loader(1, styleimg_path)

def StyleTransfer(input, model, alpha):
    model_output_temp = {}
    img_output_temp = {}
    possibility = []
    i = 0
    # 用15种风格进行风格转移，每种风格有一个预测结果
    # 多种模板结果如何融合！！！！
    # 更换pic_transfer算法！！！
    #方法一：15个里面取最大值的最大值
    for batch_data in style_loader:
        styleTensor = batch_data[0]
        img_with_style = img_styler.pic_transfer(input, styleTensor, vgg, decoder, alpha)
        img_output_temp[i] = img_with_style.to(device)
        model_output_temp[i] = model(img_with_style)
        possibility.append(model_output_temp[i].data.max(1, keepdim=True)[0])
        i = i + 1
    img_output = img_output_temp[possibility.index(max(possibility))]

    #方法二：基于投票，不太行
    # vote = torch.tensor([0] * 15)
    # for batch_data in style_loader:
    #     styleTensor = batch_data[0]
    #     img_with_style = img_styler.pic_transfer(input, styleTensor, vgg, decoder, 1.1)
    #     img_output_temp[i] = img_with_style
    #     model_output_temp[i] = model(img_with_style)
    #     vote[model_output_temp[i].data.max(1)[1]] += 1 #投票
    #     i = i + 1
    # img_output = img_output_temp[vote.max(0)[1].item()]


    #方法三：均值法, 15张图片的均值
    # temp = torch.zeros((1, 3, 224, 224)).to(device)
    #
    # for batch_data in style_loader:
    #     styleTensor = batch_data[0]
    #     img_with_style = img_styler.pic_transfer(input, styleTensor, vgg, decoder, 1.1)
    #     temp = temp.add(img_with_style/style_loader.dataset.size())
    #     i = i + 1
    # img_output = temp
    # torch.cuda.empty_cache()
    return img_output

    # 方法四：排序平均法？
    # 方法五：

def test_generalization(test_model, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_data in dataloader:
            inputs = batch_data[0].to(device)
            targets = batch_data[1].to(device)
            with torch.no_grad():
                outputs = test_model(inputs)

            _, pred_idx = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += pred_idx.eq(targets.data).cpu().sum().float()

            sys.stdout.write("\rGeneralization... Acc: %.3f%% (%d/%d)"
                             % (100. * correct / total, correct, total))
            sys.stdout.flush()

    return 100. * correct / total

def test_pgd_attack(test_model, dataloader):
    correct = 0
    total = 0


    for batch_data in dataloader:
        inputs = batch_data[0].to(device)
        targets = batch_data[1].to(device)

        #假如传入的不是test_model而是resnet_model，那么这是黑盒吗？
        x_adv, deltaBd = pgd_attack_01(inputs, targets, resnet_model, epsilon=8/255)
        outputs = test_model(x_adv)
        _, pred_idx = torch.max(outputs.data, 1)
        
        total += targets.size(0)
        # correct += pred_idx.eq(pred_original.data).cpu().sum().float()
        correct += pred_idx.eq(targets.data).cpu().sum().float()
        
        sys.stdout.write("\rGeneralization... Acc: %.3f%% (%d/%d)"
                            % (100. * correct / total, correct, total))
        sys.stdout.flush()
        torch.cuda.empty_cache()

    return 100. * correct / total

# the input data is within [0 1]  
def pgd_attack_01(X, true_target, model, mask=None, epsilon=8/255, alpha=0.1, num_iter=10, loss_func=None):
    model.eval()

    if loss_func is None:
        loss_func = nn.CrossEntropyLoss()
    
    small_constant = 1e-10
    assert(X.min() >= 0)
    assert(X.max() <= 1.0+small_constant)
    
    y = true_target
    nData = len(y)

    randomize=False
    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True)

    # 传入正常样本X时的模型输出
    tmp_output = model(X)
    # 输出预测标签，保持维度
    pred_init = tmp_output.max(1,keepdim=True)[1]
    
    if mask is None:
        mask = torch.ones_like(X, requires_grad=False)

    deltaBd = delta.data.zero_()
    attackLength = [0] * nData
    loss_list = []
    #num_iter:10 10步PGD攻击
    for t in range(num_iter):
        x_adv = X + delta*mask
        #torch.clamp 值压缩到0.0~1.0之间
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        
        tmp_output = model(x_adv)
        pred = tmp_output.max(1,keepdim=True)[1]
        loss = loss_func(tmp_output, y)
        
        loss_list.append(loss.item())
        for imgk in range(0, nData):
            succeed = (not (pred[imgk] == pred_init[imgk]))
            if succeed == True:
                if(attackLength[imgk] == 0):
                    deltaBd[imgk] = x_adv[imgk,:] - X[imgk,:]
                    attackLength[imgk] = t + 1
                    
        model.zero_grad()
        loss.backward(retain_graph=True)
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        if(delta.data.abs().sum() < 1e-10):
            delta.data = torch.rand_like(X) * 2 * epsilon - epsilon
        delta.grad.zero_()

        if min(attackLength) > 0:
            break
        torch.cuda.empty_cache()

    for imgk in range(0, nData):
        if attackLength[imgk] == 0:
            deltaBd[imgk] = x_adv[imgk,:] - X[imgk,:]
            attackLength[imgk] = -1 * t

    # check deltaBd    
    x_adv = X + deltaBd
    for imgk in range(0, nData):
        assert(x_adv[imgk,:].min() >= 0)
        assert(x_adv[imgk,:].max() <= 1.0)

    return x_adv, deltaBd


if __name__ == "__main__":
    useCuda = False
    device = torch.device("cuda" if (useCuda and torch.cuda.is_available()) else "cpu")
    dbhome = '../dataset'
    train_loader, test_loader = prjutils.get_mini_imagenet(trainBS=32, testBS=1, dbhome=dbhome)
    model_path = './resnet_model/resnet34_20_94.pth'

    model = torch.load(model_path, map_location=device)

    model.eval()
    model.to(device)

    style_model = StyleDefenseNet(model, 0.1)
    style_model.eval()
    print('=====> Generalization of StyleDefense model... Acc: %.3f%%' % test_generalization(style_model, test_loader))
    print('=====> White-box PGD on StyleDefense model... Acc: %.3f%%' % test_pgd_attack(style_model, test_loader))
