import os
import torchvision.models as models
import torch

homedir = r'D:\WORK'
os.sys.path.append(os.path.join(homedir,'WORK','CodeLibrary'))

from pylib.pytorch import torchsummary as torchsummary

device = torch.device("cuda")

model = models.resnet18(pretrained=False)
model.load_state_dict(torch.load('./pretrained/resnet18.pth'))
model.to(device)
torchsummary.summary(model, (3, 256,256))
os.sys.exit(0)

resnet18 = models.resnet18(pretrained=True)
resnet34 = models.resnet34(pretrained=True)
resnet34 = models.resnet50(pretrained=True)
resnet34 = models.resnet101(pretrained=True)
resnet34 = models.resnet152(pretrained=True)

vgg16 = models.vgg16(pretrained=True)
vgg16 = models.vgg19(pretrained=True)
vgg16 = models.vgg16_bn(pretrained=True)
vgg16 = models.vgg19_bn(pretrained=True)

densenet = models.densenet121(pretrained=True)
densenet = models.densenet169(pretrained=True)
densenet = models.densenet201(pretrained=True)
densenet = models.densenet161(pretrained=True)


alexnet = models.alexnet(pretrained=True)
squeezenet = models.squeezenet1_0(pretrained=True)

inception = models.inception_v3(pretrained=True)
googlenet = models.googlenet(pretrained=True)
shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
mobilenet = models.mobilenet_v2(pretrained=True)
resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
wide_resnet50_2 = models.wide_resnet50_2(pretrained=True)
mnasnet = models.mnasnet1_0(pretrained=True)

