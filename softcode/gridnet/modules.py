import torch
import torch.nn as nn
import numpy as np
import cv2
from sklearn.decomposition import  PCA
from PIL import Image
class Basic_layer1(nn.Module):
    def __init__(self,block_index):
        super(Basic_layer1, self).__init__()
        # 第一层
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.RReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.RReLU(inplace=False)
        )
    def forward(self,x):
        return self.layer(x)

class Basic_layer2(nn.Module):
    def __init__(self,block_index):
        super(Basic_layer2, self).__init__()
        # 第二层
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.RReLU(inplace=False),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.RReLU(inplace=False)
        )
    def forward(self,x):
        return self.layer(x)
class Basic_layer3(nn.Module):
    def __init__(self,block_index):
        super(Basic_layer3, self).__init__()
        # 第三层
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
            nn.RReLU(inplace=False),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.RReLU(inplace=False)
        )
    def forward(self,x):
        return self.layer(x)

class Feature_extractor(nn.Module):
    def __init__(self,use_pretrained = True):
        super(Feature_extractor, self).__init__()
        self.use_pretrained = use_pretrained
        self.replace = False
        if use_pretrained:

            model = models.resnet18(pretrained=False)
            # model.load_state_dict(torch.load('weights/resnet18-5c106cde.pth'))
            # 只保留第一层 conv
            self.resnet_layer = nn.Sequential(*list(model.children())[:1])
            if self.replace:
                self.resnet_layer[0] =  nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=1,padding=3)
                self.resnet_layer[4] = self.resnet_layer[4][0]
            for each in self.resnet_layer:
                print(each)
            # print(self.resnet_layer)
            # for k,v in self.resnet_layer.named_parameters():
            #     print(k)
            # print(self.resnet_layer.conv1.weight)
            # self.resnet_layer.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=1,padding=3)
            # for k,v in self.resnet_layer.named_parameters():
            #     # if k=='conv1.weight':
            #     #     print(v)
            #
            #     print(k,v.requires_grad)
        else:
            self.layer1 = Basic_layer1()
            self.layer2 = Basic_layer2()
            self.layer3 = Basic_layer3()
    def forward(self,x):
        if self.use_pretrained:
            return torch.mean( self.resnet_layer(x),axis=1)
        else:
            self.feature_map1 = self.layer1(x)
            self.feature_map2 = self.layer2(self.feature_map1)
            self.feature_map3 = self.layer3(self.feature_map2)
            #把金字塔返回出来
            return [x,self.feature_map1,self.feature_map2,self.feature_map3]


class LateralBlock(nn.Module):

    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.f = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1)
        )
        if ch_in != ch_out:
            self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1)

    def forward(self, x):
        fx = self.f(x)
        if fx.shape[1] != x.shape[1]:
            x = self.conv(x)

        return fx + x

class DownSamplingBlock(nn.Module):

    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.f = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(ch_in, ch_out, kernel_size = 3, stride = 2, padding = 1),
            nn.PReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size = 3, padding = 1)
        )

    def forward(self, x):
        return self.f(x)

class UpSamplingBlock(nn.Module):

    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.f = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = False),
            # nn.UpsamplingNearest2d(scale_factor = 2),
            nn.PReLU(),
            nn.Conv2d(ch_in, ch_out, kernel_size = 3, padding = 1),
            nn.PReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size = 3, padding = 1)
        )

    def forward(self, x):
        return self.f(x)
if __name__=='__main__':
    Fe = Feature_extractor()
    img_dir = '/media/zyt/新加卷/data/big4/code/softsplat-impl/test_data'
    img = img_dir+ '/' + 'first.png'
    # img = 'leaf.png'
    img  = Image.open(img)
    # img = torch.FloatTensor(np.ascontiguousarray(
    #     cv2.imread(filename=img, flags=-1).reshape(-1,3,320,320)))

    
    transform = transforms.Compose([transforms.Resize((224, 224)),
                         transforms.ToTensor()])
                     
    img = transform(img)
    img = img.unsqueeze(0)
    res = Fe(img)
    res = res.detach().numpy().reshape(112,112,1)*255
    res = np.stack([res,res,res],axis=2).squeeze()*255
    print(res.shape)
    # res = res.detach().numpy().reshape(160*160,64)
    # print(res.shape)
    # pca = PCA(n_components=3)
    # fit_res = pca.fit_transform(res).reshape(160,160,3)*255
    # cv2.imshow('fit_res.jpg',fit_res)
    cv2.imwrite('fitres.png',res)
    # cv2.waitKey(0)
    # print(fit_res.shape)

