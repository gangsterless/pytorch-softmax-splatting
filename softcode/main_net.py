# @Time : 2021/1/22 17:53 

# @Author : xx

# @File : main_net.py 

# @Software: PyCharm

# @description=''
import torch.nn as nn
from pwc.pwc_network import Network as flow_net
from softsplatting import softsplat
from softsplatting.run import backwarp
import torch
from other_modules import context_extractor_layer , Matric_UNet
from gridnet.net_module import GridNet
from pwc.utils.flow_utils import show_compare
import cv2
#这里会把几个模块综合起来 呜呜呜


#一个 Beta 还是两个 Beta 是个大问题！！！！！！！

class Main_net(nn.Module):
    def __init__(self,shape):
        super(Main_net, self).__init__()
        #直接把原始图片的分辨率传进来，方便确认图片大小
        self.shape = shape
        # 从第一张到第二张的extracor
        self.feature_extractor_1 = context_extractor_layer()
        # 从第二张到第一张的extracor
        self.feature_extractor_2 = context_extractor_layer()
        self.flow_extractor1to2 = flow_net()
        self.flow_extractor2to1 = flow_net()
        #注意，这个参数是可以学出来的！！！
        self.beta1 = nn.Parameter(-torch.ones(1))
        self.beta2 = nn.Parameter(-torch.ones(1))
        self.Matric_UNet = Matric_UNet()
        self.grid_net = GridNet()
    #把估计的flow 尺寸也减小
    def scale_flow(self,flow):
        #注意 pwc-net  那边传回来的flow的分辨率是原始大小的1/4
        intHeight,intWidth = self.shape[2:]
        #到底应该乘几是个谜我觉得确实应该乘的是 2,4
        # https://github.com/sniklaus/softmax-splatting/issues/12
        flow_scale_half =(20.0/2.0) * nn.functional.interpolate(input=flow,
                                                                     size=(int(intHeight/2), int(intWidth /2)),
                                                                     mode='bilinear', align_corners=False)
        flow_scale_raw =(20.0)* nn.functional.interpolate(input=flow,size=(int(intHeight), int(intWidth)),
                                                                     mode='bilinear', align_corners=False)
        return [flow_scale_raw,flow_scale_half,flow*(20/4.0)]

    def scale_tenMetric(self, tenMetric):
        intHeight, intWidth = self.shape[2:]
        tenMetric_scale_half =  nn.functional.interpolate(input=tenMetric,size=(int(intHeight / 2), int(intWidth / 2)), mode='bilinear', align_corners=False)
        tenMetric_scale_quarter = nn.functional.interpolate(input=tenMetric, size=(int(intHeight/4), int(intWidth/4)),
                                                            mode='bilinear', align_corners=False)
        return [tenMetric,tenMetric_scale_half,tenMetric_scale_quarter ]
    def forward(self,img1,img2):
        feature_pyrr1 = self.feature_extractor_1(img1)
        feature_pyrr2 = self.feature_extractor_2(img2)

        flow_1to2 = self.flow_extractor1to2(img1,img2)

        flow_1to2_pyri = self.scale_flow(flow_1to2)
        #可以实时的查看光流  因为我们没有gt 就把预测的都传进去了
        # show_compare(flow_1to2_pyri[0].squeeze().cpu().detach().numpy().transpose(1,2,0), flow_1to2_pyri[0].squeeze().cpu().detach().numpy().transpose(1,2,0))
        flow_2to1 = self.flow_extractor1to2(img2, img1)
        flow_2to1_pyri = self.scale_flow(flow_2to1)

        #这个是important metric的输入之一 注意结果一定是单通道的 对应原论文的公式15
        tenMetric_1to2 = nn.functional.l1_loss(input=img1, target=backwarp(tenInput=img2, tenFlow=flow_1to2_pyri[0]),
                                                reduction='none').mean(1, True)
        tenMetric_1to2 = self.Matric_UNet(tenMetric_1to2,img1)
        tenMetric_ls_1to2 = self.scale_tenMetric(tenMetric_1to2)
        # 我们插中间的 所以是0.5 嗷 这是对图片的warp
        warped_img1 = softsplat.FunctionSoftsplat(tenInput=img1, tenFlow=flow_1to2_pyri[0] * 0.5,
                                                 tenMetric=self.beta1* tenMetric_ls_1to2[0],
                                                 strType='softmax')  # -20.0 is a hyperparameter, called 'beta' in the paper, that could be learned using a torch.Parameter
        # print('beta 1',self.beta1)
        # print('beta 2', self.beta2)
        # warped_img1_out = warped_img1.squeeze().cpu().detach().numpy().transpose(1,2,0)
        # cv2.imshow(warped_img1_out)
        # cv2.waitKey(0)
        warped_pyri1_1 = softsplat.FunctionSoftsplat(tenInput=feature_pyrr1[0], tenFlow=flow_1to2_pyri[0] * 0.5,
                                                 tenMetric=self.beta1* tenMetric_ls_1to2[0],
                                                 strType='softmax')
        warped_pyri1_2 = softsplat.FunctionSoftsplat(tenInput=feature_pyrr1[1], tenFlow=flow_1to2_pyri[1] * 0.5,
                                                    tenMetric=self.beta1 * tenMetric_ls_1to2[1],
                                                    strType='softmax')
        warped_pyri1_3 = softsplat.FunctionSoftsplat(tenInput=feature_pyrr1[2], tenFlow=flow_1to2_pyri[2] * 0.5,
                                                    tenMetric=self.beta1 * tenMetric_ls_1to2[2],
                                                    strType='softmax')

        tenMetric_2to1 = nn.functional.l1_loss(input=img2, target=backwarp(tenInput=img1, tenFlow=flow_1to2_pyri[0]),
                                               reduction='none').mean(1, True)
        tenMetric_2to1 = self.Matric_UNet(tenMetric_2to1, img2)
        tenMetric_ls_2to1 = self.scale_tenMetric(tenMetric_2to1)
        # 我们插中间的 所以是0.5 嗷 这是对图片的warp
        warped_img2 = softsplat.FunctionSoftsplat(tenInput=img2, tenFlow=flow_2to1_pyri[0] * 0.5,
                                                  tenMetric=self.beta2 * tenMetric_ls_2to1[0],
                                                  strType='softmax')  # -20.0 is a hyperparameter, called 'beta' in the paper, that could be learned using a torch.Parameter

        warped_pyri2_1= softsplat.FunctionSoftsplat(tenInput=feature_pyrr2[0], tenFlow=flow_2to1_pyri[0] * 0.5,
                                                    tenMetric=self.beta2 * tenMetric_ls_2to1[0],
                                                    strType='softmax')
        warped_pyri2_2 = softsplat.FunctionSoftsplat(tenInput=feature_pyrr2[1], tenFlow=flow_2to1_pyri[1] * 0.5,
                                                    tenMetric=self.beta2 * tenMetric_ls_2to1[1],
                                                    strType='softmax')
        warped_pyri2_3 = softsplat.FunctionSoftsplat(tenInput=feature_pyrr2[2], tenFlow=flow_2to1_pyri[2] * 0.5,
                                                    tenMetric=self.beta2 * tenMetric_ls_2to1[2],
                                                    strType='softmax')
        grid_input_l1 = torch.cat([warped_img1, warped_pyri1_1,warped_img2,warped_pyri2_1], dim=1)

        grid_input_l2 = torch.cat([ warped_pyri1_2,warped_pyri2_2], dim=1)

        grid_input_l3 = torch.cat([ warped_pyri1_3,warped_pyri2_3], dim=1)

        out = self.grid_net(grid_input_l1,grid_input_l2,grid_input_l3)

        return out
if __name__=='__main__':
    W = 448
    H = 256
    N = 1
    tenFirst = torch.rand(size=(N, 3, H, W)).cuda()
    tenSecond = torch.rand(size=(N, 3, H, W)).cuda()

    model = Main_net(tenFirst.shape).cuda()

    res = model(tenFirst,tenSecond)
    print(res.shape)







