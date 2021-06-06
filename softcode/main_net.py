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


class Main_net(nn.Module):
    def __init__(self,shape):
        super(Main_net, self).__init__()
       
        self.shape = shape
       
        self.feature_extractor = context_extractor_layer()
       
       
        self.flow_extractor = flow_net()
        
       
        self.alpha = nn.Parameter(-torch.ones(1))
      
        self.Matric_UNet = Matric_UNet()
        self.grid_net = GridNet()
    
    def scale_flow(self,flow):
        
        intHeight,intWidth = self.shape[2:]
       
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
        feature_pyrr1 = self.feature_extractor(img1)
        feature_pyrr2 = self.feature_extractor(img2)

        flow_1to2 = self.flow_extractor(img1,img2)

        flow_1to2_pyri = self.scale_flow(flow_1to2)
      
        # show_compare(flow_1to2_pyri[0].squeeze().cpu().detach().numpy().transpose(1,2,0), flow_1to2_pyri[0].squeeze().cpu().detach().numpy().transpose(1,2,0))
        flow_2to1 = self.flow_extractor(img2, img1)
        flow_2to1_pyri = self.scale_flow(flow_2to1)

  
        tenMetric_1to2 = nn.functional.l1_loss(input=img1, target=backwarp(tenInput=img2, tenFlow=flow_1to2_pyri[0]),
                                                reduction='none').mean(1, True)
      
        tenMetric_1to2 = self.Matric_UNet(tenMetric_1to2,img1)
        tenMetric_ls_1to2 = self.scale_tenMetric(tenMetric_1to2)
      
        warped_img1 = softsplat.FunctionSoftsplat(tenInput=img1, tenFlow=flow_1to2_pyri[0] * 0.5,
                                                 tenMetric=self.alpha* tenMetric_ls_1to2[0],
                                                 strType='softmax')  # -20.0 is a hyperparameter, called 'beta' in the paper, that could be learned using a torch.Parameter
        # print('beta 1',self.alpha)
        # print('beta 2', self.beta2)
        # warped_img1_out = warped_img1.squeeze().cpu().detach().numpy().transpose(1,2,0)
        # cv2.imshow(warped_img1_out)
        # cv2.waitKey(0)
        warped_pyri1_1 = softsplat.FunctionSoftsplat(tenInput=feature_pyrr1[0], tenFlow=flow_1to2_pyri[0] * 0.5,
                                                 tenMetric=self.alpha* tenMetric_ls_1to2[0],
                                                 strType='softmax')
        warped_pyri1_2 = softsplat.FunctionSoftsplat(tenInput=feature_pyrr1[1], tenFlow=flow_1to2_pyri[1] * 0.5,
                                                    tenMetric=self.alpha * tenMetric_ls_1to2[1],
                                                    strType='softmax')
        warped_pyri1_3 = softsplat.FunctionSoftsplat(tenInput=feature_pyrr1[2], tenFlow=flow_1to2_pyri[2] * 0.5,
                                                    tenMetric=self.alpha * tenMetric_ls_1to2[2],
                                                    strType='softmax')
        
        tenMetric_2to1 = nn.functional.l1_loss(input=img2, target=backwarp(tenInput=img1, tenFlow=flow_2to1_pyri[0]),
                                               reduction='none').mean(1, True)
        tenMetric_2to1 = self.Matric_UNet(tenMetric_2to1, img2)
        tenMetric_ls_2to1 = self.scale_tenMetric(tenMetric_2to1)
      
        warped_img2 = softsplat.FunctionSoftsplat(tenInput=img2, tenFlow=flow_2to1_pyri[0] * 0.5,
                                                  tenMetric=self.alpha * tenMetric_ls_2to1[0],
                                                  strType='softmax')  # -20.0 is a hyperparameter, called 'beta' in the paper, that could be learned using a torch.Parameter

        warped_pyri2_1= softsplat.FunctionSoftsplat(tenInput=feature_pyrr2[0], tenFlow=flow_2to1_pyri[0] * 0.5,
                                                    tenMetric=self.alpha * tenMetric_ls_2to1[0],
                                                    strType='softmax')
        warped_pyri2_2 = softsplat.FunctionSoftsplat(tenInput=feature_pyrr2[1], tenFlow=flow_2to1_pyri[1] * 0.5,
                                                    tenMetric=self.alpha * tenMetric_ls_2to1[1],
                                                    strType='softmax')
        warped_pyri2_3 = softsplat.FunctionSoftsplat(tenInput=feature_pyrr2[2], tenFlow=flow_2to1_pyri[2] * 0.5,
                                                    tenMetric=self.alpha * tenMetric_ls_2to1[2],
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







