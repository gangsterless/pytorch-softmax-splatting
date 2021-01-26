import numpy as  np
import torch
import cv2
from main_net import Main_net
tenFirst = torch.FloatTensor(np.ascontiguousarray(cv2.imread(filename='./pair_img/im1.png', flags=-1).transpose(2, 0, 1)[None, :, :, :].astype(np.float32) * (1.0 / 255.0))).cuda()

tenSecond = torch.FloatTensor(np.ascontiguousarray(cv2.imread(filename='./pair_img/im3.png', flags=-1).transpose(2, 0, 1)[None, :, :, :].astype(np.float32) * (1.0 / 255.0))).cuda()

gt = cv2.imread('./pair_img/im2.png')
H = 256
W = 448
shape = [1,3,H,W]
model = Main_net(shape).cuda().eval()
with torch.no_grad():
    model.load_state_dict(torch.load('weights/model_weight_16.pth'))
    img_out = model(tenFirst,tenSecond)
    img_out = img_out.squeeze().detach().cpu().numpy().transpose(1,2,0)
    cv2.imshow('out',img_out)
    cv2.imshow('ground truth',gt)
    cv2.waitKey(0)





