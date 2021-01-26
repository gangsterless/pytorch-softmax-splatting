from run import Network
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import Sintel
import torch
import time
import math
from torchvision import transforms
from utils.constv import ConstV
from utils.losses import MultiScale, robust_training_loss, L1Loss
import numpy as np
import torch.nn.functional as F
from utils.flow_utils import (vis_flow, save_flow)
from logger import Logger
import matplotlib.pyplot as plt
from  utils.losses import EPE
from pwc.utils.flow_utils import show_compare
model = None
#
# 以下都是muti_scale
# 1. img 不除以255 从头开始：：：
 # loss ->   0.1279,  0.1260,  0.1271 降不下去了！！！ epe 16.3左右
#这个loss跟epe降不下来也属于正常 一方面我得训练代码估计不太对 另一方面主要是因为 没有在flychair上面训练

#2 img除以255 从头开始  0.1499->0.1337->0.1313->0.1291->0.1276->0.1246->1270 不降了
# 作者说必须要用set_device 指定设备

#3 用了random crop 1,2都是center crop 0.1333->0.1302->0.1292->1273->1277
# You are correct, it should be set_device instead of just device. Thank you for letting me know! No need to issue a pull request,
# I just updated it and removed that line altogether. It is usually better to set CUDA_VISIBLE_DEVICES
# and I only had this line to avoid issues with multi-GPU systems.
# 4 使用预训练模型 loss 0.0812->0.0740->0.0722->0.0741 6.3左右的epe 1


#以下是 一个scale
# 使用预训练模型 loss 8.8289 epe 13.27 明显不行
#不使用预训练模型 0.1470 0.1343 。。。。。 最好的 1296 epe 16.6 看来还是muti比较牛逼

def train():
    def estimate(tenFirst, tenSecond):

        # if model is None:
        #     model = Network().cuda().train()
        intWidth = tenFirst.shape[3]
        intHeight = tenFirst.shape[2]
        tenPreprocessedFirst = tenFirst.cuda().view(-1, 3, intHeight, intWidth)
        tenPreprocessedSecond = tenSecond.cuda().view(-1, 3, intHeight, intWidth)
        intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
        intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))

        tenPreprocessedFirst = torch.nn.functional.interpolate(input=tenPreprocessedFirst,
                                                               size=(intPreprocessedHeight, intPreprocessedWidth),
                                                               mode='bilinear', align_corners=False)
        tenPreprocessedSecond = torch.nn.functional.interpolate(input=tenPreprocessedSecond,
                                                                size=(intPreprocessedHeight, intPreprocessedWidth),
                                                                mode='bilinear', align_corners=False)
        flow_out_ls = model(tenPreprocessedFirst, tenPreprocessedSecond)
        if ConstV.mutil_l:
            return flow_out_ls

        else:
            tenFlow = 20.0 * torch.nn.functional.interpolate(input=flow_out_ls, size=(intHeight, intWidth),
                                                             mode='bilinear',
                                                             align_corners=False)
            #
            tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
            tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)
            return tenFlow
    # global model
    crop_shape = [384, 448]
    sintel_dataset = Sintel(dataset_dir=ConstV.dataset_dir, train_or_test=ConstV.train_or_test, crop_shape=crop_shape)
    model = Network().cuda().train()
    train_loader = DataLoader(sintel_dataset,
                              batch_size=ConstV.batch_size,
                              shuffle=True,
                              num_workers=ConstV.num_workers,
                              pin_memory=True)
    forward_time = 0
    backward_time = 0
    optimizer = torch.optim.Adam(params=model.parameters(), lr=ConstV.lr, weight_decay=ConstV.weight_decay)

    total_loss_levels = [0] * ConstV.num_levels
    total_epe_levels = [0] * ConstV.num_levels
    # 金字塔L1 loss
    # loss = robust_training_loss(args, flows, flow_gt_pyramid)
    if ConstV.mutil_l:
        criteration = MultiScale()
    else:
        criteration = L1Loss()

    # logger = Logger('log')
    for step in range(1, ConstV.total_step + 1):
        total_loss = 0
        total_epe = 0
        for ix, data in enumerate(train_loader):
            data = [each.to(ConstV.my_device) for each in data]
            img1, img2, flow_gt = data

            t_forward = time.time()
            out_flow = estimate(img1, img2)
            if ConstV.mutil_l:
                forward_time += time.time() - t_forward
                # print(forward_time)
                # loss, epe, loss_levels, epe_levels = criteration(out_flow, flow_gt)
                loss, epe, loss_levels, epe_levels = criteration(out_flow, flow_gt)
                total_loss += loss
                total_epe+=epe
                for l, (loss_, epe_) in enumerate(zip(loss_levels, epe_levels)):
                    total_loss_levels[l] += loss_.item()
                    total_epe_levels[l] += epe_.item()
                t_backward = time.time()
                optimizer.zero_grad()

                loss.backward()
                optimizer.step()
                backward_time += time.time() - t_backward
                print('step,,' + str(step) + '  data:  ' + str(ix) + '\\' + str(len(train_loader)) + '    loss :' + str(
                    loss.item()))
                print('epe,,' + str(epe.item()))
            else:

                # print(img1.shape)
                # print(out_flow.shape)
                forward_time += time.time() - t_forward
                # print(forward_time)
                # loss, epe, loss_levels, epe_levels = criteration(out_flow, flow_gt)
                lossvalue, epevalue = criteration(out_flow, flow_gt)
                total_loss += lossvalue
                total_epe+=epevalue
                # total_loss += loss.item()
                # total_epe += epe.item()
                # for l, (loss_, epe_) in enumerate(zip(loss_levels, epe_levels)):
                #     total_loss_levels[l] += loss_.item()
                #     total_epe_levels[l] += epe_.item()

                t_backward = time.time()
                optimizer.zero_grad()
                # loss.backward()
                lossvalue.backward()
                optimizer.step()
                backward_time += time.time() - t_backward
                print('step,,,' + str(step) + '   ' + str(ix) + '\\    loss ' + str(len(train_loader)))
                print('loss value :', lossvalue)
                print('epe :', epevalue)
                # if step % ConstV.summary_interval == 0:
                #     print('step:::',step)
                # # Scalar Summaries
                # # ============================================================
                #     print('lr', optimizer.param_groups[0]['lr'], step)
                #     print('loss', total_loss / step, step)
                #     print('EPE', total_epe / step, step)
                #
                # for l, (loss_, epe_) in enumerate(zip(loss_levels, epe_levels)):
                #     print(f'loss_lv{l}', total_loss_levels[l] / step, step)
                #     print(f'EPE_lv{l}', total_epe_levels[l] / step, step)
                #     # ============================================================
                #     B = out_flow[0].size(0)
                #     vis_batch = []
                #     for b in range(B):
                #         batch = [np.array(
                #             F.upsample(out_flow[l][b].unsqueeze(0), scale_factor=2 ** ((len(out_flow) - l + 1))).cpu().detach().squeeze(
                #                 0)).transpose(1, 2, 0) for l in range(len(out_flow) - 1)]
                #         # for i in batch:
                #         #     print(i.shape)
                #         # print(flows[-1][b].detach().cpu().numpy().transpose(1,2,0))
                #         # print(flow_gt[b].detach().cpu().numpy().transpose(1,2,0).shape)
                #         vis = batch + [out_flow[-1][b].detach().cpu().numpy().transpose(1, 2, 0),
                #                        flow_gt[b].detach().cpu().numpy().transpose(1, 2, 0)]
                #         vis = np.concatenate(list(map(vis_flow, vis)), axis=1)
                #         vis_batch.append(vis.transpose(2, 0, 1))
                #     logger.image_summary(f'flow', vis_batch, step)
        print('epoch ' + str(step) + 'avg  loss :' + str(total_loss / len(train_loader))+ 'avg  epe :' + str(total_epe / len(train_loader)))
def test():
    model_path = './weights/network-default.pytorch'
    num_workers = 0
    batch_size = 4
    print('load model...')
    model = Network().cuda().eval()
    model.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.load(model_path).items()})

    print('build eval dataset...')

    test_dataset =Sintel(dataset_dir=ConstV.dataset_dir, train_or_test='test')
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=True,
                             num_workers=num_workers,
                             pin_memory=True)

    total_batches = len(test_loader)

    # logs
    # ============================================================

    total_epe = 0
    print(len(test_loader))
    for batch_idx, (img1,img2 ,flow_gt) in enumerate(test_loader):
        # Forward Pass
        # ============================================================
        # t_start = time.time()
        # data, target = [d.to(args.device) for d in data], [t.to(args.device) for t in target]

        with torch.no_grad():
            img1 = img1.to(ConstV.my_device)
            img2 = img2.to(ConstV.my_device)
            flow_gt = flow_gt.to(ConstV.my_device)
            intWidth = img1.shape[3]
            intHeight = img2.shape[2]
            tenPreprocessedFirst = img1.view(-1, 3, intHeight, intWidth)
            tenPreprocessedSecond = img2.view(-1, 3, intHeight, intWidth)
            intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
            intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))

            tenPreprocessedFirst = torch.nn.functional.interpolate(input=tenPreprocessedFirst,
                                                                   size=(intPreprocessedHeight, intPreprocessedWidth),
                                                                   mode='bilinear', align_corners=False)
            tenPreprocessedSecond = torch.nn.functional.interpolate(input=tenPreprocessedSecond,
                                                                    size=(intPreprocessedHeight, intPreprocessedWidth),
                                                                    mode='bilinear', align_corners=False)
            flow_out = model(tenPreprocessedFirst,tenPreprocessedSecond)

            #要把形状改的跟gt一样
            flow_out =  torch.nn.functional.interpolate(input=flow_out,
                                                                    size=(intHeight, intWidth),
                                                                    mode='bilinear', align_corners=False)
            show_compare(flow_out.cpu().numpy().squeeze().transpose(1,2,0),flow_gt.cpu().numpy().squeeze().transpose(1,2,0))


            epe = EPE(flow_out, flow_gt)
            total_epe+=epe
            print(f'[{batch_idx}/{total_batches}]  Time: {batch_idx:.2f}s')  # EPE:{total_epe / batch_idx}')
            print('avg epe :',total_epe/batch_idx)
        # time_logs.append(time.time() - t_start)

        # Compute EPE

if __name__ == '__main__':
    if ConstV.train_or_test=='train':
        train()
    else:
        test()