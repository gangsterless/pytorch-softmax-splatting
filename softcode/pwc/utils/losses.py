import torch
import torch.nn as nn
import torch.nn.functional as F
from .constv import ConstV

#const value 




def L1loss(x, y): return (x - y).abs().mean()
def L2loss(x, y): return torch.norm(x - y, p = 2, dim = 1).mean()

def training_loss( flow_pyramid, flow_gt_pyramid):
    return sum(w * L2loss(flow, gt) for w, flow, gt in zip(ConstV.weights, flow_pyramid, flow_gt_pyramid))
    
def robust_training_loss( flow_pyramid, flow_gt_pyramid):
    return sum((w * L1loss(flow, gt) + ConstV.epsilon) ** ConstV.q for w, flow, gt in zip(ConstV.weightsls, flow_pyramid, flow_gt_pyramid))
    


def EPE(input_flow, target_flow):
    return torch.norm(target_flow-input_flow,p=2,dim=1).mean()


class L1(nn.Module):
    def __init__(self):
        super(L1, self).__init__()
    def forward(self, output, target):
        lossvalue = torch.abs(output - target).mean()
        return lossvalue


class L2(nn.Module):
    def __init__(self):
        super(L2, self).__init__()
    def forward(self, output, target):
        lossvalue = torch.norm(output-target,p=2,dim=1).mean()
        return lossvalue


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()
        self.loss = L1()
        self.loss_labels = ['L1', 'EPE']

    def forward(self, outputs, target):
        lossvalue = self.loss(outputs[-1], target)
        epevalue = EPE(outputs[-1], target)
        return [lossvalue, epevalue]


class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()
        self.loss = L2()
        self.loss_labels = ['L2', 'EPE']

    def forward(self, outputs, target):
        lossvalue = self.loss(outputs[-1], target)
        epevalue = EPE(output, target)
        return [lossvalue, epevalue]


class MultiScale(nn.Module):
    def __init__(self, startScale = 5, numScales = 6, l_weight= 0.32, norm= 'L1'):
        super(MultiScale,self).__init__()

        self.startScale = startScale
        self.numScales = numScales
        self.loss_weights = torch.FloatTensor([(l_weight / 2 ** scale) for scale in range(self.numScales)]).to(ConstV.my_device)
      
        self.l_type = norm
        self.div_flow = 0.05
        assert(len(self.loss_weights) == self.numScales)

        if self.l_type == 'L1': self.loss = L1()
        else: self.loss = L2()

        self.multiScales = [nn.AvgPool2d(2**l, 2**l) for l in range(ConstV.num_levels)][::-1][:ConstV.output_level]
        self.loss_labels = ['MultiScale-'+self.l_type, 'EPE'],

    def forward(self, outputs, target):
       
        # if flow is normalized, every output is multiplied by its size
        # correspondingly, groundtruth should be scaled at each level
        targets = [avg_pool(target) / 2 ** (ConstV.num_levels - l -1 ) for l, avg_pool in enumerate(self.multiScales)] + [target]
        loss, epe = 0, 0
        loss_levels, epe_levels = [], []
        for w, o, t in zip(ConstV.weightsls, outputs, targets):
            # print(f'flow值域: ({o.min()}, {o.max()})')
            # print(f'gt值域: ({t.min()}, {t.max()})')
            # print(f'EPE:', EPE(o, t))
            loss += w * self.loss(o, t)
            epe += EPE(o, t)
            loss_levels.append(self.loss(o, t))
            epe_levels.append(EPE(o, t))
        return [loss, epe, loss_levels, epe_levels]

