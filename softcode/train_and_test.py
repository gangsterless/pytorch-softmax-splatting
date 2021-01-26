
from vimo_dataset import Vimeo
from main_net import Main_net
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from loss_f import LapLoss


vimo_data_dir = 'D:/dataset/vimeo_triplet'

# class LaplacianPyramid(nn.Module):
#     def __init__(self, max_level=5):
#         super(LaplacianPyramid, self).__init__()
#         self.gaussian_conv = GaussianConv()
#         self.max_level = max_level
#
#     def forward(self, X):
#         t_pyr = []
#         current = X
#         for level in range(self.max_level):
#             t_guass = self.gaussian_conv(current)
#             t_diff = current - t_guass
#             t_pyr.append(t_diff)
#             current = F.avg_pool2d(t_guass, 2)
#         t_pyr.append(current)
#
#         return t_pyr
#
# class LaplacianLoss(nn.Module):
#     def __init__(self):
#         super(LaplacianLoss, self).__init__()
#
#         self.criterion = nn.L1Loss()
#         self.lap = LaplacianPyramid()
#
#     def forward(self, x, y):
#         x_lap, y_lap = self.lap(x), self.lap(y)
#         return sum(self.criterion(a, b) for a, b in zip(x_lap, y_lap))
def train():
    batch_size = 1
    total_step = 100
    num_workers = 0
    crop = False
    if crop:
        W = 256
        H = 256
        
    else:
        W = 448
        H = 256
    lr = 1e-4
    criteration = LapLoss()
    vimo_dataset = Vimeo(base_dir=vimo_data_dir)
    train_loader = DataLoader(vimo_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True)
    shape= [batch_size,3,H,W]
    model = Main_net(shape).cuda().train(True)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=4e-4)

    for step in range(total_step):
        total_loss = 0
        for ix,data in enumerate(train_loader):

            img1,img2 ,tar= data
            img1 = img1.cuda()
            img2 = img2.cuda()
            tar  = tar.cuda()
            img_out  = model(img1,img2)
            # loss = torch.nn.functional.l1_loss(img_out,tar)
            loss = criteration(img_out, tar)

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            if ix %5==0:
                print('data idx:' +' lr  :'+str(lr)+'  epoch:  ' +str(ix)+'  /  '+str(len(train_loader)))
                print('loss value :', loss.item())
            total_loss+=loss
        f.write('epoch:  '+str(step)+'    avg loss   :'+str(total_loss.item()/len(train_loader)))
        f.write('\n')
        print('epoch:  '+str(step)+'    avg loss   :'+str(total_loss.item()/len(train_loader)))
        if (step+1)%3==0:
            torch.save(model,'./weights/'+'model_weight_'+str(step+1)+'.pth')


if __name__=='__main__':
    train()