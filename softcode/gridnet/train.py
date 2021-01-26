#我们要试试 只用两张图片合成，总应该有一个baseline
from gridnet.single_net import GridNet
import torch
from vimo_dataset import Vimeo
from torch.utils.data import DataLoader
from loss_f import LapLoss
vimo_data_dir = 'D:/dataset/vimeo_triplet'
def EPE(input_flow, target_flow):
    return torch.norm(target_flow-input_flow,p=2,dim=1).mean()
def train():
    batch_size = 1
    total_step = 100
    num_workers = 0
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

    model = GridNet().cuda().train()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=4e-4)

    print('这是  测试  gridnet')
    for step in range(total_step):
        total_loss = 0
        total_epe = 0
        for ix, data in enumerate(train_loader):
            img1, img2, tar = data
            img1 = img1.cuda()
            img2 = img2.cuda()
            tar = tar.cuda()
            img_out = model(img1, img2)
            # loss = torch.nn.functional.l1_loss(img_out,tar)
            loss = criteration(img_out, tar)

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            print()
            print('data idx:' + ' lr  :' + str(lr) + '  epoch:  ' + str(ix) + '  /  ' + str(len(train_loader)))
            print('loss value :', loss.item())
            epe = EPE(img_out,tar)
            print('EPE :',epe.item())
            total_loss += loss
            total_epe+=epe
        # f.write('epoch:  ' + str(step) + '    avg loss   :' + str(total_loss.item() / len(train_loader)))
        # f.write('\n')
        print('epoch:  ' + str(step) + '    avg loss   :' + str(total_loss.item() / len(train_loader)))
        print('epoch:  ' + str(step) + '    avg epe   :' + str(total_epe.item() / len(train_loader)))
if __name__=='__main__':
    train()