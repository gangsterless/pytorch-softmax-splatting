from torch.utils.data import Dataset

import numpy as np
import imageio
import torch
import random
class StaticRandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        h, w = image_size
        self.h1 = random.randint(0, h - self.th)
        self.w1 = random.randint(0, w - self.tw)

    def __call__(self, img):
        return img[self.h1:(self.h1+self.th), self.w1:(self.w1+self.tw),:]


class StaticCenterCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        self.h, self.w = image_size
    def __call__(self, img):
        return img[(self.h-self.th)//2:(self.h+self.th)//2, (self.w-self.tw)//2:(self.w+self.tw)//2,:]
class Vimeo(Dataset):
    def __init__(self, base_dir, augument=False, transform=None):

        self.width = 448
        self.height = 256
        self.augument = augument
        self.transform = transform
        self.base_dir = base_dir
        self.crop_shape = (256,256)
        self.cropper = 'random'
        self.files_dir_ls = self.get_file_ls()

    def get_file_ls(self):
        tri_img = ['im1.png', 'im2.png', 'im3.png']
        total_res_ls = []
        with open(self.base_dir+'/'+'tri_trainlist.txt','r') as f:
            each_ls = f.readlines()
            for each in each_ls:
                each =  each.strip('\n')
                tmp_p_ls = []
                for sub in tri_img:
                    tmp_p_ls.append(self.base_dir + '/' + 'sequences' + '/' + each + '/' + sub)
                total_res_ls.append(tmp_p_ls)
        return total_res_ls

    def __getitem__(self, idx):
        # files_dir_ls 包含了三张图片
        # device = torch.device(type='cuda')
        img1_path = self.files_dir_ls[idx][0]
        img2_path = self.files_dir_ls[idx][-1]
        target_path = self.files_dir_ls[idx][1]

        img1, img2,target = map(imageio.imread, (img1_path, img2_path,target_path))
        # 如果是要反向
        H, W = img1.shape[:2]
        # pad to 64
        img1 = np.pad(img1, ((0, (64 - H % 64) if H % 64 else 0), (0, (64 - W % 64) if H % 64 else 0), (0, 0)),
                      mode='constant')
        img2 = np.pad(img2, ((0, (64 - H % 64) if H % 64 else 0), (0, (64 - W % 64) if H % 64 else 0), (0, 0)),
                      mode='constant')
        target = np.pad(target, ((0, (64 - H % 64) if H % 64 else 0), (0, (64 - W % 64) if H % 64 else 0), (0, 0)),
                      mode='constant')
        images = [img1, img2, target]
        if self.augument:
            cropper = StaticRandomCrop(img1.shape[:2],
                                       self.crop_shape) if self.cropper == 'random' else StaticCenterCrop(
                img1.shape[:2], self.crop_shape)
            # print(cropper)
            images = list(map(cropper, images))
        images = [torch.from_numpy( each.transpose(2, 0, 1).astype(np.float32)) * (1.0 / 255.0) for each in images]

        # img1 = img1.transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)
        # img2 = img2.transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)
        # target = target.transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)



        return images
        # x = torch.Tensor(x).to(device)

    def __len__(self):
        return len(self.files_dir_ls)