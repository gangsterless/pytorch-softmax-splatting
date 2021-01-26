from abc import abstractmethod
from torch.utils.data import Dataset
import imageio
import cv2
import numpy as np
import random
from pathlib import Path
from itertools import islice
import torch
from utils.flow_utils import load_flow
from utils.constv import ConstV

class StaticRandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        h, w = image_size
        self.h1 = random.randint(0, h - self.th)
        self.w1 = random.randint(0, w - self.tw)

    def __call__(self, img):
        return img[self.h1:(self.h1 + self.th), self.w1:(self.w1 + self.tw), :]


class StaticCenterCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        self.h, self.w = image_size

    def __call__(self, img):
        return img[(self.h - self.th) // 2:(self.h + self.th) // 2, (self.w - self.tw) // 2:(self.w + self.tw) // 2, :]


def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


class BaseDataset(Dataset):
    @abstractmethod
    def __init__(self):
        pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        img1_path, img2_path, flow_path = self.samples[idx]
        img1, img2 = map(imageio.imread, (img1_path, img2_path))

        flow = load_flow(flow_path)

        if self.color == 'gray':
            img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis]
            img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis]

        if self.crop_shape is not None:
            # cropper = StaticRandomCrop(img1.shape[:2], self.crop_shape) if self.cropper == 'random' else StaticCenterCrop(img1.shape[:2], self.crop_shape)
            # print(cropper)
            cropper = StaticRandomCrop(img1.shape[:2], self.crop_shape)
            img1 = cropper(img1)
            img2 = cropper(img2)
            flow = cropper(flow)
        if self.resize_shape is not None:
            resizer = partial(cv2.resize, dsize=(0, 0), dst=self.resize_shape)
            images = list(map(resizer, images))
            flow = resizer(flow)
        elif self.resize_scale is not None:
            resizer = partial(cv2.resize, dsize=(0, 0), fx=self.resize_scale, fy=self.resize_scale)
            images = list(map(resizer, images))
            flow = resizer(flow)

        # if self.train_or_test == 'test':
        #     H, W = img1.shape[:2]
        #     img1, img2 = images
        #     img1 = np.pad(img1, ((0, (64 - H % 64) if H % 64 else 0), (0, (64 - W % 64) if H % 64 else 0), (0, 0)),
        #                   mode='constant')
        #     img1 = img1.transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)
        #     img2 = np.pad(img2, ((0, (64 - H % 64) if H % 64 else 0), (0, (64 - W % 64) if H % 64 else 0), (0, 0)),
        #                   mode='constant')
        #     img2 = img2.transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)
        #     images = [img1, img2]
        # 在这里标准化一下
        # img_pair = [img1,img2]
        # rgb_mean = x.contiguous().view(x.size()[:2] + (-1,)).mean(dim=-1).view(x.size()[:2] + (1, 1, 1,))
        # x = (x - rgb_mean) / args.rgb_max
        # img1,img2 = self.pair_norm(np.array([img1,img2]))
        if ConstV.train_or_test=='train':
            img1 = np.ascontiguousarray(img1.transpose(2, 0, 1).astype(np.float32))*(1.0/255.0)
            img2 = np.ascontiguousarray(img2.transpose(2, 0, 1).astype(np.float32))*(1.0/255.0)
        else:

            img1 = np.ascontiguousarray(img1.transpose(2, 0, 1).astype(np.float32))*(1.0/255.0)
            img2 = np.ascontiguousarray(img2.transpose(2, 0, 1).astype(np.float32))*(1.0/255.0)

        # images = np.array(images).transpose(3,0,1,2)
        flow = flow.transpose(2, 0, 1)
        flow.astype('float32')
        img1 = torch.from_numpy(img1.astype(np.float32))
        img2 = torch.from_numpy(img2.astype(np.float32))
        flow = torch.from_numpy(flow.astype(np.float32))
        # images = torch.from_numpy(images.astype(np.float32))
        # flow = torch.from_numpy(flow.astype(np.float32))

        return img1, img2, flow
    def pair_norm(self,imgls):
        R_mean = np.mean(imgls[:,:,:,0])
        R_std = np.std(imgls[:,:,:,0])
        G_mean = np.mean(imgls[:,:,:,1])
        G_std = np.std(imgls[:,:,:,1])
        B_mean = np.mean(imgls[:,:,:,2])
        B_std = np.std(imgls[:, :, :, 2])
        return (imgls[0]-(R_mean,G_mean,B_mean))/(R_std,G_std,B_std),(imgls[1]-(R_mean,G_mean,B_mean))/(R_std,G_std,B_std)
    def has_txt(self):

        p = Path(self.dataset_dir) / (self.train_or_test + '.txt')
        print('self.dataset_dir :', p)
        self.samples = []
        with open(p, 'r') as f:
            for i in f.readlines():
                img1, img2, flow = i.split(',')
                flow = flow.strip()
                self.samples.append((img1, img2, flow))

    def split(self, samples):
        p = Path(self.dataset_dir)
        if ConstV.train_or_test=='train':
            test_ratio = 0.0
        else:
            test_ratio = 1.0
        random.shuffle(samples)
        idx = int(len(samples) * (1 - test_ratio))
        train_samples = samples[:idx]
        test_samples = samples[idx:]

        with open(p / 'train.txt', 'w') as f: f.writelines((','.join(i) + '\n' for i in train_samples))
        with open(p / 'test.txt', 'w') as f: f.writelines((','.join(i) + '\n' for i in test_samples))

        self.samples = train_samples if self.train_or_test == 'train' else test_samples


class Sintel(BaseDataset):

    def __init__(self, dataset_dir, train_or_test, mode='final', color='rgb', cropper='random', rgb_mean=True,
                 crop_shape=None, resize_shape=None, resize_scale=None):
        super(Sintel, self).__init__()
        self.mode = mode
        self.color = color
        self.cropper = cropper
        self.crop_shape = crop_shape
        self.resize_shape = resize_shape
        self.resize_scale = resize_scale

        self.dataset_dir = dataset_dir
        self.train_or_test = train_or_test
        p = Path(dataset_dir) / (train_or_test + '.txt')
        # if p.exists():
        #     print('exist')
        #     self.has_txt()
        # else:
        # print('not exist')
        self.has_no_txt()

    def has_no_txt(self):
        p = Path(self.dataset_dir)
        p_img = p / 'training' / self.mode
        p_flow = p / 'training/flow'
        samples = []

        collections_of_scenes = sorted(map(str, p_img.glob('**/*.png')))
        from itertools import groupby
        collections = [list(g) for k, g in groupby(collections_of_scenes, lambda x: x.split('\\')[-2])]

        samples = [(*i, i[0].replace(self.mode, 'flow').replace('.png', '.flo')) for collection in collections for i in
                   window(collection, 2)]

        self.split(samples)


if __name__ == '__main__':

    dataset_dir = '/home/zhangyuantong/dataset/Sintel'
    train_or_test = 'train'

    sintel_dataset = Sintel(dataset_dir=dataset_dir, train_or_test=train_or_test)
    print('len', len(sintel_dataset))
    for idx, data in enumerate(sintel_dataset):
        # print(len(imgls))
        img1, img2, flowls = data
        print(img1.shape)
        print(flowls.shape)