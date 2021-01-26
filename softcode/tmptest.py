#临时测试使用

import os
import shutil
base_dir = 'D:/dataset/vimeo_triplet'
def generate_sub_dataset():

    sub_train_tri_ls = os.path.join(base_dir, 'sub_test_tri_ls.txt')
    tri_names = ['im1.png', 'im2.png', 'im3.png']
    sub_dataset_root = 'D:/dataset'
    with open(sub_train_tri_ls, 'r') as f:
        sub_train_tri_ls = f.readlines()
        for each in sub_train_tri_ls:

            each = each.strip('\n')
            new_img_dir = os.path.join(sub_dataset_root, 'vimeo_sub_triplet', 'sequences', each)
            if not os.path.exists(new_img_dir):
                os.makedirs(new_img_dir)
            for name in tri_names:
                img_dir = os.path.join(base_dir, 'sequences', each, name)
                dest = os.path.join(sub_dataset_root, 'vimeo_sub_triplet', 'sequences', each, name)
                print(dest)
                shutil.copy(img_dir,dest)
if __name__=='__main__':
    generate_sub_dataset()