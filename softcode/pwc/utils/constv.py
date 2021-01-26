import torch
class ConstV:
    mutil_l = True
    #是否拿出来嵌入整个softsplat的框架，如果嵌入整个框架最后返回的flow的分辨率是不一样的
    integrated = True
    epsilon = 0.02
    q = 0.4
    weightsls = [0.32,0.08,0.02,0.01,0.005]
    dataset_dir = 'D:/dataset/Sintel'
    train_or_test = 'train'
    num_workers = 0
    batch_size = 4
    lr = 1e-4
    weight_decay = 4e-4
    total_step = 100
    my_device = torch.device('cuda')
    num_levels = 7
    output_level = 4
    summary_interval = 2
    lv_chs =[529, 661, 629, 597, 565]
    