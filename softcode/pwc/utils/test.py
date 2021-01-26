# import torch
# # self.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.load('weights/network-default.pytorch').items()})
# for each in torch.load('../weights/network-default.pytorch'):
#     print(each)
# print(50*'-')
# for strKey, tenWeight in torch.load('../weights/network-default.pytorch').items():
#     print(strKey)
#     # print(tenWeight)
# new_state_dict = { strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.load('../weights/network-default.pytorch').items()}
# print(50*'-')
# for k,v in new_state_dict.items():
#     print(k)
#
import numpy as np
dd = np.cumsum([128,128,96,64,32])
for each in dd:
    print(each)