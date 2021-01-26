import torch
import torch.nn as nn

from gridnet.modules import LateralBlock, DownSamplingBlock, UpSamplingBlock
class GridNet(nn.Module):
    def __init__(self, out_chs=3, grid_chs=[32, 64, 96]):
        # n_row = 3, n_col = 6, n_chs = [32, 64, 96]):
        super().__init__()

        self.n_row = 3
        self.n_col = 6
        self.n_chs = grid_chs
        assert len(grid_chs) == self.n_row, 'should give num channels for each row (scale stream)'

        self.lateral_init = LateralBlock(70, self.n_chs[0])

        for r, n_ch in enumerate(self.n_chs):
            for c in range(self.n_col - 1):

                setattr(self, f'lateral_{r}_{c}', LateralBlock(n_ch, n_ch))

        for r, (in_ch, out_ch) in enumerate(zip(self.n_chs[:-1], self.n_chs[1:])):
            for c in range(int(self.n_col / 2)):
                # 00 10 要特殊设置
                if r==0 and c==0:
                    setattr(self, f'down_{r}_{c}', LateralBlock(128, out_ch))
                elif r==1 and c==0:
                    setattr(self, f'down_{r}_{c}', LateralBlock(192, out_ch))
                else:
                    setattr(self, f'down_{r}_{c}', DownSamplingBlock(in_ch, out_ch))

        for r, (in_ch, out_ch) in enumerate(zip(self.n_chs[1:], self.n_chs[:-1])):
            for c in range(int(self.n_col / 2)):
                setattr(self, f'up_{r}_{c}', UpSamplingBlock(in_ch, out_ch))

        self.lateral_final = LateralBlock(self.n_chs[0], out_chs)

    def forward(self, x_l1,x_l2,x_l3):
        # torch.Size([2, 32, 328, 448])
        state_00 = self.lateral_init(x_l1)
        #torch.Size([2, 64, 164, 224])
        state_10 = self.down_0_0(x_l2)
        #torch.Size([2, 96, 82, 112])
        state_20 = self.down_1_0(x_l3)
        #01的输入换成 warp的结果
        # torch.Size([2, 32, 164, 224])
        state_01 = self.lateral_0_0(state_00)
        state_11 = self.down_0_1(state_01) + self.lateral_1_0(state_10)
        state_21 = self.down_1_1(state_11) + self.lateral_2_0(state_20)

        state_02 = self.lateral_0_1(state_01)
        state_12 = self.down_0_2(state_02) + self.lateral_1_1(state_11)
        state_22 = self.down_1_2(state_12) + self.lateral_2_1(state_21)

        state_23 = self.lateral_2_2(state_22)
        state_13 = self.up_1_0(state_23) + self.lateral_1_2(state_12)
        state_03 = self.up_0_0(state_13) + self.lateral_0_2(state_02)

        state_24 = self.lateral_2_3(state_23)
        state_14 = self.up_1_1(state_24) + self.lateral_1_3(state_13)
        state_04 = self.up_0_1(state_14) + self.lateral_0_3(state_03)

        state_25 = self.lateral_2_4(state_24)
        state_15 = self.up_1_2(state_25) + self.lateral_1_4(state_14)
        state_05 = self.up_0_2(state_15) + self.lateral_0_4(state_04)
        return self.lateral_final(state_05)
if __name__=='__main__':
    W = 448
    H = 328
    N = 2
    x_l1 = torch.rand(size=(N, 70, H, W)).cuda()
    x_l2 = torch.rand(size=(N, 128, H//2, W//2)).cuda()
    x_l3 = torch.rand(size=(N, 192, H//4, W//4)).cuda()
    model = GridNet(out_chs=3).cuda()
    res = model(x_l1,x_l2,x_l3)
    print(res.shape)