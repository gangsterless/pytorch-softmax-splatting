#!/usr/bin/env python

import torch

import getopt
import math
import numpy
import os
import PIL
import PIL.Image
import sys
import torch.nn.functional as F
from pwc.utils.constv import ConstV
import torch.nn as nn
try:
	from .correlation import correlation # the custom cost volume layer
except:
	sys.path.insert(0, './correlation'); import correlation # you should consider upgrading python
# end

##########################################################

assert(int(str('').join(torch.__version__.split('.')[0:2])) >= 13) # requires at least pytorch version 1.3.0

# torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################

arguments_strModel = 'default'
arguments_strFirst = './images/first.png'
arguments_strSecond = './images/second.png'
arguments_strOut = './out.flo'

for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [ strParameter[2:] + '=' for strParameter in sys.argv[1::2] ])[0]:
	if strOption == '--model' and strArgument != '': arguments_strModel = strArgument # which model to use
	if strOption == '--first' and strArgument != '': arguments_strFirst = strArgument # path to the first frame
	if strOption == '--second' and strArgument != '': arguments_strSecond = strArgument # path to the second frame
	if strOption == '--out' and strArgument != '': arguments_strOut = strArgument # path to where the output should be stored
# end

##########################################################

backwarp_tenGrid = {}
backwarp_tenPartial = {}

def backwarp(tenInput, tenFlow):
	if str(tenFlow.shape) not in backwarp_tenGrid:
		tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3]).view(1, 1, 1, -1).expand(-1, -1, tenFlow.shape[2], -1)
		tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).view(1, 1, -1, 1).expand(-1, -1, -1, tenFlow.shape[3])

		backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([ tenHor, tenVer ], 1).cuda()
	# end

	if str(tenFlow.shape) not in backwarp_tenPartial:
		backwarp_tenPartial[str(tenFlow.shape)] = tenFlow.new_ones([ tenFlow.shape[0], 1, tenFlow.shape[2], tenFlow.shape[3] ])
	# end

	tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)
	tenInput = torch.cat([ tenInput, backwarp_tenPartial[str(tenFlow.shape)] ], 1)

	tenOutput = torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=False)

	tenMask = tenOutput[:, -1:, :, :]; tenMask[tenMask > 0.999] = 1.0; tenMask[tenMask < 1.0] = 0.0

	return tenOutput[:, :-1, :, :] * tenMask
# end

##########################################################

class Network(torch.nn.Module):
	def __init__(self):
		super(Network, self).__init__()
		#

		class Extractor(torch.nn.Module):
			def __init__(self):
				super(Extractor, self).__init__()

				self.netOne = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netTwo = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netThr = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netFou = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netFiv = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netSix = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=128, out_channels=196, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)
			# end

			def forward(self, tenInput):
				tenOne = self.netOne(tenInput)
				tenTwo = self.netTwo(tenOne)
				tenThr = self.netThr(tenTwo)
				tenFou = self.netFou(tenThr)
				tenFiv = self.netFiv(tenFou)
				tenSix = self.netSix(tenFiv)

				return [ tenOne, tenTwo, tenThr, tenFou, tenFiv, tenSix ]
			# end
		# end

		class Decoder(torch.nn.Module):
			def __init__(self, intLevel):
				super(Decoder, self).__init__()
				self.intLevel = intLevel
				intPrevious = [ None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None ][intLevel + 1]
				intCurrent = [ None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None ][intLevel + 0]

				if intLevel < 6: self.netUpflow = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)
				if intLevel < 6: self.netUpfeat = torch.nn.ConvTranspose2d(in_channels=intPrevious + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=4, stride=2, padding=1)
				if intLevel < 6: self.fltBackwarp = [ None, None, None, 5.0, 2.5, 1.25, 0.625, None ][intLevel + 1]

				self.netOne = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netTwo = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netThr = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128 + 128, out_channels=96, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netFou = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96, out_channels=64, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netFiv = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netSix = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=3, stride=1, padding=1)
				)
			# end

			def forward(self, tenFirst, tenSecond, objPrevious):
				tenFlow = None
				tenFeat = None

				if objPrevious is None:
					tenFlow = None
					tenFeat = None

					tenVolume = torch.nn.functional.leaky_relu(input=correlation.FunctionCorrelation(tenFirst=tenFirst, tenSecond=tenSecond), negative_slope=0.1, inplace=False)

					tenFeat = torch.cat([ tenVolume ], 1)


				elif objPrevious is not None:
					tenFlow = self.netUpflow(objPrevious['tenFlow'])
					tenFeat = self.netUpfeat(objPrevious['tenFeat'])

					tenVolume = torch.nn.functional.leaky_relu(input=correlation.FunctionCorrelation(tenFirst=tenFirst, tenSecond=backwarp(tenInput=tenSecond, tenFlow=tenFlow * self.fltBackwarp)), negative_slope=0.1, inplace=False)
					#corr 对应  tenVolume tenFirst对应x1  flow 对应tenflow
					tenFeat = torch.cat([ tenVolume, tenFirst, tenFlow, tenFeat ], 1)


				# end
				# #如果是最后一层
				# if self.intLevel==6:


				tenFeat = torch.cat([ self.netOne(tenFeat), tenFeat ], 1)
				tenFeat = torch.cat([ self.netTwo(tenFeat), tenFeat ], 1)
				tenFeat = torch.cat([ self.netThr(tenFeat), tenFeat ], 1)
				tenFeat = torch.cat([ self.netFou(tenFeat), tenFeat ], 1)
				tenFeat = torch.cat([ self.netFiv(tenFeat), tenFeat ], 1)

				tenFlow = self.netSix(tenFeat)

				return {
					'tenFlow': tenFlow,
					'tenFeat': tenFeat
				}
			# end
		# end

		class Refiner(torch.nn.Module):
			def __init__(self,in_ch):
				super(Refiner, self).__init__()

				self.netMain = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=in_ch, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1),
					# nn.BatchNorm2d(128),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2, dilation=2),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=8, dilation=8),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=16, dilation=16),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1)
				)
			# end

			def forward(self, tenInput):
				return self.netMain(tenInput)
			# end
		# end

		# 初始化 context refiner
		# self.context_refiners = []
		# for l, ch in enumerate(ConstV.lv_chs):
		# 	layer = Refiner(ch).to(ConstV.my_device)
		# 	# self.add_module(f'ContextNetwork(Lv{l})', layer)
		# 	self.context_refiners.append(layer)
		self.netExtractor = Extractor()

		self.netTwo = Decoder(2)
		self.netThr = Decoder(3)
		self.netFou = Decoder(4)
		self.netFiv = Decoder(5)
		self.netSix = Decoder(6)

		self.netRefiner = Refiner(565)


		#一定用高斯初始化
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				if m.bias is not None: nn.init.uniform_(m.bias)
				nn.init.xavier_uniform_(m.weight)

			if isinstance(m, nn.ConvTranspose2d):
				if m.bias is not None: nn.init.uniform_(m.bias)
				nn.init.xavier_uniform_(m.weight)
		# print('http://content.sniklaus.com/github/pytorch-pwc/network-' + arguments_strModel + '.pytorch')
		#不清楚为啥 不能load 似乎只会起反作用，应该是估计的光流没有起到很好的效果？？？
		#看来输入的光流必须除以255否则不顶用 这根预训练模型绝对有关系
		self.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.load('pwc/weights/network-default.pytorch').items()})
	# end

	def forward(self, tenFirst, tenSecond):
		tenFirst = self.netExtractor(tenFirst)
		tenSecond = self.netExtractor(tenSecond)
		if ConstV.mutil_l:
			objEstimate = self.netSix(tenFirst[-1], tenSecond[-1], None)
			# objEstimate['tneFlow'] = [2,2,6,7]

			flow_l1 = objEstimate['tenFlow']

			objEstimate = self.netFiv(tenFirst[-2], tenSecond[-2], objEstimate)
			# objEstimate['tneFlow'] = [2,2,12,14] [2,529,6,7]

			flow_l2 = objEstimate['tenFlow']
			# flow_l2 = objEstimate['tenFlow'] + self.context_refiners[1](objEstimate['tenFeat'])
			objEstimate = self.netFou(tenFirst[-3], tenSecond[-3], objEstimate)

			# objEstimate['tneFlow'] = [2,2,24,28] [2, 661, 12, 14]
			flow_l3 = objEstimate['tenFlow']
			# flow_l3 = objEstimate['tenFlow'] + self.context_refiners[2](objEstimate['tenFeat'])
			objEstimate = self.netThr(tenFirst[-4], tenSecond[-4], objEstimate)

			# objEstimate['tneFlow'] = [2,2,48,56]
			flow_l4 = objEstimate['tenFlow']
			# flow_l4 = objEstimate['tenFlow'] + self.context_refiners[3](objEstimate['tenFeat'])
			objEstimate = self.netTwo(tenFirst[-5], tenSecond[-5], objEstimate)
		# shapels.append(objEstimate['tenFeat'].shape[1])
		# print(shapels)
			flow_l5 = objEstimate['tenFlow'] + self.netRefiner(objEstimate['tenFeat'])
		# objEstimate['tneFlow'] = [2,2,96,112]
		# 貌似是把最后一层的尺寸强行拉到 原始输入尺寸 即扩大四倍 然后数值也乘4
		# if l == args.output_level:
		#                 flow = F.upsample(flow, scale_factor = 2 ** (args.num_levels - args.output_level - 1), mode = 'bilinear') * 2 ** (args.num_levels - args.output_level - 1)
		#                 flows.append(flow)

		# flow_l5 = F.upsample(flow_l5, scale_factor=2 ** (2), mode='bilinear')
			if ConstV.integrated:
				return flow_l5
			if ConstV.train_or_test == 'train':
				#后面会× 20的，不用再乘了
				flow_l5 = F.upsample(flow_l5, scale_factor=2 ** (2), mode='bilinear')
				return [flow_l1, flow_l2, flow_l3, flow_l4, flow_l5]
			else:
				return flow_l5
		else:
		# shapels = []
			objEstimate = self.netSix(tenFirst[-1], tenSecond[-1], None)
			# objEstimate['tneFlow'] = [2,2,6,7]

			objEstimate = self.netFiv(tenFirst[-2], tenSecond[-2], objEstimate)
			# objEstimate['tneFlow'] = [2,2,12,14] [2,529,6,7]

			objEstimate = self.netFou(tenFirst[-3], tenSecond[-3], objEstimate)
			# shapels.append(objEstimate['tenFeat'].shape[1])

			objEstimate = self.netThr(tenFirst[-4], tenSecond[-4], objEstimate)

			# flow_l4 = objEstimate['tenFlow'] + self.context_refiners[3](objEstimate['tenFeat'])
			objEstimate = self.netTwo(tenFirst[-5], tenSecond[-5], objEstimate)


			# flow_coarse + flow_fine
			return objEstimate['tenFlow'] + self.netRefiner(objEstimate['tenFeat'])
	# end
# end

# netNetwork = None
#
# ##########################################################
#
# def estimate(tenFirst, tenSecond):
# 	global netNetwork
#
# 	if netNetwork is None:
# 		netNetwork = Network().cuda().eval()
# 	# end
#
# 	assert(tenFirst.shape[1] == tenSecond.shape[1])
# 	assert(tenFirst.shape[2] == tenSecond.shape[2])
#
# 	intWidth = tenFirst.shape[2]
# 	intHeight = tenFirst.shape[1]
#
# 	assert(intWidth == 1024) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
# 	assert(intHeight == 436) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
#
# 	tenPreprocessedFirst = tenFirst.cuda().view(1, 3, intHeight, intWidth)
# 	tenPreprocessedSecond = tenSecond.cuda().view(1, 3, intHeight, intWidth)
#
# 	intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
# 	intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))
#
# 	tenPreprocessedFirst = torch.nn.functional.interpolate(input=tenPreprocessedFirst, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
# 	tenPreprocessedSecond = torch.nn.functional.interpolate(input=tenPreprocessedSecond, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
#
# 	tenFlow = 20.0 * torch.nn.functional.interpolate(input=netNetwork(tenPreprocessedFirst, tenPreprocessedSecond), size=(intHeight, intWidth), mode='bilinear', align_corners=False)
#
# 	tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
# 	tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)
#
# 	return tenFlow[0, :, :, :].cpu()
# # end
#
# ##########################################################
#
# if __name__ == '__main__':
# 	tenFirst = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(PIL.Image.open(arguments_strFirst))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
# 	tenSecond = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(PIL.Image.open(arguments_strSecond))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
# 	#(3, 436, 1024)
# 	tenOutput = estimate(tenFirst, tenSecond)
#
# 	objOutput = open(arguments_strOut, 'wb')
#
# 	numpy.array([ 80, 73, 69, 72 ], numpy.uint8).tofile(objOutput)
# 	numpy.array([ tenOutput.shape[2], tenOutput.shape[1] ], numpy.int32).tofile(objOutput)
# 	numpy.array(tenOutput.detach().numpy().transpose(1, 2, 0), numpy.float32).tofile(objOutput)
#
# 	objOutput.close()
# # end