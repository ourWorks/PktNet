from pkgutil import get_loader
from typing import Union
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

class PacketUnit(torch.nn.Module):

    def __init__(self, num_classes, feature_length_list):

        super(PacketUnit, self).__init__()
        # 不变的参数
        self.num_classes = num_classes
        self.linList = nn.ModuleList(
                [
                    # nn.LayerNorm([feature_length_list[idx]]),
                    nn.Linear(feature_length_list[idx], 64)
                    for idx in range(len(feature_length_list))
                ]
            )
        self.layerLin = nn.Linear(len(feature_length_list)*64, 64)
        self.leakyRelu = nn.LeakyReLU()
 
    def forward(self, featureList):

        # 1. 计算并堆叠
        x = torch.cat([self.linList[idx](featureList[idx]) for idx in range(len(featureList))], dim = 1)
        # 2. 计算堆叠之后的线性层
        x = self.layerLin(x)
        x = self.leakyRelu(x)
        return x

class DeepSearchUnit(torch.nn.Module):

    def __init__(self, channel_num):

        super(DeepSearchUnit, self).__init__()

        num = channel_num
        self.conv1 = nn.Sequential(nn.Conv3d(num, num, kernel_size=1),
                                    nn.Conv3d(num, num, kernel_size=3, padding=1),
                                    nn.Conv3d(num, num, kernel_size=1))
        self.conv2 = nn.Sequential(nn.Conv3d(num, num, kernel_size=1),
                                    nn.Conv3d(num, num, kernel_size=5, padding=2),
                                    nn.Conv3d(num, num, kernel_size=1))
        self.conv3 = nn.Sequential(nn.Conv3d(num, num, kernel_size=1), 
                                    nn.Conv3d(num, num, kernel_size=3, padding=1),
                                    nn.Conv3d(num, num, kernel_size=1))

        self.relu = nn.ReLU()
 
    def forward(self, x):

        x = self.relu(x + self.conv1(x))
        x = self.relu(x + self.conv2(x))
        x = self.relu(x + self.conv3(x))

        return x

class FeatureExtraLayer(nn.Module):
    def __init__(self):
        super(FeatureExtraLayer, self).__init__()

        # 3D卷积层
        self.conv1 = nn.Sequential(nn.Conv3d(1, 64, kernel_size=7, padding=1, stride=2))
        # self.conv1_1 = nn.Sequential(nn.Conv3d(1, 16, kernel_size=7, padding=1, dilation=2), nn.ReLU(), nn.AdaptiveMaxPool3d((46, 46, 46)))
        # self.conv1_1_1 = nn.Sequential(nn.Conv3d(1, 16, kernel_size=7, padding=1, dilation=3), nn.ReLU(), nn.AdaptiveMaxPool3d((46, 46, 46)))
        
        # self.deepSearchUnit_1 = DeepSearchUnit(64)

        self.change_conv1 = nn.Sequential(nn.Conv3d(64, 128, kernel_size=3, padding=1), nn.MaxPool3d(2))

        # self.deepSearchUnit_2 = DeepSearchUnit(128)

        self.change_conv2 = nn.Sequential(nn.Conv3d(128, 256, kernel_size=3, padding=1), nn.MaxPool3d(2))

        # self.deepSearchUnit_3 = DeepSearchUnit(256)

        self.change_conv3 = nn.Sequential(nn.Conv3d(256, 512, kernel_size=3, padding=1), nn.MaxPool3d(2))

        self.fc1 = nn.Linear(512 * 5 * 5 * 5, 512)

        # 激活函数
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()

    def forward(self, x):

        x = self.relu(self.conv1(x))
        # x = self.deepSearchUnit_1(x)

        x = self.relu(self.change_conv1(x))
        # x = self.deepSearchUnit_2(x)
        
        x = self.gelu(self.change_conv2(x))
        # x = self.deepSearchUnit_3(x)

        x = self.gelu(self.change_conv3(x))
        x = x.view(x.size(0), -1)     
        x = self.relu(self.fc1(x))

        return x

class BaseConvExtraLayer(nn.Module):
    def __init__(self):
        super(BaseConvExtraLayer, self).__init__()

        # 3D卷积层
        self.conv1 = nn.Sequential(nn.Conv3d(1, 64, kernel_size=7, padding=1, stride=2),nn.MaxPool3d(8))
        self.baseConv = nn.Sequential(nn.Conv3d(64, 512, kernel_size=3, padding=1))
        
        self.fc1 = nn.Linear(512 * 5 * 5 * 5, 512)

        # 激活函数
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()

    def forward(self, x):

        x = self.relu(self.conv1(x))
        x = self.gelu(self.baseConv(x))
        x = x.view(x.size(0), -1)     
        x = self.relu(self.fc1(x))

        return x


class PacketLayer(nn.Module):
    def __init__(self, num_classes):
        super(PacketLayer, self).__init__()

        self.channel_num = 512
        self.subsetSize = self.channel_num//5

        self.sizeList1 = [self.subsetSize, self.channel_num-self.subsetSize]
        self.sizeList2 = [self.subsetSize*2, self.channel_num-self.subsetSize*2]
        self.sizeList3 = [self.subsetSize*3, self.channel_num-self.subsetSize*3]
        self.sizeList4 = [self.subsetSize*4, self.channel_num-self.subsetSize*4]

        self.packet1 = PacketUnit(num_classes=num_classes, feature_length_list=self.sizeList1)
        self.packet2 = PacketUnit(num_classes=num_classes, feature_length_list=self.sizeList2)
        self.packet3 = PacketUnit(num_classes=num_classes, feature_length_list=self.sizeList3)
        self.packet4 = PacketUnit(num_classes=num_classes, feature_length_list=self.sizeList4)

        self.lastLin = nn.Linear(258, num_classes)

    def split_list(self, tensor, indexes):
        sum_indexes = [sum(indexes[0:idx]) for idx in range(len(indexes))] + [self.channel_num]
        return [tensor[..., sum_indexes[i]:sum_indexes[i+1]] for i in range(len(sum_indexes)-1)]

    def forward(self, x, age, MMSE):

        a = self.packet1(self.split_list(x, self.sizeList1))        
        b = self.packet2(self.split_list(x, self.sizeList2))        
        c = self.packet3(self.split_list(x, self.sizeList3))        
        d = self.packet4(self.split_list(x, self.sizeList4))
        x = torch.concat([age.unsqueeze(-1).float(), MMSE.unsqueeze(-1).float(), a, b, c, d], dim = 1)
        x = self.lastLin(x)

        return x
     
class BaseModel(nn.Module):

    def __init__(self, num_classes):
        super(BaseModel, self).__init__()

        self.baseConv = BaseConvExtraLayer()
        self.featureExtraLayer = FeatureExtraLayer()
        self.packetLayer = PacketLayer(num_classes)
        self.removePackerLayerLin = nn.Linear(512, num_classes)

    def forward(self, x, age, MMSE):

        # 可以先训练一段时间
        # x = self.featureExtraLayer(x)
        x = self.baseConv(x)

        # x = self.packetLayer(x, age, MMSE)
        x = self.removePackerLayerLin(x)

        return x