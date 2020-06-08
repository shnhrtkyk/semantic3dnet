

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, with_se=False, normalize=True, num_cls=3, num_scale=5):
        super().__init__()

        self.num_scale = num_scale

        voxel_layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm3d(out_channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
            nn.MaxPool3d(3),
            nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm3d(out_channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
            nn.MaxPool3d(3),
            nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm3d(out_channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
            nn.MaxPool3d(3),
        ]
        self.voxel_layers =  nn.Sequential(*voxel_layers).cuda()


    def forward(self, inputs):
        # print(self.voxel_layers)
        # print(inputs.size())
        features = self.voxel_layers(inputs)
        return features



class Semantic3D_1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, with_se=False, normalize=True, num_cls=3, num_scale=5):
        super().__init__()

        self.num_scale = num_scale
        self.convblocks_1 = ConvBlock(in_channels, out_channels, kernel_size).cuda()
        self.convblocks_2 = ConvBlock(in_channels, out_channels, kernel_size).cuda()
        self.convblocks_3 = ConvBlock(in_channels, out_channels, kernel_size).cuda()
        self.convblocks_4 = ConvBlock(in_channels, out_channels, kernel_size).cuda()
        self.convblocks_5 = ConvBlock(in_channels, out_channels, kernel_size).cuda()
        
        # for i in range(num_scale):
        #     self.convblocks.append(ConvBlock(in_channels, out_channels, kernel_size).cuda())

        # voxel_layers = [
        #     nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2),
        #     nn.BatchNorm3d(out_channels, eps=1e-4),
        #     nn.LeakyReLU(0.1, True),
        #     nn.MaxPool3d(3),
        #     nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2),
        #     nn.BatchNorm3d(out_channels, eps=1e-4),
        #     nn.LeakyReLU(0.1, True),
        #     nn.MaxPool3d(3),
        #     nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2),
        #     nn.BatchNorm3d(out_channels, eps=1e-4),
        #     nn.LeakyReLU(0.1, True),
        #     nn.MaxPool3d(3),
        # ]
        # self.voxel_layers = []
        # self.voxel_layers.append(nn.Sequential(*voxel_layers).cuda())
        # if (self.num_scale > 2):
        #     for i in range(self.num_scale - 1):
        #         print(i)
        #         self.voxel_layers.append(nn.Sequential(*voxel_layers).cuda())


        concate_layers = [
            nn.Conv3d(out_channels * num_scale, out_channels, 1, stride=1),
            nn.BatchNorm3d(out_channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
        ]
        self.concate_layers = nn.Sequential(*concate_layers)

        # self.fc_lyaer = nn.Sequential(
        #     nn.Conv1d(64, 64, kernel_size=1, bias=False),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(True),
        #     nn.Dropout(0.5),
        #     nn.Conv1d(64, 2, kernel_size=1),
        # )
        self.fc = nn.Sequential(
            nn.Linear(32, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(32, num_cls, bias=False),
            nn.Softmax(1),
        )

    def forward(self, inputs):
        # print(self.convblocks_1)
        print(inputs.size())

        features = self.convblocks_1(inputs)
        features = torch.cat((features, self.convblocks_2(inputs)), axis = 1)
        features = torch.cat((features, self.convblocks_3(inputs)), axis = 1)
        features = torch.cat((features, self.convblocks_4(inputs)), axis = 1)
        features = torch.cat((features, self.convblocks_5(inputs)), axis = 1)
                # print(features.size())
        features = self.concate_layers(features)

        # print(features.size())
        features = torch.flatten(features, start_dim=1)
        # print(features.size())
        pred = self.fc(features)
        # pred = F.log_softmax(pred, dim=1)
        # print(pred.size())
                #        pred = pred.transpose(1, 2)

        return pred


if __name__ == '__main__':
    backbone_net = Semantic3D_1(in_channels=1, out_channels=32, kernel_size=3).cuda()

    print(backbone_net)
    backbone_net.eval()
    out = backbone_net(torch.rand(2, 1, 32, 32, 32).cuda())
    check = out.cpu().data.numpy()
    check = np.argmax(check, axis=1)
    print(check)
    print (out)

    # for key in sorted(out.keys()):
    #     print(key, '\t', out[key].shape)
