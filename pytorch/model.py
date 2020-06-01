# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os






class Semantic3D_1(nn.Module):
    r"""
    """

    def __init__(self, in_channels, out_channels, kernel_size, with_se=False, normalize=True, num_cls=2):
        super().__init__()

        voxel_layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm3d(out_channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
            nn.MaxPool3d(3),
            nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm3d(out_channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
            nn.MaxPool3d(3),
         ]
        self.voxel_layers = nn.Sequential(*voxel_layers)



        self.fc_lyaer = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(64, 2, kernel_size=1),
        )
        self.fc = nn.Sequential(
            nn.Linear(864, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, num_cls, bias=False),
        )

    def forward(self, inputs):
        
        features = self.voxel_layers (inputs)        
        # flatten
        
        print(features.size())
        features = torch.flatten(features, start_dim=1)
        print(features.size())
        pred = self.fc(features)
        pred = F.log_softmax(pred, dim=1)
        print(pred.size())
#        pred = pred.transpose(1, 2)
         



        return pred


if __name__ == '__main__':
    backbone_net = Semantic3D_1(in_channels = 1,out_channels =32, kernel_size = 3 ).cuda()
    print(backbone_net)
    backbone_net.eval()
    out = backbone_net(torch.rand(2, 1, 32,32,32).cuda())
    check = out.cpu().data.numpy()
    check = np.argmax(check, axis=1)
    print(check)
    print (out)
    # for key in sorted(out.keys()):
    #     print(key, '\t', out[key].shape)
