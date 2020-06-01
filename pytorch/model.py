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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
print(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))

from pointnet2_modules import PointnetSAModuleVotes, PointnetFPModule, PointnetSAModule


class Semantic3D_1(nn.Module):
    r"""
    """

    def __init__(self, in_channels, out_channels, kernel_size, resolution, with_se=False, normalize=True, eps=0):
        super().__init__()

        self.voxel_layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm3d(out_channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
            nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm3d(out_channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
         ]



        self.fc_lyaer = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(64, NUM_CLASSES, kernel_size=1),
        )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].contiguous()
            # pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, inputs):
        
        features = self.voxel_layers (inputs)        
        # flatten
        
        
        pred = self.fc_lyaer(features)
        pred = F.log_softmax(pred, dim=1)
        pred = pred.transpose(1, 2)
        # print(pred.size())



        return pred


if __name__ == '__main__':
    backbone_net = Pointnet2Backbone(input_feature_dim=0).cuda()
    print(backbone_net)
    backbone_net.eval()
    out = backbone_net(torch.rand(1, 32,32,32).cuda())
    check = out.cpu().data.numpy()
    check = np.argmax(check, axis=2)
    print(check)
    print (out)
    # for key in sorted(out.keys()):
    #     print(key, '\t', out[key].shape)
