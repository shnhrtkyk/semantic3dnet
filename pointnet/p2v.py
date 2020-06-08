# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 15:37:55 2020

@author: shino
"""

import neuralnet_pytorch as nnt
import torch as T
from torch_scatter import scatter_add
import numpy as np
import colorsys
import collections
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('TkAgg')

def pointcloud2voxel_fast(pc: T.Tensor, voxel_size: int, grid_size=1., filter_outlier=True):
    b, n, _ = pc.shape
    half_size = grid_size / 2.
    valid = (pc >= -half_size) & (pc <= half_size)
    valid = T.all(valid, 2)
    pc_grid = (pc + half_size) * (voxel_size - 1.)
    indices_floor = T.floor(pc_grid)
    indices = indices_floor.long()
    batch_indices = T.arange(b).to(pc.device)
    batch_indices = nnt.utils.shape_padright(batch_indices)
    batch_indices = nnt.utils.tile(batch_indices, (1, n))
    batch_indices = nnt.utils.shape_padright(batch_indices)
    indices = T.cat((batch_indices, indices), 2)
    indices = T.reshape(indices, (-1, 4))

    r = pc_grid - indices_floor
    rr = (1. - r, r)
    if filter_outlier:
        valid = valid.flatten()
        indices = indices[valid]

    def interpolate_scatter3d(pos):
        updates_raw = rr[pos[0]][..., 0] * rr[pos[1]][..., 1] * rr[pos[2]][..., 2]
        updates = updates_raw.flatten()

        if filter_outlier:
            updates = updates[valid]

        indices_shift = T.tensor([[0] + pos]).to(pc.device)
        indices_loc = indices + indices_shift
        out_shape = (b,) + (voxel_size,) * 3
        out = T.zeros(*out_shape).to(pc.device).flatten()
        voxels = scatter_add(updates, nnt.utils.ravel_index(indices_loc.t(), out_shape), out=out).view(*out_shape)
        return voxels

    voxels = [interpolate_scatter3d([k, j, i]) for k in range(2) for j in range(2) for i in range(2)]
    voxels = sum(voxels)
    voxels = T.clamp(voxels, 0., 1.)
    return voxels


def voxelize(pc, vox_size=32):
    vox = pointcloud2voxel_fast(pc, vox_size, grid_size = 1)
    vox = T.clamp(vox, 0., 1.)
    vox = T.squeeze(vox)
    return vox


def iou(pred, gt, th=.5):
    pred = pred > th
    gt = gt > th
    intersect = T.sum(pred & gt).float()
    union = T.sum(pred | gt).float()
    iou_score = intersect / (union + 1e-8)
    return iou_score


def batch_iou(bpc1, bpc2, voxsize=32, thres=.4):
    def _iou(pc1, pc2):
        pc1 = pc1 - T.mean(pc1, -2, keepdim=True)
        pc1 = voxelize(pc1[None], voxsize)

        pc2 = pc2 - T.mean(pc2, -2, keepdim=True)
        pc2 = voxelize(pc2[None], voxsize)
        return iou(pc1, pc2, thres)

    total = map(_iou, bpc1, bpc2)
    return sum(total) / len(bpc1)

def int2hue(integer, in_max=255, out_max=255, is_alpha=False):
    if is_alpha:
        return tuple( float(v * out_max) for v in  colorsys.hsv_to_rgb(integer/in_max, 1, 1) ) + (out_max,)
    else:
        return tuple( float(v * out_max) for v in  colorsys.hsv_to_rgb(integer/in_max, 1, 1) )

def flatten(xs):
    result = []
    for x in xs:
        if isinstance(x, collections.Iterable) and not isinstance(x, dict)and not isinstance(x, str):
            result.extend(flatten(x))
        else:
            result.append(x)
    return result

def draw3d(tensor):
    # tensor = tensor.transpose(2,0,1) # 軸を入れ替える。※ 機械学習で画像を扱うときによく使う軸の順番にしただけで、それ以外の深い意味は無い
    fig = plt.figure(figsize=(5, 5), dpi=100) # 図の大きさを変えられる
    ax = fig.gca(projection='3d')
    org_shape = tensor.shape

    tensor_min = tensor.min() # すべての要素が正になるように、もし負の値があったらその絶対値を全要素に加算する
    if tensor_min < 0:
        tensor = tensor - tensor_min

    eps = 1e-10
    # tensor = tensor + eps # 要素の値が0だと表示できないことがあるので、それを回避するために微小な数を与える

    tensor_max = tensor.max()
    print(tensor_max)
    colors = [int2hue(x, in_max=tensor_max, out_max=1.0, is_alpha=True) for x in flatten(tensor)]
    colors = np.reshape(colors, org_shape + (4,)) # 各座標に、RGBAを表すサイズ4の配列を埋め込むためシェイプに4を追加する

    # 立方体になるように軸を調整
    xsize, ysize, zsize = tensor.shape
    max_range = np.array([xsize, ysize, zsize]).max() * 0.5
    mid_x = xsize * 0.5
    mid_y = ysize * 0.5
    mid_z = zsize * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # 目盛りを削除
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    plt.setp(ax.get_zticklabels(), visible=False)

    ax.voxels(tensor, facecolors=colors, linewidth=0.3, edgecolor='k')
    plt.savefig("./3Dvoxel_rgb_cube_rgb000_axisequal.png", dpi=130,transparent = False)
    # plt.show()

if __name__ == '__main__':
    fname = "../data/test.txt"
    bunny = np.loadtxt(fname, dtype="float")
    print(bunny.shape)
    pc = T.rand(1, bunny.shape[0], 3).cuda()  # bach * num.of point * xyz
    # pc =  T.rand(1, 256, 3).cuda() #bach * num.of point * xyz
    pc[0,:,:] = T.from_numpy(bunny[:,:3])
    voxel = voxelize(pc, vox_size=32)
    # draw3d(T.Tensor.numpy(voxel.cpu()))
    print(np.unique(T.Tensor.numpy(voxel.cpu())))
    print(voxel.size())


