# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 18:53:05 2020

@author: shino
"""
import numpy as np
import pandas as pd

from pyntcloud import PyntCloud

import binvox_rw

cloud = PyntCloud.from_file("test/00000.txt",
                            sep=" ",
                            header=0,
                            names=["x","y","z"])

# cloud.plot(mesh=True, backend="threejs")

voxelgrid_id = cloud.add_structure("voxelgrid", n_x=32, n_y=32, n_z=32)
voxelgrid = cloud.structures[voxelgrid_id]
# voxelgrid.plot(d=3, mode="density", cmap="hsv")

x_cords = voxelgrid.voxel_x
y_cords = voxelgrid.voxel_y
z_cords = voxelgrid.voxel_z

voxel = np.zeros((32, 32, 32)).astype(np.bool)

for x, y, z in zip(x_cords, y_cords, z_cords):
    voxel[x][y][z] = True

with open("test/00000.binvox", 'wb') as f:
    v = binvox_rw.Voxels(voxel, (32, 32, 32), (0, 0, 0), 1, 'xyz')
    v.write(f)