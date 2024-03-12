
import plyfile # pip install plyfile
import numpy as np
from typing import NamedTuple

from nr3d_lib.fmt import log

def export_pcl_ply(pcl: np.ndarray, pcl_color: np.ndarray = None, filepath: str = ...):
    """
    pcl_color: if provided, should be uint8_t
    """
    num_pts = pcl.shape[0]
    if pcl_color is not None:
        verts_tuple = np.zeros((num_pts,), dtype=[(
            "x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        data = [tuple(p1.tolist() + p2.tolist()) for p1, p2 in zip(pcl, pcl_color)]
        verts_tuple[:] = data[:]
    else:
        verts_tuple = np.zeros((num_pts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
        data = [tuple(p.tolist()) for p in pcl]
        verts_tuple[:] = data[:]

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    ply_data = plyfile.PlyData([el_verts])
    log.info(f"=> Saving pointclouds to {str(filepath)}")
    ply_data.write(filepath)