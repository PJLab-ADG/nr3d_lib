from typing import Literal

from .occgrid import *
from .occgrid_accel import OccupancyGridAS
from .occgrid_batch_accel import OccupancyGridBatchAS
from .occgrid_forest_accel import OccupancyGridForestAS
from .utils import *

def get_accel(type: Literal['octree', 'occ_grid'] = None, **params):
    if type is None or type == 'none':
        return None
    elif type == 'occ_grid':
        return OccupancyGridAS(**params)
    elif type == 'octree':
        from .octree_accel import OctreeAS
        return OctreeAS(**params)
    else:
        raise RuntimeError(f"Invalid type={type}")

def get_forest_accel(type: str = None, **params):
    if type is None or type == 'none':
        return None
    elif type == 'occ_grid_forest':
        return OccupancyGridForestAS(**params)
    elif type == 'octforest':
        from .octforest_accel import OctForestAS
        return OctForestAS(**params)
    else:
        raise RuntimeError(f"Invalid type={type}")

def get_batched_accel(type: str = None, **params):
    if type is None or type == 'none':
        return None
    elif type == 'occ_grid_batch':
        return OccupancyGridBatchAS(**params)
    elif type == 'octbatch':
        from .octbatch_accel import OctBatchAS
        return OctBatchAS(**params)
    else:
        raise RuntimeError(f"Invalid type={type}")