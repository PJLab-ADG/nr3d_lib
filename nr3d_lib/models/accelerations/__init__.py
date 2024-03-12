
from typing import Union

from .utils import *
from .occgrid_accel import *

accel_types_single = (
    OccGridAccel, 
)
accel_types_single_t = Union[
    OccGridAccel, 
]

accel_types_dynamic = (
    OccGridAccelDynamic, 
    OccGridAccelStaticAndDynamic, 
)
accel_types_dynamic_t = Union[
    OccGridAccelDynamic, 
    OccGridAccelStaticAndDynamic, 
]

accel_types_batched = (
    OccGridAccelBatched_Ema, 
    OccGridAccelBatched_Getter, 
)
accel_types_batched_t = Union[
    OccGridAccelBatched_Ema, 
    OccGridAccelBatched_Getter, 
]

accel_types_batched_dynamic = (
    OccGridAccelBatchedDynamic_Ema, 
    OccGridAccelBatchedDynamic_Getter, 
)
accel_types_batched_dynamic_t = Union[
    OccGridAccelBatchedDynamic_Ema, 
    OccGridAccelBatchedDynamic_Getter, 
]

accel_types_forest = (
    OccGridAccelForest, 
)
accel_types_forest_t = Union[
    OccGridAccelForest, 
]

def get_accel_class(type: str = None):
    type = type.lower()
    if type is None or type == 'none':
        return None
    elif type in ['occ_grid', 'occgrid']:
        return OccGridAccel
    elif type in ['occ_grid_batched_ema', 'occgrid_batched_ema']:
        return OccGridAccelBatched_Ema
    elif type in ['occ_grid_batched_getter', 'occgrid_batched_getter']:
        return OccGridAccelBatched_Getter
    elif type in ['occ_grid_dynamic', 'occgrid_dynamic']:
        return OccGridAccelDynamic
    elif type in ['occ_grid_static_and_dynamic', 'occgrid_static_and_dynamic']:
        return OccGridAccelStaticAndDynamic
    elif type in ['occ_grid_batched_dynamic_ema', 'occgrid_batched_dynamic_ema']:
        return OccGridAccelBatchedDynamic_Ema
    elif type in ['occ_grid_batched_dynamic_getter', 'occgrid_batched_dynamic_getter']:
        return OccGridAccelBatchedDynamic_Getter
    elif type in ['occ_grid_forest', 'occgrid_forest']:
        return OccGridAccelForest
    elif type in ['oct', 'octree']:
        from .octree_accel import OctreeAS
        return OctreeAS
    elif type in ['oct_batch']:
        from .octbatch_accel import OctAccelBatched
        return OctAccelBatched
    elif type in ['oct_forest']:
        from .octforest_accel import OctAccelForest
        return OctAccelForest
    else:
        raise RuntimeError(f"Invalid type={type}")

def get_accel(type: str = None, **kwargs):
    type = type.lower()
    cls = get_accel_class(type)
    if cls is None:
        return None
    else:
        return cls(**kwargs)

# def get_accel(type: str = None, **kwargs):
#     type = type.lower()
#     if type is None or type == 'none':
#         return None
#     elif type in ['occ_grid', 'occgrid']:
#         return OccGridAccel(**kwargs)
#     elif type in ['occ_grid_batched_ema', 'occgrid_batched_ema']:
#         return OccGridAccelBatched_Ema(**kwargs)
#     elif type in ['occ_grid_batched_getter', 'occgrid_batched_getter']:
#         return OccGridAccelBatched_Getter(**kwargs)
#     elif type in ['occ_grid_dynamic', 'occgrid_dynamic']:
#         return OccGridAccelDynamic(**kwargs)
#     elif type in ['occ_grid_static_and_dynamic', 'occgrid_static_and_dynamic']:
#         return OccGridAccelStaticAndDynamic(**kwargs)
#     elif type in ['occ_grid_batched_dynamic_ema', 'occgrid_batched_dynamic_ema']:
#         return OccGridAccelBatchedDynamic_Ema(**kwargs)
#     elif type in ['occ_grid_batched_dynamic_getter', 'occgrid_batched_dynamic_getter']:
#         return OccGridAccelBatchedDynamic_Getter(**kwargs)
#     elif type in ['occ_grid_forest', 'occgrid_forest']:
#         return OccGridAccelForest(**kwargs)
#     elif type in ['oct', 'octree']:
#         from .octree_accel import OctreeAS
#         return OctreeAS(**kwargs)
#     elif type in ['oct_batch']:
#         from .octbatch_accel import OctAccelBatched
#         return OctAccelBatched(**kwargs)
#     elif type in ['oct_forest']:
#         from .octforest_accel import OctAccelForest
#         return OctAccelForest(**kwargs)
#     else:
#         raise RuntimeError(f"Invalid type={type}")
