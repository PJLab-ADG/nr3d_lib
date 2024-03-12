from .single import *
from .batched import *
from .forest import *
from .dynamic import *
from .batched_dynamic import *

# def get_occ_grid_accel(type: str = None, **kwargs):
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
#     else:
#         raise RuntimeError(f"Invalid type={type}")
