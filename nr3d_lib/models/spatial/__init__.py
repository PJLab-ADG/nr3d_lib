from typing import Union

from .aabb import *
from .aabb_dynamic import *
from .batched import *
from .batched_dynamic import *
from .forest import *
from .utils import *

def get_space(space_cfg: Union[str, dict] = None):
    if space_cfg is None:
        return None
    
    if isinstance(space_cfg, str):
        space_cfg = {'type': space_cfg}
    else:
        space_cfg = space_cfg.copy()
    space_type = space_cfg.pop('type').lower()
    if space_type == 'aabb':
        space = AABBSpace(**space_cfg)
    elif space_type == 'aabb_dynamic':
        space = AABBDynamicSpace(**space_cfg)
    elif space_type == 'batched':
        space = BatchedBlockSpace(**space_cfg)
    elif space_type == 'batched_dynamic':
        space = BatchedDynamicSpace(**space_cfg)
    elif space_type == 'forest':
        space = ForestBlockSpace(**space_cfg)
    elif space_type == 'unbounded' or space_type == 'none':
        space = None
    else:
        raise RuntimeError(f"Invalid space_type={space_type}")
    return space