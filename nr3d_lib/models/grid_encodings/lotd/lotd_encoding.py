"""
@file   lotd_encoding.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Basic single-block lotd-encoding module.
"""

__all__ = [
    'LoTDEncoding'
]

import re
import numpy as np
from math import sqrt, prod
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from nr3d_lib.fmt import log
from nr3d_lib.config import ConfigDict
from nr3d_lib.utils import tensor_statistics, torch_dtype

from nr3d_lib.models.spatial import AABBSpace, BatchedBlockSpace
from nr3d_lib.models.utils import batchify_query, clip_norm_
from nr3d_lib.models.grid_encodings.multires_annealer import MultiresAnnealer
from nr3d_lib.models.grid_encodings.lotd.lotd import LoTD, LoDType
from nr3d_lib.models.grid_encodings.lotd.lotd_cfg import get_lotd_cfg
from nr3d_lib.models.grid_encodings.lotd.lotd_helpers import \
    param_interpolate, param_vertices, level_param_index_shape

try:
    import tensorly
    tensorly.set_backend('pytorch')
    from tensorly.decomposition import parafac_power_iteration
except:
    log.info("tensorly is not installed.")

class LoTDEncoding(nn.Module):
    def __init__(
        self, 
        input_ch=3, *,
        lotd_cfg: dict = None, 
        lotd_auto_compute_cfg: dict = None, # (Optional) auto-compute LoTD config from aabb
        lotd_use_cuboid = False, # Whether allow cuboid-shaped LoTD
        space: nn.Module = None,
        space_cfg: dict = None, 
        anneal_cfg: dict = None,
        param_init_cfg={'type': 'uniform_to_type', 'bound': 1.0e-4}, 
        clip_level_grad_ema_factor: float=0, 
        # save_and_load_configs=True, # Whether to save configs to state_dict() and load them next time
        dtype=torch.half, device=None) -> None:
        super().__init__()
        
        self.dtype = torch_dtype(dtype)

        self.clip_level_grad_ema_factor = clip_level_grad_ema_factor
        self.param_init_cfg = param_init_cfg

        #------- Valid representing space
        if space is not None:
            # Directly use the externally provided space object definition
            self.space = space
        elif space_cfg is not None:
            space_cfg = space_cfg.copy()
            space_type = space_cfg.pop('type').lower()
            if space_type == 'aabb':
                space = AABBSpace(**space_cfg)
            elif space_type == 'batched': # Batched AABB
                space = BatchedBlockSpace(**space_cfg)
            elif space_type == 'unbounded' or space_type == 'none':
                space = None
            else:
                raise RuntimeError(f"Invalid space_type={space_type}")
            self.space = space
        else:
            # Do not need input space definition and conversion
            self.space = None
        
        #------- LoTD Metadata
        assert bool(lotd_cfg is not None) != bool(lotd_auto_compute_cfg is not None), "Please specify one and only one of `lotd_cfg` and `lotd_auto_compute_cfg`"
        if lotd_auto_compute_cfg is not None:
            lotd_cfg = get_lotd_cfg(
                **lotd_auto_compute_cfg, input_ch=input_ch, 
                stretch=(1 if not lotd_use_cuboid else (self.space.radius3d*2).tolist()))
        
        self.lotd_cfg = lotd_cfg.to_dict() if not isinstance(lotd_cfg, dict) else lotd_cfg
        self.lotd = LoTD(input_ch, **lotd_cfg, dtype=self.dtype, device=device)
        self.in_features: int = input_ch
        self.out_features: int = self.lotd.out_features

        #------- LoTD parameter register
        # NOTE: Parameters should always be stored as float32. `self.dtype` is only respected when forward.
        p = torch.zeros(self.lotd.n_params, device=device, dtype=torch.float)
        self.register_parameter("flattened_params", nn.Parameter(p, requires_grad=True))

        #------- Param random initialization
        self.init_param_random()

        #------- LoTD Encoding anneal
        if anneal_cfg is not None:
            self.annealer = MultiresAnnealer(self.lotd.level_n_feats, **anneal_cfg, dtype=self.dtype, device=device)
        else:
            self.annealer = None
        # (Optional) float tensor that soft masks encoding's feature.
        self.window: torch.FloatTensor = None 
        # Discard levels that [lvl > max_level], where lvl is in range [0, lotd.n_levels-1]
        # Set `max_level=-1` to discard all levels; set `max_level=lotd.n_levels-1` to remain all levels
        self.max_level: int = None 
        
        #------- Ema grad storage
        if self.clip_level_grad_ema_factor > 0:
            self.register_buffer("level_grad_norm_ema", torch.full([self.lotd.n_levels], 0.1, dtype=torch.float, device=device))
        
        #------- Bind lod param getter/setter
        """
        NOTE: Possible APIs (getter & setter):
              - lod0, lod1, ..., lodn
              - lod0_vec, lod1_mat, ...
              - lod0_vec0, lod1_mat2, ...
        """
        self.pattern = re.compile(r"^lod(?P<level>[0-9]+)(_(?P<op>[a-z]+)(?P<dim>[0-9]+){0,1}){0,1}")
    
    @property
    def device(self) -> torch.device:
        return self.flattened_params.device
    
    @property
    def level_n_feats(self):
        return self.lotd.level_n_feats
    
    @property
    def meta(self):
        return self.lotd.meta
    
    @property
    def lod_meta(self):
        """
        LoTD metadata
        """
        return self.lotd.meta

    @property
    def inference_param(self):
        return self.flattened_params.data.to(self.dtype)

    def set_anneal_iter(self, cur_it: int):
        if self.annealer is not None:
            self.max_level, self.window = self.annealer(cur_it)

    def forward(self, input: torch.Tensor, max_level: int=None) -> torch.Tensor:
        """ Forward
            NOTE: Input must be in range [-1,1]

        Args:
            input (torch.Tensor): Relative position/coords in range [-1,1]
            max_level (int, optional): Maximum lotd level bypass. 
                Only levels l <= `max_level` will be used if specify. Defaults to None.

        Returns:
            torch.Tensor: [..., out_features] The interpolated features
        """
        output = self.lotd.forward(input / 2. + 0.5, self.flattened_params, max_level=(max_level or self.max_level))
        return (output * self.window) if self.window is not None else output
    
    def forward_dydx(self, input: torch.Tensor, max_level: int=None, need_dL_dinput: Optional[bool]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Forward with calculating dy_dx in the mean time.
            NOTE: Input must be in range [-1,1]

        Args:
            input (torch.Tensor): [..., in_features] Input relative position/coords in range [-1,1]
            max_level (int, optional): Maximum lotd level bypass. Defaults to None.
            need_dL_dinput (bool, optional): Whether to propagate the gradient from the loss back to the input. Defaults to False.
                NOTE: Only used for pose refinement. If you need nablas, use `backward_dydx` to calculate instead.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                [..., out_features] The interpolated features
                [..., out_features*in_features] dy_dx
        """
        output, dy_dx = self.lotd.forward_dydx(input / 2. + 0.5, self.flattened_params, max_level=(max_level or self.max_level), need_dL_dinput=need_dL_dinput)
        return ((output * self.window) if self.window is not None else output), dy_dx
    
    def backward_dydx(self, dL_dy: torch.Tensor, dy_dx: torch.Tensor, input: torch.Tensor, max_level: int=None, grad_guard=None) -> torch.Tensor:
        """ Caculate `nablas` given `dL_dy` (loss propagated or assigned to feature output) and `dy_dx` (calculated by `forward_dydx`)
            
            For example, if LoTD decoder is an network that maps `lotd_feature` to `mlp_output`:
            
            >>> # First, calculate the interpolated `lotd_feature` and `dy_dx`:
            >>> lotd_feature, dy_dx = lotd_encoding.forward_dydx(x)
            
            >>> # Then, decode the interpolated `lotd_feature` with `lotd_decoder`: 
            >>> mlp_output = lotd_decoder(lotd_feature)
            
            >>> # Back-propagate unit output loss to decoder's input (i.e. encoding's output) `lotd_feature`:
            >>> dL_dfeature = autograd.grad(mlp_output, lotd_feature, torch.ones_like(mlp_output))
            
            >>> # Further back-propagate gradients on encoding's output to input to get the final `nablas`:
            >>> nablas = lotd_encoding.backward_dydx(dL_dfeature, dy_dx, x)
            
            NOTE: See nr3d_lib/models/fields/sdf/lotd_sdf.py::`forward_sdf_nablas` for detailed examples.

        Args:
            dL_dy (torch.Tensor): [..., out_features], Loss propagated/assigned to lotd feature output.
            dy_dx (torch.Tensor): [..., out_features*in_features], pre-computed dy_dx tensor by `forward_dydx`
            input (torch.Tensor): [..., in_features], Input relative position/coords in range [-1,1]
            max_level (int, optional): Maximum lotd level bypass. Defaults to None.
            grad_guard (LoTD2ndGradGuard, optional): Optional gradients guard to avoid explosion / unsafety. Defaults to None.

        Returns:
            torch.Tensor: [..., in_features], Back-propagated nablas
        """
        nablas = self.lotd.backward_dydx(dL_dy, dy_dx, input / 2. + 0.5, self.flattened_params, max_level=(max_level or self.max_level), grad_guard=grad_guard)
        return nablas / 2. # Divided by 2 since `input` is also divided by 2 during the forward_dydx process

    def get_level_param(self, l: int, op: str = None, dim: int = None, grad=False) -> torch.Tensor:
        """ Get the grid features (or gradients) at a certain level, 
            which is sliced from the `flattened_params`, and viewed (reshaped) as its human-readable shape

        Args:
            l (int): The given level
            op (str, optional): Optionally for VM to specify `mat` or `vec`. Defaults to None.
            dim (int, optional): Optionally for multi-dim lotd types to specify `dim`. Defaults to None.
            grad (bool, optional): Are we getting parameters' gradients (T) or just the parameters (F). Defaults to False.

        Returns:
            torch.Tensor: The required features / gradients 
        """
        index, shape = level_param_index_shape(self.lod_meta, l, op, dim)
        return (self.flattened_params if not grad else self.flattened_params.grad)[index].view(shape)

    def set_level_param(self, l: int, op: str = None, dim: int = None, value: torch.Tensor = ...):
        """ Set the grid features at a certain level

        Args:
            l (int): The given level
            op (str, optional): Optionally for VM to specify `mat` or `vec`. Defaults to None.
            dim (int, optional): Optionally for multi-dim lotd types to specify `dim`. Defaults to None.
            value (torch.Tensor, optional): New grid feature values. 
                Any shape is acceptable as long they can be reshaped into the correponding human-readable shape. 
        """
        index, shape = level_param_index_shape(self.lod_meta, l, op, dim)
        self.flattened_params[index] = value.contiguous().view(prod(shape))

    @torch.no_grad()
    def init_param_random(self):
        param_init_cfg = self.param_init_cfg
        param_init_method = param_init_cfg['type']
        
        if param_init_method == 'uniform':
            bound = param_init_cfg['bound']
            self.flattened_params.uniform_(-bound, bound)
        
        elif param_init_method == 'normal':
            std = param_init_cfg['std']
            self.flattened_params.normal_(0, std)
        
        elif param_init_method == 'uniform_to_type':
            bound = param_init_cfg['bound']
            for l, tp in enumerate(self.lotd.level_types):
                tp = LoDType(tp)
                if tp == LoDType.Dense or tp == LoDType.Hash:
                    b = bound
                elif tp == LoDType.VectorMatrix:
                    b = sqrt(bound)
                elif tp == LoDType.NPlaneSum:
                    b = bound
                elif tp == LoDType.NPlaneMul or tp == LoDType.CP or tp == LoDType.CPfast:
                    b = bound ** (1/3.)
                else:
                    raise RuntimeError(f"Invalid tp={tp}")
                self.get_level_param(l).uniform_(-b, b)

        elif param_init_method == 'normal_to_type':
            std = param_init_cfg['std']
            for l, tp in enumerate(self.lotd.level_types):
                tp = LoDType(tp)
                if tp == LoDType.Dense or tp == LoDType.Hash:
                    s = std
                elif tp == LoDType.VectorMatrix:
                    s = sqrt(std)
                elif tp == LoDType.NPlaneSum:
                    s = std / 3.
                elif tp == LoDType.NPlaneMul or tp == LoDType.CP or tp == LoDType.CPfast:
                    s = std ** (1/3.)
                else:
                    raise RuntimeError(f"Invalid tp={tp}")
                self.get_level_param(l).normal_(0., s)
        else:
            raise RuntimeError(f"Invalid param_init_method={param_init_method}")

    def __getattr__(self, name: str) -> torch.Tensor:
        if ('lod' in name) and (hasattr(self, 'pattern')) and (t:=self.pattern.match(name)):
            # l = int(t.group(1))
            ret = t.groupdict()
            l, op, dim = int(ret['level']), ret['op'], ret['dim']
            if dim is None or dim == '':
                dim = None
            else:
                dim = int(dim)
            if op is None or op == '':
                op = None
            return self.get_level_param(l, op, dim)
        else:
            return super().__getattr__(name)
    
    def __setattr__(self, name: str, value: torch.Tensor) -> None:
        if ('lod' in name) and (hasattr(self, 'pattern')) and (t:=self.pattern.match(name)):
            # l = int(t.group(1))
            ret = t.groupdict()
            l, op, dim = int(ret['level']), ret['op'], ret['dim']
            if dim is None or dim == '':
                dim = None
            else:
                dim = int(dim)
            if op is None or op == '':
                op = None
            self.set_level_param(l, op, dim, value)
        else:
            return super().__setattr__(name, value)

    # Backup lotd_cfg just in case
    def get_extra_state(self) -> Any:
        return self.lotd_cfg
    
    def set_extra_state(self, state: Any):
        self.lotd_cfg = state

    @torch.no_grad()
    def rescale_volume(self, new_aabb: torch.Tensor):
        """ Shrink space and re-interpolate feature grids to their original full resolutions.
        NOTE: Only tensor-decomposition based LoDType(s) support this action. Not for Hash!

        Args:
            new_aabb (torch.Tensor): New AABB to shrink to.
        """
        lod_meta = self.lod_meta
        assert lod_meta.level_res[0] > 0, f"Expects equal resolution on each level! \nWhile current lod_res={lod_meta.level_res_multidim}"
        device = self.device
        new_aabb = new_aabb.view(2,3)
        old_aabb = self.space.aabb
        #----------- Rescale encoding parameter
        origin, scale = (old_aabb[1]+old_aabb[0])/2., (old_aabb[1] - old_aabb[0])/2.
        new_origin, new_scale = (new_aabb[1]+new_aabb[0])/2., (new_aabb[1] - new_aabb[0])/2.
        for l,(R,M,tp) in enumerate(zip(lod_meta.level_res, lod_meta.level_n_feats, lod_meta.level_types)):
            tp = LoDType(tp)
            if tp == LoDType.Dense:
                # Vertices in normalized new aabb
                v = param_vertices(R, 3, False, device=device, dtype=torch.float)
                # Vertices in world
                v = v * new_scale + new_origin
                # Vertices in old aabb
                v = (v - origin) / scale
                # Param in old aabb
                p = self.get_level_param(l, 'vol')
                # Upsampled param in new aabb
                p_new = param_interpolate(p.unsqueeze(0), v.unsqueeze(0), R, False).squeeze(0).contiguous()
                # Set new param
                self.set_level_param(l, value=p_new)
            
            elif tp == LoDType.VectorMatrix:
                # Rescale vector params
                v1d = param_vertices(R, 1, is_forest=True, device=device)
                v1d3 = v1d.view(R,1).expand(R,3)
                v1d3 = v1d3 * new_scale + new_origin
                v1d3 = (v1d3 - origin) / scale
                p = self.get_level_param(l, 'vec') # [3,R,M]
                p_new = param_interpolate(p, v1d3.permute(1,0).view(3,R,1), R, False).contiguous()
                self.set_level_param(l, 'vec', value=p_new)
                
                # Rescale matrix params
                v2d = param_vertices(R, 2, is_forest=True, device=device).view(R,R,2)
                # Define as [xyzxyz]
                v2d3 = v2d[..., [0,1,1,0,0,1]].view(R,R,2,3)
                v2d3 = v2d3 * new_scale + new_origin
                v2d3 = (v2d3 - origin) / scale
                # Need [yz,xz,xy]
                v2d3 = v2d3.view(R,R,6)[...,[4,5,3,2,0,1]].view(R,R,3,2).permute(2,0,1,3) # [3,R,R,2]
                p = self.get_level_param(l, 'mat') # [3,R,R,M]
                p_new = param_interpolate(p, v2d3, R, False).contiguous()
                self.set_level_param(l, 'mat', value=p_new)
            
            elif tp == LoDType.NPlaneMul or tp == LoDType.NPlaneSum:
                # Rescale plane params.
                v2d = param_vertices(R, 2, is_forest=True, device=device).view(R,R,2)
                # Define as [xyzxyz]
                v2d3 = v2d[..., [0,1,1,0,0,1]].view(R,R,2,3)
                v2d3 = v2d3 * new_scale + new_origin
                v2d3 = (v2d3 - origin) / scale
                # Need [yz,xz,xy]
                v2d3 = v2d3.view(R,R,6)[...,[4,5,3,2,0,1]].view(R,R,3,2).permute(2,0,1,3) # [3,R,R,2]
                p = self.get_level_param(l, 'plane') # [3,R,R,M]
                p_new = param_interpolate(p, v2d3, R, False).contiguous()
                self.set_level_param(l, value=p_new)
            
            elif tp == LoDType.CP or tp == LoDType.CPfast:
                # Rescale vector params.
                v1d = param_vertices(R, 1, is_forest=True, device=device)
                v1d3 = v1d.view(R,1).expand(R,3)
                v1d3 = v1d3 * new_scale + new_origin
                v1d3 = (v1d3 - origin) / scale
                p = self.get_level_param(l, 'line') # [3,R,M]
                p_new = param_interpolate(p, v1d3.permute(1,0).view(3,R,1), R, False).contiguous()
                self.set_level_param(l, value=p_new)
            
            elif tp == LoDType.Hash:
                raise RuntimeError("LoDType==hash does not support spatial operations.")

            else:
                raise RuntimeError(f"Invalid lod_type={tp}")

    @torch.no_grad()
    def init_param_from_net(self, net: nn.Module):
        """
        NOTE: Currently, this is not stable & ready-to-use. Just for reference.
        """
        device = self.device
        lod_meta = self.lod_meta
        def query_fn(x):
            prefix = x.shape[:-1]
            y = batchify_query(net.forward, x.flatten(0,-2), chunk=2**14)
            return y.unflatten(0, prefix)
        offset_table = np.cumsum([0] + lod_meta.level_n_feats)
        for l, (R,M,tp) in enumerate(zip(lod_meta.level_res, lod_meta.level_n_feats, lod_meta.level_types)):
            
            if R > 128:
                continue
            tp = LoDType(tp)
            v = param_vertices(R, 3, False, device=device, dtype=torch.float)
            h = query_fn(v)
            h_lvl = h[..., offset_table[l]:offset_table[l+1]]
            
            vecs = []
            for m in range(M):
                # TODO: This is very in-efficient --- running CP decomposition on every feat dim.
                #       Try batched CP decomposition next time.
                w, (vec1, vec2, vec3) = parafac_power_iteration(h_lvl[..., m].data, 1)
                cbrt_w = np.cbrt(w.item())
                vec = cbrt_w * torch.cat([vec1, vec2, vec3], -1).movedim(-1,0).contiguous()
                vecs.append(vec)
                # v1, v2, v3 = vec
                # tt = v1.view(R,1,1) * v2.view(1,R,1) * v3.view(1,1,R)
            vecs = torch.stack(vecs, -1) # [3, R, M]
            v1, v2, v3 = vecs
            
            if tp == LoDType.Dense:
                self.set_level_param(l, value=h_lvl)
            elif tp == LoDType.VectorMatrix:
                # TODO: This is very naive --- First CP decompose, then try to restore using VM setup
                
                # [3, R, R, M] 
                # NOTE: contiguous memory on the last dim (e.g. z of xyz)
                p_m = np.sqrt(1/3.) * torch.stack([v2.view(R,1,M) * v3.view(1,R,M), v1.view(R,1,M) * v3.view(1,R,M), v1.view(R,1,M) * v2.view(1,R,M)], 0)
                # [3, R, M]
                p_v = np.sqrt(1/3.) * vecs
                self.set_level_param(l, 'mat', value=p_m)
                self.set_level_param(l, 'vec', value=p_v)
            elif tp == LoDType.NPlaneMul:
                # [3, R, R, M] 
                sqrt_v = vecs.abs().sqrt()
                v1_, v2_, v3_ = sqrt_v
                p_m_sign = torch.stack([v1.sign().view(R,1,M).expand(R,R,M), v2.sign().view(R,1,M).expand(R,R,M), v3.sign().view(R,1,M).expand(R,R,M)], 0)
                p_m = p_m_sign * torch.stack([v2_.view(R,1,M) * v3_.view(1,R,M), v1_.view(R,1,M) * v3_.view(1,R,M), v1_.view(R,1,M) * v2_.view(1,R,M)], 0)
                # tt = p_m[0].view(1,R,R,M) * p_m[1].view(R,1,R,M) * p_m[2].view(R,R,1,M)
                self.set_level_param(l, value=p_m)
            elif tp == LoDType.NPlaneSum:
                # NOTE: Not possible
                pass
            elif tp == LoDType.CP or tp == LoDType.CPfast:
                self.set_level_param(l, value=vecs)

    @torch.no_grad()
    def clip_grad_and_update_ema(self, val: float=None):
        if self.clip_level_grad_ema_factor > 0:
            # gnorm = torch.stack([self.get_level_param(l, grad=True).data.abs().max() for l in range(self.lotd.n_levels)])
            gnorm = torch.stack([self.get_level_param(l, grad=True).data.norm() for l in range(self.lotd.n_levels)])
            
            ema = self.level_grad_norm_ema.copy_(gnorm.lerp(self.level_grad_norm_ema, 0.99))

            for lvl in range(self.lotd.n_levels):
                index, shape = level_param_index_shape(self.lod_meta, lvl)
                
                val = self.clip_level_grad_ema_factor * ema[lvl].item()
                
                # self.flattened_params.grad[index].clip_(-val, val)
                clip_norm_(self.flattened_params.grad[index], val)
    
    @torch.no_grad()
    def stat_param(self, with_grad=False, prefix: str='') -> Dict[str, float]:
        prefix_ = prefix + ('.' if prefix and not prefix.endswith('.') else '')
        ret = {}
        ret.update({prefix_ + f'total.{k}': v for k,v in tensor_statistics(self.flattened_params.data).items()})
        if with_grad and self.flattened_params.grad is not None:
            ret.update({prefix_ + f'grad_total.{k}': v for k,v in tensor_statistics(self.flattened_params.grad.data).items()})
        for lvl, tp in enumerate(self.lotd.level_types):
            tp = LoDType(tp)
            if tp == LoDType.VectorMatrix:
                ret.update({prefix_ + f'lv.{lvl}.vec.{k}': v for k,v in tensor_statistics(self.get_level_param(lvl, 'vec').data).items()})
                ret.update({prefix_ + f'lv.{lvl}.mat.{k}': v for k,v in tensor_statistics(self.get_level_param(lvl, 'mat').data).items()})
                if with_grad:
                    ret.update({prefix_ + f'grad.lv.{lvl}.vec.{k}': v for k,v in tensor_statistics(self.get_level_param(lvl, 'vec', grad=True).data).items()})
                    ret.update({prefix_ + f'grad.lv.{lvl}.mat.{k}': v for k,v in tensor_statistics(self.get_level_param(lvl, 'mat', grad=True).data).items()})
            else:
                ret.update({prefix_ + f'lv.{lvl}.{k}': v for k,v in tensor_statistics(self.get_level_param(lvl).data).items()})
                if with_grad:
                    ret.update({prefix_ + f'grad.lv.{lvl}.{k}': v for k,v in tensor_statistics(self.get_level_param(lvl, grad=True).data).items()})
        if self.clip_level_grad_ema_factor > 0:
            ret.update({prefix_ + f'grad.lv.{lvl}.ema': self.level_grad_norm_ema[lvl].item() for lvl in range(self.lotd.n_levels)})
        return ret

if __name__ == "__main__":
    def test_ops(device=torch.device('cuda')):
        m = LoTDEncoding(
            3, lotd_cfg=ConfigDict(lod_res=[13,21,34,55,89,144], lod_types=['Dense', 'VM', 'NPlaneMul', 'CP', 'Hash', 'NPlaneSum'], lod_n_feats=[2,4,6,2,4,2], size=65536), 
            param_init_cfg={'type': 'uniform_to_type', 'bound': 1.0e-4},  device=device, dtype=torch.float16)
        with torch.no_grad():
            m.lod5_mat1 = torch.randn([144,144,2], device=device, dtype=torch.float16)
            print(m)
    test_ops()