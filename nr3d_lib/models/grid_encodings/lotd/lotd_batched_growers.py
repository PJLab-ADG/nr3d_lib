"""
@file   lotd_batched_growers.py
@author Jianfei Guo, Shanghai AI Lab & Xinyang Li, Shanghai AI Lab
@brief  Batched LoTD parameters' conditional growers
"""

__all__ = [
    # Infrastructures
    'LoTDGrower',
    'MixedLoTDGrower',

    # flattended MLP growers
    'DenseLoTDGrowerFlatten',
    'TriplaneLoTDGrowerFlatten',
    'VMSplitLoTDGrowerFlatten',
    'CPLoTDGrowerFlatten',

    # Conv growers
    'DenseLoTDGrowerConv',
    'TriplaneLoTDGrowerConv',
    'VMSplitLoTDGrowerConv',
    # TODO: CPfast grower conv

    # Coordinate MLP growers
    ## FMM MLP growers
    ### Conv
    'DenseLoTDGrowerFMM',
    'NPlaneLoTDGrowerFMM',
    'VMSplitLoTDGrowerFMM',
    'CPLoTDGrowerFMM',
    
    # Concat growers
    'DenseLoTDGrowerConcat'
]

import itertools
import numpy as np
from math import prod
from typing import List, Union, Literal

import torch
import torch.nn as nn

from nr3d_lib.utils import import_str, is_scalar, torch_dtype

from nr3d_lib.models.blocks import get_blocks
from nr3d_lib.models.embedders import get_embedder
from nr3d_lib.models.layers import get_nonlinearity
from nr3d_lib.models.utils import BufferList, ParameterList
from nr3d_lib.models.modulations.modulations import ModulatedLayer, ModulatedBlock

class LoTDGrower(nn.Module):
    def __init__(
        self, z_dim: int, *, 
        lod_res: List[int], lod_n_feats: Union[int, List[int]], lod_types: Union[str, List[str]] = None, world_dim=3, 
        device=None, dtype=torch.float) -> None:
        super().__init__()
        self.dtype = torch_dtype(dtype)
        self.register_buffer('_device_marker', torch.tensor([0], device=device), persistent=False)
        self.level_res = list(lod_res)
        if is_scalar(lod_n_feats):
            # self.level_n_feats = [lod_n_feats for _ in range(len(lod_res))]
            lod_n_feats = [lod_n_feats] * len(lod_res)
        else:
            assert len(lod_n_feats) == len(lod_res)
        if lod_types is None:
            lod_types = ["Dense"] * len(lod_res)
        elif isinstance(lod_types, str):
            lod_types = [lod_types] * len(lod_res)
        else: 
            assert len(lod_types) == len(lod_res)
        self.level_n_feats = list(lod_n_feats)
        self.lod_types = list(lod_types)
        self.z_dim = z_dim
        self.world_dim = world_dim

    @property
    def device(self) -> torch.device:
        return self._device_marker.device

    def forward(self, z: torch.Tensor, max_level: int = np.inf):
        raise NotImplementedError

    def extra_repr(self) -> str:
        return f"lod_res={self.level_res}, lod_n_feats={self.level_n_feats}, lod_types={self.lod_types}"

class MixedLoTDGrower(LoTDGrower):
    def __init__(
        self, grower_configs: List[dict], world_dim=3, 
        device=None, dtype=torch.float):
        dtype = torch_dtype(dtype)
        growers = []
        for i, cfg in enumerate(grower_configs):
            cfg['param']['world_dim'] = world_dim
            cfg['param']['device'] = device
            cfg['param']['dtype'] = dtype
            g: LoTDGrower = import_str(cfg['target'])(**cfg['param'])
            growers.append(g)
            if i == 0:
                z_dim = g.z_dim
            else:
                assert g.z_dim == z_dim, "latent dim does not match"
        lod_res = list(itertools.chain.from_iterable([g.level_res for g in growers]))
        lod_n_feats = list(itertools.chain.from_iterable([g.level_n_feats for g in growers]))
        lod_types = list(itertools.chain.from_iterable([g.lod_types for g in growers]))
        super().__init__(
            z_dim, lod_res=lod_res, lod_n_feats=lod_n_feats, lod_types=lod_types, 
            world_dim=world_dim, device=device, dtype=dtype)
        self.growers = nn.ModuleList(growers)
        self.num_level_cumsum = np.cumsum([0] + [len(g.level_res) for g in growers]).tolist()[:-1]

    @staticmethod
    def from_growers(growers: List[LoTDGrower]):
        lod_res = list(itertools.chain.from_iterable([g.level_res for g in growers]))
        lod_n_feats = list(itertools.chain.from_iterable([g.level_n_feats for g in growers]))
        lod_types = list(itertools.chain.from_iterable([g.lod_types for g in growers]))
        o = LoTDGrower(lod_res, lod_n_feats, lod_types)
        o.growers = nn.ModuleList(growers)
        o.num_level_cumsum = np.cumsum([0] + [len(g.level_res) for g in growers]).tolist()[:-1]
        return o

    def forward(self, z: torch.Tensor, max_level: int = np.inf):
        max_level = np.inf if max_level is None else max_level
        return torch.cat(
            [
                g(z, max_level=(max_level-self.num_level_cumsum[i])) \
                for i, g in enumerate(self.growers)
            ], 1)


class DenseLoTDGrowerFlatten(LoTDGrower):
    def __init__(
        self,
        # input
        z_dim,
        # config
        lod_res=[3,5,8,13,21,34,55], lod_n_feats=4, world_dim=3,
        device=None, dtype=torch.float, 
        **other_net_cfg
        ) -> None:
        super().__init__(
            z_dim, lod_res=lod_res, lod_n_feats=lod_n_feats, lod_types="Dense", 
            world_dim=world_dim, device=device, dtype=dtype)

        n_params_per_level = []
        num_encoded_dim = 0
        for s, f in zip(self.level_res, self.level_n_feats):
            n_params_per_level.append((s**world_dim) * f)
            num_encoded_dim += f
        
        self.n_params_per_level = n_params_per_level
        self.mapper = get_blocks(z_dim, np.sum(n_params_per_level), **other_net_cfg, device=device, dtype=self.dtype)
        self.num_encoded_dim = num_encoded_dim

    def forward(self, z: torch.Tensor, max_level: int = np.inf):
        # NOTE: max_level is ignored.
        return self.mapper(z)

class TriplaneLoTDGrowerFlatten(LoTDGrower):
    def __init__(
        self,
        # input
        z_dim,
        # config
        lod_res=[3,5,8,13,21,34,55,89,144], lod_n_feats=4, world_dim=3,
        device=None, dtype=torch.float, 
        **other_net_cfg
        ) -> None:
        super().__init__(
            z_dim, lod_res=lod_res, lod_n_feats=lod_n_feats, lod_types="NPlane", 
            world_dim=world_dim, device=device, dtype=dtype)
        assert world_dim >= 2, "NPlane lod_type only support world_dim>=2"

        n_params_per_level = []
        num_encoded_dim = 0
        for s, f in zip(self.level_res, self.level_n_feats):
            n_params_per_level.append(world_dim * (s**(world_dim-1)) * f)
            num_encoded_dim += f
        
        self.n_params_per_level = n_params_per_level
        self.mapper = get_blocks(z_dim, np.sum(n_params_per_level), **other_net_cfg, device=device, dtype=self.dtype)
        self.num_encoded_dim = num_encoded_dim

    def forward(self, z: torch.Tensor, max_level: int = np.inf):
        # NOTE: max_level is ignored.
        return self.mapper(z)

class VMSplitLoTDGrowerFlatten(LoTDGrower):
    def __init__(
        self,
        # input
        z_dim,
        # config
        lod_res=[3,5,8,13,21,34,55,89,144], lod_n_feats=4, world_dim=3,
        device=None, dtype=torch.float, 
        **other_net_cfg
        ) -> None:
        super().__init__(
            z_dim, lod_res=lod_res, lod_n_feats=lod_n_feats, lod_types="VectorMatrix", 
            world_dim=world_dim, device=device, dtype=dtype)
        assert world_dim == 3, "VM lod_type only support world_dim==3"

        n_params_per_level = []
        num_encoded_dim = 0
        for s, f in zip(self.level_res, self.level_n_feats):
            n_params_per_level.append(3 * (s*s+s) * f)
            num_encoded_dim += f
        
        self.n_params_per_level = n_params_per_level
        self.mapper = get_blocks(z_dim, np.sum(n_params_per_level), **other_net_cfg, device=device, dtype=self.dtype)
        self.num_encoded_dim = num_encoded_dim

    def forward(self, z: torch.Tensor, max_level: int = np.inf):
        # NOTE: max_level is ignored.
        return self.mapper(z)

class CPLoTDGrowerFlatten(LoTDGrower):
    def __init__(
        self, 
        # input
        z_dim: int, 
        # config
        lod_res=[3,5,8,13,21,34,55,89,144], lod_n_feats=4, world_dim=3,
        device=None, dtype=torch.float, 
        **other_net_cfg):
        super().__init__(
            z_dim, 
            lod_res=lod_res, lod_n_feats=lod_n_feats, lod_types="CPfast", 
            world_dim=world_dim, device=device, dtype=dtype)

        n_params_per_level = []
        num_encoded_dim = 0
        for s, f in zip(self.level_res, self.level_n_feats):
            n_params_per_level.append(world_dim * s * f)
            num_encoded_dim += f
        
        self.n_params_per_level = n_params_per_level
        self.mapper = get_blocks(z_dim, np.sum(n_params_per_level), **other_net_cfg, device=device, dtype=self.dtype)
        self.num_encoded_dim = num_encoded_dim

    def forward(self, z: torch.Tensor, max_level: int = np.inf):
        # NOTE: max_level is ignored.
        return self.mapper(z)

"""
#----------------------------------------------------------------
#         Basic N-dimension LoD FMM modules
#----------------------------------------------------------------

Coherence:

    lotd_grower.py          lotd_parameterization.py        Explanation
    
    LoTDFMM                 LoTDCoordParamtrz               All different lod_res share the same coordinate network.            lod_n_feats must be the same.
    LoTDFMMShared           LoTDCoordParamtrzShared         Different lod_res share a common base, with different heads.        lod_n_feats can be different. 
    ---                     LoTDCoordParamtrzIndependent    Different lod_res uses completely different coordinate networks.    lod_n_feats can be different. 

"""

class PseudoTransform(nn.Module):
    def __init__(self, bmin, bmax) -> None:
        super().__init__()
        assert bmax > bmin
        self.bmin = bmin
        self.bmax = bmax
        self.range = bmax-bmin
        self.scale = 2./self.range
        self.offset = -(self.bmax+self.bmin) / (self.range)
    def forward(self, x: torch.Tensor):
        # NOTE: From [bmin, bmax] to [-1,1]
        # return (x-self.bmin) / (self.range/2.) - 1
        return self.scale * x + self.offset # Equivalent, with less compute graph

class LoTDFMM(nn.Module):
    def __init__(
        self,
        z_dim: int,
        lod_res: List[int], lod_n_feats: Union[int, List[int]], world_dim=3, pseudo_bound=[-1,1],
        embed_cfg:dict={'type':'identity'}, D=2, W=128, activation='relu', 
        use_learnable_embedder=False, use_shared_encoding=False, 
        device=None, dtype=torch.float, 
        **other_net_cfg
        ) -> None:
        """
            use_learnable_embedder:
                True: use leanable embedder cache.
            use_shared_encoding:
                True: use learnable shared encoding for all instances.
        """
        super().__init__()
        if not isinstance(lod_n_feats, list):
            lod_n_feats = [lod_n_feats] * len(lod_res)
        assert len(set(lod_n_feats))==1, \
            f'{self.__class__.__name__} only supports the same n_feat across different resolutions'
        assert pseudo_bound == [-1,1], 'Currently only supports [-1,1] bound.'

        self.dtype = torch_dtype(dtype)

        self.embedder, in_ch = get_embedder(embed_cfg, world_dim)

        nl, gain, init_fn, first_init_fn = get_nonlinearity(activation)
        self.net = nn.ModuleList([
            ModulatedLayer(in_ch, W, z_dim, kernel_type=0, activation=nl, gain=gain, **other_net_cfg, device=device, dtype=self.dtype),
            *[ModulatedLayer(W, W, z_dim, kernel_type=0, activation=nl, gain=gain, **other_net_cfg, device=device, dtype=self.dtype) for _ in range(D - 1)],
            ModulatedLayer(W, lod_n_feats[0], z_dim, kernel_type=0, activation=nn.Identity(), gain=1, **other_net_cfg, device=device, dtype=self.dtype)
        ])

        pseudo_x = []
        output_shapes = []
        for s, f in zip(lod_res, lod_n_feats):
            # [s, s, s, 3] or [s, s, 2] or [s, 1]
            x = torch.stack(
                    torch.meshgrid(
                        [torch.linspace(pseudo_bound[0], pseudo_bound[1], s, device=device, dtype=torch.float) for _ in range(world_dim)], 
                        indexing='ij'), 
                    dim=-1)
            # NOTE: Here we directly store the embedded x, which serves as a form of cache.
            x = self.embedder(x)
            pseudo_x.append(x)
            output_shapes.append(([s] * world_dim) + [f])
        
        self.output_shapes = output_shapes

        if use_learnable_embedder:
            self.pseudo_x = ParameterList(pseudo_x)
        else:
            self.pseudo_x = BufferList(pseudo_x, persistent=False)

        if use_shared_encoding:
            shared_encoding = []
            for s, f in zip(lod_res, lod_n_feats):
                # [s, s, s, f] or [s, s, f] or [s, f]
                e = torch.zeros(1, *[s for i in range(world_dim)], f, device=device, dtype=self.dtype).view(1, -1)
                shared_encoding.append(e)
            self.shared_encoding = ParameterList(shared_encoding)
        else: 
            self.shared_encoding = None
            
        # if pseudo_bound != [-1,1]:
        #     self.pseudo_transform = PseudoTransform(pseudo_bound[0], pseudo_bound[1])
        # else:
        #     self.pseudo_transform = None

    @property
    def device(self) -> torch.device:
        return self.net[0].device

    def forward_batched_trees(self, z: torch.Tensor, max_level: int = np.inf):
        B = z.shape[0]
        rets = []
        for i, pseudo_x in enumerate(self.pseudo_x):
            if (max_level is not None) and (i > max_level):
                h = torch.zeros([B, prod(self.output_shapes[i])], device=z.device, dtype=z.dtype)
                rets.append(h)
                continue
            
            # [B, s, s, s, 3] or [B, s, s, 2] or [B, s, 1]
            # h = self.embedder(pseudo_x)
            h = pseudo_x.tile([B, *[1]*(pseudo_x.dim())])
            # [B, s, s, s, per_scale_ch] -> flatten() -> [B, s*s*s*per_scale_ch]
            for layer in self.net:
                h = layer(h, z)
            h = h.flatten(1,-1)
            if self.shared_encoding is not None:
                h = h + self.shared_encoding[i]
            rets.append(h)
        return rets

class LoTDFMMShared(nn.Module):
    def __init__(
        self,
        z_dim: int,
        lod_res: List[int], lod_n_feats: Union[int, List[int]], world_dim=3, pseudo_bound=[-1,1],
        embed_cfg:dict={'type':'identity'}, 
        D=2, W=128, D_head: Union[int,List[int]]=None, W_head: Union[int,List[int]]=None, activation='relu', 
        use_learnable_embedder=False, use_shared_encoding=False, 
        device=None, dtype=torch.float, 
        **other_net_cfg
        ) -> None:
        super().__init__()
        
        self.dtype = torch_dtype(dtype)
        
        num_heads = len(lod_res)
        if not isinstance(lod_n_feats, list):
            lod_n_feats = [lod_n_feats] * num_heads
        if D_head is None: D_head = D
        if W_head is None: W_head = W
        if isinstance(D_head, list): assert len(D_head) == num_heads
        else: D_head = [D_head] * num_heads
        if isinstance(W_head, list): assert len(W_head) == num_heads
        else: W_head = [W_head] * num_heads
        assert pseudo_bound == [-1,1], 'Currently only supports [-1,1] bound.'

        self.world_dim = world_dim
        self.embedder, in_ch = get_embedder(embed_cfg, world_dim)

        nl, gain, init_fn, first_init_fn = get_nonlinearity(activation)
        self.base = nn.ModuleList([
            ModulatedLayer(in_ch, W, z_dim, kernel_type=0, activation=nl, gain=gain, **other_net_cfg, device=device, dtype=self.dtype),
            *[ModulatedLayer(W, W, z_dim, kernel_type=0, activation=nl, gain=gain, **other_net_cfg, device=device, dtype=self.dtype) for _ in range(D)],
        ])

        pseudo_x = []
        grid_gen_heads = []
        output_shapes = []
        for s, f, d, w in zip(lod_res, lod_n_feats, D_head, W_head):
            # [s, s, s, 3] or [s, s, 2] or [s, 1]
            x = torch.stack(
                    torch.meshgrid(
                        [torch.linspace(pseudo_bound[0], pseudo_bound[1], s, device=device, dtype=torch.float) for _ in range(world_dim)], 
                        indexing='ij'), 
                    dim=-1)
            # NOTE: Here we directly store the embedded x, which serves as a form of cache.
            x = self.embedder(x)
            pseudo_x.append(x)

            head = nn.ModuleList([
                *[ModulatedLayer(W, w, z_dim, kernel_type=0, activation=nl, gain=gain, **other_net_cfg, device=device, dtype=self.dtype) for _ in range(d - 1)],
                ModulatedLayer(w, f, z_dim, kernel_type=0, activation=nn.Identity(), gain=1, demodulation=False, **other_net_cfg, device=device, dtype=self.dtype),
            ])
            grid_gen_heads.append(head)
            output_shapes.append(([s] * world_dim) + [f])

        self.output_shapes = output_shapes

        # self.pseudo_x = BufferList(pseudo_x, persistent=False)
        self.grid_gen_heads = nn.ModuleList(grid_gen_heads)

        if use_learnable_embedder:
            self.pseudo_x = ParameterList(pseudo_x)
        else:
            self.pseudo_x = BufferList(pseudo_x, persistent=False)

        if use_shared_encoding:
            shared_encoding = []
            for s, f in zip(lod_res, lod_n_feats):
                # [s, s, s, f] or [s, s, f] or [s, f]
                e = torch.zeros(1, *[s for i in range(world_dim)], f, device=device, dtype=torch.float).view(1, -1)
                shared_encoding.append(e)
            self.shared_encoding = ParameterList(shared_encoding)
        else: 
            self.shared_encoding = None

    @property
    def device(self) -> torch.device:
        return self.base[0].device

    def forward_batched_trees(self, z: torch.Tensor, max_level: int = np.inf):
        B = z.shape[0]
        rets = []
        for i, head in enumerate(self.grid_gen_heads):
            if (max_level is not None) and (i > max_level):
                h = torch.zeros([B, prod(self.output_shapes[i])], device=z.device, dtype=z.dtype)
                rets.append(h)
                continue
            
            # [B, s, s, s, 3] or [B, s, s, 2] or [B, s, 1]
            # h = self.embedder(pseudo_x)
            pseudo_x = self.pseudo_x[i].tile([B, *[1]*(self.pseudo_x[i].dim())])

            h = pseudo_x
            # [B, s, s, s, per_scale_ch] -> flatten() -> [B, s*s*s*per_scale_ch]
            for layer in self.base:
                h = layer(h, z)
            for layer in head:
                h = layer(h, z)
            h = h.flatten(1,-1)
            if self.shared_encoding is not None:
                h = h + self.shared_encoding[i]
            rets.append(h)
        return rets


def get_fmm(
    world_dim, z_dim: int, lod_res, lod_n_feats, 
    pseudo_net_type: Literal['same', 'shared'], pseudo_net_param):
    if pseudo_net_type.lower() == 'same':
        return LoTDFMM(z_dim, lod_res=lod_res, lod_n_feats=lod_n_feats, world_dim=world_dim, **pseudo_net_param)
    elif pseudo_net_type.lower() == 'shared':
        return LoTDFMMShared(z_dim, lod_res=lod_res, lod_n_feats=lod_n_feats, world_dim=world_dim, **pseudo_net_param)
    else:
        raise RuntimeError(f"Invalid pseudo_net_type={pseudo_net_type}")

class DenseLoTDGrowerFMM(LoTDGrower):
    def __init__(
        self, 
        # input
        z_dim, 
        # config
        lod_res: List[int], lod_n_feats: Union[int, List[int]], world_dim=3,
        pseudo_net_type: Literal['same', 'shared'] = 'shared',
        pseudo_net_param: dict = {}, 
        device=None, dtype=torch.float
        ) -> None:
        super().__init__(
            z_dim, lod_res=lod_res, lod_n_feats=lod_n_feats, lod_types="Dense", 
            world_dim=world_dim, device=device, dtype=dtype)
        pseudo_net_param.setdefault('device', device)
        pseudo_net_param.setdefault('dtype', dtype)
        self.fmm = get_fmm(world_dim, z_dim, lod_res, lod_n_feats, pseudo_net_type, pseudo_net_param)
    def forward(self, z: torch.Tensor, max_level: int = np.inf):
        rets = self.fmm.forward_batched_trees(z, max_level=max_level)
        return torch.cat(rets, 1)

class NPlaneLoTDGrowerFMM(LoTDGrower):
    def __init__(
        self, 
        z_dim: int, 
        lod_res: List[int], lod_n_feats: Union[int, List[int]], world_dim=3, 
        pseudo_net_type: Literal['same', 'shared'] = 'shared',
        pseudo_net_param: dict = {}, 
        device=None, dtype=torch.float
        ) -> None:
        super().__init__(
            z_dim, lod_res=lod_res, lod_n_feats=lod_n_feats, lod_types="NPlaneSum", 
            world_dim=world_dim, device=device, dtype=dtype)
        pseudo_net_param.setdefault('device', device)
        pseudo_net_param.setdefault('dtype', dtype)
        self.fmm2d = get_fmm(2, z_dim, lod_res, [n*3 for n in self.level_n_feats], pseudo_net_type, pseudo_net_param)
    def forward(self, z: torch.Tensor, max_level: int = np.inf):
        lod_param = self.fmm2d.forward_batched_trees(z, max_level=max_level)
        return torch.cat(lod_param, 1)

class VMSplitLoTDGrowerFMM(LoTDGrower):
    def __init__(
        self, 
        # input
        z_dim, 
        # config
        lod_res: List[int], lod_n_feats: Union[int, List[int]], world_dim=3,
        pseudo_net_type: Literal['same', 'shared'] = 'shared',
        pseudo_net_param: dict = {}, 
        device=None, dtype=torch.float
        ) -> None:
        super().__init__(
            z_dim, lod_res=lod_res, lod_n_feats=lod_n_feats, lod_types="VM", 
            world_dim=world_dim, device=device, dtype=dtype)
        assert world_dim == 3, "VM lod_type only support world_dim==3"
        pseudo_net_param.setdefault('device', device)
        pseudo_net_param.setdefault('dtype', dtype)
        self.fmm2d = get_fmm(2, z_dim, lod_res, [n*3 for n in self.level_n_feats], pseudo_net_type, pseudo_net_param)
        self.fmm1d = get_fmm(1, z_dim, lod_res, [n*3 for n in self.level_n_feats], pseudo_net_type, pseudo_net_param) 
    def forward(self, z: torch.Tensor, max_level: int = np.inf):
        rets1d = self.fmm1d.forward_batched_trees(z, max_level=max_level)
        rets2d = self.fmm2d.forward_batched_trees(z, max_level=max_level)
        lod_param = []
        for r1d, r2d in zip(rets1d, rets2d):
            lod_param.append(r1d)
            lod_param.append(r2d)
        # [line, plane, line, plane, ...]
        return torch.cat(lod_param, 1)

class CPLoTDGrowerFMM(LoTDGrower):
    def __init__(
        self, 
        # input
        z_dim, 
        # config
        lod_res: List[int], lod_n_feats: Union[int, List[int]], world_dim=3,
        pseudo_net_type: Literal['same', 'shared'] = 'shared',
        pseudo_net_param: dict = {}, 
        device=None, dtype=torch.float
        ) -> None:
        super().__init__(
            z_dim, lod_res=lod_res, lod_n_feats=lod_n_feats, lod_types="CPfast", 
            world_dim=world_dim, device=device, dtype=dtype)
        pseudo_net_param.setdefault('device', device)
        pseudo_net_param.setdefault('dtype', dtype)
        self.fmm1d = get_fmm(1, z_dim, lod_res, [n*world_dim for n in self.level_n_feats], pseudo_net_type, pseudo_net_param)
    def forward(self, z: torch.Tensor, max_level: int = np.inf):
        rets1d = self.fmm1d.forward_batched_trees(z, max_level=max_level)
        return torch.cat(rets1d, 1)

#----------------------------------------------------------------------------

class DenseLoTDGrowerConv(LoTDGrower):
    def __init__(self,
        w_dim,
        resolution      = 128,          # Output image resolution.
        channel_base    = 32768 * 4,    # Overall multiplier for the number of channels.
        channel_max     = 256,          # Maximum number of channels in any layer.
        channel_mul     = 4,
        world_dim       = 3
    ):
        assert world_dim == 3, "Conv growers only support world_dim==3"
        assert resolution >= 4 and resolution & (resolution - 1) == 0
        self.w_dim = w_dim
        self.resolution = resolution
        self.resolution_log2 = int(np.log2(resolution))
        self.block_resolutions = [2 ** i for i in range(2, self.resolution_log2 + 1)]
        channels_dict = {res: min(channel_base // channel_mul ** int(np.log2(res)), channel_max) for res in self.block_resolutions}

        super().__init__(z_dim=w_dim, lod_res=self.block_resolutions, lod_n_feats=[channels_dict[res] for res in self.block_resolutions], lod_types="Dense", world_dim=world_dim)            

        n_params_per_level = []
        num_encoded_dim = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            block = ModulatedBlock(in_channels, out_channels, w_dim=w_dim, kernel_type=3)
            setattr(self, f'b{res}', block)

            out_block = ModulatedLayer(out_channels, out_channels, w_dim=w_dim, kernel_type=3, kernel_size=1, activation=nn.Identity(), gain=1, demodulation=False)
            setattr(self, f'out{res}', out_block)

            n_params_per_level.append((res**3) * out_channels)
            num_encoded_dim += out_channels
        
        self.n_params_per_level = n_params_per_level
        self.num_encoded_dim = num_encoded_dim

    def forward(self, z: torch.Tensor, max_level: int = np.inf):
        outs = []
        x = None
        for i, res in enumerate(self.block_resolutions):
            if (max_level is not None) and (i > max_level):
                outs.append(torch.zeros([z.shape[0], self.n_params_per_level[i]], device=z.device, dtype=z.dtype))
                continue
            x = getattr(self, f'b{res}')(x, z)
            outs.append(getattr(self, f'out{res}')(x, z))
        lod_params = torch.cat([out.view(z.shape[0], -1) for out in outs], 1)
        return lod_params

#----------------------------------------------------------------------------

class TriplaneLoTDGrowerConv(LoTDGrower):
    def __init__(self,
        w_dim,
        resolution      = 256,          # Output image resolution.
        channel_base    = 32768,    # Overall multiplier for the number of channels.
        channel_max     = 512,          # Maximum number of channels in any layer.
        channel_mul     = 2,
        world_dim       = 3
    ):
        assert world_dim == 3, "Conv growers only support world_dim==3"
        assert resolution >= 4 and resolution & (resolution - 1) == 0
        self.w_dim = w_dim
        self.resolution = resolution
        self.resolution_log2 = int(np.log2(resolution))
        self.block_resolutions = [2 ** i for i in range(2, self.resolution_log2 + 1)]
        channels_dict = {res: min(channel_base // channel_mul ** int(np.log2(res)), channel_max) for res in self.block_resolutions}

        super().__init__(z_dim=w_dim, lod_res=self.block_resolutions, lod_n_feats=[channels_dict[res] for res in self.block_resolutions], lod_types="NPlane", world_dim=world_dim)            

        n_params_per_level = []
        num_encoded_dim = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            block = ModulatedBlock(in_channels, out_channels, w_dim=w_dim, kernel_type=2, up=nn.Upsample(scale_factor=2, mode='bilinear'))
            setattr(self, f'b{res}', block)

            enc_channels = out_channels * 3
            out_block = ModulatedLayer(out_channels, enc_channels, w_dim=w_dim, kernel_type=2, kernel_size=1, activation=nn.Identity(), gain=1, demodulation=False)
            setattr(self, f'out{res}', out_block)

            n_params_per_level.append((res**2) * enc_channels)
            num_encoded_dim += enc_channels
        
        self.n_params_per_level = n_params_per_level
        self.num_encoded_dim = num_encoded_dim

    def forward(self, z: torch.Tensor, max_level: int = np.inf):
        outs = []
        x = None
        for i, res in enumerate(self.block_resolutions):
            if (max_level is not None) and (i > max_level):
                outs.append(torch.zeros([z.shape[0], self.n_params_per_level[i]], device=z.device, dtype=z.dtype))
                continue
            
            x = getattr(self, f'b{res}')(x, z)
            outs.append(getattr(self, f'out{res}')(x, z))

        lod_params = torch.cat([out.view(z.shape[0], -1) for out in outs], 1)
        return lod_params

#----------------------------------------------------------------------------
   
class VMSplitLoTDGrowerConv(LoTDGrower):
    def __init__(self,
        w_dim,
        resolution      = 256,          # Output image resolution.
        channel_base    = 32768,    # Overall multiplier for the number of channels.
        channel_max     = 512,          # Maximum number of channels in any layer.
        channel_mul     = 2,
        world_dim       = 3
    ):
        assert world_dim == 3, "Conv growers only support world_dim==3"
        assert resolution >= 4 and resolution & (resolution - 1) == 0
        self.w_dim = w_dim
        self.resolution = resolution
        self.resolution_log2 = int(np.log2(resolution))
        self.block_resolutions = [2 ** i for i in range(2, self.resolution_log2 + 1)]
        channels_dict = {res: min(channel_base // channel_mul ** int(np.log2(res)), channel_max) for res in self.block_resolutions}

        super().__init__(z_dim=w_dim, lod_res=self.block_resolutions, lod_n_feats=[channels_dict[res] for res in self.block_resolutions], lod_types="VM", world_dim=world_dim)            

        output_plane_shapes = []
        output_line_shapes = []
        num_encoded_dim = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            block = ModulatedBlock(in_channels, out_channels, w_dim=w_dim, kernel_type=2, up=nn.Upsample(scale_factor=2, mode='bilinear'))
            setattr(self, f'b{res}', block)

            enc_channels = out_channels * 3
            out_block_plane = ModulatedLayer(out_channels, enc_channels, w_dim=w_dim, kernel_type=2, kernel_size=1, activation=nn.Identity(), gain=1, demodulation=False)
            setattr(self, f'out_plane{res}', out_block_plane)

            out_block_line = ModulatedLayer(out_channels, enc_channels, w_dim=w_dim, kernel_type=2, kernel_size=1, activation=nn.Identity(), gain=1, demodulation=False)
            setattr(self, f'out_line{res}', out_block_line)

            output_plane_shapes.append([res, res, enc_channels])
            output_line_shapes.append([res, enc_channels])
            num_encoded_dim += enc_channels
        
        self.output_plane_shapes = output_plane_shapes
        self.output_line_shapes = output_line_shapes
        self.num_encoded_dim = num_encoded_dim

    def forward(self, z: torch.Tensor, max_level: int = np.inf):
        
        # TODO: Refactor into a method that first allocates and then assigns indices to avoid the copying brought by cat
        outs_plane = []
        outs_line = []
        x = None
        for i, res in enumerate(self.block_resolutions):
            if (max_level is not None) and (i > max_level):
                outs_plane.append(torch.zeros([z.shape[0], prod(self.output_plane_shapes[i])], dtype=z.dtype, device=z.device))
                outs_line.append(torch.zeros([z.shape[0], prod(self.output_line_shapes[i])], dtype=z.dtype, device=z.device))
                continue
            
            x = getattr(self, f'b{res}')(x, z)

            outs_plane.append(getattr(self, f'out_plane{res}')(x, z))
            # needtocheck xy, yz, zx

            lines = getattr(self, f'out_line{res}')(x, z)
            outs_line.append((lines.mean(3), lines.mean(2)))
            # x,y,y,z,z,x

        outs_line = [outs_line[1][1] + outs_line[2][0],
                     outs_line[0][0] + outs_line[2][1],
                     outs_line[0][1] + outs_line[1][0]]

        lod_forest_grid_tensor = torch.cat([torch.cat([out_plane.view(z.shape[0], -1), out_line.view(z.shape[0], -1)], 1) for out_plane, out_line in zip(outs_plane, outs_line)], 1)
        return lod_forest_grid_tensor.flatten()

#----------------------------------------------------------------------------

class DenseLoTDGrowerConcat(LoTDGrower):
    def __init__(
        self, 
        # input
        z_dim, 
        # config
        lod_res=[3,5,8,13,21,34,55,89], lod_n_feats=4, world_dim=3, pseudo_bound=[-1,1], 
        embed_cfg:dict={'type':'identity'}, D=2, W=128, activation='relu', **other_net_cfg
        ) -> None:
        super().__init__(z_dim, lod_res, lod_n_feats, "Dense", world_dim)
        self.embedder, in_ch = get_embedder(embed_cfg, world_dim)
        # self.grid_feat_gen = MLP(in_ch + z_dim, W, D=D, W=W, activation=activation, output_activation=activation, **other_net_cfg)
        self.base = get_blocks(in_ch + z_dim, W, D=D, W=W, activation=activation, output_activation=activation, **other_net_cfg)
        
        # It's a pseudo x, only used in the grow tree process, just for expanding into grid feat.
        pseudo_x = []
        grid_gen_heads = []
        n_params_per_level = []
        num_encoded_dim = 0
        for s, f in zip(self.level_res, self.level_n_feats):
            n_params_per_level.append((s**world_dim) * f)
            # [s, s, s, 3] or [s, s, 2] or [s, 1]
            x = torch.stack(
                    torch.meshgrid(
                        [torch.linspace(pseudo_bound[0], pseudo_bound[1], s) for _ in range(world_dim)], 
                        indexing='ij'), 
                    dim=-1)
            # NOTE: Here we directly store the embedded x, which serves as a form of cache.
            x = self.embedder(x)
            pseudo_x.append(x)
            
            cur_gen_head = get_blocks(W, f, D=D, W=W, activation=activation, output_activation=None, **other_net_cfg)
            grid_gen_heads.append(cur_gen_head)
            
            num_encoded_dim += f
        
        self.pseudo_x = BufferList(pseudo_x, persistent=False)
        self.grid_gen_heads = nn.ModuleList(grid_gen_heads)
        self.n_params_per_level = n_params_per_level
        self.lod_offsets_with_f = np.concatenate([np.array([0]), np.cumsum(n_params_per_level)])
        self.num_encoded_dim = num_encoded_dim

    def forward(self, z: torch.Tensor, max_level: int = np.inf):
        B = z.shape[0]
        lod_params = z.new_zeros([B, self.lod_offsets_with_f[-1]])
        for i, head in enumerate(self.grid_gen_heads):
            _0 = self.lod_offsets_with_f[i]
            _1 = self.lod_offsets_with_f[i+1]
            if (max_level is not None) and (i > max_level):
                continue
            
            pseudo_x = self.pseudo_x[i].tile([B, *[1]*(self.pseudo_x[i].dim())])
            # [B, s, s, s, 3]
            # pseudo_x = self.embedder(pseudo_x)
            # NOTE: Core operation: concat
            h = torch.cat([pseudo_x, z.tile([*pseudo_x.shape[:-1], 1, 1])], dim=-1)
            # [B, s, s, s, W]
            h = self.base(h)
            # NOTE: Consider such flattening, it changes continuously on z, so when using on the CUDA side, the index should also change continuously on z.
            # [B, s, s, s, per_scale_ch] -> flatten() -> [B, s*s*s*per_scale_ch]
            lod_params[:, _0:_1] = head(h).flatten(1, -1)
        
        return lod_params
