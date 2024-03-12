"""
@file   MLL.py
@author Jianfei Guo, Shanghai AI Lab && Xinyang Li, XMU 
@brief  MLL (Multi-Layer Lattices), a highly potent alternative to MLP (Multi-Layer Perceptrons) for representational networks.

- Versatility: 
    MLL serves as a powerful non-linear function regressor, superior to MLPs. It is universally applicable \
    across various tasks, including high-dimensional representation learning (up to 64D), \
    2D/3D/4D reconstruction, and 2D/3D/4D generative networks with inputs up to 64D.

- "Feature=Hyper-space": 
    MLL seeks the fundamental simplicity of representational networks. You can use MLLs just like using MLPs. \
    Each "hidden layer" of MLL is represented by a high-dimensional hash-encoding based on permutohedral lattices, \
    which is a local implicit network that takes the input feature as the hyper-space positions. \

- Powerful and rapid Learning: 
    A single layer of MLL possesses a capacity equivalent to a deep and wide MLP with tens of layers. \
    A single layer of MLL learns at a pace tens of times faster than traditional MLPs.

- Data Dimension Auto Adaptability:
    Feel free to concatenate any combination of embedding dimensions to create a high-dimensional input. \
    MLL will automatically warp the inputs to underlying hyper-spaces in a data-driven and self-adaptive manner.
    
    For example, for 4D (3D+t) generative models, simply concatenate the 3D positions, 1D timestamps, N-D latents to \
    form (3+1+N)-dimensional inputs.
"""

__all__ = [
    'PermutohedralLatticeLayer', 
    'MLL', 
    'MLLNet', 
]

from numbers import Number
from typing import Dict, List, Literal, Optional, Union

import torch
import torch.nn as nn
from torch import autograd
import torch.nn.functional as F

from nr3d_lib.utils import tensor_statistics, torch_dtype
from nr3d_lib.models.blocks import get_blocks
from nr3d_lib.models.layers import DenseLayer
from nr3d_lib.models.spatial import AABBSpace, AABBDynamicSpace, BatchedBlockSpace, BatchedDynamicSpace
from nr3d_lib.models.grid_encodings.permuto.permuto_encoding import PermutoEncoding


class PermutohedralLatticeLayer(nn.Module):
    def __init__(
        self, 
        in_features: int, *, 
        decoder_out_features: int = None, # By default, the same with `lattice_out_feats`; Specify -1 for no decoder
        residual_in_features: int = -1, 
        # W=16x2 by default
        n_levels: int = 16, n_feats: int = 2, 
        #---- lattice_cfg / layer_kwargs
        pos_scale: float = 1.0, 
        coarsest_res: float = 10.0, 
        finest_res: float = 1000.0, 
        log2_hashmap_size: int = 18, 
        apply_random_shifts_per_level=True, 
        anneal_cfg: dict = None,
        param_init_cfg={'type': 'uniform', 'bound': 1.0e-4}, 
        #---- Factory kwargs
        dtype: Union[str, torch.dtype] = None, device: torch.device = None, 
        ) -> None:
        super().__init__()
        self.dtype = torch_dtype(dtype)
        self.residual_in_features = residual_in_features

        #------- Permutohedral encoding
        permuto_auto_compute_cfg = dict(
            type='multi_res', n_levels=n_levels, n_feats=n_feats, log2_hashmap_size=log2_hashmap_size, 
            pos_scale=pos_scale, coarsest_res=coarsest_res, finest_res=finest_res, 
            apply_random_shifts_per_level=apply_random_shifts_per_level)
        self.encoding = PermutoEncoding(
            in_features, permuto_auto_compute_cfg=permuto_auto_compute_cfg, space_type='unbounded', 
            anneal_cfg=anneal_cfg, param_init_cfg=param_init_cfg, dtype=self.dtype, device=device)
        
        self.in_features = self.encoding.in_features

        #------- Optional per-layer decoder
        if decoder_out_features is None:
            decoder_out_features = self.encoding.out_features
        
        if decoder_out_features > 0:
            decoder_in_ch = self.encoding.out_features
            # self.decoder = nn.Sequential(
            #     nn.LeakyReLU(.2), 
            #     DenseLayer(decoder_in_ch, decoder_out_features, dtype=self.dtype, device=device), 
            #     nn.LeakyReLU(.2), 
            # )
            self.decoder = DenseLayer(decoder_in_ch, decoder_out_features, dtype=self.dtype, device=device)
            
            # self.decoder = get_blocks(
            #     decoder_in_ch, decoder_out_features, D=1, W=decoder_in_ch, activation='relu', 
            #     output_activation=None, dtype=self.dtype, device=device, use_tcnn_backend=False)
            
            self.out_features = decoder_out_features
        else:
            self.decoder = None
            self.out_features = self.encoding.out_features

        #------- Optional for skip connection / residual connection
        if self.residual_in_features > 0:
            assert self.residual_in_features <= self.out_features, f"out_features={self.out_features} should >= residual_in_features={self.residual_in_features}"
            self.pad_size = self.out_features - self.residual_in_features
            self.zero = nn.Parameter(torch.zeros([], dtype=torch.float, device=device), requires_grad=True)

    @property
    def device(self) -> torch.device:
        return self.encoding.device

    def set_anneal_iter(self, cur_it: int):
        self.encoding.set_anneal_iter(cur_it)

    # def _encode(self, input: torch.Tensor, max_level: int=None, need_dL_dinput: Optional[bool]=None):
    #     return self.encoding(input, max_level=max_level, need_dL_dinput=need_dL_dinput)

    def _decode(self, h: torch.Tensor, residual_input: torch.Tensor = None):
        if self.decoder is not None:
            h = self.decoder(h)
        if self.residual_in_features > 0:
            assert residual_input is not None and residual_input.size(-1) == self.residual_in_features, \
                f"Expects `residual_input` to be of shape [..., {self.residual_in_features}] "
            h = self.zero * h + F.pad(residual_input, (0, self.pad_size), "constant", 0)
        return h

    def forward(
        self, input: torch.Tensor, residual_input: torch.Tensor = None, 
        max_level: int=None, need_dL_dinput: Optional[bool]=None, return_encoded=False):
        encoded = self.encoding(input, max_level=max_level, need_dL_dinput=need_dL_dinput)
        output = self._decode(encoded, residual_input=residual_input)
        if return_encoded:
            return output, encoded
        else:
            return output

    @torch.no_grad()
    def stat_param(self, with_grad=False, prefix: str='') -> Dict[str, float]:
        prefix_ = prefix + ('.' if prefix and not prefix.endswith('.') else '')
        ret = {}
        #-- Encoding
        ret.update(self.encoding.stat_param(with_grad=with_grad, prefix=prefix_ + 'encoding'))
        #-- Decoder
        if self.decoder is not None:
            ret.update({prefix_ + f'decoder.total.{k}' : v for k,v in tensor_statistics(torch.cat([p.data.flatten() for p in self.decoder.parameters()])).items()})
            ret.update({prefix_ + f"decoder.{n}.{k}": v for n, p in self.decoder.named_parameters() for k, v in tensor_statistics(p.data).items()})
            if with_grad:
                param_grads = [p.grad.data.flatten() for p in self.decoder.parameters() if p.grad is not None]
                if len(param_grads) > 0:
                    ret.update({prefix_ + f'decoder.grad.total.{k}': v for k, v in tensor_statistics(torch.cat()).items()})
                ret.update({prefix_ + f"decoder.grad.{n}.{k}": v for n, p in self.decoder.named_parameters() if p.grad is not None for k, v in  tensor_statistics(p.grad.data).items() })
        return ret

class MLL(nn.Module):
    def __init__(
        self, 
        in_features: int, *, 
        D: int = 2, 
        use_residual=True, 
        
        lattice_pos_scale: Union[int, List[int]] = 1.0,
        lattice_n_levels: Union[int, List[int]] = 16,
        lattice_n_feats: Union[int, List[int]] = 2, 
        lattice_cfg = dict(), 
        decoder_out_feats: Union[int, List[int]] = None, # By default, the same with `lattice_out_feats``; Specify -1 for no decoder
        
        space: nn.Module = None,
        space_cfg: dict = None, 
        dtype: Union[str, torch.dtype] = None, device: torch.device = None, 
        ) -> None:
        super().__init__()
        self.dtype = torch_dtype(dtype)
        self.in_features = in_features
        self.use_residual = use_residual
        
        #------- Valid representing space
        if space is not None:
            # Directly use the externally provided space object
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
        
        #------- Lattice layers
        if isinstance(lattice_n_feats, int):
            lattice_n_feats = [lattice_n_feats] * D
        if isinstance(lattice_n_levels, int):
            lattice_n_levels = [lattice_n_levels] * D
        if isinstance(lattice_pos_scale, Number):
            lattice_pos_scale = [lattice_pos_scale] * D
        assert len(lattice_n_feats) == len(lattice_n_levels) == len(lattice_pos_scale) == D
        
        if isinstance(decoder_out_feats, int):
            decoder_out_feats = [decoder_out_feats] * (D-1) # No decoder for the last lattice layer
        elif decoder_out_feats is not None:
            assert len(decoder_out_feats) == (D-1)

        layers = []
        last_out_features = None
        for l, (nl, nf) in enumerate(zip(lattice_n_levels, lattice_n_feats)):
            if l == 0:
                in_dim = in_features
            else:
                in_dim = last_out_features
            
            if l == D-1:
                dec_out_dim = -1 # Not decoder at the last lattice layer
            elif decoder_out_feats is None:
                dec_out_dim = None # The same with encoding.out_features
            else:
                dec_out_dim = decoder_out_feats[l] # Manually specify

            if l == D-1:
                residual_in_dim = -1 # No residual at the last layer
            else:
                residual_in_dim = in_dim if self.use_residual else -1
            
            layer = PermutohedralLatticeLayer(
                in_dim, 
                decoder_out_features=dec_out_dim, 
                residual_in_features=residual_in_dim, 
                n_levels=nl, n_feats=nf, 
                pos_scale=lattice_pos_scale[l], 
                **lattice_cfg, 
                dtype=self.dtype, device=device
            )
            last_out_features = layer.out_features
            layers.append(layer)

        self.lattice_layers = nn.ModuleList(layers)
        self.last_encoded_features = last_out_features
        self.out_features = last_out_features
        self.D = D
    
    @property
    def device(self) -> torch.device:
        return self.lattice_layers[0].device

    def set_anneal_iter(self, cur_it: int):
        for layer in self.lattice_layers:
            layer.set_anneal_iter(cur_it)

    def forward(
        self, input: torch.Tensor, *, 
        max_level: int = None, need_dL_dinput: Optional[bool]=None):
        
        if isinstance(self.space, (AABBSpace, AABBDynamicSpace, BatchedBlockSpace, BatchedDynamicSpace)):
            # Maps input from [-1,1] to [0,1]
            h0 = input / 2. + 0.5
        else:
            h0 = input
        h = h0
        for l, layer in enumerate(self.lattice_layers):
            # residual_input = None if (l == self.D-1) or not self.use_residual else h
            h = layer.forward(
                h, h, max_level=max_level, 
                # NOTE: !!! For l > 0, input is actually the previous layer's output `h`
                need_dL_dinput=need_dL_dinput if l == 0 else h.requires_grad)
        return h

    def get_weight_reg(self, norm_type: float = 2.0):
        norms = []
        for layer in self.lattice_layers:
            if layer.decoder is not None:
                norms.append([p.norm(p=norm_type) for n,p in layer.decoder.named_parameters()])
        return torch.stack(norms)

    @torch.no_grad()
    def clip_grad_and_update_ema(self):
        pass

    @torch.no_grad()
    def stat_param(self, with_grad=False, prefix: str='') -> Dict[str, float]:
        prefix_ = prefix + ('.' if prefix and not prefix.endswith('.') else '')
        ret = {}
        #-- Lattice layers
        for l, layer in enumerate(self.lattice_layers):
            ret.update(layer.stat_param(with_grad=with_grad, prefix=prefix_+f"lattice_layers.{l}"))
        return ret

class MLLNet(MLL):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, *, 
        D: int = 2, 
        use_residual=False, 
        lattice_n_levels: Union[int, List[int]] = 16,
        decoder_out_feats: Union[int, List[int]] = None,
        lattice_n_feats: Union[int, List[int]] = 2, 
        lattice_cfg = dict(), 
        output_activation: Union[str, dict]=None, 
        
        space: nn.Module = None,
        space_cfg: dict = None, 
        
        dtype: Union[str, torch.dtype] = None, device: torch.device = None, 
        ) -> None:
        super().__init__(
            in_features, D=D, 
            use_residual=use_residual, 
            lattice_n_levels=lattice_n_levels, lattice_n_feats=lattice_n_feats, 
            decoder_out_feats=decoder_out_feats, lattice_cfg=lattice_cfg, 
            space=space, space_cfg=space_cfg, dtype=dtype, device=device)
        self.out_features = out_features
        
        #------- Output mapper
        self.to_output = DenseLayer(self.last_encoded_features, out_features, activation=output_activation, dtype=self.dtype, device=device)
        
        # self.to_output = get_blocks(
        #     self.last_encoded_features, out_features, 
        #     D=1, W=self.last_encoded_features, 
        #     activation=dict(type='softplus', beta=100.0), 
        #     output_activation=output_activation, dtype=self.dtype, device=device, use_tcnn_backend=False)

    def forward(
        self, input: torch.Tensor, *, 
        max_level: int = None, need_dL_dinput: Optional[bool]=None, return_h=False):
        h = super().forward(input, max_level=max_level, need_dL_dinput=need_dL_dinput)
        output = self.to_output(h)
        if return_h:
            return dict(output=output, h=h)
        else:
            return dict(output=output)

    def forward_with_nablas(
        self, input: torch.Tensor, *, 
        need_dL_dinput: Optional[bool]=None, 
        has_grad: bool=None, nablas_has_grad: bool=None, 
        max_level: int=None, max_pos_dims: int=None, max_out_dims: int=None):
        
        has_grad = torch.is_grad_enabled() if has_grad is None else has_grad
        nablas_has_grad = has_grad if nablas_has_grad is None else (nablas_has_grad and has_grad)
        
        #---- Forward: from l=0 to l=D-1
        with torch.enable_grad():
            input.requires_grad_(True)
            if isinstance(self.space, (AABBSpace, AABBDynamicSpace, BatchedBlockSpace, BatchedDynamicSpace)):
                # Maps input from [-1,1] to [0,1]
                h0 = input / 2. + 0.5
            else:
                h0 = input
            h = h0
            num_layers = len(self.lattice_layers)
            per_layer_input = [None] * num_layers
            per_layer_encoded = [None] * num_layers
            per_layer_decoded = [None] * num_layers
            for l, layer in enumerate(self.lattice_layers):
                layer_input = h
                per_layer_input[l] = layer_input
                
                h = encoded = layer.encoding(
                    h, max_level=max_level, 
                    # NOTE: !!! For l > 0, input is actually the previous layer's output `h`
                    need_dL_dinput=need_dL_dinput if l == 0 else h.requires_grad)
                per_layer_encoded[l] = encoded
                
                if layer.decoder is not None:
                    h = decoded = layer.decoder(h)
                    per_layer_decoded[l] = decoded
                
                if layer.residual_in_features > 0:
                    h = layer.zero * h + F.pad(layer_input, (0, layer.pad_size), "constant", 0)
                
            output = self.to_output(h)

            # NOTE: The real output used to calculate nablas.
            #       For example, if `input` is formed by concatenating [x,z_ins,z_time], \
            #           then you might only want to calculate dL_dx rather than dL_dinput
            max_out_dims = max_out_dims or output.size(-1)
            output_used = output[..., :max_out_dims] 
        
        #---- Backward dL_dx: from l=D-1 to l=0
        grad = autograd.grad(output_used, h, torch.ones_like(output_used), retain_graph=has_grad, create_graph=nablas_has_grad, only_inputs=True)[0]
        for l in range(self.D-1, -1, -1):
            layer_grad = grad
            layer = self.lattice_layers[l]
            
            #-- Grads on encoding-decoder branch
            grad = (layer.zero * grad) if layer.residual_in_features > 0 else grad
            if layer.decoder is not None:
                grad = autograd.grad(per_layer_decoded[l], per_layer_encoded[l], grad, retain_graph=has_grad, create_graph=nablas_has_grad, only_inputs=True)[0]
            grad = layer.encoding.backward_dydx(grad, per_layer_input[l], max_level=max_level, max_pos_dims=max_pos_dims if l == 0 else None)
            
            #-- Append with grads on residual branch
            if layer.residual_in_features > 0:
                # grad = grad + autograd.grad(per_layer_remapped[l], per_layer_input[l], layer_grad, retain_graph=has_grad, create_graph=nablas_has_grad, only_inputs=True)[0]
                grad = grad + layer_grad[..., :layer.residual_in_features]

        if not nablas_has_grad:
            grad = grad.detach()
        if max_pos_dims is not None:
            grad = grad[..., :max_pos_dims]
        if isinstance(self.space, (AABBSpace, AABBDynamicSpace, BatchedBlockSpace, BatchedDynamicSpace)):
            grad = grad / 2. # Divided by 2 since `input` is also divided by 2 during the forward process
        
        if not has_grad:
            output, h = output.detach(), h.detach()
        
        return dict(output=output, h=h, nablas=grad)

    @torch.no_grad()
    def stat_param(self, with_grad=False, prefix: str='') -> Dict[str, float]:
        prefix_ = prefix + ('.' if prefix and not prefix.endswith('.') else '')
        ret = {}
        #-- Lattice layers
        for l, layer in enumerate(self.lattice_layers):
            ret.update(layer.stat_param(with_grad=with_grad, prefix=prefix_+f"lattice_layers.{l}"))
        #-- Output mapper
        if self.to_output is not None:
            ret.update({prefix_ + f'to_output.total.{k}' : v for k,v in tensor_statistics(torch.cat([p.data.flatten() for p in self.to_output.parameters()])).items()})
            ret.update({prefix_ + f"to_output.{n}.{k}": v for n, p in self.to_output.named_parameters() for k, v in tensor_statistics(p.data).items()})
            if with_grad:
                ret.update({prefix_ + f'to_output.grad.total.{k}': v for k, v in tensor_statistics(torch.cat([p.grad.data.flatten() for p in self.to_output.parameters() if p.grad is not None])).items()})
                ret.update({prefix_ + f"to_output.grad.{n}.{k}": v for n, p in self.to_output.named_parameters() if p.grad is not None for k, v in  tensor_statistics(p.grad.data).items() })
        return ret

if __name__ == "__main__":
    def unit_test():
        from icecream import ic
        from torch.utils.benchmark import Timer
        
        device = torch.device('cuda')
        m = MLLNet(
            3 + 7, 1, D=2, 
            use_residual=True, 
            lattice_n_levels=[8,16], 
            lattice_n_feats=2, 
            output_activation=None, 
            dtype='half', device=device)
        print(m)
        
        num_pts = 3653653
        x = torch.randn([num_pts, 3], dtype=torch.float, device=device)
        z = torch.randn([7,], dtype=torch.float, device=device).tile(num_pts,1)
        pos = torch.cat((x,z), dim=-1)
        # sdf = m.forward(pos)[..., 0]
        ret = m.forward_with_nablas(pos, max_pos_dims=3)
        sdf, nablas = ret['output'][..., 0], ret['nablas']
        
        #---- 49.59 ms (2 linear) vs. 59.87 ms (2 MLP)
        print(Timer(stmt="m.forward(pos)", 
                    globals={'m':m, 'pos':pos}).blocked_autorange())

        #---- 163.42 ms (2 linear) vs. 188.34 ms (2 MLP)
        print(Timer(stmt="m.forward_with_nablas(pos)", 
                    globals={'m':m, 'pos':pos}).blocked_autorange())

        #---- 157.29 ms (2 linear) vs. 182.93 ms (2 MLP)
        print(Timer(stmt="m.forward_with_nablas(pos, max_pos_dims=3)", 
                    globals={'m':m, 'pos':pos}).blocked_autorange())
        
        ic(sdf.shape)
        ic(nablas.shape)
    
    def unit_test_single_layer():
        from icecream import ic
        from torch.utils.benchmark import Timer
        
        device = torch.device('cuda')
        m = MLLNet(3 + 7, 1, D=1, lattice_n_levels=16, lattice_n_feats=2, output_activation='sigmoid', dtype='half', device=device)
        print(m)
        
        num_pts = 3653653
        x = torch.randn([num_pts, 3], dtype=torch.float, device=device)
        z = torch.randn([7,], dtype=torch.float, device=device).tile(num_pts,1)
        pos = torch.cat((x,z), dim=-1)
        # sdf = m.forward(pos)[..., 0]
        ret = m.forward_with_nablas(pos, max_pos_dims=3)
        sdf, nablas = ret['output'][..., 0], ret['nablas']
        
        print(Timer(stmt="m.forward_with_nablas(pos, max_pos_dims=3)", 
                    globals={'m':m, 'pos':pos}).blocked_autorange())
        
        ic(sdf.shape)
        ic(nablas.shape)

    unit_test()
    # unit_test_single_layer()