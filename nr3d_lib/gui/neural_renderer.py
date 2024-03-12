"""
@file   neural_renderer.py
@author Jianfei Guo, Shanghai AI Lab
@brief  A kaolin-wisp interface for nr3d_lib-defined renderers
"""

import torch
import torch.nn as nn

from nr3d_lib.config import ConfigDict
from nr3d_lib.gui.datalayers.forest_datalayers import ForestDatalayers
from nr3d_lib.gui.datalayers.occgrid_datalayers import OccGridDatalayers

from nr3d_lib.models.spatial import ForestBlockSpace
from nr3d_lib.models.accelerations.occgrid_accel import OccGridAccel


class NR3DKaolinWispRenderer(object):
    def __init__(
            self, model: nn.Module, config: dict, image_embed_code: torch.Tensor = None) -> None:
        self.model = model
        self.config = config
        self.image_embed_code = image_embed_code

        if isinstance(self.model.space, ForestBlockSpace):
            self.layers_painter = ForestDatalayers()
            self._data_layers = self.layers_painter.regenerate_data_layers(
                self.model.space, alpha=0.2)
        elif isinstance(self.model.accel, OccGridAccel):
            self.layers_painter = OccGridDatalayers()
            self._data_layers = self.layers_painter.regenerate_data_layers(self.model.accel)
        else:
            self._data_layers = {}

    def data_layers(self):
        return {} #self._data_layers

    def render(self, rays_o: torch.Tensor, rays_d: torch.Tensor, near, far, res_x, res_y):
        ray_input = dict(rays_o=rays_o, rays_d=rays_d, near=near, far=far,
                         rays_h_appear=None if self.image_embed_code is None
                         else self.image_embed_code.expand(rays_o.shape[0], -1))
        ray_tested = self.model.space.ray_test(**ray_input)
        ret = self.model.ray_query(ray_input=ray_input, ray_tested=ray_tested, config=self.config,
                                   render_per_obj_individual=True)

        # To render buffer
        rendered = ret['rendered']
        rgb = rendered['rgb_volume'].view([res_y, res_x, 3]) * 255.
        alpha = rendered['mask_volume'].view([res_y, res_x, 1]) * 255.
        rgba = torch.cat([rgb, alpha], dim=-1)
        # depth = (rendered['depth_volume'] / far).view([res_y, res_x, 1])
        # depth = (rendered['depth_volume'] * dir_scale).view([res_y, res_x, 1])
        depth = rendered['depth_volume'].view([res_y, res_x, 1])
        return rgba, depth
