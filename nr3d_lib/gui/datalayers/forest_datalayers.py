"""
Modified from https://github.com/NVIDIAGameWorks/kaolin-wisp
"""

# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from typing import Dict
from abc import ABC, abstractmethod

import torch

from nr3d_lib.models.spatial import ForestBlockSpace
from nr3d_lib.models.grid_encodings.utils import points_to_corners
from nr3d_lib.gui.kaolin_wisp_modified.core.primitives import PrimitivesPack
from nr3d_lib.plot import soft_blue, soft_red, lime_green, purple, gold, light_pink

class Datalayers(ABC):

    @abstractmethod
    def needs_redraw(self, *args, **kwargs) -> bool:
        raise NotImplementedError

    @abstractmethod
    def regenerate_data_layers(self, *args, **kwargs) -> Dict[str, PrimitivesPack]:
        raise NotImplementedError


class ForestDatalayers(Datalayers):

    def __init__(self):
        self._last_state = dict()

    def needs_redraw(self, grid: ForestBlockSpace) -> True:
        # Pyramids contain information about the number of cells per level,
        # it's a plausible heuristic to determine whether the frame should be redrawn
        return not ('block_ks' in self._last_state and torch.equal(self._last_state['block_ks'], grid.block_ks))

    def regenerate_data_layers(self, grid: ForestBlockSpace, alpha=1.0) -> Dict[str, PrimitivesPack]:
        data_layers = dict()
        # lod_colors = [
        #     torch.tensor((*soft_blue, alpha)),
        #     torch.tensor((*soft_red, alpha)),
        #     torch.tensor((*lime_green, alpha)),
        #     torch.tensor((*purple, alpha)),
        #     torch.tensor((*gold, alpha)),
        # ]

        colors = torch.tensor((*light_pink, alpha))

        cells = PrimitivesPack()
        
        corners_int = points_to_corners(grid.block_ks)
        corners = grid.world_origin + corners_int * grid.world_block_size
        
        grid_lines = corners[:, [(0, 1), (1, 3), (3, 2), (2, 0),
                                    (4, 5), (5, 7), (7, 6), (6, 4),
                                    (0, 4), (1, 5), (2, 6), (3, 7)]]

        grid_lines_start = grid_lines[:, :, 0].reshape(-1, 3)
        grid_lines_end = grid_lines[:, :, 1].reshape(-1, 3)
        color_tensor = colors
        grid_lines_color = color_tensor.repeat(grid_lines_start.shape[0], 1)
        cells.add_lines(grid_lines_start, grid_lines_end, grid_lines_color)

        data_layers[f'Occupancy Grid'] = cells

        self._last_state['block_ks'] = grid.block_ks
        return data_layers
