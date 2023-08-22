"""
@file sphere_trace.py
@author Nianchen Deng, Shanghai AI Lab
@brief Python wrapper of the sphere tracing algorithm implemented in C++.
"""
import torch
import logging
from typing import Callable, Dict, List, Union
from operator import itemgetter

import nr3d_lib_bindings._sphere_trace as _backend
from nr3d_lib.profile import profile
from nr3d_lib.fmt import init_log

__all__ = ["SphereTracer", "DenseGrid"]


DenseGrid = _backend.DenseGrid
logger = init_log(f'sphere_trace')


class SphereTracer():
    """
    A class for performing sphere tracing.
    """

    def __init__(self, grid: DenseGrid, *,
                    zero_offset: float = 0.0, distance_scale: float = 0.95,
                    hit_threshold: float = 5e-4, hit_at_neg: bool = False,
                    max_steps_between_compact: int = 4, max_march_iters: int = 1000,
                    drop_alive_rate: float = 0.):
        """
        A class for performing sphere tracing.

        Args:
            grid (DenseGrid): The dense occupancy grid, can be created by `DenseGrid(resolution, occ_grid_tensor)`
            zero_offset (float, optional): The offset of SDF value defining a surface. Defaults to 0.0.
            distance_scale (float, optional): The scale factor to apply to the distance traveled at each step. Defaults to 0.95.
            hit_threshold (float, optional): The threshold for considering a hit to have occurred. Defaults to 5e-4.
            hit_at_neg (bool, optional): Whether to consider hits at negative distances. Defaults to False.
            max_steps_between_compact (int, optional): The maximum number of advancing steps between performing compaction of the traced rays. Defaults to 4.
            max_march_iters (int, optional): The maximum number of iterations to perform when marching along the trace. Defaults to 1000.
            drop_alive_rate (float, optional): The rate at which to drop alive rays. Defaults to 0..
        """        
        self.grid = grid
        self.zero_offset = zero_offset
        self.distance_scale = distance_scale
        self.hit_threshold = hit_threshold
        self.hit_at_neg = hit_at_neg
        self.max_steps_between_compact = max_steps_between_compact
        self.max_march_iters = max_march_iters
        self.drop_alive_rate = drop_alive_rate
        self.backend = _backend.SphereTracer()

    @profile
    def trace(self, rays: Dict[str, Union[int, torch.Tensor]],
              sdf_query: Callable[[torch.Tensor], Union[torch.Tensor, Dict[str, torch.Tensor]]],
              print_debug_log: bool = False,
              debug_trace_data: List[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Trace the given rays through the scene using sphere tracing.

        Args:
            rays (dict[str, int | torch.Tensor]): A dictionary containing the properties of rays including:
                ray_inds (torch.Tensor[N]): The indices of the rays.
                rays_o (torch.Tensor[N, 3]): The origins of the rays.
                rays_d (torch.Tensor[N, 3]): The directions of the rays.
                near (torch.Tensor[N]): The near distances of the rays, trace will only perform between near and far distances along rays.
                far (torch.Tensor[N]): The far distances of the rays, trace will only perform between near and far distances along rays.
            sdf_query ((torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]): A function that takes positions and returns either SDF values or a dictionary containing the SDF values and their gradients.
            print_debug_log (bool, optional): Whether to print debug log messages. Defaults to False.
            debug_trace_data (list[dict[str, torch.Tensor]], optional): If provide, the detail trace data of each step will be stored in the list. Only used for debug. Defaults to None.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing the hit information including:
                pos (torch.Tensor[N', 3]): The hit positions.
                dir (torch.Tensor[N', 3]): The directions of hit rays.
                idx (torch.Tensor[N']): The indices of hit rays.
                voxel_idx (torch.Tensor[N']): The voxel indices of hit rays.
                t (torch.Tensor[N']): The distances between the hit positions and the origins of hit rays.
                n_steps (torch.Tensor[N']): The number of steps taken to reach the hit positions.
        
        Remarks:
            N is the number of input rays.
            N' is the number of hit rays.
        """
        logger.setLevel(logging.DEBUG if print_debug_log else logging.INFO)
        logger.handlers[0].setLevel(logging.DEBUG if print_debug_log else logging.INFO)
        n_rays = rays["rays_o"].shape[0]
        n_drop_alive = self.drop_alive_rate * n_rays
        with profile("sphere_tracer.init_rays"):
            self.backend.init_rays(*itemgetter("rays_o", "rays_d", "ray_inds", "near", "far")(rays),
                                   self.grid)
        with profile("sphere_tracer.compact_rays"):
            n_rays_alive = self.backend.compact_rays()
        logger.debug(f"Trace init - {n_rays_alive} "
                     f"({n_rays_alive / n_rays * 100:.2f}%) rays alive")
        i = 0
        distance_scale = self.distance_scale
        while i < self.max_march_iters and \
                (i < 100 and n_rays_alive > 0 or n_rays_alive > n_drop_alive):
            compact_step_size = min(i + 1, self.max_steps_between_compact)
            for _ in range(compact_step_size):
                with profile("sphere_tracer.query_sdf"):
                    with profile("sphere_tracer.get_trace_positions"):
                        pts = self.backend.get_trace_positions()
                    query_ret = sdf_query(pts)

                if isinstance(query_ret, dict):
                    raw_distances = query_ret["sdf"]
                    nablas = query_ret.get("nablas")
                else:
                    raw_distances = query_ret
                    nablas = None

                if nablas is not None:
                    # Mitigate sway by normalizing distance if gradient is available
                    normal_norm = nablas.norm(dim=-1)
                    fixed_distances = raw_distances / normal_norm.clip(1.0)
                else:
                    # Mitigate sway by decay distance scale
                    normal_norm = None
                    fixed_distances = raw_distances
                    if i >= 100 and i % 10 == 0:
                        distance_scale = max(distance_scale * 0.9, 0.1)

                if debug_trace_data is not None:
                    debug_trace_data.append({
                        "x": pts.clone(),
                        "d": fixed_distances,
                        "raw_d": raw_distances,
                        "nn": normal_norm,
                        "rays_alive": self.backend.get_rays(_backend.ALIVE),
                        "rays_hit": self.backend.get_rays(_backend.HIT)
                    })
                with profile("sphere_tracer.advance_rays"):
                    self.backend.advance_rays(fixed_distances.to(torch.float), self.zero_offset,
                                              distance_scale, self.hit_threshold, self.hit_at_neg,
                                              self.grid)
                i += 1

            with profile("sphere_tracer.compact_rays"):
                n_rays_alive = self.backend.compact_rays()
            logger.debug(f"Trace step {i} - {n_rays_alive} "
                         f"({n_rays_alive / n_rays * 100:.2f}%) rays alive")

        return self.backend.get_rays(_backend.HIT)
