"""
@file sphere_trace.py
@author Nianchen Deng, Shanghai AI Lab
@brief Python wrapper of the sphere tracing algorithm implemented in C++.
"""
import torch
import logging
from typing import Callable, Dict, Union, Any
from operator import itemgetter

import nr3d_lib.bindings._sphere_trace as _backend
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
                 zero_offset: float = 0., distance_scale: float = 1.,
                 min_step: float = .1, hit_threshold: float = 1e-3,
                 max_steps_between_compact: int = 4, max_march_iters: int = 1000,
                 drop_alive_rate: float = 0., tail_sample_threshold: int = 0,
                 tail_sample_step_size: float = None):
        """
        A class for performing sphere tracing.

        Args:
            grid (DenseGrid): The dense occupancy grid, can be created by `DenseGrid(resolution, occ_grid_tensor)`
            zero_offset (float, optional): The offset of SDF value defining a surface. Defaults to 0.0.
            distance_scale (float, optional): The scale factor to apply to the distance traveled at each step. Defaults to 1.0.
            hit_threshold (float, optional): The threshold for considering a hit to have occurred. Defaults to 1e-3.
            max_steps_between_compact (int, optional): The maximum number of advancing steps between performing compaction of the traced rays. Defaults to 4.
            max_march_iters (int, optional): The maximum number of iterations to perform when marching along the trace. Defaults to 1000.
            drop_alive_rate (float, optional): The rate at which to drop alive rays. Defaults to 0.0.
            tail_sample_threshold (int, optional): The threshold for performing tail sampling. Defaults to 0.
            tail_sample_step_size (float, optional): The step size to use when performing tail sampling. Defaults to None.
        """
        self.grid = grid
        self.zero_offset = zero_offset
        self.distance_scale = distance_scale
        self.min_step = min_step
        self.max_steps_between_compact = max_steps_between_compact
        self.max_march_iters = max_march_iters
        self.drop_alive_rate = 0. if tail_sample_threshold else drop_alive_rate
        self.tail_sample_threshold = tail_sample_threshold
        self.tail_sample_step_size = tail_sample_step_size if tail_sample_step_size is not None \
            else min_step
        self.backend = _backend.SphereTracer(min_step, distance_scale, zero_offset, hit_threshold)

    @profile
    @torch.no_grad()
    def trace(self, rays: Dict[str, Union[int, torch.Tensor]],
              sdf_query: Callable[[torch.Tensor], Union[torch.Tensor, Dict[str, torch.Tensor]]],
              print_debug_log: bool = False,
              debug_output: Dict[str, Any] = None, debug_replay: bool = False) -> Dict[str, torch.Tensor]:
        """
        Trace the given rays through the scene using sphere tracing.

        Args:
            rays (dict[str, int | torch.Tensor]): A dictionary containing the properties of rays including:
                rays_inds (torch.Tensor[N]): The indices of the rays.
                rays_o (torch.Tensor[N, 3]): The origins of the rays.
                rays_d (torch.Tensor[N, 3]): The directions of the rays.
                near (torch.Tensor[N]): The near distances of the rays, trace will only perform between near and far distances along rays.
                far (torch.Tensor[N]): The far distances of the rays, trace will only perform between near and far distances along rays.
            sdf_query ((torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]): A function that takes positions and returns either SDF values or a dictionary containing the SDF values and their gradients.
            print_debug_log (bool, optional): Whether to print debug log messages. Defaults to False.
            debug_output (dict[str, Any], optional): Only used for debug. Defaults to None.

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

        def query_sdf(pts):
            query_ret = sdf_query(pts)
            return (query_ret["sdf"] if isinstance(query_ret, dict) else query_ret).to(torch.float)
        
        with profile("sphere_tracer.get_init_segments"):
            valid_rays_idx, segs_pack_info, segs, segs_endpoints, debug_tensors = _backend.ray_march(
                self.grid, *itemgetter("rays_o", "rays_d", "near", "far")(rays), enable_debug=False)
        # >>> Debug ray_march
        # segs_pack_info, segs, debug_tensors = _backend.ray_march(
        #     self.grid, *itemgetter("rays_o", "rays_d", "near", "far")(rays), enable_debug=True)
        # problem_rays_i = debug_tensors["flag"].nonzero(as_tuple=True)[0]
        # problem_march_poses = debug_tensors["march_poses"][problem_rays_i]
        # problem_march_voxels = debug_tensors["march_voxels"][problem_rays_i]
        # problem_march_next_grids = debug_tensors["march_next_grids"][problem_rays_i]
        # problem_march_txyzs = debug_tensors["march_txyzs"][problem_rays_i]
        # if problem_rays_i.numel() > 0:
        #     for i in range(1000):
        #         print(problem_march_poses[0, i].tolist(), problem_march_voxels[0, i].tolist(),
        #             problem_march_next_grids[0, i].tolist(), problem_march_txyzs[0, i].tolist())
        # <<<

        n_rays_alive = valid_rays_idx.numel()
        logger.debug(f"Trace raymarch - {n_rays_alive} "
                     f"({n_rays_alive / n_rays * 100:.2f}%) rays alive")

        if debug_output is not None:
            debug_output["segs_pack_info"] = segs_pack_info
            debug_output["segs"] = segs
            
        with profile("sphere_tracer.init_rays"):
            self.backend.init_rays(*itemgetter("rays_o", "rays_d")(rays), valid_rays_idx,
                                   segs_pack_info, segs)
        
        logger.debug(f"Trace initial - {n_rays_alive} "
                     f"({n_rays_alive / n_rays * 100:.2f}%) rays alive")
        
        i = 0
        with profile("sphere_tracer.march"):
            while (debug_replay and i < self.last_march_iters and n_rays_alive > 0) \
                or (i < self.max_march_iters and n_rays_alive > n_drop_alive
                    and n_rays_alive > self.tail_sample_threshold):
                compact_step_size = min(i + 1, self.max_steps_between_compact)
                for _ in range(compact_step_size):
                    with profile("sphere_tracer.get_trace_positions"):
                        pts = self.backend.get_trace_positions()
                    distances = query_sdf(pts)
                    if debug_output is not None:
                        debug_output.setdefault("trace_data", []).append({
                            "x": pts.clone(),
                            "d": distances,
                            "rays_alive": self.backend.get_rays(_backend.ALIVE),
                            "rays_hit": self.backend.get_rays(_backend.HIT)
                        })
                    with profile("sphere_tracer.advance_rays"):
                        self.backend.advance_rays(distances)
                    i += 1

                with profile("sphere_tracer.compact_rays"):
                    n_rays_alive = self.backend.compact_rays()
                logger.debug(f"Trace step {i} - {n_rays_alive} "
                            f"({n_rays_alive / n_rays * 100:.2f}%) rays alive")

        self.last_march_iters = i

        if self.tail_sample_threshold > 0 and n_rays_alive > 0:
            with profile("sphere_tracer.tail_sample"):
                with profile("sphere_tracer.sample_on_segments"):
                    rays_samples_offset, rays_n_samples, rays_sample_depths, rays_sample_positions \
                        = self.backend.sample_on_segments(self.tail_sample_step_size)
                rays_sample_distances = query_sdf(rays_sample_positions)
                with profile("sphere_tracer.trace_on_samples"):
                    self.backend.trace_on_samples(rays_samples_offset, rays_n_samples,
                                                  rays_sample_depths, rays_sample_distances)
        return self.backend.get_rays(_backend.HIT)
