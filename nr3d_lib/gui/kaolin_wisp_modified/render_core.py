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

import torch
import numpy as np
from typing import Dict, List
from collections import defaultdict

from nr3d_lib.config import ConfigDict
from nr3d_lib.gui.kaolin_wisp_modified.core.primitives import PrimitivesPack

from kaolin.render.camera import Camera, PinholeIntrinsics, OrthographicIntrinsics
from kaolin.render.camera.intrinsics import CameraFOV

def generate_default_grid(width, height, device=None):
    h_coords = torch.arange(height, device=device)
    w_coords = torch.arange(width, device=device)
    return torch.meshgrid(h_coords, w_coords, indexing='ij')  # return pixel_y, pixel_x

def generate_centered_pixel_coords(img_width, img_height, res_x=None, res_y=None, device=None):
    pixel_y, pixel_x = generate_default_grid(res_x, res_y, device)
    scale_x = 1.0 if res_x is None else float(img_width) / res_x
    scale_y = 1.0 if res_y is None else float(img_height) / res_y
    pixel_x = pixel_x * scale_x + 0.5   # scale and add bias to pixel center
    pixel_y = pixel_y * scale_y + 0.5   # scale and add bias to pixel center
    return pixel_y, pixel_x

def _to_ndc_coords(pixel_x, pixel_y, camera):
    pixel_x = 2 * (pixel_x / camera.width) - 1.0
    pixel_y = 2 * (pixel_y / camera.height) - 1.0
    return pixel_x, pixel_y

def generate_pinhole_rays(camera: Camera, coords_grid: torch.Tensor):
    """Default ray generation function for pinhole cameras.

    This function assumes that the principal point (the pinhole location) is specified by a 
    displacement (camera.x0, camera.y0) in pixel coordinates from the center of the image. 

    The Kaolin camera class does not enforce a coordinate space for how the principal point is specified,
    so users will need to make sure that the correct principal point conventions are followed for 
    the cameras passed into this function.

    Args:
        camera (kaolin.render.camera): The camera class. 
        coords_grid (torch.FloatTensor): Grid of coordinates of shape [H, W, 2].

    Returns:
        (wisp.core.Rays): The generated pinhole rays for the camera.
    """
    if camera.device != coords_grid[0].device:
        raise Exception(f"Expected camera and coords_grid[0] to be on the same device, but found {camera.device} and {coords_grid[0].device}.")
    if camera.device != coords_grid[1].device:
        raise Exception(f"Expected camera and coords_grid[1] to be on the same device, but found {camera.device} and {coords_grid[1].device}.")
    # coords_grid should remain immutable (a new tensor is implicitly created here)
    pixel_y, pixel_x = coords_grid
    pixel_x = pixel_x.to(camera.device, camera.dtype)
    pixel_y = pixel_y.to(camera.device, camera.dtype)

    # Account for principal point (offsets from the center)
    pixel_x = pixel_x - camera.x0
    pixel_y = pixel_y + camera.y0

    # pixel values are now in range [-1, 1], both tensors are of shape res_y x res_x
    pixel_x, pixel_y = _to_ndc_coords(pixel_x, pixel_y, camera)

    ray_dir = torch.stack((pixel_x * camera.tan_half_fov(CameraFOV.HORIZONTAL),
                           -pixel_y * camera.tan_half_fov(CameraFOV.VERTICAL),
                           -torch.ones_like(pixel_x)), dim=-1)

    ray_dir = ray_dir.reshape(-1, 3)    # Flatten grid rays to 1D array
    ray_orig = torch.zeros_like(ray_dir)

    # Transform from camera to world coordinates
    ray_orig, ray_dir = camera.extrinsics.inv_transform_rays(ray_orig, ray_dir)
    ray_dir /= torch.linalg.norm(ray_dir, dim=-1, keepdim=True)
    ray_orig, ray_dir = ray_orig[0], ray_dir[0]  # Assume a single camera

    return ray_orig, ray_dir, camera.near,camera.far

class RendererCore:
    """
    The original purpose of this class was to collect various possible neural renderers together, including those using scene_graphs;
    However, our own general_volume_rendering.py and general_rendering.py already support operations with scene_graphs, and they are not integrated here yet.
    TODO: Consider integrating the rendering output of our own scene graph framework into this class in the future.
    """
    def __init__(self, state:ConfigDict, neural_renderer) -> None:
        
        self.state = state
        self.device = state.renderer.device

        # Create a camera for user view
        self.camera = self._setup_camera(state)
        # self._camera_layers = CameraDatalayers()

        # Set up the list of available bottom level object renderers, according to scene graph
        # self._renderers = None
        # self.refresh_bl_renderers(state)
        
        

        # self._tlas = self._setup_tlas(state)

        self.res_x, self.res_y = None, None
        self.set_full_resolution()

        self._last_state = dict()
        self._last_renderbuffer = None

        # Minimal resolution supported by RendererCore
        self.MIN_RES = 128
        
        self.neural_renderer = neural_renderer

    def _setup_camera(self, state: ConfigDict):
        # Use selected camera to control canvas
        camera = state.renderer.selected_camera
        if camera is None:
            # # Create a default camera
            # lens_type = self.state.renderer.selected_camera_lens
            # camera = self._default_camera(lens_type)
            raise NotImplementedError

        camera = camera.to(self.device)
        return camera

    def set_full_resolution(self):
        self.res_x = self.camera.width
        self.res_y = self.camera.height
        # self.interactive_mode = False

    def set_low_resolution(self):
        self.res_x = self.camera.width // 4
        self.res_y = self.camera.height // 4
        # self.interactive_mode = True

    def raygen(self, camera, res_x, res_y):
        ray_grid = generate_centered_pixel_coords(camera.width, camera.height, res_x, res_y, device=self.device)
        if camera.lens_type == 'pinhole':
            rays = generate_pinhole_rays(camera, ray_grid)
        # elif camera.lens_type == 'ortho':
        #     rays = generate_ortho_rays(camera, ray_grid)
        else:
            raise ValueError(f'RenderCore failed to raygen on unknown camera lens type: {camera.lens_type}')
        return rays

    def resize_canvas(self, width, height):
        self.camera.intrinsics.width = width
        self.camera.intrinsics.height = height
        # self.set_low_resolution()
        self.set_full_resolution()

    def _update_scene_graph(self):
        """ Update scene graph information about objects and their data layers """
        # New data layers maybe have been generated, update the scene graph about their existence.
        # Some existing layers may have been regenerated, in which case we copy their previous "toggled on/off" status.
        # data_layers = self._bl_renderer_data_layers()
        # for object_id, bl_renderer_state in self.state.graph.bl_renderers.items():
        #     object_layers = data_layers[object_id]
        #     bl_renderer_state.data_layers = object_layers
        #     toggled_data_layers = defaultdict(bool)
        #     for layer_name, layer in object_layers.items():  # Copy over previous toggles if such exist
        #         toggled_data_layers[layer_name] = bl_renderer_state.toggled_data_layers.get(layer_name, False)
        #     bl_renderer_state.toggled_data_layers = toggled_data_layers
        
        object_layers = self._bl_renderer_data_layers()['neural']
        for layer_name, layer in object_layers.items():
            self.state.nr_data_layers[layer_name] = layer

    @torch.no_grad()
    def redraw(self) -> None:
        """ Allow bottom-level renderers to refresh internal information, such as data layers. """
        # Read phase: sync with scene graph, create renderers for new objects added
        # self.refresh_bl_renderers(self.state)

        # Invoke internal redraw() on all current visible renderers, to imply it's time to refresh content
        # (i.e. data layers may get generate here behind the scenes)
        # scene_graph = self.state.graph
        # for obj_id, renderer in self._renderers.items():
        #     if obj_id in scene_graph.visible_objects:
        #         renderer.redraw()

        # Write phase: update scene graph back with latest info from render core, i.e. new data layers generated
        self._update_scene_graph()

    @torch.no_grad()
    def render(self, time_delta=None, force_render=False):
        camera = self.camera
        clear_color = self.state.renderer.clear_color_value
        res_x, res_y = self.res_x, self.res_y
        
        # Generate rays
        rays_o, rays_d, near, far = self.raygen(camera, res_x, res_y)
        
        # Invoke ray query
        ret = self.neural_renderer.render(rays_o, rays_d, near, far, res_x, res_y)
        return ret

    def _bl_renderer_data_layers(self) -> Dict[str, PrimitivesPack]:
        """ Returns the bottom level object data layers"""
        layers = dict()
        # for renderer_id, renderer in self._renderers.items():
        #     layers[renderer_id] = renderer.data_layers()
        # return layers
        layers['neural'] = self.neural_renderer.data_layers()
        
        return layers

    def active_data_layers(self) -> List[PrimitivesPack]:
        layers_to_draw = []
        
        # for obj_state in self.state.graph.bl_renderers.values():
        #     for layer_id, layer in obj_state.data_layers.items():
        #         if obj_state.toggled_data_layers[layer_id]:
        #             layers_to_draw.append(layer)
        
        for layer_id, layer in self.state.nr_data_layers.items():
            if True:
                layers_to_draw.append(layer)
        
        # camera_data_layers = self._cameras_data_layers()
        # layers_to_draw.extend(camera_data_layers)
        
        return layers_to_draw