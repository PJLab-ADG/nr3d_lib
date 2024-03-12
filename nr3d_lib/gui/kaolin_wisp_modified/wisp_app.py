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

import math
import torch
import pycuda # git clone https://github.com/inducer/pycuda; ./configure.py --cuda-root=/usr/local/cuda --cuda-enable-gl; python setup.py install
import pycuda.gl
import numpy as np
import imgui
from glumpy import app, gloo, gl, ext
from contextlib import contextmanager

from typing import Optional, Dict

from nr3d_lib.config import ConfigDict
from nr3d_lib.gui.kaolin_wisp_modified.render_core import RendererCore
from nr3d_lib.gui.kaolin_wisp_modified.camera_control import CameraControlMode, TrackballCameraMode
from nr3d_lib.gui.kaolin_wisp_modified.gizmos import PrimitivesPainter, Gizmo, WorldGrid, AxisPainter

@contextmanager
def cuda_activate(img):
    """Context manager simplifying use of pycuda.gl.RegisteredImage.
        Boilerplate code based in part on pytorch-glumpy.
    """
    mapping = img.map()
    yield mapping.array(0, 0)
    mapping.unmap()

class APP():
    def __init__(
        self, 
        window_name="wisp app", 
        state=ConfigDict(
            renderer=ConfigDict(
                antialiasing='msaa_4x', 
                clear_color_value=(0,0,0),
                canvas_height=320, 
                canvas_width=480, 
                target_fps=24, 
                device=torch.device('cuda'), 
                selected_camera=None, 
                selected_camera_lens="perspective", 
                reference_grids=['xz']
            ), 
            nr_data_layers=ConfigDict()
        ), 
        neural_renderer=None
        ):
        self.state = state

        # Create main app window & initialize GL context
        # glumpy with a specialized glfw backend takes care of that (backend is imgui integration aware)
        window = self._create_window(self.width, self.height, window_name)

        # Initialize gui, assumes the window is managed by glumpy with glfw
        imgui.create_context()
        self.canvas_dirty = False

        render_core = RendererCore(self.state, neural_renderer)
        
        self.window = window                    # App window with a GL context & oversees event callbacks
        self.render_core = render_core          # Internal renderer, responsible for painting over canvas
        self.render_clock = app.clock.Clock()
        self.render_clock.tick()
        self.interactions_clock = app.clock.Clock()
        self.interactions_clock.tick()
        self._was_interacting_prev_frame = False

        # The initialization of these fields is deferred util "on_resize" is first prompted.
        # There we generate a simple billboard GL program with a shared CUDA resource
        # Canvas content will be blitted onto it
        self.cuda_buffer: Optional[pycuda.gl.RegisteredImage] = None    # CUDA buffer, as a shared resource with OpenGL
        self.depth_cuda_buffer: Optional[pycuda.gl.RegisteredImage] = None
        self.canvas_program: Optional[gloo.Program] = None              # GL program used to paint a single billboard
        
        self.user_mode = TrackballCameraMode(self.render_core, self.state)

        self.gizmos = self.create_gizmos()          # Create canvas widgets for this app
        self.prim_painter = PrimitivesPainter()

        self.redraw()

    def run(self):
        """ Initiate events message queue """
        app.run()   # App clock should always run as frequently as possible (background tasks should not be limited)

    def _create_window(self, width, height, window_name):
        # Currently assume glfw backend due to integration with imgui
        app.use("glfw_imgui")

        win_config = app.configuration.Configuration()
        if self.state.renderer.antialiasing == 'msaa_4x':
            win_config.samples = 4

        # glumpy implicitly sets the GL context as current
        window = app.Window(width=width, height=height, title=window_name, config=win_config)
        window.on_draw = self.on_draw
        window.on_resize = self.on_resize
        # window.on_key_press = self.on_key_press
        # window.on_key_release = self.on_key_release
        window.on_mouse_press = self.on_mouse_press
        window.on_mouse_drag = self.on_mouse_drag
        window.on_mouse_release = self.on_mouse_release
        window.on_mouse_scroll = self.on_mouse_scroll
        # window.on_mouse_motion = self.on_mouse_motion

        if self.state.renderer.antialiasing == 'msaa_4x':
            gl.glEnable(gl.GL_MULTISAMPLE)

        return window

    @staticmethod
    def _create_gl_depth_billboard_program(texture: np.ndarray, depth_texture: np.ndarray):
        vertex = """
                    uniform float scale;
                    attribute vec2 position;
                    attribute vec2 texcoord;
                    varying vec2 v_texcoord;
                    void main()
                    {
                        v_texcoord = texcoord;
                        gl_Position = vec4(scale*position, 0.0, 1.0);
                    } """

        fragment = """
                    uniform sampler2D tex;
                    uniform sampler2D depth_tex;
                    varying vec2 v_texcoord;
                    void main()
                    {
                        gl_FragColor = texture2D(tex, v_texcoord);
                        gl_FragDepth = texture2D(depth_tex, v_texcoord).r;
                    } """
        # TODO (operel): r component is a waste?

        # Compile GL program
        canvas = gloo.Program(vertex, fragment, count=4)

        # Upload fixed values to GPU
        canvas['position'] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
        canvas['texcoord'] = [(0, 0), (0, 1), (1, 0), (1, 1)]
        canvas['scale'] = 1.0
        canvas['tex'] = texture
        canvas['depth_tex'] = depth_texture
        return canvas

    @staticmethod
    def _create_cugl_shared_texture(res_h, res_w, channel_depth, map_flags=pycuda.gl.graphics_map_flags.WRITE_DISCARD,
                                    dtype=np.uint8):
        """ Create and return a Texture2D with gloo and pycuda views. """
        if issubclass(dtype, np.integer):
            tex = np.zeros((res_h, res_w, channel_depth), dtype).view(gloo.Texture2D)
        elif issubclass(dtype, np.floating):
            tex = np.zeros((res_h, res_w, channel_depth), dtype).view(gloo.TextureFloat2D)
        else:
            raise ValueError(f'_create_cugl_shared_texture invoked with unsupported texture dtype: {dtype}')
        tex.activate()  # Force gloo to create on GPU
        tex.deactivate()
        cuda_buffer = pycuda.gl.RegisteredImage(int(tex.handle), tex.target,
                                                map_flags)  # Create shared GL / CUDA resource
        return tex, cuda_buffer

    def _blit_to_gl_renderbuffer(self, img, depth_img, canvas_program, cuda_buffer, depth_cuda_buffer, height):
        shared_tex = canvas_program['tex']
        shared_tex_depth = canvas_program['depth_tex']

        # copy from torch into buffer
        assert shared_tex.nbytes == img.numel() * img.element_size()
        assert shared_tex_depth.nbytes == depth_img.numel() * depth_img.element_size()    # TODO: using a 4d tex
        cpy = pycuda.driver.Memcpy2D()
        with cuda_activate(cuda_buffer) as ary:
            cpy.set_src_device(img.data_ptr())
            cpy.set_dst_array(ary)
            cpy.width_in_bytes = cpy.src_pitch = cpy.dst_pitch = shared_tex.nbytes // height
            cpy.height = height
            cpy(aligned=False)
        torch.cuda.synchronize()
        # TODO (operel): remove double synchronize after depth debug
        cpy = pycuda.driver.Memcpy2D()
        with cuda_activate(depth_cuda_buffer) as ary:
            cpy.set_src_device(depth_img.data_ptr())
            cpy.set_dst_array(ary)
            cpy.width_in_bytes = cpy.src_pitch = cpy.dst_pitch = shared_tex_depth.nbytes // height
            cpy.height = height
            cpy(aligned=False)
        torch.cuda.synchronize()

        canvas_program.draw(gl.GL_TRIANGLE_STRIP)






    def render_canvas(self, render_core, time_delta, force_render):
        rgba, depth = render_core.render(time_delta, force_render)
        # buffer_attachment = renderbuffer.image().rgba
        rgba = rgba.flip([0])  # Flip y axis
        img = rgba.byte().contiguous()

        # buffer_attachment_depth = renderbuffer.depth
        depth = depth.flip([0])
        depth_img = depth.repeat(1,1,4).contiguous()

        return img, depth_img

    @torch.no_grad()
    def redraw(self):
        # Let the renderer redraw the data layers if needed
        self.render_core.redraw()

        # Regenerate the GL primitives according to up-to-date data layers
        layers_to_draw = self.render_core.active_data_layers()
        self.prim_painter.redraw(layers_to_draw)

    @torch.no_grad()
    def render(self):
        dt = self.render_clock.tick()  # Tick render clock: dt is now the exact time elapsed since last render

        # Populate the scene state with the most recent information about the interactive renderer.
        # The scene state, for example, may be used by the GUI widgets to display up to date information.
        # self.update_renderer_state(self.wisp_state, dt)

        # Clear color / depth buffers before rendering the next frame
        clear_color = (*self.state.renderer.clear_color_value, 1.0)    # RGBA
        self.window.clear(color=clear_color)

        # imgui renders first
        # self.render_gui(self.wisp_state)

        if self.canvas_dirty:
            self.redraw()

        # Invoke the timer tick event, and let the camera controller update the state of any interactions
        # of the user which involve the time elapsed (i.e: velocity, acceleration of movements).
        self.user_mode.handle_timer_tick(dt)

        if self.user_mode.is_interacting():
            pass
            # self.render_core.set_low_resolution()
        else:
            # Allow a fraction of a second before turning full resolution on.
            # User interactions sometimes behave like a rapid burst of short and quick interactions.
            if self._was_interacting_prev_frame:
                self.interactions_clock.tick()
            time_since_last_interaction = self.interactions_clock.time() - self.interactions_clock.last_ts
            # if time_since_last_interaction > self.COOLDOWN_BETWEEN_RESOLUTION_CHANGES:
            #     self.render_core.set_full_resolution()

        self._was_interacting_prev_frame = self.user_mode.is_interacting()

        # render canvas: core proceeds by invoking internal renderers tracers
        # output is rendered on a Renderbuffer object, backed by torch tensors
        img, depth_img = self.render_canvas(self.render_core, dt, self.canvas_dirty)

        # glumpy code injected within the pyimgui render loop to blit the rendercore output to the actual canvas
        # The torch buffers are copied by pycuda to CUDA buffers, connected as shared resources as 2d GL textures
        self._blit_to_gl_renderbuffer(img, depth_img, self.canvas_program, self.cuda_buffer,
                                      self.depth_cuda_buffer, self.height)

        # Finally, render OpenGL gizmos on the canvas.
        # This may include the world grid, or vectorial lines / points belonging to data layers
        camera = self.render_core.camera
        # for gizmo in self.gizmos.values():
        #     gizmo.render(camera)
        self.prim_painter.render(camera)
        self.canvas_dirty = False




    def create_gizmos(self) -> Dict[str, Gizmo]:
        """ Override to define which gizmos are used by the wisp app. """
        gizmos = dict()
        planes = self.state.renderer.reference_grids
        axes = set(''.join(planes))
        grid_size = 10.0
        for plane in planes:
            gizmos[f'world_grid_{plane}'] = WorldGrid(squares_per_axis=20, grid_size=grid_size,
                                                      line_color=(128, 128, 128), line_size=1, plane=plane)
            gizmos[f'world_grid_fine_{plane}'] = WorldGrid(squares_per_axis=200, grid_size=grid_size,
                                                           line_color=(128, 128, 128), line_size=2, plane=plane)
        # Axes on top of the reference grid
        gizmos['grid_axis_painter'] = AxisPainter(axes_length=grid_size, line_width=1,
                                                  axes=axes, is_bidirectional=False)
        return gizmos



    def on_draw(self, dt=None):
        # dt arg comes from the app clock, the renderer clock is maintained separately from the background tasks
        # Interactive mode on, or interaction have just started
        
        # is_interacting = self.wisp_state.renderer.interactive_mode or self.user_mode.is_interacting()
        is_interacting = self.user_mode.is_interacting()
        if is_interacting or self.is_time_to_render():
            self.render()     # Render objects uploaded to GPU

    def is_time_to_render(self):
        time_since_last_render = self.render_clock.time() - self.render_clock.last_ts
        target_fps = self.state.renderer.target_fps
        if target_fps is None or ((target_fps > 0) and time_since_last_render >= (1 / target_fps)):
            return True
        return False

    def on_resize(self, width, height):
        self.width = width
        self.height = height

        # Handle pycuda shared resources
        if self.cuda_buffer is not None:
            del self.cuda_buffer    # TODO(operel): is this proper pycuda deallocation?
            del self.depth_cuda_buffer
        tex, cuda_buffer = self._create_cugl_shared_texture(height, width, self.channel_depth)
        depth_tex, depth_cuda_buffer = self._create_cugl_shared_texture(height, width, 4, dtype=np.float32)   # TODO: Single channel
        self.cuda_buffer = cuda_buffer
        self.depth_cuda_buffer = depth_cuda_buffer
        if self.canvas_program is None:
            self.canvas_program = self._create_gl_depth_billboard_program(texture=tex, depth_texture=depth_tex)
        else:
            if self.canvas_program['tex'] is not None:
                self.canvas_program['tex'].delete()
            if self.canvas_program['depth_tex'] is not None:
                self.canvas_program['depth_tex'].delete()
            self.canvas_program['tex'] = tex
            self.canvas_program['depth_tex'] = depth_tex

        self.render_core.resize_canvas(height=height, width=width)
        self.window.activate()
        gl.glViewport(0, 0, width, height)
        self._is_reposition_imgui_menu = True   # Signal menu it needs to shift after resize

    def on_mouse_press(self, x, y, button):
        # if self.is_canvas_event():
        self.user_mode.handle_mouse_press(x, y, button)

    def on_mouse_drag(self, x, y, dx, dy, button):
        # if self.is_canvas_event():
        self.user_mode.handle_mouse_drag(x, y, dx, dy, button)

    def on_mouse_release(self, x, y, button):
        # if self.is_canvas_event():
        self.user_mode.handle_mouse_release(x, y, button)

    def on_mouse_scroll(self, x, y, dx, dy):
        """ The mouse wheel was scrolled by (dx,dy). """
        # if self.is_canvas_event():
        self.user_mode.handle_mouse_scroll(x, y, dx, dy)





    @property
    def width(self):
        return self.state.renderer.canvas_width

    @width.setter
    def width(self, value: int):
        self.state.renderer.canvas_width = value

    @property
    def height(self):
        return self.state.renderer.canvas_height

    @height.setter
    def height(self, value: int):
        self.state.renderer.canvas_height = value

    @property
    def channel_depth(self):
        return 4  # Assume the framebuffer keeps RGBA