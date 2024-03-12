"""
Borrowed from https://github.com/NVIDIAGameWorks/kaolin-wisp
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
from copy import deepcopy
from glumpy import app, gloo, gl, ext


class CameraControlMode:
    def __init__(self, render_core, state):
        self.state = state
        self.render_core = render_core

        self.PI = torch.tensor(np.pi, device=render_core.device)
        self.planes_forbidden_zooming_through = []

        #-----------------------------------
        # Sensitivity & velocity parameters
        #-----------------------------------
        # The controller uses kinematics for smooth FPS considerate transitions
        # Pan operations with the keyboard:
        self._key_pan_initial_velocity = 0.8           # start with this v0 velocity
        self._key_pan_deacceleration = 3.2             # deaccelerate with this acceleration factor a ~ 4*v0
        self._key_pan_distance_weight = 0.5            # pan amount is distance dependant, weighted by this factor

        # Pan operations with the mouse:
        self._mouse_pan_distance_weight = 0.002        # pan amount is distance dependant, weighted by this factor

        # Zoom operations with the mouse:
        self._mouse_zoom_initial_velocity = 10.0       # start with this v0 velocity
        self._mouse_zoom_deacceleration = 40.0         # deaccelerate with this acceleration factor a ~ 4*v0
        self._zoom_persp_distance_weight = 0.25        # zoom is cam-pos dependant, weighted by this factor
        self._zoom_ortho_distance_weight = 0.2         # zoom is fov-distance dependant, weighted by this factor
        self._zoom_ortho_fov_dist_range = (1e-4, 1e2)  # zoom in ortho mode is limited to this fov-distance range

        # Current state
        self._current_pan_velocity = 0
        self._current_pan_direction = None
        self._remaining_pan_time = 0

        self.interactions_stack = list()

    def handle_timer_tick(self, dt):
        self.progress_pan(dt)

    def handle_mouse_scroll(self, x, y, dx, dy):
        """ The mouse wheel was scrolled by (dx,dy). """
        self.stop_all_current_interactions()
        if dy < 0:
            self.start_pan(pan_direction='forward',
                           velocity=self._mouse_zoom_initial_velocity, deaccelaration=self._mouse_zoom_deacceleration)
        else:
            self.start_pan(pan_direction='backward',
                           velocity=self._mouse_zoom_initial_velocity, deaccelaration=self._mouse_zoom_deacceleration)

    # def handle_key_press(self, symbol, modifiers):
    #     self.stop_all_current_interactions()
    #     if symbol == WispKey.LEFT:
    #         self.start_pan(pan_direction='left',
    #                        velocity=self._key_pan_initial_velocity, deaccelaration=self._key_pan_deacceleration)
    #     elif symbol == WispKey.RIGHT:
    #         self.start_pan(pan_direction='right',
    #                        velocity = self._key_pan_initial_velocity, deaccelaration=self._key_pan_deacceleration)
    #     elif symbol == WispKey.UP:
    #         self.start_pan(pan_direction='up',
    #                        velocity=self._key_pan_initial_velocity, deaccelaration=self._key_pan_deacceleration)
    #     elif symbol == WispKey.DOWN:
    #         self.start_pan(pan_direction='down',
    #                        velocity=self._key_pan_initial_velocity, deaccelaration=self._key_pan_deacceleration)

    def start_pan(self, pan_direction, velocity=None, deaccelaration=None):
        self.start_interaction(f'pan_{pan_direction}')
        self._current_pan_velocity = velocity
        self._current_pan_deacceleration = deaccelaration
        self._current_pan_direction = pan_direction
        self._remaining_pan_time = abs(self._current_pan_velocity / self._current_pan_deacceleration)
        if pan_direction in ('left', 'down', 'backward'):  # Directions that move opposite to camera axes
            self._current_pan_velocity *= -1

    def zoom(self, amount):
        if self.camera.lens_type == 'ortho':
            # Under orthographic projection, objects are not affected by distance to the camera
            amount = self._zoom_ortho_distance_weight * self.camera.fov_distance * abs(amount) * np.sign(amount)
            self.camera.zoom(amount)
            # Keep distance at reasonable range
            self.camera.fov_distance = torch.clamp(self.camera.fov_distance, *self._zoom_ortho_fov_dist_range)
        else:
            dist = self.camera.cam_pos().norm()
            amount *= self._zoom_persp_distance_weight * dist
            self.camera.move_forward(amount)

    def progress_pan(self, dt):
        if self._current_pan_direction is None or self._current_pan_velocity == 0:
            return
        dt = min(self._remaining_pan_time, dt)
        pos_delta = dt * self._current_pan_velocity
        if self._current_pan_direction in ('forward', 'backward'):
            cam_pos, cam_forward = self.camera.cam_pos().squeeze(), self.camera.cam_forward().squeeze()
            new_pos = (cam_pos + cam_forward * pos_delta)
            if ('xz' in self.planes_forbidden_zooming_through and new_pos[1].sign() * cam_pos[1].sign() == -1) or \
                ('xy' in self.planes_forbidden_zooming_through and new_pos[2].sign() * cam_pos[2].sign() == -1) or \
                ('yz' in self.planes_forbidden_zooming_through and new_pos[0].sign() * cam_pos[0].sign() == -1):
                self._remaining_pan_time = 0
            else:
                self.zoom(pos_delta)
        elif self._current_pan_direction in ('right', 'left'):
            dist = self.camera.cam_pos().norm()
            pos_delta *= self._key_pan_distance_weight * dist
            self.camera.move_right(pos_delta)
        elif self._current_pan_direction in ('up', 'down'):
            dist = self.camera.cam_pos().norm()
            pos_delta *= self._key_pan_distance_weight * dist
            self.camera.move_up(pos_delta)
        velocity_sign = np.sign(self._current_pan_velocity)
        deaccel_amount = velocity_sign * self._current_pan_deacceleration * dt
        self._current_pan_velocity -= deaccel_amount
        self._remaining_pan_time = max(0, self._remaining_pan_time - dt)
        if np.sign(self._current_pan_velocity) != velocity_sign or \
                self._current_pan_velocity == 0 or self._remaining_pan_time == 0:
            self.end_pan()

    def end_pan(self):
        self.end_interaction()
        if not self.is_interacting():   # End only if other interactions have not taken place meanwhile
            self._current_pan_velocity = 0
            self._current_pan_direction = None
            self._remaining_pan_time = 0

    def handle_key_release(self, symbol, modifiers):
        pass

    def handle_mouse_press(self, x, y, button):
        self.start_interaction('pan_withmouse')

    def handle_mouse_drag(self, x, y, dx, dy, button):
        dist_normalization = self._mouse_pan_distance_weight * self.camera.cam_pos().norm()
        self.camera.move_right(dist_normalization * -dx)
        self.camera.move_up(dist_normalization * dy)

    def handle_mouse_release(self, x, y, button):
        self.end_pan()

    def handle_mouse_motion(self, x, y, dx, dy):
        """ The mouse was moved with no buttons held down. """
        pass

    def start_interaction(self, interaction_id):
        self.interactions_stack.append(interaction_id)

    def end_interaction(self):
        # On some occassions the app may be out of focus and some interactions are not registered properly.
        # Silently ignore that.
        if len(self.interactions_stack) > 0:
            self.interactions_stack.pop()

    def stop_all_current_interactions(self):
        while self.is_interacting():
            last_interaction_started = self.get_last_interaction_started()
            if last_interaction_started.startswith('pan'):
                self.end_pan()
            else:
                self.end_interaction()

    def is_interacting(self):
        return len(self.interactions_stack) > 0

    def get_last_interaction_started(self):
        return self.interactions_stack[-1] if self.is_interacting() else None

    def has_interaction(self, interaction_id):
        return interaction_id in self.interactions_stack

    @property
    def camera(self):
        return self.render_core.camera


def quat_mul(Q1, Q2):
    return torch.tensor([Q1[0] * Q2[3] + Q1[3] * Q2[0] - Q1[2] * Q2[1] + Q1[1] * Q2[2],
                         Q1[1] * Q2[3] + Q1[2] * Q2[0] + Q1[3] * Q2[1] - Q1[0] * Q2[2],
                         Q1[2] * Q2[3] - Q1[1] * Q2[0] + Q1[0] * Q2[1] + Q1[3] * Q2[2],
                         Q1[3] * Q2[3] - Q1[0] * Q2[0] - Q1[1] * Q2[1] - Q1[2] * Q2[2]])


def quat_matrix(q): # True only for unit quaternions
    xx = q[0] * q[0]
    xy = q[0] * q[1]
    xz = q[0] * q[2]
    xw = q[0] * q[3]
    yy = q[1] * q[1]
    yz = q[1] * q[2]
    yw = q[1] * q[3]
    zz = q[2] * q[2]
    zw = q[2] * q[3]
    ww = q[3] * q[3]
    return torch.tensor([[ww + xx - yy - zz, 2.0 * (xy - zw), 2.0 * (xz + yw), 0.0],
                         [2.0 * (xy + zw), ww - xx + yy - zz, 2.0 * (yz - xw), 0.0],
                         [2.0 * (xz - yw), 2.0 * (yz + xw), ww - xx - yy + zz, 0.0],
                         [0.0, 0.0, 0.0, 1.0]], dtype=torch.float)


class TrackballCameraMode(CameraControlMode):

    def __init__(self, render_core, state):
        super().__init__(render_core, state)
        self.radius = 1.0
        self.tb_scale = 1.1
        self.sensitivity = 1.0
        self.reset_center_of_focus(reset_radius=True)

        self.q = torch.tensor([0.0, 0.0, 0.0, 1.0])
        self.q0 = torch.tensor([0.0, 0.0, 0.0, 1.0])
        self.q1 = None
        self.v0 = None
        self.v1 = None

        self.initial_camera = None
        self.planes_forbidden_zooming_through = self.state.renderer.reference_grids

    @classmethod
    def name(cls):
        return "Trackball"
 
    def reset_center_of_focus(self, reset_radius=False):
        pos = self.camera.cam_pos().squeeze()
        forward_axis = self.camera.cam_forward().squeeze()
        forward_axis /= forward_axis.norm()
        if reset_radius:
            self.radius = torch.dot(pos, forward_axis)
        self.focus_at = pos - self.radius * forward_axis   

    def mouse2vector(self, mx, my):
        """ Converts mouse position in screen coordinates:
            [0,0] at top left of screen to [width, height] at bottom-right
            to coordinates on arcball
        """
        # First we convert the mouse coordinates to normalized device coordinates, multiplied by the arcball radius.
        # v is a vector of (x,y,z), and will contain coordinates projected to the arcball surface
        # TODO: Is x and y rotated inversely?
        half_width = 0.5 * self.camera.width
        half_height = 0.5 * self.camera.height
        v = torch.tensor([half_width - float(mx), float(my) - half_height, 0.0])
        normalization_factor = half_height if half_width >= half_height else half_width
        v *= self.tb_scale / float(normalization_factor)

        # v is now in Normalized Device Coordinates
        # Next we need to calculate the z coordinate, which is currently set to 0

        # xy_power = x^2 + y^2
        xy_power = torch.pow(v, 2).sum()

        if xy_power < 1.0:
            v[2] = -torch.sqrt(1.0 - xy_power)   # z = -sqrt(R - x^2 - y^2)
        else:
            v /= torch.sqrt(xy_power)   # x = x/sqrt(x^2 + y^2) ; y = y/sqrt(x^2 + y^2) ; z = 0.0

        return v

    def handle_mouse_press(self, x, y, button):
        self.stop_all_current_interactions()
        if button == app.window.mouse.LEFT:
            self.start_interaction('trackball_rotate')
            self.initial_camera = deepcopy(self.camera)
            self.v0 = self.mouse2vector(x, y)
        elif button == app.window.mouse.MIDDLE:
            super().handle_mouse_press(x, y, button)

    def handle_mouse_release(self, x, y, button):
        if self.has_interaction('trackball_rotate'):
            self.end_interaction()
            self.q0 = torch.tensor([0.0, 0.0, 0.0, 1.0])
            self.initial_camera = None
        else:
            super().handle_mouse_release(x, y, button)

    def handle_mouse_drag(self, x, y, dx, dy, button):
        if not self.has_interaction('trackball_rotate'):
            super().handle_mouse_drag(x, y, dx, dy, button)
        else:
            camera = self.camera

            if torch.allclose(self.v0, torch.zeros_like(self.v0)):
                self.v0 = self.mouse2vector(x, y)   # Start position

            self.v1 = self.mouse2vector(x, y)       # Current position

            # find quaterion for previous to current frame rotation
            axis = torch.cross(self.v1, self.v0)
            angle = -torch.dot(self.v1, self.v0).unsqueeze(0)
            angle *= self.sensitivity
            self.q1 = torch.cat((axis, angle))

            # Apply rotation to starting rotation quaternion
            self.q = quat_mul(self.q1, self.q0)

            # To rotation matrix
            Q = quat_matrix(self.q)
            Q = Q.to(camera.device)

            # find length of vector from eye to focus at point
            pos = self.initial_camera.cam_pos().reshape(-1)
            vec = self.focus_at - pos
            length = torch.sqrt(torch.dot(vec, vec))

            # create translation in z direction to/from focua at point (in camera space)
            T = torch.eye(4, device=camera.device, dtype=camera.dtype).repeat(1, 1)
            T[2, 3] = length
            Tinv = torch.eye(4, device=camera.device, dtype=camera.dtype).repeat(1, 1)
            Tinv[2, 3] = -length

            view_matrix = self.initial_camera.view_matrix()[0]   # Unbatch

            # apply transforms
            view_matrix = Tinv @ Q @ T @ view_matrix

            # update extrinsic matrix
            camera.update(view_matrix)

