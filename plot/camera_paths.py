"""
@file   camera_paths.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Camera trajectory/paths generation tools
"""

__all__ = [
    "get_path_spherical_spiral", 
    "get_path_spherical_spiral_extend_half", 
    "get_path_small_circle", 
    "get_path_front_left_lift_then_spiral_forward", 
    "get_path_interpolation"
]

import math
import numpy as np
from numbers import Number
from typing import Literal, Union

from scipy import interpolate
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R

import torch

from nr3d_lib.models.attributes.transform import TransformMat4x4
from nr3d_lib.utils import check_to_torch
from nr3d_lib.geometry import normalize, look_at_opencv


def _smoothed_motion_interpolation(full_range, num_samples, uniform_proportion=1/3.):
    half_acc_proportion = (1-uniform_proportion) / 2.
    num_uniform_acc = max(math.ceil(num_samples*half_acc_proportion), 2)
    num_uniform = max(math.ceil(num_samples*uniform_proportion), 2)
    num_samples = num_uniform_acc * 2 + num_uniform
    seg_velocity = np.arange(num_uniform_acc)
    seg_angle = np.cumsum(seg_velocity)
    # NOTE: full angle = 2*k*x_max + k*v_max*num_uniform
    ratio = full_range / (2.0*seg_angle.max()+seg_velocity.max()*num_uniform)
    # uniform acceleration sequence
    seg_acc = seg_angle * ratio

    acc_angle = seg_acc.max()
    # uniform sequence
    seg_uniform = np.linspace(acc_angle, full_range-acc_angle, num_uniform+2)[1:-1]
    # full sequence
    all_samples = np.concatenate([seg_acc, seg_uniform, full_range-np.flip(seg_acc)])
    return all_samples

def get_path_spherical_spiral(
    three_cam_centers_ref: np.ndarray, # [3, 3] Three camera locations on a small circle for reference, CCW order
    num_frames: int, # Number of frames / waypoints to generate
    n_rots: float = 2.2, # Number of spins / rotation rounds
    up_angle_start: float = 0., # The starting lift angle (in rad)
    up_angle: float = np.pi / 3.,  # The ending lift angle (in rad)
    focus_center: Union[Literal['origin', 'small_circle_center'], np.ndarray, Number] = 'origin', 
    verbose=False, **verbose_kwargs
    ) -> np.ndarray:
    """
    https://en.wikipedia.org/wiki/Spiral#Spherical_spirals
        assume three input views are on a small circle, then generate a spherical spiral path based on the small circle
    """
    centers = three_cam_centers_ref
    centers_norm = np.linalg.norm(centers, axis=-1)
    radius = np.max(centers_norm)
    centers = centers * radius / centers_norm
    vec0 = centers[1] - centers[0]
    vec1 = centers[2] - centers[0]
    # The axis vertical to the small circle's area
    up_vec = normalize(np.cross(vec0, vec1))
    
    # Key rotations of a spherical spiral path
    sphere_thetas = np.linspace(0, np.pi * 2. * n_rots, num_frames)
    sphere_phis = np.linspace(up_angle_start, up_angle, num_frames)
    
    if isinstance(focus_center, str):
        if focus_center == 'origin':
            # Use the origin as the focus center
            focus_center = np.zeros([3])
        elif focus_center == 'small_circle_center':
            # Use the center of the small circle as the focus center
            focus_center = np.dot(up_vec, centers[0]) * up_vec
        else:
            raise RuntimeError(f"Invalid focus_center={focus_center}")
    elif isinstance(focus_center, Number):
        # A height vertically lifted from the center of the small circle
        focus_center = focus_center * up_vec
    else:
        # A absolute coordinate
        focus_center = np.asarray(focus_center)
    
    # First rotate about up vec
    rots_theta = R.from_rotvec(sphere_thetas[:, None] * up_vec[None, :])
    render_centers = rots_theta.apply(centers[0])
    # Then rotate about horizontal vec
    horizontal_vec = normalize(np.cross(render_centers-focus_center[None, :], up_vec[None, :], axis=-1))
    rots_phi = R.from_rotvec(sphere_phis[:, None] * horizontal_vec)
    render_centers = rots_phi.apply(render_centers)
    
    # c2w
    render_c2ws_all = look_at_opencv(render_centers, focus_center[None, :], up=up_vec)

    def debug_vis(intrs: np.ndarray, H: int, W: int, cam_size: float=0.1, font_size: int = 12):
        import matplotlib
        import matplotlib.pyplot as plt
        from nr3d_lib.plot import vis_camera_mplot
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        # ax.set_aspect("equal")
        ax.set_aspect("auto")    

        matplotlib.rcParams.update({'font.size': font_size})
        #----------- Draw cameras
        vis_camera_mplot(ax, intrs, render_c2ws_all, 
            H=H, W=W, cam_size=cam_size, 
            annotation=True, per_cam_axis=False)

        radius = np.linalg.norm(centers[0])
        
        #----------- Draw small circle
        # key rotations of a spherical spiral path
        num_pts = int(n_rots * 180.)
        sphere_thetas = np.linspace(0, np.pi * 2. * n_rots, num_pts)
        sphere_phis = np.linspace(0, up_angle, num_pts)
        # first rotate about up vec
        rots_theta = R.from_rotvec(sphere_thetas[:, None] * up_vec[None, :])
        pts = rots_theta.apply(centers[0])
        # then rotate about horizontal vec
        horizontal_vec = normalize(np.cross(pts-focus_center[None, :], up_vec[None, :], axis=-1))
        rots_phi = R.from_rotvec(sphere_phis[:, None] * horizontal_vec)
        pts = rots_phi.apply(pts)
        # [x, y, z]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color='black')
        
        #----------- Draw sphere
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = radius * np.outer(np.cos(u), np.sin(v))
        y = radius * np.outer(np.sin(u), np.sin(v))
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='grey', linewidth=0, alpha=0.1)
        
        #----------- Draw axis
        axis = np.array([[0,0,0,1,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1]])
        X, Y, Z, U, V, W = zip(*axis) 
        ax.quiver(X[0], Y[0], Z[0], U[0], V[0], W[0], color='red')
        ax.quiver(X[1], Y[1], Z[1], U[1], V[1], W[1], color='green')
        ax.quiver(X[2], Y[2], Z[2], U[2], V[2], W[2], color='blue')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        
        plt.show()

    if verbose:
        debug_vis(**verbose_kwargs)

    return render_c2ws_all

def get_path_small_circle(
    three_cam_centers_ref: np.ndarray, # [3, 3] Three camera locations on a small circle for reference, CCW order
    num_frames: int, # Number of frames / waypoints to generate
    verbose=False, **verbose_kwargs
    ) -> np.array:
    centers = three_cam_centers_ref
    centers_norm = np.linalg.norm(centers, axis=-1)
    radius = np.max(centers_norm)
    centers = centers * radius / centers_norm
    vec0 = centers[1] - centers[0]
    vec1 = centers[2] - centers[0]
    # the axis vertical to the small circle
    up_vec = normalize(np.cross(vec0, vec1))
    # length of the chord between c0 and c2
    len_chord = np.linalg.norm(vec1, axis=-1)
    # angle of the smaller arc between c0 and c1
    full_angle = np.arcsin(len_chord/2/radius) * 2.
    
    all_angles = _smoothed_motion_interpolation(full_angle, num_frames)
    
    rots = R.from_rotvec(all_angles[:, None] * up_vec[None, :])
    centers = rots.apply(centers[0])
    
    # get c2w matrices
    render_c2ws_all = look_at_opencv(centers, np.zeros_like(centers), up=up_vec)
    
    def debug_vis(intrs: np.ndarray, H: int, W: int, cam_size: float=0.1, font_size: int = 12):
        import matplotlib
        import matplotlib.pyplot as plt
        from nr3d_lib.plot import vis_camera_mplot
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        # ax.set_aspect("equal")
        ax.set_aspect("auto")    

        matplotlib.rcParams.update({'font.size': font_size})
        #----------- draw cameras
        vis_camera_mplot(ax, intrs, render_c2ws_all, 
            H=H, W=W, cam_size=cam_size, 
            annotation=True, per_cam_axis=False)

        radius = np.linalg.norm(centers[0])
        
        #----------- Draw small circle
        angles = np.linspace(0, np.pi * 2., 180)
        rots = R.from_rotvec(angles[:, None] * up_vec[None, :])
        # [180, 3]
        pts = rots.apply(centers[0])
        # [x, y, z]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color='black')
        
        #----------- Draw sphere
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = radius * np.outer(np.cos(u), np.sin(v))
        y = radius * np.outer(np.sin(u), np.sin(v))
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='grey', linewidth=0, alpha=0.1)
        
        #----------- Draw axis
        axis = np.array([[0,0,0,1,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1]])
        X, Y, Z, U, V, W = zip(*axis) 
        ax.quiver(X[0], Y[0], Z[0], U[0], V[0], W[0], color='red')
        ax.quiver(X[1], Y[1], Z[1], U[1], V[1], W[1], color='green')
        ax.quiver(X[2], Y[2], Z[2], U[2], V[2], W[2], color='blue')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        
        plt.show()
    
    if verbose:
        debug_vis(**verbose_kwargs)
    
    return render_c2ws_all

def get_path_front_left_lift_then_spiral_forward(
    pose_ref: np.ndarray, # [N0,4,4], Original ego pose; will generate trajectories along this
    num_frames: int, # Number of frames / waypoints to generate
    duration_frames: int = 48, # Number of frames per round / cycle
    # Spiral configs
    up_max: float = 1.8, up_min: float = 0.2, 
    left_max: float = 1.0, left_min: float = 0., 
    elongation: float = 1., first_with_forward=False, 
    forward_vec: np.ndarray = np.array([0., 0., 1.]),  # Frontal direction vector of OpenCV
    up_vec: np.ndarray = np.array([0., -1., 0.]),  # Frontal direction vector of OpenCV
    left_vec: np.ndarray = np.array([-1., 0., 0.]),  # Frontal direction vector of OpenCV
    verbose=False, **verbose_kwargs
    ) -> np.ndarray:
    """
    First lift ego in the front left direction, then do spiral forward
    """
    pose_ref: TransformMat4x4 = TransformMat4x4(pose_ref, device=torch.device('cpu'), dtype=torch.float)
    n_rots = num_frames / duration_frames
    track_ref = pose_ref.translation()
    rot_ref = pose_ref.rotation()
    # NOTE: Convert from observer's coords dir to world coords dir;
    #       If the pose_ref is the pose of camera, `forward/up/left_vec` can be used as-is;
    #           (since the cameras are already OpenCV cameras if dataio's xxx_dataset.py is correctly implemented)
    #       otherwise, you should specify the coords dir vector of your given pose_ref.
    forward_vecs_ref = pose_ref.forward(check_to_torch(forward_vec, ref=pose_ref)) - track_ref
    up_vecs_ref = pose_ref.forward(check_to_torch(up_vec, ref=pose_ref)) - track_ref
    left_vecs_ref = pose_ref.forward(check_to_torch(left_vec, ref=pose_ref)) - track_ref
    
    forward_vecs_ref = forward_vecs_ref.numpy()
    up_vecs_ref = up_vecs_ref.numpy()
    left_vecs_ref = left_vecs_ref.numpy()
    track_ref = track_ref.numpy()
    rot_ref = rot_ref.numpy()
    
    nvs_seqs = []
    
    verti_radius = (up_max - up_min) / 2.
    up_offset = (up_max + up_min) / 2.
    horiz_radius = (left_max - left_min) / 2.
    left_offset = (left_max + left_min) / 2.
    assert (verti_radius >= -1e-5) and (horiz_radius >= -1e-5)
    
    #----------------------------------------
    #---- First: lift up & left
    #----------------------------------------
    first_frames = int(0.25 * duration_frames)+1
    remain_frames = num_frames - first_frames
    pace = np.linalg.norm(track_ref[0] - track_ref[-1]) * elongation
    forward_1st = ((first_frames / num_frames) * pace) if first_with_forward else 0
    
    track = track_ref[0] \
        + np.linspace(0, verti_radius+up_offset, first_frames)[..., None] * up_vecs_ref[0] \
        + np.linspace(0, horiz_radius+left_offset, first_frames)[..., None] * left_vecs_ref[0] \
        + np.linspace(0, forward_1st, first_frames)[..., None] * forward_vecs_ref[0]
    pose = np.eye(4)[None,...].repeat(first_frames, 0)
    pose[:, :3, 3] = track
    pose[:, :3, :3] = rot_ref[0]
    nvs_seqs.append(pose)
    
    #----------------------------------------
    #---- Then:   Sprial forward
    #----------------------------------------
    w = np.linspace(0,1,remain_frames) # [0->1], data key time
    t = np.arange(len(track_ref)) / (len(track_ref)-1) # [0->1], render key time (could be extended)
    track_interp = interpolate.interp1d(t, track_ref, axis=0, fill_value='extrapolate')
    up_vec_interp = interpolate.interp1d(t, up_vecs_ref, axis=0, fill_value='extrapolate')
    left_vec_interp = interpolate.interp1d(t, left_vecs_ref, axis=0, fill_value='extrapolate')
    
    up_vecs_all = up_vec_interp(w * elongation)
    left_vecs_all = left_vec_interp(w * elongation)
    #---- Base: left * [1], up * [1]
    track_base_all = track_interp(w * elongation) + (up_offset+verti_radius) * up_vecs_all + (left_offset+horiz_radius) * left_vecs_all + forward_1st * forward_vecs_ref[None, 0]
    
    # up_vecs_all = np.percentile(up_vecs_ref, w*100, 0)
    # left_vecs_all = np.percentile(left_vecs_ref, w*100, 0)
    # track_base_all = np.percentile(track_ref, w*100, 0) + (up_max+up_offset) * up_vecs_all + (left_max+left_offset) * left_vecs_all
    
    key_rots = R.from_matrix(rot_ref)
    rot_slerp = Slerp(t, key_rots)
    if elongation > 1:
        mask = (w * elongation) < 1.
        rot_base_all = np.eye(3)[None,...].repeat(remain_frames, 0)
        rot_base_all[mask] = rot_slerp(w[mask]).as_matrix()
        rot_base_all[~mask] = rot_ref[-1]
    else:
        rot_base_all = rot_slerp(w).as_matrix()
    
    rads = np.linspace(0, remain_frames / duration_frames * np.pi * 2., remain_frames)
    #---- Spiral: 
    # left: [0, -1, -2, -1, 0] + base: [1] -> [1, 0, -1, 0, 1]
    # up: [0, 1, 0, -1, 0] + base [1] -> [1, 2, 1, 0, 1]
    track = track_base_all \
        + (np.cos(rads)-1)[..., None] * horiz_radius * left_vecs_all\
        + (np.sin(rads))[..., None] * verti_radius * up_vecs_all
    pose = np.eye(4)[None,...].repeat(remain_frames, 0)
    pose[:, :3, 3] = track
    pose[:, :3, :3] = rot_base_all
    nvs_seqs.append(pose)
    
    render_pose_all = np.concatenate(nvs_seqs, 0)
    
    def debug_vis():
        import vedo
        track0 = vedo.Line(track_ref).color("blue")
        track = vedo.Line(render_pose_all[:, :3, 3]).color("red")
        vedo.show(track0, track, axes=1, new=True)
    
    if verbose:
        debug_vis(**verbose_kwargs)
    
    return render_pose_all

def get_path_interpolation(
    pose_ref: np.ndarray, # [N0,4,4], Original ego pose; will generate trajectories along this
    num_frames: int, # Number of frames / waypoints to generate
    ) -> np.ndarray:
    key_rots = R.from_matrix(pose_ref[:, :3, :3])
    key_trans = pose_ref[:, :3, 3]
    key_times = list(range(len(key_rots)))
    slerp = Slerp(key_times, key_rots) # Rotation interpolation
    interp = interpolate.interp1d(key_times, key_trans, axis=0) # Translation interpolation
    
    render_pose_all = []
    for i, time in enumerate(np.linspace(0, len(pose_ref)-1, num_frames)):
        cam_location = interp(time)
        cam_rot = slerp(time).as_matrix()
        c2w = np.eye(4)
        c2w[:3, :3] = cam_rot
        c2w[:3, 3] = cam_location
        render_pose_all.append(c2w)
    
    render_pose_all = np.array(render_pose_all)
    return render_pose_all
