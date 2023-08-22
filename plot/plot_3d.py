"""
@file   plot_3d.py
@author Jianfei Guo, Shanghai AI Lab
@brief  3D Plot utilities
"""

__all__ = [
    'create_camera_frustum_mplot', 
    'vis_camera_mplot',
    'create_camera_frustum',
    'frustums2lineset_o3d',
    'create_camera_frustum_o3d',
    'vis_camera_o3d',
    'vis_camera_o3d_from_arrays',
    'create_camera_frustum_vedo',
    
    'create_occ_grid_lines',
    'create_occ_grid_lines_from_k',
    'expand_cube_lines_from_corners',
    'create_occ_grid_lineset_o3d',
    'create_occ_grid_mesh',
    'create_occ_grid_mesh_o3d',
    'vis_spc_voxels_o3d',
    'vis_occgrid_voxels_o3d', 
    'create_vox_grid_lines',
    'create_vox_grid_lineset_o3d',
    
    'vis_lidar_vedo',
    'vis_lidar_o3d',
    'get_box_corners',
    'create_box_o3d',
    'vis_lidar_and_boxes_o3d',
    'create_continuous_curve_o3d',
]

import numpy as np
from numbers import Number
from typing import Iterable, List, Tuple, Union

from matplotlib import cm

import torch

from nr3d_lib.fmt import log
from nr3d_lib.geometry import gmean
from nr3d_lib.utils import check_to_torch
from nr3d_lib.plot.plot_basic import *
from nr3d_lib.models.grids.utils import points_to_corners

def create_camera_frustum_mplot(width: Number, height: Number, frustum_length: Number, draw_frame_axis=False):
    """
    OpenCV camera
    """
    # Draw image plane
    X_img_plane = np.ones((4, 5))
    X_img_plane[0:3, 0] = [-width, height, frustum_length]
    X_img_plane[0:3, 1] = [width, height, frustum_length]
    X_img_plane[0:3, 2] = [width, -height, frustum_length]
    X_img_plane[0:3, 3] = [-width, -height, frustum_length]
    X_img_plane[0:3, 4] = [-width, height, frustum_length]

    # Draw triangle above the image plane
    X_triangle = np.ones((4, 3))
    X_triangle[0:3, 0] = [-width, -height, frustum_length]
    X_triangle[0:3, 1] = [0, -2*height, frustum_length] # Above -> because `y` points downward in openCV
    X_triangle[0:3, 2] = [width, -height, frustum_length]

    # Draw camera
    X_center1 = np.ones((4, 2))
    X_center1[0:3, 0] = [0, 0, 0]
    X_center1[0:3, 1] = [-width, height, frustum_length]

    X_center2 = np.ones((4, 2))
    X_center2[0:3, 0] = [0, 0, 0]
    X_center2[0:3, 1] = [width, height, frustum_length]

    X_center3 = np.ones((4, 2))
    X_center3[0:3, 0] = [0, 0, 0]
    X_center3[0:3, 1] = [width, -height, frustum_length]

    X_center4 = np.ones((4, 2))
    X_center4[0:3, 0] = [0, 0, 0]
    X_center4[0:3, 1] = [-width, -height, frustum_length]

    # Draw camera frame axis
    X_frame1 = np.ones((4, 2))
    X_frame1[0:3, 0] = [0, 0, 0]
    X_frame1[0:3, 1] = [frustum_length/2, 0, 0]

    X_frame2 = np.ones((4, 2))
    X_frame2[0:3, 0] = [0, 0, 0]
    X_frame2[0:3, 1] = [0, frustum_length/2, 0]

    X_frame3 = np.ones((4, 2))
    X_frame3[0:3, 0] = [0, 0, 0]
    X_frame3[0:3, 1] = [0, 0, frustum_length/2]

    if draw_frame_axis:
        return [X_img_plane, X_triangle, X_center1, X_center2, X_center3, X_center4, X_frame1, X_frame2, X_frame3]
    else:
        return [X_img_plane, X_triangle, X_center1, X_center2, X_center3, X_center4]

def vis_camera_mplot(
    ax, intrs: np.ndarray, c2ws: np.ndarray, H: int, W: int, 
    cam_size=0.1, annotation=True, per_cam_axis=True, custom_origin=None):
    """
    Modified from https://github.com/opencv/opencv/blob/master/samples/python/camera_calibration_show_extrinsics.py

    example usage:

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        vis_camera_mplot(ax, intrs, c2ws, H, W, cam_size, annotation=True, per_cam_axis=True)
        plt.show()
    """

    # ax.set_aspect("equal")
    ax.set_aspect("auto")

    min_values = np.zeros((3, 1))
    min_values = np.inf
    max_values = np.zeros((3, 1))
    max_values = -np.inf

    fx = intrs[...,0,0].mean()
    fy = intrs[...,1,1].mean()
    cam_width = W/2./fx*cam_size
    cam_height = H/2./fy*cam_size

    # Draw cameras
    # [X_img_plane, X_triangle, X_center1, X_center2, X_center3, X_center4, (X_frame1, X_frame2, X_frame3)]
    X_cam = create_camera_frustum_mplot(cam_width, cam_height, frustum_length=cam_size, draw_frame_axis=per_cam_axis)

    cm_subsection = np.linspace(0.0, 1.0, c2ws.shape[0])
    colors = [cm.jet(x) for x in cm_subsection]

    for idx in range(c2ws.shape[0]):
        c2w = c2ws[idx]
        for i in range(len(X_cam)):
            X = np.zeros(X_cam[i].shape)
            for j in range(X_cam[i].shape[1]):
                X[0:4, j] = c2w.dot(X_cam[i][0:4, j])
            
            if per_cam_axis and i >= (len(X_cam)-3):
                irgb = i-(len(X_cam)-3)
                color = ['red', 'green', 'blue'][irgb]
            else:
                color = colors[idx]
            ax.plot3D(X[0, :], X[1, :], X[2, :], color=color)
            min_values = np.minimum(min_values, X[0:3, :].min(1))
            max_values = np.maximum(max_values, X[0:3, :].max(1))
        # Modified: add an annotation of number
        if annotation:
            X = c2w.dot(X_cam[0][0:4, 0])
            ax.text(X[0], X[1], X[2], str(idx), color=colors[idx])


    # Draw 3d coordinate frame
    # axis_len = (max_values - min_values) / 10.
    # ax.plot3D([min_values[0], min_values[0]+axis_len[0]], [min_values[1], min_values[1]], [min_values[2], min_values[2]], color='red')
    # ax.plot3D([min_values[0], min_values[0]], [min_values[1], min_values[1]+axis_len[1]], [min_values[2], min_values[2]], color='green')
    # ax.plot3D([min_values[0], min_values[0]], [min_values[1], min_values[1]], [min_values[2], min_values[2]+axis_len[2]], color='blue')

    axis_len = ((max_values - min_values) / 10.).max()
    o = [0.,0.,0.] if custom_origin is None else custom_origin
    ax.plot3D([o[0], o[0]+axis_len],[o[1], o[1]],         [o[2], o[2]], color='red')
    ax.plot3D([o[0], o[0]],         [o[1], o[1]+axis_len],[o[2], o[2]], color='green')
    ax.plot3D([o[0], o[0]],         [o[1], o[1]],         [o[2], o[2]+axis_len], color='blue')

    # Set coordinate limits, with equal aspect ratio
    max_range = (max_values - min_values).max()
    mids = (max_values + min_values) / 2.0
    mins = mids - max_range/2.
    maxs = mids + max_range/2.
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    return min_values, max_values

def create_camera_frustum(
    img_wh: Tuple[Number,Number], intr: np.ndarray, c2w: np.ndarray, 
    frustum_length=0.5, color=[0., 1., 0.]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Create camera frustum lineset

    Args:
        img_wh (Tuple[Number,Number]): Viewport size, in pixels
        intr (np.ndarray): [3x3] Pinhole intrinsics mat
        c2w (np.ndarray): [3x4] or [4x4] Camera to world transform mat
        frustum_length (float, optional): Frustum length, in world coords. Defaults to 0.5.
        color (list, optional): Frustum color. Defaults to [0., 1., 0.].

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Frustum lineset (composed of frustum vertices, line indices, and colors of frustum lines)
    """
    W, H = img_wh
    hfov = np.rad2deg(np.arctan(W / 2. / intr[0, 0]) * 2.)
    vfov = np.rad2deg(np.arctan(H / 2. / intr[1, 1]) * 2.)
    half_w = frustum_length * np.tan(np.deg2rad(hfov / 2.))
    half_h = frustum_length * np.tan(np.deg2rad(vfov / 2.))

    # build view frustum for camera (I, 0)
    frustum_points = np.array([[0., 0., 0.],                          # frustum origin
                               [-half_w, -half_h, frustum_length],    # top-left image corner
                               [half_w, -half_h, frustum_length],     # top-right image corner
                               [half_w, half_h, frustum_length],      # bottom-right image corner
                               [-half_w, half_h, frustum_length]])    # bottom-left image corner
    frustum_lines = np.array([[0, i] for i in range(1, 5)] + [[i, (i+1)] for i in range(1, 4)] + [[4, 1]])
    frustum_colors = np.tile(np.array(color).reshape((1, 3)), (frustum_lines.shape[0], 1))

    # frustum_colors = np.vstack((np.tile(np.array([[1., 0., 0.]]), (4, 1)),
    #                            np.tile(np.array([[0., 1., 0.]]), (4, 1))))

    frustum_points = np.dot(np.hstack((frustum_points, np.ones_like(frustum_points[:, 0:1]))), c2w.T)
    frustum_points = frustum_points[:, :3] / frustum_points[:, 3:4]

    return frustum_points, frustum_lines, frustum_colors

def frustums2lineset_o3d(frustums):
    import open3d as o3d
    N = len(frustums)
    merged_points = np.zeros((N*5, 3))      # 5 vertices per frustum
    merged_lines = np.zeros((N*8, 2))       # 8 lines per frustum
    merged_colors = np.zeros((N*8, 3))      # each line gets a color

    for i, (frustum_points, frustum_lines, frustum_colors) in enumerate(frustums):
        merged_points[i*5:(i+1)*5, :] = frustum_points
        merged_lines[i*8:(i+1)*8, :] = frustum_lines + i*5
        merged_colors[i*8:(i+1)*8, :] = frustum_colors

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(merged_points)
    lineset.lines = o3d.utility.Vector2iVector(merged_lines)
    lineset.colors = o3d.utility.Vector3dVector(merged_colors)

    return lineset

def create_camera_frustum_o3d(
    img_wh: Tuple[int,int], intr: np.ndarray, c2w: np.ndarray, 
    frustum_length=0.5, color=[0., 1., 0.]):
    import open3d as o3d
    frustum_points, frustum_lines, frustum_colors = create_camera_frustum(img_wh, intr, c2w, frustum_length, color)
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(frustum_points)
    lineset.lines = o3d.utility.Vector2iVector(frustum_lines)
    lineset.colors = o3d.utility.Vector3dVector(frustum_colors)
    return lineset

def vis_camera_o3d(
    colored_camera_dicts, *,
    cam_size=0.1, sphere_radius=None, per_cam_axis=True,
    geometry_file=None, geometry_type='mesh', backface=False, 
    custom_origin=None, custom_framerot=None, custom_geometry_transform=None,
    key_to_callback={}, show=True):
    """
    Modified from NeRF++.  https://github.com/Kai-46/nerfplusplus/blob/master/colmap_runner/extract_sfm.py
    
    colored_camera_dicts:
        list of dicts: {
            'img_wh': (W,H)
            'intr': intrinsic matrix 
            'c2w'/'w2c': extrinsic matrix
            'color': RGB value of the camera. defaults to sequential color 
        }
    """
    try:
        import open3d as o3d
    except:
        raise RuntimeError("Need open3d installed to run vis_camera_o3d_with_surface.")
    
    things_to_draw = []
    
    if 'color' not in colored_camera_dicts[0]:
        cmap = cm.get_cmap('rainbow')

    c2ws = []
    # Draw cameras
    idx = 0
    for i, camera_dict in enumerate(colored_camera_dicts):
        idx += 1

        intr = np.array(camera_dict['intr'])
        if 'c2w' not in camera_dict:
            assert 'w2c' in camera_dict, 'Need specify at least one of c2w or w2c matrices'
            w2c = np.array(camera_dict['w2c'])
            c2w = np.linalg.inv(w2c)
        else:
            c2w = np.array(camera_dict['c2w'])
        c2ws.append(c2w)
        img_wh = camera_dict['img_wh']
        color = camera_dict.get('color', list(cmap(float(i)/len(colored_camera_dicts))[:3]))
        # camera = frustums2lineset_o3d([create_camera_frustum(img_wh, intr, c2w, frustum_length=cam_size, color=color)])
        camera = frustums2lineset_o3d([create_camera_frustum(img_wh, intr, c2w, frustum_length=cam_size, color=color)])
        things_to_draw.append(camera)

        if per_cam_axis:
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=cam_size/3., origin=c2w[:3,3])
            coord_frame.rotate(c2w[:3,:3])
            things_to_draw.append(coord_frame)
    c2ws = np.array(c2ws)

    # Draw coordinate frame
    max_range = np.linalg.norm(c2ws[..., :3, 3], axis=-1).max()
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=max_range/2., origin=[0., 0., 0.] if custom_origin is None else custom_origin)
    if custom_framerot is not None:
        coord_frame.rotate(custom_framerot[:3,:3])
    things_to_draw.append(coord_frame)

    # Draw an optional sphere
    if sphere_radius is not None and sphere_radius > 0:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius, resolution=10)
        sphere = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
        sphere.paint_uniform_color((1, 0, 0))
        if custom_origin is not None:
            sphere.translate(custom_origin, relative=True)
        things_to_draw.append(sphere)

    # Draw an optional geometry
    if geometry_file is not None:
        if geometry_type == 'mesh':
            geometry = o3d.io.read_triangle_mesh(geometry_file)
            geometry.compute_vertex_normals()
        elif geometry_type == 'pointcloud':
            geometry = o3d.io.read_point_cloud(geometry_file)
        else:
            raise Exception('Unknown geometry_type: ', geometry_type)
        if custom_geometry_transform is not None:
            geometry.transform(custom_geometry_transform)
        things_to_draw.append(geometry)
    if backface:
        o3d.visualization.RenderOption.mesh_show_back_face = True
    
    # Register key callbacks if needed
    if show:
        if len(key_to_callback) == 0:
            o3d.visualization.draw_geometries(things_to_draw)
        else:
            o3d.visualization.draw_geometries_with_key_callbacks(things_to_draw, key_to_callback)
    else:
        return things_to_draw

def vis_camera_o3d_from_arrays(intrs, c2ws, Hs, Ws, **kwargs):
    Hs = np.asarray([Hs] * len(c2ws) if isinstance(Hs, Number) else Hs)
    Ws = np.asarray([Ws] * len(c2ws) if isinstance(Ws, Number) else Ws)
    intrs = np.tile(intrs, (len(c2ws), 1, 1)) if len(intrs.shape)==2 else intrs
    assert len(intrs) == len(c2ws) == len(Hs) == len(Ws)
    camera_dicts = []
    for intr, c2w, H, W in zip(intrs, c2ws, Hs, Ws):
        camera_dicts.append({
            'intr': intr,
            'img_wh': (W, H),
            'c2w': c2w
        })
    vis_camera_o3d(camera_dicts, **kwargs)

def create_camera_frustum_vedo(
    img_wh: Tuple[int,int], intr: np.ndarray, c2w: np.ndarray, 
    frustum_length=0.5, color='k4', lw: int=1):
    from vedo import Lines
    frustum_points, frustum_lines, frustum_colors = create_camera_frustum(
        img_wh, intr=intr, c2w=c2w, frustum_length=frustum_length)
    cam_actor = Lines(frustum_points[frustum_lines[:,0]], frustum_points[frustum_lines[:,1]])
    cam_actor.color(color)
    cam_actor.lw(lw)
    return cam_actor

def create_occ_grid_lines(grid: torch.BoolTensor, origin = None, block_size = None) -> Tuple[np.ndarray, np.ndarray]:
    assert grid.dtype == torch.bool, "Grid must be binary"
    points = grid.nonzero().short()
    return create_occ_grid_lines_from_k(points, origin=origin, block_size=block_size)

def create_occ_grid_lines_from_k(k: torch.ShortTensor, origin = None, block_size = None) -> Tuple[np.ndarray, np.ndarray]:
    corners = points_to_corners(k) * block_size + (origin if origin is not None else 0)
    corners = corners.reshape(-1, 3)
    return corners.data.cpu().numpy(), expand_cube_lines_from_corners(corners)

def expand_cube_lines_from_corners(corners) -> Tuple[np.ndarray, np.ndarray]:
    assert corners.dim() == 2 and corners.shape[-1] == 3, "Expect flattened input corners"
    # [num_voxels, 1, 1] + [12, 2] -> [num_voxels, 12, 2] -> [num_voxels*12, 2]; range: [0, num_voxels*8-1]
    grid_line_idx = torch.tensor([(0, 1), (1, 3), (3, 2), (2, 0),
                                (4, 5), (5, 7), (7, 6), (6, 4),
                                (0, 4), (1, 5), (2, 6), (3, 7)], dtype=torch.long, device=corners.device)
    line_idx = (torch.arange(len(corners), device=corners.device)[:,None,None]*8 + grid_line_idx).reshape(-1,2)
    # [num_voxels, 8, 3] -> [num_voxels*8, 3]
    return line_idx.data.cpu().numpy()

def create_occ_grid_lineset_o3d(grid: torch.BoolTensor, color=light_pink, origin = None, block_size = None):
    import open3d as o3d
    assert grid.dtype == torch.bool, "Grid must be binary"
    points, lines = create_occ_grid_lines(grid, origin=origin, block_size=block_size)
    colors = np.tile(color, (lines.shape[0], 1))
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(points)
    lineset.lines = o3d.utility.Vector2iVector(lines)
    lineset.colors = o3d.utility.Vector3dVector(colors)
    return lineset

def create_occ_grid_mesh(grid: torch.BoolTensor, origin = None, block_size = None) -> Tuple[torch.Tensor, torch.Tensor]:
    from kaolin.ops.conversions import voxelgrids_to_cubic_meshes
    assert grid.dtype == torch.bool, "Grid must be binary"
    mesh_verts, mesh_faces = voxelgrids_to_cubic_meshes(grid.unsqueeze(0))
    mesh_verts = mesh_verts[0] * block_size + origin
    return mesh_verts, mesh_faces[0]

def create_occ_grid_mesh_o3d(grid: torch.BoolTensor, color=[0.6,0.6,0.6], origin = None, block_size = None):
    import open3d as o3d
    assert grid.dtype == torch.bool, "Grid must be binary"
    mesh_verts, mesh_faces = create_occ_grid_mesh(grid, origin=origin, block_size=block_size)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(mesh_verts.data.cpu().numpy())
    mesh.triangles = o3d.utility.Vector3iVector(mesh_faces.data.cpu().numpy())
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(np.array(color))
    return mesh

def vis_spc_voxels_o3d(
    spc, 
    show=True, finest_only=False, finest_mesh=True, finest_mesh_only=False, finest_mesh_mat='lit', max_level=None, origin = None, block_size = None):
    if finest_mesh_only: finest_mesh=True
    if max_level is None: max_level = spc.max_level
    
    import open3d as o3d
    from kaolin.ops.spc import unbatched_get_level_points, to_dense
    from kaolin.rep.spc import Spc
    
    lod_colors = np.array([
        soft_blue,
        soft_red, 
        lime_green, 
        purple, 
        gold
    ])
    
    if block_size is None:
        block_size = 1./ (2**level)
    if origin is None:
        origin = torch.tensor([0.,0.,0.], spc.octrees.device)
    
    line_sets = []
    mesh = None
    to_plot_levels = list(range(max_level+1)) if not finest_only else [max_level]
    for level in to_plot_levels:
        # [num_voxels, 3]
        level_points = unbatched_get_level_points(spc.point_hierarchies, spc.pyramids[0], level)

        if not finest_mesh_only:
            # [num_voxels, 8, 3]
            corners = points_to_corners(level_points) * block_size + origin
            
            # corners = corners * 2.0 - 1.0
            grid_line_idx = np.array([(0, 1), (1, 3), (3, 2), (2, 0),
                                        (4, 5), (5, 7), (7, 6), (6, 4),
                                        (0, 4), (1, 5), (2, 6), (3, 7)])
            
            # [num_voxels, 1, 1] + [12, 2] -> [num_voxels, 12, 2] -> [num_voxels*12, 2]; range: [0, num_voxels*8-1]
            flattened_line_idx = (np.arange(len(corners))[:,None,None]*8 + grid_line_idx).reshape(-1,2)
            # [num_voxels, 8, 3] -> [num_voxels*8, 3]
            flattended_corners = corners.reshape(-1,3).cpu().numpy()

            colors = lod_colors[level % len(lod_colors)]
            grid_lines_color = np.tile(colors, (flattened_line_idx.shape[0], 1))
            
            lineset = o3d.geometry.LineSet()
            lineset.points = o3d.utility.Vector3dVector(flattended_corners)
            lineset.lines = o3d.utility.Vector2iVector(flattened_line_idx)
            lineset.colors = o3d.utility.Vector3dVector(grid_lines_color)
    
            line_sets.append(lineset)
        
        if finest_mesh and level == max_level:
            occ_grid = to_dense(spc.point_hierarchies, spc.pyramids, level_points.new_ones([*level_points[:, :1].shape], dtype=torch.float), level=level)[:,0]
            mesh = create_occ_grid_mesh_o3d(occ_grid, origin=origin, block_size=block_size)
    if show:
        # TODO: Add a slider bar in the gui to adjust the maximum display level; 
        #       Add a checkbox for whether to accumulate multiple levels.
        
        import open3d.visualization.gui as gui
        app = gui.Application.instance
        app.initialize()
        w = app.create_window(f"spc visualization", 1024, 768)
        widget3d = gui.SceneWidget()
        w.add_child(widget3d)
        widget3d.scene = o3d.visualization.rendering.Open3DScene(w.renderer)
        
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        
        for level, lineset in enumerate(line_sets):
            line_mat = o3d.visualization.rendering.MaterialRecord()
            line_mat.line_width = float(max_level-level+1)
            line_mat.shader = "unlitLine"
            
            widget3d.scene.add_geometry(f"spc.lv{level}", lineset, line_mat)

        if mesh is not None:
            mesh_mat = o3d.visualization.rendering.MaterialRecord()
            if finest_mesh_mat == 'transparent':
                mesh_mat.shader = "defaultLitTransparency"
                # mesh_mat.shader = "defaultUnlitTransparency"
                mesh_mat.base_color = np.array([1, 1, 1, 0.8])
                mesh_mat.base_roughness = 1.0
                mesh_mat.base_reflectance = 0.0
                mesh_mat.base_clearcoat = 0.0
            elif finest_mesh_mat == 'unlit':
                mesh_mat.shader = "defaultUnlit"
            elif finest_mesh_mat == 'lit':
                mesh_mat.shader = "defaultLit"
                mesh_mat.base_color = np.array([1, 1, 1, 0.8])
                mesh_mat.base_roughness = 1.0
                mesh_mat.base_reflectance = 0.0
                mesh_mat.base_clearcoat = 0.0
            else:
                raise RuntimeError(f"Invalid finest_mesh_mat={finest_mesh_mat}")
            widget3d.scene.add_geometry(f"spc.lv{max_level}.voxels", mesh, mesh_mat)

        size = gmean(block_size) * (2**level)
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=origin)
        widget3d.scene.add_geometry('coord', coord_frame, mat)
        
        bbox = widget3d.scene.bounding_box
        widget3d.setup_camera(60.0, bbox, bbox.get_center())
        widget3d.scene.set_background([0, 0, 0, 1]) 
        app.run()
    else:
        return line_sets

def vis_occgrid_voxels_o3d(occ_grid: torch.BoolTensor, draw_lines=True, draw_mesh=True,
                           draw_mesh_mat='lit', origin = None, block_size = None,
                           lines_color = soft_red, mesh_color = [1, 1, 1, 0.8],
                           background_color = [1, 1, 1, 1], 
                           show=True) -> List:    
    import open3d as o3d
    if draw_lines:
        # [0, 1] range
        lineset = create_occ_grid_lineset_o3d(occ_grid, lines_color, origin=origin,
            block_size=block_size)
    else:
        lineset = None
    if draw_mesh:
        mesh = create_occ_grid_mesh_o3d(occ_grid, (0.6,0.6,0.6), origin=origin, block_size=block_size)
    else:
        mesh = None
    if show:
        import open3d.visualization.gui as gui
        app = gui.Application.instance
        app.initialize()
        w = app.create_window(f"spc visualization", 1024, 768)
        widget3d = gui.SceneWidget()
        w.add_child(widget3d)
        widget3d.scene = o3d.visualization.rendering.Open3DScene(w.renderer)
        
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        
        if draw_lines:
            line_mat = o3d.visualization.rendering.MaterialRecord()
            line_mat.line_width = 4.
            line_mat.shader = "unlitLine"
            widget3d.scene.add_geometry(f"occ_grid", lineset, line_mat)
        
        if draw_mesh:
            mesh_mat = o3d.visualization.rendering.MaterialRecord()
            if draw_mesh_mat == 'transparent':
                mesh_mat.shader = "defaultLitTransparency"
                # mesh_mat.shader = "defaultUnlitTransparency"
                mesh_mat.base_color = np.array(mesh_color)
                mesh_mat.base_roughness = 1.0
                mesh_mat.base_reflectance = 0.0
                mesh_mat.base_clearcoat = 0.0
            elif draw_mesh_mat == 'unlit':
                mesh_mat.shader = "defaultUnlit"
            elif draw_mesh_mat == 'lit':
                mesh_mat.shader = "defaultLit"
                mesh_mat.base_color = np.array(mesh_color)
                mesh_mat.base_roughness = 1.0
                mesh_mat.base_reflectance = 0.0
                mesh_mat.base_clearcoat = 0.0
            else:
                raise RuntimeError(f"Invalid draw_mesh_mat={draw_mesh_mat}")
            widget3d.scene.add_geometry(f"occ_grid.voxels", mesh, mesh_mat)

        if origin is None: origin = [0.,0.,0.]
        if block_size is None: block_size = 1.0
        size = gmean(block_size).item() * max(occ_grid.shape)
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=origin.tolist())
        widget3d.scene.add_geometry('coord', coord_frame, mat)
        
        bbox = widget3d.scene.bounding_box
        widget3d.setup_camera(60.0, bbox, bbox.get_center())
        widget3d.scene.set_background(background_color) 
        app.run()
    else:
        return [item for item in [lineset, mesh] if item is not None]

def create_vox_grid_lines(
    grid_points: Union[np.ndarray, torch.FloatTensor], 
    spacing: Union[float, List[float], np.ndarray, torch.Tensor] = None) -> Tuple[np.ndarray, np.ndarray]:
    grid_points = check_to_torch(grid_points)
    if isinstance(spacing, Number):
        spacing = [spacing]
    if spacing is not None:
        spacing = torch.tensor(spacing, dtype=grid_points.dtype, device=grid_points.device).view(-1).expand([3])
    grid_points = grid_points.flatten(0, -2)
    corners = points_to_corners(grid_points, spacing=spacing)
    grid_line_idx = np.array([(0, 1), (1, 3), (3, 2), (2, 0),
                                (4, 5), (5, 7), (7, 6), (6, 4),
                                (0, 4), (1, 5), (2, 6), (3, 7)])
    # [num_voxels, 1, 1] + [12, 2] -> [num_voxels, 12, 2] -> [num_voxels*12, 2]; range: [0, num_voxels*8-1]
    flattened_line_idx = (np.arange(len(corners))[:,None,None]*8 + grid_line_idx).reshape(-1,2)
    # [num_voxels, 8, 3] -> [num_voxels*8, 3]
    flattended_corners = corners.reshape(-1,3).cpu().numpy()
    return flattended_corners, flattened_line_idx

def create_vox_grid_lineset_o3d(grid_points: torch.FloatTensor, color=light_pink, spacing: Union[float, List[float], np.ndarray, torch.Tensor] = None):
    import open3d as o3d
    points, lines = create_vox_grid_lines(grid_points, spacing=spacing)
    colors = np.tile(color, (lines.shape[0], 1))
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(points)
    lineset.lines = o3d.utility.Vector2iVector(lines)
    lineset.colors = o3d.utility.Vector3dVector(colors)
    return lineset

def vis_lidar_vedo(points: np.ndarray, colormap: str = 'rainbow', vmin=None, vmax=None, r: int=2, pcl_val: np.ndarray=None):
    """ Visualize 3D point cloud with vedo

    Args:
        points (np.ndarray): 3D point cloud data with shape [N, 3].
        colormap (str, optional): pcl colormap. Defaults to 'rainbow'.
        vmin (_type_, optional): min pcl val for color mapping. Defaults to -2..
        vmax (_type_, optional): max pcl val for color mapping. Defaults to 9..
        r (int, optional): radius of points. Defaults to 2.
        pcl_val (np.ndarray, optional): Scalar values corresponding to each point for color mapping. Defaults to None and will use the z coords of pcls.
    """
    import vedo
    if pcl_val is None:
        pcl_val = points[:, 2]
        vmin = vmin or -2.
        vmax = vmax or 9.
    pts_c = (vedo.color_map(pcl_val, colormap, vmin=vmin, vmax=vmax) * 255.).clip(0,255).astype(np.uint8)
    pts_c = np.concatenate([pts_c, np.full_like(pts_c[:,:1], 255)], axis=-1) # RGBA is ~50x faster
    pts_actor = vedo.Points(points, c=pts_c, r=r)
    vedo.show(pts_actor, axes=None)

def vis_lidar_o3d(points: np.ndarray):
    """
    Visualize 3D point cloud with open3d
    
    Args:
        points (numpy.array): 3D point cloud data with shape [N, 3].
    """
    # Create point cloud object and set color
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries(pcd)

def get_box_corners(center: np.ndarray, size: np.ndarray) -> np.ndarray:
    """ Calculates 8 box vertices according to box center and box sidelength
    
    Args:
        center (np.ndarray): Box center coord
        size (np.ndarray): Box sidelength, [l,w,h] for x,y,z axis
    
    Returns:
        np.ndarray: [8, 3] Box vertices
    """
    x, y, z = center
    l, w, h = size
    corners = np.array([ 
        [x-l/2, y-w/2, z-h/2], [x-l/2, y-w/2, z+h/2], [x-l/2, y+w/2, z-h/2], [x-l/2, y+w/2, z+h/2], 
        [x+l/2, y-w/2, z-h/2], [x+l/2, y-w/2, z+h/2], [x+l/2, y+w/2, z-h/2], [x+l/2, y+w/2, z+h/2]
    ])
    return corners

def create_box_o3d(center: np.ndarray, size: np.ndarray, color = [1,0,0]):
    corners = get_box_corners(center, size)
    lines = np.array([(0, 1), (1, 3), (3, 2), (2, 0),
                    (4, 5), (5, 7), (7, 6), (6, 4),
                    (0, 4), (1, 5), (2, 6), (3, 7)])
    colors = [color for _ in range(len(lines))]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners.tolist())
    line_set.lines = o3d.utility.Vector2iVector(lines.tolist())
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def vis_lidar_and_boxes_o3d(points: np.ndarray, boxes: np.ndarray):
    """ Visualize 3D point cloud with bounding boxes using Open3D library.
    
    Args:
        points (numpy.array): 3D point cloud data with shape [N, 3].
        boxes (numpy.array): 3D bounding boxes data with shape [M, 6],
            where M is the number of boxes, and the last 3 elements
            represent the dimensions of the box in (length, width, height)
            order, and the first 3 elements represent the center of the box
            in (x, y, z) order.
    """
    import open3d as o3d
    things_to_draw = []
    
    # Create point cloud object and set color
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    things_to_draw.append(pcd)
    
    # Create coordinate frame
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    things_to_draw.append(axes)
    
    for box in boxes:
        if box.shape[-1] == 6: # AABB; center (3) + size (3)
            transform = np.eye(4)
            center = box[:3]
            size = box[3:]
        elif box.shape[-1] == 15: # OBB; transform (12) + size (3)
            transform = np.eye(4)
            transform[:3] = box[:12].reshape(3,4)
            size = box[12:]
            center = np.array([0., 0., 0.])
        else:
            raise RuntimeError(f"Invalid box.shape = {box.shape}. Supported: [N,6], [N,15]")
        # Create line set containing bounding boxes as wireframes
        box = create_box_o3d(center, size)
        box.transform(transform)
        things_to_draw.append(box)
        
    # Visualize point cloud, bounding boxes, and coordinate frame
    o3d.visualization.draw_geometries(things_to_draw)

def create_continuous_curve_o3d(points: np.ndarray, color=[1,0,0]):
    import open3d as o3d
    lines = [[i, i+1] for i in range(len(points)-1)]
    lines = np.array(lines)
    colors = [color for _ in range(len(lines))]
    colors = np.array(colors)
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

if __name__ == "__main__":
    import open3d as o3d
    def unit_test_points_to_corners(device=torch.device('cuda')):
        from kaolin.ops.spc import points_to_corners as points_to_corners1
        occgrid = torch.rand([32,32,32], device=device) > 0.5
        points = occgrid.nonzero().long()
        corners = points_to_corners1(points.short()).long()
        corners2 = points_to_corners(points)    
        print(torch.equal(corners, corners2))
    
    def unit_test_vis_occgrid_o3d(device=torch.device('cuda')):
        #--------- test 1: voxels from occupancy grid
        import open3d.visualization.gui as gui
        occgrid = torch.rand([8,8,8], device=device) > 0.5
        lineset = create_occ_grid_lineset_o3d(occgrid)
        
        app = gui.Application.instance
        app.initialize()
        w = app.create_window(f"spc visualization", 1024, 768)
        widget3d = gui.SceneWidget()
        w.add_child(widget3d)
        widget3d.scene = o3d.visualization.rendering.Open3DScene(w.renderer)

        line_mat = o3d.visualization.rendering.MaterialRecord()
        line_mat.line_width = 4.
        line_mat.shader = "unlitLine"
        widget3d.scene.add_geometry(f"occ_grid", lineset, line_mat)
        
        bbox = widget3d.scene.bounding_box
        widget3d.setup_camera(60.0, bbox, bbox.get_center())
        widget3d.scene.set_background([0, 0, 0, 1]) 
        app.run()
    
    def unit_test_vis_voxgrid_o3d(device=torch.device('cuda')):
        #--------- test 2: voxels from list of grid points
        import open3d.visualization.gui as gui
        occgrid = torch.rand([8,8,8], device=device) > 0.5
        spacing = torch.tensor([0.2, 0.5, 0.3], device=device)
        voxgrid = occgrid.nonzero().float() * spacing
        lineset = create_vox_grid_lineset_o3d(voxgrid, spacing=spacing)
        
        app = gui.Application.instance
        app.initialize()
        w = app.create_window(f"spc visualization", 1024, 768)
        widget3d = gui.SceneWidget()
        w.add_child(widget3d)
        widget3d.scene = o3d.visualization.rendering.Open3DScene(w.renderer)

        line_mat = o3d.visualization.rendering.MaterialRecord()
        line_mat.line_width = 4.
        line_mat.shader = "unlitLine"
        widget3d.scene.add_geometry(f"occ_grid", lineset, line_mat)
        
        bbox = widget3d.scene.bounding_box
        widget3d.setup_camera(60.0, bbox, bbox.get_center())
        widget3d.scene.set_background([0, 0, 0, 1]) 
        app.run()
        

    # unit_test_points_to_corners()
    unit_test_vis_occgrid_o3d()
    unit_test_vis_voxgrid_o3d()
        