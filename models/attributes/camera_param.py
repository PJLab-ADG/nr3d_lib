"""
@file   camera_param.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Camera param related attr definitions.
"""

__all__ = [
    #---- Intrinsics mat
    "CameraMatrixBase", 
    "CameraMatrix3x3", 
    "CameraMatrix4x4", 
    
    #---- Focal ratio
    "FocalRatio", 
    "FocalRatioSquare", 
    "FocalRatioExp", 
    
    "FocalRatioRefinedAdd", 
    "FocalRatioSquareRefinedAdd", 
    "FocalRatioExpRefinedAdd", 
    "FocalRatioRefinedMul", 
    
    #---- Pinhole camera model (H, W, camera_matrix)
    "CameraBase", 
    "PinholeCameraMatHW", 
    "PinholeCameraHWFxy", 
    "PinholeCameraHWFxyRatio", 
    "PinholeCameraHWF", 
    "PinholeCameraHWF", 
    "PinholeCameraHWF", 
    "PinholeCameraHWFRatio", 
    
    #---- Orthogonal camera model (H, W, phyH, phyW)
    "OrthoCameraIntrinsics", 
    
    #---- OpenCV camera model (H, W, camera_matrix, distortion)
    "OpenCVCameraMatHW", 
    
    #---- Fisheye camera model (H, W, camera_matrix, distortion)
    "FisheyeCameraMatHW", 
]

import numpy as np
from typing import Tuple, Union

import torch

from .attr import *

from nr3d_lib.geometry import *
from nr3d_lib.utils import check_to_torch, is_scalar
from nr3d_lib.render.cameras import \
    pinhole_lift, pinhole_lift_cf, pinhole_view_frustum, \
    opencv_distort_points, opencv_undistort_points, \
    fisheye_distort_points_cpu, fisheye_undistort_points_cpu

################################################
####    Camera params (Pinhole, OpenCV)    #####
################################################
#---- Intrinsics mat
@AttrBase
class CameraMatrixBase(Attr):
    def focal(self) -> torch.Tensor:
        raise NotImplementedError
    def principle(self) -> torch.Tensor:
        raise NotImplementedError
    def mat_3x3(self):
        raise NotImplementedError
    def mat_4x4(self):
        raise NotImplementedError    
class CameraMatrix3x3(CameraMatrixBase):
    default = torch.eye(3)
    def focal(self):
        fx = self.tensor[..., 0, 0]
        fy = self.tensor[..., 1, 1]
        return torch.stack([fx, fy], dim=-1)
    def principle(self):
        cx = self.tensor[..., 0, 2]
        cy = self.tensor[..., 1, 2]
        return torch.stack([cx, cy], dim=-1)
    def mat_3x3(self):
        return self.tensor[..., :3, :3]
    def mat_4x4(self):
        return torch.cat(
            [
                torch.cat([self.tensor, check_to_torch([[0],[0],[0]], ref=self).tile([*self.prefix, 1, 1])], dim=-1),
                check_to_torch([[0,0,0,1]], ref=self).tile([*self.prefix, 1, 1])
            ], dim=-2)
    def set_focal(self, fx, fy=None):
        self.tensor[..., 0, 0] = fx
        self.tensor[..., 1, 1] = fx if fy is None else fy
class CameraMatrix4x4(CameraMatrix3x3):
    default = torch.eye(4)
    def mat_4x4(self):
        return self.tensor[:]

#---- Focal ratio
@AttrBase
class FocalRatio(Attr):
    default = torch.tensor([1.,1.])
    def ratio(self):
        # return torch.movedim(self.tensor, -1, 0)
        return self.tensor[:]
    @staticmethod
    def from_fov(fov_deg=53., size=(), learnable=False):
        sx = 0.5 / np.tan((.5 * fov_deg * np.pi/180.))
        sy = 0.5 / np.tan((.5 * fov_deg * np.pi/180.))
        fxy_r = torch.tensor([sx, sy], dtype=torch.float32).tile([*size,1])
        return FocalRatio(fxy_r, learnable=learnable)
class FocalRatioSquare(FocalRatio):
    def ratio(self):
        return super().ratio() ** 2
    @staticmethod
    def from_fov(fov_deg=53., size=(), learnable=False):
        fxy_r = FocalRatio.from_fov(fov_deg=fov_deg, size=size)
        with torch.no_grad():
            fxy_r = FocalRatioSquare(torch.sqrt(fxy_r.tensor.data), learnable=learnable)
        return fxy_r
class FocalRatioExp(FocalRatio):
    default = torch.tensor([0.,0.])
    def ratio(self):
        return torch.exp(super().ratio())
    @staticmethod
    def from_fov(fov_deg=53., size=(), learnable=False):
        fxy_r = FocalRatio.from_fov(fov_deg=fov_deg, size=size)
        with torch.no_grad():
            fxy_r = FocalRatioExp(torch.log(fxy_r.tensor.data), learnable=learnable)
        return fxy_r
FocalRatioRefinedAdd = make_refined_add_cls(FocalRatio)
FocalRatioExpRefinedAdd = make_refined_add_cls(FocalRatioExp)
FocalRatioSquareRefinedAdd = make_refined_add_cls(FocalRatioSquare)
class FocalRatioRefinedMul(AttrNested, FocalRatio):
    default = {'attr0': FocalRatio(), 'delta': FocalRatio()}
    def ratio(self):
        return self.subattr.attr0.ratio() * self.subattr.delta.ratio()
    @staticmethod
    def from_fov(*args, **kwargs):
        raise RuntimeError("Please set attr0 and delta manually")
    @property
    def prefix(self):
        return self.subattr.attr0.prefix

#---- Pinhole camera model (H, W, camera_matrix)
@AttrBase
class CameraBase(AttrNested):
    default = {'H': Scalar(torch.tensor([100.])), 'W':Scalar(torch.tensor([100.]))}
    # Common functions
    def unscaled_wh(self):
        return torch.stack([self.subattr.W.tensor, self.subattr.H.tensor], dim=-1)
    def wh(self):
        return self.unscaled_wh() / self.scale_xy
    def set_downscale(self, ratio: Union[int, float, Tuple[Union[int, float], Union[int, float]]]):
        """
        single input: set equal ratio to x(W) / y(H)
        double input: set different ratio to x(W) and y(H)
        """
        if is_scalar(ratio):
            assert ratio > 0, f"Invalid downscale ratio {ratio}"
            self._scale_x = ratio
            self._scale_y = ratio
            self._scale_xy = torch.tensor([self.scale_x, self.scale_y], dtype=torch.float, device=self.device)
        else:
            ratio = check_to_torch(ratio, dtype=torch.float, device=self.device)
            assert ratio.shape[-1] == 2 and (ratio[..., 0] > 0).all() and (ratio[..., 1] > 0).all(), f"Invalid downscale ratio {ratio}"
            self._scale_x = ratio[..., 0]
            self._scale_y = ratio[..., 1]
            self._scale_xy = ratio
        self._scale_3x3 = torch.ones([*self.prefix,3,3], device=self.device)
        self._scale_3x3[..., 0, 0] = self._scale_x
        self._scale_3x3[..., 1, 1] = self._scale_y
        self._scale_3x3[..., 0, 2] = self._scale_x
        self._scale_3x3[..., 1, 2] = self._scale_y
        self._scale_4x4 = torch.ones([*self.prefix,4,4], device=self.device)
        self._scale_4x4[..., :3, :3] = self._scale_3x3
    @property
    def scale_x(self):
        return 1 if not hasattr(self, '_scale_x') else self._scale_x
    @property
    def scale_y(self):
        return 1 if not hasattr(self, '_scale_y') else self._scale_y
    @property
    def scale_xy(self):
        return 1 if not hasattr(self, '_scale_xy') else  self._scale_xy
    @property
    def scale_3x3(self):
        # return torch.ones([3,3], device=self.device) if not hasattr(self, '_scale_3x3') else self._scale_3x3
        return 1 if not hasattr(self, '_scale_3x3') else self._scale_3x3
    @property
    def scale_4x4(self):
        # return torch.ones([4,4], device=self.device) if not hasattr(self, '_scale_4x4') else self._scale_4x4
        return 1 if not hasattr(self, '_scale_4x4') else self._scale_4x4
    @property
    def H(self):
        H = torch.floor(self.subattr.H.tensor / self.scale_y + 0.5).long() # Rounding to the nearest integer
        if is_scalar(H):
            return int(H.item())
        else:
            H_first = H.flatten()[0]
            assert (H==H_first).all(), "Can not get `H` when it's different across batches."
            return int(H_first.item())
    @property
    def W(self):
        W = torch.floor(self.subattr.W.tensor / self.scale_x + 0.5).long() # Rounding to the nearest integer
        if is_scalar(W):
            return int(W.item())
        else:
            W_first = W.flatten()[0]
            assert (W==W_first).all(), "Can not get `W` when it's different across batches."
            return int(W_first.item())
    @H.setter
    def H(self, value: int):
        self.subattr.H.tensor[:] = value
    @W.setter
    def W(self, value: int):
        self.subattr.W.tensor[:] = value
    # Interfaces
    def focal(self) -> torch.Tensor:
        return torch.tensor([1., 1.], device=self.device)
    def focal_ratio(self) -> torch.Tensor:
        raise NotImplementedError
    def fov(self, degree=True) -> torch.Tensor:
        fov_rad = 2 * torch.arctan(0.5 * self.wh() / self.focal())
        if degree:
            return fov_rad * 180. / np.pi
        else:
            return fov_rad
        
    def set_focal(self, fx, fy=None):
        raise NotImplementedError
    def principle(self):
        return self.wh() / 2.
    def mat_3x3(self) -> torch.Tensor:
        fx, fy = self.focal().movedim(-1,0)
        cx, cy = self.principle().movedim(-1,0)
        mat1 = check_to_torch(torch.zeros([3,3]),ref=self).tile([*fx.shape,1,1])
        mat1[...,0,0] = fx
        mat1[...,1,1] = fy
        mat2 = check_to_torch(torch.zeros([3,3]),ref=self).tile([*cx.shape,1,1])
        mat2[...,0,2] = cx
        mat2[...,1,2] = cy
        # notice there might be broadcasting happens here
        mat = mat1 + mat2
        mat[..., 2,2] = 1
        return mat
    def mat_4x4(self) -> torch.Tensor:
        mat_3x3 = self.mat_3x3()
        prefix = mat_3x3.shape[:-2]
        return torch.cat(
            [
                torch.cat([mat_3x3, check_to_torch([[0],[0],[0]], ref=self).tile([*prefix, 1, 1])], dim=-1),
                check_to_torch([[0,0,0,1]], ref=self).tile([*prefix, 1, 1])
            ], dim=-2)
    def lift(self, u: torch.Tensor, v: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """ 2D Image coords to 3D world coords

        Args:
            u (torch.Tensor): image pixel coordinates in range [0,W)
            v (torch.Tensor): image pixel coordinates in range [0,H)
            d (torch.Tensor): depth of the pixels

        Returns:
            torch.Tensor: The lifted 3D points
        """
        raise NotImplementedError
    def proj(self, xyz: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ 3D world coords to 2D image coords

        Args:
            xyz (torch.Tensor): 3D points in camera coordinates

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: u, v, d
                u: image pixel coordinates in range [0,W)
                v: image pixel coordinates in range [0,H)
                d: depth of the pixels
        """
        raise NotImplementedError
    def get_view_frustum(self, c2w: torch.Tensor, near: float, far: float):
        fx, fy = self.focal().movedim(-1,0)
        cx, cy = self.principle().movedim(-1,0)
        planes = pinhole_view_frustum(c2w, cx, cy, fx, fy, near_clip=near, far_clip=far)
        return planes
    @property
    def prefix(self):
        raise NotImplementedError
    @property
    def datashape(self):
        raise NotImplementedError

class PinholeCameraMatHW(CameraBase):
    default = {
        'mat':CameraMatrix3x3(), 
        'H': Scalar(torch.tensor([100.])), 
        'W':Scalar(torch.tensor([100.]))
    }
    @property
    def prefix(self):
        return self.subattr.mat.prefix
    def focal(self):
        return self.subattr.mat.focal() / self.scale_xy
    def focal_ratio(self):
        return self.subattr.mat.focal() / self.unscaled_wh()
    def set_focal(self, fx, fy=None):
        self.subattr.mat.set_focal(fx, fy)
    def principle(self):
        return self.subattr.mat.principle() / self.scale_xy
    def mat_3x3(self) -> torch.Tensor:
        return self.subattr.mat.mat_3x3()/self.scale_3x3
    def mat_4x4(self) -> torch.Tensor:
        return self.subattr.mat.mat_4x4()/self.scale_4x4
    def lift(self, u, v, d):
        # NOTE: u./v./d.shape == [*self.prefix, ...] or [*[1]*len(self.prefix), ...]
        mat_3x3 = self.mat_3x3()
        if len(prefix:=self.prefix) > 0:
            # If batched: Assume `u` has the same prefix dims as self
            mat_3x3 = mat_3x3.view(*prefix, *[1]*(u.dim()-len(prefix)), 3, 3)
        return pinhole_lift(u, v, d, mat_3x3)
    def proj(self, xyz: torch.Tensor):
        # NOTE: xyz.shape == [*self.prefix, ..., 3] or [*[1]*len(self.prefix), ..., 3]
        mat_3x3 = self.mat_3x3()
        if len(prefix:=self.prefix) > 0:
            # If batched: Assume `xyz` has the same prefix dims as self
            mat_3x3 = mat_3x3.view(*prefix, *[1]*(xyz.dim()-1-len(prefix)), 3, 3)
        uvd = (mat_3x3 * xyz.unsqueeze(-2)).sum(-1)
        uv = uvd[..., 0:2] / torch.abs(uvd[..., 2:3]).clamp(1e-7)
        return uv[...,0], uv[...,1], uvd[..., 2]

class PinholeCameraHWFxy(CameraBase):
    default = {
        'fx': Scalar(torch.tensor([100.])), 
        'fy': Scalar(torch.tensor([100.])), 
        'H': Scalar(torch.tensor([100.])), 
        'W':Scalar(torch.tensor([100.]))
    }
    @property
    def prefix(self):
        return self.subattr.fx.prefix
    def focal(self):
        return torch.stack([self.subattr.fx.tensor/self.scale_x, self.subattr.fy.tensor/self.scale_y], dim=-1)
    def focal_ratio(self):
        return self.focal() / self.unscaled_wh()
    def set_focal(self, fx: torch.Tensor, fy: torch.Tensor=None):
        self.subattr.fx[:] = fx
        self.subattr.fy[:] = fx if fy is None else fy
    def lift(self, u, v, d):
        # NOTE: u./v./d.shape == [*self.prefix, ...] or [*[1]*len(self.prefix), ...]
        principle = self.principle().movedim(-1,0)
        focal = self.focal().movedim(-1,0)
        if len(prefix:=self.prefix) > 0:
            # If batched: Assume `u` has the same prefix dims as self
            principle = principle if list(principle.shape) == [2] else principle.view(2, *prefix, *[1]*(u.dim()-len(prefix)))
            focal = focal if list(focal.shape) == [2] else focal.view(2, *prefix, *[1]*(u.dim()-len(prefix)))
        return pinhole_lift_cf(u, v, d, *principle, *focal)
    def proj(self, xyz: torch.Tensor):
        # NOTE: xyz.shape == [*self.prefix, ..., 3] or [*[1]*len(self.prefix), ..., 3]
        # uvd = torch.einsum('...ij,...j->...i', self.mat_3x3(), xyz)
        mat_3x3 = self.mat_3x3()
        if len(prefix:=self.prefix) > 0:
            # If batched: Assume `xyz` has the same prefix dims as self
            mat_3x3 = mat_3x3.view(*prefix, *[1]*(xyz.dim()-1-len(prefix)), 3, 3)
        uvd = (mat_3x3 * xyz.unsqueeze(-2)).sum(-1)
        uv = uvd[..., 0:2] / torch.abs(uvd[..., 2:3]).clamp(1e-7)
        return uv[...,0], uv[...,1], uvd[..., 2]
        
class PinholeCameraHWFxyRatio(CameraBase):
    default = {
        'fxy_r': FocalRatio(torch.ones([2,])), 
        'H': Scalar(torch.tensor([100.])), 
        'W':Scalar(torch.tensor([100.]))
    }
    @property
    def prefix(self):
        return self.subattr.fxy_r.prefix
    def focal(self) -> torch.Tensor:
        return self.focal_ratio() * self.wh()
    def focal_ratio(self) -> torch.Tensor:
        return self.subattr.fxy_r.ratio()
    def set_focal(self, fx: torch.Tensor, fy: torch.Tensor = None):
        fxy = torch.stack([fx, fx if fy is None else fy], dim=-1)
        self.subattr.fxy_r[:] = fxy / self.wh()
    def lift(self, u, v, d):
        # NOTE: u./v./d.shape == [*self.prefix, ...] or [*[1]*len(self.prefix), ...]
        principle = self.principle().movedim(-1,0)
        focal = self.focal().movedim(-1,0)
        if len(prefix:=self.prefix) > 0:
            # If batched: Assume `u` has the same prefix dims as self
            principle = principle if list(principle.shape) == [2] else principle.view(2, *prefix, *[1]*(u.dim()-len(prefix)))
            focal = focal if list(focal.shape) == [2] else focal.view(2, *prefix, *[1]*(u.dim()-len(prefix)))
        return pinhole_lift_cf(u, v, d, *principle, *focal)
    def proj(self, xyz: torch.Tensor):
        # NOTE: xyz.shape == [*self.prefix, ..., 3] or [*[1]*len(self.prefix), ..., 3]
        # uvd = torch.einsum('...ij,...j->...i', self.mat_3x3(), xyz)
        mat_3x3 = self.mat_3x3()
        if len(prefix:=self.prefix) > 0:
            # If batched: Assume `xyz` has the same prefix dims as self
            mat_3x3 = mat_3x3.view(*prefix, *[1]*(xyz.dim()-1-len(prefix)), 3, 3)
        uvd = (mat_3x3 * xyz.unsqueeze(-2)).sum(-1)
        uv = uvd[..., 0:2] / torch.abs(uvd[..., 2:3]).clamp(1e-7)
        return uv[...,0], uv[...,1], uvd[..., 2]

class PinholeCameraHWF(CameraBase):
    default = {
        'f': Scalar(torch.tensor([100.])), 
        'H': Scalar(torch.tensor([100.])), 
        'W': Scalar(torch.tensor([100.]))
    }
    @property
    def prefix(self):
        return self.subattr.f.prefix
    def focal(self):
        prefix = self.subattr.f.prefix
        return self.subattr.f.tensor.unsqueeze(-1).tile([*[1]*len(prefix), 2]) / self.scale_xy
    def focal_ratio(self):
        return self.focal() / self.wh()
    def set_focal(self, f: torch.Tensor):
        self.subattr.f[:] = f
    def lift(self, u, v, d):
        # NOTE: u./v./d.shape == [*self.prefix, ...] or [*[1]*len(self.prefix), ...]
        principle = self.principle().movedim(-1,0)
        focal = self.focal().movedim(-1,0)
        if len(prefix:=self.prefix) > 0:
            # If batched: Assume `u` has the same prefix dims as self
            principle = principle if list(principle.shape) == [2] else principle.view(2, *prefix, *[1]*(u.dim()-len(prefix)))
            focal = focal if list(focal.shape) == [2] else focal.view(2, *prefix, *[1]*(u.dim()-len(prefix)))
        return pinhole_lift_cf(u, v, d, *principle, *focal)
    def proj(self, xyz: torch.Tensor):
        # NOTE: xyz.shape == [*self.prefix, ..., 3] or [*[1]*len(self.prefix), ..., 3]
        # uvd = torch.einsum('...ij,...j->...i', self.mat_3x3(), xyz)
        mat_3x3 = self.mat_3x3()
        if len(prefix:=self.prefix) > 0:
            # If batched: Assume `xyz` has the same prefix dims as self
            mat_3x3 = mat_3x3.view(*prefix, *[1]*(xyz.dim()-1-len(prefix)), 3, 3)
        uvd = (mat_3x3 * xyz.unsqueeze(-2)).sum(-1)
        uv = uvd[..., 0:2] / torch.abs(uvd[..., 2:3]).clamp(1e-5)
        return uv[...,0], uv[...,1], uvd[..., 2]

class PinholeCameraHWFRatio(CameraBase):
    pass

#---- Orthogonal camera model (H, W, phyH, phyW)
class OrthoCameraIntrinsics(CameraBase):
    default = {
        'phyH': Scalar(), 'phyW': Scalar(), 
        'H': Scalar(), 'W': Scalar()
    }
    
    @property
    def phyW(self): return self.subattr.phyW.item()

    @property
    def phyH(self): return self.subattr.phyH.item()
    
    def lift(self, u, v, d):
        x = (u / self.W - .5) * self.phyW
        y = (v / self.H - .5) * self.phyH
        z = d
        return torch.stack([x, y, z], dim=-1)

    def proj(self, xyz: torch.Tensor):
        u = (xyz[..., 0] / self.phyW + .5) * self.W
        v = (xyz[..., 1] / self.phyH + .5) * self.H
        d = xyz[..., 2]
        return u, v, d

#---- OpenCV camera model (H, W, camera_matrix, distortion)
class OpenCVCameraMatHW(CameraBase):
    default = {
        'mat':CameraMatrix3x3(), 
        'H': Scalar(torch.tensor([100.])), 
        'W': Scalar(torch.tensor([100.])), 
        'distortion': Vector_5(torch.zeros([5,])) # :math:`(k_1,k_2,p_1,p_2[,k_3[,k_4,k_5,k_6[,s_1,s_2,s_3,s_4[,\tau_x,\tau_y]]]])`
    }
    @property
    def prefix(self):
        return self.subattr.mat.prefix
    def focal(self):
        return self.subattr.mat.focal() / self.scale_xy
    def focal_ratio(self):
        return self.subattr.mat.focal() / self.unscaled_wh()
    def set_focal(self, fx, fy=None):
        self.subattr.mat.set_focal(fx, fy)
    def principle(self):
        return self.subattr.mat.principle() / self.scale_xy
    def mat_3x3(self) -> torch.Tensor:
        return self.subattr.mat.mat_3x3()/self.scale_3x3
    def mat_4x4(self) -> torch.Tensor:
        return self.subattr.mat.mat_4x4()/self.scale_4x4
    def lift(self, u, v, d):
        prefix = self.prefix
        # NOTE: u./v./d.shape == [*self.prefix, ...] or [*[1]*len(self.prefix), ...]
        mat_3x3 = self.mat_3x3() # Might-scaled new K
        # mat0_3x3 = self.subattr.mat.mat_3x3() # Un-scaled original K
        distortion = self.subattr.distortion.tensor
        if len(prefix:=self.prefix) > 0:
            mat_3x3 = mat_3x3.view(*prefix, *[1]*(u.dim()-len(prefix)), 3, 3)
            distortion = distortion.view(*prefix, *[1]*(u.dim()-len(prefix)), distortion.shape[-1])
        uv = torch.stack([u,v], dim=-1)
        # From opencv image's uv to pinhole uv
        uv1 = opencv_undistort_points(uv.unsqueeze(-2), K=mat_3x3, dist=distortion).squeeze(-2)
        # Lift from pinhole 2d uv to 3d xyz
        return pinhole_lift(
            uv1[..., 0], uv1[..., 1], d, 
            mat_3x3 if len(prefix)==0 else mat_3x3.view(*prefix, *[1]*(u.dim()-len(prefix)), 3, 3))
    def proj(self, xyz: torch.Tensor, min_radial: float = 0.8, max_radial: float = 1.2):
        prefix = self.prefix
        mat_3x3 = self.mat_3x3() # Might-scaled new K
        # mat0_3x3 = self.subattr.mat.mat_3x3() # Un-scaled original K
        distortion = self.subattr.distortion.tensor
        if len(prefix:=self.prefix) > 0:
            mat_3x3 = mat_3x3.view(*prefix, *[1]*(xyz.dim()-1-len(prefix)), 3, 3)
            distortion = distortion.view(*prefix, *[1]*(xyz.dim()-1-len(prefix)), distortion.shape[-1])
        
        uvd = (mat_3x3 * xyz.unsqueeze(-2)).sum(-1)
        uv = uvd[..., 0:2] / torch.abs(uvd[..., 2:3]).clamp(1e-7)

        # From pinhole uv to opencv uv
        uv1, valid = opencv_distort_points(
            uv, 
            K=mat_3x3 if len(prefix)==0 else mat_3x3.view(*prefix, *[1]*(xyz.dim()-2-len(prefix)), 3, 3),
            dist=distortion if len(prefix)==0 else distortion.view(*prefix, *[1]*(xyz.dim()-2-len(prefix)), distortion.shape[-1]), 
            min_radial=min_radial, max_radial=max_radial
        )
        
        uvd1 = uvd.new_full(uvd.shape, -1)
        wh = self.wh()
        wh = wh if len(prefix)==0 else wh.view(*prefix, *[1]*(xyz.dim()-1-len(prefix)), 2)
        in_bound = valid & (uvd[..., 2] > 0) & (uv1[..., 0] >= 0) & (uv1[..., 0] < wh[..., 0]) & (uv1[..., 1] >= 0) & (uv1[..., 1] < wh[..., 1])
        
        uvd1[in_bound] = torch.cat([uv1[in_bound], uvd[..., 2:3][in_bound]], dim=-1)
        return uvd1[...,0], uvd1[...,1], uvd1[..., 2]
    def lift_pinhole(self, u, v, d):
        return PinholeCameraMatHW.lift(self, u, v, d)
    def proj_pinhole(self, xyz: torch.Tensor):
        return PinholeCameraMatHW.proj(self, xyz)

class FisheyeCameraMatHW(CameraBase):
    # https://docs.opencv.org/4.1.1/db/d58/group__calib3d__fisheye.html
    default = {
        'mat':CameraMatrix3x3(), 
        'H': Scalar(torch.tensor([100.])), 
        'W': Scalar(torch.tensor([100.])), 
        'distortion': Vector_4(torch.zeros([4,])) # k1,k2,k3,k4
    }
    @property
    def prefix(self):
        return self.subattr.mat.prefix
    def focal(self):
        return self.subattr.mat.focal() / self.scale_xy
    def focal_ratio(self):
        return self.subattr.mat.focal() / self.unscaled_wh()
    def set_focal(self, fx, fy=None):
        self.subattr.mat.set_focal(fx, fy)
    def principle(self):
        return self.subattr.mat.principle() / self.scale_xy
    def mat_3x3(self) -> torch.Tensor:
        return self.subattr.mat.mat_3x3()/self.scale_3x3
    def mat_4x4(self) -> torch.Tensor:
        return self.subattr.mat.mat_4x4()/self.scale_4x4
    def lift(self, u, v, d):
        prefix = self.prefix
        # NOTE: u./v./d.shape == [*self.prefix, ...] or [*[1]*len(self.prefix), ...]
        mat_3x3 = self.mat_3x3() # Might-scaled new K
        # mat0_3x3 = self.subattr.mat.mat_3x3() # Un-scaled original K
        distortion = self.subattr.distortion.tensor
        if len(prefix:=self.prefix) > 0:
            mat_3x3 = mat_3x3.view(*prefix, *[1]*(u.dim()-len(prefix)), 3, 3)
            distortion = distortion.view(*prefix, *[1]*(u.dim()-len(prefix)), distortion.shape[-1])
        uv = torch.stack([u,v], dim=-1)
        # From fisheye image's uv to pinhole uv
        raise NotImplementedError
        uv1 = fisheye_undistort_points_cpu(uv.unsqueeze(-2), K=mat_3x3, dist=distortion).squeeze(-2)
        # uv1 = fisheye_undistort_points_cpu(uv.unsqueeze(-2), K=mat_3x3.reshape(-1,3,3)[0], dist=distortion.reshape(-1,4)[0]).squeeze(-2)
        # Lift from pinhole 2d uv to 3d xyz
        return pinhole_lift(
            uv1[..., 0], uv1[..., 1], d, 
            mat_3x3 if len(prefix)==0 else mat_3x3.view(*prefix, *[1]*(u.dim()-len(prefix)), 3, 3))
    def proj(self, xyz: torch.Tensor):
        prefix = self.prefix
        mat_3x3 = self.mat_3x3() # Might-scaled new K
        # mat0_3x3 = self.subattr.mat.mat_3x3() # Un-scaled original K
        distortion = self.subattr.distortion.tensor
        if len(prefix:=self.prefix) > 0:
            mat_3x3 = mat_3x3.view(*prefix, *[1]*(xyz.dim()-1-len(prefix)), 3, 3)
            distortion = distortion.view(*prefix, *[1]*(xyz.dim()-1-len(prefix)), distortion.shape[-1])
        
        # Project 3d xyz to pinhole 2d uv
        uvd = (mat_3x3 * xyz.unsqueeze(-2)).sum(-1)
        uv = uvd[..., 0:2] / torch.abs(uvd[..., 2:3]).clamp(1e-7)

        # From pinhole uv to fisheye uv
        raise NotImplementedError
        uv1 = fisheye_distort_points_cpu(
            uv, 
            K=mat_3x3 if len(prefix)==0 else mat_3x3.view(*prefix, *[1]*(xyz.dim()-2-len(prefix)), 3, 3),
            dist=distortion if len(prefix)==0 else distortion.view(*prefix, *[1]*(xyz.dim()-2-len(prefix)), distortion.shape[-1]), 
        )
        # uv1 = fisheye_distort_points_cpu(
        #     uv, 
        #     K=mat_3x3.reshape(-1,3,3)[0],
        #     dist=distortion.reshape(-1,4)[0], 
        # )
        
        # Check inbound uv
        uvd1 = uvd.new_full(uvd.shape, -1)
        wh = self.wh()
        wh = wh if len(prefix)==0 else wh.view(*prefix, *[1]*(xyz.dim()-1-len(prefix)), 2)
        in_bound = (uvd[..., 2] > 0) & (uv1[..., 0] >= 0) & (uv1[..., 0] < wh[..., 0]) & (uv1[..., 1] >= 0) & (uv1[..., 1] < wh[..., 1])
        
        uvd1[in_bound] = torch.cat([uv1[in_bound], uvd[..., 2:3][in_bound]], dim=-1)
        return uvd1[...,0], uvd1[...,1], uvd1[..., 2]