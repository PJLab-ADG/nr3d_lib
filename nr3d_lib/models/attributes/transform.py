"""
@file   transform.py
@author Jianfei Guo, Shanghai AI Lab
@brief  3D transformation related attr definitions.
        
        NOTE For matrix multiplication:
            - Do not use bmm/mm! 
                In practive, bmm/mm would introduce errors of 4e-3 magnitude to rays_d, 
                which would be broadcasted to coordinate x and enlarged to an error of ~0.1 @(depth=10) and ~1.0 @(depth=100)
            - Do not use einsum!
                Einsum often just falls back to bmm/mm.
            - Use unsqueezing, element-wise multiplication, and reducing instead.
"""

__all__ = [
    #---- Rotation representations
    "RotationBase", 
    "RotationMat3x3",
    "RotationQuaternion",
    "RotationAxisAngle",
    "Rotation6D",
    "Rotation", 
    
    "RotationQuaternionRefinedAdd", 
    "RotationAxisAngleRefinedAdd", 
    "Rotation6DRefinedAdd", 
    "RotationRefinedMul", 
    
    #---- Translation representations
    "Translation",
    "TranslationRefinedAdd", 
    
    #---- Transform representations
    "TransformBase",
    "TransformRT",
    "TransformMat3x4",
    "TransformMat4x4",
    "TransformExpSE3",
    "Transform",
    
    #---- Scaling representations
    "Scale",
    "ScaleExp",
    "ScaleSquare", 
]

from math import prod
from typing import Tuple

import torch

from .attr import *

from nr3d_lib.maths import *
from nr3d_lib.utils import check_to_torch, torch_consecutive_interp1d, torch_consecutive_interp1d_general

################################################
################    Transform    ###############
################################################
#---- Rotation representations
@AttrBase
class RotationBase(Attr):
    default = None
    def inv(self) -> 'RotationBase':
        raise NotImplementedError
    def interp1d(self, ts_keyframes: torch.Tensor, ts: torch.Tensor) -> 'RotationBase':
        # NOTE: For most of the rotation representations, interpolation should not be linear.
        raise NotImplementedError
    def mat_3x3(self) -> torch.Tensor:
        raise NotImplementedError
    def forward(self, x: torch.Tensor, inv=False) -> torch.Tensor:
        raise NotImplementedError
    @classmethod
    def from_mat_3x3(cls, mat_3x3: torch.Tensor):
        raise NotImplementedError
    @classmethod
    def random(cls, size=(), device=None, dtype=None):
        from scipy.spatial.transform import Rotation as R
        num = int(prod(size))
        mat_3x3 = R.random(num).as_matrix().reshape([*size,3,3])
        mat_3x3 = torch.tensor(mat_3x3, device=device, dtype=dtype)
        return cls.from_mat_3x3(mat_3x3)
class RotationMat3x3(RotationBase):
    default = torch.eye(3)
    def inv(self):
        return RotationMat3x3(torch.transpose(self.tensor, -1, -2))
    def interp1d(self, ts_keyframes: torch.Tensor, ts: torch.Tensor):
        interp_m = torch_consecutive_interp1d_general(
            ts_keyframes, self.tensor, ts, interp_fn=matrix_slerp)
        return RotationMat3x3(interp_m, device=self.device, dtype=self.dtype)
    def mat_3x3(self):
        return self.tensor[:]
    def forward(self, x: torch.Tensor, inv=False):
        # NOTE: x.shape == [*self.prefix, ..., 3] or [*[1]*len(self.prefix), ..., 3]
        if inv:
            mat_3x3 = torch.transpose(self.tensor, -1, -2)
        else:
            mat_3x3 = self.tensor
        # return torch.einsum('...ij,...j->...i', mat_3x3, x)
        if len(prefix:=self.prefix) > 0:
            # If batched: Assume `x` has the same prefix dims as self
            mat_3x3 = mat_3x3.view(*prefix, *[1]*(x.dim()-1-len(prefix)), 3, 3)
        return (mat_3x3 * x.unsqueeze(-2)).sum(-1)
    @classmethod
    def from_mat_3x3(cls, mat_3x3: torch.Tensor):
        return cls(mat_3x3)
class RotationQuaternion(RotationBase):
    default = torch.tensor([1., 0., 0., 0.])
    def inv(self):
        return RotationQuaternion(quat_invert(normalize_quat(self.tensor)))
    def interp1d(self, ts_keyframes: torch.Tensor, ts: torch.Tensor):
        interp_q = torch_consecutive_interp1d_general(
            ts_keyframes, self.tensor, ts, interp_fn=lambda q0, q1, w: \
                unitquat_slerp(normalize_quat(q0), normalize_quat(q1), w))
        return RotationQuaternion(interp_q, device=self.device, dtype=self.dtype)
    def mat_3x3(self):
        return quat_to_rot(normalize_quat(self.tensor))
    def forward(self, x: torch.Tensor, inv=False):
        # NOTE: x.shape == [*self.prefix, ..., 3] or [*[1]*len(self.prefix), ..., 3]
        if inv:
            quat = quat_invert(normalize_quat(self.tensor))
        else:
            quat = self.tensor
        if len(prefix:=self.prefix) > 0:
            # If batched: Assume `x` has the same prefix dims as self
            quat = quat.view(*prefix, *[1]*(x.dim()-1-len(prefix)), 4)
        return quat_apply(normalize_quat(quat), x)
    @classmethod
    def from_mat_3x3(cls, mat_3x3: torch.Tensor):
        quat = rot_to_quat(mat_3x3)
        return cls(quat)
class RotationAxisAngle(RotationBase):
    default = torch.tensor([0., 0., 0.])
    def mat_3x3(self):
        return axis_angle_to_rot(self.tensor)
    def inv(self):
        return RotationAxisAngle(-self.tensor)
    def interp1d(self, ts_keyframes: torch.Tensor, ts: torch.Tensor):
        interp_aa = torch_consecutive_interp1d_general(
            ts_keyframes, self.tensor, ts, interp_fn=axis_angle_slerp)
        return RotationAxisAngle(interp_aa, device=self.device, dtype=self.dtype)
    def forward(self, x: torch.Tensor, inv=False):
        # NOTE: x.shape == [*self.prefix, ..., 3] or [*[1]*len(self.prefix), ..., 3]
        quat = axis_angle_to_quat(self.tensor)
        if inv:
            quat = quat_invert(quat)
        if len(prefix:=self.prefix) > 0:
            # If batched: Assume `x` has the same prefix dims as self
            quat = quat.view(*prefix, *[1]*(x.dim()-1-len(prefix)), 4)
        return quat_apply(quat, x)
    @classmethod
    def from_mat_3x3(cls, mat_3x3: torch.Tensor):
        quat = rot_to_quat(mat_3x3)
        aa = quat_to_axis_angle(quat)
        return cls(aa)
class Rotation6D(RotationBase):
    default = torch.tensor([1., 0., 0., 0., 1., 0.])
    def mat_3x3(self):
        return rot6d_to_rot(self.tensor)
    def interp1d(self, ts_keyframes: torch.Tensor, ts: torch.Tensor):
        #---- Convert at the last step of interpolation
        interp_mat = torch_consecutive_interp1d_general(
            ts_keyframes, self.tensor, ts, interp_fn=lambda R0, R1, w: \
                matrix_slerp(rot6d_to_rot(R0), rot6d_to_rot(R1), w))
        interp_R = rot_to_rot6d(interp_mat)
        # #---- First convert then interpolate
        # interp_mat = torch_consecutive_interp1d_general(
        #     ts_keyframes, rot6d_to_rot(self.tensor), ts, 
        #     interp_fn=lambda R0, R1, w: matrix_slerp(R0, R1, w))
        # interp_R = rot_to_rot6d(interp_mat)
        return Rotation6D(interp_R, device=self.device, dtype=self.dtype)
    def forward(self, x: torch.Tensor, inv=False):
        return RotationMat3x3(self.mat_3x3())(x, inv=inv)
    @classmethod
    def from_mat_3x3(cls, mat_3x3: torch.Tensor):
        rot6d = rot_to_rot6d(mat_3x3)
        return cls(rot6d)
Rotation = RotationMat3x3
RotationQuaternionRefinedAdd = make_refined_add_cls(RotationQuaternion)
RotationAxisAngleRefinedAdd = make_refined_add_cls(RotationAxisAngle)
Rotation6DRefinedAdd = make_refined_add_cls(Rotation6D)
class RotationRefinedMul(AttrNested, RotationBase):
    # NOTE: attr0 and delta could be of different types
    default = {'attr0': Rotation(), 'delta': Rotation()}
    def inv(self) -> 'RotationBase':
        # TODO: a maybe better solution
        return RotationMat3x3(torch.transpose(self.mat_3x3(), -1, -2))
    def mat_3x3(self) -> torch.Tensor:
        # return torch.bmm(self.subattr.delta.mat_3x3(), self.subattr.attr0.mat_3x3())
        # return torch.einsum("...ij,...jk->...ik", self.subattr.delta.mat_3x3(), self.subattr.attr0.mat_3x3())
        return (self.subattr.delta.mat_3x3().unsqueeze(-1) * self.subattr.attr0.mat_3x3().unsqueeze(-3)).sum(-2)
    def forward(self, x: torch.Tensor, inv=False) -> torch.Tensor:
        if inv:
            return self.subattr.attr0(self.subattr.delta(x, inv=True), inv=True)
        else:
            return self.subattr.delta(self.subattr.attr0(x))
    @property
    def prefix(self):
        return self.subattr.attr0.prefix

#---- Translation representations
@AttrBase
class Translation(Attr):
    default = torch.zeros([3,])
    def vec_3(self):
        return self.tensor[:]
    def vec_4_homo(self):
        vec_3 = self.vec_3()
        prefix = vec_3.shape[:-1]
        return torch.cat([vec_3, vec_3.new_ones([*prefix,1])], dim=-1)
TranslationRefinedAdd = make_refined_add_cls(Translation)

#---- Transform representations
@AttrBase
class TransformBase(Attr):
    def mat_3x4(self):
        raise NotImplementedError
    def mat_4x4(self):
        raise NotImplementedError
    @property
    def rot(self) -> RotationBase:
        """
        Returns the underlying rotation representation
        """
        return RotationMat3x3(self.rotation())
    def rotation(self) -> torch.Tensor:
        """
        Returns [3,3] rotation matrix
        """
        raise NotImplementedError
    def translation(self) -> torch.Tensor:
        """
        Returns [3,] transformation vector
        """
        raise NotImplementedError
    def rotation_translation(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns [3,3] rotation matrix and [3,] transformation vector
        """
        raise NotImplementedError
    def rotate(self, x: torch.Tensor, inv=False):
        raise NotImplementedError
    def inv(self):
        raise NotImplementedError
    def forward(self, x: torch.Tensor, inv=False):
        raise NotImplementedError
    def interp1d(self, ts_keyframes: torch.Tensor, ts: torch.Tensor):
        # NOTE: For most of the rotation representations, interpolation should not be linear.
        raise NotImplementedError
class TransformRT(AttrNested, TransformBase):
    # NOTE: AttrNested must be in front of TransformBase
    default = {'rot': Rotation(), 'trans': Translation()}
    def mat_3x4(self):
        return torch.cat([self.subattr.rot.mat_3x3(), self.subattr.trans.vec_3().unsqueeze(-1)], dim=-1)
    def mat_4x4(self):
        mat_3x4 = self.mat_3x4()
        return torch.cat([mat_3x4, check_to_torch([[0,0,0,1]], ref=self).tile([*mat_3x4.shape[:-2], 1, 1])], dim=-2)
    @property
    def rot(self) -> RotationBase:
        return self.subattr.rot
    def rotation(self) -> torch.Tensor:
        return self.subattr.rot.mat_3x3()
    def translation(self) -> torch.Tensor:
        return self.subattr.trans.vec_3()
    def inv(self):
        return TransformRT(rot=self.subattr.rot.inv(), trans=self.subattr.rot(-self.subattr.trans.vec_3(), inv=True))
    def interp1d(self, ts_keyframes: torch.Tensor, ts: torch.Tensor):
        return AttrNested.interp1d(self, ts_keyframes, ts)
    def rotate(self, x: torch.Tensor, inv=False):
        return self.rot(x, inv=inv)
    def forward(self, x: torch.Tensor, inv=False):
        # NOTE: x.shape == [*self.prefix, ..., 3] or [*[1]*len(self.prefix), ..., 3]
        rot = self.rot
        translation = self.translation()
        if len(prefix:=self.prefix) > 0:
            # If batched: Assume `x` has the same prefix dims as self
            translation = translation.view(*prefix, *[1]*(x.dim()-1-len(prefix)), 3)
        if inv:
            return rot.inv()(x - translation)
        else:
            return rot(x) + translation
    @property
    def prefix(self):
        return self.subattr.trans.prefix
class TransformMat4x4(TransformBase):
    default = torch.eye(4)
    def mat_3x4(self):
        return self.tensor[..., :3, :4]
    def mat_4x4(self):
        return self.tensor[...]
    @property
    def rot(self) -> RotationMat3x3:
        return RotationMat3x3(self.tensor[..., :3, :3])
    def interp1d(self, ts_keyframes: torch.Tensor, ts: torch.Tensor):
        rotation = self.rotation()
        translation = self.translation()
        interp_r = torch_consecutive_interp1d_general(
            ts_keyframes, rotation, ts, interp_fn=matrix_slerp)
        interp_t = torch_consecutive_interp1d(ts_keyframes, translation, ts)
        interp_m3x4 = torch.cat([interp_r, interp_t.unsqueeze(-1)], dim=-1)
        return TransformMat4x4(torch.cat([interp_m3x4, check_to_torch([[0,0,0,1]], ref=self).tile([*ts.shape, 1, 1])], dim=-2))
    def rotation(self) -> torch.Tensor:
        return self.tensor[..., :3, :3]
    def translation(self) -> torch.Tensor:
        return self.tensor[..., :3, 3]
    def inv(self):
        rot_inv = torch.transpose(self.tensor[..., :3, :3], -1, -2)
        # mat_3x4 = torch.cat([rot_inv, -torch.einsum('...ij,...j->...i', rot_inv, self.translation()).unsqueeze(-1)], dim=-1)
        mat_3x4 = torch.cat([rot_inv, -(rot_inv*self.translation().unsqueeze(-2)).sum(-1,keepdim=True)], dim=-1)
        return TransformMat4x4(torch.cat([mat_3x4, check_to_torch([[0,0,0,1]], ref=self).tile([*self.prefix, 1, 1])], dim=-2))
    def rotate(self, x: torch.Tensor, inv=False):
        # NOTE: x.shape == [*self.prefix, ..., 3] or [*[1]*len(self.prefix), ..., 3]
        rotation = self.rotation()
        if len(prefix:=self.prefix) > 0:
            # If batched: Assume `x` has the same prefix dims as self
            rotation = rotation.view(*prefix, *[1]*(x.dim()-1-len(prefix)), 3, 3)
        if inv:
            # return torch.einsum('...ij,...j->...i', torch.transpose(rotation, -1, -2), x)
            return (torch.transpose(rotation,-1,-2) * x.unsqueeze(-2)).sum(-1)
        else:
            # return torch.einsum('...ij,...j->...i', rotation, x)
            return (rotation*x.unsqueeze(-2)).sum(-1)
    def forward(self, x: torch.Tensor, inv=False):
        # NOTE: x.shape == [*self.prefix, ..., 3] or [*[1]*len(self.prefix), ..., 3]
        rotation = self.rotation()
        translation = self.translation()
        if len(prefix:=self.prefix) > 0:
            # If batched: Assume `x` has the same prefix dims as self
            rotation = rotation.view(*prefix, *[1]*(x.dim()-1-len(prefix)), 3, 3)
            translation = translation.view(*prefix, *[1]*(x.dim()-1-len(prefix)), 3)
        if inv:
            # return torch.einsum('...ij,...j->...i', torch.transpose(rotation, -1, -2), x - translation)
            return (torch.transpose(rotation,-1,-2) * (x-translation).unsqueeze(-2)).sum(-1)
        else:
            # return torch.einsum('...ij,...j->...i', rotation, x) + translation
            return (rotation * x.unsqueeze(-2)).sum(-1) + translation
class TransformMat3x4(TransformMat4x4):
    default = torch.tensor([[1.,0.,0.,0.], [0.,1.,0.,0.], [0.,0.,1.,0.]])
    def mat_4x4(self):
        return torch.cat([self.tensor, check_to_torch([[0,0,0,1]], ref=self).tile([*self.tensor.shape[:-2], 1, 1])], dim=-2)
class TransformExpSE3(AttrNested, TransformBase):
    """
    Modified from inerf: https://github.com/salykovaa/inerf/blob/11c03a1bfe4c1007736e6c199838e3b48cf4d51c/inerf_helpers.py#L18
    """
    default = {'w': Vector_3(torch.zeros([3,])), 'v': Vector_3(torch.zeros([3,])), 'theta': Scalar(torch.zeros([]))}
    def mat_3x4(self) -> torch.Tensor:
        rotation, translation = self.rotation_translation()
        return torch.cat([rotation, translation.unsqueeze(-1)], -1)
    def mat_4x4(self) -> torch.Tensor:
        rotation, translation = self.rotation_translation()
        prefix = translation.shape[:-1]
        exp_i = torch.eye([*prefix, 4, 4])
        exp_i[..., :3, :3] = rotation
        exp_i[..., :3, 3] = translation
        return exp_i
    def rotation(self) -> torch.Tensor:
        # w = self.subattr.w.tensor; w_ss = skew_symmetric(w); w_ss2 = torch.einsum('...ij,...jk->...ik', w_ss, w_ss)
        w = self.subattr.w.tensor; w_ss = skew_symmetric(w); w_ss2 = (w_ss.unsqueeze(-1) * w_ss.unsqueeze(-3)).sum(-2)
        theta = self.subattr.theta.tensor[...,None,None]; sin_theta = torch.sin(theta); cos_theta = torch.cos(theta)
        prefix = w.shape[:-1]
        I = torch.eye(3, dtype=self.dtype, device=self.device).tile([*prefix, 1,1])
        return I + sin_theta * w_ss + (1 - cos_theta) * w_ss2
    def translation(self) -> torch.Tensor:
        # w = self.subattr.w.tensor; w_ss = skew_symmetric(w); w_ss2 = torch.einsum('...ij,...jk->...ik', w_ss, w_ss)
        w = self.subattr.w.tensor; w_ss = skew_symmetric(w); w_ss2 = (w_ss.unsqueeze(-1) * w_ss.unsqueeze(-3)).sum(-2)
        v = self.subattr.v.tensor
        theta = self.subattr.theta.tensor[...,None,None]; sin_theta = torch.sin(theta); cos_theta = torch.cos(theta)
        prefix = w.shape[:-1]
        I = torch.eye(3, dtype=self.dtype, device=self.device).tile([*prefix, 1,1])
        # return torch.matmul(I * theta + (1 - cos_theta) * w_ss + (theta - sin_theta) * w_ss2, v)
        # return torch.einsum('...ij,...j->...i', I * theta + (1 - cos_theta) * w_ss + (theta - sin_theta) * w_ss2, v)
        return ((I * theta + (1 - cos_theta) * w_ss + (theta - sin_theta) * w_ss2) * v.unsqueeze(-2)).sum(-1)
    def rotation_translation(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # w = self.subattr.w.tensor; w_ss = skew_symmetric(w); w_ss2 = torch.einsum('...ij,...jk->...ik', w_ss, w_ss)
        w = self.subattr.w.tensor; w_ss = skew_symmetric(w); w_ss2 = (w_ss.unsqueeze(-1) * w_ss.unsqueeze(-3)).sum(-2)
        v = self.subattr.v.tensor
        theta = self.subattr.theta.tensor[...,None,None]; sin_theta = torch.sin(theta); cos_theta = torch.cos(theta)
        prefix = w.shape[:-1]
        I = torch.eye(3, dtype=self.dtype, device=self.device).tile([*prefix, 1,1])
        rotation = I + sin_theta * w_ss + (1 - cos_theta) * w_ss2
        # translation = torch.matmul(I * theta + (1 - cos_theta) * w_ss + (theta - sin_theta) * w_ss2, v)
        # translation = torch.einsum('...ij,...j->...i', I * theta + (1 - cos_theta) * w_ss + (theta - sin_theta) * w_ss2, v)
        translation = ((I * theta + (1 - cos_theta) * w_ss + (theta - sin_theta) * w_ss2) * v.unsqueeze(-2)).sum(-1)
        return rotation, translation
    def rotate(self, x: torch.Tensor, inv=False):
        # NOTE: x.shape == [*self.prefix, ..., 3] or [*[1]*len(self.prefix), ..., 3]
        rotation = self.rotation()
        if len(prefix:=self.prefix) > 0:
            # If batched: Assume `x` has the same prefix dims as self
            rotation = rotation.view(*prefix, *[1]*(x.dim()-1-len(prefix)), 3, 3)
        if inv:
            # return torch.einsum('...ij,...j->...i', rotation.transpose(-1,-2), x)
            return (torch.transpose(rotation,-1,-2)*x.unsqueeze(-2)).sum(-1)
        else:
            # return torch.einsum('...ij,...j->...i', rotation, x)
            return (rotation*x.unsqueeze(-2)).sum(-1)
    def forward(self, x: torch.Tensor, inv=False):
        # NOTE: x.shape == [*self.prefix, ..., 3] or [*[1]*len(self.prefix), ..., 3]
        rotation, translation = self.rotation_translation()
        if len(prefix:=self.prefix) > 0:
            # If batched: Assume `x` has the same prefix dims as self
            rotation = rotation.view(*prefix, *[1]*(x.dim()-1-len(prefix)), 3, 3)
            translation = translation.view(*prefix, *[1]*(x.dim()-1-len(prefix)), 3)
        if inv:
            # return torch.einsum('...ij,...j->...i', rotation.transpose(-1,-2), x-translation)
            return (torch.transpose(rotation,-1,-2)*(x-translation).unsqueeze(-2)).sum(-1)
        else:
            # return torch.einsum('...ij,...j->...i', rotation, x) + translation
            return (rotation*x.unsqueeze(-2)).sum(-1) + translation
    @property
    def prefix(self):
        return self.subattr.w.prefix
Transform = TransformMat4x4

#---- Scaling representations
@AttrBase
class Scale(Attr):
    default = torch.ones([3,])
    def vec_3(self):
        return self.tensor[:]
    def vec_4_homo(self):
        vec_3 = self.vec_3()
        prefix = vec_3.shape[:-1]
        return torch.cat([vec_3, vec_3.new_ones([*prefix,1])], dim=-1)
    def mat_3x3(self):
        return torch.diagflat(self.vec_3())
    def mat_4x4(self):
        return torch.diagflat(self.vec_4_homo())
    def forward(self, x: torch.Tensor, inv=False):
        if inv:
            return x / self.vec_3()
        else:
            return x * self.vec_3()
class ScaleExp(Scale):
    default = torch.zeros([3,])
    def vec_3(self):
        return torch.exp(self.tensor)
class ScaleSquare(Scale):
    def vec_3(self):
        return self.tensor ** 2
