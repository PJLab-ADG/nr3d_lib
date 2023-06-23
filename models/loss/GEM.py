"""
@file   GEM.py
@author Qiusheng Huang, Shanghai AI Lab
@brief  GEM loss, for optimizing the latent space z
"""

__all__ = [
    'Llm_process', 
    'Iso_loss'
]

# tmp
import torch
import torch.nn as nn
import torch.nn.functional as F

from nr3d_lib.fmt import log
import torch.nn as nn
from nr3d_lib.models.layers import DenseLayer

class Llm_process(nn.Module):

    def __init__(self, w, manifold_dim):
        super(Llm_process, self).__init__()
        self.w = w
        self.manifold_dim = manifold_dim

    def forward(self, gt_x, latents, z):
        """
            inputs :
                gt_x : input coordinary [B, num_points, 3]
                gt_sdf : input coordinary [B, num_points, 1]
                latents : all latents z
                z : selected latents [B, dim]
            returns :
                out : self attention value + input feature 
                attention: B X C X C
        """
        xs = gt_x
        latent_chunk = 1
        with torch.no_grad():
            chunks = max(z.size(0) // 16, 1)
            zs = torch.chunk(z, chunks, 0)
            idxs = []

            for zi in zs:
                diff = zi[:, None] - latents[None, :]
                s = diff.size()
                diff =  diff.view(s[0], s[1], latent_chunk, -1)

                diff_val = torch.norm(diff, p='fro', dim=-1)
                _, idx = torch.topk(diff_val, self.manifold_dim+1, dim=1, largest=False)
                idxs.append(idx)

            idx = torch.cat(idxs, dim=0)

            # Don't include the current entry
            idx = idx[:, 1:].permute(0, 2, 1).contiguous()
            idx = idx[:, :, :, None].repeat(1, 1, 1, z.size(1))
        
        s = idx.size()
        idx = idx.view(-1, s[-1])
        latents_dense = torch.gather(latents, 0, idx)

        latents_select = latents_dense.view(s[0], s[1], s[2], s[3])
        select_idx = torch.arange(s[3]).view(latent_chunk, -1).to(latents_select.device)
        select_idx = select_idx[None, :, None, :].repeat(s[0], 1, self.manifold_dim, 1)
        subset_latent = torch.gather(latents_select, -1, select_idx)

        s = subset_latent.size()
        latents_select = subset_latent.view(s[0] * s[1], s[2], s[3])
        z_dense = z.view(-1, latents_select.size(-1))

        latents_center =  latents_select - z_dense[:, None, :]

        latents_permute = latents_center.permute(0, 2, 1)

        dot_matrix = torch.bmm(latents_center, latents_permute)
        ones = torch.ones(latents_center.size(0), latents_center.size(1), 1).to(latents_center.device)

        dot_matrix_inv = torch.inverse(dot_matrix)
        weights = torch.bmm(dot_matrix_inv, ones)
        weights_sum = weights.sum(dim=1, keepdim=True)
        weights = weights / weights_sum

         # Regenerate with grad
        latents_linear = (weights * latents_select).sum(dim=1)
        latents_linear = latents_linear.view(-1, latents.size(-1))

        # regularization for weight
        inv_weight = torch.clamp(-weights, 0, 1000)
        loss_weight = inv_weight.mean()
        return loss_weight, latents_linear

class Iso_loss(nn.Module):

    def __init__(self, w, min_sdf=0.1):
        super(Iso_loss, self).__init__()
        self.w = w
        self.min_sdf = min_sdf
        self.register_parameter('scale_factor',
                                nn.Parameter(torch.ones(1, 1) * 100, requires_grad=True))

    def forward(self, gt_x, gt_sdf, latents, idx):
        """
            inputs :
                gt_x : input coordinary [B, num_points, 3]
                gt_sdf : input coordinary [B, num_points, 1]
                latents : all latents z
                idx : current batch idx
            returns :
                out : distance
        """
        xs = gt_x
        occs = torch.where(gt_sdf<self.min_sdf, 1., 0.)
        occs = torch.where(occs>(-self.min_sdf), 1., 0.)
        with torch.no_grad():
            xs_perm = torch.cat([xs[1:], xs[:1]], dim=0)

            occs_perm = torch.cat([occs[1:], occs[:1]], dim=0)

            xss = torch.chunk(xs, 4, dim=0)
            xss_perm = torch.chunk(xs_perm, 4, dim=0)
            occss = torch.chunk(occs, 4, dim=0)
            occss_perm = torch.chunk(occs_perm, 4, dim=0)
            mses = []

            for xsi, xsi_perm, occsi, occsi_perm in zip(xss, xss_perm, occss, occss_perm):
                dist = torch.norm(xsi[:, :, None, :] - xsi_perm[:, None, :, :], dim=-1)

                filter_mask = torch.ones_like(dist)

                # Prevent matching of spurious points
                dist = dist + filter_mask * 1000. * (1 - occsi[:, :, None, 0]) + filter_mask * 1000. * (1 - occsi_perm[:, None, :, 0])

                min_dist_perm = (dist.min(dim=1)[0] * occsi_perm[:, :, 0]).sum(dim=1) / occsi_perm[:, :, 0].sum(dim=1)
                min_dist = (dist.min(dim=2)[0] * occsi[:, :, 0]).sum(dim=1) / occsi[:, :, 0].sum(dim=1)

                mse = min_dist + min_dist_perm
                mses.append(mse)

            mse = torch.cat(mses, dim=0)

        # latents = prediction['latents']
        # idx = gt['idx']
        latents_i = latents[idx]
        latents_j = torch.cat([latents_i[1:], latents_i[:1]], dim=0)

        dist = (latents_i - latents_j).pow(2).mean(dim=-1)
        # dist = dist[:, 0]

        mse_idx = torch.argsort(mse, dim=0)
        mse = mse[mse_idx[:32]]
        dist = dist[mse_idx[:32]] * self.scale_factor.squeeze()

        return self.w * torch.abs(dist - mse).mean()
