"""
@file   importance.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Pytorch importance sampling on frames & on pixels using accumulated error maps.
"""

import numpy as np
from numbers import Number
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn

class ErrorMap(nn.Module):
    """
    Pixel importance sampling via 2D invert CDF sampling on accumulated error maps
    
    NOTE: Code convention:
    x, y: float tensor in range [0,1]
    w, h: integer tensor in range [0,W-1] or [0,H-1]
    x <=> w;   y <=> h
    """
    def __init__(
        self, 
        n_images: int, error_map_hw: Tuple[int, int],
        # image_res: Tuple[int, int], 
        *, 
        #---- PDF configs
        min_pdf: float = 0.01, 
        max_pdf: float = None, 
        #---- Update configs
        n_steps_init: int = 128, 
        n_steps_max: int = None, 
        n_steps_growth_factor: float = 1.5, 
        dtype=torch.float, device=None
        ) -> None:
        """ Pixel importance sampling via 2D invert CDF sampling on accumulated error maps

        Args:
            n_images (int): Number of images in the dataset.
            hw (Tuple[int, int]): Resolution of the error map. Usually much coarser than the original image.
            min_pdf (float, optional): Minimum pdf value to prevent all zero or too small error maps. Defaults to 0.01.
            n_steps_init (int, optional): The initial CDF update period (`n_steps_between_update`), in iters unit. Default is 128.
            n_steps_growth_factor (float, optional): The growth factor of cdf update period of the next update w.r.t the previous one. Defaults to 1.5
            n_steps_max (int, optional): The maximum cdf update period. If None, there is no limitations. Defaults to None.
            device (torch.device, optional): torch device of the error maps. Defaults to torch.device('cuda').
            dtype (torch.dtype, optional): torch dtype of the error maps. Defaults to torch.float.
        """
        super().__init__()
    
        self.dtype = dtype

        res_y, res_x = error_map_hw
        # img_res_y, img_res_x = image_res
        error_map = torch.zeros([n_images, res_y, res_x], device=device, dtype=dtype)
        
        self.n_images, self.res_y, self.res_x = n_images, res_y, res_x
        # self.image_res_y, self.image_res_x = img_res_y, img_res_x
        self.register_buffer('error_map', error_map, persistent=True) # [N, H, W]

        self.cdf_x_cond_y: torch.Tensor = None # [N, H, W]
        self.cdf_y: torch.Tensor = None # [N, H]
        self.cdf_img: torch.Tensor = None # [N]
        
        self.min_pdf = min_pdf
        self.max_pdf = max_pdf
        min_cdf_x_cond_y = torch.arange(res_x, dtype=dtype, device=device).add_(1).div_(res_x).tile(n_images,res_y,1) # [N, H, W]
        min_cdf_y = torch.arange(res_y, dtype=dtype, device=device).add_(1).div_(res_y).tile(n_images,1) # [N, H]
        min_cdf_img = torch.arange(n_images, dtype=dtype, device=device).add_(1).div_(n_images) # [N]
        self.register_buffer('min_cdf_x_cond_y', min_cdf_x_cond_y, persistent=False)
        self.register_buffer('min_cdf_y', min_cdf_y, persistent=False)
        self.register_buffer('min_cdf_img', min_cdf_img, persistent=False)
        
        #-------------------------------------
        # Training update related
        #-------------------------------------
        self.n_steps_since_update = 0
        self.n_steps_growth_factor = n_steps_growth_factor
        self.n_steps_between_update = n_steps_init
        self.n_steps_max = n_steps_max
        
    @property
    def device(self) -> torch.device:
        return self.error_map.device

    @torch.no_grad()
    def update_error_map(self, i: Union[int, torch.LongTensor], xy: torch.Tensor, val: torch.Tensor):
        """
        Args:
            i: frame indices; [int] or torch.tensor of shape [...]
            xy: pixel coords in [0,1] of shape [..., 2]
            val: error value on the sampled i+xy, of shape [...]
        """
        assert not (val < 0).any(), "Found negative err when updaing error_map. Please check to make sure only non-negative err."
        
        res_y, res_x = self.res_y, self.res_x
        
        x_img, y_img = xy.movedim(-1,0)
        wf, hf = x_img * res_x, y_img * res_y
        w, h = wf.long(), hf.long()
        
        # NOTE: Tri-linear error_map update.
        w_w, w_h = (wf - w), (hf-h)
        w.clamp_(0, res_x-2)
        h.clamp_(0, res_y-2)
        self.error_map[i, h,   w  ] += (1-w_h) * (1-w_w) * val
        self.error_map[i, h+1, w  ] +=    w_h  * (1-w_w) * val
        self.error_map[i, h  , w+1] += (1-w_h) *    w_w  * val
        self.error_map[i, h+1, w+1] +=    w_h  *    w_w  * val

    @torch.no_grad()
    def get_normalized_error_map(self, frame_ind = None):
        error_map = self.error_map[frame_ind] if frame_ind is not None else self.error_map
        if self.max_pdf is not None:
            error_map = error_map.clone().clamp_max_(self.max_pdf)
        else:
            error_map = error_map / error_map.max().clamp_min(1e-5)
        return error_map

    @torch.no_grad()
    def construct_cdf(self):
        """ Construct 2D cdf maps viewing each current error map as 2D pdf
        """
        if self.max_pdf is not None:
            self.error_map.clamp_max_(self.max_pdf)
        
        min_pdf = self.min_pdf
        error_map: torch.Tensor = self.error_map # [N, H, W]
        
        # cdf_x_cond_y
        cdf_x_cond_y = (error_map + 1e-10).cumsum(dim=2) # [N, H, W]
        cumu = pdf_y = cdf_x_cond_y[:,:,-1] # [N, H]
        self.cdf_x_cond_y = (1-min_pdf) * cdf_x_cond_y / cumu.unsqueeze(-1) + min_pdf * self.min_cdf_x_cond_y
        
        # cdf_y
        cdf_y = pdf_y.cumsum(dim=1) # [N, H]
        cumu = pdf_img = cdf_y[:, -1] # [N]
        self.cdf_y = (1-min_pdf) * cdf_y / cumu.unsqueeze(-1) + min_pdf * self.min_cdf_y
        
        # cdf_img
        cdf_img = pdf_img.cumsum(dim=0) # [N]
        self.cdf_img = (1-min_pdf) * cdf_img / cdf_img[-1:] + min_pdf * self.min_cdf_img
    
    @torch.no_grad()
    def construct_cdf_and_clean_error_map(self):
        """ Construct 2D cdf maps and clean current error maps
        """
        self.construct_cdf()
        self.error_map.zero_()
    
    @torch.no_grad()
    def step_error_map(self, i: Union[int, torch.LongTensor], xy: torch.Tensor, val: torch.Tensor):
        self.update_error_map(i=i, xy=xy, val=val)
        self.n_steps_since_update += 1
        if self.n_steps_since_update >= self.n_steps_between_update:
            self.construct_cdf_and_clean_error_map()
            self.n_steps_since_update = 0
            self.n_steps_between_update = int(self.n_steps_growth_factor * self.n_steps_between_update)
            if self.n_steps_max is not None:
                self.n_steps_between_update = min(self.n_steps_between_update, self.n_steps_max)
    
    @torch.no_grad()
    def get_pdf_image(self) -> torch.Tensor:
        """ Get the pdf (pmf) of different images

        Returns:
            torch.Tensor: [num_images,] pdf (pmf) of different image frames.
        """
        cdf_img = self.cdf_img
        pdf_img = cdf_img.diff(prepend=cdf_img.new_zeros([1,]))
        return pdf_img
    
    @torch.no_grad()
    def sample_pixel(self, num_samples: int, frame_ind: Union[int, torch.LongTensor]) -> torch.Tensor:
        """ Sample some pixel locations using weighted sampling.
        NOTE: Conduct 2D invert CDF sampling (2D weighted sampling) on the given frame(s) of image

        Args:
            num_samples (int): number of sampling points.
            frame_ind (Union[int, torch.LongTensor]): the given frame indice(s) of the image frame to sample

        Returns:
            torch.Tensor: ([..., num_samples,2], [..., num_samples,2]). The sampled xy (for WH) values, \
                having the same prefix with frame_ind, with range [0,1]
        """
        res_y, res_x = self.res_y, self.res_x
        
        #----------- Sample CDF 2D
        x, y = torch.rand([2, num_samples], dtype=self.dtype, device=self.device).clamp_(1e-6, 1-1e-6)
        
        cdf_y = self.cdf_y
        h = torch.searchsorted(cdf_y[frame_ind], y.unsqueeze(-1), right=False).squeeze(-1)
        prev = torch.where(h > 0, cdf_y[frame_ind, h-1], cdf_y.new_zeros([]))
        y = ((y - prev) / (cdf_y[frame_ind, h] - prev) + h) / res_y
        
        cdf_x_cond_y = self.cdf_x_cond_y
        w = torch.searchsorted(cdf_x_cond_y[frame_ind, h], x.unsqueeze(-1), right=False).squeeze(-1)
        prev = torch.where(w > 0, cdf_x_cond_y[frame_ind, h, w-1],  cdf_x_cond_y.new_zeros([]))
        x = ((x - prev) / (cdf_x_cond_y[frame_ind, h, w] - prev) + w) / res_x
        xy = torch.stack([x, y], dim=-1)
        return xy

    @torch.no_grad()
    def sample_img(self, num_samples: int) -> torch.Tensor:
        """ Sample some image frame inds using weighted sampling.
        NOTE: Conduct 1D invert CDF sampling (weighted sampling) of frame inds.

        Args:
            num_samples (int): number of frames to sample

        Returns:
            torch.Tensor: [num_samples,] The sampled frame indices.
        """
        i = torch.rand([num_samples], device=self.device, dtype=self.dtype).clamp_(1e-6, 1-1e-6)
        i = torch.searchsorted(self.cdf_img, i, right=False)
        return i
    
    @torch.no_grad()
    def sample_img_pixel(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Sample image frame inds and pixel locations simultaneously.

        Args:
            num_samples (int): Number of samples

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: ([num_samples, ], [num_samples, 2]) The sampled image frame indices and pixel xy (WH) locations
        """

        i = self.sample_img(num_samples)
        xy = self.sample_pixel(num_samples, i)
        return i, xy

class ImpSampler(nn.Module):
    """
    Pixel importance sampling via 2D invert CDF sampling on accumulated error maps
    
    NOTE: Code convention:
    x, y: float tensor in range [0,1]
    w, h: integer tensor in range [0,W-1] or [0,H-1]
    x <=> w;   y <=> h
    """
    def __init__(
        self, 
        #---- Multiple sampling choices
        error_maps: Dict[str, Tuple[ErrorMap, float]] = {}, 
        frac_uniform: float = 0.5, 
        ) -> None:
        super().__init__()
        
        error_map_names: List[str] = list(error_maps.keys())
        error_map_list: List[ErrorMap] = [v[0] for v in error_maps.values()]
        error_map_fracs: List[float] = [v[1] for v in error_maps.values()]
        
        for error_map in error_map_list:
            if error_map.cdf_img is None:
                error_map.construct_cdf()

        assert all(error_map_list[0].n_images == e.n_images for e in error_map_list), \
            "`error_maps` should have the same `n_images`"
        self.error_maps: Dict[str, ErrorMap] = nn.ModuleDict(dict(zip(error_map_names, error_map_list)))
        self.n_images = error_map_list[0].n_images
        
        # Normalize fracs
        non_uniform_fracs = np.array(error_map_fracs)
        self.error_map_fracs = ((1-frac_uniform) * non_uniform_fracs / non_uniform_fracs.sum()).tolist()
        self.frac_uniform = frac_uniform
        self.error_map_names = error_map_names

    @property
    def device(self) -> torch.device:
        return self.error_maps[self.error_map_names[0]].device
    
    @property
    def dtype(self):
        return self.error_maps[self.error_map_names[0]].dtype

    @torch.no_grad()
    def get_pdf_image(self) -> torch.Tensor:
        pdf_list = []
        if self.frac_uniform > 0:
            pdf_list.append(self.frac_uniform * torch.full((self.n_images,), 1./self.n_images, dtype=self.dtype, device=self.device))
        for frac, error_map in zip(self.error_map_fracs, self.error_maps.values()):
            pdf_list.append(frac * error_map.get_pdf_image())
        pdf_list = torch.stack(pdf_list, 0).sum(0)
        return pdf_list

    @torch.no_grad()
    def sample_img(self, num_samples: int) -> torch.Tensor:
        n_uniform = int(num_samples * self.frac_uniform)
        n_error_map = [int(num_samples * frac) for frac in self.error_map_fracs]
        n_error_map[-1] = num_samples - (n_uniform + sum(n_error_map[:-1])) # Make sure sum is exactly `num_samples`
        i = []
        if n_uniform > 0:
            i.append(torch.randint(self.n_images, [n_uniform,], dtype=torch.long, device=self.device))
        for n, error_map in zip(n_error_map, self.error_maps.values()):
            if n > 0:
                i.append(error_map.sample_img(n))
        i = torch.cat(i, dim=0)
        return i

    @torch.no_grad()
    def sample_pixel(self, num_samples: int, frame_ind: int) -> torch.Tensor:
        assert isinstance(frame_ind, Number), "`frame_ind` must be a single frame index"
        n_uniform = int(num_samples * self.frac_uniform)
        n_error_map = [int(num_samples * frac) for frac in self.error_map_fracs]
        n_error_map[-1] = num_samples - (n_uniform + sum(n_error_map[:-1])) # Make sure sum is exactly `num_samples`
        xy = []
        if n_uniform > 0:
            xy.append(torch.rand([n_uniform, 2], dtype=self.dtype, device=self.device).clamp_(1e-6, 1-1e-6))
        for n, error_map in zip(n_error_map, self.error_maps.values()):
            if n > 0:
                xy.append(error_map.sample_pixel(n, frame_ind))
        xy = torch.cat(xy, dim=0)
        return xy

    @torch.no_grad()
    def sample_img_pixel(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        All fractions = uniform + error_map0 + error_map1 + .... = 1.0
        """
        i = []
        xy = []
        n_uniform = int(num_samples * self.frac_uniform)
        n_error_map = [int(num_samples * frac) for frac in self.error_map_fracs]
        n_error_map[-1] = num_samples - (n_uniform + sum(n_error_map[:-1])) # Make sure sum is exactly `num_samples`
        if n_uniform > 0:
            i.append(torch.randint(self.n_images, [n_uniform,], dtype=torch.long, device=self.device))
            xy.append(torch.rand([n_uniform, 2], dtype=self.dtype, device=self.device).clamp_(1e-6, 1-1e-6))
        for n, (name, error_map) in zip(n_error_map, self.error_maps.items()):
            if n > 0:
                _i,_xy = error_map.sample_img_pixel(n)
                i.append(_i)
                xy.append(_xy)
        i = torch.cat(i, dim=0)
        xy = torch.cat(xy, dim=0)
        return i, xy

if __name__ == "__main__":
    def unit_test(device=torch.device('cuda')):
        from torch.utils.benchmark import Timer
        from icecream import ic
        
        m = ErrorMap(7, (128,128), device=device, dtype=torch.float)
        m.error_map.uniform_(0.2, 1)
        m.update_error_map(0, torch.rand([130, 2], device=device, dtype=torch.float), torch.rand([130], device=device, dtype=torch.float))
        m.construct_cdf()
        
        pdf = m.get_pdf_image()
        ic(pdf.shape, pdf.sum().item())
        
        xy = m.sample_pixel(13, 0)
        ic(xy.shape)
        xy = m.sample_pixel(13, torch.randint(7, [13,], device=device, dtype=torch.long))
        ic(xy.shape)
        i, xy = m.sample_img_pixel(13)
        ic(i.shape, xy.shape)
        
        sampler = ImpSampler({'name': (m, 0.5)})
        i, xy = sampler.sample_img_pixel(13)
        ic(i.shape, xy.shape)
        
        pdf = sampler.get_pdf_image()
        ic(pdf.shape, pdf.sum().item())
        
        # 338 us
        print(Timer(
            stmt='m.sample_img_pixel(16384)', 
            globals={'m':m}
        ).blocked_autorange())

        # 395 us
        print(Timer(
            stmt='sampler.sample_img_pixel(16384)', 
            globals={'sampler':sampler}
        ).blocked_autorange())

    unit_test()
