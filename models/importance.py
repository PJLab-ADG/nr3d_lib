"""
@file   importance.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Pytorch importance sampling on frames & on pixels using accumulated error maps.
"""

from typing import Tuple, Union

import torch
import torch.nn as nn

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
        n_images: int, error_res: Tuple[int, int],
        # image_res: Tuple[int, int], 
        *, 
        #---- PDF configs
        min_pdf: float = 0.01, uniform_sampling_fraction: float = 0.5, 
        #---- Update configs
        n_steps_init: int = 128, n_steps_growth_factor: float = 1.5, n_steps_max: int = None, 
        device=torch.device('cuda'), dtype=torch.float
        ) -> None:
        """ Pixel importance sampling via 2D invert CDF sampling on accumulated error maps

        Args:
            n_images (int): Number of images in the dataset.
            error_res (Tuple[int, int]): Resolution of the error map. Usually much coarser than the original image.
            min_pdf (float, optional): Minimum pdf value to prevent all zero or too small error maps. Defaults to 0.01.
            uniform_sampling_fraction (float, optional): A part of pixels use uniform sampling. Defaults to 0.5.
            n_steps_init (int, optional): The initial CDF update period (`n_steps_between_update`), in iters unit. Default is 128.
            n_steps_growth_factor (float, optional): The growth factor of cdf update period of the next update w.r.t the previous one. Defaults to 1.5
            n_steps_max (int, optional): The maximum cdf update period. If None, there is no limitations. Defaults to None.
            device (torch.device, optional): torch device of the error maps. Defaults to torch.device('cuda').
            dtype (torch.dtype, optional): torch dtype of the error maps. Defaults to torch.float.
        """
        super().__init__()
    
        self.device = device
        self.dtype = dtype
    
        res_y, res_x = error_res
        # img_res_y, img_res_x = image_res
        error_map = torch.zeros([n_images, res_y, res_x], device=device, dtype=dtype)
        
        self.n_images, self.res_y, self.res_x = n_images, res_y, res_x
        # self.image_res_y, self.image_res_x = img_res_y, img_res_x
        self.register_buffer('error_map', error_map, persistent=True) # [N, H, W]
        
        self.cdf_x_cond_y: torch.Tensor = None # [N, H, W]
        self.cdf_y: torch.Tensor = None # [N, H]
        self.cdf_img: torch.Tensor = None # [N]
        
        self.min_pdf = min_pdf
        self.min_cdf_x_cond_y = torch.arange(res_x, dtype=dtype, device=device).add_(1).div_(res_x).tile(n_images,res_y,1) # [N, H, W]
        self.min_cdf_y = torch.arange(res_y, dtype=dtype, device=device).add_(1).div_(res_y).tile(n_images,1) # [N, H]
        self.min_cdf_img = torch.arange(n_images, dtype=dtype, device=device).add_(1).div_(n_images) # [N]

        self.uniform_sampling_fraction = uniform_sampling_fraction
        
        #-------------------------------------
        # Training update related
        #-------------------------------------
        self.n_steps_since_update = 0
        self.n_steps_growth_factor = n_steps_growth_factor
        self.n_steps_between_update = n_steps_init
        self.n_steps_max = n_steps_max

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
    def construct_cdf(self):
        """ Construct 2D cdf maps viewing each current error map as 2D pdf
        """
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
    def _get_pdf_image(self) -> torch.Tensor:
        """ Get the pdf (pmf) of different images

        Returns:
            torch.Tensor: [num_images,] pdf (pmf) of different image frames.
        """
        cdf_img = self.cdf_img
        pdf_img = cdf_img.diff(prepend=cdf_img.new_zeros([1,]))
        return pdf_img

    @torch.no_grad()
    def get_pdf_image(self) -> torch.Tensor:
        """ Get the pdf (pmf) of different images, considering uniform sapling fraction

        Returns:
            torch.Tensor: [num_images,] pdf (pmf) of different image frames.
        """
        # A mixture of CDF sampling & uniform sampling
        cdf_img = self.cdf_img
        pdf_img = cdf_img.diff(prepend=cdf_img.new_zeros([1,]))
        return self.uniform_sampling_fraction / self.n_images + (1.-self.uniform_sampling_fraction) * pdf_img

    @torch.no_grad()
    def _sample_pixel(self, num_samples: int, frame_ind: Union[int, torch.LongTensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Sample some pixel locations, using only weighted sampling.
        NOTE: Conduct 2D invert CDF sampling (2D weighted sampling) on the given frame(s) of image

        Args:
            num_samples (int): number of sampling points.
            frame_ind (Union[int, torch.LongTensor]): the given frame indice(s) of the image frame to sample

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: ([..., num_samples,], [..., num_samples,]). The sampled y, x (H, W) values, having the same prefix with frame_ind, with range [0,1]
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
        
        return y, x
    
    @torch.no_grad()
    def sample_pixel(self, num_samples: int, frame_ind: Union[int, torch.LongTensor]) -> torch.Tensor:
        """ Sample some pixel locations, using weighted sampling and uniform sampling.
        NOTE: A part of pixel locations are uniformly sampled; others are from 2D invert CDF sampling (2D weighted sampling) on the given frame(s) of image

        Args:
            num_samples (int): number of sampling points.
            frame_ind (Union[int, torch.LongTensor]): the given frame indice(s) of the image frame to sample

        Returns:
            torch.Tensor: [..., num_samples, 2]. The sampled xy (WH) values, having the the same prefix with frame_ind, with range [0,1]
        """
        num_uniform = int(num_samples * self.uniform_sampling_fraction)
        num_importance = num_samples - num_uniform
        if isinstance(frame_ind, torch.Tensor):
            assert frame_ind.numel() == num_samples, f"Expect `frame_ind` to have size={[num_samples]}"
            frame_ind = frame_ind[:num_importance]
        y, x = self._sample_pixel(num_importance, frame_ind)
        xy = torch.stack([x, y], dim=-1)
        if num_uniform > 0:
            xy2 = torch.rand([num_uniform, 2], dtype=self.dtype, device=self.device).clamp_(1e-6, 1-1e-6)
            xy = torch.cat([xy, xy2], dim=0)
        return xy
    
    @torch.no_grad()
    def _sample_img(self, num_frames: int) -> torch.Tensor:
        """ Sample some image frame inds, using only weighted sampling.
        NOTE: Conduct 1D invert CDF sampling (weighted sampling) of frame inds.

        Args:
            num_frames (int): number of frames to sample

        Returns:
            torch.Tensor: [num_frames,] The sampled frame indices.
        """
        i = torch.rand([num_frames], device=self.device, dtype=self.dtype).clamp_(1e-6, 1-1e-6)
        i = torch.searchsorted(self.cdf_img, i, right=False)
        return i
    
    @torch.no_grad()
    def sample_img(self, num_frames: int):
        """ Sample some image frame inds, using weighted sampling and uniform sampling.
        NOTE: A part of frames are uniformly sampled and others are weighted sampled.

        Args:
            num_frames (int): number of frames to sample

        Returns:
            torch.Tensor: The sampled frame indices.
        """
        pdf = self.get_pdf_image()
        i = torch.multinomial(pdf, num_frames, replacement=False)
        return i
    
    @torch.no_grad()
    def sample_img_pixel(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Sample image frame inds and pixel locations simultaneously.
        NOTE: A part of data is sampled uniformly; others weighted sampled

        Args:
            num_samples (int): Number of samples

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: ([num_samples, ], [num_samples, 2]) The sampled image frame indices and pixel xy (WH) locations
        """
        num_uniform = int(num_samples * self.uniform_sampling_fraction)
        num_importance = num_samples - num_uniform

        i = torch.rand([num_importance], device=self.device, dtype=self.dtype).clamp_(1e-6, 1-1e-6)
        i = torch.searchsorted(self.cdf_img, i, right=False)
        y, x = self._sample_pixel(num_importance, i)
        xy = torch.stack([x,y], dim=-1)
        if num_uniform > 0:
            i2 = torch.randint(len(self.cdf_img), [num_uniform], device=self.device, dtype=torch.long)
            i = torch.cat([i, i2], dim=0)
            xy2 = torch.rand([num_uniform, 2], dtype=self.dtype, device=self.device).clamp_(1e-6, 1-1e-6)
            xy = torch.cat([xy, xy2], dim=0)
        return i, xy

if __name__ == "__main__":
    def unit_test(device=torch.device('cuda')):
        from torch.utils.benchmark import Timer
        from icecream import ic
        m = ImpSampler(7, (128,128), device=device, dtype=torch.float)
        m.error_map.uniform_(0.2, 1)
        m.update_error_map(0, torch.rand([2, 130], device=device, dtype=torch.float), torch.rand([130], device=device, dtype=torch.float))
        m.construct_cdf()
        
        pdf = m.get_pdf_image()
        ic(pdf.shape, pdf.sum().item())
        
        xy = m.sample_pixel(13, 0)
        ic(xy.shape)
        xy = m.sample_pixel(13, torch.randint(7, [13,], device=device, dtype=torch.long))
        ic(xy.shape)
        i, xy = m.sample_img_pixel(13)
        ic(i.shape, xy.shape)
        
        # 288 us
        print(Timer(
            stmt='m.sample_img_pixel(16384)', 
            globals={'m':m}
        ).blocked_autorange())
        
    unit_test()
