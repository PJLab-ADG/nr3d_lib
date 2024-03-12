"""
@file   plot_2d.py
@author Jianfei Guo, Shanghai AI Lab
@brief  2D Image ploting utilities
"""

__all__ = [
    'draw_2dbox_on_im', 
    'draw_bool_mask_on_im', 
    'draw_int_mask_on_im', 
    'draw_patch_on_im'
]

import cv2
import numpy as np
from numbers import Number

from nr3d_lib.plot.plot_basic import choose_opposite_color

def draw_2dbox_on_im(
    im: np.ndarray, # np.uint8, 0~255, [H, W, 3]
    center_x: Number, center_y: Number, width: Number, height: Number, 
    color = (255, 0, 0), 
    fillalpha: float = 0.1, 
    linewidth: int = 2, 
    # Optionally draw label
    label: str = None, # Label on the first line
    label2: str = None, # Optional label on the second line
    fontscale: float = 1.,
    thickness: int = 1, 
):
    im = im.copy()
    top_left = (int(center_x - width / 2), int(center_y - height / 2))
    bottom_right = (int(center_x + width / 2), int(center_y + height / 2))
    
    # Draw translucent fill
    overlay = np.zeros_like(im)
    cv2.rectangle(overlay, top_left, bottom_right, color, -1)
    im = cv2.addWeighted(im, 1.0, overlay, fillalpha, 1)
    
    # Draw opaque border
    cv2.rectangle(im, top_left, bottom_right, color, linewidth)
    
    # [Optional] Draw text
    if label is not None:
        label_color = choose_opposite_color(color)
        
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fontscale, thickness)[0]
        if label2 is not None:
            label2_size = cv2.getTextSize(label2, cv2.FONT_HERSHEY_SIMPLEX, fontscale, thickness)[0]
            label_w, label_h = max(label_size[0], label2_size[0]), label_size[1] + label2_size[1] + int(8 * fontscale)
        else:
            label_w, label_h = label_size
        
        cv2.rectangle(
            im, 
            (top_left[0], top_left[1]),
            (top_left[0] + label_w + thickness + int(8 * fontscale), 
             top_left[1] + label_h + thickness + int(8 * fontscale)), 
            color, thickness=-1)
        
        cv2.putText(
            im, label, 
            (top_left[0] + 1, top_left[1] + label_size[1] + 1),
            cv2.FONT_HERSHEY_DUPLEX, fontscale, label_color, thickness, cv2.LINE_AA)
        
        if label2 is not None:
            cv2.putText(
                im, label2, 
                (top_left[0] + 1, top_left[1] + label_h + int(4 * fontscale)),
                cv2.FONT_HERSHEY_DUPLEX, fontscale, label_color, thickness, cv2.LINE_AA)
    return im

def draw_bool_mask_on_im(
    im: np.ndarray, # np.uint8, 0~255, [H, W, 3]
    mask: np.ndarray, # bool, [Hm, Wm]
    color = (255, 0, 0), 
    alpha: float = 0.5, 
    h0: int = 0, w0: int = 0 # start of the patch in im
):
    color = np.array(color)[None, :]
    h, w, *_ = im.shape
    hm, wm, *_ = mask.shape
    if h0 >= h or w0 >= w:
        return im
    im = im.copy()
    
    # the new end of the patch in im
    h1, w1 = min(h0 + hm, h), min(w0 + wm, w)
    # the real start, end of the patch
    h_start, w_start = max(0, -h0), max(0, -w0)
    h_end, w_end = min(hm, h-h0), min(wm, w-w0)
    # the new start of the patch in im
    h0, w0 = max(0, h0), max(0, w0)
    
    mask = mask[h_start:h_end, w_start:w_end]
    
    im_sel = im[h0:h1, w0:w1]
    im_new = im_sel * (1 - alpha) + alpha * color
    im_new = np.where(mask, im_new, im_sel).astype(np.uint8)
    im[h0:h1, w0:w1] = im_new
    return im

def draw_int_mask_on_im(
    im: np.ndarray, # np.uint8, 0~255, [H, W, 3]
    mask: np.ndarray, # int, [Hm, Wm]
    cmap: np.ndarray, # np.uint8, 0~255, [N, 3]
    alpha: float = 1.0, 
    h0: int = 0, w0: int = 0 # start of the patch in im
):
    cmap = np.array(cmap)
    h, w, *_ = im.shape
    hm, wm, *_ = mask.shape
    if h0 >= h or w0 >= w:
        return im
    im = im.copy()

    # the new end of the patch in im
    h1, w1 = min(h0 + hm, h), min(w0 + wm, w)
    # the real start, end of the patch
    h_start, w_start = max(0, -h0), max(0, -w0)
    h_end, w_end = min(hm, h-h0), min(wm, w-w0)
    # the new start of the patch in im
    h0, w0 = max(0, h0), max(0, w0)

    mask = mask.reshape(hm, wm)[h_start:h_end, w_start:w_end]
    colored = cmap[mask]
    
    im_sel = im[h0:h1, w0:w1]
    im_new = im_sel * (1 - alpha) + alpha * colored
    im[h0:h1, w0:w1] = im_new
    return im

def draw_patch_on_im(
    im: np.ndarray, # np.uint8, 0~255, [H, W, 3]
    patch: np.ndarray, # np.uint8, 0~255, [Hm, Wm, 3]
    alpha: float = 1.0, 
    h0: int = 0, w0: int = 0 # start of the patch in im
):
    h, w, *_ = im.shape
    hm, wm, *_ = patch.shape
    if h0 >= h or w0 >= w:
        return im
    im = im.copy()

    # the new end of the patch in im
    h1, w1 = min(h0 + hm, h), min(w0 + wm, w)
    # the real start, end of the patch
    h_start, w_start = max(0, -h0), max(0, -w0)
    h_end, w_end = min(hm, h-h0), min(wm, w-w0)
    # the new start of the patch in im
    h0, w0 = max(0, h0), max(0, w0)

    patch = patch[h_start:h_end, w_start:w_end]
    
    im_sel = im[h0:h1, w0:w1]
    im_new = im_sel * (1 - alpha) + alpha * patch
    im[h0:h1, w0:w1] = im_new
    return im

if __name__ == "__main__":
    def unit_test():
        import matplotlib.pyplot as plt
        bgcolor = np.random.randint(255, size=[3,], dtype=np.uint8)
        fgcolor = np.random.randint(255, size=[3,], dtype=np.uint8)
        im = np.tile(bgcolor, [100, 200, 1])
        patch1 = np.tile(fgcolor, [50, 50, 1])
        patch2 = np.tile(fgcolor, [200, 400, 1])
        
        im1 = draw_patch_on_im(im, patch1, h0=-25, w0=0)
        im2 = draw_patch_on_im(im, patch1, h0=80, w0=0)
        im3 = draw_patch_on_im(im, patch1, h0=75, w0=75)
        im4 = draw_patch_on_im(im, patch2, h0=75, w0=75)
        im5 = draw_patch_on_im(im, patch2, h0=-175, w0=-375)
        plt.subplot(1, 5, 1)
        plt.imshow(im1)
        plt.subplot(1, 5, 2)
        plt.imshow(im2)
        plt.subplot(1, 5, 3)
        plt.imshow(im3)
        plt.subplot(1, 5, 4)
        plt.imshow(im4)
        plt.subplot(1, 5, 5)
        plt.imshow(im5)
        plt.show()
    unit_test()
        