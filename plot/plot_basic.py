"""
@file   plot_basic.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Basic plotting utilities.
"""

import colorsys
import itertools
import numpy as np
from numbers import Number
from fractions import Fraction
from typing import Iterable, List, Literal, Tuple, Union

from matplotlib import cm

from nr3d_lib.fmt import log

white = (1.0, 1.0, 1.0)
black = (0.0, 0.0, 0.0)
dark_gray = (0.25, 0.25, 0.25)
light_purple = (0.788, 0.580, 1.0)
lime = (0.746, 1.0, 0.0)
red = (1.0, 0.0, 0.0)
green = (0.0, 1.0, 0.0)
blue = (0.0, 0.0, 1.0)
orange = (1.0, 0.5, 0.0)
light_cyan = (0.796, 1.0, 1.0)
light_pink = (1.0, 0.796, 1.0)
light_yellow = (1.0, 1.0, 0.796)
light_teal = (0.757, 1.0, 0.949)
gray = (0.5, 0.5, 0.5)
soft_blue = (0.721, 0.90, 1.0)
soft_red = (1.0, 0.0, 0.085)
lime_green = (0.519, 0.819, 0.0)
purple = (0.667, 0.0, 0.429)
gold = (1.0, 0.804, 0.0)

#---------------------------------------------
#--------     Color depth map     ------------
#---------------------------------------------
# def color_depth(depths, scale=None):
#     """
#     Color an input depth map.

#     Arguments:
#         depths -- HxW numpy array of depths
#         [scale=None] -- scaling the values (defaults to the maximum depth)

#     Returns:
#         colored_depths -- HxWx3 numpy array visualizing the depths
#     """

#     _color_map_depths = np.array([
#         [0, 0, 0],  # 0.000
#         [0, 0, 255],  # 0.114
#         [255, 0, 0],  # 0.299
#         [255, 0, 255],  # 0.413
#         [0, 255, 0],  # 0.587
#         [0, 255, 255],  # 0.701
#         [255, 255, 0],  # 0.886
#         [255, 255, 255],  # 1.000
#         [255, 255, 255],  # 1.000
#     ]).astype(float)
#     _color_map_bincenters = np.array([
#         0.0,
#         0.114,
#         0.299,
#         0.413,
#         0.587,
#         0.701,
#         0.886,
#         1.000,
#         2.000,  # doesn't make a difference, just strictly higher than 1
#     ])

#     if scale is None:
#         scale = depths.max()

#     values = np.clip(depths.flatten() / scale, 0, 1)
#     # for each value, figure out where they fit in in the bincenters: what is the last bincenter smaller than this value?
#     lower_bin = ((values.reshape(-1, 1) >=
#                   _color_map_bincenters.reshape(1, -1)) * np.arange(0, 9)).max(axis=1)
#     lower_bin_value = _color_map_bincenters[lower_bin]
#     higher_bin_value = _color_map_bincenters[lower_bin + 1]
#     alphas = (values - lower_bin_value) / (higher_bin_value - lower_bin_value)
#     colors = _color_map_depths[lower_bin] * (1 - alphas).reshape(-1, 1) + _color_map_depths[
#         lower_bin + 1] * alphas.reshape(-1, 1)
#     return colors.reshape(depths.shape[0], depths.shape[1], 3).astype(np.uint8)

def color_depth(depths: np.ndarray, scale=None, cmap='viridis', out: Literal['uint8,0,255', 'float,0,1']='uint8,0,255'):
    if scale is None:
        scale = depths.max()+1e-10
    colors = cm.get_cmap(cmap)(depths/scale)[...,:3]
    if out == 'uint8,0,255':
        return  np.clip(((colors)*255.).astype(np.uint8), 0, 255)
    elif out == 'float,0,1':
        return colors
    else:
        raise RuntimeError(f"Invalid out={out}")

#---------------------------------------------
#--------     Color ind map     ------------
#---------------------------------------------
def zenos_dichotomy() -> Iterable[Fraction]:
    """
    http://en.wikipedia.org/wiki/1/2_%2B_1/4_%2B_1/8_%2B_1/16_%2B_%C2%B7_%C2%B7_%C2%B7
    """
    for k in itertools.count():
        yield Fraction(1,2**k)

def fracs() -> Iterable[Fraction]:
    """
    [Fraction(0, 1), Fraction(1, 2), Fraction(1, 4), Fraction(3, 4), Fraction(1, 8), Fraction(3, 8), Fraction(5, 8), Fraction(7, 8), Fraction(1, 16), Fraction(3, 16), ...]
    [0.0, 0.5, 0.25, 0.75, 0.125, 0.375, 0.625, 0.875, 0.0625, 0.1875, ...]
    """
    yield Fraction(0)
    for k in zenos_dichotomy():
        i = k.denominator # [1,2,4,8,16,...]
        for j in range(1,i,2):
            yield Fraction(j,i)

# can be used for the v in hsv to map linear values 0..1 to something that looks equidistant
# bias = lambda x: (math.sqrt(x/3)/Fraction(2,3)+Fraction(1,3))/Fraction(6,5)

HSVTuple = Tuple[Fraction, Fraction, Fraction]
RGBTuple = Tuple[float, float, float]

def hue_to_tones(h: Fraction) -> Iterable[HSVTuple]:
    for s in [Fraction(6,10)]: # optionally use range
        for v in [Fraction(8,10),Fraction(5,10)]: # could use range too
            yield (h, s, v) # use bias for v here if you use range

def hsv_to_rgb(x: HSVTuple) -> RGBTuple:
    return colorsys.hsv_to_rgb(*map(float, x))

flatten = itertools.chain.from_iterable

def hsvs() -> Iterable[HSVTuple]:
    return flatten(map(hue_to_tones, fracs()))

def rgbs() -> Iterable[RGBTuple]:
    return map(hsv_to_rgb, hsvs())

def rgb_to_css(x: RGBTuple) -> str:
    uint8tuple = map(lambda y: int(y*255), x)
    return [*uint8tuple]

def css_colors() -> Iterable[str]:
    return map(rgb_to_css, rgbs())

def get_n_ind_colors(num):
    return list(itertools.islice(css_colors(), num))

def get_n_ind_pallete(num: int, palette: str="bright"):
    import seaborn as sns
    return sns.color_palette(palette, num)

def gallery(array, ncols=None, nrows=None):
    if isinstance(array, list):
        array = np.array(array)
    if ncols is None and nrows is None:
        ncols = 3
    nindex, height, width, intensity = array.shape
    if nrows is None:
        nrows = nindex//ncols
    else:
        ncols = nindex//nrows
    # assert nindex == nrows*ncols
    if nindex > nrows*ncols:
        nrows += 1
        array = np.concatenate([array, np.zeros([nrows*ncols-nindex, height, width, intensity], dtype=array.dtype)])
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols, intensity))
    return result

def figure_to_image(figures, close=True):
    # Modified from tensorboardX   https://github.com/lanpa/tensorboardX
    """Render matplotlib figure to numpy format.

    Note that this requires the ``matplotlib`` package.

    Args:
        figure (matplotlib.pyplot.figure) or list of figures: figure or a list of figures
        close (bool): Flag to automatically close the figure

    Returns:
        numpy.array: image in [CHW] order
    """
    import numpy as np
    try:
        import matplotlib.pyplot as plt
        import matplotlib.backends.backend_agg as plt_backend_agg
    except ModuleNotFoundError:
        print('please install matplotlib')

    def render_to_rgb(figure):
        canvas = plt_backend_agg.FigureCanvasAgg(figure)
        canvas.draw()
        data = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        w, h = figure.canvas.get_width_height()
        image_hwc = data.reshape([h, w, 4])[:, :, 0:3]
        # image_chw = np.moveaxis(image_hwc, source=2, destination=0)
        if close:
            plt.close(figure)
        return image_hwc

    if isinstance(figures, list):
        images = [render_to_rgb(figure) for figure in figures]
        return np.stack(images)
    else:
        image = render_to_rgb(figures)
        return image

def choose_opposite_color(bg_color: Tuple[int, int, int]):
    # Calculate brightness and choose the opposite color based on the background color
    brightness = (bg_color[0] * 299 + bg_color[1] * 587 + bg_color[2] * 114) / 1000
    if brightness > 127:
        return (0, 0, 0)
    else:
        return (255, 255, 255)