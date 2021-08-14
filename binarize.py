import numpy as np
import scipy as sp
from PIL import Image

assert '1.21' in np.__version__ 

def create_sliding_window(image, shape=(3, 3)):
	## Inspired from this stackoverflow thread: https://stackoverflow.com/a/48216770
    r_extra = np.floor(shape[0] / 2).astype(int)
    c_extra = np.floor(shape[1] / 2).astype(int)
    out = np.empty((image.shape[0] + 2 * r_extra, image.shape[1] + 2 * c_extra))
    out[:] = np.nan
    out[r_extra:-r_extra, c_extra:-c_extra] = image
    out = np.lib.stride_tricks.sliding_window_view(out, shape)
    return out


def binarize_textual_components(image, window_size=(5,5), k=0.2, R=128):
 	
    # window size should always be odd
    assert window_size[0] % 2 == 1, "window size should always be odd for pixel to be at the center"
    
    # create a view of a sliding window of shape (window_size, window_size) over every pixel of the image
    view = create_sliding_window(image, shape=window_size)
    
    # ignore nans while computing mean and std at local window level
    local_mean = np.nanmean(view, axis=(2, 3))
    local_std = np.nanstd(view, axis=(2, 3))
    
    pixel_threshold = local_mean * (1 + k * ((local_std/R) - 1))
    
    binarized_image = np.where(image >= pixel_threshold, 0, 255)
    return binarized_image