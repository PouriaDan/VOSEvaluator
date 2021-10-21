import numpy as np
import matplotlib.cm as cm
from PIL import Image


def open_image(img_path):
    img = Image.open(img_path)
    img = np.array(img)
    return img

def mask_to_seg(mask, num_objs=None):
    cmap = cm.get_cmap('jet')
    num_objs = num_objs if num_objs is not None else mask.max()
    seg = np.zeros([*mask.shape,3])
    for o in range(1, num_objs+1):
        seg[mask==o] = np.array(cmap(o/num_objs))[:3]
    return seg

def merge_images(im1, w1, im2, w2):
    im1 = im1/im1.max()
    im2 = im2/im2.max()
    return im1*w1+im2*w2