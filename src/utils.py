import torch
import cv2
import matplotlib
import numpy as np


def normalize_img(img):  # normalize  -1-1
    return (img/255-0.5)*2


def denormalize_img(img):

    out = torch.zeros_like(img)
    out[img != 0] = (img[img != 0]/2+0.5)*255

    return out


def read_img(path, img_size, K):  # should merge this with the modify K function

    img = cv2.imread(path)

    img_h, img_w, _ = img.shape

    c_w = K[0, 2]
    c_h = K[1, 2]

    if img_h != img_w:
        D = min(img_h, img_w)
        if img_h > img_w:
            up_left = int(c_h-D//2)
            img = img[up_left:up_left + D, 0:D]
        else:
            up_left = int(c_w-D//2)
            img = img[0:D, up_left:up_left + D]

    if img_size != img.shape[0]:
        img = cv2.resize(img, (img_size, img_size))

    return normalize_img(img[:, :, [2, 1, 0]].astype(np.float32))


def depth_inv_to_color(depth_inv):
    color_map = matplotlib.colormaps['viridis']
    depth_color = color_map(depth_inv*3)[:, :, 0:3]
    depth_color[depth_inv == 0] = 0

    return depth_color


def positional_encoding(positions, freqs, ori=False):
    # from PointNeRF codebase
    '''encode positions with positional encoding
        positions: :math:`(...,D)`
        freqs: int
    Return:
        pts: :math:`(..., 2DF)`
    '''
    freq_bands = (2**torch.arange(freqs).float()).to(positions.device)  # (F,)
    ori_c = positions.shape[-1]
    pts = (positions[..., None] * freq_bands).reshape(positions.shape[:-1] +
                                                      (freqs * positions.shape[-1], ))  # (..., DF)
    if ori:
        pts = torch.cat([positions, torch.sin(pts), torch.cos(pts)], dim=-
                        1).reshape(pts.shape[:-1]+(pts.shape[-1]*2+ori_c,))
    else:
        pts = torch.stack([torch.sin(pts), torch.cos(pts)], dim=-1).reshape(pts.shape[:-1]+(pts.shape[-1]*2,))
    return pts


def clip_values(x):
    x[x > 1] = 1
    x[x < -1] = -1
    return x
