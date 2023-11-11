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


def read_img(path, img_size, K, return_raw_size=False):  # should merge this with the modify K function

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


    img_normalized =  normalize_img(img[:, :, [2, 1, 0]].astype(np.float32))

    if return_raw_size:
        return img_normalized, (img_h, img_w)

    else:
        return img_normalized


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



def get_ray_dir(uv, K, E):

    x = (uv[:, 0] + 0.5 - K[0, 2]) / K[0, 0]
    y = (uv[:, 1] + 0.5 - K[1, 2]) / K[1, 1]
    z = torch.ones_like(x)
    dirs = torch.stack([x, y, z], axis=-1)

    dirs = dirs @ E[0:3, 0:3]  

    dirs = dirs / (torch.norm(dirs, dim=1)[:, None] + 1e-5)

    return dirs

def modify_K_resize(K, resize, img_raw_size):

    if img_raw_size[0] != img_raw_size[1]:  # argo case
 
        D = min(img_raw_size)

        if img_raw_size[0] > img_raw_size[1]:
            up_left = int(K[1, 2]-D//2)
            K[1, 2] = K[1, 2] - up_left
        else:
            up_left = int(K[0, 2]-D//2)
            K[0, 2] = K[0, 2] - up_left

    K = K.copy()
    K /= resize
    K[2, 2] = 1
    return K
