"""Helpers from See-through's common/utils (torch_utils.py, cv.py), only those the layer model uses."""

import random
from typing import Union

import cv2
import numpy as np
import torch
from einops import rearrange
from PIL import Image
from torch import Tensor
import torchvision.transforms.functional as tv_functional
from torchvision.transforms.functional import pil_to_tensor


def tensor2img(t: torch.Tensor, output_type='numpy', denormalize=False, mean = 0., std = 255., from_mode: str = 'RGB', convert_mode: str = None, src_dim_order: str = 'chw', dtype=np.uint8, clip=(0, 255)) -> Union[Image.Image, np.array]:
    
    def _check_denormalize_params(values, num_channels):
        if isinstance(values, (int, float, np.ScalarType)):
            return values
        else:
            if isinstance(values, list):
                values = np.array(values)
            elif isinstance(values, torch.Tensor):
                values = values.to(device='cpu', dtype=torch.float32).numpy()
            else:
                raise Exception(f'invalid normalizing values: {values}')

            if len(values) > num_channels:
                values = values[:num_channels]
            assert len(values) == num_channels
            values = values.reshape((1, 1, -1))
            return values

    t = t.detach().to(device='cpu', dtype=torch.float32).squeeze().numpy()

    if t.ndim == 3:
        if src_dim_order == 'chw':
            t = rearrange(t, 'c h w -> h w c')
        c = t.shape[-1]
    else:
        assert t.ndim == 2, "t.ndim should be 2 or 3 after squeeze"
        c = 1

    if denormalize:
        t = (t * _check_denormalize_params(std, c)) + _check_denormalize_params(mean, c)

    if clip is not None:
        t = np.clip(t, clip[0], clip[1])
    image = t.astype(dtype)

    if output_type == 'pil':
        if len(image.shape) == 2:
            from_mode = 'L'
        image = Image.fromarray(image, mode=from_mode)
        if convert_mode is not None:
            image = image.convert(convert_mode)
    else:
        image = np.ascontiguousarray(image)
        assert output_type == 'numpy'
    return image


_IMG2TENSOR_IMGTYPE = (Image.Image, np.ndarray, str)


_IMG2TENSOR_DIMORDER = ('bchw', 'chw', 'hwc')


def img2tensor(img: Union[Image.Image, np.ndarray, str, torch.Tensor], normalize = False, mean = 0., std = 255., dim_order: str = 'bchw', dtype=torch.float32, device: str = 'cpu', imread_mode='RGB') -> Tensor:

    def _check_normalize_values(values, num_channels):
        if isinstance(values, tuple):
            values = list(values)
        elif isinstance(values, (int, float, np.ScalarType)):
            values = [values] * num_channels
        else:
            assert isinstance(values, (np.ndarray, list))
        if len(values) > num_channels:
            values = values[:num_channels]
        assert len(values) == num_channels
        return values

    assert isinstance(img, _IMG2TENSOR_IMGTYPE)
    assert dim_order in _IMG2TENSOR_DIMORDER

    if isinstance(img, str):
        img = load_image(img, mode=imread_mode)

    if isinstance(img, Image.Image):
        img = pil_to_tensor(img)
        if dim_order == 'bchw':
            img = img.unsqueeze(0)
        elif dim_order == 'hwc':
            img = img.permute((1, 2, 0))
    else:
        if img.ndim == 2:
            img = img[..., None]
        else:
            assert img.ndim == 3
        if dim_order == 'bchw':
            img = rearrange(img, 'h w c -> c h w')[None, ...]
        elif dim_order == 'chw':
            img = rearrange(img, 'h w c -> c h w')
        img = torch.from_numpy(np.ascontiguousarray(img))


    img = img.to(device=device, dtype=dtype)

    if normalize:

        if dim_order == 'bchw':
            c = img.shape[1]
        elif dim_order == 'chw':
            c = img.shape[0]
        else:
            c = img.shape[2]

        if mean is not None and std is not None:
            mean = _check_normalize_values(mean, c)
            std = _check_normalize_values(std, c)
            img = tv_functional.normalize(img, mean=mean, std=std)

    return img


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def smart_resize(src: np.ndarray, target_size, upscale_interpolation=cv2.INTER_LINEAR, downscale_interpolation=cv2.INTER_AREA):
    h, w = src.shape[:2]
    th, tw = target_size
    if th == h and tw == w:
        return src.copy()
    if th * tw < h * w:
        interpolation = downscale_interpolation
    else:
        interpolation = upscale_interpolation
    return cv2.resize(src, (tw, th), interpolation=interpolation)


def center_square_pad_resize(img: np.ndarray, target_size, pad_value=0, upscale_interpolation=cv2.INTER_LINEAR, downscale_interpolation=cv2.INTER_AREA, return_pad_info=False):
    h, w = img.shape[:2]
    pad_size = (w, h)
    pad_pos = (0, 0)
    if h != w:
        sz = max(h, w)
        px1 = (sz - w) // 2
        py1 = (sz - h) // 2
        shape = (sz, sz) if img.ndim == 2 else (sz, sz, img.shape[-1])
        padded = np.full(shape, pad_value, dtype=img.dtype)
        padded[py1: py1 + h, px1: px1 + w] = img
        h, w = padded.shape[:2]
        img = padded
        pad_size = (w, h)
        pad_pos = (px1, py1)
    if h != target_size or w != target_size:
        img = smart_resize(img, (target_size, target_size), upscale_interpolation=upscale_interpolation, downscale_interpolation=downscale_interpolation)
    if return_pad_info:
        return img, pad_size, pad_pos
    else:
        return img


def build_alpha_pyramid(color, alpha, dk=1.2):
    # Written by lvmin at Stanford
    # Massive iterative Gaussian filters are mathematically consistent to pyramid.

    pyramid = []
    current_premultiplied_color = color * alpha
    current_alpha = alpha

    while True:
        pyramid.append((current_premultiplied_color, current_alpha))

        H, W, C = current_alpha.shape
        if min(H, W) == 1:
            break

        current_premultiplied_color = cv2.resize(current_premultiplied_color, (int(W / dk), int(H / dk)), interpolation=cv2.INTER_AREA)
        current_alpha = cv2.resize(current_alpha, (int(W / dk), int(H / dk)), interpolation=cv2.INTER_AREA)[:, :, None]
    return pyramid[::-1]


def pad_rgb(np_rgba_hwc_uint8, return_format='rgb', to_uint8=False, keep_ori_pixel=True):
    # Written by lvmin at Stanford
    # Massive iterative Gaussian filters are mathematically consistent to pyramid.


    np_rgba_hwc = np_rgba_hwc_uint8.astype(np.float32) / 255.0

    if keep_ori_pixel:
        ori_rgb = np_rgba_hwc[..., :3].copy()
        ori_alpha = np_rgba_hwc[..., [-1]].copy()

    pyramid = build_alpha_pyramid(color=np_rgba_hwc[..., :3], alpha=np_rgba_hwc[..., 3:])

    top_c, top_a = pyramid[0]
    fg = np.sum(top_c, axis=(0, 1), keepdims=True) / np.sum(top_a, axis=(0, 1), keepdims=True).clip(1e-8, 1e32)

    for layer_c, layer_a in pyramid:
        layer_h, layer_w, _ = layer_c.shape
        fg = cv2.resize(fg, (layer_w, layer_h), interpolation=cv2.INTER_LINEAR)
        fg = layer_c + fg * (1.0 - layer_a)

    if keep_ori_pixel:
        fg = np.clip(ori_alpha * ori_rgb + (1-ori_alpha) * fg, 0, 1)

    if return_format == 'argb':
        fg = np.concatenate([np_rgba_hwc[..., 3:], fg], axis=2)

    if to_uint8:
        fg = (fg * 255).astype(np.uint8)

    return fg


def checkerboard(shape):
    return np.indices(shape).sum(axis=0) % 2
