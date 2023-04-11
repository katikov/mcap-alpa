import numpy as np
import flax.linen as nn
import jax.numpy as jnp
from typing import Sequence

def window_partition(x, window_size):
    x
    if len(x.shape) == 5:
        b, d, h, w, c = x.shape
        x = x.reshape((
            b,
            d // window_size[0],
            window_size[0],
            h // window_size[1],
            window_size[1],
            w // window_size[2],
            window_size[2],
            c)
        )
        windows = (
            x.transpose(0, 1, 3, 5, 2, 4, 6, 7).reshape((-1, window_size[0] * window_size[1] * window_size[2], c))
        )
    elif len(x.shape) == 4:
        b, h, w, c = x.shape
        x = x.reshape((b, h // window_size[0], window_size[0], w // window_size[1], window_size[1], c))
        windows = x.transpose(0, 1, 3, 2, 4, 5).reshape((-1, window_size[0] * window_size[1], c))
    return windows


def window_reverse(windows, window_size, dims):
    if len(dims) == 4:
        b, d, h, w = dims
        x = windows.reshape((
            b,
            d // window_size[0],
            h // window_size[1],
            w // window_size[2],
            window_size[0],
            window_size[1],
            window_size[2],
            -1)
        )
        x = x.transpose(0, 1, 4, 2, 5, 3, 6, 7).reshape((b, d, h, w, -1))

    elif len(dims) == 3:
        b, h, w = dims
        x = windows.reshape((b, h // window_size[0], w // window_size[1], window_size[0], window_size[1], -1))
        x = x.transpose(0, 1, 3, 2, 4, 5).reshape((b, h, w, -1))
    return x


def compute_mask(window_size, shift_size):
    cnt = 0
    img_mask = np.zeros((window_size))
    if len(window_size) == 3:
        for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
            for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
                for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                    img_mask[d, h, w] = cnt
                    cnt += 1

    elif len(window_size) == 2:
        for h in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
            for w in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
                img_mask[h, w] = cnt
                cnt += 1

    # print(img_mask)
    mask_windows = img_mask.reshape(-1)
    
    attn_mask = np.expand_dims(mask_windows, 0) - np.expand_dims(mask_windows, 1)# mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask[attn_mask != 0] = float(-100.0)
    attn_mask[attn_mask == 0] = float(0.0)

    return attn_mask



if __name__ == "__main__":
    a = compute_mask((5,5), (2,2))
    print(a, a.shape)
