
import flax.linen as nn
import jax.numpy as jnp
from typing import Sequence, Any
import itertools
import numpy as np

from .swin_utils import compute_mask
from .swin_transformer_block import SwinTransformerBlock


class PatchMerging(nn.Module):
    input_channels: int
    output_channels: int
    dims: int = 2
    norm_layer: type = nn.LayerNorm
    
    # TODO: implement merging by convolution
    def setup(self):
        self.reduction = nn.Dense(self.output_channels, use_bias=False)
        # patch_size = (2,) * self.dims
        # self.conv = nn.Conv(self.output_channels, kernel_size=patch_size, strides=patch_size, use_bias=False)
        self.norm = self.norm_layer()
            

    def __call__(self, x):
        # x = self.conv(x)
        # x = self.norm(x)

        # b, h, w, c = x.shape
        # if (h % 2 == 1) or (w % 2 == 1):
        #     x = jnp.pad(x, ((0, 0), (0, w%2), (0, h%2), (0, 0)) )
        # b, h, w, c = x.shape
        # x = x.reshape(b, h//2, 2, w//2, 2, c).transpose((0, 1, 3, 2, 4, 5)).reshape(b, h//2, w//2, c*4)

        # x = jnp.concatenate([x[:, 0::2, 0::2, :], x[:, 0::2, 1::2, :], x[:, 1::2, 0::2, :], x[:, 1::2, 1::2, :]], axis=-1)
        if self.dims == 3:
            b, d, h, w, c = x.shape
            pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1)
            if pad_input:
                x = jnp.pad(x, ((0, 0), (0, w % 2), (0, h % 2), (0, d % 2), (0, 0)) )
            x = jnp.concatenate(
                [x[:, i::2, j::2, k::2, :] for i, j, k in itertools.product(range(2), range(2), range(2))], axis=-1
            )
        elif self.dims == 2:
            b, h, w, c = x.shape
            if (h % 2 == 1) or (w % 2 == 1):
                x = jnp.pad(x, ((0, 0), (0, w%2), (0, h%2), (0, 0)) )

            x = jnp.concatenate([x[:, i::2, j::2, :] for i, j in itertools.product(range(2), range(2))], axis=-1)
        
        # print("---:", x.shape)
        x = self.norm(x)
        x = self.reduction(x)
        return x



class SwinTransformerStage(nn.Module):
        input_channels: int
        output_channels: int
        num_layers: int
        num_heads: int
        window_size: Sequence[int]
        dropout_path: Sequence[float]
        mlp_ratio: float = 4.0
        qkv_bias: bool = True
        dropout_rate: float = 0.0
        attn_dropout_rate: float = 0.0
        norm_layer: type = nn.LayerNorm


        def setup(self):
            self.shift_size = tuple(i // 2 for i in self.window_size)
            self.no_shift = tuple(0 for i in self.window_size)
            self.dims = len(self.window_size)

            self.blocks = [SwinTransformerBlock(
                            output_channels=self.input_channels,
                            num_heads=self.num_heads,
                            window_size=self.window_size,
                            shift_size=self.no_shift if (i % 2 == 0) else self.shift_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=self.qkv_bias,
                            dropout_rate=self.dropout_rate,
                            attn_dropout_rate=self.attn_dropout_rate,
                            dropout_path=self.dropout_path[i],
                            norm_layer=self.norm_layer
                    ) for i in range(self.num_layers)
                ]



            self.downsample = PatchMerging(input_channels=self.input_channels, 
                                        output_channels=self.output_channels,
                                        dims=self.dims,
                                        norm_layer=self.norm_layer)

            # from .simple_unet import DownBlock
            # self.downsample = DownBlock(in_channels=self.input_channels, out_channels=self.output_channels, dropout_rate=0.0)


            attn_mask = compute_mask(self.window_size, self.shift_size).reshape(
                                1,1,np.prod(self.window_size), np.prod(self.window_size))

            self.attn_mask = jnp.array(attn_mask)


        def __call__(self, x, train=True):
            for blk in self.blocks:
                x = blk(x, self.attn_mask)
            x = self.downsample(x)
            # print("-:", x.shape)
            return x


