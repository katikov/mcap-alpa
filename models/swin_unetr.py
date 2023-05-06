
import flax.linen as nn
import jax.numpy as jnp
from typing import Sequence
import numpy as np
import alpa
from .swin_transformer_stage import SwinTransformerStage
from .basic_blocks import UnetrBasicBlock, UnetrUpBlock

"""
code from MONAI Swin UnetR:
https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/swin_unetr.py
TODO: checkpointing
TODO: support 3d images
TODO: set dtype
"""


class PatchEmbed(nn.Module):
    patch_size: Sequence[int] = (2, 2)
    in_channels: int = 1
    out_channels: int = 48
    dropout_rate:float = 0.0

    def setup(self):
        # TODO: padding
        self.conv = nn.Conv(self.out_channels, kernel_size=self.patch_size, strides=self.patch_size)
        self.pos_drop = nn.Dropout(self.dropout_rate, deterministic=True)
        

    def __call__(self, x, train=True):
        x = self.conv(x)
        if train:
            x = self.pos_drop(x)
        return x






class SwinUNETR(nn.Module):
    img_size: Sequence[int] = (512, 512)
    in_channels: int = 1
    out_channels: int = 1
    num_layers: Sequence[int] = (2, 2, 2, 2)
    num_heads: Sequence[int] = (3, 6, 12, 24)
    patch_size: Sequence[int] = (2, 2)
    window_size: Sequence[int] = (7, 7)
    feature_size: int = 48  
    use_v2: bool = False
    mlp_ratio: float = 4.0
    dtype: jnp.dtype = jnp.float32
    qkv_bias: bool = True

    dropout_rate: float = 0.0 
    attn_dropout_rate: float = 0.0 
    dropout_path_rate: float = 0.0 
    normalize: bool = True
    norm_layer:type = nn.LayerNorm


    use_checkpoint: bool = False # TODO: 
    downsample="merging" # TODO:
        
    

    def setup(self):
        self.dims = len(self.img_size)
        if self.dims != 2:
            raise Exception("dim should be 2 ! (dim=3 not yet implemented)")
        if len(self.window_size) != self.dims or len(self.patch_size) != self.dims:
            raise Exception("window_size and/or patch_size do not match the image dimensions!")
        if self.feature_size % 12 != 0:
            raise Exception("feature_size should be divisible by 12.")

        self.num_stages = len(self.num_layers)
        
        # embed from input, (output as vit0)
        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            out_channels=self.feature_size,
            dropout_rate = self.dropout_rate
        )

        self.x0_norm = self.norm_layer(epsilon=1e-5)
        
        temp = [[] for i in range(self.num_stages)]

        dropout_rates_path = [x.item() for x in np.linspace(0, self.dropout_path_rate, sum(self.num_layers))]
        # vit transformer, input -> vit1-4
        for i in range(self.num_stages):
            input_channels = int(self.feature_size * 2**i)
            output_channels = input_channels * 2

            if self.use_v2:
                layerc = UnetrBasicBlock(
                        input_channels=input_channels,
                        output_channels=input_channels,
                        dims=2
                    )
                temp[i].append(layerc)


            layer = SwinTransformerStage(
                input_channels=input_channels,
                output_channels=output_channels,
                num_layers=self.num_layers[i],
                num_heads=self.num_heads[i],
                window_size=self.window_size,
                dropout_path=dropout_rates_path[sum(self.num_layers[:i]) : sum(self.num_layers[: i + 1])],
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                dropout_rate=self.dropout_rate,
                attn_dropout_rate=self.attn_dropout_rate,
                norm_layer=self.norm_layer
            )

            # from .simple_unet import DownBlock
            # layer = DownBlock(in_channels=input_channels, out_channels=output_channels, dropout_rate=self.dropout_rate)


            temp[i].append(layer)


        self.vit_layers = temp
        self.norms = [self.norm_layer() for i in range(self.num_stages)]

        # encoders: fisrt, 0-2, last
        kernel_size = (3 for i in range(self.dims))
        self.encoder_first = UnetrBasicBlock(
            input_channels=self.in_channels,
            output_channels=self.feature_size,
            kernel_size=kernel_size,
            dims=self.dims,
            stride=1,
            res_block=True,
        )

        self.encoders = [UnetrBasicBlock(
            input_channels=(2**i) * self.feature_size,
            output_channels=(2**i) * self.feature_size,
            kernel_size=kernel_size,
            dims=self.dims,
            stride=1,
            res_block=True,
            ) for i in range(self.num_stages-1)]
        

        self.encoder_last = UnetrBasicBlock(
            input_channels=(2**self.num_stages) * self.feature_size,
            output_channels=(2**self.num_stages) * self.feature_size,
            kernel_size=kernel_size,
            dims=self.dims,
            stride=1,
            res_block=True,
        )

        # decoders: 0-3, last
        self.decoders = [ UnetrUpBlock(input_channels=(2**(i+1))* self.feature_size,
                                       output_channels=(2**i) * self.feature_size,
                                       dims = 2,
                                       kernel_size = (3,3),
                                       upsample_kernel_size = (2, 2),
                                       res_block = True)
            for i in range(self.num_stages)
        ]

        self.decoder_last = UnetrUpBlock(input_channels=self.feature_size,
                                         output_channels=self.feature_size,
                                         dims = 2,
                                         kernel_size = (3,3),
                                         upsample_kernel_size = (2, 2),
                                         res_block = True)
            



        # out conv: reduce to k (output type) channels
        self.out = nn.Conv(self.out_channels, kernel_size=(1,1), strides=(1,1))




    def __call__(self, x, train=True):
        vit_out = []
        x_in = x

        x = self.patch_embed(x, train)
        x0_out = self.x0_norm(x) if self.normalize else x
        vit_out.append(x0_out)
        alpa.mark_pipeline_boundary()

        for i in range(self.num_stages):
            for layer in self.vit_layers[i]:
                x = layer(x, train)
            x_out = self.norms[i](x)
            # alpa.mark_pipeline_boundary()
            vit_out.append(x_out)

            

        dec = self.encoder_last(vit_out[self.num_stages])
        dec = self.decoders[self.num_stages-1](dec, vit_out[self.num_stages-1])
        alpa.mark_pipeline_boundary()
        for i in range(self.num_stages-2, -1, -1):
            enc = self.encoders[i](vit_out[i])
            dec = self.decoders[i](dec, enc)
            alpa.mark_pipeline_boundary()

        enc = self.encoder_first(x_in)
        dec = self.decoder_last(dec, enc)

        logits = self.out(dec)
        return logits

        