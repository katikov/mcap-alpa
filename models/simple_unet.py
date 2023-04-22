
import flax.linen as nn
import jax.numpy as jnp
from typing import Sequence, Tuple
import numpy as np
import alpa
import jax

class ConvBlock(nn.Module):
    in_channels: int
    out_channels: int
    dropout_rate: float = 0.5
    # TODO: padding for all modules
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if self.in_channels is None or self.out_channels is None:
            raise Exception("in_channels or out_channels is None!!")

        self.conv_1 = nn.Conv(self.out_channels, kernel_size=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)), dtype=self.dtype)
        self.dropout_1 = nn.Dropout(rate=self.dropout_rate)
        self.activation_1 = nn.activation.relu

        self.conv_2 = nn.Conv(self.out_channels, kernel_size=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)), dtype=self.dtype)
        self.dropout_2 = nn.Dropout(rate=self.dropout_rate)
        self.activation_2 = nn.activation.relu
      

    def __call__(self, x, train=True):
        x = self.conv_1(x)
        if train:
            x = self.dropout_1(x, deterministic=True)
        x = self.activation_1(x)
        # alpa.mark_pipeline_boundary()
        x = self.conv_2(x)
        if train:
            x = self.dropout_2(x, deterministic=True)
        x = self.activation_2(x)
        return x

class DownBlock(nn.Module):
    in_channels: int
    out_channels: int
    dropout_rate: float = 0.5
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.conv = ConvBlock(self.in_channels, self.out_channels, dropout_rate=self.dropout_rate)

    def __call__(self, x, train=True):
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = self.conv(x, train)
        return x


class UpconvBlock(nn.Module):
    in_channels: int
    out_channels: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if self.in_channels is None or self.out_channels is None:
            raise Exception("in_channels or out_channels is None!!")

        
        self.conv = nn.Conv(
            self.out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

        self.activation = nn.activation.relu

    def __call__(self, x):
        batch, height, width, channels = x.shape
        # print(x.shape)
        x = jax.image.resize(x, shape=(batch, height * 2, width * 2, channels), method="nearest")
        x = self.conv(x)
        x = self.activation(x)
        # print(x.shape)
        return x


class CropConcatBlock(nn.Module):
    def setup(self):
        pass
    
    def __call__(self, x, down_layer):
        x1_shape = down_layer.shape
        x2_shape = x.shape
        height_diff = (x1_shape[1] - x2_shape[1]) // 2
        width_diff = (x1_shape[2] - x2_shape[2]) // 2

        down_layer_cropped = down_layer[:,
                                        height_diff: (x2_shape[1] + height_diff),
                                        width_diff: (x2_shape[2] + width_diff),
                                        :]
        x = jnp.concatenate([down_layer_cropped, x], axis=-1)
        return x
        

        
class UpBlock(nn.Module):
    in_channels: int
    out_channels: int
    dropout_rate: float = 0.5
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.upconv = UpconvBlock(self.in_channels, self.out_channels)
        self.concat = CropConcatBlock()
        self.conv = ConvBlock(self.out_channels, self.out_channels, dropout_rate=self.dropout_rate)

    def __call__(self, x, down_layer, train=True):
        x = self.upconv(x)
        # print(x.shape, down_layer.shape)
        x = self.concat(x, down_layer)
        # alpa.mark_pipeline_boundary()
        x = self.conv(x, train)
        return x

class UNet(nn.Module):
    in_channels: int = 1
    out_channels: int = 2
    dropout_rate: float = 0.5
    layer_depth: int = 4
    hidden_channels: Tuple[int] = (32, 64, 128, 256)
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if len(self.hidden_channels) != self.layer_depth:
            raise Exception("size of hidden channels does not match the layer depth (layer_channels==len(hidden_channels))!!")
        # (conv + maxpooling)*layer_depth
        temp = [ConvBlock(in_channels=self.in_channels, out_channels=self.hidden_channels[0], dropout_rate=self.dropout_rate)]
        for i in range(self.layer_depth-1):
            conv = DownBlock(in_channels=self.hidden_channels[i], out_channels=self.hidden_channels[i+1], dropout_rate=self.dropout_rate)
            temp.append(conv)
        self.convs = temp


        # (upconv + crop + conv) * layer_depth
        temp = []
        for i in range(self.layer_depth-1):
            conv = UpBlock(in_channels=self.hidden_channels[i+1], out_channels=self.hidden_channels[i], dropout_rate=self.dropout_rate)
            temp.append(conv)
        temp.reverse()
        self.upconvs = temp

        # outconv(single conv) + relu + softmax
        self.outconv = nn.Conv(self.out_channels, kernel_size=(1, 1), strides=(1, 1), dtype=self.dtype)
        self.out_activation = nn.activation.relu
        self.out = nn.activation.softmax

    def __call__(self, x, train=True):
        hiddens = []
        for conv in self.convs:
            x = conv(x, train)
            hiddens.append(x)
            alpa.mark_pipeline_boundary()

        # alpa.mark_pipeline_boundary()
        hiddens = hiddens[:-1]
        hiddens.reverse()

        for down_layer, conv in zip(hiddens, self.upconvs):
            # print(x.shape, down_layer.shape)
            x = conv(x, down_layer, train)
            alpa.mark_pipeline_boundary()

        x = self.outconv(x)
        x = self.out_activation(x)
        x = self.out(x)

        return x
