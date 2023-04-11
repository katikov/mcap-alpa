
import flax.linen as nn
import jax.numpy as jnp
from typing import Sequence, Union
import jax
import numpy as np

class UnetrUpBlock(nn.Module):
    input_channels: int
    output_channels: int
    dims: int = 2
    kernel_size: Sequence[int] = (3, 3)
    upsample_kernel_size: Sequence[int] = (2, 2)
    res_block: bool = False

    def setup(self):
        # if dims != len(kernel_size):
        self.conv1 = nn.Conv(
            self.output_channels,
            kernel_size=(1, 1),
            strides=(1, 1)
        )
        if self.res_block:
            self.conv2 = UnetResBlock(
                input_channels = self.output_channels * 2,
                output_channels = self.output_channels,
                dims = 2,
                kernel_size = self.kernel_size,
                stride = (1, 1)
            )
        else:
            self.conv2 = UnetBasicBlock(
                input_channels = self.output_channels * 2,
                output_channels = self.output_channels,
                dims = 2,
                kernel_size = self.kernel_size,
                stride = (1, 1)
            )
        

    def __call__(self, x, skip):
        batch, height, width, channels = x.shape
        sh, sw = self.upsample_kernel_size
        x = jax.image.resize(x, shape=(batch, height * sh, width * sw, channels), method="nearest")
        x = self.conv1(x)
        x = jnp.concatenate([x, skip], axis=-1)
        x = self.conv2(x)
        return x








class UnetResBlock(nn.Module):
    input_channels: int
    output_channels: int
    dims: int = 2
    kernel_size: Union[Sequence[int], int] = (3, 3)
    stride: Union[Sequence[int], int] = (1, 1)
    dropout: Union[float, None] = None
    activation: type = nn.activation.leaky_relu

    def setup(self):
        if self.dims == 2:
            axes = (-2, -1)
        elif self.dims == 3:
            axes = (-3, -2, -1)
        self.downsample = self.input_channels != self.output_channels
        stride_np = np.atleast_1d(self.stride)
        if not np.all(stride_np == 1):
            self.downsample = True

        self.conv1 = nn.Conv(self.output_channels, self.kernel_size, strides=self.stride)
        if self.dropout:
            self.drop1 = nn.Dropout(self.dropout, deterministic=True)
        self.norm1 = nn.LayerNorm(reduction_axes=axes)

        self.lrelu = self.activation

        self.conv2 = nn.Conv(self.output_channels, self.kernel_size, strides=1)
        if self.dropout:
            self.drop2 = nn.Dropout(self.dropout, deterministic=True)
        self.norm2 = nn.LayerNorm(reduction_axes=axes)

        if self.downsample:
            self.conv3 = nn.Conv(self.output_channels, kernel_size=(1,1), strides=self.stride)
            if self.dropout:
                self.drop3 = nn.Dropout(self.dropout, deterministic=True)
            self.norm3 = nn.LayerNorm(reduction_axes=axes)


    def __call__(self, x, train=True):
        residual = x
        
        x = self.conv1(x)
        if self.dropout and train:
            x = self.drop1(x)
        x = self.norm1(x)
        x = self.lrelu(x)

        x = self.conv2(x)
        if self.dropout and train:
            x = self.drop2(x)
        x = self.norm2(x)
        x = self.lrelu(x)

        if hasattr(self, "conv3"):
            residual = self.conv3(residual)
            if self.dropout and train:
                residual = self.drop3(residual)
            residual = self.norm3(residual)

        x += residual
        x = self.lrelu(x)

        return x


class UnetBasicBlock(nn.Module):
    input_channels: int
    output_channels: int
    dims: int = 2
    kernel_size: Union[Sequence[int], int] = (3, 3)
    stride: Union[Sequence[int], int] = (1, 1)
    dropout: Union[float, None] = None
    activation: type = nn.activation.leaky_relu

    def setup(self):
        if dims == 2:
            axes = (-2, -1)
        elif dims == 3:
            axes = (-3, -2, -1)

        self.conv1 = nn.Conv(self.output_channels, self.kernel_size, strides=self.stride)
        if self.dropout:
            self.drop1 = nn.Dropout(self.dropout, deterministic=True)
        self.norm1 = nn.LayerNorm(reduction_axes=axes)

        self.lrelu = self.activation

        self.conv2 = nn.Conv(self.output_channels, self.kernel_size, strides=self.stride)
        if self.dropout:
            self.drop2 = nn.Dropout(self.dropout, deterministic=True)
        self.norm2 = nn.LayerNorm(reduction_axes=axes)


    def __call__(self, x, train=True):
        x = self.conv1(x)
        if self.dropout and train:
            x = self.drop1(x)
        x = self.norm1(x)
        x = self.lrelu(x)

        x = self.conv2(x)
        if self.dropout and train:
            x = self.drop2(x)
        x = self.norm2(x)
        x = self.lrelu(x)
        return x


        




class UnetrBasicBlock(nn.Module):
    input_channels: int
    output_channels: int
    kernel_size: Union[Sequence[int], int] = (3, 3)
    dims: int = 2
    stride: Union[Sequence[int], int] = (1, 1)
    # norm_name: str = "instance"
    res_block: bool = True

    def setup(self):
        # if len(kernel_size) != dims:
        #     raise Exception("kernel size should be the same as dims!")
        if self.res_block:
            self.layer = UnetResBlock(  
                dims=self.dims,
                input_channels=self.input_channels,
                output_channels=self.output_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
            )
            
        else:
            self.layer = UnetBasicBlock(  
                dims=self.dims,
                input_channels=self.input_channels,
                output_channels=self.output_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
            )


    def __call__(self, x, train=True):
        return self.layer(x, train)