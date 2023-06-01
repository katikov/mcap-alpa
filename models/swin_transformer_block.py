import flax.linen as nn
import jax.numpy as jnp
from typing import Sequence
import numpy as np
import alpa
import jax


from .swin_utils import window_partition, window_reverse

class DropPath(nn.Module):
    dropout_rate: float = 0.0
    scale_by_keep: bool = True

    def setup(self):
        self.keep_rate = 1-self.dropout_rate
        self.drop = nn.Dropout(self.dropout_rate, broadcast_dims=(1,2), deterministic=True )

    def __call__(self, x, train=True):
        if train:
            # TODO: check correctness of drop path
            x = self.drop(x)
            if self.keep_rate > 0.0 and self.scale_by_keep:
                x = x / self.keep_prob
        return x



class MLPBlock(nn.Module):
    input_channels: int
    hidden_channels: int
    dropout_rate: float = 0.0
    activation: type = nn.activation.gelu
    # dropout_mode = swin

    def setup(self):
        self.linear1 = nn.Dense(self.hidden_channels)
        self.linear2 = nn.Dense(self.input_channels)
        self.fn = self.activation
        self.drop1 = nn.Dropout(self.dropout_rate, deterministic=True)
        self.drop2 = self.drop1 # nn.Dropout(self.dropout_rate)

    def __call__(self, x, train=True):
        x = self.fn(self.linear1(x))
        if train:
            x = self.drop1(x)
        x = self.linear2(x)
        if train:
            x = self.drop2(x)
        return x


class WindowAttention(nn.Module):
    output_channels: int
    num_heads: int
    window_size: Sequence[int]
    qkv_bias: bool = True
    attn_dropout_rate: float = 0.0
    proj_dropout_rate: float = 0.0

    def setup(self):
        self.head_channels = self.output_channels // self.num_heads
        self.scale = self.head_channels**-0.5

        if len(self.window_size) == 3:
            pass # TODO: 3d
        elif len(self.window_size) == 2:
            bias_shape = ((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_heads)
            # TODO: init bias with normal distribution     trunc_normal_(self.relative_position_bias_table, std=0.02)
            self.relative_position_bias_table = self.param('relative_position_bias_table', 
                                                lambda rng, shape: jax.random.normal(rng, shape=shape, dtype="float32"), 
                                                bias_shape)
            

            coords_h = np.arange(self.window_size[0])
            coords_w = np.arange(self.window_size[1])
            coords = np.stack(np.meshgrid(coords_h, coords_w, indexing="ij"))
            coords_flatten = coords.reshape(2,-1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.transpose(1, 2, 0)
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1


        self.relative_position_index = jnp.array(relative_coords.sum(-1).reshape(-1))
        
        self.qkv = nn.Dense(self.output_channels * 3, use_bias=self.qkv_bias)
        self.attn_drop = nn.Dropout(self.attn_dropout_rate, deterministic=True)
        self.proj = nn.Dense(self.output_channels)
        self.proj_drop = nn.Dropout(self.proj_dropout_rate, deterministic=True)

        self.softmax = nn.activation.softmax


    def __call__(self, x, mask, train=True):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_channels).transpose((2, 0, 3, 1, 4))
        q,k,v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose((0,1,3,2))

        relative_position_bias = self.relative_position_bias_table[
                                self.relative_position_index].reshape(n, n, self.num_heads)
        relative_position_bias = jnp.expand_dims(relative_position_bias.transpose((2, 0, 1)), 0)
        attn = attn + relative_position_bias
        
        if mask is not None:
            attn = attn + mask
        
        attn = self.softmax(attn)  # b, num_heads, n, n
        
        if train:
            attn = self.attn_drop(attn)

        x = (attn @ v).transpose((0,2,1,3)).reshape(b, n, c)
        x = self.proj(x)
        if train:
            x = self.proj_drop(x)
        return x



class SwinTransformerBlock(nn.Module):
    output_channels: int
    num_heads: int
    window_size: Sequence[int]
    shift_size: Sequence[int]
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    dropout_rate: float = 0.0
    attn_dropout_rate: float = 0.0
    dropout_path: float = 0.0
    activation: type = nn.activation.gelu
    norm_layer: type = nn.LayerNorm


    def setup(self):
        self.shifted = self.shift_size and any(i > 0 for i in self.shift_size)
        self.norm1 = self.norm_layer(epsilon=1e-5, reduction_axes=(-3, -2, -1))
        self.attn = WindowAttention(
            output_channels = self.output_channels,
            window_size=self.window_size,
            num_heads=self.num_heads,
            qkv_bias=self.qkv_bias,
            attn_dropout_rate=self.attn_dropout_rate,
            proj_dropout_rate=self.dropout_rate,
        )

        self.drop_path = DropPath(self.dropout_path) if self.dropout_path > 0.0 else None
        self.norm2 = self.norm_layer(epsilon=1e-5, reduction_axes=(-3, -2, -1))
        self.mlp_hidden_channels = int(self.output_channels * self.mlp_ratio)
        self.mlp = MLPBlock(input_channels=self.output_channels, 
                        hidden_channels=self.mlp_hidden_channels, 
                        dropout_rate=self.dropout_rate,
                        activation=self.activation)
    
    
    def forward_part1(self, x, mask_matrix, train=True):
        x_shape = x.shape
        x = self.norm1(x)
        if len(self.window_size) == 2:
            b, h, w, c = x.shape
            # window_size, shift_size = get_window_size((h, w), self.window_size, self.shift_size)
            pad_l = pad_t = 0
            pad_b = (self.window_size[0] - h % self.window_size[0]) % self.window_size[0]
            pad_r = (self.window_size[1] - w % self.window_size[1]) % self.window_size[1]
            x = jnp.pad(x, ((0, 0), (pad_l, pad_r), (pad_t, pad_b), (0, 0)) )
            _, hp, wp, _ = x.shape
            dims = [b, hp, wp]
        else:
            pass # TODO: 3d models

        if self.shifted:
            if len(self.window_size) == 3:
                shifted_x = jnp.roll(x, shift=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), axis=(1, 2, 3))
            elif len(self.window_size) == 2:
                shifted_x = jnp.roll(x, shift=(-self.shift_size[0], -self.shift_size[1]), axis=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, self.window_size)
        attn_windows = x_windows
        attn_windows = self.attn(x_windows, attn_mask, train)
        attn_windows = attn_windows.reshape(-1, *(self.window_size + (c,)) )
        shifted_x = window_reverse(attn_windows, self.window_size, dims)

        if self.shifted:
            if len(self.window_size) == 3:
                x = jnp.roll(shifted_x, shift=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), axis=(1, 2, 3))
            elif len(self.window_size) == 2:
                x = jnp.roll(shifted_x, shift=(self.shift_size[0], self.shift_size[1]), axis=(1, 2))
        else:
            x = shifted_x

        if len(self.window_size) == 3:
            if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
                x = x[:, :d, :h, :w, :]
        elif len(self.window_size) == 2:
            if pad_r > 0 or pad_b > 0:
                x = x[:, :h, :w, :]

        return x

    def forward_part2(self, x, train=True):
        x = self.norm2(x)
        x = self.mlp(x, train)
        if self.drop_path:
            x = self.drop_path(x, train)
        return x



    def __call__(self, x, mask_matrix, train=True):
        shortcut = x
        
        x = self.forward_part1(x, mask_matrix, train)


        if self.drop_path:
            x = self.drop_path(x, train)
        x = x + shortcut
        alpa.mark_pipeline_boundary()
        
        x = x + self.forward_part2(x, train)
        alpa.mark_pipeline_boundary()
        return x




if __name__ == "__main__":
    net = SwinTransformerBlock(output_channels=48, num_heads=3, window_size=(7,7), shift_size=(3,3))
