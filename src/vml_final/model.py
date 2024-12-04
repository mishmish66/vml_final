import jax
from einops import rearrange
from flax import linen as nn
from flax import nnx
from jax import numpy as jnp


def relu(x):
    return jnp.maximum(x, 0.0)


class MaxPool(nnx.Module):
    def __init__(self, window_size, dim=-1):
        self.window_size = window_size
        self.dim = dim

    def __call__(self, x):
        window_count = x.shape[self.dim] // self.window_size

        # Bring the relevant dim to the front
        x = jnp.moveaxis(x, self.dim, 0)
        # Grab the remainder after windowifying
        x_leftover = x[window_count * self.window_size :]
        x = x[: window_count * self.window_size]
        x = rearrange(
            x, "(windows in_window) ... -> in_window windows ...", windows=window_count
        )
        # Now do the max
        x = jnp.max(x, axis=0)
        # Handle the leftover
        x_leftover = jnp.max(x_leftover, axis=0, keepdims=True)

        x = jnp.concatenate([x, x_leftover], axis=0)

        return jnp.moveaxis(x, 0, self.dim)


class TemporalBlock(nnx.Module):
    def __init__(
        self,
        in_features,
        out_features,
        kernel_size,
        dropout=0.2,
        *,
        rngs: nnx.Rngs,
    ):
        # First convolution layer
        self.conv1 = nnx.Conv(
            in_features,
            out_features,
            kernel_size,
            rngs=rngs,
        )
        self.bn1 = nnx.BatchNorm(out_features, rngs=rngs)
        self.relu = relu
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

        # Second convolution layer
        self.conv2 = nnx.Conv(
            out_features,
            out_features,
            kernel_size,
            rngs=rngs,
        )
        self.bn2 = nnx.BatchNorm(out_features, rngs=rngs)

        self.net = nnx.Sequential(
            *(self.conv1, self.bn1, self.relu, self.dropout),
            *(self.conv2, self.bn2, self.relu, self.dropout),
        )

        # Residual connection
        self.downsample = (
            nnx.Conv(in_features, out_features, kernel_size=1, rngs=rngs)
            if in_features != out_features
            else None
        )

    def __call__(self, x):
        out = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        # Slice output to match input length (in case padding added extra time steps)
        out = x[..., : out.shape[-1]]
        return out + x


class TemporalConvolutionalNetwork(nnx.Module):
    def __init__(
        self,
        input_channels,
        conv_hidden_dims,
        mlp_hidden_dims=[],
        kernel_size=3,
        dropout=0.2,
        stride=3,
        *,
        rngs: nnx.Rngs,
    ):
        # Extractor does not have strides
        self.initial_extractor = nnx.Conv(
            input_channels,
            conv_hidden_dims[0],
            kernel_size,
            rngs=rngs,
        )
        layers = []

        for layer_in_channels, layer_out_channels in zip(
            conv_hidden_dims[:-1], conv_hidden_dims[1:]
        ):
            layers += [
                nnx.Dropout(dropout, rngs=rngs),
                nnx.BatchNorm(layer_in_channels, rngs=rngs),
                relu,
                nnx.Conv(
                    layer_in_channels,
                    layer_out_channels,
                    kernel_size,
                    strides=stride,
                    rngs=rngs,
                ),
            ]

        self.convnet = nnx.Sequential(*layers)

        mlp_dims = conv_hidden_dims[-1:] + mlp_hidden_dims + [1]
        mlp_layer_list = []
        for mlp_layer_in_dim, mlp_layer_out_dim in zip(mlp_dims[:-1], mlp_dims[1:]):
            mlp_layer_list += [
                relu,
                nnx.Linear(mlp_layer_in_dim, mlp_layer_out_dim, rngs=rngs),
            ]

        self.mlp = nnx.Sequential(*mlp_layer_list)

    def __call__(self, x):
        x = self.initial_extractor(x)
        y = self.convnet(x)
        if y.shape[-2] > 1:
            print(
                f"Warning: the output is indexing only the last time step, this may cause information loss. The final time dim is {y.shape[-2]}"
            )
        y = y[..., -1, :]  # Take the last time step output
        y = self.mlp(y)
        return y[..., 0]  # Squeeze the last dim
