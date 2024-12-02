import jax
from flax import nnx
from jax import numpy as jnp
from flax import linen as nn


def relu(x):
    return jnp.maximum(x, 0.0)


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
        self.relu = nnx.relu
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
        hidden_dims,
        kernel_size=3,
        dropout=0.2,
        stride=3,
        *,
        rngs: nnx.Rngs,
    ):
        self.extractor = nnx.Conv(
            input_channels,
            hidden_dims[0],
            kernel_size,
            rngs=rngs,
        )
        layers = []

        for layer_in_channels, layer_out_channels in zip(
            hidden_dims[:-1], hidden_dims[1:]
        ):
            layers += [
                TemporalBlock(
                    layer_in_channels,
                    layer_out_channels,
                    kernel_size,
                    dropout,
                    rngs=rngs,
                ),
                relu,
                # Strided conv to downscale
                nnx.Conv(
                    layer_out_channels,
                    layer_out_channels,
                    kernel_size,
                    strides=stride,
                    feature_group_count=layer_out_channels,
                    rngs=rngs,
                ),
                relu,
            ]

        self.network = nnx.Sequential(*layers)
        self.linear = nnx.Linear(hidden_dims[-1], 1, rngs=rngs)

    def __call__(self, x):
        x = self.extractor(x)
        y = self.network(x)
        y = y[..., -1, :]  # Take the last time step output
        return self.linear(y)  # Squeeze the time dim
