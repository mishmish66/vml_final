from flax import nnx


class TemporalBlock(nnx.Module):
    def __init__(
        self,
        in_features,
        out_features,
        kernel_size,
        stride,
        dilation,
        dropout=0.2,
        *,
        rngs: nnx.Rngs,
    ):
        # Calculate padding to keep output size the same as input size
        padding = "SAME"

        # First convolution layer
        self.conv1 = nnx.Conv(
            in_features,
            out_features,
            kernel_size,
            strides=stride,
            padding=padding,
            kernel_dilation=dilation,
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
            strides=stride,
            padding=padding,
            kernel_dilation=dilation,
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
        out = out[..., : x.shape[-1]]
        return self.relu(out + x)


class TemporalConvolutionalNetwork(nnx.Module):
    def __init__(
        self,
        input_channels,
        extractor_hidden_features,
        extractor_groups,
        extractor_kernel_size,
        hidden_dims,
        kernel_size=3,
        dropout=0.2,
        *,
        rngs: nnx.Rngs,
    ):
        layers = []

        initial_pointwise = nnx.Conv(
            input_channels,
            extractor_hidden_features,
            1,
            # strides=1,
            # padding="SAME",
            # kernel_dilation=1,
            rngs=rngs,
        )
        initial_depthwise = nnx.Conv(
            extractor_hidden_features,
            hidden_dims[0],
            extractor_kernel_size,
            padding="SAME",
            feature_group_count=extractor_groups,
            rngs=rngs,
        )

        for layer_in_channels, layer_out_channels in zip(
            hidden_dims[:-1], hidden_dims[1:]
        ):
            layers += [
                TemporalBlock(
                    layer_in_channels,
                    layer_out_channels,
                    kernel_size,
                    stride=1,
                    dilation=1,
                    dropout=dropout,
                    rngs=rngs,
                ),
                # Strided convolution to downscale
                nnx.Conv(
                    layer_out_channels,
                    layer_out_channels,
                    kernel_size,
                    2,
                    rngs=rngs,
                ),
            ]

        self.network = nnx.Sequential(initial_pointwise, initial_depthwise, *layers)
        self.linear = nnx.Linear(hidden_dims[-1], 1, rngs=rngs)

    def __call__(self, x):
        # Input shape: [batch_size, channels, timesteps]
        y = self.network(x)
        y = y[..., -1, :]  # Take the last time step output
        return self.linear(y)  # Squeeze the time dim
