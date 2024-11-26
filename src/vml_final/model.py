from flax import nnx


class TemporalBlock(nnx.Module):
    def __init__(
        self, in_features, out_features, kernel_size, stride, dilation, dropout=0.2
    ):
        # Calculate padding to keep output size the same as input size
        padding = (kernel_size - 1) * dilation

        # First convolution layer
        self.conv1 = nnx.Conv(
            in_features,
            out_features,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.bn1 = nnx.BatchNorm(out_features)
        self.relu = nnx.relu
        self.dropout = nnx.Dropout(dropout)

        # Second convolution layer
        self.conv2 = nnx.Conv(
            out_features,
            out_features,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.bn2 = nnx.BatchNorm(out_features)

        self.net = nnx.Sequential(
            *(self.conv1, self.bn1, self.relu, self.dropout),
            *(self.conv2, self.bn2, self.relu, self.dropout),
        )

        # Residual connection
        self.downsample = (
            nnx.Conv1d(in_features, out_features, kernel_size=1)
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
    def __init__(self, num_inputs, feature_dim, kernel_size=3, dropout=0.2):
        layers = []
        num_levels = len(feature_dim)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else feature_dim[i - 1]
            out_channels = feature_dim[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    dropout=dropout,
                ),
                # Strided convolution to downscale
                nnx.Conv(out_channels, out_channels, 5, 3),
            ]

        self.network = nnx.Sequential(*layers)
        self.linear = nnx.Linear(feature_dim[-1], 1)

    def __call__(self, x):
        # Input shape: [batch_size, channels, timesteps]
        y = self.network(x)
        y = y[..., -1]  # Take the last time step output
        return self.linear(y)[..., 0]  # Squeeze the last dim
