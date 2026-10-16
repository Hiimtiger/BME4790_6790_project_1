
import torch
import torch.nn as nn


class BloodCellCNN(nn.Module):
    """
    CNN for classifying 8 blood cell types.

    Supported depths: 2, 3, or 4 convolutional blocks.

    Each block:
        Conv2D -> ReLU -> MaxPool2D

    Classifier:
        Global Average Pooling -> Linear -> 8 logits
    """

    def __init__(self, num_conv_layers=4, num_classes=8):
        super().__init__()

        if num_conv_layers not in [2, 3, 4]:
            raise ValueError(
                "num_conv_layers must be 2, 3, or 4."
            )

        self.num_conv_layers = num_conv_layers
        self.num_classes = num_classes

        filter_sizes = [32, 64, 128, 256]

        layers = []
        in_channels = 3  # RGB input

        for i in range(num_conv_layers):
            out_channels = filter_sizes[i]

            layers.extend([
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ])

            in_channels = out_channels

        # Convolutional encoder
        self.features = nn.Sequential(*layers)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification layer
        self.classifier = nn.Linear(
            in_features=in_channels,
            out_features=num_classes
        )

    def forward(self, x):
        x = self.features(x)

        x = self.global_pool(x)

        x = torch.flatten(x, start_dim=1)

        logits = self.classifier(x)

        return logits

    def get_last_conv_layer(self):
        """
        Returns the last Conv2D layer for Grad-CAM.
        """
        conv_layers = [
            layer for layer in self.features
            if isinstance(layer, nn.Conv2d)
        ]

        return conv_layers[-1]


# =====================================
# MODEL TEST
# =====================================

if __name__ == "__main__":

    # Small synthetic batch:
    # 2 RGB images, each 360 x 363
    dummy_input = torch.randn(2, 3, 360, 363)

    for depth in [2, 3, 4]:

        model = BloodCellCNN(
            num_conv_layers=depth,
            num_classes=8
        )

        with torch.no_grad():
            output = model(dummy_input)

        num_params = sum(
            p.numel() for p in model.parameters()
        )

        print(f"\n{depth}-Layer CNN")
        print(f"Output shape: {output.shape}")
        print(f"Trainable parameters: {num_params:,}")
        print(f"Grad-CAM target: {model.get_last_conv_layer()}")
