"""Architectural components for the Compact Source Removal project.

Includes the PConvUNet model and VGG16-based feature extractor for style loss computation.
Users can define custom architectures here by adding a new class and using its name in the configuration file.
"""

import torch
from torch import nn
from torchvision.models import vgg16, VGG16_Weights
from .blocks import PCDown, PCUp


class PConvUNet(nn.Module):  # pylint: disable=too-many-instance-attributes
    """Partial Convolutional based U-Net architecture for compact source removal task in an inpainting setting."""

    def __init__(self, input_channels: int, output_channels: int):
        """
        Initialize the PConvUNet model.

        Args:
            input_channels (int): Number of input channels.
            output_channels (int): Number of output channels.
        """
        super().__init__()
        self.freeze_enc_bn = False

        # Encoders
        self.e1 = PCDown(input_channels, 32, k=3, s=1, p=1)  # 96x96
        self.e2 = PCDown(32, 64, k=4, s=2, p=1)  # 48x48
        self.e3 = PCDown(64, 64, k=3, s=1, p=1)  # 48x48
        self.e4 = PCDown(64, 128, k=4, s=2, p=1)  # 24x24
        self.e5 = PCDown(128, 128, k=3, s=1, p=1)  # 24x24
        self.e6 = PCDown(128, 256, k=4, s=2, p=1)  # 12x12
        self.e7 = PCDown(256, 256, k=3, s=1, p=1)  # 12x12
        self.e8 = PCDown(256, 512, k=4, s=2, p=1)  # 6x6
        self.e9 = PCDown(512, 512, k=3, s=1, p=1)  # 6x6
        self.e10 = PCDown(512, 512, k=4, s=2, p=1)  # 3x3
        self.e11 = PCDown(512, 512, k=3, s=1, p=0)  # 1x1

        # Decoders
        self.d1 = PCUp(512, 512, 3, 1, 0, activ='relu')  # 3x3
        self.d2 = PCUp(1024, 512, 4, 2, 1, activ='relu')  # 6x6
        self.d3 = PCUp(1024, 512, 3, 1, 1, activ='relu')  # 6x6
        self.d4 = PCUp(1024, 256, 4, 2, 1, activ='relu')  # 12x12
        self.d5 = PCUp(512, 256, 3, 1, 1, activ='relu')  # 12x12
        self.d6 = PCUp(512, 128, 4, 2, 1, activ='relu')  # 24x24
        self.d7 = PCUp(256, 128, 3, 1, 1, activ='relu')  # 24x24
        self.d8 = PCUp(256, 64, 4, 2, 1, activ='relu')  # 48x48
        self.d9 = PCUp(128, 64, 3, 1, 1, activ='relu')  # 48x48
        self.d10 = PCUp(128, 32, 4, 2, 1, activ='relu')  # 96x96
        self.d11 = PCUp(64, output_channels, 3, 1, 1, bn=False, activ=None, conv_bias=True)  # 96x96

    def forward(self, input_image, input_mask):  # pylint: disable=too-many-locals
        """
        Forward pass of the PConvUNet model.

        Args:
            input_image (torch.Tensor): The input image tensor.
            input_mask (torch.Tensor): The input mask tensor.

        Returns:
            torch.Tensor: The output tensor after inpainting.
        """
        # Encoders
        h1, h_mask1 = self.e1(input_image, input_mask)
        h2, h_mask2 = self.e2(h1, h_mask1)
        h3, h_mask3 = self.e3(h2, h_mask2)
        h4, h_mask4 = self.e4(h3, h_mask3)
        h5, h_mask5 = self.e5(h4, h_mask4)
        h6, h_mask6 = self.e6(h5, h_mask5)
        h7, h_mask7 = self.e7(h6, h_mask6)
        h8, h_mask8 = self.e8(h7, h_mask7)
        h9, h_mask9 = self.e9(h8, h_mask8)
        h10, h_mask10 = self.e10(h9, h_mask9)
        h11, h_mask = self.e11(h10, h_mask10)

        # Decoders
        h, h_mask = self.d1(h11, h_mask)
        x, y = torch.cat([h, h10], axis=1), torch.cat([h_mask, h_mask10], axis=1)
        h, h_mask = self.d2(x, y)
        x, y = torch.cat([h, h9], axis=1), torch.cat([h_mask, h_mask9], axis=1)
        h, h_mask = self.d3(x, y)
        x, y = torch.cat([h, h8], axis=1), torch.cat([h_mask, h_mask8], axis=1)
        h, h_mask = self.d4(x, y)
        x, y = torch.cat([h, h7], axis=1), torch.cat([h_mask, h_mask7], axis=1)
        h, h_mask = self.d5(x, y)
        x, y = torch.cat([h, h6], axis=1), torch.cat([h_mask, h_mask6], axis=1)
        h, h_mask = self.d6(x, y)
        x, y = torch.cat([h, h5], axis=1), torch.cat([h_mask, h_mask5], axis=1)
        h, h_mask = self.d7(x, y)
        x, y = torch.cat([h, h4], axis=1), torch.cat([h_mask, h_mask4], axis=1)
        h, h_mask = self.d8(x, y)
        x, y = torch.cat([h, h3], axis=1), torch.cat([h_mask, h_mask3], axis=1)
        h, h_mask = self.d9(x, y)
        x, y = torch.cat([h, h2], axis=1), torch.cat([h_mask, h_mask2], axis=1)
        h, h_mask = self.d10(x, y)
        x, y = torch.cat([h, h1], axis=1), torch.cat([h_mask, h_mask1], axis=1)
        h, h_mask = self.d11(x, y)

        return h

    def train(self, mode=True):
        """Override train to freeze BatchNorm layers in the encoder."""
        super().train(mode)
        if self.freeze_enc_bn:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d) and 'enc' in name:
                    module.eval()


class VGG16FeatureExtractor(nn.Module):
    """Feature extractor based on VGG16 for computing style loss components in the inpainting model."""

    def __init__(self):
        super().__init__()
        vgg16_model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.enc_1 = nn.Sequential(*vgg16_model.features[:5])
        self.enc_2 = nn.Sequential(*vgg16_model.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16_model.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, f'enc_{i + 1}').parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, f'enc_{i + 1}')
            results.append(func(results[-1]))
        return results[1:]
