"""Building blocks for partial convolution operations in the Compact Source Removal project.

This module provides classes for partial convolutional layers and downsampling/upsampling blocks,
with mask handling for inpainting tasks.
"""


import math
import torch
from torch import nn


def weights_init(init_type: str = 'gaussian'):
    """Initializes the weights of the network layers.

    Args:
        init_type (str): Type of initialization ('gaussian', 'xavier', 'kaiming', 'orthogonal', 'default').
    """

    def init_fun(m: nn.Module):
        classname = m.__class__.__name__
        # Apply weight initialization to Conv and Linear layers
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                raise ValueError(f"Unsupported initialization: {init_type}")
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun


class PartialConv(nn.Module):
    """Partial Convolution layer with mask handling for inpainting tasks."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1,
                 padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = True): # pylint: disable=too-many-arguments, too-many-positional-arguments
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolution kernel.
            stride (int): Stride of the convolution. Default is 1.
            padding (int): Zero-padding added to both sides of the input. Default is 0.
            dilation (int): Spacing between kernel elements. Default is 1.
            groups (int): Number of blocked connections from input channels to output channels. Default is 1.
            bias (bool): If True, adds a learnable bias to the output. Default is True.
        """
        super().__init__()
        # Initialize input and mask convolutions
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, False)
        self.input_conv.apply(weights_init('kaiming'))
        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        # Ensure the mask convolution parameters are not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input: torch.Tensor, mask: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """Performs forward pass of the partial convolution.

        Args:
            input (torch.Tensor): Input tensor.
            mask (torch.Tensor): Mask tensor.

        Returns:
            torch.Tensor: Output tensor after partial convolution.
            torch.Tensor: Updated mask tensor.
        """
        # http://masc.cs.gmu.edu/wiki/partialconv
        # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
        # W^T* (M .* X) / sum(M) + b = [C(M .* X) – C(0)] / D(M) + C(0)

        # Perform convolution with masked input
        output = self.input_conv(input * mask)
        output_bias = (self.input_conv.bias.view(1, -1, 1, 1).expand_as(output)
                       if self.input_conv.bias is not None else torch.zeros_like(output))

        # Perform convolution on the mask
        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        # Handle regions where the mask is zero
        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)

        # Normalize the output and handle holes
        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)

        # Create new mask for the output
        new_mask = torch.ones_like(output).masked_fill_(no_update_holes, 0.0)

        return output, new_mask


class PartialConvTranspose(nn.Module):
    """Partial Transposed Convolution layer with mask handling for upsampling."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1,
                 padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = True): # pylint: disable=too-many-arguments, too-many-positional-arguments
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolution kernel.
            stride (int): Stride of the convolution. Default is 1.
            padding (int): Zero-padding added to both sides of the input. Default is 0.
            output_padding (int): Additional size added to one side of each dimension in the output shape. Default is 0.
            groups (int): Number of blocked connections from input channels to output channels. Default is 1.
            bias (bool): If True, adds a learnable bias to the output. Default is True.
        """
        super().__init__()
        # Initialize input and mask transposed convolutions
        self.input_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                             stride, padding, output_padding, groups, bias)
        self.mask_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                            stride, padding, output_padding, groups, False)
        self.input_conv.apply(weights_init('kaiming'))
        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        # Ensure the mask convolution parameters are not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input: torch.Tensor, mask: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """Performs forward pass of the partial transposed convolution.

        Args:
            input (torch.Tensor): Input tensor.
            mask (torch.Tensor): Mask tensor.

        Returns:
            torch.Tensor: Output tensor after partial transposed convolution.
            torch.Tensor: Updated mask tensor.
        """
        # Perform transposed convolution with masked input
        output = self.input_conv(input * mask)
        output_bias = (self.input_conv.bias.view(1, -1, 1, 1).expand_as(output)
                       if self.input_conv.bias is not None else torch.zeros_like(output))

        # Perform transposed convolution on the mask
        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        # Handle regions where the mask is zero
        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)

        # Normalize the output and handle holes
        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)

        # Create new mask for the output
        new_mask = torch.ones_like(output).masked_fill_(no_update_holes, 0.0)

        return output, new_mask


class PCDown(nn.Module):
    """Partial Convolutional Downsampling block with optional BatchNorm and activation."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 2, p: int = 1,
                 bn: bool = True, activ: str = 'relu', conv_bias: bool = False):
        """
        Args:
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels.
            k (int): Kernel size. Default is 3.
            s (int): Stride size. Default is 2.
            p (int): Padding size. Default is 1.
            bn (bool): If True, applies BatchNorm. Default is True.
            activ (str): Activation function ('relu' or 'leaky'). Default is 'relu'.
            conv_bias (bool): If True, adds a learnable bias to the convolution. Default is False.
        """
        super().__init__()
        # Initialize partial convolution
        self.conv = PartialConv(in_ch, out_ch, k, s, p, bias=conv_bias)

        # Optional BatchNorm and activation
        if bn:
            self.bn = nn.BatchNorm2d(out_ch)
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input: torch.Tensor, input_mask: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """Performs forward pass of the downsampling block.

        Args:
            input (torch.Tensor): Input tensor.
            input_mask (torch.Tensor): Input mask tensor.

        Returns:
            torch.Tensor: Output tensor after partial convolution and optional BatchNorm and activation.
            torch.Tensor: Updated mask tensor.
        """
        h, h_mask = self.conv(input, input_mask)
        if hasattr(self, 'bn'):
            h = self.bn(h)
        if hasattr(self, 'activation'):
            h = self.activation(h)
        return h, h_mask


class PCUp(nn.Module):
    """Partial Convolutional Upsampling block with optional BatchNorm and activation."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 2, p: int = 1,
                 bn: bool = True, activ: str = 'relu', conv_bias: bool = False):
        """
        Args:
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels.
            k (int): Kernel size. Default is 3.
            s (int): Stride size. Default is 2.
            p (int): Padding size. Default is 1.
            bn (bool): If True, applies BatchNorm. Default is True.
            activ (str): Activation function ('relu' or 'leaky'). Default is 'relu'.
            conv_bias (bool): If True, adds a learnable bias to the convolution. Default is False.
        """
        super().__init__()
        # Initialize partial transposed convolution
        self.conv = PartialConvTranspose(in_ch, out_ch, k, s, p, bias=conv_bias)

        # Optional BatchNorm and activation
        if bn:
            self.bn = nn.BatchNorm2d(out_ch)
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input: torch.Tensor, input_mask: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """Performs forward pass of the upsampling block.

        Args:
            input (torch.Tensor): Input tensor.
            input_mask (torch.Tensor): Input mask tensor.

        Returns:
            torch.Tensor: Output tensor after partial transposed convolution and optional BatchNorm and activation.
            torch.Tensor: Updated mask tensor.
        """
        h, h_mask = self.conv(input, input_mask)
        if hasattr(self, 'bn'):
            h = self.bn(h)
        if hasattr(self, 'activation'):
            h = self.activation(h)
        return h, h_mask
