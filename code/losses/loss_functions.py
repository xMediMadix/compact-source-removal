"""Loss functions for the Compact Source Removal project.

This module includes already implemented loss functions (InpaintingLoss, L1Loss, and MSELoss).
Users can define custom loss functions here by adding a new class and using its name in the configuration file.
"""

import torch
from torch import nn
from kornia.losses import SSIMLoss
from code.losses.utils import gram_matrix


class InpaintingLoss(nn.Module):
    """Inpainting loss function combining L1, style, SSIM, and source consistency losses.

    This code is mainly based on: https://github.com/naoto0804/pytorch-inpainting-with-partial-conv/blob/master/loss.py

    Args:
        extractor (nn.Module): A feature extractor model for style loss.
    """

    def __init__(self, extractor):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.extractor = extractor
        self.ssim = SSIMLoss(window_size=5)

    def forward(self, input_img, mask, output, gt):
        """
        Args:
            input_img (torch.Tensor): The input image.
            mask (torch.Tensor or None): The binary mask, or None if not used.
            output (torch.Tensor): The output image from the model.
            gt (torch.Tensor): The ground truth image.

        Returns:
            dict: A dictionary containing the computed losses.
        """
        loss_dict = {}

        output_comp = self.compute_output_comp(input_img, mask, output)

        loss_dict['reconstruction'] = self.compute_reconstruction_loss(output, gt, mask)
        loss_dict['style'] = self.compute_style_loss(output, output_comp, gt)
        loss_dict['ssim'] = self.ssim(output, gt)
        loss_dict['source'] = self.compute_source_loss(input_img, gt, output_comp)

        return loss_dict

    def compute_output_comp(self, input_img, mask, output):
        """Compute the composite output image."""
        if mask is not None:
            return mask * input_img + (1 - mask) * output
        return output

    def compute_reconstruction_loss(self, output, gt, mask):
        """Compute the reconstruction loss."""
        if mask is not None:
            return (
                    self.l1(mask * output, mask * gt) +
                    self.l1((1 - mask) * output, (1 - mask) * gt)
            )
        return self.l1(output, gt)

    def compute_style_loss(self, output, output_comp, gt):
        """Compute the style loss."""
        if output.shape[1] == 3:
            feat_output_comp = self.extractor(output_comp)
            feat_output = self.extractor(output)
            feat_gt = self.extractor(gt)
        elif output.shape[1] == 1:
            feat_output_comp = self.extractor(torch.cat([output_comp] * 3, 1))
            feat_output = self.extractor(torch.cat([output] * 3, 1))
            feat_gt = self.extractor(torch.cat([gt] * 3, 1))
        else:
            raise ValueError('Only gray and RGB images are supported.')

        style_loss = 0.0
        for i in range(3):
            style_loss += self.l1(gram_matrix(feat_output[i]), gram_matrix(feat_gt[i]))
            style_loss += self.l1(gram_matrix(feat_output_comp[i]), gram_matrix(feat_gt[i]))

        return style_loss

    def compute_source_loss(self, input_img, gt, output_comp):
        """Compute the source consistency loss."""
        return self.l1(input_img - gt, input_img - output_comp)


class L1Loss(nn.Module):
    """Simple L1 loss function."""

    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, _input, _mask, output, gt):
        """
        Args:
            _input (torch.Tensor): The input image (unused).
            _mask (torch.Tensor or None): The binary mask (unused).
            output (torch.Tensor): The output image from the model.
            gt (torch.Tensor): The ground truth image.

        Returns:
            dict: A dictionary containing the computed loss.
        """
        return {'reconstruction': self.l1(output, gt)}


class MSELoss(nn.Module):
    """Simple MSE loss function."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, _input, _mask, output, gt):
        """
        Args:
            _input (torch.Tensor): The input image (unused).
            _mask (torch.Tensor or None): The binary mask (unused).
            output (torch.Tensor): The output image from the model.
            gt (torch.Tensor): The ground truth image.

        Returns:
            dict: A dictionary containing the computed loss.
        """
        return {'reconstruction': self.mse(output, gt)}
