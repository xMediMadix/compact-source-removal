"""
Custom data transformations and interpolation functions for inpainting tasks in the Compact Source Removal project.

This module provides transformations like random flips and rotations, as well as interpolation for filling NaN values.
Users can extend the data augmentation pipeline or add additional preprocessing functions here.
"""

import random
import torch
import kornia

from scipy.interpolate import griddata
import numpy as np


def interpolate_nans(img):
    """
    Interpolates NaN values in a 2D image array by filling them based on surrounding pixel values.
    This function uses nearest-neighbor interpolation to estimate and replace NaN values.

    Args:
        img (np.ndarray): 2D numpy array (image) containing NaN values where interpolation is needed.

    Returns:
        np.ndarray: The input image with NaN values interpolated using nearest neighbors.
    """
    # Create arrays representing the indices of the image
    x = np.arange(0, img.shape[1])
    y = np.arange(0, img.shape[0])

    # Create a meshgrid for the indices
    x_grid, y_grid = np.meshgrid(x, y)

    # Get the indices of where the NaNs are
    array_nans = np.isnan(img)
    nans_x = x_grid[array_nans]
    nans_y = y_grid[array_nans]

    # Get the indices of where the NaNs aren't
    array_not_nans = ~array_nans
    not_nans_x = x_grid[array_not_nans]
    not_nans_y = y_grid[array_not_nans]

    # Get the values of the pixels where the NaNs aren't
    img_values = img[array_not_nans]

    # Interpolate the values for the NaN pixels
    interpolated_values = griddata(
        (not_nans_y, not_nans_x),
        img_values,
        (nans_y, nans_x),
        method='nearest'
    )

    # Fill the NaN pixels in the original image with the interpolated values
    img[array_nans] = interpolated_values

    return img


class CustomTransform:  # pylint: disable=too-few-public-methods
    """Custom data transformation class."""

    def __init__(self, p: float = 0.5):
        """
        Args:
            p (float): Probability of applying the transformation.
        """
        self.p = p
        self.angles = [0, 90, 180, 270]

    def __call__(self, sample: dict):
        """Apply transformations to the sample.

        Args:
            sample (dict): A dictionary with 'input', 'target', and 'mask' tensors.

        Returns:
            dict: Transformed sample.
        """
        input_data, target, mask = sample['input'], sample['target'], sample['mask']

        # Random horizontal flip with probability p
        if random.random() < self.p:
            input_data = torch.flip(input_data, [-1])
            target = torch.flip(target, [-1])
            if mask is not None:
                mask = torch.flip(mask, [-1])

        # Random vertical flip with probability p
        if random.random() < self.p:
            input_data = torch.flip(input_data, [-2])
            target = torch.flip(target, [-2])
            if mask is not None:
                mask = torch.flip(mask, [-2])

        # Random rotation with one of the angles
        angle = random.choice(self.angles)
        angle = torch.tensor(angle, dtype=torch.float32)

        # Rotate input and target
        input_data = kornia.geometry.transform.rotate(
            input_data.unsqueeze(0), angle, mode='bicubic'
        ).squeeze(0)
        target = kornia.geometry.transform.rotate(
            target.unsqueeze(0), angle, mode='bicubic'
        ).squeeze(0)

        # Rotate mask if it exists
        if mask is not None:
            mask = kornia.geometry.transform.rotate(mask.unsqueeze(0), angle, mode='nearest').squeeze(0)

        return {'input': input_data, 'target': target, 'mask': mask}
