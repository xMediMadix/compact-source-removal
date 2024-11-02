"""Custom data transformations for inpainting tasks in the Compact Source Removal project.

This module provides transformations like random flips and rotations. Users can define additional
custom transformations here to extend the data augmentation pipeline.
"""

import random
import torch
import kornia


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
