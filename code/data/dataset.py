"""Dataset classes for loading Herschel Numpy data with various mask options.

This module provides a flexible data loading pipeline for inpainting tasks, allowing for custom mask types.
Users can define their own dataset classes here by following the structure of the provided dataset
class and referencing the class name in the configuration file.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


class HerschelNumpyDatasetMultiMasks(Dataset):
    """PyTorch Dataset for loading Herschel Numpy data with multiple mask options."""

    def __init__(self, data_dir: str, mode: str, length: int, mask_type: str, input_size: tuple,
                 transform=None):  # pylint: disable=too-many-arguments, too-many-positional-arguments
        """
        Args:
            data_dir (str): Path to the directory containing the data.
            mode (str): The mode of the dataset (e.g., 'train', 'valid').
            length (int): Number of samples in the dataset.
            mask_type (str): Type of mask to use, e.g., 'large_mask', 'medium_mask', 'small_mask', or 'snr_mask'.
            input_size (tuple): The expected size of the input data (height, width).
            transform (callable, optional): Transform to apply to the samples.
        """
        self.data_dir = data_dir
        self.mode = mode
        self.length = length
        self.mask_type = mask_type
        self.input_size = input_size
        self.transform = transform

    def __len__(self) -> int:
        """Returns the total number of samples."""
        return self.length

    def __getitem__(self, idx: int):
        """Loads and returns a single sample from the dataset."""
        # Construct file paths
        input_file = os.path.join(self.data_dir, self.mode, f'{idx}.npy')
        target_file = os.path.join(self.data_dir, f'{self.mode}_target', f'{idx}.npy')
        mask_file = os.path.join(self.data_dir, f"{self.mode}_mask", f'{idx}.npy')

        # Load the data
        input_data = np.load(input_file)
        target_data = np.load(target_file)
        mask_data = np.load(mask_file).astype(np.float32)

        # Handle invalid data shapes
        invalid_input_shape = input_data.shape != tuple(self.input_size)
        invalid_target_shape = target_data.shape != tuple(self.input_size)
        invalid_mask_shape = mask_data.shape != (4, *self.input_size)

        if invalid_input_shape or invalid_target_shape or invalid_mask_shape:
            return self._handle_invalid_data()

        # Normalize the input and target data
        input_data, (min_val, max_val) = self.normalize(input_data)
        target_data, _ = self.normalize(target_data, min_val, max_val)

        # Process input and mask according to mask type
        input_data, mask_data = self.process_mask(input_data, mask_data)

        # Convert numpy arrays to PyTorch tensors
        input_data = torch.from_numpy(input_data).float()
        target_data = torch.from_numpy(target_data).unsqueeze(0).float()

        if mask_data is not None:
            mask_data = torch.from_numpy(np.expand_dims(mask_data, axis=0)).float()

        # Apply transformations if any
        if self.transform:
            sample = {'input': input_data, 'target': target_data, 'mask': mask_data}
            transformed_sample = self.transform(sample)
            input_data = transformed_sample['input']
            target_data = transformed_sample['target']
            mask_data = transformed_sample['mask']

        return input_data, target_data, mask_data

    def process_mask(self, input_data: np.ndarray, mask_data: np.ndarray):
        """Adjust input and mask based on the mask type.

        Args:
            input_data (np.ndarray): The input data array.
            mask_data (np.ndarray): The mask data array.

        Returns:
            tuple: Processed input data and mask data.
        """
        input_data = np.expand_dims(input_data, axis=0)

        if self.mask_type == "large_mask":
            mask_data = mask_data[0, :, :]
        elif self.mask_type == "medium_mask":
            mask_data = mask_data[1, :, :]
        elif self.mask_type == "small_mask":
            mask_data = mask_data[2, :, :]
        elif self.mask_type == 'snr_mask':
            mask_data = mask_data[3, :, :]
        else:
            mask_data = None

        return input_data, mask_data

    def _handle_invalid_data(self):
        """Returns zero tensors with appropriate shapes based on mask type."""
        height, width = self.input_size
        zero_input = torch.zeros((1, height, width), dtype=torch.float32)
        zero_target = torch.zeros((1, height, width), dtype=torch.float32)
        zero_mask = None if self.mask_type == "no_mask" else torch.zeros((1, height, width), dtype=torch.float32)

        return zero_input, zero_target, zero_mask

    @staticmethod
    def normalize(data: np.ndarray, min_val: float = None, max_val: float = None):
        """Normalize data to the range [-1, 1].

        Args:
            data (np.ndarray): Data to normalize.
            min_val (float, optional): Minimum value for normalization. If None, it will be computed.
            max_val (float, optional): Maximum value for normalization. If None, it will be computed.

        Returns:
            tuple: Normalized data and (min_val, max_val) used for scaling.
        """
        if min_val is None:
            min_val = np.min(data)
        if max_val is None:
            max_val = np.max(data)

        # Scale data to [0, 1]
        data = (data - min_val) / (max_val - min_val + 1e-7)

        # Scale data to [-1, 1]
        data = data * 2 - 1

        return data, (min_val, max_val)
