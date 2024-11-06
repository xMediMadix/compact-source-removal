"""
Dataset classes for handling Herschel data with multiple mask options and photometric evaluation.

This module provides dataset classes for both training and evaluation of compact source removal on
Herschel photometric data. It includes functionalities to dynamically apply different mask types,
normalize input data, and provide photometric evaluation on processed data. Users can customize
their datasets by defining parameters in the configuration file.
"""

import os
import copy

import cv2
import numpy as np
import pandas as pd
import torch
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from torch.utils.data import Dataset

from code.data.transforms import interpolate_nans


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


class EvaluationHerschelDataset(Dataset):
    """Dataset class for evaluating models on Herschel test data, processing one FITS file at a time."""

    def __init__(self, fits_file: str, csv_file: str, original_map_path: str, crop_size: tuple, mask_type: str,
                 snr_parameters: dict):
        """
        Initialize the dataset for a single FITS file and its corresponding CSV file.

        Args:
            fits_file (str): Path to the FITS file to process.
            csv_file (str): Path to the CSV file with source coordinates.
            original_map_path (str or None): Path to the original map (target) FITS file, or None if not in simulation mode.
            crop_size (tuple): The size of the crop to extract from the data (e.g., (96, 96)).
            mask_type (str): Type of mask to apply (e.g., "large_mask", "medium_mask", "small_mask", "snr_mask").
            snr_parameters (dict): Dictionary containing SNR calculation parameters and mask sizes based on observing mode and band.
        """
        self.fits_file = fits_file
        self.csv_file = csv_file
        self.original_map_path = original_map_path
        self.crop_size = crop_size
        self.mask_type = mask_type
        self.snr_params = snr_parameters

        # Check mode (simulation or real_data)
        self.simulation_mode = self.original_map_path is not None

        # Extract relevant parameters for the current observing mode and band
        self.inner_radius = self.snr_params["inner_radius"]
        self.outer_radius = self.snr_params["outer_radius"]
        self.source_radius = self.snr_params["source_radius"]

        # Load coordinates from CSV; flux data only in simulation mode
        self.meta_df = pd.read_csv(self.csv_file, encoding="utf-8")
        self.coords = self.meta_df.iloc[:, :2].to_numpy()  # RA and DEC
        self.fluxes = self.meta_df.iloc[:, 2].to_numpy() if self.simulation_mode else None

        # Load the original target map if in simulation mode
        if self.simulation_mode:
            with fits.open(self.original_map_path) as hdulist:
                self.target_img = hdulist['image'].data
                self.target_img = interpolate_nans(self.target_img)

        # Load the FITS file (input)
        with fits.open(self.fits_file) as hdulist:
            # Original before processing
            self.input_img_original = copy.deepcopy(hdulist)
            self.input_img = interpolate_nans(
                copy.deepcopy(self.input_img_original['image'].data)
            )
            self.wcs = WCS(hdulist['image'].header)

        # Create a binary mask based on the mask type
        self.mask_data = self.create_mask()

    def create_mask(self):
        """
        Create a binary mask for the input image based on the specified mask type.

        Returns:
            np.ndarray: A binary mask array based on the specified mask type.
        """
        # A binary mask with the same shape as the input image
        binary_mask = np.ones_like(self.input_img)

        # Adding circular masks for each position
        for _, row in self.meta_df.iterrows():
            coord = SkyCoord(row['ra'] * u.deg, row['dec'] * u.deg)
            pixel_center = coord.to_pixel(self.wcs)
            y, x = np.ogrid[:binary_mask.shape[0], :binary_mask.shape[1]]

            if self.mask_type == "large_mask":
                mask_radius = 21  # Radius for large mask
            elif self.mask_type == "medium_mask":
                mask_radius = 11  # Radius for medium mask
            elif self.mask_type == "small_mask":
                mask_radius = 8  # Radius for small mask
            elif self.mask_type == "snr_mask":
                # Calculate SNR for dynamic mask size
                cutout = Cutout2D(self.input_img, coord, self.crop_size, wcs=self.wcs, copy=True).data
                snr = self.calculate_snr(cutout)
                mask_radius = self.determine_mask_size(snr)  # Use SNR-based radius from configuration
            else:
                raise ValueError(f"Unsupported mask type: {self.mask_type}")

            # Draw the circular mask using the determined radius
            mask = (x - pixel_center[0]) ** 2 + (y - pixel_center[1]) ** 2 <= mask_radius ** 2
            binary_mask[mask] = 0

        return binary_mask

    def calculate_snr(self, cutout):
        """
        Calculate SNR using a cutout of the image centered around the source.

        Args:
            cutout (ndarray): A cutout of the image centered on the source.

        Returns:
            float: The calculated SNR value.
        """
        # Create grid for distance calculation
        y, x = np.ogrid[:cutout.shape[0], :cutout.shape[1]]
        center = (cutout.shape[0] // 2, cutout.shape[1] // 2)
        dist_from_center = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)

        # Mask for the annular region (background estimation)
        annular_mask = (dist_from_center >= self.inner_radius) & (dist_from_center <= self.outer_radius)

        # Calculate the background mean and standard deviation in the annular region
        background_mean = np.mean(cutout[annular_mask])
        background_std = np.std(cutout[annular_mask])

        # Mask for the main source region
        source_mask = dist_from_center <= self.source_radius
        peak_flux = np.max(cutout[source_mask])

        # Calculate the SNR
        snr = (peak_flux - background_mean) / background_std

        return snr

    def determine_mask_size(self, snr):
        """
        Determine the radius of the mask based on the SNR value.

        Args:
            snr (float): The calculated SNR value.

        Returns:
            int: The radius of the mask corresponding to the SNR.
        """
        snr_thresholds = self.snr_params["snr_thresholds"]
        mask_sizes = self.snr_params["mask_sizes"]

        for threshold, size in zip(snr_thresholds, mask_sizes):
            if snr >= threshold:
                return size
        return mask_sizes[-1]

    def get_cutout(self, coord: SkyCoord):
        """
        Get cutouts for input and target images around the source position.

        Args:
            coord (SkyCoord): The sky coordinates of the source.

        Returns:
            tuple: Cutouts of the input and target images centered around the source.
        """
        input_cutout = Cutout2D(data=self.input_img, position=coord, size=self.crop_size, wcs=self.wcs, copy=True)
        target_cutout = None

        if self.simulation_mode:
            target_cutout = Cutout2D(data=self.target_img, position=coord, size=self.crop_size, wcs=self.wcs, copy=True)

        return input_cutout, target_cutout

    def get_and_process_mask(self, input_data: np.ndarray, coord: SkyCoord):
        """
        Retrieve and process the mask data based on the mask type specified in the configuration.

        Args:
            input_data (np.ndarray): The input data array.
            coord (SkyCoord): The sky coordinates of the source.

        Returns:
            tuple: The processed input data, selected mask, and mask ROI, adjusted for the specified mask type.
        """
        # Create a cutout for the selected mask around the source
        selected_mask_cutout = Cutout2D(data=self.mask_data, position=coord, size=self.crop_size, wcs=self.wcs,
                                        copy=True)
        selected_mask = selected_mask_cutout.data

        # Create a mask for just the current source using the SNR-based radius
        # No near sources get masks on this binary image
        mask_roi = np.ones(self.crop_size, dtype=np.float32)
        snr = self.calculate_snr(input_data)
        dynamic_mask_radius = self.determine_mask_size(snr)

        # Create a circular mask for the current source only
        center_x, center_y = self.crop_size[1] // 2, self.crop_size[0] // 2  # Center of the cutout
        y, x = np.ogrid[:self.crop_size[0], :self.crop_size[1]]
        source_circle = (x - center_x) ** 2 + (y - center_y) ** 2 <= dynamic_mask_radius ** 2
        mask_roi[source_circle] = 0  # Apply the SNR-based mask for just the current source

        # Erode and blur the mask_roi to get a smoother transition
        # noinspection PyUnresolvedReferences
        eroded_mask = cv2.erode(mask_roi.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1)
        # noinspection PyUnresolvedReferences
        blurred_mask = cv2.GaussianBlur(eroded_mask.astype(np.float32), (5, 5), 5)
        mask_roi = np.clip(blurred_mask, 0, 1)

        return input_data, selected_mask, mask_roi

    def __len__(self) -> int:
        """Return the total number of sources (coordinates) in the dataset."""
        return len(self.coords) if self.coords is not None else 0

    def __getitem__(self, idx: int):
        """Retrieve a cutout for a specific source from the FITS file."""
        ra, dec = self.coords[idx]
        flux_level = self.fluxes[idx] if self.simulation_mode else None

        # Create the SkyCoord object for the coordinates
        coord = SkyCoord(ra * u.deg, dec * u.deg)

        # Create a cutout around the source's position and normalize it
        input_cutout, target_cutout = self.get_cutout(coord)
        input_cutout = np.expand_dims(input_cutout.data, axis=0)
        input_data, (min_val, max_val) = self.normalize(input_cutout)

        # We have target only in simulation mode
        if self.simulation_mode:
            target_data, _ = self.normalize(target_cutout.data, min_val, max_val)
            target_tensor = torch.from_numpy(target_data).unsqueeze(0).float()
        else:
            target_tensor = None

        # Get and process the mask and input data based on the configuration
        input_data, selected_mask, mask_roi = self.get_and_process_mask(input_data, coord)

        img_tensor = torch.from_numpy(input_data).float()
        if selected_mask is not None:
            selected_mask = np.asarray(selected_mask, dtype=selected_mask.dtype.newbyteorder('='))
            selected_mask = torch.from_numpy(np.expand_dims(selected_mask, axis=0)).float()

        return (ra, dec), (img_tensor, target_tensor, selected_mask, mask_roi), (min_val, max_val), flux_level

    def get_original_fits_image(self):
        """
        Returns the original FITS image before any processing (interpolation, starlet).
        Useful for replacing sources with predictions.

        Returns:
            np.ndarray: The original input FITS image.
        """
        return self.input_img_original

    @staticmethod
    def normalize(data: np.ndarray, min_val: float = None, max_val: float = None):
        """
        Normalize the data to the range [-1, 1].
        If min_val and max_val are given, use them; otherwise calculate them.
        Returns the normalized data along with min_val and max_val.

        Args:
            data (np.ndarray): The data to normalize.
            min_val (float): The minimum value for scaling.
            max_val (float): The maximum value for scaling.

        Returns:
            np.ndarray: Normalized data.
            tuple: min_val and max_val used for normalization.
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

    @staticmethod
    def denormalize(data: np.ndarray, min_val: float, max_val: float):
        """
        Denormalize the data back to its original range from [-1, 1] to [min_val, max_val].

        Args:
            data (np.ndarray): The normalized data to denormalize.
            min_val (float): The original minimum value used for normalization.
            max_val (float): The original maximum value used for normalization.

        Returns:
            np.ndarray: The denormalized data.
        """
        # Shift data from [-1, 1] to [0, 1]
        data = (data + 1) / 2

        # Scale data back to the original range
        data = data * (max_val - min_val + 1e-7) + min_val

        return data
