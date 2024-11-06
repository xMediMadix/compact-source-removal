"""
Dataloader and dataset definitions for the Compact Source Removal project.

This module provides data loading functionalities tailored for both training and evaluation phases.
It includes custom collate functions, dataset definitions, and dataloader configurations based
on user specifications in configuration files. During evaluation, custom handling of target and flux levels
is provided for real and simulated datasets, supporting a seamless evaluation process.
"""

import os
import json
import importlib

import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from code.data.dataset import EvaluationHerschelDataset
from code.data.transforms import CustomTransform


def custom_collate(batch):
    """Custom collate function to handle None masks in the batch."""
    inputs, targets, masks = zip(*batch)

    # Collate inputs and targets normally
    inputs = default_collate(inputs)
    targets = default_collate(targets)

    # Handle masks: if any mask is None, return None for masks, else collate normally
    if any(mask is None for mask in masks):
        masks = None
    else:
        masks = default_collate(masks)

    return inputs, targets, masks


def custom_evaluation_collate(batch):
    """
    Custom collate function for handling cases where target_tensor or flux_level might be None.

    Args:
        batch (list): List of tuples from the dataset, where each tuple consists of:
                      (RA, DEC), (img_tensor, target_tensor, selected_mask, mask_roi), (min_val, max_val), flux_level

    Returns:
        tuple: Collated data with None values handled appropriately.
    """
    # Separate each part of the batch
    coords, tensors, norm_vals, flux_levels = zip(*batch)

    # Process img_tensor, target_tensor, selected_mask, and mask_roi
    img_tensor, target_tensor, selected_mask, mask_roi = zip(*tensors)

    # If any target_tensor or flux_level is None, set them to None for the whole batch
    if any(t is None for t in target_tensor) or any(f is None for f in flux_levels):
        target_tensor_collated = None
        flux_level_collated = None
    else:
        # Otherwise, use default_collate to batch them normally
        target_tensor_collated = default_collate(target_tensor)
        flux_level_collated = default_collate(flux_levels)

    # Use default_collate for other elements and return None for target_tensor and flux_level if needed
    return (
        default_collate(coords),
        (
            default_collate(img_tensor), target_tensor_collated, default_collate(selected_mask),
            default_collate(mask_roi)),
        default_collate(norm_vals),
        flux_level_collated
    )


def define_dataloader(config: dict):
    """Define train and validation dataloaders based on configuration.

    Args:
        config (dict): Configuration parameters.

    Returns:
        tuple: Training and validation dataloaders.
    """
    # Define the datasets
    train_dataset, valid_dataset = define_dataset(config)

    # Create the DataLoader for training
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['dataloader']['train']['batch_size'],
        shuffle=config['dataloader']['train']['shuffle'],
        num_workers=config['dataloader']['train']['num_workers'],
        pin_memory=config['dataloader']['train']['pin_memory'],
        collate_fn=custom_collate
    )

    # Create the DataLoader for validation
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config['dataloader']['validation']['batch_size'],
        shuffle=config['dataloader']['validation']['shuffle'],
        num_workers=config['dataloader']['validation']['num_workers'],
        pin_memory=config['dataloader']['validation']['pin_memory'],
        collate_fn=custom_collate
    )

    return train_loader, valid_loader


def define_dataset(config: dict):
    """Dynamically define the dataset based on the configuration.

    Args:
        config (dict): Configuration parameters.

    Returns:
        Dataset: An instance of the specified dataset class.
    """
    dataset_name = config['data']['dataset_name']
    module_name = "code.data.dataset"
    dataset_class = getattr(importlib.import_module(module_name), dataset_name)

    data_dir = config['data']['data_dir']
    mask_type = config['data']['mask_type']
    augmentation = config['data']['augmentation']
    input_size = tuple(config['data']['input_size'])

    # Load CSV files for train and validation
    train_path = os.path.join(config['data']['csv_dir'], 'train.csv')
    valid_path = os.path.join(config['data']['csv_dir'], 'valid.csv')

    train_csv = pd.read_csv(train_path)
    valid_csv = pd.read_csv(valid_path)

    # Instantiate the dataset objects
    transform = CustomTransform(p=augmentation['probability']) if augmentation['use_augmentation'] else None

    train_dataset = dataset_class(
        data_dir=data_dir,
        mode='train',
        length=len(train_csv),
        mask_type=mask_type,
        input_size=input_size,
        transform=transform
    )

    valid_dataset = dataset_class(
        data_dir=data_dir,
        mode='valid',
        length=len(valid_csv),
        mask_type=mask_type,
        input_size=input_size,
        transform=None
    )

    return train_dataset, valid_dataset


def load_snr_config(config_path: str):
    """Load SNR configuration parameters from a JSON file."""
    with open(config_path, "r", encoding="utf-8") as file:
        snr_config = json.load(file)
    return snr_config


def define_evaluation_dataset(config: dict, fits_file: str, csv_file: str, original_map: str):
    """
    Define the dataset for evaluation based on a single FITS file and its corresponding CSV file.

    Args:
        config (dict): Configuration parameters from the JSON file.
        fits_file (str): Path to the FITS file to process.
        csv_file (str): Path to the CSV file with source coordinates.
        original_map (str or None): Path to the original target map (FITS file) or None if not available.

    Returns:
        EvaluationHerschelDataset: Instantiated dataset object for evaluation.
    """
    input_size = tuple(config['data']['input_size'])
    mask_type = config['data']['mask_type']
    observing_params = config['observing_parameters']

    # Load SNR-related parameters based on observing mode and band
    snr_config = load_snr_config(config_path=os.path.join("config", "parameters", "snr_config.json"))
    observing_mode = observing_params["observing_mode"]
    band = observing_params["band"]

    # Extract the specific SNR parameters for the mode and band
    snr_parameters = snr_config[observing_mode][band]

    # Instantiate the evaluation dataset with SNR parameters
    dataset = EvaluationHerschelDataset(
        fits_file=fits_file,
        csv_file=csv_file,
        original_map_path=original_map,
        crop_size=input_size,
        mask_type=mask_type,
        snr_parameters=snr_parameters
    )

    return dataset


def define_evaluation_dataloader(config: dict, fits_file: str, csv_file: str, original_map: str) -> DataLoader:
    """
    Define the evaluation DataLoader for a single FITS file.

    Args:
        config (dict): Configuration parameters from the JSON file.
        fits_file (str): Path to the FITS file to process.
        csv_file (str): Path to the CSV file with source coordinates.
        original_map (str or None): Path to the original target map (FITS file) or None if not available.

    Returns:
        DataLoader: DataLoader object for loading the evaluation dataset.
    """
    dataset = define_evaluation_dataset(config, fits_file, csv_file, original_map)

    # Create DataLoader for evaluation
    data_loader = DataLoader(
        dataset,
        batch_size=config['dataloader']['batch_size'],
        shuffle=False,
        num_workers=config['dataloader']['num_workers'],
        pin_memory=True,
        collate_fn=custom_evaluation_collate
    )

    return data_loader
