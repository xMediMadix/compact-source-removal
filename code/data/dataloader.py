"""Dataloader and dataset definition for the Compact Source Removal project.

This module provides flexible dataloaders with custom collation and dynamic dataset definitions based
on configuration. Users can modify or extend dataloader settings here as needed.
"""


import os
import importlib
import pandas as pd

from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
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
