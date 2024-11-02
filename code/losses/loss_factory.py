"""Factory function to define and initialize loss functions based on configuration."""

import torch
from code.losses.loss_functions import InpaintingLoss, L1Loss, MSELoss
from code.models.architectures import VGG16FeatureExtractor


def define_loss(config: dict, device: torch.device):
    """Define the loss function based on the configuration.

    Args:
        config (dict): Configuration parameters.
        device (torch.device): The device to place the loss function on.

    Returns:
        nn.Module: The initialized loss function.
    """
    loss_name = config['loss']['name']

    if loss_name == "InpaintingLoss":
        return InpaintingLoss(VGG16FeatureExtractor()).to(device)
    if loss_name == "L1LossCustom":
        return L1Loss().to(device)
    if loss_name == "MSELossCustom":
        return MSELoss().to(device)

    raise ValueError(f"Unknown loss function: {loss_name}")
