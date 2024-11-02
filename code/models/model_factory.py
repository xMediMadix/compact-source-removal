"""Factory function to dynamically define and initialize models based on configuration.

This module allows for the flexible creation of models by specifying the model name in the configuration
file. Models are defined in the `architectures.py` module and are instantiated here.
"""

import importlib
import torch


def define_model(config: dict, device: torch.device):
    """Dynamically define the model based on the configuration and place it on the device.

    Args:
        config (dict): Configuration parameters.
        device (torch.device): The device to place the model on.

    Returns:
        torch.nn.Module: The initialized PyTorch model.
    """
    model_name = config['model']['name']
    input_channels = config['model']['input_channels']
    output_channels = config['model']['output_channels']

    module_name = "code.models.architectures"
    model_class = getattr(importlib.import_module(module_name), model_name)

    model = model_class(input_channels=input_channels, output_channels=output_channels)

    model = model.to(device)

    return model
