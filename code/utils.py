"""Utility functions for setting random seeds, defining optimizers, and configuring schedulers."""

import random
import numpy as np
import torch
from torch import optim
from torch import nn
from typing import Tuple, Optional, Dict, Any


def set_seed(seed: int):
    """Sets the random seed for reproducibility.

    Args:
        seed (int): The seed value.
    """
    if seed >= 0:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    if seed >= 0:  # Slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # Faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def define_optimizer_and_scheduler(
        model: nn.Module, config: Dict[str, Any]
) -> Tuple[optim.Optimizer, Optional[optim.lr_scheduler._LRScheduler]]:
    """Define the optimizer and scheduler based on the configuration.

    Args:
        model (nn.Module): The model to optimize.
        config (dict): Configuration parameters.

    Returns:
        optimizer: The initialized optimizer.
        scheduler: The initialized scheduler (or None if not used).
    """
    # Initialize the optimizer
    optimizer_config = config['optimization']['optimizer']
    optimizer_type = optimizer_config['type']
    lr = optimizer_config['learning_rate']
    weight_decay = optimizer_config['weight_decay']

    # Get the optimizer class from torch.optim
    optimizer_class = getattr(optim, optimizer_type, None)
    if optimizer_class is None:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    optimizer = optimizer_class(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Initialize the scheduler
    scheduler = None
    if 'scheduler' in config['optimization']:
        scheduler_config = config['optimization']['scheduler']
        scheduler_type = scheduler_config['type']
        scheduler_params = scheduler_config['params']

        # Get the scheduler class from torch.optim.lr_scheduler
        scheduler_class = getattr(optim.lr_scheduler, scheduler_type, None)
        if scheduler_class is None:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

        scheduler = scheduler_class(optimizer, **scheduler_params)

    return optimizer, scheduler
