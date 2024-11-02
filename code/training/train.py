"""Training functions for the Compact Source Removal project.

This module includes functions to train and validate models, calculate losses, and handle logging.
"""

import torch
from torch import nn
from tqdm import tqdm


def calculate_loss(loss_fn, lambdas, inputs, masks, outputs,
                   targets):  # pylint: disable=too-many-arguments, too-many-positional-arguments
    """Calculate the weighted loss using the provided loss function and lambda coefficients.

    Args:
        loss_fn (callable): Function to compute loss, returning a dictionary of loss components.
        lambdas (dict): Coefficients for each loss component.
        inputs (torch.Tensor): Batch of input images.
        masks (torch.Tensor or None): Batch of masks, or None if not used.
        outputs (torch.Tensor): Model predictions for the batch.
        targets (torch.Tensor): Ground truth images for the batch.

    Returns:
        float: Total computed loss for the batch.
    """
    loss_dict = loss_fn(
        inputs[:, 0:1, :, :],
        masks[:, 0:1, :, :] if masks is not None else None,
        outputs,
        targets
    )

    total_loss = 0.0
    for key, coef in lambdas.items():
        if key in loss_dict:
            total_loss += coef * loss_dict[key]

    return total_loss


def _train_one_epoch(model, train_loader, calculate_loss, loss_fn, lambdas, optimizer, device):
    """Train the model for one epoch.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        calculate_loss (callable): Function to compute weighted loss.
        loss_fn (callable): Function to compute loss components.
        lambdas (dict): Coefficients for each loss component.
        optimizer (torch.optim.Optimizer): Optimizer for model training.
        device (torch.device): Device to run training on.

    Returns:
        float: The average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0

    for _, (inputs, targets, masks) in enumerate(tqdm(train_loader, desc='Training Loop:')):
        inputs, targets = inputs.to(device), targets.to(device)
        masks = masks.to(device) if masks is not None else None

        # Forward pass
        outputs = model(inputs, masks)
        loss = calculate_loss(loss_fn, lambdas, inputs, masks, outputs, targets)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print(f"Training Loss: {epoch_loss:.4f}")
    return epoch_loss


def _validate(model, valid_loader, calculate_loss, loss_fn, lambdas, device, logger, visualization_batches, epoch):
    """Validate the model on the validation set.

    Args:
        model (torch.nn.Module): The model to validate.
        valid_loader (torch.utils.data.DataLoader): DataLoader for the validation data.
        calculate_loss (callable): Function to compute weighted loss.
        loss_fn (callable): Function to compute loss components.
        lambdas (dict): Coefficients for each loss component.
        device (torch.device): Device to run validation on.
        logger (Logger): Logger instance for visualizations and metric tracking.
        visualization_batches (list): List of batch indices to visualize.
        epoch (int): Current epoch number.

    Returns:
        tuple: Average validation loss and L1 loss for the epoch.
    """
    model.eval()
    running_val_loss = 0.0
    running_val_l1 = 0.0

    with torch.no_grad():
        for i, (inputs, targets, masks) in enumerate(tqdm(valid_loader, desc='Validation Loop:')):
            inputs, targets = inputs.to(device), targets.to(device)
            masks = masks.to(device) if masks is not None else None

            # Forward pass
            outputs = model(inputs, masks)

            # Calculate loss
            val_loss = calculate_loss(loss_fn, lambdas, inputs, masks, outputs, targets)
            running_val_loss += val_loss.item()

            # Calculate L1 loss separately
            l1_loss = nn.L1Loss()(outputs, targets)
            running_val_l1 += l1_loss.item()

            # Visualization
            if i in visualization_batches:
                logger.visualize(inputs, masks, outputs, targets, epoch, i)

    return running_val_loss / len(valid_loader), running_val_l1 / len(valid_loader)


def train(model, train_loader, valid_loader, loss_fn, optimizer, scheduler, logger, config: dict,
          device: torch.device):  # pylint: disable=too-many-arguments, too-many-positional-arguments
    """Train and validate the model.

    Args:
        model (torch.nn.Module): Model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        valid_loader (torch.utils.data.DataLoader): DataLoader for the validation data.
        loss_fn (callable): Function to compute loss, returning a dictionary of loss components.
        optimizer (torch.optim.Optimizer): Optimizer for model training.
        scheduler (torch.optim.lr_scheduler or None): Learning rate scheduler, or None if not used.
        logger (Logger): Logger instance for visualizations and metric tracking.
        config (dict): Configuration dictionary with training parameters.
        device (torch.device): Device to run the training on.
    """
    best_val_loss = float('inf')
    best_val_l1 = float('inf')
    lambdas = config['loss'].get('lambdas', {})
    visualization_batches = config['training']['visualization_batches']
    epochs = config['training']['epochs']

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        # Training loop
        epoch_loss = _train_one_epoch(
            model, train_loader, calculate_loss, loss_fn, lambdas, optimizer, device
        )

        # Validation loop
        epoch_val_loss, epoch_val_l1 = _validate(
            model, valid_loader, calculate_loss, loss_fn, lambdas, device, logger, visualization_batches, epoch
        )
        print(f"Validation Loss: {epoch_val_loss:.6f}; L1: {epoch_val_l1:.6f}")

        # Check if both main metrics are better this time, save the model if so
        if epoch_val_loss < best_val_loss and epoch_val_l1 < best_val_l1:
            logger.save_best_model(model, epoch, metric_type=f"epoch_{epoch + 1}")

        # Check if we have a new best validation loss
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            logger.save_best_model(model, epoch, metric_type='loss')

        # Check if we have a new best validation L1
        if epoch_val_l1 < best_val_l1:
            best_val_l1 = epoch_val_l1
            logger.save_best_model(model, epoch, metric_type='l1')

        # Step the scheduler if applicable
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_val_loss)
            else:
                scheduler.step()

        # Log metrics with wandb if enabled
        logger.log_metrics(train_loss=epoch_loss,
                           val_loss=epoch_val_loss,
                           best_val_loss=best_val_loss,
                           val_l1=epoch_val_l1,
                           best_val_l1=best_val_l1)

        torch.cuda.empty_cache()

    print("Training completed!")
    logger.finish_wandb()
