"""Run training script for the Compact Source Removal project.

This script loads configuration settings,
initializes the training components,
and starts the training process.
"""

import argparse

import torch
import json5

from code.training.train import train
from code.training.logger import Logger
from code.losses.loss_factory import define_loss
from code.data.dataloader import define_dataloader
from code.models.model_factory import define_model
from code.utils import set_seed, define_optimizer_and_scheduler


def main():
    """Main function to parse arguments and initiate the training procedure."""
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the training config file"
    )
    args = parser.parse_args()

    print("---\tLoading configuration...")
    with open(args.config, "r", encoding='utf-8') as f:
        config = json5.load(f)
    print("---\tConfiguration loaded successfully.")

    train_procedure(config)


def train_procedure(config: dict):
    """Sets up the training environment and initiates the training loop.

    Args:
        config (dict): Configuration parameters from the JSON file.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"---\tUsing device: {device}")

    seed = config.get("random_seed", -1)
    if seed >= 0:
        set_seed(seed)
        print(f"---\tRandom seed set to: {seed}")
    else:
        print("---\tRandom seed not set, using randomization.")

    # Initialize the logger
    logger = Logger(config)
    print("---\tLogger initialized.")

    # Load data
    train_loader, valid_loader = define_dataloader(config)
    print("---\tData loaded successfully.")

    # Define the model and move it to the device
    model = define_model(config, device)
    print(f"---\tModel {config['model']['name']} defined and moved to device.")

    # Define the loss function and move it to the device
    loss_fn = define_loss(config, device)
    print(f"---\tLoss function {config['loss']['name']} defined.")

    # Define the optimizer and scheduler
    optimizer, scheduler = define_optimizer_and_scheduler(model, config)
    print("---\tOptimizer and scheduler set up successfully.")

    # Start training
    print("---\tStarting training...")
    train(model, train_loader, valid_loader, loss_fn, optimizer, scheduler, logger, config, device)
    print("---\tTraining completed successfully.")


if __name__ == "__main__":
    main()
