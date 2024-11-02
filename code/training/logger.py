"""
This module defines the Logger class, which manages logging,
experiment directory structure, visualization, and integration with WandB.
"""

import os
import json
import shutil
from datetime import datetime

import cv2
import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval

# Set environment variable
os.environ["WANDB_SILENT"] = "true"


class Logger:
    """Logger class to manage experiment logging, directory setup, and wandb integration."""

    def __init__(self, config):
        """
        Initialize the logger, set up the experiment folder, and handle wandb setup.

        Args:
            config (dict): Configuration parameters.
        """
        self.config = config
        self.timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.experiment_name = f"{config['experiment']}+{self.timestamp}"
        self.model_name = config['model']['name']

        # Define the paths
        self.experiment_folder = os.path.join('./experiments', self.model_name, self.experiment_name)
        self.trained_model_dir = os.path.join(self.experiment_folder, 'trained_model')
        self.visualization_dir = os.path.join(self.experiment_folder, 'visualization')
        self.code_dir = os.path.join(self.experiment_folder, 'code')

        # Create the folder structure
        self._create_folders()

        # Copy the code and configuration to the experiment folder
        self._copy_code()
        self._copy_config()

        # Initialize wandb if enabled in the configuration
        self.wandb_run = self._init_wandb()

    def _create_folders(self):
        """Create the necessary folder structure for the experiment."""
        os.makedirs(self.experiment_folder, exist_ok=True)
        os.makedirs(os.path.join(self.experiment_folder, 'code'), exist_ok=True)
        os.makedirs(self.trained_model_dir, exist_ok=True)
        os.makedirs(self.visualization_dir, exist_ok=True)

    def _copy_code(self):
        """Copy the entire code directory to the experiment folder."""
        shutil.copytree('./code', self.code_dir, dirs_exist_ok=True)

    def _copy_config(self):
        """Copy the configuration file to the experiment folder."""
        config_path = os.path.join(self.experiment_folder, "config.json")
        with open(config_path, 'w', encoding='utf-8') as config_file:
            json.dump(self.config, config_file, indent=4)

    def _init_wandb(self):
        """Initialize wandb if it's enabled in the configuration."""
        # Log in to wandb
        wandb.login()

        # Add requirement for wandb core
        wandb.require("core")

        if self.config['logging'].get('log_to_wandb', False):
            tags = [
                self.config['model']['name'],
                self.config['optimization']['optimizer']['type'],
                self.config['optimization']['scheduler']['type'],
                self.config['loss']['name'],
                self.config['data']['dataset_name'],
                self.config['data']['mask_type']
            ]

            return wandb.init(
                project=self.config['logging']['wandb_project'],
                name=self.experiment_name,
                tags=tags,
                config=self.config,
                settings=wandb.Settings(_disable_stats=True, _disable_meta=True)
            )

        return None

    def visualize(self, inputs, masks, outputs, targets, epoch, batch_index):
        """Visualize and save model predictions for a batch.

        Args:
            inputs (torch.Tensor): Batch of input images.
            masks (torch.Tensor or None): Batch of masks, or None.
            outputs (torch.Tensor): Model predictions for the batch.
            targets (torch.Tensor): Ground truth images for the batch.
            epoch (int): Current epoch number, used for file naming.
            batch_index (int): Current batch index, used for file naming.
        """
        masked_inputs = inputs[:, 0:1, :, :]

        if masks is not None:
            masked_outputs = masks[:, 0:1, :, :] * inputs[:, 0:1, :, :] + (1 - masks[:, 0:1, :, :]) * outputs
            num_cols = 6  # 6 columns: input, mask, target, masked output, smoother output, raw output
        else:
            masked_outputs = None
            num_cols = 3  # 3 columns: input, target, raw output

        plt.figure(figsize=(3 * num_cols, 25))

        for j in range(min(8, inputs.shape[0])):  # Handles cases where batch size < 8
            # Calculate Z-scale limits separately for each input image
            interval = ZScaleInterval()
            vmin, vmax = interval.get_limits(masked_inputs[j].squeeze().cpu().numpy())

            # First Column: Input
            plt.subplot(8, num_cols, num_cols * j + 1)
            plt.imshow(masked_inputs[j].squeeze().cpu().numpy(), cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
            plt.title('Input')
            plt.axis('off')

            # Second Column: Mask (if available)
            if masks is not None:
                plt.subplot(8, num_cols, num_cols * j + 2)
                plt.imshow(masks[j].squeeze().cpu().numpy(), cmap='gray', origin='lower')
                plt.title('Mask')
                plt.axis('off')

            # Third Column: Target
            plt.subplot(8, num_cols, num_cols * j + (3 if masks is not None else 2))
            plt.imshow(targets[j].squeeze().cpu().numpy(), cmap='gray', origin='lower')
            plt.title('Target')
            plt.axis('off')

            # Fourth Column: Masked Output (if available)
            if masked_outputs is not None:
                plt.subplot(8, num_cols, num_cols * j + 4)
                plt.imshow(masked_outputs[j].detach().squeeze().cpu().numpy(), cmap='gray', origin='lower')
                plt.title('Masked Output')
                plt.axis('off')

            # Fifth Column: Smoother Masked Output (only if masks are available)
            if masks is not None:
                # Create the smoother mask using OpenCV operations
                kernel_size = 5
                eroded_mask = cv2.erode(
                    masks[j].squeeze().cpu().numpy().astype(np.uint8),
                    np.ones((kernel_size, kernel_size), np.uint8), iterations=1
                )
                blurred_mask = cv2.GaussianBlur(eroded_mask.astype(np.float32), (5, 5), 5)
                smooth_mask = np.clip(blurred_mask, 0, 1)

                # Blend input and output using the smoother mask
                smoother_output = smooth_mask * inputs[j, 0:1, :, :].cpu().numpy() + \
                                  (1 - smooth_mask) * outputs[j].detach().cpu().numpy()

                plt.subplot(8, num_cols, num_cols * j + 5)
                plt.imshow(smoother_output.squeeze(), cmap='gray', origin='lower')
                plt.title('Smoother Output')
                plt.axis('off')

            # Last Column: Raw Output
            last_col_index = num_cols * j + (num_cols if masks is not None else 3)
            plt.subplot(8, num_cols, last_col_index)
            plt.imshow(outputs[j].detach().squeeze().cpu().numpy(), cmap='gray', origin='lower')
            plt.title('Raw Output')
            plt.axis('off')

        plt.tight_layout()
        vis_save_path = os.path.join(self.visualization_dir, str(batch_index))
        os.makedirs(vis_save_path, exist_ok=True)
        plt.savefig(os.path.join(vis_save_path, f"{epoch + 1}_{batch_index}.png"))
        plt.close()

    def save_best_model(self, model, epoch, metric_type='loss'):
        """Save the best model based on validation metrics.

        Args:
            model (torch.nn.Module): The model to save.
            epoch (int): The current epoch number.
            metric_type (str): The type of metric that triggered the save ('loss' or 'l1').
        """
        model_path = os.path.join(self.trained_model_dir, f"best_model_{metric_type}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"New Best Validation {metric_type.capitalize()}: Saving model at epoch {epoch + 1}.")

    def log_metrics(self, train_loss, val_loss, best_val_loss, val_l1, best_val_l1):
        """Log metrics to wandb if enabled.

        Args:
            train_loss (float): The training loss for the current epoch.
            val_loss (float): The validation loss for the current epoch.
            best_val_loss (float): The best validation loss observed so far.
            val_l1 (float): The validation L1 loss for the current epoch.
            best_val_l1 (float): The best validation L1 loss observed so far.
        """
        if self.wandb_run is not None:
            log_data = {
                'train/loss': train_loss,
                'valid/loss': val_loss,
                'valid/best_loss': best_val_loss,
                'valid/l1': val_l1,
                'valid/best_l1': best_val_l1
            }
            self.wandb_run.log(log_data)

    def finish_wandb(self):
        """Finish the wandb run if logging to wandb."""
        if self.wandb_run is not None:
            self.wandb_run.finish()
