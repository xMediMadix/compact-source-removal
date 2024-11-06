"""
Evaluation output management for Compact Source Removal.

This module defines the `EvaluationOutputManager` class, which handles saving predictions,
photometry results, and configuration details. It also manages the evaluation folder
structure and code backup to preserve the state of the experiment.
"""

import os
import shutil
from datetime import datetime
import numpy as np
import pandas as pd


class EvaluationOutputManager:
    """Manager class to handle evaluation outputs including predictions, metrics, and code backup."""

    def __init__(self, config: dict):
        """
        Initialize the EvaluationOutputManager, set up the evaluation folder structure, and handle configuration backup.

        Args:
            config (dict): Configuration parameters for the evaluation.
        """
        self.config = config
        self.timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.experiment_name = f"{config['experiment']}+{self.timestamp}"
        self.model_name = config['model']['name']

        # Define paths
        self.evaluation_folder = os.path.join('./evaluation_output', self.model_name, self.experiment_name)
        self.code_dir = os.path.join(self.evaluation_folder, 'code')
        self.predictions_dir = None
        self.results_dir = None

        # Create folder structure
        self._create_folders()

        # Copy code and config files
        self._copy_code()
        self._copy_config()

    def _create_folders(self):
        """Create the necessary folder structure for evaluation."""
        os.makedirs(self.evaluation_folder, exist_ok=True)
        os.makedirs(self.code_dir, exist_ok=True)

        # Create predictions folder if 'save_predictions' is specified
        if 'save_predictions' in self.config['evaluation_functionalities']:
            self.predictions_dir = os.path.join(self.evaluation_folder, 'predictions')
            os.makedirs(self.predictions_dir, exist_ok=True)

        # Create results folder if photometry is defined
        if 'photometry' in self.config['evaluation_functionalities']:
            self.results_dir = os.path.join(self.evaluation_folder, 'results')
            os.makedirs(self.results_dir, exist_ok=True)

    def _copy_code(self):
        """Copy the entire code directory to the evaluation folder."""
        shutil.copytree('./code', self.code_dir, dirs_exist_ok=True)

    def _copy_config(self):
        """Copy the configuration directory to the evaluation folder."""
        config_dir = os.path.join(self.evaluation_folder, "config")
        shutil.copytree('./config', config_dir, dirs_exist_ok=True)

    def save_prediction(self, original_hdulist, prediction_hdulist, file_name: str):
        """
        Save both the original and modified FITS files if 'save_predictions' is in functionalities.
        Adds a '_pred' suffix to the prediction file for differentiation.

        Args:
            original_hdulist (fits.HDUList): HDUList object containing the original FITS image data.
            prediction_hdulist (fits.HDUList): HDUList object containing the modified FITS image data.
            file_name (str): The base name of the file to save.
        """
        if self.predictions_dir:
            # Save the original file
            original_file_path = os.path.join(self.predictions_dir, file_name)
            original_hdulist.writeto(original_file_path, overwrite=True)
            print(f"---\tSaved original FITS file to {original_file_path}")

            # Save the prediction file with a '_pred' suffix
            prediction_file_name = file_name.replace('.fits', '_pred.fits')
            prediction_file_path = os.path.join(self.predictions_dir, prediction_file_name)
            prediction_hdulist.writeto(prediction_file_path, overwrite=True)
            print(f"---\tSaved prediction FITS file to {prediction_file_path}")

    def save_photometry_results(self, all_flux_results: list) -> None:
        """
        Save the detailed photometry results to a CSV file, excluding columns as needed in real_data mode.

        Args:
            all_flux_results (list): List of photometry results for all sources.
        """
        if 'photometry' in self.config['evaluation_functionalities']:
            # Check if simulation mode is on to determine columns
            simulation_mode = self.config['mode'] == 'simulation'

            # Initialize lists to accumulate detailed data for CSV
            filename_list = []
            ra_list = []
            dec_list = []
            predicted_flux_list = []
            standard_flux_list = []
            apertures_list = []
            if simulation_mode:
                true_flux_list = []

            # Flatten results to collect RA, DEC, and fluxes
            for res in all_flux_results:
                filename_list.append(res['Filename'])
                ra_list.append(res['RA'])
                dec_list.append(res['DEC'])
                predicted_flux_list.extend(res['predicted_flux'])
                standard_flux_list.extend(res['standard_flux'])
                apertures_list.extend(self.config['data']['aperture_sizes'])
                if simulation_mode:
                    true_flux_list.extend(res['true_flux'])

            # Create DataFrame based on mode
            if simulation_mode:
                detailed_results_df = pd.DataFrame({
                    'Filename': np.repeat(filename_list, len(self.config['data']['aperture_sizes'])),
                    'RA': np.repeat(ra_list, len(self.config['data']['aperture_sizes'])),
                    'DEC': np.repeat(dec_list, len(self.config['data']['aperture_sizes'])),
                    'Aperture Size (arcsec)': apertures_list,
                    'True Flux (mJy)': true_flux_list,
                    'Predicted Flux (mJy)': predicted_flux_list,
                    'Standard Flux (mJy)': standard_flux_list
                })
            else:
                detailed_results_df = pd.DataFrame({
                    'Filename': np.repeat(filename_list, len(self.config['data']['aperture_sizes'])),
                    'RA': np.repeat(ra_list, len(self.config['data']['aperture_sizes'])),
                    'DEC': np.repeat(dec_list, len(self.config['data']['aperture_sizes'])),
                    'Aperture Size (arcsec)': apertures_list,
                    'Predicted Flux (mJy)': predicted_flux_list,
                    'Standard Flux (mJy)': standard_flux_list
                })

            # Save detailed results to CSV
            detailed_results_file_path = os.path.join(self.results_dir, 'detailed_photometry_results.csv')
            detailed_results_df.to_csv(detailed_results_file_path, index=False)
            print(f"---\tSaved detailed photometry results to {detailed_results_file_path}")
