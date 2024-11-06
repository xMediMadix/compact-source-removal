"""
Evaluation functions for the Compact Source Removal project.

This module includes functions to evaluate a model on simulated or real data for source removal.
It handles loading datasets, applying photometry corrections, and saving evaluation results.

Main Functions:
    - evaluate_model: Main function for evaluating the model on test data.
    - evaluate_single_fits: Processes a single FITS file to replace compact sources with model predictions.
    - perform_aperture_photometry: Executes aperture photometry and background correction.
"""

import os
import re
import copy
import glob
import json
import warnings
from typing import Optional

import numpy as np
import torch
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from photutils.aperture import (
    CircularAperture, aperture_photometry, CircularAnnulus, ApertureStats
)
from photutils.centroids import centroid_quadratic
from code.data.dataloader import define_evaluation_dataloader
from code.evaluation.evaluation_output import EvaluationOutputManager

# Suppress specific warnings from photutils (e.g., DeprecationWarnings)
warnings.filterwarnings("ignore", category=DeprecationWarning, module='photutils.centroids.core')


def extract_field_name(csv_file: str) -> str:
    """
    Extract the field name from the CSV file name.

    Args:
        csv_file (str): Path to the CSV file.

    Returns:
        str: Field name extracted from the file name, or an empty string if no match is found.
    """
    pattern = r'^(.*?)_sim(\d+)\.csv$'
    match = re.search(pattern, os.path.basename(csv_file))

    if match:
        field_name = match.group(1)
        return field_name
    return ''


def evaluate_model(config: dict, model: torch.nn.Module, device: torch.device, eval_manager: EvaluationOutputManager):
    """
    Main function to evaluate the model on test data.

    Args:
        config (dict): Configuration parameters from the JSON file.
        model (torch.nn.Module): The trained model for evaluation.
        device (torch.device): Device to perform computations on.
        eval_manager (EvaluationOutputManager): Output manager for handling evaluation output and results.
    """
    # Get the test data path from config
    data_path = config['data']['test_data_dir']
    # Default to 'real_data' if mode not specified
    mode = config.get('mode', 'real_data')
    # Get the path to the original maps if given
    original_maps = config['data'].get('original_maps_path', '')

    # Find all FITS files in the test data path
    fits_files = glob.glob(os.path.join(data_path, '*.fits'))
    total_files = len(fits_files)
    print(f"---\t{total_files} FITS files found in {data_path}.")

    # To accumulate photometry results for all FITS files
    all_flux_results = []

    # Process each FITS file
    for file_idx, fits_file in enumerate(fits_files):
        print(f"\nProcessing file {file_idx + 1}/{total_files}: {os.path.basename(fits_file)}")

        # Replace .fits extension with .csv to get the corresponding CSV file path
        csv_file = fits_file.replace('.fits', '.csv')

        # If in simulation mode, extract field name and load original map
        if mode == "simulation":
            # Extract the field name and simulation number from the CSV file name
            field_name = extract_field_name(csv_file)
            if not field_name:
                print(f"---\tWarning: Could not extract field information from {csv_file}. Skipping file.")
                continue

            print(f"---\tProcessing field: {field_name}")

            # Define path to the original map
            original_map = os.path.join(original_maps, f"{field_name}.fits")
        else:
            original_map = None  # No original map needed for real_data mode

        # Define the dataloader for this FITS file, passing None for original_map if not in simulation mode
        data_loader = define_evaluation_dataloader(config, fits_file, csv_file, original_map)

        # Evaluate this FITS file and collect photometry results
        flux_results = evaluate_single_fits(
            model,
            data_loader,
            device,
            eval_manager,
            config,
            os.path.basename(fits_file)
        )

        # Append the results if photometry was performed
        if flux_results:
            all_flux_results.extend(flux_results)  # Extend because each FITS file could have multiple results

        print(f"---\tCompleted processing for {fits_file}")

        # Save all photometry results after all files are processed
    if all_flux_results:
        eval_manager.save_photometry_results(all_flux_results)


def load_aperture_corrections(config, apertures):
    """
    Load and validate aperture correction values for the specified observing mode and band.

    Args:
        config (dict): Configuration dictionary that contains mode and observing parameters.
        apertures (list or np.array): List of aperture sizes to validate against the correction file.

    Returns:
        list: A list of aperture correction values corresponding to the specified apertures.

    Raises:
        ValueError: If an aperture size is missing in the correction file.
    """
    # Determine the mode and band for aperture correction
    mode = config["observing_parameters"]["observing_mode"]
    band = config["observing_parameters"]["band"]

    # Select the appropriate correction file based on simulation or real data mode
    aperture_correction_file = os.path.join(
        "config", "parameters",
        "vesta_aperture_correction.json" if config['mode'] == 'simulation' else "eef_aperture_correction.json"
    )

    # Load the correction data from the specified file
    with open(aperture_correction_file, 'r', encoding='utf-8') as f:
        correction_data = json.load(f)

    # Retrieve the corrections for the current mode and band
    correction_apertures = correction_data[mode][band]["apertures"]
    aperture_corrections = []

    # Validate each aperture size and collect its correction value
    for aperture in apertures:
        if str(aperture) in correction_apertures:
            aperture_corrections.append(correction_apertures[str(aperture)])
        else:
            raise ValueError(f"Aperture size {aperture} has no correction value defined in {aperture_correction_file}.")

    return aperture_corrections


def evaluate_single_fits(
        model: torch.nn.Module,
        data_loader,
        device: torch.device,
        eval_manager: EvaluationOutputManager,
        config: dict,
        filename: str
):
    """
    Evaluates a single FITS file by running inference for each source, replacing the source with the model prediction.

    Args:
        model (torch.nn.Module): The trained model to use for inference.
        data_loader (torch.utils.data.DataLoader): DataLoader for the current FITS file.
        device (torch.device): Device to perform computations on.
        eval_manager (EvaluationOutputManager): Output manager for handling evaluation output and results.
        config (dict): Configuration dictionary that contains aperture photometry settings.
        filename (str): The name of the FITS file being processed.

    Returns:
        list: A list of photometry results for each source in the FITS file.
    """
    # Switch model to evaluation mode
    model.eval()

    # Access the dataset from the DataLoader to get the original FITS image
    dataset = data_loader.dataset
    original_hdulist = dataset.get_original_fits_image()  # Get the original HDUList
    original_fits_image = original_hdulist['image'].data  # Extract the original image data
    # Duplicate the original HDUList to preserve the original state
    original_file_to_save = copy.deepcopy(original_hdulist)

    # Check if photometry is required
    perform_photometry = 'photometry' in config['evaluation_functionalities']
    apertures = np.array(config['data']['aperture_sizes'])

    # Load and validate aperture corrections
    aperture_corrections = load_aperture_corrections(config, apertures)

    # Determine if we should refine the center based on the config
    use_center_refinement = config.get("center_refinement", False)

    # To store photometry results for all sources in this FITS file
    all_flux_results = []

    # Iterate through the sources in the current FITS file
    with torch.no_grad():
        for (ra, dec), (input_data, target_data, mask_data, mask_roi), (min_val, max_val), _ in data_loader:
            expected_size = (1, 1, config['data']['input_size'][0], config['data']['input_size'][1])
            if input_data.shape != expected_size or mask_data.shape != expected_size:
                print(f"Skipping source at Ra: {ra}, Dec: {dec} due to shape mismatch.")
                continue

            # Extract values
            min_val, max_val = min_val.item(), max_val.item()

            # Move data to the device (GPU/CPU)
            input_data, mask_data = input_data.to(device), mask_data.to(device)

            # Forward pass: get model predictions
            outputs = model(input_data, mask_data)

            # Convert ra, dec to pixel coordinates
            coord = SkyCoord(ra * u.deg, dec * u.deg)
            pixel_cutout = Cutout2D(data=original_fits_image, position=coord, size=dataset.crop_size, wcs=dataset.wcs,
                                    copy=False)

            # Denormalize the predicted output
            predicted_data = outputs.squeeze().cpu().numpy()
            predicted_data = dataset.denormalize(predicted_data, min_val, max_val)

            # Blend the predicted output with the original image using mask_roi
            blended_cutout = ((1 - mask_roi) * predicted_data) + (mask_roi * pixel_cutout.data)

            # Replace the region in the original FITS image with the blended cutout
            original_fits_image[pixel_cutout.slices_original] = blended_cutout

            # Perform photometry if defined
            if perform_photometry:
                # Prepare denormalized data for photometry
                denormalized_input = dataset.denormalize(input_data.squeeze().cpu().numpy(), min_val, max_val)
                denormalized_output = blended_cutout.squeeze().cpu().numpy()
                denormalized_target = dataset.denormalize(target_data.squeeze().cpu().numpy(), min_val,
                                                          max_val) if target_data is not None else None

                # Refine center based on mask if enabled
                mask_for_recentering = mask_roi.squeeze().cpu().numpy()
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", module="photutils.centroids.core")
                    refined_center = refine_center(denormalized_input,
                                                   mask_for_recentering) if use_center_refinement else None

                flux_results = perform_aperture_photometry(
                    denormalized_input,
                    denormalized_target,
                    denormalized_output,
                    dataset,
                    apertures,
                    aperture_corrections,
                    refined_center,
                    ra.item(),
                    dec.item(),
                    filename
                )
                all_flux_results.append(flux_results)

    # Save the modified FITS file if saving is enabled
    if "save_predictions" in config['evaluation_functionalities']:
        eval_manager.save_prediction(original_file_to_save, original_hdulist, os.path.basename(dataset.fits_file))

    # Return photometry results for this FITS file
    return all_flux_results


def refine_center(input_data: np.ndarray, mask: np.ndarray) -> tuple:
    """
    Refine the center of a source region based on a masked area in the input image.

    Args:
        input_data (np.ndarray): The input image data containing the source.
        mask (np.ndarray): A binary mask.

    Returns:
        tuple: The refined (x, y) coordinates of the source center.
    """
    # Apply the mask to the input data to focus on the source region
    masked_input_data = input_data * mask

    # Attempt to refine the center using the masked data
    refined_center = centroid_quadratic(masked_input_data)

    # Fallback to the default center if the refined center contains NaNs
    if np.isnan(refined_center).any():
        center_x, center_y = input_data.shape[1] // 2, input_data.shape[0] // 2
    else:
        center_x, center_y = refined_center

    return center_x, center_y


def perform_aperture_photometry(
        input_data: np.ndarray,
        target_data: Optional[np.ndarray],
        predicted_data: np.ndarray,
        dataset,
        apertures: np.ndarray,
        apcorr: np.ndarray,
        refined_center: Optional[tuple],
        ra: float,
        dec: float,
        filename: str
) -> dict:
    """
    Perform aperture photometry on both the ground truth (input_data - target_data) and the predicted data.
    Pixel center is dynamically calculated based on the shape of input_data.

    Args:
        input_data (np.ndarray): Input image data.
        target_data (Optional[np.ndarray]): Target (ground truth) image data, or None if not in simulation mode.
        predicted_data (np.ndarray): Model predicted image data.
        dataset: The dataset object providing WCS and other metadata.
        apertures (np.ndarray): Array of aperture sizes in arcseconds.
        apcorr (np.ndarray): Array of aperture corrections.
        refined_center (Optional[tuple]): Refined center coordinates, or None if not using center refinement.
        ra (float): Right Ascension of the source.
        dec (float): Declination of the source.
        filename (str): The filename of the current FITS file.

    Returns:
        dict: A dictionary containing the photometry results.
    """

    # Default to the center if refined_center is not provided
    center_x, center_y = refined_center if refined_center else (input_data.shape[1] // 2, input_data.shape[0] // 2)

    # Convert aperture sizes to pixel units
    wcs = dataset.wcs
    ps = np.abs(dataset.get_original_fits_image()['image'].header['CDELT1']) * 3600.0  # arcseconds per pixel
    radii_in_pixels = np.array(apertures) / ps
    apertures_pixel = [CircularAperture((center_x, center_y), r=r) for r in radii_in_pixels]

    # Perform photometry on input - target and predicted data
    true_flux = aperture_photometry(input_data - target_data, apertures_pixel, method='subpixel',
                                    wcs=wcs) if target_data is not None else None
    predicted_flux = aperture_photometry(input_data - predicted_data, apertures_pixel, method='subpixel', wcs=wcs)

    # Perform standard photometry on input data without target subtraction (for comparison)
    standard_photometry = aperture_photometry(input_data, apertures_pixel, method='subpixel', wcs=wcs)

    # Calculate the background using a circular annulus
    annulus_aperture = CircularAnnulus((center_x, center_y), r_in=25 / ps, r_out=35 / ps)
    annulus_stats = ApertureStats(input_data, annulus_aperture)

    # Background mean and area calculation
    bkg_mean = annulus_stats.mean
    areas = np.pi * (radii_in_pixels ** 2)  # Vector of areas for each aperture in pixels

    # Extract the aperture sums by matching the column names dynamically
    true_flux_sums = np.array(
        [true_flux[f'aperture_sum_{i}'][0] for i in range(len(apertures))]) if true_flux is not None else None
    predicted_flux_sums = np.array([predicted_flux[f'aperture_sum_{i}'][0] for i in range(len(apertures))])
    standard_flux_sums = np.array([standard_photometry[f'aperture_sum_{i}'][0] for i in range(len(apertures))])

    # Perform background subtraction for standard method
    bkg_contributions = bkg_mean * areas
    standard_flux_corrected = standard_flux_sums - bkg_contributions

    # Collect flux results, apply aperture corrections
    results = {
        'Filename': filename,
        'RA': ra,
        'DEC': dec,
        'true_flux': (true_flux_sums / apcorr) * 1000.0 if true_flux_sums is not None else None,  # Convert to mJy
        'predicted_flux': (predicted_flux_sums / apcorr) * 1000.0,  # Convert to mJy
        'standard_flux': (standard_flux_corrected / apcorr) * 1000.0,  # Convert to mJy
    }

    return results
