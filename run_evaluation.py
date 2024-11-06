"""
Main entry point for the compact source removal evaluation on simulated or real Herschel data.

This script loads a configuration file to set up the environment, model, and evaluation parameters,
then initiates the source removal and background estimation process.
Users can specify settings like observing mode, band, and photometry parameters.
Results, including source-free images and photometric data, are exported based on the configuration.

Example:
    python run_evaluation.py --config config/evaluation_simulation.json
    python run_evaluation.py --config config/evaluation_real_data.json

"""

import argparse
import json5
import torch

from code.evaluation.evaluate import evaluate_model
from code.models.model_factory import define_model
from code.evaluation.evaluation_output import EvaluationOutputManager


def main():
    """Main function to parse arguments and initiate the evaluation procedure."""
    parser = argparse.ArgumentParser(description="Evaluation Script")

    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the evaluation config file"
    )

    args = parser.parse_args()

    print("---\tLoading configuration...")
    with open(args.config, "r", encoding="utf-8") as f:
        config = json5.load(f)
    print("---\tConfiguration loaded successfully.")

    evaluate_procedure(config)


def evaluate_procedure(config: dict):
    """Sets up the evaluation environment and initiates the evaluation loop.

    Args:
        config (dict): Configuration parameters from the JSON file.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"---\tUsing device: {device}")

    # Initialize the EvaluationOutputManager
    eval_manager = EvaluationOutputManager(config)
    print("---\tEvaluation Output Manager initialized.")

    # Define the model and move it to the device
    model = define_model(config, device)
    print(f"---\tModel {config['model']['name']} defined and moved to device.")

    # Load the trained model weights from the configuration
    weights_path = config['model'].get('weights_path', None)
    if weights_path is None:
        raise ValueError("Weights path not specified in the configuration file.")

    model.load_state_dict(torch.load(weights_path, map_location=device))
    print(f"---\tTrained weights loaded from {weights_path}.")

    # Start evaluation (placeholder for now)
    print("---\tStarting evaluation...")
    evaluate_model(config, model, device, eval_manager)
    print("---\tEvaluation completed successfully.")


if __name__ == "__main__":
    main()
