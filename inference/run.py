"""
Script loads the latest trained model, data for inference and predicts results.
Imports necessary packages and modules.
"""

import argparse
import json
import logging
import os
import pickle
import sys
from datetime import datetime
from typing import List
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import torch
import re
# Adds the root directory to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

from training.train import IrissNN

# Change to CONF_FILE = "settings.json" if you have problems with env variables
CONF_FILE = os.getenv('CONF_PATH')

from utils import get_project_dir, configure_logging

# Loads configuration settings from JSON
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Defines paths
DATA_DIR = get_project_dir(conf['general']['data_dir'])
MODEL_DIR = get_project_dir(conf['general']['models_dir'])
RESULTS_DIR = get_project_dir(conf['general']['results_dir'])

# Initializes parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--infer_file", 
                    help="Specify inference data file", 
                    default=conf['inference']['inp_table_name'])
parser.add_argument("--out_path", 
                    help="Specify the path to the output table")


def get_latest_model_path() -> str:
    """Gets the path of the latest saved model"""
    latest = None
    for (dirpath, dirnames, filenames) in os.walk(MODEL_DIR):
        for filename in filenames:
            if not latest or datetime.strptime(latest, conf['general']['datetime_format'] + '.pth') < \
                    datetime.strptime(filename, conf['general']['datetime_format'] + '.pth'):
                latest = filename
    return os.path.join(MODEL_DIR, latest)

def get_model_by_path(path: str):
    """Loads and returns the specified model"""
    try:
        # Initialize the model architecture
        model = IrissNN()

        # Load the state dictionary
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        
        # Set the model to evaluation mode
        model.eval()

        logging.info(f"Model successfully loaded from: {path}")
        return model
    except Exception as e:
        logging.error(f"An error occurred while loading the model: {e}")
        sys.exit(1)


def get_inference_data(path: str) -> pd.DataFrame:
    """loads and returns data for inference from the specified csv file"""
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        logging.error(f"An error occurred while loading inference data: {e}")
        sys.exit(1)


def predict_results(model, infer_data: pd.DataFrame) -> pd.DataFrame:
    """
    Predict the results using a PyTorch model and join them with the input DataFrame.

    Args:
        model (nn.Module): The PyTorch model.
        infer_data (pd.DataFrame): Input data as a Pandas DataFrame.

    Returns:
        pd.DataFrame: Input DataFrame with predictions added as a new column.
    """
    model.eval()  # Ensure the model is in evaluation mode

    # Convert DataFrame to a PyTorch tensor
    infer_tensor = torch.tensor(infer_data.values, dtype=torch.float32)

    # Perform inference
    with torch.no_grad():  # Disable gradient computation
        outputs = model(infer_tensor)  # Forward pass
        predictions = torch.argmax(outputs, dim=1).numpy()  # Convert logits to class predictions

    # Add predictions to the DataFrame
    infer_data['results'] = predictions
    return infer_data


def store_results(results: pd.DataFrame, path: str = None) -> None:
    """Store the prediction results in 'results' directory with current datetime as a filename"""
    if not path:
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
        path = datetime.now().strftime(conf['general']['datetime_format']) + '.csv'
        path = os.path.join(RESULTS_DIR, path)
    pd.DataFrame(results).to_csv(path, index=False)
    logging.info(f'Results saved to {path}')


def main():
    """Main function"""
    configure_logging()
    args = parser.parse_args()
    print(get_latest_model_path())
    model = get_latest_model_path()
    #"22.01.2025_01.18.pth"
    model_path = os.path.join(MODEL_DIR, model)

    model = get_model_by_path(model_path)
    infer_file = args.infer_file
    infer_data = get_inference_data(os.path.join(DATA_DIR, infer_file))
    results = predict_results(model, infer_data)
    store_results(results, args.out_path)

    logging.info(f'Prediction results: {results}')


if __name__ == "__main__":
    main()