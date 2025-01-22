"""
This script prepares the data, runs the training, and saves the model.
"""

import argparse
import os
import sys
import pickle
import json
import logging
import pandas as pd
import time
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import List
import torch
from torch.utils.data import DataLoader, TensorDataset
# Comment this lines if you have problems with MLFlow installation
#import mlflow
from torch import nn
import torch.nn.init as init
import torch
from torch import nn
from copy import deepcopy

#mlflow.autolog()

# Adds the root directory to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))


# Change to CONF_FILE = "settings.json" if you have problems with env variables
CONF_FILE = os.getenv('CONF_PATH') 

from utils import get_project_dir, configure_logging

# Loads configuration settings from JSON
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Defines paths
DATA_DIR = get_project_dir(conf['general']['data_dir'])
MODEL_DIR = get_project_dir(conf['general']['models_dir'])
TRAIN_PATH = os.path.join(DATA_DIR, conf['train']['table_name'])

# Initializes parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--train_file", 
                    help="Specify inference data file", 
                    default=conf['train']['table_name'])
parser.add_argument("--model_path", 
                    help="Specify the path for the output model")

class DataProcessor():
    def __init__(self) -> None:
        pass

    def prepare_data(self, max_rows: int = None) -> pd.DataFrame:
        logging.info("Preparing data for training...")
        df = self.data_extraction(TRAIN_PATH)
        df = self.data_rand_sampling(df, max_rows)
        return df

    def data_extraction(self, path: str) -> pd.DataFrame:
        logging.info(f"Loading data from {path}...")
        return pd.read_csv(path)
    
    def data_rand_sampling(self, df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
        if not max_rows or max_rows < 0:
            logging.info('Max_rows not defined. Skipping sampling.')
        elif len(df) < max_rows:
            logging.info('Size of dataframe is less than max_rows. Skipping sampling.')
        else:
            df = df.sample(n=max_rows, replace=False, random_state=conf['general']['random_state'])
            logging.info(f'Random sampling performed. Sample size: {max_rows}')
        return df 
      
class IrissNN(nn.Module):
    def __init__(self):
        super(IrissNN, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
class Training():
    def __init__(self) -> None:
        self.model = IrissNN()
        self.best_model = None
        self.best_accuracy = 0.0

    def run_training(self, df: pd.DataFrame, out_path: str = None, test_size: float = 0.33, 
                 n_epochs: int = 50, batch_size: int = 32, learning_rate: float = 0.001) -> None:
        
        logging.info("Running training...")
        categorical_columns = ['species']
        df = self.label_encode(df, categorical_columns)
        X_train, X_test, y_train, y_test = self.data_split(df, test_size=test_size)
        X_train, X_test = self.scale_data(X_train, X_test)
        X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = self.convert_to_tensors(X_train, y_train, X_test, y_test)
        train_loader, test_loader = self.prepare_dataloader(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, batch_size)
        start_time = time.time()
        logging.info(train_loader)
        self.train(train_loader, test_loader, n_epochs, learning_rate)
        end_time = time.time()
        logging.info(f"Training completed in {end_time - start_time} seconds.")
        self.save(out_path)
    
    def label_encode(self, dataset: pd.DataFrame, columns: List[str]):
        logging.info("Labling categorical variables")
        label_encoders = {}
        for column in columns:
            label_encoder = LabelEncoder()
            dataset[column] = label_encoder.fit_transform(dataset[column])
            label_encoders[column] = label_encoder
        return dataset

   
    def data_split(self, df: pd.DataFrame, test_size: float = 0.33) -> tuple:
        logging.info("Splitting data into training and test sets...")
        feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        X = df[feature_names].values
        Y = df["species"].values
        return train_test_split(X, Y, test_size=test_size, 
                                random_state=conf['general']['random_state'])
    
    def scale_data(self, X_train, X_test) -> pd.DataFrame:
        logging.info("Scaling data")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test
    

    def convert_to_tensors(self, X_train, y_train, X_test, y_test):

        train_tensors = (
            torch.tensor(X_train, dtype=torch.float32),  # Features as float32
            torch.tensor(y_train, dtype=torch.long),     # Targets as long
        )
        test_tensors = (
            torch.tensor(X_test, dtype=torch.float32),   # Features as float32
            torch.tensor(y_test, dtype=torch.long),      # Targets as long
        )

        return *train_tensors, *test_tensors
   

    def prepare_dataloader(self, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, batch_size):
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader

    def train(self, train_dataloader, validation_dataloader, epochs: int, lr: float):
        """
        Train the model.

        Args:
            train_dataloader (DataLoader): DataLoader for training data.
            validation_dataloader (DataLoader): DataLoader for validation data.
            epochs (int): Number of training epochs.
            lr (float): Learning rate for the optimizer.
        """
        criterion = nn.CrossEntropyLoss()  
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0

            # Training loop
            for data, targets in train_dataloader:
               #data, targets = data.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                output = self.model(data).squeeze()
                loss = criterion(output, targets.long())
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Validation after each epoch
            validation_accuracy = self.evaluate_model(validation_dataloader)
            if validation_accuracy > self.best_accuracy:
                self.best_accuracy = validation_accuracy
                self.best_model = deepcopy(self.model)

            logging.info(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}, Validation Accuracy: {validation_accuracy:.4f}")

    def evaluate_model(self, validation_dataloader):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, targets in validation_dataloader:
                # data, targets = data.to(device), targets.to(device)

                # Forward Pass
                output = self.model(data)  # Output: (batch_size, num_classes)
                predictions = torch.argmax(output, dim=1)  # Predicted class indices

                # Compare predictions with targets
                correct += (predictions == targets).sum().item()
                total += targets.size(0)

        # Final Accuracy Calculation
        accuracy = correct / total if total > 0 else 0.0
        return accuracy
    
   
    def save(self, path: str) -> None:
        logging.info("Saving the model state dictionary...")
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

        if not path:
            # Generate a filename with the current timestamp
            path = os.path.join(MODEL_DIR, datetime.now().strftime(conf['general']['datetime_format']) + '.pth')

        # Save only the state dictionary
        torch.save(self.model.state_dict(), path)
        logging.info(f"Model state dictionary saved to {path}")

def main():
    configure_logging()

    data_proc = DataProcessor()
    tr = Training()

    df = data_proc.prepare_data(max_rows=conf['train']['data_sample'])
    tr.run_training(df, test_size=conf['train']['test_size'], n_epochs = conf["train"]["n_epochs"], batch_size = conf["train"]["batch_size"], learning_rate = conf["train"]["learning_rate"])


if __name__ == "__main__":
    main()