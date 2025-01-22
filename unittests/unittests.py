import unittest
import pandas as pd
import os
import sys
import json
from sklearn.datasets import load_iris

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))
CONF_FILE = os.getenv('CONF_PATH')

from training.train import DataProcessor, Training


class TestDataProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if CONF_FILE:
            with open(CONF_FILE, "r") as file:
                conf = json.load(file)
            cls.data_dir = conf['general']['data_dir']
            cls.train_path = os.path.join(cls.data_dir, conf['train']['table_name'])
        else:
            cls.train_path = "path_to_iris_data.csv"  # Replace with the actual path if needed.

    def test_data_extraction(self):
        dp = DataProcessor()
        df = dp.data_extraction(self.train_path)
        self.assertIsInstance(df, pd.DataFrame)

    def test_prepare_data(self):
        dp = DataProcessor()
        df = dp.prepare_data(150)  # Iris dataset has 150 rows
        self.assertEqual(df.shape[0], 150)

class TestTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the Iris dataset for testing
        iris = load_iris()
        iris.feature_names =  ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        cls.df = pd.DataFrame(iris.data, columns=iris.feature_names)
        cls.df['species'] = iris.target  # Add the target column

        # Define training parameters
        cls.test_size = 0.33
        cls.n_epochs = 2  # Use a small number for quick testing
        cls.batch_size = 16
        cls.learning_rate = 0.001
        cls.output_path = "test_model.pth"  # Temporary model save path

    def test_run_training(self):
        trainer = Training()

        # Run the training pipeline
        trainer.run_training(
            df=self.df,
            out_path=self.output_path,
            test_size=self.test_size,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
        )

        # Check that the best model is saved
        self.assertIsNotNone(trainer.best_model)
        self.assertGreater(trainer.best_accuracy, 0.0)
        self.assertTrue(os.path.exists(self.output_path))

    def test_data_split(self):
        trainer = Training()
        X_train, X_test, y_train, y_test = trainer.data_split(self.df, test_size=self.test_size)

        # Check shapes of the splits
        self.assertEqual(len(X_train) + len(X_test), len(self.df))
        self.assertEqual(len(y_train) + len(y_test), len(self.df))
        self.assertAlmostEqual(len(X_test) / len(self.df), self.test_size, delta=0.05)

    def test_scale_data(self):
        trainer = Training()
        X_train, X_test, y_train, y_test = trainer.data_split(self.df, test_size=self.test_size)
        X_train_scaled, X_test_scaled = trainer.scale_data(X_train, X_test)

        # Check scaling
        self.assertAlmostEqual(X_train_scaled.mean(), 0, delta=0.1)
        self.assertAlmostEqual(X_train_scaled.std(), 1, delta=0.1)

    def test_prepare_dataloader(self):
        trainer = Training()
        X_train, X_test, y_train, y_test = trainer.data_split(self.df, test_size=self.test_size)
        X_train_scaled, X_test_scaled = trainer.scale_data(X_train, X_test)
        train_tensors = trainer.convert_to_tensors(X_train_scaled, y_train, X_test_scaled, y_test)

        train_loader, test_loader = trainer.prepare_dataloader(*train_tensors, self.batch_size)

        # Check batch sizes
        for data, targets in train_loader:
            self.assertEqual(data.shape[0], self.batch_size)
            self.assertEqual(data.shape[1], 4)  # 4 features in Iris dataset
            self.assertEqual(targets.shape[0], self.batch_size)
            break  # Test only the first batch

    def tearDown(self):
        # Clean up the saved model file
        if os.path.exists(self.output_path):
            os.remove(self.output_path)


if __name__ == '__main__':
    unittest.main()


