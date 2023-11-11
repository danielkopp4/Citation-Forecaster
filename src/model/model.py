import os
import json
import numpy as np
import tensorflow as tf
from src.data_extraction.dataset_api import Dataset
from src.model.callback import TrainingCallback
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, LeakyReLU, Concatenate, BatchNormalization, Input
from tensorflow.keras.regularizers import L2
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanAbsoluteError
import logging

class Forecaster:
    """
    A class for forecasting using a neural network model.
    """
    model_params = {}  # Class variable for model parameters

    def __init__(self, params: dict) -> None:
        """
        Initialize the Forecaster with the provided parameters.

        Args:
            params (dict): A dictionary of parameters for configuring the model and training.

        Returns:
            None
        """
        self._params = params
        self.__load_dataset()
        self.__create_model()

        if self._params.get('load_prev', False):
            self._load_model()

        self.__compile_model()

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Model:
        """
        Train the model using the provided training and validation data.

        Args:
            X_train (np.ndarray): Input data for training.
            y_train (np.ndarray): Target data for training.
            X_val (np.ndarray): Input data for validation.
            y_val (np.ndarray): Target data for validation.

        Returns:
            Model: Trained model.
        """
        try:
            self.keras_model.fit(
                X_train,
                y_train,
                batch_size=self.batch_size,
                epochs=self._params.get('epochs', 10),
                validation_data=(X_val, y_val),
                verbose=0,
                callbacks=[TrainingCallback(
                    self._params['log_dir'],
                    self._params['model_name'],
                    plot_freq=self._params['plot_freq'],
                    print_freq=self._params['print_freq'],
                    checkpoint_freq=self._params['checkpoint_freq'],
                    checkpoint_fn=lambda epoch: self.__save_model(epoch)
                )]
            )

            self.__save_model()

            print(30 * '=', '|', ' Finished Training ', '|', 30 * '=', sep='')

            return self.keras_model
        except Exception as e:
            logging.error(f"Error during training: {e}")
            return None

    @property
    def batch_size(self) -> int:
        return self._params['batch_size']

    @property
    def data_path(self) -> str:
        return self._params['data_path']

    @property
    def keras_model(self) -> Model:
        return self._model

    def __load_dataset(self):
        """
        Load datasets for training, validation, and testing.
        """
        self._datasets = {
            name: Dataset(os.path.join(self.data_path, name) + '.' + self._params['ds_format'], self.batch_size)
            for name in ['train', 'val', 'test']
        }
        self._inpt_shape = self._datasets['train'].shape()
        self._output_shape = self._datasets['train'].output_shape()

    def __dense_layer(self, units, x):
        """
        Create a dense layer with specified units and apply regularization.
        """
        return Dense(units, kernel_regularizer=L2(1), kernel_initializer=glorot_uniform())(x)

    def __relu(self, x):
        """
        Apply LeakyReLU activation function with the specified alpha.
        """
        return LeakyReLU(self._params['alpha'])(x)

    def __batch_norm(self, x):
        """
        Apply Batch Normalization to the input.
        """
        return BatchNormalization()(x)

    def __residual_layer(self, units, x):
        """
        Create a residual layer by combining a dense layer with the input.
        """
        x_skp = self.__dense_layer(units, x)
        x_skp = self.__batch_norm(x_skp)
        x_skp = self.__relu(x_skp)
        return Concatenate()([x, x_skp])

    def __create_model(self):
        """
        Create the neural network model based on the provided configuration.
        """
        inpt = Input(shape=self._inpt_shape)
        x = inpt

        for dim in self._params['model_dim']:
            x = self.__residual_layer(dim, x)

        assert len(self._output_shape) == 1
        x = Dense(self._output_shape[0], kernel_regularizer=L2(1), kernel_initializer="he_uniform")(x)
        self._model = Model(inputs=inpt, outputs=x)

    def __load_model(self):
        """
        Load a previously trained model from the specified location.
        """
        params_loc = os.path.join(self._params['prev_dir'], 'params.json')
        model_loc = os.path.join(self._params['prev_dir'], self._params['prev_name'])

        try:
            with open(params_loc, 'r') as file:
                old_params = json.load(file)

            for inherited in self._params.get('inherit', []):
                self._params[inherited] = old_params.get(inherited, self._params.get(inherited, ''))

            self._model = load_model(model_loc, compile=False)
        except Exception as e:
            logging.error(f"Error loading model: {e}")

    def __save_model(self, epoch=None):
        """
        Save the trained model and its parameters to the specified directory.
        """
        model_dir = os.path.join(self._params['models_dir'], self._params['model_name'])
        model_dir = os.path.join(model_dir, 'final') if epoch is None else os.path.join(model_dir, 'checkpoint')

        try:
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            params_loc = os.path.join(model_dir, 'params.json')
            model_loc = os.path.join(model_dir, f'trained_model{"_" + str(epoch) if epoch is not None else ""}')

            with open(params_loc, 'w') as file:
                json.dump(self._params, file)

            self.keras_model.save(model_loc, include_optimizer=False)
        except Exception as e:
            logging.error(f"Error saving model: {e}")

    def __compile_model(self):
        """
        Compile the model with the specified optimizer, loss function, and metrics.
        """
        optimizer = Adam(
            self._params['learning_rate'],
            self._params['beta_1'],
            self._params['beta_2']
        )

        try:
            self.keras_model.compile(
                optimizer=optimizer,
                loss='mse',
                metrics=[MeanAbsoluteError()]
            )
        except Exception as e:
            logging.error(f"Error compiling model: {e}")

    def print_params(self):
        """
        Print model parameters and summary.
        """
        print('Model Parameters:')
        print(json.dumps(self._params, indent=4))
        print('Model Summary:')
        self.keras_model.summary()
        print(end='', flush=True)

    def train(self):
        """
        Train the model using the provided dataset and parameters.
        """
        print(30 * '=', '|', ' Starting Training ', '|', 30 * '=', sep='')

        try:
            self.keras_model.fit(
                self._datasets['train'],
                batch_size=self.batch_size,
                epochs=self._params.get('epochs', 10),
                validation_data=self._datasets['val'].data,
                verbose=0,
                callbacks=[TrainingCallback(
                    self._params['log_dir'],
                    self._params['model_name'],
                    plot_freq=self._params['plot_freq'],
                    print_freq=self._params['print_freq'],
                    checkpoint_freq=self._params['checkpoint_freq'],
                    checkpoint_fn=lambda epoch: self.__save_model(epoch)
                )]
            )

            self.__save_model()

            print(30 * '=', '|', ' Finished Training ', '|', 30 * '=', sep='')
            self.keras_model.evaluate(
                self._datasets['test'].data,
                verbose=0,
                callbacks=[TrainingCallback(
                    log_dir=self._params['log_dir'],
                    model_name=self._params['model_name']
                )]
            )
        except Exception as e:
            logging.error(f"Error during training: {e}")

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.

        Args:
            input_data (np.ndarray): Input data for making predictions.

        Returns:
            np.ndarray: Model predictions.
        """
        try:
            predictions = self.keras_model.predict(input_data)
            return predictions
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            return None

# Example of injecting datasets using dependency injection
def get_forecaster(params: dict, datasets: dict) -> Forecaster:
    """ 
    Create a Forecaster instance with the provided parameters and injected datasets.

    Args:
        params (dict): Model parameters.
        datasets (dict): Datasets for training, validation, and testing.

    Returns:
        Forecaster: Initialized Forecaster instance.
    """
    model = Forecaster(params)
    model._datasets = datasets  # Inject datasets
    return model
