import os
import json
import tensorflow as tf
from src.data_extraction.dataset_api import Dataset
from src.model.callback import TrainingCallback
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, LeakyReLU, Concatenate, BatchNormalization
from tensorflow.keras.regularizers import L2
from tensorflow.keras.initializers import glorot_uniform  # Import glorot_uniform initializer
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanAbsoluteError
from src.model.callback import TrainingCallback
import os
import json
import pprint
import numpy as np  # Import numpy for type hint

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
        self._datasets = {
            name: Dataset(os.path.join(self.data_path, name) + '.' + self._params['ds_format'], self.batch_size)
            for name in ['train', 'val', 'test']
        }
        self._inpt_shape = self._datasets['train'].shape()
        self._output_shape = self._datasets['train'].output_shape()

    def __dense_layer(self, units, x):
        return Dense(units, kernel_regularizer=L2(1), kernel_initializer=glorot_uniform())(x)  # Use glorot_uniform initializer

    def __relu(self, x):
        return LeakyReLU(self._params['alpha'])(x)

    def __batch_norm(self, x):
        return BatchNormalization()(x)

    def __residual_layer(self, units, x):
        x_skp = self.__dense_layer(units, x)
        x_skp = self.__batch_norm(x_skp)  # Apply Batch Normalization here
        x_skp = self.__relu(x_skp)
        return Concatenate()([x, x_skp])

    def __create_model(self):
        inpt = Input(shape=self._inpt_shape)
        x = inpt

        for dim in self._params['model_dim']:
            x = self.__residual_layer(dim, x)

        assert len(self._output_shape) == 1
        x = Dense(self._output_shape[0], kernel_regularizer=tf.keras.regularizers.L2(1), kernel_initializer="he_uniform")(x)
        return Model(inputs=inpt, outputs=x)

    def __load_model(self):
        params_loc = os.path.join(self._params['prev_dir'], 'params.json')
        model_loc = os.path.join(self._params['prev_dir'], self._params['prev_name'])

        with open(params_loc, 'r') as file:
            old_params = json.load(file)

        for inherited in self._params.get('inherit', []):
            self._params[inherited] = old_params.get(inherited, self._params.get(inherited, ''))

        self._model = load_model(model_loc, compile=False)

    def __save_model(self, epoch=None):
        if not os.path.exists(self._params['models_dir']):
            os.mkdir(self._params['models_dir'])

        model_dir = os.path.join(self._params['models_dir'], self._params['model_name'])

        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        if epoch is None:
            # final
            model_dir = os.path.join(model_dir, 'final')
        elif epoch % self._params['checkpoint_freq'] == 0:  # Check if epoch is a checkpoint based on checkpoint_freq
            # checkpoint
            model_dir = os.path.join(model_dir, 'checkpoint')

        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        params_loc = os.path.join(model_dir, 'params.json')
        model_loc = os.path.join(model_dir, f'trained_model')
        if epoch is not None:
            model_loc += f'_{epoch}'

        with open(params_loc, 'w') as file:
            json.dump(self._params, file)

        self.keras_model.save(model_loc, include_optimizer=False)

    def __compile_model(self):
        optimizer = Adam(
            self._params['learning_rate'],
            self._params['beta_1'],
            self._params['beta_2']
        )

        self.keras_model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=[tf.keras.metrics.MeanAbsoluteError()]
        )

    def print_params(self):
        self._logger.info('Model Parameters:')
        self._logger.info(json.dumps(self._params, indent=4))
        self._logger.info('Model Summary:')
        self.keras_model.summary()
        print(end='', flush=True)

    def train(self):
        """
        Train the model using the provided dataset and parameters.
        """
        print(30 * '=', '|', ' Starting Training ', '|', 30 * '=', sep='')
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

        print(30*'=', '|', ' Finished Training ', '|',  30*'=', sep='')
        self.keras_model.evaluate(
            self._datasets['test'].data,
            verbose=0,
            callbacks=[TrainingCallback(
                log_dir=self._params['log_dir'],  # Fix this
                model_name=self._params['model_name']  # Fix this
            )]
        )

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.

        Args:
            input_data (np.ndarray): Input data for making predictions.

        Returns:
            np.ndarray: Model predictions.
        """
        # Add code here to perform predictions using the trained model
        predictions = self.keras_model.predict(input_data)
        return predictions

# Example of injecting datasets using dependency injection
def get_forecaster(params: dict, datasets: dict) -> Forecaster:
    model = Forecaster(params)
    model._datasets = datasets  # Inject datasets
    return model
