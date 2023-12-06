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
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from tensorflow.keras.layers import LSTM, Dropout, Add, BatchNormalization
import logging
import traceback
import configparser

# Set up logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='model_log.log'
)

# Custom exception for model-related errors
class ModelError(Exception):
    pass

class Forecaster:
    """
    A class for forecasting using a neural network model.
    """
    model_params = {}  # Class variable for model parameters

    def __init__(self, config_file: str, load_prev: bool = False) -> None:
        """
        Initializes the Forecaster class with model parameters and datasets.

        Args:
            config_file (str): Path to the configuration file.
            load_prev (bool): Flag to load a previously trained model.
        Raises:
            ValueError: If required parameters are missing or invalid.
        """
        config = configparser.ConfigParser()
        config.read(config_file)

        model_params = config['ModelParams']

        # Parameter validation checks
        required_params = ['batch_size', 'data_path', 'model_dim', 'learning_rate', 'beta_1', 'beta_2']
        missing_params = [param for param in required_params if param not in model_params]
        if missing_params:
            raise ValueError(f"Missing required parameters: {', '.join(missing_params)}")

        if not model_params.getint('batch_size') > 0:
            raise ValueError("Batch size must be a positive integer")

        # Initialize parameters
        self._params = {
            'batch_size': model_params.getint('batch_size'),
            'data_path': model_params['data_path'],
            'model_dim': list(map(int, model_params['model_dim'].split(','))),
            'learning_rate': model_params.getfloat('learning_rate'),
            'beta_1': model_params.getfloat('beta_1'),
            'beta_2': model_params.getfloat('beta_2'),
        }

        self.__load_dataset()
        self.__create_model()

        if self._params.get('load_prev', False):
            self._load_model()

        self.__compile_model()
        
        self.log_dir = os.path.join('logs', self._params['model_name'])  # Directory for TensorBoard logs
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1)

        
        self.lr_scheduler = LearningRateScheduler(cosine_decay_scheduler(self._params['learning_rate'], self._params.get('epochs', 10)))
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=3)
        
        self._params['load_prev'] = load_prev
        if load_prev:
            self.__load_model()
            
    def cosine_decay_scheduler(initial_lr, total_epochs):
        """
        Custom learning rate scheduler implementing cosine decay.

        Args:
            initial_lr (float): Initial learning rate.
            total_epochs (int): Total number of epochs.
        Returns:
            function: Scheduler function.
        """
        def scheduler(epoch):
            """
            Cosine decay scheduler function.

            Args:
                epoch (int): Current epoch.
            Returns:
                float: Updated learning rate.
            """
            cosine_decay = 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))
            updated_lr = initial_lr * cosine_decay
            return updated_lr

        return scheduler


    def __load_dataset(self):
        """
        Load datasets for training, validation, and testing.
        """
        self._datasets = {
            name: Dataset(os.path.join(self.data_path, name) + '.' + self._params['ds_format'], self.batch_size)
            for name in ['train', 'val', 'test']
        }
        # Feature engineering - Lagged features
        for ds_name, dataset in self._datasets.items():
            dataset.df['lag_1'] = dataset.df['target'].shift(1)  # Example lagged feature
        # Extract validation data for monitoring during training
        self._val_data = self._datasets['val'].data
        self._inpt_shape = self._datasets['train'].shape()
        self._output_shape = self._datasets['train'].output_shape()

    def __residual_layer(self, units, x, kernel_regularizer=None):
        """
        Create a residual layer by combining a dense layer with the input.
        """
        x_skp = self.__dense_layer(units, x, kernel_regularizer=kernel_regularizer)
        x_skp = self.__batch_norm(x_skp)
        x_skp = self.__relu(x_skp)
        x = Concatenate()([x, x_skp])
        return self.__dense_layer(units, x, kernel_regularizer=kernel_regularizer)

    def __create_residual_blocks(self):
        """
        Create residual blocks in the model based on model_dim parameter.
        """
        inpt = Input(shape=self._inpt_shape)
        x = inpt
        residual = x  # Initialize residual connection

        for dim in self._params['model_dim']:
            # Apply LSTM layers with dropout and residual connections
            x = LSTM(dim, return_sequences=True)(x)
            x = Dropout(0.2)(x)  # Add dropout for regularization
            x = BatchNormalization()(x)  # Add batch normalization
            x = Add()([residual, x])  # Add residual connection
            residual = self.__residual_layer(dim, x, kernel_regularizer=L2(1))

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
                validation_data=(X_val, y_val),
                batch_size=self.batch_size,
                epochs=self._params.get('epochs', 10),
                verbose=0,
                callbacks=[
                    self.lr_scheduler,
                    self.early_stopping,
                    self.tensorboard_callback,  # Add TensorBoard callback
                    TrainingCallback(
                        self._params['log_dir'],
                        self._params['model_name'],
                        plot_freq=self._params['plot_freq'],
                        print_freq=self._params['print_freq'],
                        checkpoint_freq=self._params['checkpoint_freq'],
                        checkpoint_fn=lambda epoch: self.__save_model(epoch)
                            )
                    ]
            )

            self.__save_model()

            print(30 * '=', '|', ' Finished Training ', '|', 30 * '=', sep='')

            return self.keras_model
        except Exception as e:
            error_msg = f"Error during training: {e}. Traceback: {traceback.format_exc()}"
            logging.error(error_msg)
            raise ModelError(error_msg)

    @property
    def batch_size(self) -> int:
        return self._params['batch_size']

    @property
    def data_path(self) -> str:
        return self._params['data_path']

    @property
    def keras_model(self) -> Model:
        return self._model

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

    def __create_model(self):
        """
        Create the neural network model based on the provided configuration.
        """
        inpt, x = self.__create_residual_blocks()

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
        # Experiment with different optimizers (e.g., RMSprop, Adagrad, Nadam)
        optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=self._params['learning_rate'],
            momentum=0.9  # Adjust momentum if needed
        )

        try:
            self.keras_model.compile(
                optimizer=optimizer,
                loss='mse',
                metrics=[tf.keras.metrics.MeanAbsoluteError()]
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
                callbacks=[
                    self.lr_scheduler,
                    self.early_stopping,
                    self.tensorboard_callback,  # Add TensorBoard callback
                    TrainingCallback(
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
