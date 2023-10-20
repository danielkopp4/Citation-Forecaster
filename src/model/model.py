import os
import json
import tensorflow as tf
from src.data_extraction.dataset_api import Dataset
from src.model.callback import TrainingCallback
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import logging

class Forecaster:
    def __init__(self, config_file):
        self._logger = self._setup_logger()
        try:
            with open(config_file, 'r') as file:
                self._params = json.load(file)
        except Exception as e:
            self._logger.error(f"Error loading config file: {e}")
            raise

        self._datasets = self._load_datasets()
        self._model = self._create_model()

        if self._params.get('load_prev', False):
            self._load_model()

        self._compile_model()
        self.validate_config()

    @property
    def batch_size(self) -> int:
        return self._params.get('batch_size', 32)

    @property
    def data_path(self) -> str:
        return self._params.get('data_path', '')

    @property
    def keras_model(self) -> Model:
        return self._model

    def _setup_logger(self):
        logger = logging.getLogger(__name__)
        log_level = self._params.get('log_level', 'INFO')
        logging.basicConfig(level=logging.getLevelName(log_level), format="%(asctime)s [%(levelname)s] %(message)s")
        return logger

    def _load_datasets(self):
        datasets = {}
        formats = self._params.get('ds_format', 'train,val,test').split(',')
        for name in formats:
            dataset = Dataset(os.path.join(self.data_path, f"{name}.{self._params.get('ds_format', 'csv')}"), self.batch_size)
            datasets[name] = dataset
        self._inpt_shape = datasets['train'].shape
        self._output_shape = datasets['train'].output_shape
        return datasets

    def _create_model(self):
        inpt = Input(shape=(self._inpt_shape,))
        x = inpt

        for dim in self._params.get('model_dim', []):
            x = Dense(dim, kernel_regularizer=tf.keras.regularizers.L2(1), kernel_initializer="he_uniform")(x)
            x = tf.keras.layers.LeakyReLU(self._params.get('alpha', 0.01))(x)

        assert len(self._output_shape) == 1
        x = Dense(self._output_shape[0], kernel_regularizer=tf.keras.regularizers.L2(1), kernel_initializer="he_uniform")(x)
        return Model(inputs=inpt, outputs=x)

    def _load_model(self):
        params_loc = os.path.join(self._params.get('prev_dir', ''), 'params.json')
        model_loc = os.path.join(self._params.get('prev_dir', ''), self._params.get('prev_name', ''))

        with open(params_loc, 'r') as file:
            old_params = json.load(file)

        for inherited in self._params.get('inherit', []):
            self._params[inherited] = old_params.get(inherited, self._params.get(inherited, ''))

        self._model = load_model(model_loc, compile=False)

    def validate_config(self):
        required_params = ['data_path', 'batch_size', 'model_dim', 'output_shape', 'learning_rate', 'log_level']
        missing_params = [param for param in required_params if param not in self._params]
        if missing_params:
            self._logger.error(f"Missing required parameters in config: {missing_params}")
            raise ValueError(f"Missing required parameters in config: {missing_params}")

    def _save_model(self, epoch=None):
        model_dir = os.path.join(self._params.get('models_dir', ''), self._params.get('model_name', ''))
        model_dir = os.path.join(model_dir, 'final' if epoch is None else 'checkpoint')

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        params_loc = os.path.join(model_dir, 'params.json')
        model_loc = os.path.join(model_dir, 'trained_model.h5')
        if epoch is not None:
            model_loc = os.path.join(model_dir, f'trained_model_{epoch}.h5')

        with open(params_loc, 'w') as file:
            json.dump(self._params, file)

        self.keras_model.save(model_loc, include_optimizer=False)

    def _compile_model(self):
        optimizer = Adam(
            self._params.get('learning_rate', 0.001),
            self._params.get('beta_1', 0.9),
            self._params.get('beta_2', 0.999)
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

    def train(self):
        self._logger.info('Starting Training')
        early_stopping = EarlyStopping(patience=self._params.get('early_stopping_patience', 10), monitor='val_loss', restore_best_weights=True)
        checkpoint = ModelCheckpoint(self._params.get('checkpoint_path', 'checkpoints/model_{epoch:02d}.h5'), save_best_only=True)
        tensorboard = TensorBoard(log_dir=self._params.get('log_dir', ''))

        history = self.keras_model.fit(
            self._datasets['train'].data,
            batch_size=self.batch_size,
            epochs=self._params.get('epochs', 10),
            validation_data=self._datasets['val'].data,
            verbose=0,
            callbacks=[early_stopping, checkpoint, tensorboard]
        )

        # Log training metrics
        self._logger.info('Training metrics:')
        self._logger.info(history.history)

        self._save_model()
        self._logger.info('Finished Training')

        # Perform model evaluation and log additional metrics
        test_metrics = self.keras_model.evaluate(self._datasets['test'].data, verbose=0)
        self._logger.info('Test metrics:')
        self._logger.info(test_metrics)

        self.keras_model.evaluate(
            self._datasets['test'].data,
            verbose=0,
            callbacks=[TrainingCallback(
                self._params.get('log_dir', ''),
                self._params.get('model_name', '')
            )]
        )

def get_model(config_file: str) -> Forecaster:
    model = Forecaster(config_file)
    return model
