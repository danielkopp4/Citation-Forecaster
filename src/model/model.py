from src.data_extraction.dataset_api import Dataset
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanAbsoluteError
from src.model.callback import TrainingCallback
import os

class Forecaster:
    def __init__(self, params: dict):
        self._params = params
        self.__load_dataset()
        self.__create_model()

        if self._params['load_prev']:
            self.__load_model()

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


    def __dense_layer(self, units):
        return Dense(units)


    def __relu(self):
        return LeakyReLU(self._params['alpha'])
    

    def __create_model(self):
        inpt = Input(shape=self._inpt_shape)
        x = self.__dense_layer(256)(inpt)
        x = self.__relu()(x)
        assert len(self._output_shape) == 1
        x = self.__dense_layer(self._output_shape[0])(inpt)
        self._model = Model(inputs=inpt, outputs=x)


    def __load_model(self):
        assert False


    def __compile_model(self):
        optimizer = Adam(
            self._params['learning_rate'],
            self._params['beta_1'],
            self._params['beta_2']            
        )

        self.keras_model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=[MeanAbsoluteError()]
        )


    def train(self):
        self.keras_model.fit(
            self._datasets['train'],
            batch_size=self.batch_size,
            epochs=self._params['epochs'],
            validation_data=self._datasets['val'],
            verbose='none',
            callbacks=[TrainingCallback()]
        )

        self.keras_model.evaluate(
            self._datasets['test'],
            verbose='none',
            callbacks=[TrainingCallback()]
        )



def get_model(params: dict) -> Forecaster:
    model = Forecaster(params)
    return model
