from src.data_extraction.dataset_api import Dataset
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from src.model.callback import TrainingCallback
import os

class Forecaster:
    def __init__(self, params: dict):
        self._params = params
        self.__load_dataset()
        self.__create_model()

        if self._params["load_prev"]:
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
            name: Dataset(self.batch_size, os.path.join(self.data_path, name)) 
            for name in ['train', 'val', 'test']
        }

    def __create_model(self):
        self._model = Model()

    def __load_model(self):
        pass

    def __compile_model(self):
        optimizer = Adam(
            self._params['learning_rate'],
            self._params['beta_1'],
            self._params['beta_2']            
        )

        self._model.compile(
            optimizer=optimizer,
            loss="mse",
            metrics=[]
        )

    def train(self):
        self.keras_model.fit(
            self._datasets['train'],
            batch_size=self.batch_size,
            epochs=self.params['epochs'],
            validation_data=self._datasets['val'],
            verbose="none",
            callbacks=[TrainingCallback()]
        )

        self.keras_model.evaluate(
            self._datasets['test'],
            verbose="none",
            callbacks=[TrainingCallback()]
        )



def get_model(params: dict) -> Forecaster:
    model = Forecaster(params)
    return model
