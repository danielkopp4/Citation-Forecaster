from src.data_extraction.dataset_api import Dataset
from tensorflow.keras import Model, Input
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanAbsoluteError
from src.model.callback import TrainingCallback
import os
import json
import pprint

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
        params_loc = os.path.join(self._params['prev_dir'], 'params.json')
        model_loc = os.path.join(self._params['prev_dir'], self._params['prev_name'])

        with open(params_loc, 'r') as file:
            old_params = json.load(file)

        for inherited in self._params['inherit']:
            self._params[inherited] = old_params[inherited]

        self._model = load_model(model_loc, compile=False)


    def __save_model(self, epoch=None):
        if not os.path.exists(self._params['models_dir']):
            os.mkdir(self._params['models_dir'])

        model_dir = os.path.join(self._params['models_dir'], self._params['model_name'])

        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        if epoch == None:
            # final
            model_dir = os.path.join(model_dir, 'final')
        else:
            # checkpoint
            model_dir = os.path.join(model_dir, 'checkpoint')

        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        params_loc = os.path.join(model_dir, 'params.json')
        model_loc = os.path.join(model_dir, f'trained_model')
        if epoch != None:
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
            metrics=[MeanAbsoluteError()]
        )


    def print_params(self):
        print("model params:")
        pprint.pprint(self._params)
        

    def train(self):
        print(30*'=', '|', ' Starting Training ', '|',  30*'=', sep='')
        self.keras_model.fit(
            self._datasets['train'],
            batch_size=self.batch_size,
            epochs=self._params['epochs'],
            validation_data=self._datasets['val'],
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
            self._datasets['test'],
            verbose=0,
            callbacks=[TrainingCallback(
                self._params['log_dir'], 
                self._params['model_name']
            )]
        )



def get_model(params: dict) -> Forecaster:
    model = Forecaster(params)
    return model
