from src.data_extraction.dataset_api import Dataset
from tensorflow.keras import Model, Input
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, LeakyReLU, Concatenate, Add, BatchNormalization
from tensorflow.keras.regularizers import L2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanAbsoluteError
from src.model.callback import TrainingCallback
import os
import json
import pprint
import tensorflow as tf

class Forecaster:
    def __init__(self, params: dict):
        self._params = params
        self._datasets = self._load_datasets()
        self._model = self._create_model()

        if self._params.get('load_prev', False):
            self._load_model()

        self._compile_model()

    @property
    def batch_size(self) -> int:
        return self._params.get('batch_size', 32)

    @property
    def data_path(self) -> str:
        return self._params.get('data_path', '')

    @property
    def keras_model(self) -> tf.keras.Model:
        return self._model

    def _load_datasets(self):
        datasets = {}
        formats = self._params.get('ds_format', 'train,val,test').split(',')
        for name in formats:
            dataset = Dataset(os.path.join(self.data_path, f"{name}.{self._params.get('ds_format', 'csv')}"), self.batch_size)
            datasets[name] = dataset
        self._inpt_shape = datasets['train'].shape()
        self._output_shape = datasets['train'].output_shape()
        return datasets


    def __dense_layer(self, units, x):
        return Dense(units, kernel_regularizer=L2(1), kernel_initializer="he_uniform")(x)


    def __relu(self, x):
        return LeakyReLU(self._params['alpha'])(x)
    
    
    def __batch_norm(self, x):
        return BatchNormalization()(x)
    
    
    def __residual_layer(self, units, x):
        x_skp = self.__dense_layer(units, x)
        # x_skp = self.__batch_norm(x_skp)
        x_skp = self.__relu(x_skp)
        return Concatenate()([x, x_skp])
        

    def __create_model(self):
        inpt = Input(shape=self._inpt_shape)
        x = inpt

        for dim in self._params['model_dim']:
            x = self.__residual_layer(dim, x)
        
        assert len(self._output_shape) == 1
        x = self.__dense_layer(self._output_shape[0], x)
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
        print('model params:')
        pprint.pprint(self._params)
        print('model summary:')
        self.keras_model.summary()
        print(end='', flush=True)
        

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
