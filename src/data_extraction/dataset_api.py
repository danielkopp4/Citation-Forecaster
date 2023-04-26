from tensorflow.keras.utils import Sequence
from typing import Tuple
import numpy as np


'''
The dataset that contins the specfic shape and 
properties dataset should be able to perform

used by the downloader to the be able to keep the aspects of the dataset in one place
'''
class Dataset(Sequence):
    def __init__(self, file_path: str, batch_size: int = 64):
        self._batch_size = batch_size
        self._file_path = file_path
        self.__load_data()
        self.shuffle_data()


    @property
    def n_observations(self) -> int:
        return self._n_observations


    def shape(self, include_batch=False) -> np.shape:
        shape = [self._n_features]
        if include_batch:
            shape = [self._batch_size] + shape
        return tuple(shape)


    def output_shape(self) -> np.shape:
        return (1,)
        

    def __load_data(self):
        raw_data = np.load(self._file_path)
        self._n_observations = len(raw_data)
        self._data = raw_data
        self._indices = np.arange(self._n_observations)
        self._n_features = self._data.shape[1] - 1 # last col is y var


    def __len__(self) -> int:
        return self.n_observations // self._batch_size
    

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        index_batch = self._indices[index * self._batch_size: (index + 1) * self._batch_size]
        batch = self._data[index_batch]
        return batch[:,:-1], np.reshape(batch[:,-1], (self._batch_size, self.output_shape()[0]))
    

    def shuffle_data(self):
        np.random.shuffle(self._indices)


    def on_epoch_end(self):
        return self.shuffle_data()
