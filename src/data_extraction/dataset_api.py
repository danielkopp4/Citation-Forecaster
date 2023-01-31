from tensorflow.keras.utils import Sequence
from typing import Tuple
import numpy as np

class Dataset(Sequence):
    def __init__(self, batch_size: int, file_path: str):
        self._batch_size = batch_size
        self._file_path = file_path
        self.__load_data()
        self.shuffle_data()

    @property
    def n_observations(self) -> int:
        return self.n_observations

    def shape(self, include_batch=False) -> np.shape:
        if include_batch:
            return np.shape([self._batch_size] + list(self.shape()))
        
        return np.shape([self._n_features])

    def __load_data(self):
        self._n_observations = None
        self._data = None
        self._indices = None
        self._n_features = None

    def __len__(self) -> int:
        return self.n_observations  // self.batch_size
    
    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        index_batch = self._indices[index * self._batch_size: (index + 1) * self._batch_size]
        batch = self._data[index_batch]
        return batch[:,:-1], batch[:,-1]
    
    def shuffle_data(self):
        np.random.shuffle(self.indices)

    def on_epoch_end(self):
        return self.shuffle_data()
    