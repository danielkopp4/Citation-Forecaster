from tensorflow.keras.utils import Sequence
from typing import Tuple
import numpy as np

class Dataset(Sequence):
    def __init__(self, batch_size):
        pass

    def __len__(self) -> int:
        pass

    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray]:
        pass