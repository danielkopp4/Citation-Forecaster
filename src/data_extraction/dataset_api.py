from abc import ABC, abstractmethod, abstractproperty
from typing import Set, Tuple
from src.betting_env.odds import Odds
import numpy as np
from copy import deepcopy

class DataAPI(ABC):
    def get_data_generator():
        pass

    @abstractproperty
    def observation_space(self):
        pass


class HistoricalBettingDataAPI(ABC):
    def __init__(self):
        self._order = np.arange(self._length_raw)

    def shuffle(self):
        np.random.shuffle(self._order)

    @abstractmethod
    def _get_item_raw(self, index) -> Tuple[Odds, Odds, str, str, int, int]:
        pass

    @abstractproperty
    def _length_raw(self) -> int:
        pass

    def copy(self) -> 'HistoricalBettingDataAPI':
        return deepcopy(self) 
    
    def __len__(self):
        return len(self._order)

    def __getitem__(self, index) -> Tuple[Odds, Odds, str, str, int, int]:
        if isinstance(index, slice):
            cpy = self.copy()
            cpy._order = self._order[index]
            return cpy

        return self._get_item_raw(self._order[index])  

    @abstractmethod
    def get_unique_teams(self) -> Set[str]:
        pass