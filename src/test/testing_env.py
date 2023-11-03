import gym
from gym import spaces
import numpy as np
from typing import Dict, List, Optional, Set, Tuple
import logging
from stable_baselines3.common.env_checker import check_env


from src.betting_env.odds import Odds
from src.data_api import HistoricalBettingDataAPI


LAMBDA_GAINS: float = 1
LAMBDA_LOSSES: float = -1.1
ACTION_MULTIPLIER: float = 3


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def redistribute(x: np.ndarray, scalar: Optional[float] = None) -> np.ndarray:
    if scalar == None:
        return x

    
    x_t = softmax(x * ACTION_MULTIPLIER)
    amnt_0 = x_t[0] / scalar
    amnt_1 = x_t[1] / scalar
    amnt = amnt_0 + amnt_1
    x_t[2] += amnt
    x_t[0] -= amnt_0
    x_t[1] -= amnt_1
    return np.log(x_t) / ACTION_MULTIPLIER


def n_to_vec_one_hot(n: int, length: int) -> np.ndarray:
    result = np.zeros((length,))
    result[n] = 1
    return result


def calculate_one_hot_mapping(items: Set) -> Dict:
    mapping = {}
    for i, item in enumerate(items):
        mapping[item] = n_to_vec_one_hot(i, len(items))

    return mapping


def make_observation_space(N_teams: int) -> spaces.Space:
    # observation is [g, oa, ob, date, *team_a_oh, *team_b_oh]
    # in a 4+2*N matrix where N is number of teams
    return spaces.Box(
        low=np.concatenate((np.array([1,1,-1]), np.zeros((2*N_teams,))-0.01), dtype=np.float32), # date floor end at 2010 may need to go lower than -1 
        high=np.concatenate((np.array([np.inf,np.inf,1]), 1.01*np.ones((2*N_teams,))), dtype=np.float32), # ceiling ends at 2040 may need to increase if doing this a while
        shape=(3+2*N_teams,),
        dtype=np.float32
    )



class SportsBettingOddsEnv(gym.Env):
    metadata = {'render.modes': ['console', 'none']}

    def __init__(self, initial_pot: float, data_api: HistoricalBettingDataAPI): # takes some variable that biases the betting odds o_a, o_b
        super(SportsBettingOddsEnv, self).__init__()
        self.action_space = spaces.Box(
            low=np.array([-1,-1,-1],dtype=np.float32), 
            high=np.array([1,1,1], dtype=np.float32), 
            shape=(3,), 
            dtype=np.float32
        )

        self._team_mapping = calculate_one_hot_mapping(data_api.get_unique_teams())
        self.observation_space = make_observation_space(len(self._team_mapping))

        self._data_api = data_api
        self._order = np.arange(len(self._data_api))
        self._iterations = 0

        self.setup_next_game()

    @property
    def o_a(self) -> Odds:
        return self._o_a

    @property
    def o_b(self) -> Odds:
        return self._o_b

    @property
    def o(self) -> List[Odds]:
        return [self.o_a, self.o_b]

    @property
    def team_a(self) -> np.ndarray:
        return self._team_mapping[self.get_team_a()]

    @property
    def team_b(self) -> np.ndarray:
        return self._team_mapping[self.get_team_b()]

    @property
    def teams(self) -> np.ndarray:
        return np.array([self.team_a, self.team_b])

    @property
    def date(self) -> int:
        # utc timestamp just normalize between 2010 and 2040
        # -> between -1 and 1
        _2010_utc = 1262322000
        _2040_utc = 2209006800
        return 2 * (self._date - _2010_utc) / (_2040_utc - _2010_utc) - 1

    def get_team_a(self) -> str:
        return self._team_a

    def get_team_b(self) -> str:
        return self._team_b

    def get_teams(self) -> List[str]:
        return [self.get_team_a(), self.get_team_b()]

    def get_winner(self) -> int:
        return self._winner

    @property
    def data_point(self) -> Tuple[Odds, Odds, str, str, int, int]:
        return self._data_api[self.index]
        
    def calculate_observation(self) -> np.ndarray:
        return np.concatenate((
            np.array(
                    [self.o_a.get_decimal(), self.o_b.get_decimal(), self.date], 
                    dtype=np.float32
            ),
            self.team_a,
            self.team_b
        ), dtype=np.float32)

    @property
    def index(self) -> int:
        return self._order[self._iterations]

    def setup_next_game(self):
        self._o_a, self._o_b, self._team_a, self._team_b, self._date, self._winner = self.data_point

    def reset(self) -> np.ndarray:
        self._iterations += 1

        if self._iterations == len(self._data_api):
            np.random.shuffle(self._order)
            self._iterations = 0

        observation = self.calculate_observation()
        return observation

    def step(self, action):
        action = softmax(ACTION_MULTIPLIER * action)
        # print("action:", action)
        self._action = action
        winning_team = self.get_winner()

        # calculate gains, losses
        self._gains = action[winning_team] * (self.o[winning_team].get_winnings_multiplier() - 1)
        self._losses = action[1 - winning_team]

        # calculate reward
        # print("winning odds:", action[winning_team])
        # print("gains:", self._gains)
        # print("losing odds:", action[1 - winning_team])
        # print("losses:", self._losses)
        # print("ngains:",  self._gains / self.pot)
        # print("nlosses:", self._losses / self.pot)
        # reward = LAMBDA_GAINS * (self._gains / self.pot) + LAMBDA_LOSSES * (self._losses / self.pot)
        reward = LAMBDA_GAINS * self._gains + LAMBDA_LOSSES * self._losses 
        # print("reward:", reward)

        self.setup_next_game()

        # return step results
        observation = self.calculate_observation()
        done = True
        info = {
            "pot": 1 + self._gains - self._losses
        }
        return observation, reward, done, info

    def render(self, mode='console'):
        if mode == 'console':
            if self._iterations == 0:
                logging.info("oa=%02.3f ob=%02.3f ta=%s tb=%s", self.o_a.get_decimal(), self.o_b.get_decimal(), self.get_team_a(), self.get_team_b())
            else:
                logging.info("action taken: [%.2f, %.2f, %.2f]", *self._action)
                logging.info("oa=%02.3f ob=%02.3f ta=%s tb=%s | ga=%.3e ls=%.5e", self.o_a.get_decimal(), self.o_b.get_decimal(), self.get_team_a(), self.get_team_b(), self._gains, self._losses)

    def close(self):
        pass

class MultiGameSimulation:
    def __init__(self, model, env):
        self._model = model
        self._env = env

    def run_sims(self, length, trials, scalar=None, verbose=0):
        amounts = []
        for _ in range(trials):
            pot = 1
            for _ in range(length):
                obs = self._env.reset()
                action, _ = self._model.predict(obs)
                action = redistribute(action, scalar)
                _, _, _, info = self._env.step(action)
                pot *= info["pot"]

                if verbose > 0:
                    logging.info("pot=%s, action=%s, obs=%s", str(pot), str(softmax(ACTION_MULTIPLIER * action)), str(obs[0:4]))
                    logging.info("="*10)

                if pot == 0:
                    break
                    
            amounts.append(pot)

        return amounts

        
if __name__ == "__main__":
    from src.NBA.data_api import NBAHistoricalBettingAPI
    api = NBAHistoricalBettingAPI()
    env = SportsBettingOddsEnv(100000, api)
    check_env(env)