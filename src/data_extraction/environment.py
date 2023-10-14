import gym
from gym import spaces
import numpy as np
from odds import Odds

def clamp(low, high, value):
    if value < low:
        return low

    if value > high:
        return high

    return value

WIN_STD = 0.1

EPS_MU = 0.00
EPS_STD = 0.00

LAMBDA_GAINS = 1
LAMBDA_LOSSES = -1

ZERO_REWARD = -100

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

class BettingEnvironment(gym.Env):
    metadata = {'render.modes': ['console', 'none']}

    def __init__(self, bias_factor, g_0, episode_length): # takes some variable that biases the betting odds o_a, o_b
        super(BettingEnvironment, self).__init__()
        self.action_space = spaces.Box(low=np.array([-1,-1,-1]), high=np.array([1,1,1]), shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([0, 1, 1, 0, 0]), high=np.array([np.inf, np.inf, np.inf, 1, 1]),
                                        shape=(5,), dtype=np.float32)

        self.g_0 = g_0
        self.bias_factor = bias_factor
        self.g = g_0
        self.episode_length = episode_length
        self.iterations = 0

    def setup_next_game(self):
        self.p_pred = clamp(0, 1, np.random.normal(0.5, WIN_STD))
        self.eps = clamp(0, 1, abs(np.random.normal(EPS_MU, EPS_STD)))
        self.p_gt = clamp(0, 1, np.random.normal(self.p_pred, self.eps))
        odds_percent = clamp(0.01, 0.99, np.random.normal(0, self.bias_factor) + self.p_gt)
        self.o_a: Odds = Odds.from_win_percentage(odds_percent)
        self.o_b: Odds = Odds.from_win_percentage(1 - odds_percent)
        self.o = [self.o_a, self.o_b]

    def calculate_winner(self):
        return np.random.choice([0,1], p=[self.p_gt, 1-self.p_gt]) 

    def step(self, action):
        action = self.g * softmax(action*4)
        self.iterations += 1
        # get winnings
        winning_team = self.calculate_winner()
        self.gains = action[winning_team] * self.o[winning_team].get_winnings_multiplier()
        self.losses = action[1 - winning_team]
        self.g = self.g + self.gains - self.losses

        reward = LAMBDA_GAINS * self.gains + LAMBDA_LOSSES * self.losses

        self.setup_next_game()
        self.observation = np.array([self.g, self.o_a.get_decimal(), self.o_b.get_decimal(), self.eps, self.p_pred], dtype=np.float32)

        done = self.g <= 0 or self.iterations >= self.episode_length
        
        self.info = {}

        return self.observation, reward, done, self.info
        
    def reset(self):
        self.g = self.g_0
        self.iterations = 0

        self.setup_next_game()

        self.observation = np.array([self.g, self.o_a.get_decimal(), self.o_b.get_decimal(), self.eps, self.p_pred], dtype=np.float32)


        return self.observation  # reward, done, info can't be included

    def render(self, mode='console'):
        if mode == 'console':
            if self.iterations == 0:
                print("{} {} {} {} {} {}".format(self.iterations, self.g, self.o_a, self.o_b, self.p_pred, self.eps))
            else:
                print("{} {} {} {} {} {} | {} {} | {}".format(self.iterations, self.g, self.p_pred, self.o_a, self.o_b, self.eps, self.gains, self.losses, self.p_gt))

    def close (self):
        pass

if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env
    env = BettingEnvironment(1, 1, 100)
    check_env(env)