import logging
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np

from src.data_api import HistoricalBettingDataAPI
from src.sports_betting_odds_env import SportsBettingOddsEnv


logging.basicConfig(level=logging.DEBUG)

# notation
# tn = team n, ex. t1 = team 1
# p[w = t1] = prob team 1 wins

train_test_split = 0.8
initial_amount = 1
test_trials = 20
episode_length = 100
training_steps = episode_length * 100

def evaluate_betting_odds_model(model, test_env: SportsBettingOddsEnv, trials: int):
    env = test_env
    rewards = []
    for _ in range(trials):
        done = False
        obs = env.reset()
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

        rewards.append(env.pot)

    obs = env.reset()
    

    return sum(rewards) / len(rewards)

def train_sports_betting_odds_env(betting_odds_api: HistoricalBettingDataAPI):
    logging.info("starting training on raw sports betting odds...")

    betting_odds_api.shuffle()

    split_index = int(train_test_split * len(betting_odds_api))
    betting_odds_training = betting_odds_api[:split_index]
    # betting_odds_training = betting_odds_api.copy()
    # betting_odds_training.set_end(split_index)
    # betting_odds_testing = betting_odds_api.copy()
    # betting_odds_testing.set_start(split_index)
    betting_odds_testing = betting_odds_api[split_index:]

    env = SportsBettingOddsEnv(initial_amount, betting_odds_training, episode_length=episode_length)

    # be able to split into training and testing

    env = make_vec_env(lambda: env, n_envs=1)

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
    model.learn(total_timesteps=training_steps, log_interval=1)

    # env = model.get_env()
    env = SportsBettingOddsEnv(1, betting_odds_testing, episode_length=episode_length)
    logging.info('finished')
    obs = env.reset()

    logging.info('starting eval...')
    logging.info('eval results %.5e', evaluate_betting_odds_model(model, env, test_trials))
    logging.info('press enter...')

    input()

    while True:
        env.render('console')

        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(np.array([0,0,0]))
        # obs, rewards, dones, info = env.step(action)

        if dones:
            env.render('console')

            input()
            env.reset()

    # obs = env.reset()
    # env.render()

    # print(obs)
    # print(env.observation_space)
    # print(env.action_space)
    # print(env.action_space.sample())

    # # Hardcoded best agent: always go left!
    # n_steps = 1000
    # for step in range(n_steps):
    #     print("Step {}".format(step + 1))
    #     obs, reward, done, info = env.step(env.action_space.sample())
    #     # obs, reward, done, info = env.step(np.array([1,1,-1]))
    #     print('obs=', obs, 'reward=', reward, 'done=', done)
    #     env.render()
    #     if done:
    #         print("Goal reached!", "reward=", reward)
    #         break





if __name__ == "__main__":
    from src.NBA.data_api import NBAHistoricalBettingAPI
    api = NBAHistoricalBettingAPI()
    # api.set_end(10)
    api = api
    train_sports_betting_odds_env(api)