import gymnasium as gym
import numpy as np
from gym_example.learning_agent.N_step_SARSA import QEstimator
from gym_example.learning_agent.N_step_SARSA import sarsa_n
from gym_example.learning_agent.lamda_SARSA import sarsa_lambda
import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0')
observation, info = env.reset()


def policy(env):
    return env.action_space.sample()

if __name__ == '__main__':
    episode_num = 1000
    Qest_hat = QEstimator(env, 0.5, num_of_filings = 8, tiles_per_dim =8, max_size=2048, epsilon=0.0)
    
    # rewards = sarsa_n(env, Qest_hat, step_size = 0.5, epsilon = 0.0, n=5, gamma =1, episode_cnt = episode_num)
    rewards = sarsa_lambda(env, Qest_hat, max_size = 2048, gamma =1, episode_cnt = episode_num)

    print('rewards',rewards)
    print('mean reward',np.mean(rewards[-10:]))
    plt.plot(range(episode_num),rewards)
    plt.show()


