import numpy as np
import sys
# from gym.envs.toy_text import discrete
import gym.envs
from gym import spaces
from contextlib import closing
from io import  StringIO
import pygame
from gym_example.env import cliff_world_env
from gym_example.learning_agent.SARSA_agent import SARSAAgent
from gym_example.learning_agent.Q_learning_agent import QLearningAgent
from gym_example.learning_agent.expected_sarsa import ExpectedSARSAAgent
from gym_example.learning_agent.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt



class CliffWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.size = (4,10)  # The size of the square grid
        self.window_size = (200,500)  # The size of the PyGame window
        self.policy = np.ones((1,40,4))
        self.policy = np.multiply(self.policy, 0.25)
        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, 9, shape=(2,), dtype=int),
                "target": spaces.Box(0, 9, shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}
    
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = np.array([3, 0])

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = np.array([3,9])

        observation = self._get_obs()
        info = self._get_info()

        # if self.render_mode == "human":
        #     self._render_frame()

        return observation, info
    
    def step(self,action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        rand_action = np.random.randint(4)
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, [0,0], [3,9]
        )
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        cliff_falled = self.cliff_fall_check()
        reward = -100 if cliff_falled else -1  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        # if self.render_mode == "human":
        #     self._render_frame()

        return observation, reward, terminated, False, info
    
    def cliff_fall_check(self):
        on_cliff_level =  self._agent_location[0] == 3
        on_cliff_range = self._agent_location[1] >= 1 and self._agent_location[1] <= 8
        cliff_fall = on_cliff_level and on_cliff_range
        if cliff_fall:
            self._agent_location = np.array([0,3]) #send back to starting location
        return cliff_fall
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    
    def get_action_to_direction(self):
        return self._action_to_direction
    
    #training algorithm with reply buffer
    def train_agent(self, agent, episode_cnt=1000, tmax=1000, anneal_eps=True, replay_buffer = None, batch_size=16):

        episode_rewards = []
        for i in range(episode_cnt):
            G = 0
            obs, info = self.reset()
            state = obs["agent"]
            for t in range(tmax):
                action = agent.get_action(state)
                observation, reward, done, _, _ = self.step(action)
                next_state = observation["agent"]
                if replay_buffer:
                    replay_buffer.add(state, action, reward, next_state, done)
                    states, actions, rewards, next_states, done_flags = replay_buffer.sample(batch_size)
                    for i in range(batch_size):
                        agent.update(states[i], actions[i], rewards[i], next_states[i], done_flags[i])
                else:
                    agent.update(state, action, reward, next_state, done)
                G+=reward
                if done:
                    episode_rewards.append(G)
                    if anneal_eps:
                        agent.epsilon = agent.epsilon * 0.99
                    break
                state = next_state
        
        return np.array(episode_rewards)
        
    
        


if __name__ == '__main__':
    num_episodes = 100
    actions_at_each_state = np.array([0, 1, 2, 3])
    actions_all_states = np.tile(actions_at_each_state,(4,10,1))

    env = gym.make('gym_example/CliffWorldEnv-v0',render_mode = "human")
    replay_buffer = ReplayBuffer(10)
    expected_sarsa_agent = ExpectedSARSAAgent(alpha = 0.9, epsilon = 0.05, gamma=0.9, get_possible_actions = actions_all_states)
    Q_learning_agent = QLearningAgent(alpha = 0.9, epsilon = 0.05, gamma=0.9, get_possible_actions = actions_all_states)
    all_rewards = env.train_agent(Q_learning_agent,replay_buffer = replay_buffer)

    mean_reward = np.mean(all_rewards[-100:])
    # sarsa_agent.print_Q()
    print('mean reward',mean_reward)
    plt.plot(range(1000),all_rewards)
    plt.show()
