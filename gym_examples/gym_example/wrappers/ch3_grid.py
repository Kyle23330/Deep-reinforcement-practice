from gym_example.env import grid_world_env
import gym
import numpy as np
import pygame

np.random.seed(0)

class ch3_grid_wrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        size = env.size
        self.policy = np.ones((1,16,4))
        self.policy = np.multiply(self.policy, 0.25)
        self._action_to_direction = {
            0: np.array([-1, 0]),
            1: np.array([0, 1]),
            2: np.array([1, 0]),
            3: np.array([0, -1]),
        }

    def step(self):
        #take a random direction
        random_action = np.random.randint(4)
        print('choice: ', random_action)
        observation = self._get_obs()
        info = self._get_info()
        direction = self._action_to_direction[random_action]

        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        terminated = np.any(np.all(self._agent_location == self._target_location, axis=1))
        reward = -1

        if self.render_mode == "human":
            self._render_frame()
    
        return observation, reward, terminated, False, info
    
    def reset(self, seed=None, options=None):
        # super().reset(seed=seed)

        #set target location at top left and bottom right corner
        self._target_location = np.array([[0,0],[self.env.size-1,self.env.size-1]])

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the agent's location randomly until it does not coincide with the target's location
        self._agent_location = self._target_location[0]
        while np.any(np.all(self._target_location == self._agent_location, axis=1)):
            self._agent_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}
    
    def _get_info(self):
        return {"distance to be determined"}

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
                pix_square_size * self._target_location[0],
                (pix_square_size, pix_square_size),
            ),
        )
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location[1],
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
        
    def policy_eval_equal_prob(self, discount_factor=1.0,theta=0.0001):
        V = np.zeros(16)
        V_new = np.copy(V)
        r = 0
        while True:
            delta = 0
            for s in range(1,15):
                v_update_holder = 0
                position = np.unravel_index(s, (4,4))
                for key in self._action_to_direction:
                    reward = self.action_reward(position, self._target_location, key)
                    next_position = self.next_position(position, key)

                    v_update_holder += 0.25*1*(reward + discount_factor * V[np.ravel_multi_index(next_position,(4,4))])
                    # print('s:',s,' position',position,' action:',key,'next_position',next_position)
                V_new[s] = v_update_holder
                delta = max(delta, np.abs(V_new[s] - V[s]))
                print('s:',s,'V_new[s]', 'delta: ',delta)
            V = np.copy(V_new)
            r += 1
            if delta < theta:
                break
        
        return np.array(V)
    
    def policy_iteration_greedy_action(self, discount_factor=1.0, theta=0.0001):
        V_iter = np.zeros(16)
        policy_changed = True
        b = 0
        while policy_changed:
            V_iter = self.policy_eval_greedy_action()
            policy_changed = self.policy_improvement(V_iter, discount_factor)
            print('b',b)
            print('V:',V_iter)
            # print('policy:',self.policy)
            # print('policy_changed:',policy_changed)
            b += 1
        
        return V_iter



    def policy_eval_greedy_action(self, discount_factor=1.0,theta=0.0001, V = np.zeros(16)):
        V_new = np.copy(V)
        r = 0
        while True:
            delta = 0
            for s in range(1,15):
                v_update_holder = 0
                position = np.unravel_index(s, (4,4))
                for key in self._action_to_direction:
                    reward = self.action_reward(position, self._target_location, key)
                    next_position = self.next_position(position, key)

                    v_update_holder += self.policy[0,s,key]*1*(reward + discount_factor * V[np.ravel_multi_index(next_position,(4,4))])
                V_new[s] = v_update_holder
                delta = max(delta, np.abs(V_new[s] - V[s]))
            r += 1
            # print('V',V)
            # print('V_new',V_new)
            V = np.copy(V_new)
            if delta < theta:
                break
        print('r',r)
        return V

    def greedy_action(self, current_position_44):
        #find the action with maximum policy prob
        #assume current_position in (4,4)
        current_position_16 = np.ravel_multi_index(current_position_44,(4,4)) #(1,16)
        return np.argmax(self.policy[0, current_position_16])
    
    def policy_improvement(self, V, discount_factor):
        new_policy = np.zeros((1,16,4))
        for s in range(1,15):
            s_44 = np.unravel_index(s, (4,4))
            max_act = self.find_greedy_policy(s_44, V, discount_factor)
            new_policy[0,s] = 0
            new_policy[0,s,max_act] = 1/max_act.size
        
        policy_same = np.array_equal(self.policy,new_policy)
        print('self_policy',self.policy)
        print('new policy',new_policy)
        policy_changed = not policy_same
        print('policy_changed',policy_changed)
        self.policy = np.copy(new_policy)
        return policy_changed
    
    # to find the actio n that maxmize r + yV(s') in the current state
    def find_greedy_policy(self, current_position, V, discount_factor):
        #assume current_position is within (4,4)
        current_position_action_values = np.zeros((4,))
        for key in self._action_to_direction:
            current_position_action_values[key] = self.action_value_eval(current_position, key, V, discount_factor)
        max_value = np.max(current_position_action_values)
        max_indices = np.where(current_position_action_values == max_value)[0]
        return max_indices
    
    def action_value_eval(self, current_position_44, action, V, discount_factor):
        next_position_44 = self.next_position(current_position_44, action)
        reward = self.action_reward(next_position_44, self._target_location, action)
        next_position_16 = np.ravel_multi_index(next_position_44,(4,4))
        action_value = reward + discount_factor * V[next_position_16]
        return action_value

        

    def action_reward(self, position_44, target_positions, action):
        next_location = self.next_position(position_44, action)
        terminated = np.any(np.all(next_location == target_positions, axis=1))
        return -1
    
    def next_position(self, current_position_44, action):
        direction = self._action_to_direction[action]
        next_location = np.clip(current_position_44 + direction, 0, self.env.size - 1)
        return next_location


                        

                    





if __name__ == '__main__':
    print('running')
    env = gym.make('gym_example/GridWorldEnv-v0',render_mode = "human", size = 4)
    env = ch3_grid_wrapper(env)
    obs, info = env.reset()
    # observation, reward_step1, terminated, truncated, info = env.step()
    total_reward = 0
    num_actions = 0
    terminated = False

    # while not terminated:
    #     observation, reward_change, terminated, truncated, info = env.step()
    #     num_actions = num_actions + 1
    #     total_reward = total_reward + reward_change
    
    # print('total reward ',total_reward)
    # print('total actions',num_actions)

    # final_V = env.policy_eval_greedy_action()
    final_V = env.policy_iteration_greedy_action()
    print('final V: \n', final_V)
