from collections import defaultdict
import numpy as np
import gym
import matplotlib.pyplot as plt

class ExpectedSARSAAgent:
    def __init__(self, alpha, epsilon, gamma, get_possible_actions):
        self.get_possible_actions = get_possible_actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self._Q = defaultdict(lambda: defaultdict(lambda: 0))
    
    def get_Q(self,state,action):
        state_tuple = tuple(state)
        return self._Q[state_tuple][action]
    
    def set_Q(self, state, action, value):
        state_tuple = tuple(state)
        self._Q[state_tuple][action] = value

    def update(self,state,action,reward,next_state,done):
        if not done:
            best_next_action = self.max_action(next_state)
            actions = self.get_possible_actions[next_state[0],next_state[1]]
            next_q = 0
            for next_action in actions:
                if next_action == best_next_action:
                    next_q += (1-self.epsilon+self.epsilon/len(actions))*self.get_Q(next_state, next_action)
                else:
                    next_q += (self.epsilon/len(actions))*self.get_Q(next_state,next_action)
            
            td_error = reward + self.gamma * next_q - self.get_Q(state,action)
        else:
            td_error = reward - self.get_Q(state,action)
    
        new_value = self.get_Q(state,action) + self.alpha*td_error
        self.set_Q(state, action, new_value)

    def max_action(self, state):
        # actions = self.get_possible_actions(state)
        actions = self.get_possible_actions[state[0],state[1]]
        best_action = []
        best_q_value = float("-inf")

        for action in actions:
            q_s_a = self.get_Q(state, action)
            if q_s_a > best_q_value:
                best_action = [action]
                best_q_value = q_s_a
            elif q_s_a == best_q_value:
                best_action.append(action)
        return np.random.choice(np.array(best_action))
    
    # choose action as per epi_greedy policy
    def get_action(self, state):
        actions = self.get_possible_actions[state[0],state[1]]
        if len(actions) == 0:
            return None
        
        if np.random.random() < self.epsilon:
            a = np.random.choice(actions)
            return a
        else:
            a = self.max_action(state)
            return a
    
    def print_Q(self):
        print('Q:',self._Q)

if __name__ == '__main__':
    num_episodes = 100

    env = gym.make('gym_example/CliffWorldEnv-v0',render_mode = "human")
    obs, info = env.reset()
    state = obs["agent"]
    total_reward = 0
    num_actions = 0
    terminated = False
    actions_at_each_state = np.array([0, 1, 2, 3])
    actions_all_states = np.tile(actions_at_each_state,(4,10,1))
    # sarsa_agent = SARSAAgent(alpha = 0.2, epsilon = 0.05, gamma=0.9, get_possible_actions = actions_all_states)
    # Q_learning_agent = QLearningAgent(alpha = 0.2, epsilon = 0.05, gamma=0.9, get_possible_actions = actions_all_states)\
    expected_sarsa_agent = ExpectedSARSAAgent(alpha = 0.2, epsilon = 0.05, gamma=0.9, get_possible_actions = actions_all_states)
    step = 0
    all_rewards  = []
    action_to_direction = env.get_action_to_direction()

    for episodes in range(1000):
        obs, info = env.reset()
        cur_state = obs["agent"]
        total_reward = 0
        num_actions = 0
        terminated = False
        step = 0
        while not terminated:
            action = expected_sarsa_agent.get_action(state = cur_state)
            observation, reward_change, terminated, truncated, info = env.step(action)

            next_state = observation["agent"]
            action_next = expected_sarsa_agent.get_action(next_state)
            expected_sarsa_agent.update(cur_state,action,reward_change,next_state,terminated)

            num_actions = num_actions + 1
            total_reward = total_reward + reward_change

            # print('action',action)
            # print('action_next',action_next)
            # print('state',state)
            # print('state_next',next_state)
            # print('total reward',total_reward)

            cur_state = next_state
            step += 1
    
        all_rewards.append(total_reward)
    # save all workspace variables
    # workspace_variables = {name: value for name, value in globals().items() if not name.startswith('__') and not callable(value)}
    # Save variables to a file
    # with open('workspace.pkl', 'wb') as file:
    #     pickle.dump(workspace_variables, file)
    mean_reward = np.mean(all_rewards[-100:])
    # sarsa_agent.print_Q()
    print('mean reward',mean_reward)
    plt.plot(range(1000),all_rewards)
    plt.show()