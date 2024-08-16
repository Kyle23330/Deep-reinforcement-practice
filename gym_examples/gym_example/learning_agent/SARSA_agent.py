from collections import defaultdict
import numpy as np

class SARSAAgent:
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

    def update(self,state,action,reward,next_state,next_action,done):
        if not done:
            td_error = reward + self.gamma*self.get_Q(next_state, next_action) - self.get_Q(state,action)
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