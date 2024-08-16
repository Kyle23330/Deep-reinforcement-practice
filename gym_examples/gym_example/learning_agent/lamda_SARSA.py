import numpy as np
from gym_example.learning_agent.iht import IHT
from gym_example.learning_agent.iht import tiles
from gym_example.learning_agent.iht import binary_tiling_features

def accumulating_trace(trace, active_features, gamma, lambd):
    trace *= gamma * lambd
    trace[active_features] += 1
    return trace

def replacing_trace(trace, active_features, gamma, lambd):
    trace *= gamma * lambd
    trace[active_features] = 1
    return trace

class QEstimator:
    def __init__(self, env, step_size, num_of_tilings = 8, tiles_per_dim =8, max_size=2048, epsilon=0.0):
        self.max_size = max_size
        self.num_of_tilings = num_of_tilings
        self.tiles_per_dim = tiles_per_dim
        self.epsilon = epsilon
        self.step_size = step_size
        self.table = IHT(max_size)
        self.w = np.zeros(max_size)
        self.pos_scale = self.tiles_per_dim / (env.observation_space.high[0] - env.observation_space.low[0])
        self.vel_scale = self.tiles_per_dim / (env.observation_space.high[1] - env.observation_space.low[1])
        self.env = env
        self.trace = []
    # takes as input continuous two-dimensional values of S and discrete input action A to return the tile-encoded active_feature x(S,A)
    def get_active_features(self, state, action):
        pos, vel = state
        active_features = tiles(self.table, self.num_of_tilings, [self.pos_scale*(pos - self.env.observation_space.low[0]), self.vel_scale*(vel-self.env.observation_space.low[1])], [action])

        return active_features

    def q_predict(self, state, action):
        pos, vel = state
        if pos == self.env.observation_space.high[0]: #reach goal
            return 0.0
        else:
            active_features = self.get_active_features(state, action)
            return np.sum(self.w[active_features])
    
    def q_update(self, action, reward, next_state, next_action):
        #needs work
        return 0

    def get_eps_greedy_action(self, state):
        pos, vel = state
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.env.action_space.n)
        else:
            qvals = np.array([self.q_predict(state, action) for action in range(self.env.action_space.n)])
            return np.argmax(qvals)
        
    ###
def sarsa_lambda(env, qhat, max_size = 2048, gamma =1, episode_cnt = 10000, lambd = 0.5):
    episode_rewards = []

    for _ in range(episode_cnt):
        state,info = env.reset()
        # print('state',state)
        action = qhat.get_eps_greedy_action(state)
        qhat.trace = np.zeros(max_size)
        episode_reward = 0
        count = 0
        done = False
        while not done:
            # print('counttttttttt',count)
            # print('current state',state)
            # print('current action',action)
            observation, reward, done, _, _  = env.step(action)
            next_state = observation
            # print('next_state',next_state)
            next_action = qhat.get_eps_greedy_action(next_state)
            # print('next_action',next_action)
            episode_reward += reward
            # print('reward',reward)
            # print('epireward',episode_reward)
            # qhat.update(state, action, reward, next_state, next_action)
            delta = reward
            active_indices = qhat.get_active_features(state, action)

            # print('active_indices',active_indices)
            for active_indice in active_indices:
                delta = delta - qhat.w[active_indice]
                qhat.trace[active_indice] += 1 # accumulating trace

            if done:
                # print('before done update',qhat.q_predict(state, action))
                qhat.w += qhat.step_size * delta * qhat.trace / qhat.num_of_tilings
                # print('after done update',qhat.q_predict(state, action))
                episode_rewards.append(episode_reward)
                break

            active_indices = qhat.get_active_features(next_state, next_action)
            for active_indice in active_indices:
                delta = delta + gamma * qhat.w[active_indice]
            
            # print('before w update update',qhat.q_predict(state, action))
            qhat.w = qhat.w + qhat.step_size * delta * qhat.trace / qhat.num_of_tilings
            # print('after w update update',qhat.q_predict(state, action))
            qhat.trace = gamma*lambd*qhat.trace
            state = next_state
            action = next_action
            count += 1
    return np.array(episode_rewards)


    




