import numpy as np
from gym_example.learning_agent.iht import IHT
from gym_example.learning_agent.iht import tiles
from gym_example.learning_agent.iht import binary_tiling_features

class QEstimator:
    def __init__(self, env, step_size, num_of_filings = 8, tiles_per_dim =8, max_size=2048, epsilon=0.0):
        self.max_size = max_size
        self.num_of_tilings = num_of_filings
        self.tiles_per_dim = tiles_per_dim
        self.epsilon = epsilon
        self.step_size = step_size
        self.table = IHT(max_size)
        self.w = np.zeros(self.max_size)
        self.pos_scale = self.tiles_per_dim / (env.observation_space.high[0] - env.observation_space.low[0])
        self.vel_scale = self.tiles_per_dim / (env.observation_space.high[1] - env.observation_space.low[1])
        self.env = env
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
    
    def q_update(self, state, action, target):
        active_features = self.get_active_features(state, action)

        q_s_a = np.sum(self.w[active_features])
        # print('qsa',q_s_a)
        delta = (target - q_s_a)
        # print('update features',active_features)
        # print('before update predict', self.q_predict(state, action))
        # print('before update',self.w[active_features])
        self.w[active_features] += self.step_size*delta/(self.num_of_tilings - 0)
        # print('after update',self.w[active_features])
        # print('after update predict', self.q_predict(state, action))

    def get_eps_greedy_action(self, state):
        pos, vel = state
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.env.action_space.n)
        else:
            qvals = np.array([self.q_predict(state, action) for action in range(self.env.action_space.n)])
            return np.argmax(qvals)
        
    ###
def sarsa_n(env, qhat, step_size = 0.5, epsilon = 0.0, n=1, gamma =1, episode_cnt = 1000):
    episode_rewards = []

    for epi_count in range(episode_cnt):
        # print('episode done', epi_count)
        state,info = env.reset()
        # print('starting state',state)
        action = qhat.get_eps_greedy_action(state)
        # print('starting_action',action)
        T = float('inf')
        t = 0
        states = [state]
        actions = [action]
        rewards = [0.0]
        count = 0
        done = False
        while True:
            print('countttttttttttttttttttttttttttt',count)
            if t < T:
                observation, reward, done, _, _  = env.step(action)
                next_state = observation
                # print('next_state',next_state)
                # print('reward',reward)
                # print('t',t)
                states.append(next_state)
                rewards.append(reward)

                if done:
                    T = t + 1
                else:
                    next_action = qhat.get_eps_greedy_action(next_state)
                    # print('next_action',next_action)
                    actions.append(next_action)
            tau = t - n + 1
            # print('tau',tau)

            if tau >= 0:
                G = 0
                for i in range(tau+1, min(tau+n, T) + 1):
                    # print('G1 check',i,tau)
                    G+= gamma ** (i-tau-1) * rewards[i]
                    # print('G part 1',G)
                if tau+n < T:
                    # print('qhat predict for G end', qhat.q_predict(states[tau+n],actions[tau+n]))
                    G += gamma**n * qhat.q_predict(states[tau+n],actions[tau+n])
                # print('states[tau]',states[tau])
                # print('actions[tau]',actions[tau])
                # print('G part 2',G)
                qhat.q_update(states[tau], actions[tau], G)
            
            if tau == T - 1:
                # print('append condition')
                episode_rewards.append(np.sum(rewards))
                break
            else:
                t += 1
                # state = next_state
                action = next_action
            count += 1
            # print('states[]',states)
            # print('actions[]',actions)
            # print('rewards[]',rewards)
            # print('w',qhat.w)
    return np.array(episode_rewards)


    




