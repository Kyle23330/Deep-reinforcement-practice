import torch
import gymnasium as gym
from gym_example.learning_agent.dqn_agent import DQNAgent
from gym_example.learning_agent.DQN_replay_buffer import ReplayBuffer

def play_and_record(start_state, agent, env, exp_replay, n_steps=1):
    s = start_state
    sum_rewards= 0

    #Play the game for n_steps and record transitions in buffer
    for _ in range(n_steps):
        qvalues = agent.get_qvalues([s])
        a = agent.sample_actions(qvalues)[0]
        next_s, r, done, _ = env.step(a)
        sum_rewards += r
        exp_replay.add(s, a, r, next_s, done)
        if done:
            s = env.reset()
        else:
            s = next_s

    return sum_rewards


# compute TD loss in pytorch

def compute_td_loss(agent, target_network, states, actions, rewards, next_states, done_flags, device, gamma=0.99):
    #convert numpy array to torch tensors
    states = torch.tensor(states, device=device, dtype=torch.float)
    actions = torch.tensor(actions, device=device, dtype=torch.long)
    rewards = torch.tensor(rewards, device=device, dtype=torch.float)
    next_states = torch.tensor(next_states, device=device, dtype=torch.float)
    done_flags = torch.tensor(done_flags.astype('float32'), device=device, dtype=torch.float)

    #get q values for all actions in current states
    predicted_qvalues = agent(states)

    #compute q-values for all actions in next states
    predicted_next_qvalues = target_network(next_states)

    #select q-values for chosen actions
    predicted_qvalues_for_actions = predicted_qvalues[range(len(actions)), actions]

    #compute Qmax using predicted next q-values
    next_state_values, _ = torch.max(predicted_next_qvalues, dim=1)

    #compute "target q-values"
    target_qvalues_for_actions = rewards + gamma * next_state_values * (1-done_flags)

    #mean squared error loss to minimize
    loss = torch.mean((predicted_qvalues_for_actions - target_qvalues_for_actions.detach()) ** 2)

    return loss

def epsilon_schedule(start_epsilon, end_epsilon, step, eps_decay_final_step):
    epsilon_decay_val = 0.8
    epsilon = start_epsilon * (0.8 ** step)
    if step <= eps_decay_final_step:
        return epsilon
    else:
        return end_epsilon


env = gym.make('CartPole-v1')
observation, info = env.reset()

total_steps = 1000
state_shape = 4
n_actions = 2
agent = DQNAgent(state_shape, n_actions, epsilon=0)

start_epsilon = 0.9
end_epsilon = 0.001
eps_decay_final_step = 900

exp_replay = ReplayBuffer(buffer_size = 10)
batch_size = 100

if __name__ == '__main__':
    for step in range(total_steps + 1):
        #reduce exploration as progress
        agent.epsilon = epsilon_schedule(start_epsilon, end_epsilon, step, eps_decay_final_step)

        # take timesteps_per_epoch and update experience replay buffer
        _, state = play_and_record(state, agent, env, exp_replay, timesteps_per_epoch = 100)

        #train by sampling batch_size of data from experience replay
        states, actions, rewards, next_states, done_flags = exp_replay.sample(batch_size)

