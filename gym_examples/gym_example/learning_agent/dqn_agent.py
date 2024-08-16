import torch
from torch import nn
import numpy as np

class DQNAgent(nn.Module):
    def __init__(self, state_shape, n_actions, epsilon=0):
        super().__init__()
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.state_shape = state_shape

        state_dim = state_shape[0]
        #simple NN with state_dim as input vector (inout is state s)
        #self.n_actions as output vector of logits of q(s,a)

        self.network = nn.Sequential()
        self.network.add_module('layer1',nn.Linear(state_dim, 192))
        self.network.add_module('relu1',nn.ReLU())
        self.network.add_module('layer2',nn.Linear(192,256))
        self.network.add_module('relu1',nn.ReLU())
        self.network.add_module('layer2',nn.Linear(256,64))
        self.network.add_module('relu1',nn.ReLU())
        self.network.add_module('layer2',nn.Linear(64,n_actions))

        self.parameters = self.network.parameters

    
    def forward(self, state_t):
        qvalues = self.network(state_t)
        return qvalues
    
    def get_qvalues(self, states, device):
        #input array of states in numpy output Qvals as numpy array
        states = torch.tensor(states, device=device, dtype=torch.float32)
        qvalues = self.forward(states)
        return qvalues.data.cpu().numpy()
    
    def sample_actions(self, qvalues):
        #sample actions from batch of q_values using epsilon greedy policy
        epsilon = self.epsilonbatch_size, n_actions = qvalues.shape
        batch_size, n_actions = qvalues.shape
        random_actions = np.random.choice(n_actions, size = batch_size)
        best_actions = qvalues.argmax(axis=-1)
        should_explore = np.random.choice([0,1], batch_size, p = [1-epsilon, epsilon])
        return np.where(should_explore, random_actions, best_actions)
