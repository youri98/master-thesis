from abc import ABC
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.distributions.categorical import Categorical
# from torchsummary import summary
import torch
from blitz.modules import BayesianLinear, BayesianConv2d


class PolicyModel(nn.Module, ABC):
    def __init__(self, state_shape, n_actions, n_neurons=128):
        super(PolicyModel, self).__init__()
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.n_neurons = n_neurons

        self.seq = nn.Sequential(
            nn.Linear(2, self.n_neurons),
        )

        self.policy = nn.Linear(self.n_neurons, self.n_actions)
        self.int_value = nn.Linear(self.n_neurons, 1)
        self.ext_value = nn.Linear(self.n_neurons, 1)

        self.extra_value_fc = nn.Linear(self.n_neurons, self.n_neurons)
        self.extra_policy_fc = nn.Linear(self.n_neurons, self.n_neurons)

        # for layer in self.seq.children():
        #     if isinstance(layer, nn.Linear):
        #         nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                # layer.bias.data.zero_()

        # nn.init.orthogonal_(self.extra_policy_fc.weight, gain=np.sqrt(0.1))
        # # self.extra_policy_fc.bias.data.zero_()
        # nn.init.orthogonal_(self.extra_value_fc.weight, gain=np.sqrt(0.1))
        # # self.extra_value_fc.bias.data.zero_()

        # nn.init.orthogonal_(self.policy.weight, gain=np.sqrt(0.01))
        # # self.policy.bias.data.zero_()
        # nn.init.orthogonal_(self.int_value.weight, gain=np.sqrt(0.01))
        # # self.int_value.bias.data.zero_()
        # nn.init.orthogonal_(self.ext_value.weight, gain=np.sqrt(0.01))
        # self.ext_value.bias.data.zero_()

    def forward(self, state):
        #state = np.divide(state, 255., out=state, casting="unsafe")
        # print("Inside ", state.shape)
        x = state / 255.
        x = self.seq(x)
        x_value = self.extra_value_fc(x)
        x_pi = self.extra_policy_fc(x)
        int_value = self.int_value(x)
        ext_value = self.ext_value(x)
        policy = self.policy(x)
        
        if self.n_actions != 1:
            probs = F.softmax(policy, dim=1)
        else:
            probs = policy
        # dist = Categorical(probs)
        result = [int_value, ext_value, probs]

        return result  # probs, int_value, ext_value, probs


class TargetModel(nn.Module, ABC):

    def __init__(self, state_shape):
        super(TargetModel, self).__init__()
        self.state_shape = state_shape

        self.fc1 = nn.Linear(in_features=2, out_features=1000)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)

        # for layer in self.modules():
        #     if isinstance(layer, nn.Conv2d):
        #         nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
        #         # layer.bias.data.zero_()
        #     elif isinstance(layer, nn.Linear):
        #         nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
        #         # layer.bias.data.zero_()

        self.seq = nn.Sequential(self.fc1,
                                
                                 )
    def forward(self, inputs, k_samples=None):
        return self.seq(inputs)


class PredictorModel(nn.Module, ABC):

    def __init__(self, state_shape):
        super(PredictorModel, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.fc1 = nn.Linear(in_features=2, out_features=1000)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(in_features=512, out_features=512)

        # for layer in self.modules():
        #     if isinstance(layer, nn.Linear):
        #         nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                # layer.bias.data.zero_()

        self.seq = nn.Sequential(
                                 self.fc1,
                                 )


    def forward(self, inputs):
        return self.seq(inputs)


