import numpy as np
import torch.nn as nn
from torch.distributions.categorical import Categorical
from abc import ABC
from torch.nn import functional as F
from utils import conv_shape
import torch

class PolicyModel(nn.Module, ABC):
    def __init__(self, state_shape, n_actions):
        super(PolicyModel, self).__init__()
        self.state_shape = state_shape
        self.n_actions = n_actions

        color, w, h = state_shape
        conv1_out_shape = conv_shape((w, h), 8, 4)
        conv2_out_shape = conv_shape(conv1_out_shape, 4, 2)
        conv3_out_shape = conv_shape(conv2_out_shape, 3, 1)

        flatten_size = conv3_out_shape[0] * conv3_out_shape[1] * 64

        self.seq = nn.Sequential(
            nn.Conv2d(color, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64,  kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(flatten_size, 256),
            nn.ReLU(),
            nn.Linear(256, 448),
            nn.ReLU(),
        )

        self.policy = nn.Linear(448, self.n_actions)
        self.int_value = nn.Linear(448, 1)
        self.ext_value = nn.Linear(448, 1)

        self.extra_value_fc = nn.Linear(448, 448)
        self.extra_policy_fc = nn.Linear(448, 448)

        for layer in self.seq.children():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                layer.bias.data.zero_()

        nn.init.orthogonal_(self.extra_policy_fc.weight, gain=np.sqrt(0.1))
        self.extra_policy_fc.bias.data.zero_()
        nn.init.orthogonal_(self.extra_value_fc.weight, gain=np.sqrt(0.1))
        self.extra_value_fc.bias.data.zero_()

        nn.init.orthogonal_(self.policy.weight, gain=np.sqrt(0.01))
        self.policy.bias.data.zero_()
        nn.init.orthogonal_(self.int_value.weight, gain=np.sqrt(0.01))
        self.int_value.bias.data.zero_()
        nn.init.orthogonal_(self.ext_value.weight, gain=np.sqrt(0.01))
        self.ext_value.bias.data.zero_()

    def forward(self, state):
        #state = np.divide(state, 255., out=state, casting="unsafe")
        x = state / 255.
        x = self.seq(x)
        x_value = x + F.relu(self.extra_value_fc(x))
        x_pi = x + F.relu(self.extra_policy_fc(x))
        int_value = self.int_value(x_value)
        ext_value = self.ext_value(x_value)
        policy = self.policy(x_pi)
        probs = F.softmax(policy, dim=1)
        dist = Categorical(probs)

        return dist, int_value, ext_value, probs


class TargetModel(nn.Module, ABC):
    def __init__(self, state_shape):
        super(TargetModel, self).__init__()
        self.state_shape = state_shape

        color, w, h = state_shape
        conv1_out_shape = conv_shape((w, h), 8, 4)
        conv2_out_shape = conv_shape(conv1_out_shape, 4, 2)
        conv3_out_shape = conv_shape(conv2_out_shape, 3, 1)

        flatten_size = conv3_out_shape[0] * conv3_out_shape[1] * 64

        self.seq = nn.Sequential(
            nn.Conv2d(color, 32, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64,  kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(flatten_size, 512)
        )


        for layer in self.seq.children():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                layer.bias.data.zero_()

    def forward(self, state):
        return self.seq(state)


class PredictorModel(nn.Module, ABC):
    def __init__(self, state_shape):
        super(PredictorModel, self).__init__()
        self.state_shape = state_shape

        color, w, h = state_shape
        conv1_out_shape = conv_shape((w, h), 8, 4)
        conv2_out_shape = conv_shape(conv1_out_shape, 4, 2)
        conv3_out_shape = conv_shape(conv2_out_shape, 3, 1)

        flatten_size = conv3_out_shape[0] * conv3_out_shape[1] * 64

        self.seq = nn.Sequential(
            nn.Conv2d(color, 32, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64,  kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(flatten_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )


        for layer in self.seq.children():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                layer.bias.data.zero_()

    def forward(self, state):
        return self.seq(state)

class DiscriminatorModel(nn.Module, ABC):
    def __init__(self, encoding_size=512, hidden_size=32, rollout_len=1, n_layers=10):
        super(DiscriminatorModel, self).__init__()
        
        self.rnn = nn.LSTM(input_size=encoding_size, hidden_size=hidden_size, num_layers=n_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(rollout_len * hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.encoding_size = encoding_size
        self.rollout_len = rollout_len
        self.h = torch.randn(n_layers, rollout_len, hidden_size)
        self.c = torch.randn(n_layers, rollout_len, hidden_size)

        for layer in self.fc.children():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                layer.bias.data.zero_()


    def forward(self, rand_encoding, true_encoding):
        y = torch.zeros((len(rand_encoding), 1), requires_grad=True)

        #rand_encoding = rand_encoding.view(-1, self.rollout_len, true_encoding.shape[-1])
        #true_encoding = true_encoding.view(-1, self.rollout_len, true_encoding.shape[-1])

        with torch.no_grad():
            for t in range(0, len(rand_encoding)*self.rollout_len, self.rollout_len):
                lstm_rollout_output, (h_, c_) = self.rnn(rand_encoding[t:t + self.rollout_len], (self.h, self.c))
                lstm_rollout_output = nn.Flatten(start_dim=1)(lstm_rollout_output)
                temp = self.fc(lstm_rollout_output)
                y[t : t+self.rollout_len, ...] = self.fc(lstm_rollout_output)

                _, (self.h, self.c) = self.rnn(true_encoding[t:t + self.rollout_len], (self.h, self.c))

            
            #lstm_out = lstm_output.contiguous().view(-1, self.hidden_dim)
            #lstm_out = nn.Flatten()(lstm_output)

        
        return y


