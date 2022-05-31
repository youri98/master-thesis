import numpy as np
import torch.nn as nn
from torch.distributions.categorical import Categorical
from abc import ABC
from torch.nn import functional as F
from utils import conv_shape
import torch
from torch import autocast

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

        return int_value, ext_value, probs


class TargetModel(nn.Module, ABC):
    def __init__(self, state_shape, encoding_size):
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
            nn.Linear(flatten_size, encoding_size)
        )


        for layer in self.seq.children():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                layer.bias.data.zero_()

    # @autocast(device_type="cpu")
    def forward(self, state):
        return self.seq(state)

class PredictorModel(nn.Module, ABC):
    def __init__(self, state_shape, encoding_size):
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
            nn.Linear(512, encoding_size),
        )


        for layer in self.seq.children():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                layer.bias.data.zero_()
    
    #@autocast(device_type="cpu")
    def forward(self, state):
        return self.seq(state)

class DiscriminatorModel(nn.Module, ABC):
    def __init__(self, encoding_size, hidden_layers=6, timesteps=2, n_layers=1, pred_size=1, n_actions=18):
        super(DiscriminatorModel, self).__init__()

        self.timesteps = timesteps
        self.hidden_layers = hidden_layers
        self.n_layers = n_layers
        self.n_actions = n_actions
        self.device =  torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.rnn = nn.LSTM(encoding_size + n_actions, self.hidden_layers, self.n_layers)
        self.h0 = torch.randn((self.n_layers, self.timesteps, self.hidden_layers), dtype=torch.float32)
        self.c0 = torch.randn((self.n_layers, self.timesteps, self.hidden_layers), dtype=torch.float32)


        self.fc = nn.Sequential(
            nn.Linear(self.hidden_layers, 256),
            nn.ReLU(),
            nn.Linear(256, pred_size),
            nn.Sigmoid()
        )

    #@autocast(device_type="cpu")
    def forward(self, rand_encoding, actions, true_encoding):
        # self.device =  torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


        input_t = torch.cat((rand_encoding, actions), dim=-1)
        true_input_t = torch.cat((true_encoding, actions), dim=-1)

        device = input_t.device
        
        h0 = torch.ones((self.n_layers, self.timesteps, self.hidden_layers), dtype=torch.float32).to(device)
        c0 = torch.ones((self.n_layers, self.timesteps, self.hidden_layers), dtype=torch.float32).to(device)

        # print("input: ", input_t.device)
        # print("h0: ", self.h0.device)

        output, (h_n, c_n) = self.rnn(input_t, (h0, c0))
        output = self.fc(output)

        with torch.no_grad():
            _, (h0, c0) = self.rnn(true_input_t, (h0, c0))

        return output



class DiscriminatorModelGRU(nn.Module, ABC):
    def __init__(self, encoding_size, hidden_layers=128, timesteps=8, n_layers=3, pred_size=1, n_actions=18):
        super(DiscriminatorModelGRU, self).__init__()

        self.timesteps = timesteps
        self.hidden_layers = hidden_layers
        self.n_layers = n_layers
        self.n_actions = n_actions
        self.device =  torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.rnn = nn.GRUCell(encoding_size + n_actions, self.hidden_layers).to(self.device)
        self.h = torch.randn((self.hidden_layers), dtype=torch.float32).to(self.device)

        self.fc = nn.Sequential(
            nn.Linear(self.hidden_layers, 256),
            nn.ReLU(),
            nn.Linear(256, pred_size),
            nn.Sigmoid()
        ).to(self.device)

    #@autocast(device_type="cpu")
    def forward(self, rand_encoding, actions, true_encoding):
        input_t = torch.cat((rand_encoding, actions), dim=-1)
        true_input_t = torch.cat((true_encoding, actions), dim=-1)

        input_t = input_t.view(input_t.shape[0]*input_t.shape[1], input_t.shape[-1])
        true_input_t = true_input_t.view(true_input_t.shape[0]*true_input_t.shape[1], true_input_t.shape[-1])

        outputs = torch.zeros(len(input_t))
    
        for i in range(len(input_t)):
            h = self.rnn(input_t[i], self.h)
            outputs[i] = self.fc(h)

            with torch.no_grad():
                self.h = self.rnn(true_input_t[i], self.h)

        return outputs



# class DecoderModel(nn.Module, ABC):
#     def __init__(self, encoding_shape=512, state_shape=(84,84,1)):
#         super(DecoderModel, self).__init__()
#         self.state_shape = state_shape

#         self.seq = nn.Sequential(
#             nn.Linear(encoding_shape, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, 84*84),
#             nn.Sigmoid()
#         )
#         for layer in self.seq.children():
#             if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
#                 nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
#                 layer.bias.data.zero_()

#     def forward(self, encoding):
#         reconstructed = self.seq(encoding)
#         reconstructed = reconstructed.view(-1, *self.state_shape)
#         return reconstructed
