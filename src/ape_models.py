from cgitb import html
import numpy as np
import torch.nn as nn
from torch.distributions.categorical import Categorical
from abc import ABC
from torch.nn import functional as F
from utils import conv_shape, pool_shape
import torch
from torch import autocast
import pygad
import pygad.torchga
import pygad.cnn

class PolicyModel(nn.Module, ABC):
    def __init__(self, state_shape, n_actions):
        super(PolicyModel, self).__init__()
        self.state_shape = state_shape
        self.n_actions = n_actions

        color, w, h = state_shape
        conv1_out_shape = conv_shape((w, h), 8, 4)
        conv2_out_shape = conv_shape(conv1_out_shape, 4, 2)
        conv3_out_shape = conv_shape(conv2_out_shape, 3, 1)

        flatten_size = conv3_out_shape[0] * conv3_out_shape[1] * 10 # at 12 model loading becomes super slow, is it too large to fit in memory?

        self.seq = nn.Sequential(
            nn.Conv2d(color, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 10,  kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(flatten_size, 64),
            # nn.LazyLinear(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )
        # test = self.seq(torch.ones((1, 4, 84, 84))) # for lazylinear

        self.policy = nn.Linear(128, self.n_actions)
        self.int_value = nn.Linear(128, 1)
        self.ext_value = nn.Linear(128, 1)

        self.extra_value_fc = nn.Linear(128, 128)
        self.extra_policy_fc = nn.Linear(128, 128)

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
        int_value = 0
        ext_value = 0
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
    def __init__(self, encoding_size, hidden_size=6, timesteps=2, n_layers=2, pred_size=1, n_actions=18):
        super(PredictorModel, self).__init__()


        self.encoding_size = encoding_size
        self.timesteps = timesteps
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_actions = n_actions
        self.device =  torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.rnn = nn.GRU(encoding_size + n_actions, self.hidden_size, self.n_layers)
        self.h0 = torch.randn((self.n_layers, self.timesteps, self.hidden_size), dtype=torch.float32)
        self.c0 = torch.randn((self.n_layers, self.timesteps, self.hidden_size), dtype=torch.float32)


        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, encoding_size),
            # nn.Sigmoid()
        )

    def forward(self, true_encoding, actions):

        # input_frame = torch.unsqueeze(true_encoding[:, 0, :], dim=1)
        #actions = torch.unsqueeze(actions, dim=-2)
        frames_back = 4
        device = true_encoding.device
        h0 = torch.ones((self.n_layers, self.timesteps, self.hidden_size), dtype=torch.float32).to(device)

        
        # pred_frame = input_frame
        # frames = input_frame
        # for t in range(self.timesteps - 1):
        #     #action = actions[:, t, ...]
        #     #input = torch.cat((pred_frame, action), dim=-1)
        #     output, h = self.rnn(pred_frame, h)
        #     pred_frame = self.fc(output)

        #     frames = torch.cat((frames, pred_frame), dim=1)

        input_encoding = torch.tile(torch.unsqueeze(true_encoding[:, 0, :], dim=1), (1, self.timesteps, 1))

        total_input = torch.cat((input_encoding, actions), dim=-1)

        
        output, hn = self.rnn(total_input, h0)
        output = self.fc(output)

        return output




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
class PredictorModelRND(nn.Module, ABC):

    def __init__(self, state_shape):
        super(PredictorModelRND, self).__init__()
        self.state_shape = state_shape
        c, w, h = state_shape
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        color, w, h = state_shape
        conv1_out_shape = conv_shape((w, h), 8, 4)
        conv2_out_shape = conv_shape(conv1_out_shape, 4, 2)
        conv3_out_shape = conv_shape(conv2_out_shape, 3, 1)

        flatten_size = conv3_out_shape[0] * conv3_out_shape[1] * 64

        self.fc1 = nn.Linear(in_features=flatten_size, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=512)
        self.encoded_features = nn.Linear(in_features=512, out_features=512)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                layer.bias.data.zero_()
            elif isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                layer.bias.data.zero_()

    def forward(self, inputs):
        x = inputs
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu((self.conv3(x)))
        x = x.contiguous()
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.encoded_features(x)
