from abc import ABC
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.distributions.categorical import Categorical
# from torchsummary import summary
import torch
from blitz.modules import BayesianLinear, BayesianConv2d


def conv_shape(input_dims, kernel_size, stride, padding=0):
    return ((input_dims[0] + 2 * padding - kernel_size) // stride + 1,
            (input_dims[1] + 2 * padding - kernel_size) // stride + 1)


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
        # print("Inside ", state.shape)
        x = state / 255.
        x = self.seq(x)
        x_value = x + F.relu(self.extra_value_fc(x))
        x_pi = x + F.relu(self.extra_policy_fc(x))
        int_value = self.int_value(x_value)
        ext_value = self.ext_value(x_value)
        policy = self.policy(x_pi)
        probs = F.softmax(policy, dim=1)
        # dist = Categorical(probs)
        result = [int_value, ext_value, probs]

        return result  # probs, int_value, ext_value, probs


class TargetModel(nn.Module, ABC):

    def __init__(self, state_shape):
        super(TargetModel, self).__init__()
        self.state_shape = state_shape

        c, w, h = state_shape
        self.conv1 = nn.Conv2d(
            in_channels=c, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1)

        color, w, h = state_shape
        conv1_out_shape = conv_shape((w, h), 8, 4)
        conv2_out_shape = conv_shape(conv1_out_shape, 4, 2)
        conv3_out_shape = conv_shape(conv2_out_shape, 3, 1)

        flatten_size = conv3_out_shape[0] * conv3_out_shape[1] * 64

        self.fc = nn.Linear(in_features=flatten_size, out_features=512)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                layer.bias.data.zero_()
            elif isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                layer.bias.data.zero_()

        self.seq = nn.Sequential(self.conv1,
                                 nn.LeakyReLU(),
                                 self.conv2,
                                 nn.LeakyReLU(),
                                 self.conv3,
                                 nn.LeakyReLU(),
                                 nn.Flatten(),
                                 self.fc)

    def forward(self, inputs):
        return self.seq(inputs)


class PredictorModel(nn.Module, ABC):

    def __init__(self, state_shape, mcdropout=0):
        super(PredictorModel, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.state_shape = state_shape
        c, w, h = state_shape
        self.conv1 = nn.Conv2d(
            in_channels=c, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1)

        color, w, h = state_shape
        conv1_out_shape = conv_shape((w, h), 8, 4)
        conv2_out_shape = conv_shape(conv1_out_shape, 4, 2)
        conv3_out_shape = conv_shape(conv2_out_shape, 3, 1)

        flatten_size = conv3_out_shape[0] * conv3_out_shape[1] * 64

        self.fc1 = nn.Linear(in_features=flatten_size, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=512)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                layer.bias.data.zero_()
            elif isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                layer.bias.data.zero_()

        self.seq = nn.Sequential(self.conv1, nn.LeakyReLU(),
                                 nn.Dropout2d(p=mcdropout),
                                 self.conv2, nn.LeakyReLU(),
                                 nn.Dropout2d(p=mcdropout),
                                 self.conv3, nn.LeakyReLU(),
                                 nn.Dropout2d(p=mcdropout),
                                 nn.Flatten(),
                                 self.fc1, nn.ReLU(),
                                 nn.Dropout(p=mcdropout),
                                 self.fc2, nn.ReLU(),
                                 nn.Dropout(p=mcdropout),
                                 self.fc3)
        nn.Dropout(p=mcdropout),

    def forward(self, inputs, k_samples=None):
        return self.seq(inputs)


class BayesianPredictorModel(nn.Module, ABC):

    def __init__(self, state_shape):
        super(BayesianPredictorModel, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.state_shape = state_shape
        c, w, h = state_shape
        self.conv1 = BayesianConv2d(
            in_channels=c, out_channels=32, kernel_size=(8, 8), stride=4)
        self.conv2 = BayesianConv2d(
            in_channels=32, out_channels=64, kernel_size=(4, 4), stride=2)
        self.conv3 = BayesianConv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1)

        color, w, h = state_shape
        conv1_out_shape = conv_shape((w, h), 8, 4)
        conv2_out_shape = conv_shape(conv1_out_shape, 4, 2)
        conv3_out_shape = conv_shape(conv2_out_shape, 3, 1)

        flatten_size = conv3_out_shape[0] * conv3_out_shape[1] * 64

        self.fc1 = BayesianLinear(in_features=flatten_size, out_features=512)
        self.fc2 = BayesianLinear(in_features=512, out_features=512)
        self.fc3 = BayesianLinear(in_features=512, out_features=512)

        for layer in self.modules():
            if isinstance(layer, BayesianConv2d):
                nn.init.orthogonal_(layer.weight_mu, gain=np.sqrt(2))
                layer.bias_mu.data.zero_()
            elif isinstance(layer, BayesianLinear):
                nn.init.orthogonal_(layer.weight_mu, gain=np.sqrt(2))
                layer.bias_mu.data.zero_()

        self.seq = nn.Sequential(self.conv1, nn.LeakyReLU(),
                                 self.conv2, nn.LeakyReLU(),
                                 self.conv3, nn.LeakyReLU(),
                                 nn.Flatten(),
                                 self.fc1, nn.ReLU(),
                                 self.fc2, nn.ReLU(),
                                 self.fc3)

    def forward(self, inputs, k_samples=3):
        result = torch.ones(k_samples, inputs.shape[0], 512).to(self.device)
        for i in range(k_samples):
            result[i] = self.seq(inputs)

        return torch.mean(result, 0)


class KPredictorModel(nn.Module, ABC):
    def __init__(self, state_shape, k=5):
        super(KPredictorModel, self).__init__()
        self.state_shape = state_shape
        self.k = k
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.predictors = [PredictorModel(self.state_shape) for _ in range(k)]

    def forward(self, inputs, k_samples=None):
        result = torch.ones(self.k, inputs.shape[0], 512).to(self.device)

        for i in range(self.k):
            result[i] = self.predictors[i](inputs)

        return torch.mean(result, 0)
