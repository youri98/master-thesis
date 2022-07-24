from rnd_models import *
import torch
from torch import from_numpy
import numpy as np
from numpy import concatenate  # Make coder faster.
from torch.optim.adam import Adam
from utils import mean_of_list, RunningMeanStd, clip_grad_norm_
import torch.distributed.autograd as dist_autograd
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel
from torch.distributed.optim import DistributedOptimizer
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef
import torch.distributed 
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import os
from torch.utils.data import TensorDataset, DataLoader
import sys

from torch.distributions.categorical import Categorical
from PrioritizedMemory import Memory
from heap import ReplayMemory

torch.backends.cudnn.benchmark = True
gpu = True

class RND:
    def __init__(self, **config):
        self.config = config
        self.mini_batch_size = self.config["batch_size"]
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.obs_shape = self.config["obs_shape"]
        self.state_shape = self.config["state_shape"]
        self.memory_capacity = 128*56*2


        self.current_policy = PolicyModel(self.config["state_shape"], self.config["n_actions"]).to(self.device)
        if self.config['algo'] == "RND":
            self.predictor_model = PredictorModel(self.obs_shape).to(self.device)
        elif self.config['algo'] == "RND-Bayes":
            self.predictor_model = BayesianPredictorModel(self.obs_shape).to(self.device)
        elif self.config['algo'] == "RND-MC":
            self.predictor_model = PredictorModel(self.obs_shape, mcdropout=0.1).to(self.device)
        elif self.config['algo'] == "RND-K":
            self.predictor_model = KPredictorModel(self.obs_shape).to(self.device)

        
        self.target_model = TargetModel(self.obs_shape).to(self.device)
        for param in self.target_model.parameters():
            param.requires_grad = False

        self.memory = ReplayMemory(rnd_target=self.target_model, rnd_predictor=self.predictor_model, max_capacity=self.memory_capacity)


        self.n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0 
        device_ids = [x for x in range(self.n_gpus)]

        if torch.cuda.device_count() > 1 or True:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
    
            self.predictor_model = DataParallel(self.predictor_model)
            self.current_policy = DataParallel(self.current_policy)
            self.target_model = DataParallel(self.target_model)
            self.target_model.to(self.device)
            self.predictor_model.to(self.device)
            self.current_policy.to(self.device)

        self.total_trainable_params = list(self.current_policy.parameters()) + list(self.predictor_model.parameters())
        self.optimizer = Adam(self.total_trainable_params, lr=self.config["lr"])

        self.state_rms = RunningMeanStd(shape=self.obs_shape)
        self.int_reward_rms = RunningMeanStd(shape=(1,))

        self.mse_loss = torch.nn.MSELoss()

        self.priority_alpha = 0.65
        
    def prioritized_sampling(self, values, epsilon=0.01):
        if self.config["per"] == "rankbased":

            priorities = [1/i for i in range(1, len(values) + 1)]
            priority_sum = np.sum(np.power(priorities, self.priority_alpha))
            priorities = np.power(priorities, self.priority_alpha) / priority_sum
            indices = np.flip(np.argsort(values.cpu().numpy()))

        elif self.config["per"] == "proportional":

            priorities = [np.abs(v) + epsilon for v in values.cpu().numpy()]
            priority_sum = np.sum(np.power(priorities, self.priority_alpha))
            priorities = np.power(priorities, self.priority_alpha) / priority_sum
            indices = range(0, len(values))
        else:
            raise ValueError

        fraction = 1 if self.config["n_workers"] <= 32 else 32 / self.config["n_workers"]

        sampled_indices = np.random.choice(indices, size=(self.config["n_mini_batch"], int(np.ceil(self.mini_batch_size * fraction))), p=priorities, replace=True)
        
        return sampled_indices

    def get_actions_and_values(self, state, batch=False):
        if not batch:
            state = np.expand_dims(state, 0)
        state = from_numpy(state).to(self.device)
        
        # torch.cuda.empty_cache() 

        with torch.no_grad():
            outputs = self.current_policy(state)
            int_value, ext_value, action_prob = outputs
            dist = Categorical(action_prob)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.cpu().numpy(), int_value.cpu().numpy().squeeze(), \
               ext_value.cpu().numpy().squeeze(), log_prob.cpu().numpy(), action_prob.cpu().numpy()

    def choose_mini_batch(self, states, actions, int_returns, ext_returns, advs, log_probs, next_states, uniform_sampling=False):
        states = torch.ByteTensor(states).to(self.device)
        next_states = torch.Tensor(next_states).to(self.device)
        actions = torch.ByteTensor(actions).to(self.device)
        advs = torch.Tensor(advs).to(self.device)
        int_returns = torch.Tensor(int_returns).to(self.device)
        ext_returns = torch.Tensor(ext_returns).to(self.device)
        log_probs = torch.Tensor(log_probs).to(self.device)


        if self.config['per'] == "default" or uniform_sampling:
            fraction = 1 if self.config["n_workers"] <= 32 else 32 / self.config["n_workers"]
            indices = np.random.randint(0, len(states), (self.config["n_mini_batch"], int(np.ceil(self.mini_batch_size * fraction))))
        else:
            indices = self.prioritized_sampling(advs)



        for idx in indices:
            yield states[idx], actions[idx], int_returns[idx], ext_returns[idx], advs[idx], \
                  log_probs[idx], next_states[idx]

    def sample(self):
        experiences, idxs, is_weights = self.memory.sample(self.mini_batch_size)
        return experiences, idxs, is_weights

    # def learn(self, experiences, idxs, is_weights, batch_size=BATCH_SIZE, gamma=GAMMA):
    #     states, actions, int_rewards, ext_rewards, dones, int_values, ext_values, log_probs, next_int_values, next_ext_values, total_next_obs = experiences



    def add_to_memory(self, states, actions, int_rewards,
              ext_rewards, dones, int_values, ext_values, total_next_obs):

        

        for i in np.arange(len(states)):
            self.memory.add(int_rewards[i], (states[i], actions[i], int_rewards[i], ext_rewards[i], dones[i], int_values[i], ext_values[i],
                                    total_next_obs[i]))

    # @mean_of_list
    def train(self, states, actions, int_rewards,
              ext_rewards, dones, int_values, ext_values,
              log_probs, next_int_values, next_ext_values, total_next_obs):
        

        int_rets = self.get_gae(int_rewards, int_values, next_int_values,
                                np.zeros_like(dones), self.config["int_gamma"])
        ext_rets = self.get_gae(ext_rewards, ext_values, next_ext_values,
                                dones, self.config["ext_gamma"])

        ext_values = concatenate(ext_values)
        ext_advs = ext_rets - ext_values

        int_values = concatenate(int_values)
        int_advs = int_rets - int_values

        advs = ext_advs * self.config["ext_adv_coeff"] + int_advs * self.config["int_adv_coeff"]

        self.state_rms.update(total_next_obs)
        total_next_obs = ((total_next_obs - self.state_rms.mean) / (self.state_rms.var ** 0.5)).clip(-5, 5)
        

        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0 
        device_ids = [x for x in range(n_gpus)]

        pg_losses, ext_v_losses, int_v_losses, rnd_losses, entropies = [], [], [], [], []

        for experience in zip(states, actions, np.concatenate(int_rewards), np.concatenate(ext_rewards), np.concatenate(dones), int_values, ext_values, total_next_obs):
            self.memory.add(experience)


        for epoch in range(self.config["n_epochs"]):
            for state, action, int_return, ext_return, adv, old_log_prob, next_state in self.choose_mini_batch(states=states,
                                                                                                               actions=actions,
                                                                                                               int_returns=int_rets,
                                                                                                               ext_returns=ext_rets,
                                                                                                               advs=advs,
                                                                                                               log_probs=log_probs,
                                                                                                               next_states=total_next_obs,
                                                                                                               uniform_sampling=True):


                outputs = self.current_policy(state)
                int_value, ext_value, action_prob = outputs
                dist = Categorical(action_prob)

                entropy = dist.entropy().mean()
                new_log_prob = dist.log_prob(action)
                ratio = (new_log_prob - old_log_prob).exp()
                pg_loss = self.compute_pg_loss(ratio, adv)

                int_value_loss = self.mse_loss(int_value.squeeze(-1), int_return)
                ext_value_loss = self.mse_loss(ext_value.squeeze(-1), ext_return)

                critic_loss = 0.5 * (int_value_loss + ext_value_loss) 

                policy_loss = critic_loss + pg_loss - self.config["ent_coeff"] * entropy


                pg_losses.append(pg_loss.item())
                ext_v_losses.append(ext_value_loss.item())
                int_v_losses.append(int_value_loss.item())
                entropies.append(entropy.item())
                

                rnd_loss = self.memory.compute_rnd_loss(self.mini_batch_size)
                total_loss = policy_loss + rnd_loss

                self.optimize(total_loss)

                rnd_losses.append(rnd_loss.item())


                # https://github.com/openai/random-network-distillation/blob/f75c0f1efa473d5109d487062fd8ed49ddce6634/ppo_agent.py#L187
        return np.array(pg_losses).mean(), np.array(ext_v_losses).mean(), np.array(int_v_losses).mean(), np.array(rnd_losses).mean(), np.array(entropies).mean(), np.mean(advs)

    def optimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.total_trainable_params)
        # torch.nn.utils.clip_grad_norm_(self.total_trainable_params, 0.5)
        self.optimizer.step()

    def get_gae(self, rewards, values, next_values, dones, gamma):
        lam = self.config["lambda"]  # Make code faster.
        returns = [[] for _ in range(self.config["n_workers"])]
        extended_values = np.zeros((self.config["n_workers"], self.config["rollout_length"] + 1))
        for worker in range(self.config["n_workers"]):
            extended_values[worker] = np.append(values[worker], next_values[worker])
            gae = 0
            for step in reversed(range(len(rewards[worker]))):
                delta = rewards[worker][step] + gamma * (extended_values[worker][step + 1]) * (1 - dones[worker][step]) \
                        - extended_values[worker][step]
                gae = delta + gamma * lam * (1 - dones[worker][step]) * gae
                returns[worker].insert(0, gae + extended_values[worker][step])

        return concatenate(returns)

    def calculate_int_rewards(self, next_states, batch=True):
        # torch.cuda.empty_cache() 
        if not batch:
            next_states = np.expand_dims(next_states, 0)
        next_states = np.clip((next_states - self.state_rms.mean) / (self.state_rms.var ** 0.5), -5, 5,
                              dtype="float32")  # dtype to avoid '.float()' call for pytorch.
        next_states = from_numpy(next_states).to(self.device)
        predictor_encoded_features = self.predictor_model(next_states, 5)
        target_encoded_features = self.target_model(next_states)

        int_reward = (predictor_encoded_features - target_encoded_features).pow(2).mean(1)
        if not batch:
            return int_reward.detach().cpu().numpy()
        else:
            return int_reward.detach().cpu().numpy().reshape((self.config["n_workers"], self.config["rollout_length"]))

    def normalize_int_rewards(self, intrinsic_rewards):
        # OpenAI's usage of Forward filter is definitely wrong;
        # Because: https://github.com/openai/random-network-distillation/issues/16#issuecomment-488387659
        gamma = self.config["int_gamma"]  # Make code faster.
        intrinsic_returns = [[] for _ in range(self.config["n_workers"])]
        for worker in range(self.config["n_workers"]):
            rewems = 0
            for step in reversed(range(self.config["rollout_length"])):
                rewems = rewems * gamma + intrinsic_rewards[worker][step]
                intrinsic_returns[worker].insert(0, rewems)
        self.int_reward_rms.update(np.ravel(intrinsic_returns).reshape(-1, 1))

        return intrinsic_rewards / (self.int_reward_rms.var ** 0.5)

    def compute_pg_loss(self, ratio, adv):
        new_r = ratio * adv
        clamped_r = torch.clamp(ratio, 1 - self.config["clip_range"], 1 + self.config["clip_range"]) * adv
        loss = torch.min(new_r, clamped_r)
        loss = -loss.mean()
        return loss

    def calculate_rnd_loss(self, next_state):
        encoded_target_features = self.target_model(next_state)
        encoded_predictor_features = self.predictor_model(next_state, 1)
        loss = (encoded_predictor_features - encoded_target_features).pow(2).mean(-1)
        # mask = torch.rand(loss.size(), device=self.device)
        # mask = (mask < self.config["predictor_proportion"]).float()
        # loss = (mask * loss).sum() / torch.max(mask.sum(), torch.Tensor([1]).to(self.device))


        # loss = torch.mean(loss)
        return loss

    def set_from_checkpoint(self, checkpoint):
        self.current_policy.load_state_dict(checkpoint["current_policy_state_dict"])
        self.predictor_model.load_state_dict(checkpoint["predictor_model_state_dict"])
        self.target_model.load_state_dict(checkpoint["target_model_state_dict"])
        for param in self.target_model.parameters():
            param.requires_grad = False
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.state_rms.mean = checkpoint["state_rms_mean"]
        self.state_rms.var = checkpoint["state_rms_var"]
        self.state_rms.count = checkpoint["state_rms_count"]
        self.int_reward_rms.mean = checkpoint["int_reward_rms_mean"]
        self.int_reward_rms.var = checkpoint["int_reward_rms_var"]
        self.int_reward_rms.count = checkpoint["int_reward_rms_count"]

    def set_to_eval_mode(self):
        self.current_policy.eval()
