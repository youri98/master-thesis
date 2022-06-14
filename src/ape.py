from scipy.misc import derivative
import wandb
from ape_models import PolicyModel, TargetModel, PredictorModelRND
import torch
import numpy as np
from torch.optim.adam import Adam
from utils import mean_of_list, mean_of_list, RunningMeanStd
from runner import Worker
from torch.multiprocessing import Process, Pipe
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
import os
from torch.utils.data import TensorDataset, DataLoader
import sys
import pygad
from torch.distributions.categorical import Categorical

torch.backends.cudnn.benchmark = True

class APE:
    def __init__(self, timesteps=1, rnd_predictor=False, encoding_size=512, multiple_feature_pred=False, use_gan_loss=False, **config):

        self.config = config
        self.device =  torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.obs_shape = self.config["obs_shape"]
        self.state_shape = self.config["state_shape"]

        self.rnd_predictor = rnd_predictor
        self.use_gan_loss = use_gan_loss
        self.encoding_size = encoding_size
        self.multiple_feature_pred = multiple_feature_pred
        self.timesteps = timesteps
        self.pred_size = self.encoding_size if self.multiple_feature_pred else 1
        self.n_actions = self.config["n_actions"]
        self.prev_disc_losses = None

        self.current_policy = PolicyModel(self.config["state_shape"], self.config["n_actions"]).to(self.device)
        self.predictor_model = PredictorModelRND(self.obs_shape).to(self.device)
        self.target_model = TargetModel(self.obs_shape, self.encoding_size).to(self.device)

        # self.current_policy.requires_grad_(False)
    


        for param in self.target_model.parameters():
            param.requires_grad = False



        self.n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0 

        if torch.cuda.device_count() > 1 or True:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
    
            self.predictor_model = DataParallel(self.predictor_model)
            self.current_policy = DataParallel(self.current_policy)
            self.target_model = DataParallel(self.target_model)
            self.target_model.to(self.device)
            self.predictor_model.to(self.device)
            self.current_policy.to(self.device)

        # self.pol_optimizer = Adam(self.current_policy.parameters(), lr=self.config["lr"])
        self.pred_optimizer = Adam(self.predictor_model.parameters(), lr=self.config["lr"])

        # consider LBFGS

        self.state_rms = RunningMeanStd(shape=self.obs_shape)
        self.int_reward_rms = RunningMeanStd(shape=(1,))

        for param in self.target_model.parameters():
            param.requires_grad = False

    def get_discount_rewards():
        # dont know if this is a good idea, because future states we want to count as important?
        pass

    def get_gae(self, rewards, values, next_values, dones, gamma):
        lam = self.config["lambda"]  # Make code faster.

        returns = []
        extended_values = [np.array([]) for _ in range(self.config["n_workers"])]  #np.zeros((self.config["n_workers"], self.config["rollout_length"] + 1))

        extended_values = np.append(values, next_values)
        gae = 0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * (extended_values[step + 1]) * (1 - dones[step]) \
                    - extended_values[step]
            gae = delta + gamma * lam * (1 - dones[step]) * gae
            returns.insert(0, gae + extended_values[step])

        return np.array(returns)

    def get_adv(self, int_rewards, ext_rewards, int_values, ext_values, next_int_values, next_ext_values, dones):
        int_rets = self.get_gae(int_rewards, int_values, next_int_values, np.zeros_like(dones), self.config["int_gamma"])
        ext_rets = self.get_gae(ext_rewards, ext_values, next_ext_values, dones, self.config["ext_gamma"])

        ext_values = np.array(ext_values)
        ext_advs = ext_rets - ext_values

        int_values = np.array(int_values)
        int_advs = int_rets - int_values

        advs = ext_advs * self.config["ext_adv_coeff"] + int_advs * self.config["int_adv_coeff"]
        return advs

    def optimize(self, loss, optimizer, model):
        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), np.inf, 0.5)
        optimizer.step()

    def get_actions_and_values(self, state, batch=False):
        """
        returns intrinsic reward valueimate, extrinsic, action, log_prob of actions, prob of actions
        """
        if not batch:
            state = np.expand_dims(state, 0)
        state = torch.from_numpy(state).to(self.device)
        
        torch.cuda.empty_cache() 

        with torch.no_grad():
            outputs = self.current_policy(state)
            int_value, ext_value, action_prob = outputs
            dist = Categorical(action_prob)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.cpu().numpy(), int_value.cpu().numpy().squeeze(), \
               ext_value.cpu().numpy().squeeze(), log_prob.cpu().numpy(), action_prob.cpu().numpy()


    def generate_batches(self, states, actions, int_returns, ext_returns, advs, log_probs, next_states):
        states = torch.ByteTensor(states).to(self.device)
        next_states = torch.Tensor(next_states).to(self.device)
        actions = torch.ByteTensor(actions).to(self.device)
        advs = torch.Tensor(advs).to(self.device)
        int_returns = torch.Tensor(int_returns).to(self.device)
        ext_returns = torch.Tensor(ext_returns).to(self.device)
        log_probs = torch.Tensor(log_probs).to(self.device)
        
        # if self.config["algo"] == "APE":
        #     shape = (self.config['n_mini_batch'], -1, self.timesteps) 
        #     #  for example with len 1024 -> 4, 32, 8: 4 batches, 32 size of batch, 8 timesteps per individual prediction
        # else:
        #     shape = (self.config['n_mini_batch'], -1)
        #     #  for example with len 1024 -> 4, 32*8 = 256: 4 batches, 256 size of batch

        # states = states.view(*shape, *self.state_shape)
        # next_states = next_states.view(*shape, *self.state_shape)
        # actions = actions.view(*shape, 1)
        # log_probs = log_probs.view(self.config['n_mini_batch'], -1)
        # advs = advs.view(self.config['n_mini_batch'], -1)

        # int_returns = int_returns.view(self.config['n_mini_batch'], -1)
        # ext_returns = ext_returns.view(self.config['n_mini_batch'], -1)
        indices = np.random.randint(0, len(states)//self.timesteps, (self.config["n_mini_batch"], self.mini_batch_size//self.timesteps))
        if self.config["n_workers"] > 32:
            fraction = 32 / self.config["n_workers"] 
            mask = np.random.rand(indices.shape[1]) <= fraction
            indices = indices[:, mask]
        indices = np.concatenate([list(range(self.timesteps*idx, self.timesteps*idx + self.timesteps)) for batch in indices for idx in batch])
        indices = np.reshape(indices, (self.config["n_mini_batch"], -1))

        for idx in indices:
            yield states[idx], actions[idx], int_returns[idx], ext_returns[idx], advs[idx], log_probs[idx], next_states[idx]
    
    def choose_mini_batch(self, states, actions, int_returns, ext_returns, advs, log_probs, next_states):
        states = torch.ByteTensor(states).to(self.device)
        next_states = torch.Tensor(next_states).to(self.device)
        actions = torch.ByteTensor(actions).to(self.device)
        advs = torch.Tensor(advs).to(self.device)
        int_returns = torch.Tensor(int_returns).to(self.device)
        ext_returns = torch.Tensor(ext_returns).to(self.device)
        log_probs = torch.Tensor(log_probs).to(self.device)

        indices = np.random.randint(0, len(states), (self.config["n_mini_batch"], self.mini_batch_size))

        for idx in indices:
            yield states[idx], actions[idx], int_returns[idx], ext_returns[idx], advs[idx], \
                  log_probs[idx], next_states[idx]

    def choose_mini_batch_rnd(self, next_states):
        next_states = torch.Tensor(next_states).to(self.device)
        batch_size = int(np.ceil(len(next_states)/self.config["n_mini_batch"]))
        indices = np.random.randint(0, len(next_states), (self.config["n_mini_batch"], batch_size))

        for idx in indices:
            yield next_states[idx]

    def train_rnd(self, total_next_obs):
        self.state_rms.update(total_next_obs)
        total_next_obs = ((total_next_obs - self.state_rms.mean) / (self.state_rms.var ** 0.5)).clip(-5, 5)
        total_next_obs = torch.Tensor(total_next_obs).to(self.device)

        rnd_losses = []
        for epoch in range(self.config["n_epochs"]):
            for next_state in self.choose_mini_batch_rnd(total_next_obs):
 
                rnd_loss = self.calculate_rnd_loss(next_state)

                self.optimize(rnd_loss, self.pred_optimizer, self.predictor_model)

                rnd_losses.append(rnd_loss.item())
            # https://github.com/openai/random-network-distillation/blob/f75c0f1efa473d5109d487062fd8ed49ddce6634/ppo_agent.py#L187

        return np.mean(rnd_losses)
    

    def calculate_int_rewards(self, next_states, batch=True):
        if not batch:
            next_states = np.expand_dims(next_states, 0)
        next_states = np.clip((next_states - self.state_rms.mean) / (self.state_rms.var ** 0.5), -5, 5,
                              dtype="float32")  # dtype to avoid '.float()' call for pytorch.
        # torch.cuda.empty_cache() 
        next_states = torch.from_numpy(next_states).type(torch.float32).to(self.device)

        target_encoded_features = self.target_model(next_states)
        predictor_encoded_features = self.predictor_model(next_states).detach()

        loss = torch.pow(target_encoded_features - predictor_encoded_features, 2)
        loss = torch.mean(loss, 1)
            
        if not batch:
            return loss
        else:
            return loss.detach().cpu().numpy()
                

    def calculate_rnd_loss(self, next_states): 
        target_encoded_features = self.target_model(next_states) # wants flat
        predictor_encoded_features = self.predictor_model(next_states)
        loss = torch.pow(predictor_encoded_features - target_encoded_features, 2)

        mask = torch.rand(loss.size(), device=self.device)
        mask = mask < self.config["predictor_proportion"]
        # mask = np.random.choice([1,0], size=len(target_encoded_features), p=[self.config["predictor_proportion"], 1 - self.config["predictor_proportion"]])

        loss = torch.mean(loss[mask])

        return loss

    def set_from_checkpoint(self, checkpoint): 
        self.current_policy.load_state_dict(checkpoint["current_policy_state_dict"])
        self.predictor_model.load_state_dict(checkpoint["predictor_model_state_dict"])
        self.target_model.load_state_dict(checkpoint["target_model_state_dict"])
        self.discriminator.load_state_dict(checkpoint["discriminator_model_state_dict"])
        for param in self.target_model.parameters():
            param.requires_grad = False
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.state_rms.mean = checkpoint["state_rms_mean"]
        self.state_rms.var = checkpoint["state_rms_var"]
        self.state_rms.count = checkpoint["state_rms_count"]
        self.int_reward_rms.mean = checkpoint["int_reward_rms_mean"]
        self.int_reward_rms.var = checkpoint["int_reward_rms_var"]
        self.int_reward_rms.count = checkpoint["int_reward_rms_count"]
            
    def compute_pg_loss(self, ratio, adv):
        new_r = ratio * adv
        clamped_r = torch.clamp(ratio, 1 - self.config["clip_range"], 1 + self.config["clip_range"]) * adv
        loss = torch.min(new_r, clamped_r)
        loss = -loss.mean()
        return loss


    def normalize_int_rewards(self, pop_intrinsic_rewards):
        # OpenAI's usage of Forward filter is definitely wrong;
        # Because: https://github.com/openai/random-network-distillation/issues/16#issuecomment-488387659
        # if isinstance(intrinsic_rewards, torch.Tensor):
        #     intrinsic_rewards = intrinsic_rewards.to(torch.float32).cpu().numpy()

        gamma = self.config["int_gamma"]  # Make code faster.
        intrinsic_returns = [[] for _ in range(self.config["n_individuals_per_gen"])]
        for indiv_idx, individual in enumerate(pop_intrinsic_rewards):
            rewems = 0
            for step in reversed(range(len(individual))):
                rewems = rewems * gamma + individual[step]
                intrinsic_returns[indiv_idx].insert(0, rewems)
        temp = np.array(intrinsic_returns, dtype=object)
        temp = np.concatenate(temp)
        temp = np.ravel(temp)
        temp = temp.reshape(-1, 1)
        self.int_reward_rms.update(temp)

        return np.array(pop_intrinsic_rewards, dtype=object) / (self.int_reward_rms.var ** 0.5)

    def set_to_eval_mode(self):
        self.current_policy.eval()