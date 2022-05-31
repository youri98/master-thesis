from ape_models import PolicyModel, PredictorModel, TargetModel, DiscriminatorModel, DiscriminatorModelGRU
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
from sklearn.metrics import log_loss

from torch.distributions.categorical import Categorical

torch.backends.cudnn.benchmark = True

class APE:
    def __init__(self, timesteps=8, use_decoder=False, encoding_size=512, multiple_feature_pred=False, **config):

        self.config = config
        self.mini_batch_size = self.config["batch_size"]
        self.device =  torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.obs_shape = self.config["obs_shape"]
        
        self.use_decoder = use_decoder
        self.encoding_size = encoding_size
        self.multiple_feature_pred = multiple_feature_pred
        self.timesteps = timesteps
        self.pred_size = self.encoding_size if self.multiple_feature_pred else 1
        self.n_actions = self.config["n_actions"]

        self.discriminator = DiscriminatorModel(self.encoding_size, timesteps=self.timesteps, pred_size=self.pred_size, n_actions=self.config["n_actions"]).to(self.device)
        self.current_policy = PolicyModel(self.config["state_shape"], self.config["n_actions"]).to(self.device)
        self.predictor_model = PredictorModel(self.encoding_size, timesteps=self.timesteps, pred_size=self.pred_size, n_actions=self.config["n_actions"]).to(self.device)
        self.target_model = TargetModel(self.obs_shape, self.encoding_size).to(self.device)


        for param in self.target_model.parameters():
            param.requires_grad = False



        self.n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0 

        if torch.cuda.device_count() > 1 or True:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
    
            self.predictor_model = DataParallel(self.predictor_model)
            self.current_policy = DataParallel(self.current_policy)
            self.discriminator = DataParallel(self.discriminator)
            self.target_model = DataParallel(self.target_model)
            self.target_model.to(self.device)
            self.predictor_model.to(self.device)
            self.current_policy.to(self.device)
            self.discriminator.to(self.device)

        self.pol_optimizer = Adam(self.current_policy.parameters(), lr=self.config["lr"])
        self.pred_optimizer = Adam(self.predictor_model.parameters(), lr=self.config["lr"])
        self.disc_optimizer = Adam(self.discriminator.parameters(), lr=self.config["lr"])

        # consider LBFGS

        self.state_rms = RunningMeanStd(shape=self.obs_shape)
        self.int_reward_rms = RunningMeanStd(shape=(1,))
        self.loss_func = lambda y_hat,y: torch.nn.BCELoss(reduction='none')(y_hat, y.float())# if self.pred_size == 1 else torch.nn.L1Loss() 
        self.mse_loss = torch.nn.MSELoss()

        for param in self.target_model.parameters():
            param.requires_grad = False

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

        return np.concatenate(returns)

    def optimize(self, loss, optimizer, model):
        model.zero_grad()
        loss.backward(retain_graph=True)
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


    def generate_batches(self, states, next_states, actions, log_probs, advs, int_rets, ext_rets):
        if self.config["algo"] == "APE":
            shape = (self.config['n_mini_batch'], -1, self.timesteps) 
            #  for example with len 1024 -> 4, 32, 8: 4 batches, 32 size of batch, 8 timesteps per individual prediction
        else:
            shape = (self.config['n_mini_batch'], -1)
            #  for example with len 1024 -> 4, 32*8 = 256: 4 batches, 256 size of batch

        states = states.view(*shape, *self.obs_shape)
        next_states = next_states.view(*shape, *self.obs_shape)
        actions = actions.view(*shape, 1)
        log_probs = log_probs.view(self.config['n_mini_batch'], -1)
        advs = advs.view(self.config['n_mini_batch'], -1)
        int_rets = int_rets.view(self.config['n_mini_batch'], -1)
        ext_rets = ext_rets.view(self.config['n_mini_batch'], -1)

        for t in range(self.config['n_mini_batch']):
            yield states[t], next_states[t], actions[t], log_probs[t], advs[t], int_rets[t], ext_rets[t]
    
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

    def train(self, states, actions, int_rewards,
              ext_rewards, dones, int_values, ext_values,
              log_probs, next_int_values, next_ext_values, total_next_obs):

        int_rets = self.get_gae(int_rewards, int_values, next_int_values,
                                np.zeros_like(dones), self.config["int_gamma"])
        ext_rets = self.get_gae(ext_rewards, ext_values, next_ext_values,
                                dones, self.config["ext_gamma"])

        ext_values = np.concatenate(ext_values)
        ext_advs = ext_rets - ext_values

        int_values = np.concatenate(int_values)
        int_advs = int_rets - int_values

        advs = ext_advs * self.config["ext_adv_coeff"] + int_advs * self.config["int_adv_coeff"]

        self.state_rms.update(total_next_obs)
        total_next_obs = ((total_next_obs - self.state_rms.mean) / (self.state_rms.var ** 0.5)).clip(-5, 5)
        

        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0 
        device_ids = [x for x in range(n_gpus)]

        pg_losses, ext_v_losses, int_v_losses, rnd_losses, entropies, disc_losses = [], [], [], [], [], []
        for epoch in range(self.config["n_epochs"]):
            torch.cuda.empty_cache() 

            for state, action, int_return, ext_return, adv, old_log_prob, next_state in \
                    self.choose_mini_batch(states=states,
                                           actions=actions,
                                           int_returns=int_rets,
                                           ext_returns=ext_rets,
                                           advs=advs,
                                           log_probs=log_probs,
                                           next_states=total_next_obs):
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
                
                action = action.to(torch.int64).to(self.device)
                action = torch.nn.functional.one_hot(action, num_classes=self.n_actions)
                action = action.view(-1, self.timesteps, self.n_actions)
                disc_loss, gen_loss = self.calculate_loss(next_state, action)


                total_loss = critic_loss + pg_loss - self.config["ent_coeff"] * entropy

                self.optimize(total_loss, self.pol_optimizer, self.current_policy)
                self.optimize(gen_loss, self.pred_optimizer, self.predictor_model)
                self.optimize(disc_loss, self.disc_optimizer, self.discriminator)

                pg_losses.append(pg_loss.item())
                ext_v_losses.append(ext_value_loss.item())
                int_v_losses.append(int_value_loss.item())
                rnd_losses.append(gen_loss.item())
                disc_losses.append(disc_loss.item())
                entropies.append(entropy.item())
            # https://github.com/openai/random-network-distillation/blob/f75c0f1efa473d5109d487062fd8ed49ddce6634/ppo_agent.py#L187

        return np.mean(pg_losses), np.mean(ext_v_losses), np.mean(int_v_losses), np.mean(rnd_losses), np.mean(disc_losses), np.mean(entropies), np.mean(advs) #, int_values, int_rets, ext_values, ext_rets
    

    def calculate_int_rewards(self, next_states, actions, batch=True):
        if not batch:
            next_states = np.expand_dims(next_states, 0)
        next_states = np.clip((next_states - self.state_rms.mean) / (self.state_rms.var ** 0.5), -5, 5,
                              dtype="float32")  # dtype to avoid '.float()' call for pytorch.
        torch.cuda.empty_cache() 


        next_states = torch.from_numpy(next_states).type(torch.float32).to(self.device)

        actions = torch.from_numpy(actions).to(torch.int64).to(self.device)
        actions = torch.nn.functional.one_hot(actions, num_classes=self.n_actions)
        actions = actions.view(-1, self.timesteps, self.n_actions)

        target_encoded_features = self.target_model(next_states).detach()
        # predictor_encoded_features = self.predictor_model(target_encoded_features, actions).detach()

        # predictor_encoded_features = predictor_encoded_features.view(-1, self.timesteps, self.encoding_size)
        target_encoded_features = target_encoded_features.view(-1, self.timesteps, self.encoding_size)

        predictor_encoded_features = self.predictor_model(target_encoded_features, actions).detach()



        mask = torch.randint(0, 2, size=(*predictor_encoded_features.shape[:-1], self.pred_size)).to(self.device)
        #mask = torch.kron(mask, torch.ones(self.timesteps, dtype=torch.uint8))
        mask_inv = torch.where((mask==0)|(mask==1), mask^1, mask).to(self.device)

        temp_p = predictor_encoded_features*mask_inv
        temp_t = target_encoded_features*mask
        features = (temp_p + temp_t).to(self.device)

        disc_preds = self.discriminator(features, actions, target_encoded_features)
        disc_loss = self.loss_func(disc_preds[:, 0], mask[:, 0]) if self.multiple_feature_pred else self.loss_func(disc_preds, mask)

        if not batch:
            return 1/disc_loss.detach().cpu().numpy()
        else:
            return 1/disc_loss.detach().cpu().numpy().reshape((self.config["n_workers"], self.config["rollout_length"]))

    def calculate_loss(self, next_state, action): 
        
        target_encoded_features = self.target_model(next_state.view(-1, *self.obs_shape))
        
        target_encoded_features = target_encoded_features.view(-1, self.timesteps, self.encoding_size)
        # predictor_encoded_features = self.predictor_model(target_encoded_features, action).detach()

        # predictor_encoded_features = predictor_encoded_features.view(-1, self.timesteps, self.encoding_size)
        predictor_encoded_features = self.predictor_model(target_encoded_features, action)

        mask = torch.randint(0, 2, size=(*predictor_encoded_features.shape[:-1], self.pred_size)).to(self.device)
        mask_inv = torch.where((mask==0)|(mask==1), mask^1, mask).to(self.device)
            
        temp_p = predictor_encoded_features*mask_inv
        temp_t = target_encoded_features*mask
        features = temp_p + temp_t

        disc_preds = self.discriminator(features, action, target_encoded_features)
        gen_loss = torch.mean(self.loss_func(disc_preds, mask_inv))

        disc_preds =  self.discriminator(features.detach(), action, target_encoded_features)
        disc_loss = torch.mean(self.loss_func(disc_preds, mask))

        return disc_loss, gen_loss

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


    def normalize_int_rewards(self, intrinsic_rewards):
        # OpenAI's usage of Forward filter is definitely wrong;
        # Because: https://github.com/openai/random-network-distillation/issues/16#issuecomment-488387659
        if isinstance(intrinsic_rewards, torch.Tensor):
            intrinsic_rewards = intrinsic_rewards.to(torch.float32).cpu().numpy()

        gamma = self.config["int_gamma"]  # Make code faster.
        intrinsic_returns = [[] for _ in range(self.config["n_workers"])]
        for worker in range(self.config["n_workers"]):
            rewems = 0
            for step in reversed(range(self.config["rollout_length"])):
                rewems = rewems * gamma + intrinsic_rewards[worker][step]
                intrinsic_returns[worker].insert(0, rewems)
        self.int_reward_rms.update(np.ravel(intrinsic_returns).reshape(-1, 1))

        return intrinsic_rewards / (self.int_reward_rms.var ** 0.5)

    def set_to_eval_mode(self):
        self.current_policy.eval()