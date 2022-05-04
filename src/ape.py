from models import PolicyModel, PredictorModel, TargetModel, DiscriminatorModel
from rnd import RND
import torch
from torch import from_numpy
import numpy as np
from numpy import concatenate  # Make coder faster.
from torch.optim.adam import Adam
from utils import mean_of_list, RunningMeanStd

torch.backends.cudnn.benchmark = True

class APE(RND):
    def __init__(self, **config):
        super(APE, self).__init__(**config)

        self.discriminator = DiscriminatorModel()

        for param in self.target_model.parameters():
            param.requires_grad = False

        self.total_trainable_params = list(self.current_policy.parameters()) + list(self.predictor_model.parameters()) + list(self.discriminator.parameters())
        self.optimizer = Adam(self.total_trainable_params, lr=self.config["lr"])

        self.bce_loss = torch.nn.L1Loss(reduction='none')
        self.rollout_len = 4

    @mean_of_list
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
        # normalise
        total_next_obs = ((total_next_obs - self.state_rms.mean) / (self.state_rms.var ** 0.5)).clip(-5, 5)

        pg_losses, ext_value_losses, int_value_losses, rnd_losses, disc_losses, entropies = [], [], [], [], [], []
        for epoch in range(self.config["n_epochs"]):
            for state, action, int_return, ext_return, adv, old_log_prob, next_state in \
                    self.choose_mini_batch(states=states,
                                           actions=actions,
                                           int_returns=int_rets,
                                           ext_returns=ext_rets,
                                           advs=advs,
                                           log_probs=log_probs,
                                           next_states=total_next_obs):
                dist, int_value, ext_value, _ = self.current_policy(state)
                entropy = dist.entropy().mean()
                new_log_prob = dist.log_prob(action)
                ratio = (new_log_prob - old_log_prob).exp()
                pg_loss = self.compute_pg_loss(ratio, adv)

                int_value_loss = self.mse_loss(int_value.squeeze(-1), int_return)
                ext_value_loss = self.mse_loss(ext_value.squeeze(-1), ext_return)

                critic_loss = 0.5 * (int_value_loss + ext_value_loss)

                rnd_loss = self.calculate_rnd_loss(next_state)
                disc_loss = self.calculate_discriminator_loss(next_state)

                total_loss = critic_loss + pg_loss - self.config["ent_coeff"] * entropy + rnd_loss + disc_loss
                self.optimize(total_loss)

                pg_losses.append(pg_loss.item())
                ext_value_losses.append(ext_value_loss.item())
                int_value_losses.append(int_value_loss.item())
                rnd_losses.append(rnd_loss.item())
                disc_losses.append(disc_loss.item())
                entropies.append(entropy.item())
                # https://github.com/openai/random-network-distillation/blob/f75c0f1efa473d5109d487062fd8ed49ddce6634/ppo_agent.py#L187

        return pg_losses, ext_value_losses, int_value_losses, rnd_losses, disc_losses, entropies#, int_values, int_rets, ext_values, ext_rets
    
    def calculate_int_rewards(self, next_states, batch=True):
        if not batch:
            next_states = np.expand_dims(next_states, 0)
        next_states = np.clip((next_states - self.state_rms.mean) / (self.state_rms.var ** 0.5), -5, 5,
                              dtype="float32")  # dtype to avoid '.float()' call for pytorch.
        next_states = from_numpy(next_states).to(self.device)
        predictor_encoded_features = self.predictor_model(next_states)
        target_encoded_features = self.target_model(next_states)

        if batch:

            mask = np.random.choice([True, False], size=len(predictor_encoded_features)/self.rollout_len, p=[.5,.5])
            features = []

            for t, booly in enumerate(mask):
                if booly:
                    features.append(predictor_encoded_features[t : t+self.rollout_len, ...])
                else:
                    features.append(target_encoded_features[t : t+self.rollout_len, ...])



            temp = [(t,m) if m else (p,m) for p,t,m in zip(predictor_encoded_features, target_encoded_features, mask)]
            features, y_true = zip(*temp)
            y_true = torch.unsqueeze(torch.tensor(y_true, dtype=torch.float32), dim=1)
            features = torch.stack(features)
            features = torch.unsqueeze(features, dim=1)
            target_encoded_features = torch.unsqueeze(target_encoded_features, dim=1)
            disc_preds = self.discriminator(features, target_encoded_features)
            disc_loss = self.bce_loss(disc_preds, y_true)

        if not batch:
            return disc_loss.detach().cpu().numpy()
        else:
            return disc_loss.detach().cpu().numpy().reshape((self.config["n_workers"], self.config["rollout_length"]))

    def calculate_discriminator_loss(self, next_state):
        target_encoded_features = self.target_model(next_state)
        predictor_encoded_features = self.predictor_model(next_state)

        mask = np.random.choice([True, False], size=len(predictor_encoded_features), p=[.5,.5])
        temp = [(t,m) if m else (p,m) for p,t,m in zip(predictor_encoded_features, target_encoded_features, mask)]
        features, y_true = zip(*temp)
        y_true = torch.unsqueeze(torch.tensor(y_true, dtype=torch.float32), dim=1)
        features = torch.stack(features)
        features = torch.unsqueeze(features, dim=1)
        target_encoded_features = torch.unsqueeze(target_encoded_features, dim=1)

        disc_preds = self.discriminator(features, target_encoded_features)
        loss = self.bce_loss(disc_preds, y_true)
        
        mask = torch.rand(loss.size(), device=self.device)
        mask = (mask < self.config["predictor_proportion"]).float()
        loss = (mask * loss).sum() / torch.max(mask.sum(), torch.Tensor([1]).to(self.device))
        return loss


    def set_from_checkpoint(self, checkpoint): # TODO: add discr
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
