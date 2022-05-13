from models import PolicyModel, PredictorModel, TargetModel, DiscriminatorModel, DecoderModel
import torch
import numpy as np
from torch.optim.adam import Adam
from utils import mean_of_list, mean_of_list, RunningMeanStd
from runner import Worker
from torch.multiprocessing import Process, Pipe
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel

torch.backends.cudnn.benchmark = True

class APE:
    def __init__(self, timesteps=8, use_decoder=False, encoding_size=512, multiple_feature_pred=False, run_from_hpc=False, **config):

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
        self.predictor_model = PredictorModel(self.obs_shape, self.encoding_size).to(self.device)
        self.target_model = TargetModel(self.obs_shape, self.encoding_size).to(self.device)


        if run_from_hpc:
            self.current_policy = DistributedDataParallel(self.current_policy)
            self.predictor_model = DistributedDataParallel(self.predictor_model)
            self.target_model = DistributedDataParallel(self.target_model)
            self.discriminator = DistributedDataParallel(self.discriminator)

        for param in self.target_model.parameters():
            param.requires_grad = False

        self.total_trainable_params = list(self.current_policy.parameters()) + list(self.predictor_model.parameters())

        if self.config["algo"] == "APE":
            self.total_trainable_params += list(self.discriminator.parameters())
        
        if self.use_decoder:
            self.decoder = DecoderModel()
            self.total_trainable_params += list(self.decoder.parameters())

        self.optimizer = Adam(self.total_trainable_params, lr=self.config["lr"])
        # consider LBFGS

        self.state_rms = RunningMeanStd(shape=self.obs_shape)
        self.int_reward_rms = RunningMeanStd(shape=(1,))
        self.f1_loss = torch.nn.L1Loss(reduction='none')# if self.pred_size == 1 else torch.nn.L1Loss() 
        self.mse_loss = torch.nn.MSELoss()

        for param in self.target_model.parameters():
            param.requires_grad = False
        
        # self.workers = [Worker(i, **config) for i in range(config["n_workers"])] 
        # self.parents = []
        # for worker in self.workers:
        #     parent_conn, child_conn = Pipe()
        #     p = Process(target=self.run_workers_train, args=(worker, child_conn,))
        #     p.daemon = True
        #     self.parents.append(parent_conn)
        #     p.start()





        # for worker_id, parent in enumerate(parents):
        #     total_states[worker_id, t] = parent.recv()

        # for parent, a in zip(parents, total_actions[:, t]):
        #     parent.send(a)


    def run_workers_train(worker, conn):
        worker.step(conn)

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

    def optimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.total_trainable_params, np.inf, 0.5)
        self.optimizer.step()

    def get_actions_and_values(self, state, batch=False):
        """
        returns intrinsic reward valueimate, extrinsic, action, log_prob of actions, prob of actions
        """
        if not batch:
            state = np.expand_dims(state, 0)
        state = torch.from_numpy(state).to(self.device)
        with torch.no_grad():
            dist, int_value, ext_value, action_prob = self.current_policy(state)
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

    @mean_of_list
    def train(self, states, actions, int_rewards,
              ext_rewards, dones, int_values, ext_values,
              log_probs, next_int_values, next_ext_values, next_states):

        int_rets = self.get_gae(int_rewards, int_values, next_int_values,
                                np.zeros_like(dones), self.config["int_gamma"])
        ext_rets = self.get_gae(ext_rewards, ext_values, next_ext_values,
                                dones, self.config["ext_gamma"])

        ext_values = np.concatenate(ext_values)
        ext_advs = ext_rets - ext_values

        int_values = np.concatenate(int_values)
        int_advs = int_rets - int_values

        advs = ext_advs * self.config["ext_adv_coeff"] + int_advs * self.config["int_adv_coeff"]

        self.state_rms.update(next_states)

        changes_in_states = ((next_states - self.state_rms.mean) / (self.state_rms.var ** 0.5)).clip(-5, 5)

        if self.device == torch.device("cuda"):
            # optimised for 16-bit computing on gpus such as v100 and 20180ti, but not supported for cpu
            states = torch.Tensor(states).to(torch.float16).to(self.device)
            next_states = torch.Tensor(next_states).to(torch.float16).to(self.device)
            actions = torch.Tensor(actions).to(torch.int64).to(self.device)
            log_probs = torch.Tensor(log_probs).to(torch.float16).to(self.device)
            advs = torch.Tensor(advs).to(torch.float16).to(self.device)
            int_rets = torch.Tensor(int_rets).to(torch.float16).to(self.device)
            ext_rets = torch.Tensor(ext_rets).to(torch.float16).to(self.device)
        else:
            states = torch.Tensor(states).to(self.device)
            next_states = torch.Tensor(next_states).to(self.device)
            actions = torch.Tensor(actions).to(torch.int64).to(self.device)
            log_probs = torch.Tensor(log_probs).to(self.device)
            advs = torch.Tensor(advs).to(self.device)
            int_rets = torch.Tensor(int_rets).to(self.device)
            ext_rets = torch.Tensor(ext_rets).to(self.device)



        pg_losses, ext_value_losses, int_value_losses, rnd_losses, disc_losses, entropies = [], [], [], [], [], []
        for epoch in range(self.config["n_epochs"]):
            for state, next_state, action, log_prob, adv, int_ret, ext_ret in self.generate_batches(states, next_states, actions, log_probs, advs, int_rets, ext_rets):
                dist, int_value, ext_value, _ = self.current_policy(next_state.view(-1, *self.obs_shape))
                entropy = dist.entropy().mean()
                new_log_prob = dist.log_prob(action.view(-1))
                ratio = (new_log_prob - log_prob).exp()
                pg_loss = self.compute_pg_loss(ratio, adv)

                int_value_loss = self.mse_loss(int_value.squeeze(-1), int_ret)
                ext_value_loss = self.mse_loss(ext_value.squeeze(-1), ext_ret)

                critic_loss = 0.5 * (int_value_loss + ext_value_loss)

                # TODO: this is basically batch size 1, with a rollout length of 16 each time
                # so split on rollout size and multiprocess that and sum the individual losses as batch loss
                if self.config["algo"] == "APE":
                    action = torch.nn.functional.one_hot(action, num_classes=self.n_actions)
                    action = action.view(-1, self.timesteps, self.n_actions)
                disc_loss, rnd_loss = self.calculate_loss(next_state, action)


                total_loss = critic_loss + pg_loss - self.config["ent_coeff"] * entropy + rnd_loss + disc_loss

                self.optimize(total_loss)

                pg_losses.append(pg_loss.item())
                ext_value_losses.append(ext_value_loss.item())
                int_value_losses.append(int_value_loss.item())
                rnd_losses.append(rnd_loss.item())
                if self.config["algo"] == "APE":
                    disc_losses.append(disc_loss.item())
                entropies.append(entropy.item())
            # https://github.com/openai/random-network-distillation/blob/f75c0f1efa473d5109d487062fd8ed49ddce6634/ppo_agent.py#L187

        return pg_losses, ext_value_losses, int_value_losses, rnd_losses, disc_losses, entropies, advs #, int_values, int_rets, ext_values, ext_rets
    
    def calculate_int_rewards(self, next_states, actions, batch=True):
        if not batch:
            next_states = np.expand_dims(next_states, 0)
        # next_states = np.clip((next_states - self.state_rms.mean) / (self.state_rms.var ** 0.5), -5, 5,
        #                       dtype="float32")  # dtype to avoid '.float()' call for pytorch.
        next_states = torch.from_numpy(next_states).type(torch.float32).to(self.device)
        actions = torch.from_numpy(actions).to(torch.int64).to(self.device)
        actions = torch.nn.functional.one_hot(actions, num_classes=self.n_actions)

        predictor_encoded_features = self.predictor_model(next_states).detach()
        target_encoded_features = self.target_model(next_states).detach()
        predictor_encoded_features = predictor_encoded_features.view(-1, self.timesteps, self.encoding_size)
        target_encoded_features = target_encoded_features.view(-1, self.timesteps, self.encoding_size)
        actions = actions.view(-1, self.timesteps, self.n_actions)

        if self.config["algo"] == "RND":
            rnd_loss = (predictor_encoded_features - target_encoded_features).pow(2).mean(-1)
            if not batch:
                return rnd_loss.detach().cpu().numpy()
            else:
                return rnd_loss.detach().cpu().numpy().reshape((self.config["n_workers"], self.config["rollout_length"]))

        #mask = from_numpy(np.random.choice([0,1], size=(len(target_encoded_features)//self.timesteps, self.pred_size), p=[0.5, 0.5]))



        mask = torch.randint(0, 2, size=(*predictor_encoded_features.shape[:-1], self.pred_size))
        #mask = torch.kron(mask, torch.ones(self.timesteps, dtype=torch.uint8))
        mask_inv = torch.where((mask==0)|(mask==1), mask^1, mask)

        temp_p = predictor_encoded_features*mask_inv
        temp_t = target_encoded_features*mask
        features = temp_p + temp_t

        disc_preds = self.discriminator(features, actions, target_encoded_features)
        #disc_preds = torch.kron(disc_preds, torch.ones(self.timesteps)).view(len(target_encoded_features), -1)
        disc_loss = self.f1_loss(disc_preds[:, 0], mask[:, 0]) if self.multiple_feature_pred else self.f1_loss(disc_preds, mask)

        if not batch:
            return disc_loss.detach().cpu().numpy()
        else:
            return disc_loss.detach().cpu().numpy().reshape((self.config["n_workers"], self.config["rollout_length"]))

    def calculate_loss(self, next_state, action): 
        target_encoded_features = self.target_model(next_state.view(-1, *self.obs_shape))
        predictor_encoded_features = self.predictor_model(next_state.view(-1, *self.obs_shape))

        predictor_encoded_features = predictor_encoded_features.view(-1, self.timesteps, self.encoding_size)
        target_encoded_features = target_encoded_features.view(-1, self.timesteps, self.encoding_size)

        # reconstructed_img = self.decoder(target_encoded_features)

        rnd_loss = (predictor_encoded_features - target_encoded_features).pow(2).mean(-1)
        mask = torch.rand(rnd_loss.size(), device=self.device)
        mask = (mask < self.config["predictor_proportion"]).float()
        rnd_loss = (mask * rnd_loss).sum() / torch.max(mask.sum(), torch.Tensor([1]).to(self.device))

        if self.config["algo"] == "RND":
            return 0, rnd_loss

        mask = torch.randint(0, 2, size=(*predictor_encoded_features.shape[:-1], self.pred_size))
        #mask = torch.kron(mask, torch.ones(self.timesteps, dtype=torch.uint8))
        mask_inv = torch.where((mask==0)|(mask==1), mask^1, mask)
            
        #mask = from_numpy(np.random.choice([0,1], size=(len(target_encoded_features)//self.timesteps, self.pred_size), p=[0.5, 0.5]))
        temp_p = predictor_encoded_features*mask_inv
        temp_t = target_encoded_features*mask
        features = temp_p + temp_t

        disc_preds = self.discriminator(features, action, target_encoded_features)
        #disc_preds = torch.kron(disc_preds, torch.ones(self.timesteps)).view(len(target_encoded_features), -1)
        disc_loss = torch.mean(self.f1_loss(disc_preds, mask))

        return disc_loss, rnd_loss

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