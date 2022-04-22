import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from tqdm import tqdm


class PPOMemory:
    def __init__(self, batch_size) -> None:
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards_e = []
        self.rewards_i = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        #np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return T.stack(self.states),\
            np.array(self.actions),\
            np.array(self.probs), \
            np.array(self.rewards_e),\
            np.array(self.rewards_i),\
            np.array(self.dones),\
            batches

    def store_memory(self, state, action, probs, reward_e, reward_i, done):
        self.states.append(state)
        self.probs.append(probs)
        self.rewards_e.append(reward_e)
        self.rewards_i.append(reward_i)
        self.dones.append(done)
        self.actions.append(action)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards_e = []
        self.rewards_i = []
        self.dones = []


class RND(nn.Module):
    def __init__(self, chkpt_dir='tmp/ppo', alpha=0.001):
        super(RND, self).__init__()

        self.target = CommonNetwork(filename='target')
        self.predictor = CommonNetwork(filename='predictor')
        
        self.checkpoint_file = os.path.join(chkpt_dir, 'RND')

        self.optimizer = optim.Adam(self.predictor.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        z_true = self.target(state).detach()
        z_pred = self.predictor(state)
        rnd_error = T.pow(z_pred - z_true, 2).sum(1)

        return rnd_error

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class CommonNetwork(nn.Module):
    def __init__(self, conv1_dims=32, conv2_dims=64, conv3_dims=64, chkpt_dir='tmp/ppo', filename='CommonNN', alpha=0.001):
        super(CommonNetwork, self).__init__()

        self.conv1_dims = conv1_dims
        self.conv2_dims = conv2_dims
        self.conv3_dims = conv3_dims

        self.checkpoint_file = os.path.join(chkpt_dir, filename)

        self.encoder = nn.Sequential(
                nn.Conv2d(1, conv1_dims, kernel_size=8, stride=4),
                nn.ELU(),
                nn.Conv2d(conv1_dims, conv2_dims, kernel_size=4, stride=2),
                nn.ELU(),
                nn.Conv2d(conv2_dims, conv3_dims,  kernel_size=3, stride=1),
                nn.ELU(),
                nn.Flatten(start_dim=1),
                nn.Linear(22528, 512),
                nn.ELU(),
                nn.Linear(512, 512),
                nn.ELU(),
                nn.Linear(512, 512),
        )

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        enc = self.encoder(state)
        return enc

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, conv1_dims=32, conv2_dims=64, conv3_dims=64, chkpt_dir='tmp/ppo', filename='actor', alpha=0.001):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, filename)

        self.actor = nn.Sequential(
                nn.Conv2d(1, conv1_dims, kernel_size=8, stride=4),
                nn.ELU(),
                nn.Conv2d(conv1_dims, conv2_dims, kernel_size=4, stride=2),
                nn.ELU(),
                nn.Conv2d(conv2_dims, conv3_dims,  kernel_size=3, stride=1),
                nn.ELU(),
                nn.Flatten(start_dim=1),
                nn.Linear(22528, 512),
                nn.ELU(),
                nn.Linear(512, n_actions),
                nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)
        return dist
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))



class PPOAgent:
    def __init__(self, n_actions, input_dims,  gamma=.99, alpha=.0003, gae_lambda=.95, policy_clip=.1, batch_size=64, N=2048, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = ActorNetwork(n_actions)
        self.critic = RND()
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, probs, reward_e, reward_i, done):
        self.memory.store_memory(state, action, probs, reward_e, reward_i, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        state = observation.to(self.actor.device)
        state = T.unsqueeze(state, dim=0)

        dist = self.actor(state)
        reward_i = self.critic(state)
        action = dist.sample()

        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        reward_i = T.squeeze(reward_i).item()

        return action, probs, reward_i

    def learn(self, verbose=True):
        for _ in tqdm(range(self.n_epochs)):
            critic_epoch_loss = 0
            actor_epoch_loss = 0

            state_arr, action_arr, old_probs_arr, reward_arr_e, reward_arr_i, done_arr, batches = self.memory.generate_batches()

            # actor learn
            reward_arr_i = self.critic(state_arr)
            # reward_arr_i = T.from_numpy(reward_arr_i).to(self.actor.device)

            # TODO: formula

            # TODO: separate critic from actor loss
            advantage = np.zeros(len(reward_arr_e), dtype=np.float32)

            for t in range(len(reward_arr_e)-1):
                discount = 1
                a_t = 0

                for k in range(t, len(reward_arr_e)-1):
                    a_t += discount * \
                        (reward_arr_e[k] + self.gamma*reward_arr_i[k+1]
                         * (1 - int(done_arr[k])) - reward_arr_i[k])
                    discount * - self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)


            for batch in batches:
                states = state_arr[batch].to(
                    self.actor.device)
                old_probs = T.from_numpy(old_probs_arr[batch]).to(
                    self.actor.device)
                actions = T.from_numpy(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)

                #RND here
                critic_value = self.critic(states)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(
                    prob_ratio, 1-self.policy_clip, 1 + self.policy_clip)*advantage[batch]

                actor_loss = -T.min(weighted_probs,
                                    weighted_clipped_probs).mean()
                                    
                #returns = advantage[batch] + reward_arr_i[batch]
                #critic_loss = (returns - critic_value) ** 2
                #critic_loss = critic_loss.mean()

                #total_loss = actor_loss + 0.5 * critic_loss
                #total_loss.backward()
                actor_epoch_loss += actor_loss.item()

                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                critic_loss = critic_value.sum()
                critic_epoch_loss += critic_loss.item()

                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                self.critic.optimizer.step()

            if verbose:
                print(f"ACTOR LOSS: {actor_epoch_loss} \nCRITIC LOSS: {critic_epoch_loss}") 


        self.memory.clear_memory()
