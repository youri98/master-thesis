# credit to https://github.com/gribeiro2004/Continuous-control-with-DDPG-and-prioritized-experience-replay/tree/main/Code

import argparse
import binascii
import random
import numpy as np
from SumTree import SumTree
from operator import itemgetter
import matplotlib.pyplot as plt
import torch

# class Memory:  # stored as ( s, a, r, s_ ) in SumTree
#     e = 0.01 
#     a = 0.6
#     beta = 0.4
#     beta_increment_per_sampling = 0.00001

#     def __init__(self, capacity):
#         self.tree = SumTree(capacity)
#         self.capacity = capacity

#     def _get_priority(self, error):
#         return (np.abs(error) + self.e) ** self.a

#     def push(self, *args):
#         # p = self._get_priority(error)

#         p_max = np.max(self.tree.tree[self.capacity -1 :])

#         if p_max == 0:
#             p_max = 1

#         self.tree.add(p_max, args)

#     def sample(self, n):
#         batch = []
#         idxs = []
#         segment = self.tree.total() / n
#         priorities = []

#         self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

#         for i in range(n):
#             a = segment * i
#             b = segment * (i + 1)

#             s = random.uniform(a, b)
#             (idx, p, data) = self.tree.get(s)
#             priorities.append(p)
#             batch.append(data)
#             idxs.append(idx)

#         sampling_probabilities = priorities / self.tree.total()
        
#         is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        
#         is_weight /= is_weight.max()

#         return [b[0] for b in batch], idxs, is_weight

    
#     def update_priorities(self, batch_indices, batch_errors):
#         for idx, error in zip(batch_indices, batch_errors):
#             p = self._get_priority(error)
#             self.tree.update(idx, p)


class PrioritizedReplay(object):
    """
    Proportional Prioritization
    """
    def __init__(self, capacity, beta_frames=1000000, config=None):
        self.config = config
        self.alpha = config["alpha"]
        self.beta_start = config["beta"]
        self.beta_frames = beta_frames
        self.frame = 1 #for beta calculation
        self.capacity   = capacity
        self.buffer     = np.zeros((capacity, *config["obs_shape"]), dtype=np.float64)
        self.pos        = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.priority_age = np.zeros((capacity,), dtype=np.uint8)
        self.state_room = np.zeros((capacity,), dtype=np.uint8)

        self.max_prio = 1.0
        self.fix_beta = config["fix_beta"]
        self.buffer_unit_size = self.config["n_workers"] * self.config["rollout_length"]
        self.temp_counter = 0

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.config = config
        self.theta = config["theta"]
        self.c = config["c"]
        self.k = config["k"]
        self.use_gamma = self.theta and self.k or self.config["use_weight_model"]        
        self.gamma = lambda x: (x ** (self.k - 1) * torch.exp(-x/self.theta)) / ((self.theta ** self.k) * torch.exp(torch.lgamma(self.k)))
        self.current_rooms = np.zeros(config["n_workers"], dtype=np.uint8)

        self.distribution = None
        
    
    def beta_by_frame(self, frame_idx):
        """
        Linearly increases beta from beta_start to 1 over time from 1 to beta_frames.
        
        3.4 ANNEALING THE BIAS (Paper: PER)
        We therefore exploit the flexibility of annealing the amount of importance-sampling
        correction over time, by defining a schedule on the exponent 
        that reaches 1 only at the end of learning. In practice, we linearly anneal from its initial value 0 to 1
        """
        if not self.fix_beta:
            return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)
        else:
            return self.beta_start
    
    def push_per_batch(self, states, int_rewards):
        # assert state.ndim == next_state.ndim
        # state      = np.expand_dims(state, 0)
        # next_state = np.expand_dims(next_state, 0)
        
        # max_prio = self.priorities.max() if self.buffer else 1.0 # gives max priority if buffer is not empty else 1
        # temp = list(zip(self.buffer, self.priorities, self.priority_age))
        # temp.sort(key=itemgetter(1))
        self.max_prio = self.priorities.max()

        int_rewards = int_rewards.flatten()

        self.buffer[self.pos: self.pos + len(states)] = states
        
        self.priorities[self.pos:self.pos + len(states)] = int_rewards if self.config["without_maximal_appending"] else self.max_prio 
        self.pos = (self.pos + len(states)) % self.capacity # lets the pos circle in the ranges of capacity if pos+1 > cap --> new posi = 0
    
    def push_gamma(self, states, int_rewards):
        int_rewards = int_rewards.flatten()
        # since most recent are appended at the end of the states array and we want most recent to oldest
        indices = [i for i in range(self.config["rollout_length"])] * self.config["n_workers"]

        temp = list(zip(states, indices, int_rewards))
        temp.sort(key=lambda v: (v[1], np.random.random()), reverse=True)
        states = [x[0] for x in temp]
        deltas = [x[2] for x in temp]

        if self.pos >= self.capacity:

            self.buffer[len(states):] = self.buffer[:-len(states)]
            self.buffer[:len(states)] = states

            self.priorities[len(states):] = self.priorities[:-len(states)]
            self.priorities[:len(states)] = deltas
            
        else:
            self.buffer[self.pos:self.pos+len(states)] = states
            self.priorities[self.pos:self.pos+len(states)] = deltas

            self.pos += len(states)
    
    def push_batch(self, states, int_rewards, infos):
        if self.use_gamma:
            self.push_gamma(states, int_rewards)
        elif self.config["sampling_algo"] == "per":
            self.push_per_batch(states, int_rewards)
        elif self.config["sampling_algo"] == "per-v2":
            self.push_per2_batch(states)
        elif self.config["sampling_algo"] == "per-v3":
            self.push_per3_batch(states, int_rewards)
        elif self.config["sampling_algo"] == "prioritize-room":
            self.push_prioritize_room(states, int_rewards, infos)


    def push_prioritize_room(self, states, int_rewards, infos):
        int_rewards = int_rewards.flatten()

        # infos = [x for info in infos for x in info] # flatten

        infos = [[max(x["episode"]["visited_room"]) if "episode" in x else 0 for x in info] for info in infos]
        
        infos = np.array(infos, dtype=np.uint8)
        self.current_rooms = infos[-1]




        self.buffer[self.pos: self.pos + len(states)] = states
        self.priorities[self.pos: self.pos + len(states)] = int_rewards
        self.state_room[self.pos: self.pos + len(states)] = infos.flatten()

        room_priority = np.isin(self.state_room, self.current_rooms).flatten()#.astype(np.uint8)

        self.priorities[room_priority] = self.max_prio # increase corresponding rooms
        
        
        self.pos = np.mod(self.pos + len(states), self.capacity)

        self.distribution = self.priorities.copy() # for plotting distribution
        self.max_prio = self.priorities.max()




    def push_per2_batch(self, states):
        # sort
        if self.pos >= self.capacity:
            temp = list(zip(self.buffer, self.priorities, self.priority_age))
            temp.sort(key=itemgetter(1))

            for i in range(len(temp)):
                self.buffer[i], self.priorities[i], self.priority_age[i] = temp[i]


            self.buffer[:len(states)] = states
            self.priorities[:len(states)] = self.max_prio
            self.priority_age[:len(states)] = 1
            self.priority_age[len(states):] += 1
            
            self.max_prio = self.priorities[-1] # not including the maximal appendage as this results in highly skewed towards recent ones

        else:
            self.buffer[self.pos:self.pos + len(states)] = states
            self.priority_age[:min(self.pos + len(states), len(self.priority_age))] += 1
            self.priorities[:len(self.buffer)] = self.max_prio
            self.pos += len(states)

            self.max_prio = self.priorities.max() # gives max priority if buffer is not empty else 1

        
        self.distribution = self.priorities.copy() # for plotting distribution

    
    def push_per3_batch(self, states, errors):

        # sort
        if self.pos >= self.capacity:
            appended_buffer = np.concatenate([self.buffer, states])
            appended_priorities = np.concatenate([self.priorities, np.full(len(states), self.max_prio)])
            appended_priority_age = np.concatenate([self.priority_age, np.zeros(len(states))])



            temp = list(zip(appended_buffer, appended_priorities, appended_priority_age))
            temp.sort(key=itemgetter(1))

            temp = temp[len(states):]

            for i in range(len(temp)):
                self.buffer[i], self.priorities[i], self.priority_age[i] = temp[i]

            self.priority_age[:] += 1
        else:
            self.buffer[self.pos:self.pos + len(states)] = states
            self.priority_age[:min(self.pos + len(states), len(self.priority_age))] += 1
            self.priorities[:len(self.buffer)] = self.max_prio
            self.pos += len(states)

        self.max_prio = self.priorities.max() # gives max priority if buffer is not empty else 1

    def get_priority_age(self):
        bins = [0, 1, 2, 3, 4, 6, 8, 10, 15, 20, 30, 50, 100]
        
        
        counts, age = np.histogram(self.priority_age, bins=bins)
        percentage = counts / len(self.priority_age)

        return (age, counts), self.priority_age


    def sample_gamma(self, batch_size):
        
        probs = [self.gamma(x/self.buffer_unit_size) for x in range(self.pos)]
        probs = probs / sum(probs)
        self.temp_counter += 1


        padding = self.capacity - len(probs)

        if padding > 0:
            probs = np.append(probs, np.zeros(padding))


        # if self.temp_counter == 16*self.config["mem_size"]:
        #     plt.plot([i for i in range(self.capacity)], probs)
        #     plt.show()

        self.distribution = probs
            


        # while len(indices) < batch_size:
        #     idx_float = np.random.gamma(self.k, self.theta, 1)[0]
        #     idx = round(self.capacity / idx_float)
        #     if idx < self.capacity and idx < self.pos: # in beginning cases when memory isnt completely filled
        #         indices.append(idx)
        
        indices = np.random.choice([i for i in range(self.capacity)], batch_size, p=probs)
        samples = self.buffer[indices]
        samples = np.array([s[0] for s in samples], dtype=np.float32)

        return samples, None, None

    def sample_with_gamma_weights(self, batch_size):
        self.alpha = 1

        # uniform sampling

        indices = np.random.choice([i for i in range(self.pos)], batch_size) 

        # gamma weighting
        gamma_part = torch.Tensor([((x+1)/(self.buffer_unit_size+1)) for x in range(self.pos)]).to(self.device)
        gamma_part = torch.squeeze(self.gamma(gamma_part))
        gamma_part = gamma_part / torch.sum(gamma_part) # to make it proportional 

        # gamma_part = [self.gamma(x/self.buffer_unit_size) for x in range(self.pos)]
        delta_part = torch.Tensor([self.priorities[i] for i in range(self.pos)]).to(self.device)
        delta_part = delta_part / torch.sum(delta_part)

        #total_func = lambda g,d: ((g)/self.c + d/(1 - self.c)) ** self.alpha
        total_func = lambda g,d: (g*self.c + d*(1 - self.c)) ** self.alpha
        samples_weight = gamma_part * self.c + (delta_part  ** self.alpha) * (1- self.c)
        samples_weight = torch.squeeze(samples_weight)
        # samples_weight = torch.Tensor(list(map(total_func, [gamma_part, delta_part])))

        # samples_weight = [total_func(g,d) for g,d in zip(gamma_part, delta_part)]
        self.distribution = samples_weight

        # samples_weight = np.array(samples_weight, dtype=np.float32)

        samples_weight = samples_weight[indices]
        samples = self.buffer[indices]
        samples = np.array([s[0] for s in samples], dtype=np.float32)
        samples = torch.from_numpy(np.expand_dims(samples, axis=1)).to(self.device)


        return samples, indices, samples_weight


    def sample(self, batch_size):
        if self.use_gamma:
            return self.sample_with_gamma_weights(batch_size)
            #return self.sample_gamma(batch_size)
        else:

            N = len(self.buffer)
            if N == self.capacity:
                prios = self.priorities
            else:
                prios = self.priorities[:self.pos]
                
            # calc P = p^a/sum(p^a)
            probs  = prios ** self.alpha

            if probs.sum() != 0:
                P = probs/probs.sum()
            else: 
                P = np.ones(len(probs)) / len(probs)
            
            #gets the indices depending on the probability p
            indices = np.random.choice(N, batch_size, p=P) 
            samples = [self.buffer[idx] for idx in indices]
            

            beta = self.beta_by_frame(self.frame)
            self.frame+=1
                    
            #Compute importance-sampling weight
            weights  = (N * P[indices]) ** (-beta)
            # normalize weights
            weights /= weights.max() 
            weights  = np.array(weights, dtype=np.float32) 

            samples = np.array([s[0] for s in samples], dtype=np.float32)
            samples = torch.from_numpy(np.expand_dims(samples, axis=1)).to(self.device)

            weights = torch.from_numpy(weights).to(self.device)


            return samples, indices, weights

    def get_time(self):
        self.priority_age    

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = abs(prio) 

    def __len__(self):
        return len(self.buffer)

class DefaultMemory(object):
    def __init__(self, capacity, batch_size, state_shape):
        self.i = 0
        self.batch_size = batch_size
        self.memory_length = self.batch_size * capacity
        self.memory = np.zeros((self.memory_length, *state_shape), dtype=np.float32)

    def sample(self, mini_batch_size):
        indices = np.random.randint(0, len(self.memory), size=mini_batch_size)
        return self.memory[indices]

    def push_batch(self, states):
        self.memory[self.i: self.i + self.batch_size] = states
        self.i = np.mod(self.i + self.batch_size, self.memory_length)
