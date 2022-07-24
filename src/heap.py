from heapq import heapify, heappush, heappushpop, nlargest
import numpy as np
import torch
from operator import itemgetter
import time

class Experience:
    def __init__(self, priority, weight, data):
        self.priority = priority
        self.weight = weight
        self.data = data

class ReplayMemory():
    def __init__(self, rnd_target=None, rnd_predictor=None, max_capacity=1024, mode="proportional", n_parallel_env=8):
        self.heap = []
        self.max_capacity = max_capacity
        self.rnd_predictor = rnd_predictor
        self.rnd_target = rnd_target
        self.alpha = 0.6
        self.beta = 0.5
        self.mode = mode
        self.epsilon = 0.01
        self.priority_sum = 1
        self.max_w = 1
        self.max_p = 1
        self.idx = 0
        self.n_parallel_env = n_parallel_env

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        heapify(self.heap)

        self.weight_change = 0

    
        
    def add(self, element):


        # use idx to avoid clashes
        if len(self.getMax()) > 0:
            element = [self.getMax()[0][0], 1, (self.idx, element)]
        else:
            element = [1, 1, (self.idx, element)]


        if len(self.heap) < self.max_capacity:
            heappush(self.heap, element)
        else:
            heappushpop(self.heap, element)

        self.idx += 1
    
    def getAll(self):
        return nlargest(self.max_capacity, self.heap)

    def getMax(self):
        return nlargest(1, self.heap)

    def compute_rnd_loss(self, n):
        loss = 0.0
        heapify(self.heap)

        print(len(self.heap))
        k = int(np.ceil(n / self.n_parallel_env))


        for _ in range(k):
            priorities, weights, data = map(list, zip(*self.heap))
            self.max_w = max(weights)

            sample_idx = self.prioritized_sampling(priorities, self.n_parallel_env)

            weights = self.compute_is(priorities)
            
            states = np.array([d[1][7] for d in data])[sample_idx]



            delta = self.compute_delta(torch.Tensor(states).to(self.device))



            priorities = np.array(priorities, dtype=float)
            priorities[sample_idx] = delta.detach().cpu().numpy()

            loss += (delta * weights[sample_idx]).mean()

            for i in sample_idx:
                self.heap[i][0: 2] = priorities[i], weights[i].item()

        return loss
        
    
    def prioritized_sampling(self, priorities, n_parallel):         

        if self.mode == "proportional":

            # priorities = [np.abs(v) + self.epsilon for v in values]
            self.priority_sum = np.sum(np.power(priorities, self.alpha))
            new_priorities = np.power(priorities, self.alpha) / self.priority_sum
            indices = range(0, len(new_priorities))

        # if self.mode == "rankbased":

        #     priorities = [1/i for i in range(1, len(values) + 1)]
        #     priority_sum = np.sum(np.power(priorities, self.alpha))
        #     priorities = np.power(priorities, self.alpha) / priority_sum
        #     indices = np.flip(np.argsort(values.cpu().numpy()))

        else:
            raise ValueError

        # fraction = 1 if self.config["n_workers"] <= 32 else 32 / self.config["n_workers"]
        fraction = 1
        self.max_p = max(new_priorities)

        sampled_indices = np.random.choice(indices, size=n_parallel, p=new_priorities, replace=True)
        
        return sampled_indices


    def get_max_weight(self):
        pass
    
    def compute_is(self, priorities):
        N = len(priorities)
        weights = torch.pow(N * torch.Tensor(priorities), -self.beta) / self.max_w 
        return weights

    def compute_delta(self, data):
        encoded_target_features = self.rnd_target(data)
        encoded_predictor_features = self.rnd_predictor(data, 1)
        loss = (encoded_predictor_features - encoded_target_features).pow(2).mean(-1)

        return loss



if __name__ == '__main__':

    elem = [[1, 2, ("hi", 3)], [2, 2, ("AAA", 2)], [5, 0, ("ahh", 3)], [3, 3, ("ghhg", 1)], [9, 2, ("hgoisj", 98)], [52, 23, ("hgis", 4)], [3, 23, ("aaaaa", 123)]]

    heap = ReplayMemory(max_capacity=5)

    for e in elem:
        heap.add(e)

    print(heap.getAll())
    print(heap.heap)

    # print(heap.sample(40))