import torch
import numpy as np

timesteps = 16
n_min_batch = 4
parallel_envs = 56

mini_batch_size = 128*parallel_envs//4
fraction = min(32/parallel_envs, 1)
len_states = 128* parallel_envs

rollout_len = 128




iteration_idx = [x for x in range(rollout_len//timesteps)] * n_min_batch
iteration_idx = np.array(iteration_idx).reshape(4, -1)

mask = np.random.rand(4, 128//timesteps) <= fraction
iteration_idx = [iteration_idx[batch][mask[batch]] for batch in range(n_min_batch)]
iteration_idx = [list(range(timesteps*idx, timesteps*idx +timesteps)) for batch_idx in iteration_idx for idx in batch_idx]
print(mask)
print(iteration_idx)
# iteration_idx = np.random.choice(sub_array, ) for sub_array in iteration_idx






# l = [x for x in range(128*8)]
# l = torch.Tensor(l).int()
# l = l.view(-1, 16)

# iteration_idx = [x for x in range(0, len_states, timesteps)]
# print(iteration_idx, len(iteration_idx))

# chosen_idx = np.random.choice(iteration_idx, size=int(len_states//timesteps * min(32/parallel_envs, 1)), replace=False)
# print(chosen_idx, len(chosen_idx))

# chosen_idx = np.concatenate([list(range(idx - 1, idx+timesteps)) for idx in chosen_idx])
# print(chosen_idx[:100])

# indices = np.concatenate([list(range(idx, idx + timesteps)) for batch in indices for idx in batch])
# indices = np.reshape(indices, (n_min_batch, -1))
# print(indices[0])