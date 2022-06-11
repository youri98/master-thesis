from inspect import modulesbyfile
from config import get_params
import numpy as np
from numpy import concatenate
import gym
from tqdm import tqdm
from ape import APE
from rnd import RND
from logger import Logger
from torch.multiprocessing import Process, Pipe
from runner import Worker
import torch
import wandb
import multiprocessing
import os
import pygad
import time
import gym
import numpy as np
import pygad.torchga
import pygad
import torch
import torch.nn as nn
from multiprocessing import Pool
from utils import *
import pygad.gacnn
from torch.distributions.categorical import Categorical
from collections import deque

gpu = True
torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

config = get_params()
config["algo"] = "APE"
config["total_rollouts"] = 10
config["verbose"] = True
config["record"] = True
# # run 1
# config["env"] = "VentureNoFrameskip-v4"
# config["total_rollouts"] = int(7)
# config["algo"] = "RND"
# config["verbose"] = True
config["interval"] = 100

print("STARTED")
with open("key.txt", "r") as personal_key:
    if personal_key is not None:
        os.environ["WANDB_API_KEY"] = personal_key.read().strip()

temp_env = gym.make(config["env"])
config.update({"n_actions": temp_env.action_space.n})
temp_env.close()
config["n_workers"] = multiprocessing.cpu_count() #* torch.cuda.device_count() if torch.cuda.is_available() else multiprocessing.cpu_count()
config.update({"batch_size": (config["rollout_length"] * config["n_workers"]) // config["n_mini_batch"]})
config.update({"predictor_proportion": min(1, 32 / config["n_workers"])})
workers = [Worker(i, **config) for i in range(config["n_workers"])] 

agent = APE(**config)
logger = Logger(agent, **config)
logger.log_config_params()

if not config["verbose"]:
    os.environ["WANDB_SILENT"] = "true"   
else:
    print("params:", config)



recording, visited_rooms, init_iteration, episode, episode_ext_reward = [], set([1]), 0, 0, 0
if config["mode"] == "test" or config["mode"] == "train_from_chkpt":
    if config["policy_model_name"] is not None:
        logger.log_dir = config["policy_model_name"] 
    chkpt = logger.load_weights(config["policy_model_name"])
    agent.set_from_checkpoint(chkpt)
    init_iteration = int(chkpt["iteration"])
    episode = chkpt["episode"]




class PooledGA(pygad.GA):

    def cal_pop_fitness(self):
        if not hasattr(self, "frames"):
            self.frames = 0
        # if not hasattr(self, "iteration"):
        #     self.iteration = 0
        
        global pool, agent

        output = pool.map(fitness_wrapper, self.population)
        pop_fitness, episode_logs, states =  zip(*output)

        states = np.concatenate(states)

        episode_logs = list(filter(None, episode_logs))

        self.frames += states.shape[0] * states.shape[1]


        # train rnd
        agent.train_rnd(states)

        wandb.log(episode_logs[0])
        wandb.log({"N Frames": self.frames})

        print(episode_logs, self.frames)
        pop_fitness = np.array(pop_fitness)
        return pop_fitness

def callback_generation(ga_instance):

    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))

def fitness_wrapper(solution):
    return fitness_func(solution, 0)

def run_workers_env_step(worker, conn):
    worker.step(conn)

def fitness_func(solution, sol_idx):
    global policy_model, envs, device, config, agent
    # print(len(solution), sol_idx)

    current_pool_id = multiprocessing.current_process()._identity[0] - 2 # dont get why its 2 tm 9
    print ('running: ', current_pool_id)
    policy_model_weights_dict = pygad.torchga.model_weights_as_dict(model=policy_model, weights_vector=solution)
    policy_model.load_state_dict(policy_model_weights_dict)

    # initialize env
    episode_ext_reward, episode_int_reward, all_states, sum_reward, done, t = [], [], [], 0, False, 1

    state_shape = config["state_shape"]
    env = envs[current_pool_id] 
    state = env.reset() # firtst observation

    _stacked_states = np.zeros(state_shape, dtype=np.uint8) # stacking 4 observations
    _stacked_states = stack_states(_stacked_states, state, True)
    
    # rollout length / until dead
    while t <= config["max_frames_per_episode"] and not done:
        state = torch.from_numpy(_stacked_states).to(device)
        
        with torch.no_grad():
            output = policy_model(torch.unsqueeze(state.type(torch.float), 0))
            int_value, ext_value, action_prob = output
            dist = Categorical(action_prob)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        # action, int_value, ext_value, log_prob, action_prob = agent.get_actions_and_values(_stacked_states, batch=True)
        next_state, r, done, info = env.step(action)

        if t % max_episode_steps == 0:
            done = True


        _stacked_states = stack_states(_stacked_states, next_state, False)

        if "episode" in info and current_pool_id == 0 and done:
            visited_rooms = info["episode"]["visited_room"]
            episode = info["episode"]

            episode_logs = {"Ep Visited Rooms": list(visited_rooms), "Episode Ext Reward": sum(episode_ext_reward)}

        
        episode_ext_reward.append(r)
        t += 1
        all_states.append(_stacked_states)

    # r_i = agent.calculate_int_rewards(all_states)
    # r_i = agent.normalize_int_rewards(r_i)
    episode_int_reward = [0]

    sum_reward = config["ext_adv_coeff"]*sum(episode_int_reward) + config["int_adv_coeff"]*sum(episode_ext_reward)

    if "episode_logs" in locals():
        return sum_reward, episode_logs, all_states
    else:
        return sum_reward, None, all_states

#######

if config["mode"] == "test":
    pass

else:
    rollout_base_shape = config["n_workers"], config["rollout_length"]

    init_states = np.zeros(rollout_base_shape + config["state_shape"], dtype=np.uint8)
    init_actions = np.zeros(rollout_base_shape, dtype=np.uint8)
    init_action_probs = np.zeros(rollout_base_shape + (config["n_actions"],))
    init_int_rewards = np.zeros(rollout_base_shape)
    init_ext_rewards = np.zeros(rollout_base_shape)
    init_dones = np.zeros(rollout_base_shape, dtype=np.bool)
    init_int_values = np.zeros(rollout_base_shape)
    init_ext_values = np.zeros(rollout_base_shape)
    init_log_probs = np.zeros(rollout_base_shape)
    init_next_states = np.zeros((rollout_base_shape[0],) + config["state_shape"], dtype=np.uint8)
    init_next_obs = np.zeros(rollout_base_shape + config["obs_shape"], dtype=np.uint8)
    recording = []

    total_states = init_states
    total_actions = init_actions
    total_action_probs = init_action_probs
    total_int_rewards = init_int_rewards
    total_ext_rewards = init_ext_rewards
    total_dones = init_dones
    total_int_values = init_int_values
    total_ext_values = init_ext_values
    total_log_probs =init_log_probs
    next_states = init_next_states
    total_next_obs = init_next_obs
    total_next_states = np.zeros(rollout_base_shape + config["obs_shape"], dtype=np.uint8)
    # print("iteration how many frames", total_states.shape)

    config["n_workers"] = 8
    env_name = config["env"]
    max_episode_steps = config["max_frames_per_episode"]
    state_shape = config["state_shape"]
    envs = [make_atari(env_name, max_episode_steps) for i in range(config["n_workers"])] # change this outside call    
    episode_logs = {}
    policy_model = agent.current_policy
    rnd = agent.predictor_model
  

    torch_ga = pygad.torchga.TorchGA(model=policy_model, num_solutions=8)

    # Prepare the PyGAD parameters. Check the documentation for more information: https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#pygad-ga-class
    num_generations = 50  # Number of generations.
    num_parents_mating = 5  # Number of solutions to be selected as parents in the mating pool.
    initial_population = torch_ga.population_weights  # Initial population of network weights
    parent_selection_type = "sss"  # Type of parent selection.
    crossover_type = "single_point"  # Type of the crossover operator.
    mutation_type = "random"  # Type of the mutation operator.
    mutation_percent_genes = 10  # Percentage of genes to mutate. This parameter has no action if the parameter mutation_num_genes exists.
    keep_parents = -1  # Number of parents to keep in the next population. -1 means keep all parents and 0 means keep nothing.


    ga_instance = PooledGA(num_generations=num_generations,
                        num_parents_mating=num_parents_mating,
                        fitness_func=fitness_func,
                        parent_selection_type=parent_selection_type,
                        crossover_type=crossover_type,
                        mutation_type=mutation_type,
                        mutation_percent_genes=mutation_percent_genes,
                        keep_parents=keep_parents,
                        on_generation=callback_generation,
                        initial_population=initial_population)
                        # sol_per_pop=10,
                        # num_genes=300)

    with Pool(processes=config["n_workers"]) as pool:
        ga_instance.run()







    total_int_rewards = agent.calculate_int_rewards(total_next_obs)


    total_int_rewards = agent.normalize_int_rewards(total_int_rewards)
    


    training_logs = agent.train(states=concatenate(total_states),
                    actions=total_actions,
                    int_rewards=total_int_rewards,
                    ext_rewards=total_ext_rewards,
                    dones=total_dones,
                    int_values=total_int_values,
                    ext_values=total_ext_values,
                    log_probs=concatenate(total_log_probs),
                    next_int_values=next_int_values,
                    next_ext_values=next_ext_values,
                    total_next_obs=total_next_obs)

    n_frames = total_states.shape[0] * total_states.shape[1] * total_states.shape[2] * (iteration + 1)
    logger.log_iteration(iteration,
                            n_frames,
                            training_logs,
                            total_int_rewards[0].mean(),
                            total_ext_rewards[0].mean(),
                            total_action_probs[0].max(-1).mean(),
                            recording_int_rewards.mean(),
                            )
    


    # if iteration % config["interval"] == 0 or iteration == config["total_rollouts"]:
    #     logger.save_params(episode, iteration)
    #     logger.save_score_to_json()
    #     logger.time_stop()



    # print("---Pre_normalization started.---")
    # states = []
    # total_pre_normalization_steps = config["rollout_length"] * config["pre_normalization_steps"]
    # actions = np.random.randint(0, config["n_actions"], (total_pre_normalization_steps, config["n_workers"]))
    # for t in range(total_pre_normalization_steps):

    #     for worker_id, parent in enumerate(parents):
    #         parent.recv()  # Only collects next_states for normalization.

    #     for parent, a in zip(parents, actions[t]):
    #         parent.send(a)

    #     for parent in parents:
    #         s_, *_ = parent.recv()
    #         states.append(s_[-1, ...].reshape(1, 84, 84))

    #     if len(states) % (config["n_workers"] * config["rollout_length"]) == 0:
    #         agent.state_rms.update(np.stack(states))
    #         states = []
    # print("---Pre_normalization is done.---")
