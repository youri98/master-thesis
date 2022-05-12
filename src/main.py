from common.config import get_params
import numpy as np
from numpy import concatenate
import gym
from tqdm import tqdm
from common.ape import APE
from common.logger import Logger
from torch.multiprocessing import Process, Pipe
from common.runner import Worker
import torch
import wandb
import multiprocessing
import os
import pickle

import sys, os
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + "/common")

torch.autograd.set_detect_anomaly(True)


def run_workers(worker, conn):
    worker.step(conn)

def train_model():
    with open("key.txt", "r") as personal_key:
        if personal_key is not None:
            os.environ["WANDB_API_KEY"] = personal_key.read().strip()

    recording = []

    temp_env = gym.make(config["env_name"])
    config.update({"n_actions": temp_env.action_space.n})
    temp_env.close()
    
    config["n_workers"] = multiprocessing.cpu_count()

    config.update({"batch_size": (config["rollout_length"] * config["n_workers"]) // config["n_mini_batch"]})
    config.update({"predictor_proportion": 32 / config["n_workers"]})

    agent = APE(**config)

    if config["mode"] == "train_from_chkpt":
        with open("Models/" + config["model_name"] + "/logger.obj", "rb") as file_pi:
            logger = pickle.load(file_pi)
            logger.reboot()
    else:        
        logger = Logger(agent, **config)
        logger.log_config_params()

    workers = [Worker(i, **config) for i in range(config["n_workers"])] 
    
    init_iteration = 0
    episode = 0
    visited_rooms = set([1])
    workers = [Worker(i, **config) for i in range(config["n_workers"])]
    episode_ext_reward = 0

    if not config["verbose"]:
        os.environ["WANDB_SILENT"] = "true"   
    else:
        print("params:", config)


    parents = []
    for worker in workers:
        parent_conn, child_conn = Pipe()
        p = Process(target=run_workers, args=(worker, child_conn,))
        p.daemon = True
        parents.append(parent_conn)
        p.start()
        
    if config["mode"] == "test" or config["mode"] == "train_from_chkpt":
        if config["model_name"] is not None:
            logger.log_dir = config["model_name"] 
        chkpt = logger.load_weights(config["model_name"])
        agent.set_from_checkpoint(chkpt)
        init_iteration = chkpt["iteration"]
        episode = chkpt["episode"]

    if config["mode"] == "test":
        pass
    
    else:
        rollout_base_shape = config["n_workers"], config["rollout_length"]

        for iteration in tqdm(range(init_iteration, config["total_rollouts"]), disable=not config["verbose"]):
            total_states = np.zeros(rollout_base_shape + config["state_shape"], dtype=np.uint8)
            total_actions = np.zeros(rollout_base_shape, dtype=np.uint8)
            total_action_probs = np.zeros(rollout_base_shape + (config["n_actions"],))
            total_int_rewards = np.zeros(rollout_base_shape)
            total_ext_rewards = np.zeros(rollout_base_shape)
            total_dones = np.zeros(rollout_base_shape, dtype=bool)
            total_int_values = np.zeros(rollout_base_shape)
            total_ext_values = np.zeros(rollout_base_shape)
            total_log_probs = np.zeros(rollout_base_shape)
            next_states = np.zeros((rollout_base_shape[0],) + config["state_shape"], dtype=np.uint8)
            total_next_obs = np.zeros(rollout_base_shape + config["obs_shape"], dtype=np.uint8)
            total_next_states = np.zeros(rollout_base_shape + config["obs_shape"], dtype=np.uint8)

            logger.time_start()

            for t in range(config["rollout_length"]):
                infos = []

                for worker_id, parent in enumerate(parents):
                    total_states[worker_id, t] = parent.recv()

                total_actions[:, t], total_int_values[:, t], total_ext_values[:, t], total_log_probs[:, t], \
                total_action_probs[:, t] = agent.get_actions_and_values(total_states[:, t], batch=True)

                for parent, a in zip(parents, total_actions[:, t]):
                    parent.send(a)

                for worker_id, parent in enumerate(parents):
                    state_, reward, done, info = parent.recv()
                    infos.append(info)
                    total_ext_rewards[worker_id, t] = reward
                    total_dones[worker_id, t] = done
                    next_states[worker_id] = state_
                    total_next_states[worker_id, t] = state_
                    #total_next_obs[worker_id, t] = state_[-1, ...]

                    if worker_id == 0:
                        recording.append(state_)

                episode_ext_reward += total_ext_rewards[0, t]

                if total_dones[0, t]:
                    if "episode" in infos[0]:
                        visited_rooms = infos[0]["episode"]["visited_room"]
                        logger.log_episode(episode, episode_ext_reward, visited_rooms)
                    episode_ext_reward = 0

            total_next_states = concatenate(total_next_states)
            total_actions = concatenate(total_actions)

            total_int_rewards = agent.calculate_int_rewards(total_next_states, total_actions)
            _, next_int_values, next_ext_values, * \
                _ = agent.get_actions_and_values(total_next_states, batch=True)

            total_int_rewards = agent.normalize_int_rewards(total_int_rewards)

            train_logs = agent.train(states=concatenate(total_states),
                                                        actions=total_actions,
                                                        int_rewards=total_int_rewards,
                                                        ext_rewards=total_ext_rewards,
                                                        dones=total_dones,
                                                        int_values=total_int_values,
                                                        ext_values=total_ext_values,
                                                        log_probs=concatenate(
                                                            total_log_probs),
                                                        next_int_values=next_int_values,
                                                        next_ext_values=next_ext_values,
                                                        next_states=total_next_states)
            logger.time_stop()
            logger.log_iteration(iteration,
                                    train_logs,
                                    total_int_rewards[0].mean(),
                                    total_ext_rewards[0].mean(),
                                    total_action_probs[0].max(-1).mean(),
                                    )
            
            recording = np.stack(recording)
            logger.log_recording(recording)
            recording = []

            if iteration % config["interval"] == 0 or iteration == config["total_rollouts"]:
                logger.save_params(episode, iteration)
                logger.save_recording_local(recording)
                logger.time_stop()

                with open("Models/" + logger.log_dir + "/logger.obj", "wb") as file_pi:
                    pickle.dump(logger, file_pi)

if __name__ == '__main__':
    #delete_files()
    # train_model()
    # wandb.finish()

    config = get_params()

    # run 1
    config["total_rollouts"] = int(10)
    config["algo"] = "RND"
    config["verbose"] = True

    train_model()
    wandb.finish()


