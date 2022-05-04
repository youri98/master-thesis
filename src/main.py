from config import get_params
import numpy as np
from numpy import concatenate
from utils import delete_files
import gym
from tqdm import tqdm
from rnd import RND
from ape import APE
from logger import Logger
from torch.multiprocessing import Process, Pipe
from runner import Worker
import torch

torch.autograd.set_detect_anomaly(True)


def run_workers(worker, conn):
    worker.step(conn)

def train_model(model_name=None, n_to_chkpt=3):
    recording = []

    config = get_params()
    temp_env = gym.make(config["env_name"])
    config.update({"n_actions": temp_env.action_space.n})
    temp_env.close()

    config.update({"batch_size": (config["rollout_length"] * config["n_workers"]) // config["n_mini_batch"]})
    config.update({"predictor_proportion": 32 / config["n_workers"]})

    agent = APE(**config)
    logger = Logger(agent, resume=config["mode"] == "train_from_chkpt", **config)

    workers = [Worker(i, **config) for i in range(config["n_workers"])]
    
    init_iteration = 0
    episode = 0
    visited_rooms = set([1])
    workers = [Worker(i, **config) for i in range(config["n_workers"])]
    episode_ext_reward = 0

    parents = []
    for worker in workers:
        parent_conn, child_conn = Pipe()
        p = Process(target=run_workers, args=(worker, child_conn,))
        p.daemon = True
        parents.append(parent_conn)
        p.start()
        
    if config["mode"] == "test" or config["mode"] == "train_from_chkpt":
        if model_name is not None:
            logger.log_dir = model_name
        chkpt = logger.load_weights(model_name)
        agent.set_from_checkpoint(chkpt)
        init_iteration = chkpt["iteration"]
        episode = chkpt["episode"]

    if config["mode"] == "test":
        pass
    
    else:
        rollout_base_shape = config["n_workers"], config["rollout_length"]

        logger.time_start()
        for iteration in tqdm(range(init_iteration, config["total_rollouts_per_env"])):
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
                    total_next_obs[worker_id, t] = state_[-1, ...]

                    if worker_id == 0 and config["record_local"]:
                        recording.append(state_)

                episode_ext_reward += total_ext_rewards[0, t]

                if total_dones[0, t]:
                    if "episode" in infos[0]:
                        visited_rooms = infos[0]["episode"]["visited_room"]
                        logger.log_episode(episode, episode_ext_reward, visited_rooms)
                    episode_ext_reward = 0

            total_next_obs = concatenate(total_next_obs)
            total_int_rewards = agent.calculate_int_rewards(total_next_obs)
            _, next_int_values, next_ext_values, * \
                _ = agent.get_actions_and_values(next_states, batch=True)

            total_int_rewards = agent.normalize_int_rewards(total_int_rewards)

            train_logs = agent.train(states=concatenate(total_states),
                                                        actions=concatenate(
                                                            total_actions),
                                                        int_rewards=total_int_rewards,
                                                        ext_rewards=total_ext_rewards,
                                                        dones=total_dones,
                                                        int_values=total_int_values,
                                                        ext_values=total_ext_values,
                                                        log_probs=concatenate(
                                                            total_log_probs),
                                                        next_int_values=next_int_values,
                                                        next_ext_values=next_ext_values,
                                                        total_next_obs=total_next_obs)
            logger.log_iteration(iteration,
                                    train_logs,
                                    total_int_rewards[0].mean(),
                                    total_ext_rewards[0].mean(),
                                    total_action_probs[0].max(-1).mean(),
                                    )

            if iteration % n_to_chkpt == 0 or iteration == config["total_rollouts_per_env"]:
                logger.save_params(episode, iteration)
                logger.save_recording(recording)
                logger.time_stop()

if __name__ == '__main__':
    #delete_files()
    train_model()