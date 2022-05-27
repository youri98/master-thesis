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
import pickle
import torch.multiprocessing as mp

gpu = True

torch.autograd.set_detect_anomaly(True)


def run_workers_env_step(worker, conn):
    worker.step(conn)

def train_model(config, **kwargs):
    with open("key.txt", "r") as personal_key:
        if personal_key is not None:
            os.environ["WANDB_API_KEY"] = personal_key.read().strip()

    temp_env = gym.make(config["env"])
    config.update({"n_actions": temp_env.action_space.n})
    temp_env.close()
    config["n_workers"] = multiprocessing.cpu_count() * torch.cuda.device_count() if torch.cuda.is_available() else multiprocessing.cpu_count()
    config.update({"batch_size": (config["rollout_length"] * config["n_workers"]) // config["n_mini_batch"]})
    config.update({"predictor_proportion": 32 / config["n_workers"]})
    workers = [Worker(i, **config) for i in range(config["n_workers"])] 

    if config['algo'] == 'APE':
        agent = APE(**kwargs, **config)
    else:
        agent = RND(**kwargs, **config)

    if config["mode"] == "train_from_chkpt":
        with open("Models/" + config["model_name"] + "/logger.obj", "rb") as file_pi:
            logger = pickle.load(file_pi)
            logger.reboot()
    else:        
        logger = Logger(agent, **config)
        logger.log_config_params()
    if not config["verbose"]:
        os.environ["WANDB_SILENT"] = "true"   
    else:
        print("params:", config)

    # TODO: send workers to ape
    
    parents = []
    for worker in workers:
        parent_conn, child_conn = Pipe()
        p = Process(target=run_workers_env_step, args=(worker, child_conn,))
        p.daemon = True
        parents.append(parent_conn)
        p.start()



    recording, visited_rooms, init_iteration, episode, episode_ext_reward = [], set([1]), 0, 0, 0
    if config["mode"] == "test" or config["mode"] == "train_from_chkpt":
        if config["model_name"] is not None:
            logger.log_dir = config["model_name"] 
        chkpt = logger.load_weights(config["model_name"])
        agent.set_from_checkpoint(chkpt)
        init_iteration = int(chkpt["iteration"])
        episode = chkpt["episode"]
    print("init: ", init_iteration)
    if config["mode"] == "test":
        pass
    
    else:
        print("---Pre_normalization started.---")
        states = []
        total_pre_normalization_steps = config["rollout_length"] * config["pre_normalization_steps"]
        actions = np.random.randint(0, config["n_actions"], (total_pre_normalization_steps, config["n_workers"]))
        for t in range(total_pre_normalization_steps):

            for worker_id, parent in enumerate(parents):
                parent.recv()  # Only collects next_states for normalization.

            for parent, a in zip(parents, actions[t]):
                parent.send(a)

            for parent in parents:
                s_, *_ = parent.recv()
                states.append(s_[-1, ...].reshape(1, 84, 84))

            if len(states) % (config["n_workers"] * config["rollout_length"]) == 0:
                agent.state_rms.update(np.stack(states))
                states = []
        print("---Pre_normalization is done.---")

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

        for iteration in tqdm(range(init_iteration, config["total_rollouts"] + 1), disable=not config["verbose"]):
            total_states = init_states
            total_actions = init_actions
            total_action_probs =init_action_probs
            total_int_rewards = init_int_rewards
            total_ext_rewards = init_ext_rewards
            total_dones = init_dones
            total_int_values = init_int_values
            total_ext_values = init_ext_values
            total_log_probs =init_log_probs
            next_states = init_next_states
            total_next_obs = init_next_obs
            total_next_states = np.zeros(rollout_base_shape + config["obs_shape"], dtype=np.uint8)

            logger.time_start()

            for t in range(config["rollout_length"]):
                for worker_id, parent in enumerate(parents):
                    total_states[worker_id, t] = parent.recv()

                total_actions[:, t], total_int_values[:, t], total_ext_values[:, t], total_log_probs[:, t], \
                total_action_probs[:, t] = agent.get_actions_and_values(total_states[:, t], batch=True)
                for parent, a in zip(parents, total_actions[:, t]):
                    parent.send(a)

                infos = []
                for worker_id, parent in enumerate(parents):
                    s_, r, d, info = parent.recv()
                    infos.append(info)
                    total_ext_rewards[worker_id, t] = r
                    total_dones[worker_id, t] = d
                    next_states[worker_id] = s_
                    total_next_obs[worker_id, t] = s_[-1, ...]

                    if worker_id == 0:
                        recording.append(s_[-1, ...])

                episode_ext_reward += total_ext_rewards[0, t]
                if total_dones[0, t]:
                    episode += 1
                    if "episode" in infos[0]:
                        visited_rooms = infos[0]["episode"]["visited_room"]
                        logger.log_episode(episode, episode_ext_reward, visited_rooms)
                    episode_ext_reward = 0


            logger.time_stop("env step time")
            logger.time_start()
            total_next_obs = np.concatenate(total_next_obs)
            total_actions = np.concatenate(total_actions)
            total_next_states = np.concatenate(total_next_states)

            total_int_rewards = agent.calculate_int_rewards(total_next_obs) # + total actions for APE
            _, next_int_values, next_ext_values, * \
                _ = agent.get_actions_and_values(next_states, batch=True)

            total_int_rewards = agent.normalize_int_rewards(total_int_rewards)
            
            n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0 

            train_args = {"states":concatenate(total_states),
                                        "actions":total_actions,
                                        "int_rewards":total_int_rewards,
                                        "ext_rewards":total_ext_rewards,
                                        "dones":total_dones,
                                        "int_values":total_int_values,
                                        "ext_values":total_ext_values,
                                        "log_probs":concatenate(total_log_probs),
                                        "next_int_values":next_int_values,
                                        "next_ext_values":next_ext_values,
                                        "total_next_obs":total_next_obs}

            training_logs = mp.spawn(agent.train, nprocs=n_gpus, args=train_args)

            # training_logs = agent.train(args)

            logger.time_stop("training time")

            logger.time_start()
            logger.log_iteration(iteration,
                                    training_logs,
                                    total_int_rewards[0].mean(),
                                    total_ext_rewards[0].mean(),
                                    total_action_probs[0].max(-1).mean(),
                                    )
            
            recording = np.stack(recording)
            logger.log_recording(recording)
            recording = []
            logger.time_stop("logging time")
            logger.time_start()
            if iteration % config["interval"] == 0 or iteration == config["total_rollouts"]:
                logger.save_params(episode, iteration)
                logger.save_recording_local(recording)
                logger.save_score_to_json()
                logger.time_stop()

                with open("Models/" + logger.log_dir + "/logger.obj", "wb") as file_pi:
                    pickle.dump(logger, file_pi)
            logger.time_stop("param saving time")


if __name__ == '__main__':
    #delete_files()
    config = get_params()

    # # run 1
    # config["env"] = "VentureNoFrameskip-v4"
    # config["total_rollouts"] = int(7)
    # config["algo"] = "RND"
    # config["verbose"] = True
    # config["interval"] = 1

    train_model(config)
    wandb.finish()


