from config import get_params
import numpy as np
from numpy import concatenate
import gym
from tqdm import tqdm
from rnd import RND
from logger import Logger
from torch.multiprocessing import Process, Pipe
from runner import Worker
import torch
import wandb
import multiprocessing
import os


torch.autograd.set_detect_anomaly(True)


def run_workers_env_step(worker, conn):
    worker.step(conn)

def train_model(config, add_noisy_tv=False, **kwargs):
    print("STARTED")
    with open("key.txt", "r") as personal_key:
        if personal_key is not None:
            os.environ["WANDB_API_KEY"] = personal_key.read().strip()

    temp_env = gym.make(config["env"])

    if "Continuous" not in config["env"]:
        config.update({"n_actions": temp_env.action_space.n})
    else:
        config.update({"n_actions": 1})

    temp_env.close()
    config["n_workers"] = multiprocessing.cpu_count() #* torch.cuda.device_count() if torch.cuda.is_available() else multiprocessing.cpu_count()
    config.update({"batch_size": (config["rollout_length"] * config["n_workers"]) // config["n_mini_batch"]})
    config.update({"predictor_proportion": 32 / config["n_workers"]})
    workers = [Worker(i, **config) for i in range(config["n_workers"])] 


    agent = RND(**kwargs, **config)


    logger = Logger(agent, **config)
    logger.log_config_params()
    
    if not config["verbose"]:
        os.environ["WANDB_SILENT"] = "true"   
    else:
        print("params:", config)

    
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

        rollout_base_shape = config["n_workers"], config["rollout_length"]


        if "Continuous" in config["env"]:
            init_actions = np.zeros(rollout_base_shape + (config["n_actions"],), dtype=np.float32)
            init_action_probs = np.zeros(rollout_base_shape)
        else:
            init_actions = np.zeros(rollout_base_shape, dtype=np.uint8)
            init_action_probs = np.zeros(rollout_base_shape + (config["n_actions"],))

        init_int_rewards = np.zeros(rollout_base_shape)
        init_ext_rewards = np.zeros(rollout_base_shape)
        init_dones = np.zeros(rollout_base_shape, dtype=bool)
        init_int_values = np.zeros(rollout_base_shape, dtype=float)
        init_ext_values = np.zeros(rollout_base_shape, dtype=float)
        init_log_probs = np.zeros(rollout_base_shape)
        init_completion_time = np.zeros(rollout_base_shape, dtype=np.int16)

        if "MountainCar" in config["env"]:
            init_next_states = np.zeros((rollout_base_shape[0],) + config["state_shape"], dtype=np.float32)
            init_next_obs = np.zeros(rollout_base_shape + config["obs_shape"], dtype=np.float32)
            init_states = np.zeros(rollout_base_shape + config["state_shape"], dtype=np.float32)
        else:
            init_next_states = np.zeros((rollout_base_shape[0],) + config["state_shape"], dtype=np.uint8)
            init_next_obs = np.zeros(rollout_base_shape + config["obs_shape"], dtype=np.uint8)
            init_states = np.zeros(rollout_base_shape + config["state_shape"], dtype=np.uint8)

        recording = []
        
        cum_hits = 0 
        cum_ext_reward = 0
        cum_dones = 0

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
            total_completion_time = init_completion_time
            # print("iteration how many frames", total_states.shape)

            logger.time_start()
            hits = 0


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

                    if r !=0:
                        hits += 1
                        cum_hits += 1
                        cum_ext_reward += r

                    if "MountainCar" in config["env"]:
                        total_completion_time[worker_id, t] = info["completion_time"] 


                        
                    infos.append(info)
                    total_ext_rewards[worker_id, t] = r
                    total_dones[worker_id, t] = d
                    next_states[worker_id] = s_

                    if add_noisy_tv:
                        obs_ = noisy_tv(s_[-1, ...])
                    else:
                        obs_ = s_[-1, ...]

                    total_next_obs[worker_id, t] = obs_

                    if worker_id == 0:
                        recording.append(s_[-1, ...])

                episode_ext_reward += total_ext_rewards[0, t]
                
                # if any(np.ravel(total_dones)):
                #     print("heee")


                if "MountainCar" not in config["env"]:
                    if total_dones[0, t]:
                        # print("episode: ", episode)
                        episode += 1
                        if "episode" in infos[0]:

                            visited_rooms = infos[0]["episode"]["visited_room"]
                            logger.log_episode(iteration, episode, episode_ext_reward, visited_rooms)
                            # if episode_ext_reward != 0:
                            #     print(episode_ext_reward)
                        episode_ext_reward = 0
                 



            logger.time_stop("env step time")
            logger.time_start()
            total_next_obs = np.concatenate(total_next_obs)
            total_actions = np.concatenate(total_actions)


            if not config["discard_intrinsic_reward"]:
                total_int_rewards = agent.calculate_int_rewards(total_next_obs)


            recording_int_rewards = total_int_rewards[0, ...]

            _, next_int_values, next_ext_values, * \
                _ = agent.get_actions_and_values(next_states, batch=True)

            total_int_rewards = agent.normalize_int_rewards(total_int_rewards)

            # agent.add_to_memory(states=concatenate(total_states),
            #                 actions=total_actions,
            #                 int_rewards=concatenate(total_int_rewards),
            #                 ext_rewards=concatenate(total_ext_rewards),
            #                 dones=concatenate(total_dones),
            #                 int_values=concatenate(total_int_values),
            #                 ext_values=concatenate(total_ext_values),
            #                 total_next_obs=total_next_obs)

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

            logger.time_stop("training time")
            n_frames = total_states.shape[0] * total_states.shape[1] * total_states.shape[2] * (iteration + 1)
            logger.time_start()

            if config["sampling_algo"] == "per-v2":
                age_percentage, _ = agent.memory.get_priority_age()
            else:
                age_percentage = None


            logger.log_iteration(iteration,
                                        n_frames,
                                        training_logs,
                                        total_int_rewards.mean(),
                                        total_ext_rewards.mean(),
                                        total_action_probs.max(-1).mean(),
                                        recording_int_rewards.mean(),
                                        age_percentage
                                        )

            if "MountainCar" in config["env"]:
                cum_dones += concatenate(total_dones).sum()

                mean_completion_time = np.true_divide(concatenate(total_completion_time).sum(),(concatenate(total_completion_time)!=0).sum())

                logger.log_mountaincar_states(iteration,
                                        concatenate(total_states), concatenate(total_dones).sum(), hits, cum_hits, cum_ext_reward, cum_dones, mean_completion_time)
            
                

            
            recording = np.stack(recording)

            # np.save(f"frame{iteration}.npy" , recording[23])


            

            # wandb.log({"image": wandb.Image(recording[23])}, step=iteration)
            if config["record"]:
                logger.log_recording(iteration, recording)
                # logger.save_recording_local(iteration, recording)

            logger.time_stop("logging time")
            logger.time_start()

            if iteration % config["interval"] == 0 or iteration == config["total_rollouts"]:
                logger.save_params(episode, iteration)
                # logger.save_score_to_json()
                logger.time_stop()

            logger.time_stop("param saving time")
            recording = []
            recording_int_rewards = []

def noisy_tv(obs):
    selection = obs[40:60, 70:]
    frame = np.load("empty_frame.npy")[40:60, 70:]

    # check if agent is at location
    if np.mean(np.abs(selection - frame)) > 0 and np.mean(np.abs(selection - frame)) < 10: # try to dont show it for following rooms
        # add noisy tv in topright of the screen
        obs[:20, 64:] = np.random.randint(0, 128, size=(20, 20)) 

    return obs


if __name__ == '__main__':
    config = get_params()
    # # run 1
    config["env"] = "ALE/DonkeyKong-v5"

    # config["env"] = "MountainCar-v0"
    # config.update({"state_shape": (2,), "obs_shape": (2,)})
    # config["total_rollouts"] = int(7)
    # config["algo"] = "RND"
    config["total_rollouts"] = 5000
    config["verbose"] = True
    config["sampling_algo"] = "uniform"
    config["mem_size"] = 1
    config["discard_intrinsic_reward"] = False
    # config["max_frames_per_episode"] = 2000
    config["record"] = True

    train_model(config, add_noisy_tv=False)
    wandb.finish()


