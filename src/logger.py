import time
import numpy as np
import torch
import os
import datetime
import glob
from collections import deque
import wandb
import json
import cv2


class Logger:
    def __init__(self, agent, **config):
        self.config = config
        self.agent = agent
        self.log_dir = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.start_time = 0
        self.duration = 0
        self.episode = 0
        self.episode_ext_reward = 0
        self.running_ext_reward = 0
        self.running_int_reward = 0
        self.running_act_prob = 0
        self.running_training_logs = 0
        self.visited_rooms = set([1])
        self.max_episode_reward = -np.inf
        self.moving_avg_window = 10
        self.moving_weights = np.repeat(
            1.0, self.moving_avg_window) / self.moving_avg_window
        self.last_10_ep_rewards = deque(maxlen=10)
        # It is not correct but does not matter.
        self.running_last_10_ext_r = 0
        self.best_score = 0
        self.timer = {}

        self.run_id = wandb.util.generate_id()
        wandb.init(project="RND", entity="youri",
                   id=self.run_id, resume="allow")
        wandb.config = self.config

        if self.config["mode"] == "train_from_scratch":
            self.create_model_folder()

        scoreskeys = ["Iteration", "N Frames", "Visited Rooms", "Action Probability", "Intrinsic Reward", "PG Loss", "Discriminator Loss",
                      "Ext Value Loss",  "Int Value Loss", "Advantage", "RND Loss", "Entrinsic Reward", "Entropy", "Recording"]
        self.scores = {k: [] for k in scoreskeys}

        #self.exp_avg = lambda x, y: 0.9 * x + 0.1 * y if (y != 0).all() else y

    def reboot(self):
        wandb.init(project="RND", entity="youri",
                   resume="allow", id=self.run_id)
        wandb.config = self.config

    def create_model_folder(self):
        if not os.path.exists("Models"):
            os.mkdir("Models")
        os.mkdir("Models/" + self.log_dir)

    def log_config_params(self):
        with open("Models/" + self.log_dir + '/config.json', 'w') as writer:
            writer.write(json.dumps(self.config))

    def time_start(self):
        self.start_time = time.time()

    def time_stop(self, kind="training time"):
        self.timer[kind] = self.timer[kind] + (time.time(
        ) - self.start_time) if kind in self.timer else (time.time() - self.start_time)

    def log_recording(self, recording, fps=60):
        if recording is not None:
            recording = np.expand_dims(recording, 1)
            wandb.log({"video": wandb.Video(
                np.array(recording), fps=fps, format='gif')})
        self.scores["Recording"].append(recording.tolist())

    def save_recording_local(self, recording, fps=60):
        if self.config["record"]:
            frame_size = (self.config["obs_shape"][2],
                          self.config["obs_shape"][1])

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            out = cv2.VideoWriter(
                "Models/" + self.log_dir + "/recording.mp4", fourcc, fps, frame_size, 0)

            for image in recording:
                #image = np.pad(image, ((height_pad, height_pad), (width_pad,width_pad)))
                #image = np.mean(image, axis=0)
                image = image[0]
                image = image.astype(np.uint8)
                image = np.expand_dims(image, axis=2)
                image = np.tile(image, (1, 1, 3))

                out.write(image)

            out.release()

    def log_episode(self, *args):
        self.episode, self.episode_ext_reward, self.visited_rooms = args
        self.max_episode_reward = max(
            self.max_episode_reward, self.episode_ext_reward)

    def save_score_to_json(self):
        with open("Models/" + self.log_dir + '/scores.json', 'w') as file:
            norm_scores = {k: (v if k in ["N Frames", "Discriminator Loss", "Visited Rooms", "Iteration", "Recording"] or np.linalg.norm(v) == 0 else (v/np.linalg.norm(v)).tolist()) for k,v in self.scores.items()}


            file.write(json.dumps(norm_scores))

    def log_iteration(self, *args):
        if self.config["algo"] == 'APE':
            iteration, n_frames, (pg_losses, ext_value_losses, int_value_losses, rnd_losses,
                        disc_losses, entropies, advs), int_reward, ext_reward, action_prob = args
        else:
            iteration, n_frames, (pg_losses, ext_value_losses, int_value_losses, rnd_losses, entropies), int_reward, ext_reward, action_prob = args
        # self.running_act_prob = self.exp_avg(self.running_act_prob, action_prob)
        # self.running_int_reward = self.exp_avg(self.running_int_reward, int_reward)
        # self.running_training_logs = self.exp_avg(self.running_training_logs, np.array(training_logs))

        params = {
            "Visited Rooms": len(list(self.visited_rooms)),
            "N Frames": n_frames,
            "Action Probability": action_prob,
            "Intrinsic Reward": int_reward.item(),
            "Entrinsic Reward": ext_reward.item(),
            "PG Loss": pg_losses,
            "Ext Value Loss": ext_value_losses,
            "Int Value Loss": int_value_losses,
            "RND Loss": rnd_losses,
            "Entropy": entropies,
            #"Advantage": advs.item(),
            # "Intrinsic Explained variance": self.running_training_logs[5],
            # "Extrinsic Explained variance": self.running_training_logs[6],
        }

        if self.config['algo'] == 'APE':
            params["Discriminator Loss"] = disc_losses

        self.scores['Iteration'].append(iteration)
        for k, v in params.items():
            # if isinstance(v, torch.Tensor):
            #     v = v.item()
            self.scores[k].append(v)

        params.update(self.timer)
        wandb.log(params)

    def save_params(self, episode, iteration):
        params = {"current_policy_state_dict": self.agent.current_policy.state_dict(),
                    "predictor_model_state_dict": self.agent.predictor_model.state_dict(),
                    "target_model_state_dict": self.agent.target_model.state_dict(),
                    "optimizer_state_dict": self.agent.optimizer.state_dict(),
                    "state_rms_mean": self.agent.state_rms.mean,
                    "state_rms_var": self.agent.state_rms.var,
                    "state_rms_count": self.agent.state_rms.count,
                    "int_reward_rms_mean": self.agent.int_reward_rms.mean,
                    "int_reward_rms_var": self.agent.int_reward_rms.var,
                    "int_reward_rms_count": self.agent.int_reward_rms.count,
                    "iteration": iteration,
                    "episode": episode,
                    "running_reward": self.running_ext_reward,
                    "visited_rooms": self.visited_rooms
                    }

        if self.config['algo'] == "APE":
            params["discriminator_model_state_dict"] = self.agent.discriminator.state_dict()

        torch.save(params, "Models/" + self.log_dir + "/params.pth")

    def load_weights(self, model_name=None):
        if model_name is not None:
            checkpoint = torch.load("Models/" + model_name + "/params.pth")
            self.log_dir = model_name
        else:
            model_dir = glob.glob("Models/*")
            model_dir.sort()
            checkpoint = torch.load(model_dir[-1] + "/params.pth")
            self.log_dir = model_dir[-1].split(os.sep)[-1]
        return checkpoint
