import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
import os
import datetime
import glob
from collections import deque
import wandb
import json
import cv2

class Logger:
    def __init__(self, brain, **config):
        self.config = config
        self.brain = brain
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

        wandb.init(project="RND", entity="youri")
        wandb.config = self.config

        if not self.config["do_test"] and self.config["train_from_scratch"]:
            self.create_model_folder()
            self.log_hyperparams()

        self.exp_avg = lambda x, y: 0.9 * x + 0.1 * y if (y != 0).all() else y

    def create_model_folder(self):
        if not os.path.exists("Models"):
            os.mkdir("Models")
        os.mkdir("Models/" + self.log_dir)

    def log_hyperparams(self):
        with open("Models/" + self.log_dir + '/config.txt', 'w') as writer:
            writer.write(json.dumps(self.config))

    def on(self):
        self.start_time = time.time()

    def off(self):
        self.duration = time.time() - self.start_time

    def log_recording(self, recording, fps=60, save_local=True):
        wandb.log({"video": wandb.Video(np.array(recording), fps=fps, format='gif')})
        
        if save_local:
            frame_size = (self.config["obs_shape"][2], self.config["obs_shape"][1])

            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 

            out = cv2.VideoWriter("Models/" + self.log_dir + "/recording.mp4", fourcc, fps, frame_size)

            for image in recording:
                #image = np.pad(image, ((height_pad, height_pad), (width_pad,width_pad)))
                #image = np.mean(image, axis=0)
                image = image[0]
                image = image.astype(np.uint8)
                image = np.expand_dims(image, axis=2)
                image = np.tile(image, (1,1,3))

                out.write(image)

            out.release()

    def log_iteration(self, *args):
        iteration, pg_losses, ext_value_losses, int_value_losses, rnd_losses, entropies, int_reward, ext_reward, action_prob = args

        # self.running_act_prob = self.exp_avg(self.running_act_prob, action_prob)
        # self.running_int_reward = self.exp_avg(self.running_int_reward, int_reward)
        # self.running_training_logs = self.exp_avg(self.running_training_logs, np.array(training_logs))

        if iteration % (self.config["interval"] // 3) == 0:
            self.save_params(self.episode, iteration)

        # with SummaryWriter("Logs/" + self.log_dir) as writer:
        #     writer.add_scalar("Episode Ext Reward", self.episode_ext_reward, self.episode)
        #     writer.add_scalar("Running Episode Ext Reward", self.running_ext_reward, self.episode)
        #     writer.add_scalar("Visited rooms", len(list(self.visited_rooms)), self.episode)
        #     writer.add_scalar("Running last 10 Ext Reward", self.running_last_10_ext_r, self.episode)
        #     writer.add_scalar("Max Episode Ext Reward", self.max_episode_reward, self.episode)

        #     writer.add_scalar("Running Action Probability", self.running_act_prob, iteration)
        #     writer.add_scalar("Running Intrinsic Reward", self.running_int_reward, iteration)
        #     writer.add_scalar("Running PG Loss", self.running_training_logs[0], iteration)
        #     writer.add_scalar("Running Ext Value Loss", self.running_training_logs[1], iteration)
        #     writer.add_scalar("Running Int Value Loss", self.running_training_logs[2], iteration)
        #     writer.add_scalar("Running RND Loss", self.running_training_logs[3], iteration)
        #     writer.add_scalar("Running Entropy", self.running_training_logs[4], iteration)
        #     writer.add_scalar("Running Intrinsic Explained variance", self.running_training_logs[5], iteration)
        #     writer.add_scalar("Running Extrinsic Explained variance",
        #                       self.running_training_logs[6], iteration)

        wandb.log({
            "Visited rooms": len(list(self.visited_rooms)),
            "Action Probability": action_prob,
            "Intrinsic Reward": int_reward,
            "Entrinsic Reward": ext_reward,
            "PG Loss": pg_losses,
            "Ext Value Loss": ext_value_losses,
            "Int Value Loss": int_value_losses,
            "RND Loss": rnd_losses,
            "Entropy": entropies,
            # "Intrinsic Explained variance": self.running_training_logs[5],
            # "Extrinsic Explained variance": self.running_training_logs[6],
        })



        # if iteration % self.config["interval"] == 0:
        #     print("Iter:{}| "
        #           "EP:{}| "
        #           "EP_Reward:{}| "
        #           "EP_Running_Reward:{:.3f}| "
        #           "Visited_rooms:{}| "
        #           "Iter_Duration:{:.3f}| "
        #           "Time:{} "
        #           .format(iteration,
        #                   self.episode,
        #                   self.episode_ext_reward,
        #                   self.running_ext_reward,
        #                   self.visited_rooms,
        #                   self.duration,
        #                   datetime.datetime.now().strftime("%H:%M:%S"),
        #                   ))

    # def log_episode(self, *args):
    #     self.episode, self.episode_ext_reward, self.visited_rooms = args

    #     self.max_episode_reward = max(self.max_episode_reward, self.episode_ext_reward)

    #     self.running_ext_reward = self.exp_avg(self.running_ext_reward, self.episode_ext_reward)

    #     self.last_10_ep_rewards.append(self.episode_ext_reward)
    #     if len(self.last_10_ep_rewards) == self.moving_avg_window:
    #         self.running_last_10_ext_r = np.convolve(self.last_10_ep_rewards, self.moving_weights, 'valid')

    def save_params(self, episode, iteration):
        torch.save({"current_policy_state_dict": self.brain.current_policy.state_dict(),
                    "predictor_model_state_dict": self.brain.predictor_model.state_dict(),
                    "target_model_state_dict": self.brain.target_model.state_dict(),
                    "optimizer_state_dict": self.brain.optimizer.state_dict(),
                    "state_rms_mean": self.brain.state_rms.mean,
                    "state_rms_var": self.brain.state_rms.var,
                    "state_rms_count": self.brain.state_rms.count,
                    "int_reward_rms_mean": self.brain.int_reward_rms.mean,
                    "int_reward_rms_var": self.brain.int_reward_rms.var,
                    "int_reward_rms_count": self.brain.int_reward_rms.count,
                    "iteration": iteration,
                    "episode": episode,
                    "running_reward": self.running_ext_reward,
                    "visited_rooms": self.visited_rooms
                    },
                   "Models/" + self.log_dir + "/params.pth")

    def load_weights(self):
        model_dir = glob.glob("Models/*")
        model_dir.sort()
        checkpoint = torch.load(model_dir[-1] + "/params.pth")
        self.log_dir = model_dir[-1].split(os.sep)[-1]
        return checkpoint
