from utils import stack_states, make_atari
import numpy as np


class Worker:
    def __init__(self, id, **config):
        self.id = id
        self.config = config
        self.env_name = self.config["env"]
        self.max_episode_steps = self.config["max_frames_per_episode"]
        self.state_shape = self.config["state_shape"]

        if "MountainCar" in self.config["env"]:
            self._stacked_states = np.zeros(self.state_shape, dtype=np.float32)
            self.env = make_atari(self.env_name, self.max_episode_steps, sticky_action=False, max_and_skip=False, montezuma_visited_room=False, add_random_state_to_info=False)
        else:
            self.env = make_atari(self.env_name, self.max_episode_steps)
            self._stacked_states = np.zeros(self.state_shape, dtype=np.uint8)

        self.reset()

    def __str__(self):
        return str(self.id)

    def render(self):
        self.env.render()

    def reset(self):
        state = self.env.reset()

        if "MountainCar" in self.config["env"]:
            self._stacked_states = state
            # self._stacked_states = stack_states(self._stacked_states, state, True, preprocess=False)
        else:
            self._stacked_states = stack_states(self._stacked_states, state, True)

    def step(self, conn):
        t = 1
        while True:
            conn.send(self._stacked_states)
            action = conn.recv()
            next_state, r, d, info = self.env.step(action)
            t += 1

            if t % self.max_episode_steps == 0:
                d = True

            if "MountainCar" in self.config["env"]:
                if t % self.max_episode_steps != 0 and d:
                    r = np.power(self.max_episode_steps / t, 2)
                    info["completion_time"] = t
                else:
                    r = 0
                    info["completion_time"] = 0

            if "DonkeyKong" in self.config["env"] and r == 100:
                r = 0

            if self.config["render"]:
                self.render()

            if "MountainCar" in self.config["env"]:
                self._stacked_states = next_state
                # self._stacked_states = stack_states(self._stacked_states, next_state, False, preprocess=False)
            else:
                self._stacked_states = stack_states(self._stacked_states, next_state, False)
                
            conn.send((self._stacked_states, r, d, info))
            # conn.send((self._stacked_states, np.sign(r), d, info))

            if d:
                self.reset()
                t = 1
