from utils import *

class Worker:
    def __init__(self, id, **config):
        self.id = id
        self.config = config
        self.env_name = self.config["env_name"]
        self.max_episode_steps = self.config["max_frames_per_episode"]
        self.state_shape = self.config["state_shape"]
        self.env = make_atari(self.env_name, self.max_episode_steps)
        self.state = np.zeros(self.state_shape, dtype=np.uint8)
        self.reset()

    def __str__(self):
        return str(self.id)

    def render(self):
        self.env.render()

    def reset(self):
        obs = self.env.reset()
        self.state = preprocessing(obs)

    def env_step(self, conn):
        t = 1
        while True:
            conn.send(self.state)
            action = conn.recv()
            next_obs, r, d, info = self.env.step(action)
            t += 1
            if t % self.max_episode_steps == 0:
                d = True
            if self.config["render"]:
                self.render()
            self.state = preprocessing(next_obs)
            conn.send((self.state, np.sign(r), d, info))
            if d:
                self.reset()
                t = 1
