
workers = [Worker(i, **config) for i in range(config["n_workers"])] 
parents = []


for worker in workers:
    parent_conn, child_conn = Pipe()
    p = Process(target=run_workers_env_step, args=(worker, child_conn,))
    p.daemon = True
    parents.append(parent_conn)
    p.start()


# while iteration

for worker_id, parent in enumerate(parents):
    total_states[worker_id, t] = parent.recv()


for parent, a in zip(parents, total_actions[:, t]):
    parent.send(a)

for worker_id, parent in enumerate(parents):
    s_, r, d, info = parent.recv()
    infos.append(info)
    total_ext_rewards[worker_id, t] = r
    total_dones[worker_id, t] = d
    next_states[worker_id] = s_
    total_next_obs[worker_id, t] = s_[-1, ...]

def run_workers_env_step(worker, conn):
    worker.step(conn)









class Worker:
    def __init__(self, id, **config):
        self.id = id
        self.config = config
        self.env_name = self.config["env"]
        self.max_episode_steps = self.config["max_frames_per_episode"]
        self.state_shape = self.config["state_shape"]
        self.env = make_atari(self.env_name, self.max_episode_steps)
        self._stacked_states = np.zeros(self.state_shape, dtype=np.uint8)
        self.reset()

    def __str__(self):
        return str(self.id)

    def render(self):
        self.env.render()

    def reset(self):
        state = self.env.reset()
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
            if self.config["render"]:
                self.render()
            self._stacked_states = stack_states(self._stacked_states, next_state, False)
            conn.send((self._stacked_states, np.sign(r), d, info))
            if d:
                self.reset()
                t = 1
