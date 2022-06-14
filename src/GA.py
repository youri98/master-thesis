
import numpy as np
import torch
import multiprocessing
import pygad
import numpy as np
import pygad.torchga
import pygad
import torch
from utils import *
import pygad.gacnn
from torch.distributions.categorical import Categorical
import globals

class PooledGA(pygad.GA):

    def cal_pop_fitness(self):
        agent, config, logger, pool = globals.agent, globals.config, globals.logger, globals.pool
        print("calculating population fitness...")
        if not hasattr(self, "frames"):
            self.frames = 0
        if not hasattr(self, "iteration"):
            self.iteration = 0
        else:
            self.iteration += 1
        

        # play in env
        logger.time_start()
        output = pool.map(GAfunctions.fitness_wrapper, self.population)
        logger.time_stop("Env Time")

        total_ext_rewards, episode_logs, observations, int_values, ext_values, next_int_values, next_ext_values, dones = zip(*output)
        indices = [len(obs) for obs in observations]
        indices.insert(0, 0)
        indices = np.cumsum(indices)

        intervals = list(pairwise(indices))

        logger.time_start()
        total_obs = np.concatenate(observations)
        total_int_reward = agent.calculate_int_rewards(total_obs)
        total_int_reward = tuple(total_int_reward[interval[0]:interval[1]] for interval in intervals) # reshape to seperate individuals
        total_int_reward = agent.normalize_int_rewards(total_int_reward)
        int_reward = tuple(np.sum(indiv_int_reward) for indiv_int_reward in total_int_reward)
        logger.time_stop("Calc Int Reward Time")

        self.frames += indices[-1] * config["state_shape"][0] # 4 stacked frames

        # would gae work as critic will be made worse by agent, so make critic los van de rest en train dit apart
        # advs = [agent.get_adv(total_int_reward[idx], total_ext_rewards[idx], int_values[idx], ext_values[idx], next_int_values[idx], next_ext_values[idx], dones[idx]) for idx in range(config["n_workers"])]
        # pop_fitness = np.array([np.sum(adv) for adv in advs])
        ext_rewards = np.array([np.sum(indiv_int_reward) for indiv_int_reward in total_ext_rewards])
        pop_fitness = list(map(lambda r_e, r_i: config["ext_adv_coeff"]*r_e + config["int_adv_coeff"]*r_i, ext_rewards, int_reward))
        pop_fitness = np.array(pop_fitness)

        # do something with this
        episode_logs = list(filter(None, episode_logs))

        # train rnd
        logger.time_start()
        rnd_loss = agent.train_rnd(total_obs)
        logger.time_stop("RND Train Time")



        if self.iteration != 0: # ignore initial call
        
            logger.log_iteration(self.iteration, self.frames, np.mean(int_reward), np.mean(ext_rewards), np.mean(pop_fitness), rnd_loss)
            best_idx = np.argmax(pop_fitness)
            best_recording = total_obs[intervals[best_idx][0]:intervals[best_idx][1]]
            logger.log_recording(best_recording, generation=self.iteration)

            min_len = min([interval[1] - interval[0] for interval in intervals])
            total_recording = np.max([obs[:min_len] for obs in observations], 0)
            total_recording = np.pad(total_recording, ((0,0), (0,2), (0,0), (0,0)), 'constant', constant_values=0) // 2
            red_recording = np.tile(best_recording, (1, 3, 1, 1))
            total_recording = np.max((red_recording[:min_len], total_recording), 0).astype(np.uint8)

            
            logger.log_recording(total_recording, generation=self.iteration, name="All")

        print(f"pop fitness: {pop_fitness}, frames {self.frames}, rnd_loss: {rnd_loss}")
        return pop_fitness

class GAfunctions():
    
    @staticmethod
    def callback_generation(ga_instance):
        print("on_generation()")

        print("Generation = {generation}".format(generation=ga_instance.generations_completed))
        print("Fitness    = {fitness}".format(fitness=ga_instance.best_solutions_fitness))
        ga_instance.best_solutions_fitness = []
    
    @staticmethod
    def on_fitness(ga_instance, population_fitness):
        print("on_fitness()")
    
    @staticmethod
    def on_parents(ga_instance, selected_parents):
        print("on_parents()")
        globals.logger.time_start()
    
    @staticmethod
    def on_crossover(ga_instance, offspring_crossover):
        print("on_crossover()")
        globals.logger.time_stop("Crossover Time")
        globals.logger.time_start()

    @staticmethod
    def on_mutation(ga_instance, offspring_mutation):
        print("on_mutation()")
        globals.logger.time_stop("Mutation Time")

    @staticmethod   
    def fitness_wrapper(solution):
        return GAfunctions.fitness_func(solution, 0)

    @staticmethod
    def fitness_func(solution, sol_idx):
        envs, config, agent = globals.envs, globals.config, globals.agent

        current_pool_id = multiprocessing.current_process()._identity[0] - 2 # dont get why its 2 tm 9
        policy_model_weights_dict = pygad.torchga.model_weights_as_dict(model=agent.current_policy, weights_vector=solution)
        agent.current_policy.load_state_dict(policy_model_weights_dict)

        # initialize env
        episode_ext_reward, total_obs, done, t = [], [], False, 1
        total_int_values, total_ext_values, total_next_int_values, total_next_ext_values, total_dones = [], [], [], [], []
        
        state_shape = config["state_shape"]
        env = envs[current_pool_id] 
        state = env.reset() # firtst observation

        _stacked_states = np.zeros(state_shape, dtype=np.uint8) # stacking 4 observations
        _stacked_states = stack_states(_stacked_states, state, True)
        
        # rollout length / until dead
        while t <= config["max_frames_per_episode"] and not done and t <= 700:
            state = torch.from_numpy(_stacked_states)#.to(agent.device)
            
            with torch.no_grad():
                int_value, ext_value, action_prob = agent.current_policy(torch.unsqueeze(state.type(torch.float), 0))
                dist = Categorical(action_prob)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            # action, int_value, ext_value, log_prob, action_prob = agent.get_actions_and_values(_stacked_states, batch=True)
            next_state, r, done, info = env.step(action)

            if t % config["max_frames_per_episode"] == 0:
                done = True


            _stacked_states = stack_states(_stacked_states, next_state, False)
            next_obs = _stacked_states[-1, ...]

            next_state = torch.from_numpy(_stacked_states)#.to(agent.device)

            with torch.no_grad():
                next_int_value, next_ext_value, _= agent.current_policy(torch.unsqueeze(next_state.type(torch.float), 0))

            if "episode" in info and current_pool_id == 0 and done:
                visited_rooms = info["episode"]["visited_room"]
                episode = info["episode"]

                episode_logs = {"Ep Visited Rooms": list(visited_rooms), "Episode Ext Reward": sum(episode_ext_reward)}

            t += 1

            episode_ext_reward.append(r)
            total_obs.append(next_obs)
            total_int_values.append(int_value)
            total_ext_values.append(ext_value)
            total_next_int_values.append(next_int_value)
            total_next_ext_values.append(next_ext_value)
            total_dones.append(done)

        total_obs = np.array(total_obs)
        total_obs = np.expand_dims(total_obs, 1)

        if "episode_logs" in locals():
            return episode_ext_reward, episode_logs, total_obs, total_int_values, total_ext_values, total_next_int_values, total_next_ext_values, total_dones
        else:
            return episode_ext_reward, None, total_obs, total_int_values, total_ext_values, total_next_int_values, total_next_ext_values, total_dones


