import time
import gym
import numpy as np
import pygad.torchga
import pygad
import torch
import torch.nn as nn
from multiprocessing import Pool
import pygad.cnn
import pygad.gacnn
# torch.set_grad_enabled(False)

def fitness_func(solution, sol_idx):
    global model, observation_space_size, env

    print(len(solution), sol_idx)

    model_weights_dict = pygad.torchga.model_weights_as_dict(model=model, weights_vector=solution)
    model.load_state_dict(model_weights_dict)

    # play game
    observation = env.reset()
    sum_reward = 0
    done = False

    # rollout length / until dead
    while (not done) and (sum_reward < 1000):
        # env.render()
        ob_tensor = torch.tensor(observation.copy(), dtype=torch.float)
        q_values = model(ob_tensor)
        action = np.argmax(q_values).numpy()
        observation_next, reward, done, info = env.step(action)
        observation = observation_next
        sum_reward += reward

    return sum_reward


def callback_generation(ga_instance):
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))
def fitness_wrapper(solution):
    return fitness_func(solution, 0)

    
class PooledGA(pygad.GA):

    def cal_pop_fitness(self):
        global pool

        pop_fitness = pool.map(fitness_wrapper, self.population)
        print(pop_fitness)
        pop_fitness = np.array(pop_fitness)
        return pop_fitness


env = gym.make("CartPole-v1")
observation_space_size = env.observation_space.shape[0]

action_space_size = env.action_space.n


model = nn.Sequential(
    nn.Linear(observation_space_size, 16),
    nn.ReLU(),
    nn.Linear(16, 16),
    nn.ReLU(),
    nn.Linear(16, action_space_size)
)
model.requires_grad_(False)
print(sum(p.numel() for p in model.parameters()))

torch_ga = pygad.torchga.TorchGA(model=model, num_solutions=10)

# Prepare the PyGAD parameters. Check the documentation for more information: https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#pygad-ga-class
num_generations = 50  # Number of generations.
num_parents_mating = 5  # Number of solutions to be selected as parents in the mating pool.
initial_population = torch_ga.population_weights  # Initial population of network weights
parent_selection_type = "sss"  # Type of parent selection.
crossover_type = "single_point"  # Type of the crossover operator.
mutation_type = "random"  # Type of the mutation operator.
mutation_percent_genes = 10  # Percentage of genes to mutate. This parameter has no action if the parameter mutation_num_genes exists.
keep_parents = -1  # Number of parents to keep in the next population. -1 means keep all parents and 0 means keep nothing.


ga_instance = PooledGA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_func,
                       parent_selection_type=parent_selection_type,
                       initial_population=initial_population,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       keep_parents=keep_parents,
                       on_generation=callback_generation,)

with Pool(processes=6) as pool:
    ga_instance.run()

# sample_shape = (1, 84, 84)
# num_classes = 4

# input_layer = pygad.cnn.Input2D(input_shape=sample_shape)
# conv_layer1 = pygad.cnn.Conv2D(num_filters=2,
#                                kernel_size=3,
#                                previous_layer=input_layer,
#                                activation_function=None)
# relu_layer1 = pygad.cnn.Sigmoid(previous_layer=conv_layer1)
# average_pooling_layer = pygad.cnn.AveragePooling2D(pool_size=2,
#                                                    previous_layer=relu_layer1,
#                                                    stride=2)

# conv_layer2 = pygad.cnn.Conv2D(num_filters=3,
#                                kernel_size=3,
#                                previous_layer=average_pooling_layer,
#                                activation_function=None)
# relu_layer2 = pygad.cnn.ReLU(previous_layer=conv_layer2)
# max_pooling_layer = pygad.cnn.MaxPooling2D(pool_size=2,
#                                            previous_layer=relu_layer2,
#                                            stride=2)

# conv_layer3 = pygad.cnn.Conv2D(num_filters=1,
#                                kernel_size=3,
#                                previous_layer=max_pooling_layer,
#                                activation_function=None)
# relu_layer3 = pygad.cnn.ReLU(previous_layer=conv_layer3)
# pooling_layer = pygad.cnn.AveragePooling2D(pool_size=2,
#                                            previous_layer=relu_layer3,
#                                            stride=2)

# flatten_layer = pygad.cnn.Flatten(previous_layer=pooling_layer)
# dense_layer1 = pygad.cnn.Dense(num_neurons=100,
#                                previous_layer=flatten_layer,
#                                activation_function="relu")
# dense_layer2 = pygad.cnn.Dense(num_neurons=num_classes,
#                                previous_layer=dense_layer1,
#                                activation_function="softmax")

# model = pygad.cnn.Model(last_layer=dense_layer2,
#                         epochs=1,
#                         learning_rate=0.01)
# GACNN_instance = pygad.gacnn.GACNN(model=model,
#                                    num_solutions=4)

# population_vectors = pygad.gacnn.population_as_vectors(population_networks=GACNN_instance.population_networks)
# print(sum([len(pop) for pop in population_vectors]))