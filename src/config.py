import argparse
import multiprocessing

def get_params():
    parser = argparse.ArgumentParser(
        description="Variable parameters based on the configuration of the machine or user's choice")

    parser.add_argument("--n_workers", default=2, type=int, help="Number of parallel environments.")
    parser.add_argument("--interval", default=100, type=int,
                        help="The interval specifies how often different parameters should be saved and printed,"
                             " counted by iterations.")
    parser.add_argument("--algo", default="RND", help="Use RND or APE", choices=["RND", "APE"])
    parser.add_argument("--record", default=False, action="store_true", help="Save recording in model folder")
    parser.add_argument("--total_rollouts", default=int(1e1), type=int, help="Total number of rollouts per environment")
    parser.add_argument("--env", default="MontezumaRevengeNoFrameskip-v4", help="Name of the environment to run the model on")
    parser.add_argument("--mode", default="train_from_scratch", help="whether to train or test", choices=["train_from_scratch", "train_from_chkpt", "test"])
    parser.add_argument("--verbose", default=False, action="store_true")
    parser.add_argument("--model_name", help="name of model to resume training or to test")
    parser.add_argument("--multiple_gpus", default=False, action="store_true", help="whether the run is on multiple gpus")
    parser.add_argument("--n_individuals_per_gen", default=multiprocessing.cpu_count(),  help="number of individuals to simulate per generation")


    # parser.add_argument("--do_test", action="store_true",
    #                     help="The flag determines whether to train the agent or play with it.")
    parser.add_argument("--render", action="store_true",
                         help="The flag determines whether to render each agent or not.")
    # parser.add_argument("--train_from_scratch", action="store_false",
    #                     help="The flag determines whether to train from scratch or continue previous tries.")

    parser_params = parser.parse_args()

    """ 
     Parameters based on the "Exploration By Random Network Distillation" paper.
     https://arxiv.org/abs/1810.12894    
    """
    # MontezumaRevengeNoFrameskip-v4
    # region default parameters
    default_params = {
        "state_shape": (4, 84, 84),
        "obs_shape": (1, 84, 84),
        "max_frames_per_episode": 4500,  # 4500 * 4 = 18K :D
        "rollout_length": 128,
        "n_epochs": 4,
        "n_mini_batch": 4,
        "lr": 1e-4,
        "ext_gamma": 0.999,
        "int_gamma": 0.99,
        "lambda": 0.95,
        "ext_adv_coeff": 2,
        "int_adv_coeff": 1,
        "ent_coeff": 0.001,
        "clip_range": 0.1,
        "pre_normalization_steps": 5,
        "num_generations": 50,  # Number of generations.
        # Number of solutions to be selected as parents in the mating pool.
        "num_parents_mating": 5,
        "parent_selection_type": "sss",  # Type of parent selection.
        # Type of the crossover operator.
        "crossover_type": "single_point",
        # Type of the mutation operator.
        "mutation_type": "random",
        # Percentage of genes to mutate. This parameter has no action if the parameter mutation_num_genes exists.
        "mutation_percent_genes": 20,
        # Number of parents to keep in the next population. -1 means keep all parents and 0 means keep nothing.
        "keep_parents": 2,
    }

    # endregion
    total_params = {**vars(parser_params), **default_params}
    return total_params
