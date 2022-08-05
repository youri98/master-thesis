#!/usr/bin/bash
#SBATCH --partition=mcs.gpu.q
#SBATCH --nodes=1
#SBATCH --ntasks=56
#SBATCH --error=slurm-%j.err
#SBATCH --output=slurm-%j.out
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:4

module load cuda10.2/toolkit/10.2.89

python run_on_hpc.py --total_rollouts 30000 --record --algo RND --res 40 40 --mem_size 4
