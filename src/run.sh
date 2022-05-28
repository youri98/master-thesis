#!/usr/bin/bash
#SBATCH --partition=mcs.gpu.q
#SBATCH --nodes=1
#SBATCH --ntasks=56
#SBATCH --error=slurm-%j.err
#SBATCH --output=slurm-%j.out
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:4

module load python3
module load cuda10.2/toolkit/10.2.89
module load mpi

python run_on_hpc.py --total_rollouts 150 --record --algo RND