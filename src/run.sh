#!/usr/bin/bash
#SBATCH --partition=mcs.gpu.q
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --error=slurm-%j.err
#SBATCH --output=slurm-%j.out
#SBATCH --time=2:00:00
#SBATCH --gres=gpu

module load python3
module load cuda10.2/toolkit/10.2.89
module load mpi

python run_on_hpc.py --total_rollouts 50 --record --algo RND