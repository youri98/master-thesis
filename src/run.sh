#!/usr/bin/bash
#SBATCH --partition=mcs.gpu.q
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --error=slurm-%j.err
#SBATCH --output=slurm-%j.out
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

module load cuda10.2/toolkit/10.2.89

python main.py