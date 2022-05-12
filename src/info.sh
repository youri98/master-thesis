#!/usr/bin/bash
#SBATCH --partition=mcs.default.q
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --error=slurm-%j.err
#SBATCH --output=slurm-%j.out
#SBATCH --time=2:00:00
#SBATCH --gres
#SBATCH --constraint=v100

module load openmpi
module load cuda10.2/toolkit/10.2.89

mpirun ./getinfo.py