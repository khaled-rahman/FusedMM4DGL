#!/bin/bash -l

#SBATCH -N 1
#SBATCH -p azad
#SBATCH -t 150:30:00
#SBATCH -J FusedMM4DGL
#SBATCH -o FusedMM4DGL.o%j
module unload gcc
module load gcc

srun -p azad -N 1 -n 1 -c 1 bash run_all.sh
