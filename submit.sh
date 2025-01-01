#!/bin/bash
#SBATCH --account=bpms
#SBATCH --partition=gpu-h100 
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --job-name=brave
#SBATCH --mem=20GB

module purge
module load mamba cuda
conda activate prot-eng

python get_embedding.py 
