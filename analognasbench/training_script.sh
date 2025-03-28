# SLURM Script to allocate gpus and run trainings


#!/bin/bash
#SBATCH --partition=npl-2024 
#SBATCH --job-name=train1

#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8

#SBATCH --time=06:00:00
#SBATCH --output=__train_%j.out
#SBATCH --error=__error_%j.err

source ~/scratch/miniconda3x86/bin/activate
conda activate grafnas

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))

srun python training_script.py