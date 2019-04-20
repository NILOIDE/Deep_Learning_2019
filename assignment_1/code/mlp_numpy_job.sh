#!/bin/bash
#SBATCH -t 0:05:00
#SBATCH -n 1
#SBATCH -p gpu_shared_course
module load python/3.5.3
python3 train_mlp_numpy.py > mlp_numpy_job.txt

