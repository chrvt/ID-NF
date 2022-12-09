#!/bin/bash

#SBATCH --mail-user=<horvat@pyl.unibe.ch>
#SBATCH --mail-type=fail,end
#SBATCH --job-name="My_data"
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G  #32

#SBATCH --partition=gpu
#SBATCH --qos=job_gpu

#SBATCH --gres=gpu:gtx1080ti:1
#SBATCH --array=1-20 ##max number must align with args.n_sigmas

cd /storage/homefs/ch19g182/Python/ID-NF/estimate_d

nvcc --version
nvidia-smi

python my_vector_data_cluster.py --sig2_min 1e-09 --sig2_max 10 --n_sigmas 20 --dataset my_data --data_dim 3 --hidden_dim 210 --N_epochs 500 --batch_size 200 --ID_samples 100
