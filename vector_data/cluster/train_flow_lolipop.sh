#!/bin/bash

#SBATCH --mail-user=<horvat@pyl.unibe.ch>
#SBATCH --mail-type=fail,end
#SBATCH --job-name="Lolipop"
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

python main_cluster.py  --sampling --sig2_min 1e-09 --sig2_max 10 --dataset lolipop --latent_distribution uniform --noise_type gaussian --intrinsic_noise 0.0 --data_dim 3 --latent_dim 1 --hidden_dim 210 --n_hidden 3 --lr 0.1 --lr_decay 0.5 --lr_patience 2000 --sig2 0.01 --n_gradient_steps 100000 --cuda 0 --seed 0
