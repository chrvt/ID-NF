#!/bin/bash

#SBATCH --mail-user=<horvat@pyl.unibe.ch>
#SBATCH --mail-type=fail,end
#SBATCH --job-name="S(D/2)"
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

#python train.py  -c configs/train_flow_thin_spiral.config

### uncomment any D you want to train on, adapt sample simze (N_samples)

##D=20
python main_cluster.py --train   --N_samples 1000 --sampling --sig2_min 1e-09 --dataset d_sphere --latent_distribution uniform --noise_type gaussian --intrinsic_noise 0.0 --data_dim 20 --latent_dim 10 --hidden_dim 200 --n_hidden 5 --lr 0.1 --lr_decay 0.5 --lr_patience 2000 --sig2 0.01 --calculate_KS --n_gradient_steps 100000 --cuda 0 --seed 0

##D=40##
##python main_cluster.py --train --N_samples 100 --sampling --sig2_min 1e-09 --dataset d_sphere --latent_distribution uniform --noise_type gaussian --intrinsic_noise 0.0 --data_dim 40 --latent_dim 20 --hidden_dim 200 --n_hidden 5 --lr 0.1 --lr_decay 0.5 --lr_patience 2000 --sig2 0.01 --calculate_KS --n_gradient_steps 100000 --cuda 0 --seed 0

##D=60
##python main_cluster.py --train --sampling --sig2_min 1e-09 --dataset d_sphere --latent_distribution uniform --noise_type gaussian --intrinsic_noise 0.0 --data_dim 60 --latent_dim 30 --hidden_dim 240 --n_hidden 5 --lr 0.1 --lr_decay 0.5 --lr_patience 2000 --sig2 0.01 --calculate_KS --n_gradient_steps 100000 --cuda 0 --seed 0

##D=80
##python main_cluster.py --train --N_samples 100 --sampling --sig2_min 1e-09 --dataset d_sphere --latent_distribution uniform --noise_type gaussian --intrinsic_noise 0.0 --data_dim 80 --latent_dim 40 --hidden_dim 320 --n_hidden 5 --lr 0.1 --lr_decay 0.5 --lr_patience 2000 --sig2 0.01 --calculate_KS --n_gradient_steps 100000 --cuda 0 --seed 11

##D=100##
##python main_cluster.py --train --N_samples 100 --sampling --sig2_min 1e-09 --dataset d_sphere --latent_distribution uniform --noise_type gaussian --intrinsic_noise 0.0 --data_dim 100 --latent_dim 50 --hidden_dim 300 --n_hidden 5 --lr 0.1 --lr_decay 0.5 --lr_patience 2000 --sig2 0.01 --calculate_KS --n_gradient_steps 100000 --cuda 0 --seed 111

##D=120
##python main_cluster.py --train --sampling --sig2_min 1e-09 --dataset d_sphere --latent_distribution uniform --noise_type gaussian --intrinsic_noise 0.0 --data_dim 120 --latent_dim 60 --hidden_dim 240 --n_hidden 5 --lr 0.1 --lr_decay 0.5 --lr_patience 2000 --sig2 0.01 --calculate_KS --n_gradient_steps 100000 --cuda 0 --seed 0

##D=140
##python main_cluster.py --train --N_samples 1000 --sampling --sig2_min 1e-09 --dataset d_sphere --latent_distribution uniform --noise_type gaussian --intrinsic_noise 0.0 --data_dim 140 --latent_dim 70 --hidden_dim 280 --n_hidden 5 --lr 0.1 --lr_decay 0.5 --lr_patience 2000 --sig2 0.01 --calculate_KS --n_gradient_steps 100000 --cuda 0 --seed 0

##D=160
##python main_cluster.py --train --N_samples 100 --sampling --sig2_min 1e-09 --dataset d_sphere --latent_distribution uniform --noise_type gaussian --intrinsic_noise 0.0 --data_dim 160 --latent_dim 80 --hidden_dim 320 --n_hidden 5 --lr 0.1 --lr_decay 0.5 --lr_patience 2000 --sig2 0.01 --calculate_KS --n_gradient_steps 100000 --cuda 0 --seed 0

##D = 200##
##python main_cluster.py --train --N_samples 100 --sampling --sig2_min 1e-09 --dataset d_sphere --latent_distribution uniform --noise_type gaussian --intrinsic_noise 0.0 --data_dim 200 --latent_dim 100 --hidden_dim 400 --n_hidden 5 --lr 0.1 --lr_decay 0.5 --lr_patience 2000 --sig2 0.01 --calculate_KS --n_gradient_steps 100000 --cuda 0 --seed 0

##D = 300##
##python main_cluster.py --train --N_samples 1000 --sampling --sig2_min 1e-09 --dataset d_sphere --latent_distribution uniform --noise_type gaussian --intrinsic_noise 0.0 --data_dim 300 --latent_dim 150 --hidden_dim 600 --n_hidden 4 --lr 0.1 --lr_decay 0.5 --lr_patience 2000 --sig2 0.01 --calculate_KS --n_gradient_steps 100000 --cuda 0 --seed 0

##D = 400##
##python main_cluster.py --train --N_samples 1000 --sampling --N_epochs 200 --sig2_min 1e-09 --dataset d_sphere --latent_distribution uniform --noise_type uniform --intrinsic_noise 0.0 --data_dim 400 --latent_dim 200 --hidden_dim 800 --n_hidden 4 --lr 0.1 --lr_decay 0.5 --lr_patience 2000 --sig2 0.01 --calculate_KS --n_gradient_steps 100000 --cuda 0 --seed 2
