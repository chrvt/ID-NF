#!/bin/bash

#SBATCH --mail-user=<horvat@pyl.unibe.ch>
#SBATCH --mail-type=fail,end
#SBATCH --job-name="styleGAN64ds"
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4 #4
#SBATCH --mem=32G  #32

#SBATCH --partition=gpu

#SBATCH --qos=job_gpu
#SBATCH --gres=gpu:gtx1080ti:1

##SBATCH --gres=gpu:rtx3090:1

#SBATCH --array=1-3   ##11-30 #  11-30

cd /storage/homefs/ch19g182/Python/estimate_d/images/experiments

module load CUDA

######## STYLE GAN 64d ####### --resume 1
python train.py --epochs 200 --resume 1  --noise_type_preprocess non --scale_factor 1 --noise_type gaussian  --modelname july --dataset gan64d --algorithm flow --outerlayers 20 --innerlayers 8 --levels 4 --linlayers 2 --linchannelfactor 2 --lineartransform lu --splinerange 10.0 --splinebins 11 --actnorm --batchsize 25 --lr 3.0e-4 --nllfactor 1 --uvl2reg 0.0 --clip 5.0 --validationsplit 0.0 --dropout 0.0 --dir /storage/homefs/ch19g182/Python/estimate_d
python evaluate_MNIST.py --OOD_dataset gan64d  --noise_type_preprocess non --scale_factor 1 --noise_type gaussian  --modelname july --dataset gan64d --algorithm flow --outerlayers 20 --innerlayers 8 --levels 4 --linlayers 2 --linchannelfactor 2 --lineartransform lu --splinerange 10.0 --splinebins 11 --actnorm --evaluate 100 --estimate_d --dropout 0.0 --dir /storage/homefs/ch19g182/Python/estimate_d
