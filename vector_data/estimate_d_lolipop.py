

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 18:48:13 2022

@author: Horvat
"""
import os
from matplotlib import pyplot as plt
import numpy as np
import math
import matplotlib
import warnings
import sys

sys.path.append(r'D:\PROJECTS\estimate_intrinsic_dim_d\data\d_sphere')
from utils import estimate_d  

warnings.filterwarnings("ignore")    
############plotting
SMALL_SIZE = 18
MEDIUM_SIZE = 22
BIGGER_SIZE = 30

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title   
###################         
sig2_0 = 1e-09
sig2_1 = 2.0

n_start  = 0
n_sigmas = 20
delta = np.log( (sig2_1 / sig2_0)**(1/(n_sigmas-1)) )

sigmas = np.zeros(n_sigmas) + sig2_0 
for k in range(n_sigmas-1): 
    sigmas[k+1] = sigmas[k] * np.exp(delta)

data_dims = [2]  
d_hat_NF = np.zeros(len(data_dims)) 
d_std_NF = np.zeros(len(data_dims)) 

batch_size = 200 
datadim_count = -1

save_path = r'D:\PROJECTS\estimate_intrinsic_dim_d\data\lolipop'

for datadim in data_dims: 
    print('datadim ', datadim)
    datadim_count += 1       
    
    rootdir = os.path.join(r'D:\PROJECTS\estimate_intrinsic_dim_d\data\lolipop\uniform\gaussian',str(datadim))
    
    sing_values_batch = np.zeros([n_sigmas-n_start,batch_size,datadim])
    # sing_values_batch= np.zeros([n_sigmas-n_start,batch_size])
    
    #Gaussian noise: r'C:\Users\Horvat\Desktop\PhD\Projects\Toole-Box-ManifoldLearning\Python\MAF\normalizing_flows-master\results\paper_PCA_W_1_0\circle\sig2_MC_0.1sigma2_PCA_0.0s2_PCA_1.0_seed_0'
    KS_Gauss = []
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            # print('file=',file)
            for k in range(n_start,n_sigmas,1):
                if file == "sing_values_" + str(k) +".npy":  #"eig_values_" + str(k) +".npy":   #
                    S_x = np.load(os.path.join(subdir,file)) #[:,:,0]
                    # print('S_x ',S_x.shape)
                    # import pdb
                    # pdb.set_trace()
                    # mean = np.mean(S_x,axis=0)  # shape is dim
                    # std = np.std(S_x,axis=0)
                    
                    sing_values_batch[k-n_start,:,:] = S_x[0:batch_size,:]
                    # sing_values_std[k-n_start,:,:] = std
                # else: print('could not load file ',file)
    
    idx_stick = np.array([    1,   2,   4,   8,   9,  10,  11,  16,  17,  19,  21,  25,  26,
                             29,  32,  33,  36,  37,  38,  42,  46,  47,  50,  53,  55,  56,
                             57,  58,  59,  62,  63,  64,  68,  69,  71,  76,  83,  84,  87,
                             89,  90,  91,  92,  95,  96,  97, 104, 106, 108, 109, 110, 111,
                            113, 116, 117, 119, 120, 123, 125, 129, 130, 131, 134, 136, 137,
                            139, 140, 141, 142, 143, 145, 146, 147, 150, 153, 154, 156, 157,
                            159, 161, 163, 164, 167, 168, 169, 173, 175, 177, 178, 183, 185,
                            186, 189, 190, 191, 192, 194, 198])
    idx_disk = np.array([    0,   3,   5,   6,   7,  12,  13,  14,  15,  18,  20,  22,  23,
                            24,  27,  28,  30,  31,  34,  35,  39,  40,  41,  43,  44,  45,
                            48,  49,  51,  52,  54,  60,  61,  65,  66,  67,  70,  72,  73,
                            74,  75,  77,  78,  79,  80,  81,  82,  85,  86,  88,  93,  94,
                            98,  99, 100, 101, 102, 103, 105, 107, 112, 114, 115, 118, 121,
                           122, 124, 126, 127, 128, 132, 133, 135, 138, 144, 148, 149, 151,
                           152, 155, 158, 160, 162, 165, 166, 170, 171, 172, 174, 176, 179,
                           180, 181, 182, 184, 187, 188, 193, 195, 196, 197, 199])
    
    idx_disk = np.array([0])
    idx_stick = np.array([1])
    
    sing_values_batch =  sing_values_batch[:,idx_stick,:]
      
    sing_values_mean = sing_values_batch.mean(axis=1)
    
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111)    
    for d in range(datadim):
        sing_values_d = sing_values_mean[:,d]
        ax.plot(sigmas,sing_values_d) #,c=colors[n],label=labels[n]    
    plt.yscale('log')#, nonposy='clip')
    plt.xscale('log')#, nonposx='clip')       
        
    
    ax.set_title(r'lolipop: $\lambda^{i}(\sigma^2)$' ) #int(datadim) 
    ax.set_xlabel(r'inflation noise variance $\sigma^2$')
    ax.set_ylabel(r'singular value $\lambda$')
    plt.show()
    
    plt.savefig(os.path.join(save_path, str(datadim)+'.pdf'))    
    
    d_hat_NF[datadim_count], d_std_NF[datadim_count] = estimate_d(sing_values_batch,sigmas,batch_size,datadim,mean=True,mode='slope',plot=True,tag=str(datadim),save_path=save_path)
   
    print('--estimate ', d_hat_NF[datadim_count])   
    
np.save(os.path.join(save_path, 'd_hat.npy'),d_hat_NF)
np.save(os.path.join(save_path, 'd_hat_std.npy'),d_std_NF)    

