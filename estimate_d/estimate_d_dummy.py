# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 18:48:13 2022
    Dummy code for estimate ID once singular values are calculated
@author: Horvat
"""
from matplotlib import pyplot as plt
import numpy as np
import os
from utils_repo import ID_NF_estimator

#hyperparameters
n_sigmas = 20     # number N of trained NFs
datadim = 3      # data dimension D
batch_size = 10  # number K of samples where sing. values where estimated on
save_path = r'outputs'    # path where to save output files

# Define the sigma range used during training
# In the paper, we used an equidistant sigma range (in log-scale) from
# sig2_0 to sig2_1
sig2_0 = 1e-09
sig2_1 = 2.0

delta = np.log( (sig2_1 / sig2_0)**(1/(n_sigmas-1)) )
sigmas = np.zeros(n_sigmas) + sig2_0
for k in range(n_sigmas-1):
    sigmas[k+1] = sigmas[k] * np.exp(delta)

# load here your sing values of shapoe NxKxD
sing_values_batch = np.abs(np.random.randn(n_sigmas,batch_size,datadim))
# in the paper, we find that averaging the sing. values across all batches
# reduces noise in the estimate; however, if you are interested in a local estimator
# continue, this is not necessary

local_estimator = False
data_type = 'image'
# option for plotting sing. values as a functions of sig2
plot = True
if local_estimator:
    d_hat = np.zeros(batch_size)

    if plot:
        fig = plt.figure(figsize=(20,10))
        ax = fig.add_subplot(111)
        for d in range(datadim):
            sing_values_d = sing_values_batch[:,0,d]
            ax.plot(sigmas,sing_values_d) #,c=colors[n],label=labels[n]
        plt.yscale('log')#, nonposy='clip')
        plt.xscale('log')#, nonposx='clip')
        plt.savefig(os.path.join(save_path, 'sing_values_vs_sig2'+'.pdf'))

    for k in range(batch_size):
        sing_values = sing_values_batch[:,k,:]
        d_hat[k] = ID_NF_estimator(sing_values,sigmas,datadim,mode=data_type,latent_dim=2,plot=True,tag=str(k),save_path=save_path)

    print('--estimate mean ', d_hat.mean())

else:
    sing_values_mean = sing_values_batch.mean(axis=1)

    if plot:
        fig = plt.figure(figsize=(20,10))
        ax = fig.add_subplot(111)
        for d in range(datadim):
            sing_values_d = sing_values_mean[:,d]
            ax.plot(sigmas,sing_values_d) #,c=colors[n],label=labels[n]
        plt.yscale('log')#, nonposy='clip')
        plt.xscale('log')#, nonposx='clip')
        plt.savefig(os.path.join(save_path, 'sing_values_vs_sig2'+'.pdf'))

    # estimate ID based on ID_NF, see documentation of this function for details
    d_hat = ID_NF_estimator(sing_values_mean,sigmas,datadim,mode=data_type,latent_dim=2,plot=True,tag=str(datadim),save_path=save_path)
    print('--estimate ', d_hat)

np.save(os.path.join(save_path, 'd_hat.npy'),d_hat)
