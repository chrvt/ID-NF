# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 18:48:13 2022

@author: Horvat
"""
import os
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings("ignore")    
       
# from tabulate import tabulate
from utils import estimate_d, NEWestimate_d 

sigmas = [1e-09, 5e-09, 1e-08, 5e-08, 1e-07, 5e-07,1e-06,5e-06,0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.25,0.5,1.0,2.0,3.0,4.0,6.0,8.0,10.0]
n_sigmas = len(sigmas)

manifolds =  ['sphere','torus','hyperboloid','thin_spiral','swiss_roll','spheroid','stiefel'] # ['stiefel'] #
latents = ['mixture','correlated', 'exponential','unimodal']

base_path = r'path_to_singular_values'
save_path = r'path_to_save'

batch_size = 200
datadim = 3
latent_dim = 2

############plotting
SMALL_SIZE = 32
MEDIUM_SIZE = 40
BIGGER_SIZE = 50

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title   
###################      

count = -1
x = np.log(sigmas).reshape((-1, 1))  


for manifold in manifolds:
    print('manifold ',manifold)
    if manifold == 'thin_spiral':
        datadim = 2
        latent_dim = 1       
    elif manifold == 'stiefel':
        datadim = 4
        latent_dim = 1  
    else:
        datadim = 3
        latent_dim = 2
    
    for latent_distribution in latents:
            
        print('-- latent dist ',latent_distribution)
        
        # log_probs = np.zeros([n_sigmas,batch_size])
        
        data_path = os.path.join(base_path,'singular_values_'+ str(manifold) + '_' + str(latent_distribution) + '.npy' )
        if Path(data_path).is_file():
            sing_values = np.load(data_path) #[:,:,0]
            
            sing_values_mean = sing_values.mean(axis=0)
            fig = plt.figure(figsize=(20,10))
            
            # import pdb; pdb.set_trace()
            
            ax = fig.add_subplot(111)
            for d in range(datadim):
                sing_values_d = sing_values_mean[:,d]
                ax.plot(sigmas,sing_values_d) 
           
            plt.yscale('log') #, nonposy='clip')
            plt.xscale('log') #, nonposx='clip')
            
            ax.set_xlabel(r'inflation noise variance $\sigma^2$')
            ax.set_ylabel(r'singular value $\lambda$')
            # plt.show()
            plt.savefig(os.path.join(save_path,manifold+'_'+latent_distribution+'_mean.pdf'))
    
            d_hat, d_std = estimate_d(sing_values,sigmas,batch_size,datadim,tag= manifold+'_'+latent_distribution + '_mean_onsets',mean=True,plot=True,save_path=r'D:\PROJECTS\estimate_intrinsic_dim_d\data\toy_examples\plots\supp_plots') #, mode='cluster')  
            
            print('----d hat', d_hat)
            print('----d std', d_std)
            np.save(os.path.join(save_path,'d_hat_mean_onsets'+str(manifold) + '_' + str(latent_distribution) +'.npy'), d_hat)

        else: 
            continue 