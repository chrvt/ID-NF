# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 18:48:13 2022

@author: Horvat
"""
import os
from matplotlib import pyplot as plt
import numpy as np

import numpy as np
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings("ignore")    
         
sig2_0 = 1e-09
sig2_1 = 2.0
intrinsic_noise = 0.001**2

n_sigmas = 20
delta = np.log( (sig2_1 / sig2_0)**(1/(n_sigmas-1)) )

sigmas = np.zeros(n_sigmas) + sig2_0 
for k in range(n_sigmas-1): 
    sigmas[k+1] = sigmas[k] * np.exp(delta)

batch_size = 200
      
root_dir = r'path_to_data'
save_path = r'path_to_save'  

data_dims = [20,40,60,80,100, 120,140,160,200, 300 ,400] 
IDs =        [9,19,29,39,49,59,69,79,99, 149 ,199]       
#######################################
#######################################

d_estimates = np.zeros(len(data_dims))
d_estimates_std = np.zeros(len(data_dims))
count = -1

row_ours = ['ours']
row_lidl = ['LIDL']
row_twoNN = ['twoNN']

sig_start = 4
sig_end = 10

for datadim in data_dims:
    count += 1
    print('Sphere embedded in ',int(datadim))
    rootdir = os.path.join(root_dir,str(datadim))
    
    log_probs = np.zeros([n_sigmas,batch_size])
    
    #Gaussian noise: 
    KS_Gauss = []
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            # print('file=',file)
            for k in range(n_sigmas):
                if file == "log_probs_" + str(k) +".npy":  #"eig_values_" + str(k) +".npy":   #
                    log_probs_ = np.load(os.path.join(subdir,file) ) #[:,:,0]
                    log_probs[k,:] = log_probs_
    
    
    x = np.log(sigmas).reshape((-1, 1))  
    d_hat = np.zeros(batch_size)
    
    for sample in range(batch_size):
        y = log_probs[:,sample] 
        
        model = LinearRegression()
        model.fit(x[sig_start:sig_end], y[sig_start:sig_end])
        d_hat[sample] = model.coef_ 
        
    d_estimates[count] = datadim + 2*np.mean(d_hat)
    d_estimates_std[count] = np.std( datadim + 2*d_hat )
    
    print('--mean LIDL estimate ',datadim + 2*np.mean(d_hat) )
    print('--std LIDL estimate  ', np.std(datadim + 2*d_hat ))    


np.save(os.path_join(save_path,'d_LIDL_hat.npy'), d_estimates)
np.save(os.path_join(save_path,'d_LIDL_hat_std.npy'), d_estimates_std)