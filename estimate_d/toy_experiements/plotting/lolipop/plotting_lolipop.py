#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 19:43:47 2022
    + plotting lolipop for paper
    + with estimated d for certain points
@author: chrvt
"""
import warnings
import numpy as np
from lolipop_simulator import LolipopSimulator
from matplotlib import pyplot as plt
import os

save_path = r'save_path'

warnings.filterwarnings("ignore")    
############plotting
SMALL_SIZE = 22
MEDIUM_SIZE = 24
BIGGER_SIZE = 30

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title   
###################   
N_smpls = 1000
n_start = 0
n_smpls = 30

lolipop = LolipopSimulator()

samples = lolipop.sample(N_smpls)

# d_hat_smpls = lolipop.sample(n_smpls)

d_hat_smpls = np.load('first_batch.npy')[n_start:n_smpls,:,0]
d_hat =  np.load('d_hat.npy')[n_start:n_smpls]    

fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(111)  
  
ax.scatter(samples[:,0],samples[:,1])
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')


ax.scatter(d_hat_smpls[:,0],d_hat_smpls[:,1],marker='X', color = 'black',s=100)

# d_hat = []
for k in range(n_smpls-n_start):
    print(k)
    dd = str(np.round(d_hat[k],decimals=1))
    print('--',dd)
    plt.text(d_hat_smpls[k,0],d_hat_smpls[k,1]-0.08,s=dd,fontsize = 'large')
    
    
plt.savefig(os.path.join(save_path, str('lolipop')+'.pdf')) 