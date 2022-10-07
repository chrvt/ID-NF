# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 18:48:13 2022

@author: Horvat
"""
import os
from matplotlib import pyplot as plt
import numpy as np
       
from utils import estimate_d  

 ############plotting
SMALL_SIZE = 18
MEDIUM_SIZE = 22
BIGGER_SIZE = 30

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title   
###################    
  
batch_size = 200
       
sig2_0 = 1e-09
sig2_1 = 2.0
intrinsic_noise = 0.001**2

n_sigmas = 20
delta = np.log( (sig2_1 / sig2_0)**(1/(n_sigmas-1)) )

sigmas = np.zeros(n_sigmas) + sig2_0 
for k in range(n_sigmas-1): 
    sigmas[k+1] = sigmas[k] * np.exp(delta)

rootdir = r'...\data\sphere\uniform'  #sphere uniform
datadim = 3 #2
colors = ['red','orange','blue'] #['red','blue']#
labels = [r'normal',r'tangent 1',r'tangent 2'] #,
save_path = os.path.join(r'path_to_save' ) #',str('example_sphere')+'.pdf')

sing_values_mean = np.zeros([n_sigmas,datadim])
sing_values_std = np.zeros([n_sigmas,datadim])
sing_values_batch = np.zeros([n_sigmas,batch_size,datadim])

#Gaussian noise: 
KS_Gauss = []
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        # print('file=',file)
        for k in range(n_sigmas):
            if file == "sing_values_" + str(k) +".npy": 
                S_x = np.load(os.path.join(subdir,file) )
                mean = np.mean(S_x,axis=0)  # shape is dim
                std = np.std(S_x,axis=0)
                
                sing_values_batch[k,:,:] = S_x
                
                sing_values_mean[k,:] = mean
                sing_values_std[k,:] = std
  
def plt_errorbar(xplt,yplt,yerr,label=None,lw=2,c='k',marker='o',alpha=0.3,ls=None,log_scale=False):
    ax.plot(xplt,yplt,lw=lw,c=c,marker=marker,ls=ls,label=label)
    if log_scale:
        #ax.fill_between(xplt,yplt-0.43*(yerr/yplt),yplt+0.43*(yerr/yplt),color=c,alpha=alpha)
        ax.fill_between(xplt,yplt*yerr,yplt/yerr,color=c,alpha=alpha)
    else:
        ax.fill_between(xplt,yplt-yerr,yplt+yerr,color=c,alpha=alpha)


## singular values vs sigmas
fig = plt.figure(figsize=(10,10))
plt.rc('legend', fontsize=18)

ax = fig.add_subplot(111)
for n in range(datadim):
    mean = sing_values_mean[:,n]
    std = sing_values_std[:,n]
    plt_errorbar(sigmas,mean,std*0.4,c=colors[n],label=labels[n],log_scale=False)

ax.legend(loc='center left', bbox_to_anchor=(0,0.8))
ax.vlines(intrinsic_noise, 0, 490, colors='black',linestyles='dashed')
ax.text(intrinsic_noise, 490, 'intrinsic noise')

ax.vlines(1, 0, 500, colors='black',linestyles='dashed')
ax.text(0.4, 510, 'radius')

ax.set_xlabel(r'inflation noise variance $\sigma^2$')
ax.set_ylabel(r'singular value $\lambda$')

plt.yscale('log')#, nonposy='clip')
plt.xscale('log')#, nonposx='clip') 

plt.savefig(os.path.join(save_path,str('example_sphere_lambda_vs_sigma.pdf') ) )
    

from scipy.optimize import curve_fit

def myExpFunc(x, alpha,  gamma):
    return alpha * x + gamma #1 / (gamma**2 + x**2)**(alpha/2) #a * np.power(x, b)

slopes = np.zeros(datadim)

for d in range(datadim):
    mean_d = (sing_values_mean[:,d])
    popt_d, pcov_d = curve_fit(myExpFunc, np.log(sigmas), np.log(mean_d))
    print(popt_d)
    slopes[d] = popt_d[0]

sort_slopes = np.sort(slopes)
dist = np.zeros(datadim)

d_hat, d_std = estimate_d(sing_values_batch,sigmas,batch_size,datadim,mode='slope',plot=True, mean=True,tag='sphere',latent_dim=1,save_path=save_path)
