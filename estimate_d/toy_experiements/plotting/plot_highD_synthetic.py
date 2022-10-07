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

from tabulate import tabulate

headers = ['Model',r'$S(10)\subset \mathbb{R}^{20}$',r'$S(20)\subset \mathbb{R}^{40}$',r'$S(30)\subset \mathbb{R}^{60}$',r'$S(40)\subset \mathbb{R}^{80}$',r'$S(50)\subset \mathbb{R}^{100}$',r'$S(60)\subset \mathbb{R}^{120}$',r'$S(70)\subset \mathbb{R}^{140}$',r'$S(80)\subset \mathbb{R}^{160}$',r'$S(100)\subset \mathbb{R}^{200}$',r'$S(150)\subset \mathbb{R}^{300}$',r'$S(200)\subset \mathbb{R}^{400}$']
rows = [headers]

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


#######################################
### 10K################################
save_path = r'save_to_path' 
root_dir = r'...\d_sphere\uniform\gaussian\10k_samples'

d_twoNN = np.load(r'...\d_sphere\sphere_data\two_NN_10k.npy') #two_NN_1k
d_twoNN_std = np.load(r'...\d_sphere\sphere_data\two_NN_10k_std.npy')
                    
d_NF =      np.load(r'...\d_sphere\10_K_imgs\d_ID_NF_hat.npy')
d_NF_std =  np.load(r'...\d_sphere\10_K_imgs\d_ID_NF_std.npy')

d_LIDL =      np.load(r'...\d_sphere\10_K_imgs\d_LIDL_hat.npy')
d_LIDL_std =  np.load(r'...\d_sphere\10_K_imgs\d_LIDL_hat_std.npy')

img_name = str('high_dim_sphere_10k.pdf')
plot_title = r'$10^4$ samples'

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
    
    row_ours.append(str(np.round(d_NF[count],2) ) + r'$\pm $' + str( np.round(d_NF_std[count],2)) )
    row_lidl.append(str(np.round(d_LIDL[count],2)) + r'$\pm $' + str( np.round(d_LIDL_std[count],2)) )
    row_twoNN.append(str( np.round(d_twoNN[count],2)  ))
    # D = str(datadim)
    # row.append(D)
            
    # ID = str(latent_dim)
    # row.append(ID)  

rows.append(row_twoNN)
rows.append(row_lidl)
rows.append(row_ours)

print('Tabulate Table:')
print(tabulate(rows, headers='firstrow', tablefmt='latex_raw'))


############plotting
SMALL_SIZE = 22
MEDIUM_SIZE = 26
BIGGER_SIZE = 30

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title   
###################    


def plt_errorbar(xplt,yplt,yerr,label=None,lw=2,c='k',marker='o',alpha=0.3,ls=None,log_scale=False):
    ax.plot(xplt,yplt,lw=lw,c=c,marker=marker,ls=ls,label=label)
    if log_scale:
        #ax.fill_between(xplt,yplt-0.43*(yerr/yplt),yplt+0.43*(yerr/yplt),color=c,alpha=alpha)
        ax.fill_between(xplt,yplt*yerr,yplt/yerr,color=c,alpha=alpha)
    else:
        ax.fill_between(xplt,yplt-yerr,yplt+yerr,color=c,alpha=alpha)
        
        
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)

ax.plot(IDs,IDs , c='black',   label='true ID',linestyle = 'dashed') 
# ax.plot(IDs,d_NF          , label='ours' ) 
plt_errorbar(IDs,d_NF,d_NF_std,c='blue',label='ID-NF')
plt_errorbar(IDs,d_twoNN, d_twoNN_std,c='orange',label='twoNN')
# ax.plot(IDs,d_twoNN        , label='twoNN' ) 
         # 
plt_errorbar(IDs,d_estimates,d_estimates_std,c='red',label='LIDL')
ax.set_xlabel(r'$ID$')
ax.set_ylabel(r'estimate')
ax.legend(loc='upper left')
ax.set_title(plot_title)
# ax.plot(alphas,np.ones(len(alphas))*int(datadim/2))
import matplotlib
xint = range(IDs[0]+1, 200+1,20)
matplotlib.pyplot.xticks(xint)
yint = range(IDs[0]+1, 200+1,20)
matplotlib.pyplot.yticks(yint)

ax.set_xlabel(r'true intrinsic dimensionality')
ax.set_ylabel(r'estimated intrinsic dimensionality')

plt.savefig(os.path.join(save_path, img_name ))
