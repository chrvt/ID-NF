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

#sys.path.append(r'D:\PROJECTS\estimate_intrinsic_dim_d\data\d_sphere')

from utils import estimate_d, NEWestimate_d

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
# sig2_0 = 1e-09
# sig2_1 = 2.0
# intrinsic_noise = 0.001**2


label = 1
datadim = 3*64*64
datadim_ = 3*64*64

dataset = 'gan64d'
OODset = 'gan2d'
latent_dim = 64
modelname = 'july' #'august'
##############################################################################
############################################################################## /Users/chrvt/Documents/PhD/Projects/Estimate_d/data/OOD/estimate_d_stylegan_OOD.py
base_path = r'/Users/chrvt/Documents/PhD/Projects/Estimate_d/data/OOD'
save_path = os.path.join(base_path,dataset) # r'C:\Users\Horvat\iCloudDrive\draft\images'
tag = '_OOD_celeba'
sig2_0 = 255**1 * 1e-12 ##1e-12, 11.0 200.0
sig2_1 = 255**1 * 255.0 * 255 #2 , 50.0, 600
n_sigmas = 3
n_start = 0
sigmas1 = [1e-09, (0.68 * 255.0)**2, (100*255.0)**2 ]


 
rootdir = os.path.join(save_path,'ood_'+dataset+'-'+OODset+'_1-25')
#r'D:\PROJECTS\estimate_intrinsic_dim_d\data\stylegan\2d\style_gan_N3\OOD_celeba\1-25'

##############################################################################
batch_size = 25
sing_values_batch1 = np.zeros([n_sigmas-n_start,batch_size,datadim]) #+ 1
log_probs_1 = np.zeros([n_sigmas-n_start,batch_size])
# sing_values_std = np.zeros([n_sigmas-n_start,datadim,batch_size])

#Gaussian noise: r'C:\Users\Horvat\Desktop\PhD\Projects\Toole-Box-ManifoldLearning\Python\MAF\normalizing_flows-master\results\paper_PCA_W_1_0\circle\sig2_MC_0.1sigma2_PCA_0.0s2_PCA_1.0_seed_0'
KS_Gauss = []
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        # print('file=',file)
        for k in range(n_start,n_sigmas,1):
            # print('file ',file) flow_2_gan2d_july_run3_log_likelihoods_1
            if file == "flow_2_"+dataset+'_'+modelname+'_run'+ str(k+1) + "_singular_values.npy":  #"_eigen_values.npy": #"_singular_values.npy":  #
            #file == "flow_2_"+dataset+'_'+modelname+'_run'+ str(k+1) + "_singular_values.npy":  #"_eigen_values.npy": #"_singular_values.npy":  #
                # print('file found for run=',k)    
                S_x = np.abs(np.load(os.path.join(rootdir,file))) #[:,:,0]) #np.load(os.path.join(subdir,file) )
                
                #S_x_complex = np.load(os.path.join(subdir,file) )[:,:,1] 
                # print('S_x ',S_x.shape)
                # import pdb
                # pdb.set_trace()
                # mean = np.mean(S_x,axis=0)  # shape is dim
                # std = np.std(S_x,axis=0)
                
                sing_values_batch1[k-n_start,:,:] = S_x[:batch_size]
             
            elif file == "flow_2_"+dataset+'_'+modelname+'_run'+ str(k+1) + "_log_likelihoods_1.npy":  #"_eigen_values.npy": #"_singular_values.npy":  #
            #file == "flow_2_gan2d_august_run" + str(k+1) + "_log_likelihoods_1.npy":
                log_probs_ = np.load(os.path.join(rootdir,file) ) 
                log_probs_1[k-n_start,:] = log_probs_[:batch_size]
              
                
##############################################################################
############################################################################## 

##############################################################################
rootdir = os.path.join(save_path,'ood_'+dataset+'-'+OODset+'_25-50')
#rootdir = os.path.join(base_path,r'celeba/ood_celeba-gan2d_25-50')

batch_size = 25
sing_values_batch2 = np.zeros([n_sigmas-n_start,batch_size,datadim]) #+ 1
log_probs_2 = np.zeros([n_sigmas-n_start,batch_size])
# sing_values_std = np.zeros([n_sigmas-n_start,datadim,batch_size])

#Gaussian noise: r'C:\Users\Horvat\Desktop\PhD\Projects\Toole-Box-ManifoldLearning\Python\MAF\normalizing_flows-master\results\paper_PCA_W_1_0\circle\sig2_MC_0.1sigma2_PCA_0.0s2_PCA_1.0_seed_0'
KS_Gauss = []
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        # print('file=',file)
        for k in range(n_start,n_sigmas,1):
            # print('file ',file) flow_2_gan2d_july_run3_log_likelihoods_1
           # if file == "flow_2_gan64d_july_run" + str(k+1) + "_singular_values.npy":  #"_eigen_values.npy": #"_singular_values.npy":  #
            if file == "flow_2_"+dataset+'_'+modelname+'_run'+ str(k+1) + "_singular_values.npy":  #"_eigen_values.npy": #"_singular_values.npy":
                # print('file found for run=',k)
                S_x = np.abs(np.load(os.path.join(rootdir,file) ) ) #[:,:,0]) #np.load(os.path.join(subdir,file) )
                
                #S_x_complex = np.load(os.path.join(subdir,file) )[:,:,1] 
                # print('S_x ',S_x.shape)
                # import pdb
                # pdb.set_trace()
                # mean = np.mean(S_x,axis=0)  # shape is dim
                # std = np.std(S_x,axis=0)
                
                sing_values_batch2[k-n_start,:,:] = S_x[:batch_size]
             
            #elif file == "flow_2_"+dataset+'_'+modelname+'_run'+ str(k+1) + "_log_likelihoods_1.npy":  #"_eigen_values.npy": #"_singular_values.npy":  #
            #file == "flow_2_gan64d_july_run" + str(k+1) + "_log_likelihoods_1.npy":
            elif file == "flow_2_"+dataset+'_'+modelname+'_run'+ str(k+1) + "_log_likelihoods_1.npy":  #"_eigen_values.npy": #"_singular_values.npy":  #
                log_probs_ = np.load(os.path.join(rootdir,file) )
                log_probs_2[k-n_start,:] = log_probs_[:batch_size]

##############################################################################
############################################################################## 
#import pdb
#pdb.set_trace()
 
sing_values_batch = np.concatenate([sing_values_batch1,sing_values_batch2],axis=1) #,sing_values_batch_4])   [0:20,:]
log_probs = np.concatenate([log_probs_1,log_probs_2],axis=1)
np.save(os.path.join(save_path,'log_probs_'+OODset),log_probs)
    
sigmas_ = np.concatenate([sigmas1])
   
# sing_values_batch = np.concatenate([sing_values_batch_1,sing_values_batch_2,sing_values_batch_3,sing_values_batch_4])       
# sigmas_ = sigmas #np.concatenate([sigmas1,sigmas2,sigmas3,sigmas4])
    ######option cumulative
        
from scipy.optimize import curve_fit

def myExpFunc(x, alpha, beta):
    return alpha * x  + beta 

# def myExpFunc(x, alpha, beta, gamma):
#     return alpha * x**2 + beta * x + gamma  #a * np.power(x, b)

def scale(A):
    return A # (A-np.min(A))/(np.max(A) - np.min(A)) + 0.001

#import pdb; pdb.set_trace()
d_hat = np.zeros(50)
#d_hat, d_std = NEWestimate_d(sing_values_batch ,sigmas_,50,datadim,mode='cluster',plot=True, mean=True,tag=tag,latent_dim=64,save_path=save_path)#,save_path=r'D:\PROJECTS\estimate_intrinsic_dim_d\data\stylegan\2d')  
#d_hat[k] = d_hat_
#print('--d_hat mean',d_hat)   
#print('--d_hat mstd',d_std )

for k in range(50): 
    print('sample ',k)  #np.mean(sing_values_batch[, axis = 1)  #s
    sing_values_mean = sing_values_batch[:,k:k+1,:] #
    #sing_values_mean = sing_values_batch #np.mean(sing_values_batch, axis = 1) 
    #sing_values_mean = sing_values_batch[:,k:5,:]
    
    # for bb in range(1): #(batch_size
    #     fig = plt.figure(figsize=(20,10))
    #     ax = fig.add_subplot(111)
    #     # for d in range(datadim):
    #     #     # sing_values_d = sing_values_batch[:,bb,d]
    #     #     sing_values_d = sing_values_mean[:,d]
    #     #     ax = fig.add_subplot(111)
    #     #     ax.plot(sigmas_,sing_values_d) #,c=colors[n],label=labels[n]
        
    #     for d in range(64):   #1,100,1): #
    #         # sing_values_d = sing_values_batch[:,bb,d]
    #         sing_values_d = sing_values_mean[:,-d]
    #         if d in [1,2]:
    #             ax.plot(sigmas_,sing_values_d,c='red',linestyle=':',marker='x') #,c=colors[n],label=labels[n]
    #         else: ax.plot(sigmas_,sing_values_d)
            
    #     # for d in range(0,datadim,100):   #1,100,1): #
    #     #     # sing_values_d = sing_values_batch[:,bb,d]
    #     #     sing_values_d = sing_values_mean[:,d]
    #     #     # ax = fig.add_subplot(111)
    #     #     ax.plot(sigmas_,sing_values_d) #,c=colors[n],label=labels[n]
            
    #     ax.vlines( (0.68 * 255)**2, 0, 7, colors='black',linestyles='dashed')
        
    #     plt.yscale('log') #, nonposy='clip')
    #     plt.xscale('log') #, nonposx='clip')
        
    #     ax.set_title(r'Style Gan 2d: $\lambda^{i}(\sigma^2)$' ) #int(datadim) 
    #     ax.set_xlabel(r'$\sigma^2$')
    #     ax.set_ylabel(r'eigenvalues')
        
        #plt.savefig(os.path.join(save_path,str('stylegan64d_eigenvalues_mean_200epochs')+str(bb)+'.pdf') )
    
    #sing_values_batch = sing_values_batch[0:19,:,:]
    # sigmas_ = sigmas_[0:19]
          
    #import pdb; pdb.set_trace()                                                                                                                         
    d_hat_, d_std = estimate_d(sing_values_mean ,sigmas_,50,datadim,mode='cluster',plot=True, mean=True,tag=str(k),latent_dim=latent_dim,save_path=save_path)#,save_path=r'D:\PROJECTS\estimate_intrinsic_dim_d\data\stylegan\2d')  
    d_hat[k] = d_hat_


np.save(os.path.join(save_path,'d_hat_'+OODset+'.npy'),d_hat)

print('--d_hat mean',d_hat.mean())   
print('--d_hat mstd',d_hat.std() ) 




    #################################
    ######option histogram of slopes
    
    #getting sigma min and max
    # from scipy.optimize import curve_fit
    # def myExpFunc(x, a, b):
    #     return a * np.power(x, b)
    
    # slopes_highest = np.zeros([n_sigmas-2])
    # slopes_lowest = np.zeros([n_sigmas-2])
    
    # sigma_min_counter = np.zeros([n_sigmas-2])
    # sigma_max_counter = np.zeros([n_sigmas-2])
    
    
    # for k in range(n_sigmas-2):
    #     # idx_k = idx
    #     x_k = (sigmas[k:k+2])
    #     mean_highest = sing_values_mean[k:k+2,0]
    #     mean_lowest  = sing_values_mean[k:k+2,-1]
        
    #     # import pdb
    #     # pdb.set_trace()
    
    #     popt_k, pcov_k = curve_fit(myExpFunc, x_k, mean_highest, maxfev=5000)
    #     slopes_highest[k] = popt_k[1]
        
    #     popt_k, pcov_k = curve_fit(myExpFunc, x_k, mean_lowest, maxfev=5000)
    #     slopes_lowest[k] = popt_k[1]
    
        
    #     if slopes_highest[k] < -0.5 + delta and  slopes_highest[k] > -0.5 - delta:
    #         sigma_min_counter[k] = 1
            
    #     if slopes_lowest[k] < -0.5 + delta and  slopes_lowest[k] > -0.5 - delta:
    #         sigma_max_counter[k] = 1
    
    # ####
    # for k in range(n_sigmas-2):
    #     sigma_sum_min = np.sum(sigma_min_counter[k:k+2])
    #     if sigma_sum_min == 2:
    #         sig2_min = sigmas[k]   
    #         break 
    #     else:     
    #         sig2_min = sigmas[0]               
    
    
    # for k in range(n_sigmas-2):
    #     sigma_sum_max = np.sum(sigma_max_counter[k:k+2])
    #     if sigma_sum_max == 2:
    #         sig2_max = sigmas[k]   
    #         break 
    #     else:     
    #         sig2_max = sigmas[-1]               
    
    # # sig2_min = sigmas[0]  
    # # sig2_max = sigmas[-1]  
    
    # idx = np.where( (sigmas >= sig2_min ) & (sigmas <= sig2_max ) )[0]
    
    # if len(idx) > 2:
    #     slopes_ev = np.zeros([datadim, len(idx)-2])  #np.zeros([len(range(1,datadim -1)), len(idx)])
    # elif len(idx) == 2 or len(idx)  == 1:
    #     slopes_ev = np.zeros([datadim, 1])
        
        
    # for ev in range(datadim):
        
    #     if len(idx) > 2:
    #         for k in range(len(idx)-2):
    #             idx_k = idx[k:k+2]
    #             x_k = (sigmas[idx_k])
    #             mean_ev = sing_values_mean[idx_k,ev]
                
    #             popt_k, pcov_k = curve_fit(myExpFunc, x_k, mean_ev, maxfev=5000)
    #             slopes_ev[ev,k] = popt_k[1]
    #     elif len(idx) == 2:
    #         idx_k = idx
    #         x_k = (sigmas[idx_k])
    #         mean_ev = sing_values_mean[idx_k,ev]
            
    #         popt_k, pcov_k = curve_fit(myExpFunc, x_k, mean_ev, maxfev=5000)
    #         slopes_ev[ev,0] = popt_k[1]
    #     elif len(idx) == 1:
    #         idx_k = idx
    #         slopes_ev[ev,0] = sing_values_mean[idx_k,ev]
            
        
    # ###criterion: slope two times in a row -0.5 
    # slopes_highest = slopes_ev[0,:]       
    # slopes_lowest  = slopes_ev[-1,:]   
    
    # from scipy.stats import wasserstein_distance
    
    # d_estimate = 1
    # for ev in range(1,datadim-1):
    #     slope_ev = slopes_ev[ev,:]
        
    #     distance_to_highest = wasserstein_distance(slopes_highest,slope_ev)
    #     distance_to_lowest  = wasserstein_distance(slopes_lowest,slope_ev)
    #     if distance_to_lowest < distance_to_highest:
    #         d_estimate += 1
    #     #distances
        
    # print('Estimate for d=',d_estimate)
    
    # def plt_errorbar(xplt,yplt,yerr,label=None,lw=2,c='k',marker='o',alpha=0.3,ls=None,log_scale=False):
    #     ax.plot(xplt,yplt,lw=lw,c=c,marker=marker,ls=ls,label=label)
    #     if log_scale:
    #         #ax.fill_between(xplt,yplt-0.43*(yerr/yplt),yplt+0.43*(yerr/yplt),color=c,alpha=alpha)
    #         ax.fill_between(xplt,yplt*yerr,yplt/yerr,color=c,alpha=alpha)
    #     else:
    #         ax.fill_between(xplt,yplt-yerr,yplt+yerr,color=c,alpha=alpha)
    
    # ## singular values vs sigmas
    # fig = plt.figure(figsize=(20,10))
    
    # for n in range(datadim):
    #     mean = sing_values_mean[:,n]
    #     std = sing_values_std[:,n]
        
    #     ax = fig.add_subplot(111)
    #     plt_errorbar(sigmas,mean,std,log_scale=False) #,c=colors[n],label=labels[n]
    
      
    # plt.yscale('log', nonposy='clip')
    # plt.xscale('log', nonposx='clip') 
    # # ax.set_xlim(0,10**(-1))  
    # plt.savefig(os.path.join(r'D:\PROJECTS\estimate_intrinsic_dim_d\data\d_sphere',str(datadim)+'.pdf') )
    
    #################################
    
    #################################
