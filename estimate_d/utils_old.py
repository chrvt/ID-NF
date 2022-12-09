

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
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans

def double_sigmoid(x, a1,b1,c1, a2,b2,c2): #a1+a2**2
    return a1 / (1 + np.exp(-c1 * (x-b1) ) ) + a2 / (1 + np.exp(-c2 * (x- (b1 + b2**2)) ) )

def theory_fit(x, alpha, beta): # , beta)
    return -0.5 * np.log(alpha**2 + x) + np.log(beta**2)

    
def estimate_sigmoid(slopes,datadim,save_path,plot_title,tag,plot=False):
    t_max = slopes.max()
    t_min = slopes.min()
    
    alphas = np.linspace(t_min,t_max+1 ,1000)  #+ 0.1
    counts = np.zeros(len(alphas))
    
    i = -1
    for alpha in alphas:
        i += 1
        counts[i] = np.sum(slopes<alpha)
    
    KK = 500
    JUMP_left = 10 #5
    JUMP_right = 10 #5
    
    bounds = ([0,-np.inf,0,0,0,0],[datadim,np.inf,np.inf,datadim,np.inf,np.inf])
    p0 = (1,t_min,1,1,np.sqrt(t_max-t_min),1)  #[1:-2]  -2
     
    alphas_sigmoid = np.concatenate([np.linspace(t_min-JUMP_left,t_min,KK),alphas,np.linspace(t_max + 0.1 ,t_max+JUMP_right,KK)])
    counts_sigmoid = np.concatenate([np.zeros(KK),counts,np.ones(KK)*datadim])
    # import pdb; pdb.set_trace()
    sigmoid_param , pcov_d = curve_fit(double_sigmoid, alphas_sigmoid, counts_sigmoid,maxfev = 100000, method="trf",bounds=bounds,p0=p0) 
    
    d_hat = sigmoid_param[3]
            
    #######################
    #######################
    
    # import pdb; pdb.set_trace()
    if plot:
        ############plotting
        SMALL_SIZE = 26
        MEDIUM_SIZE = 30
        BIGGER_SIZE = 40
        
        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize           
        ##################                  
        fig = plt.figure(figsize=(20,10))
        ax = fig.add_subplot(111)
        ax.plot(alphas,counts) 
        ax.plot(alphas,counts,label= r'$F(\alpha)$')    
        ax.plot(alphas,double_sigmoid(alphas,*sigmoid_param), linestyle='dashed',label=r'$\hat{F}(\alpha)$') 
    
        yint = range(0, math.ceil(datadim)+1,40)
        matplotlib.pyplot.yticks(yint)
        
        ax.set_xlabel(r'$\log  \ \alpha^2$')
        ax.set_ylabel(r'# singular values')
        
        ax.set_title(r'$ a_2 =$'+str( np.round(d_hat,2) ) )
        ax.plot(alphas, datadim*np.ones(len(alphas)), c='black', linestyle='dashed')
        
        plt.savefig(os.path.join(save_path,str('sing_values_plot_')+tag+'.pdf') )
    
    return sigmoid_param[3] 
    
def estimate_d(sing_values,sigmas,batch_size,datadim,mode='vector',n_start=0,latent_dim=2,plot=False, mean=False, save_path = r'path_to_save', tag='random', plot_title = 'some title'):
    n_sigmas = len(sigmas)
    
    ## reshape sing_values to standard shape: n_sigmas x batch_size x data_dim
    if sing_values.shape[0] == batch_size and sing_values.shape[2] == datadim:
        sing_values = np.swapaxes(sing_values,0,1)
        # sing_values = sing_values.reshape([sing_values.shape[1],sing_values.shape[0],sing_values.shape[2]])
    # else: print('potential problem with singular values shape')
    
    ############plotting
    SMALL_SIZE = 22
    MEDIUM_SIZE = 26
    BIGGER_SIZE = 30
    
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title   
    ###################      
        
    if mode == 'vector':
        
        sigmas = np.inser(sigmas,0,1e-10)
        sing_values = np.insert(sing_values,0,sing_values[0,:],axis=0) #stabilizes param fit in case no intrinsic noise 
        
        slopes = np.zeros(datadim)
            
        for d in range(datadim):
            y_ = sing_values[:,d]
            #fill out missing data
            idx_0 = np.where(y_==0)[0]
            for idx in idx_0:
                y_[idx] = 0.5 * (y_[idx+1] +y_[idx-1])
                print('some missing data for d=',d)
            
            bounds = ([0,0],[np.inf,np.inf])
            params_d, pcov_d = curve_fit(theory_fit, sigmas[n_start:n_sigmas], np.log(y_),bounds=bounds,maxfev = 100000)
            a, b = params_d
            
            slopes[d] = np.log10(a)
            if a**2 < sigmas[0]: #onsets before sigma0 can be set w.l.o.g. to sigma[0]
                slopes[d] = np.log10(sigmas[0])

            if a**2 > sigmas[-1]: #onsets after sigma0 can be set w.l.o.g. to sigma[last]
                slopes[d] = np.log10(sigmas[-1])
        #import pdb; pdb.set_trace()
        d_hat_NF = estimate_sigmoid(slopes,datadim,save_path,plot_title,tag,plot=plot) #False) #
        d_std_NF = 0
        

    elif mode == 'image':
        d_hat = 0
        onsets, heights = np.zeros(datadim), np.zeros(datadim)
        
        colors  = np.empty(datadim, dtype=str)
        for i in range(datadim):
           colors[i] = "b"
            
        # import pdb; pdb.set_trace()
        sing_values_mean = np.mean(sing_values,axis=1)
        
        for d in range(datadim):
            y_ = sing_values_mean[:,d]           
            params_d, pcov_d = curve_fit(theory_fit, sigmas, np.log(y_),maxfev = 50000)
            a,b  = params_d
            # print(params_d)
            onsets[d]  =  np.log10(a**2) #params_d[2] + 1 / params_d[1]
            heights[d] = 1 * (np.log10(b**2) - 0.5 * np.log10(a**2) )#params_d[0]
        
        
            
        colors[- latent_dim:] = "red" #0.5
        df = np.transpose(np.array([onsets,heights]))
        
        fig = plt.figure(figsize=(10,10))       
        ax = fig.add_subplot(111)
        ax.scatter(df[:,0],df[:,1], c=colors ) 
        
        ax.vlines(np.log10( (255* 0.68)**2 ), -3.4, 1, colors='black',linestyles='dashed')
        
        print( 'alphas > boundardy ',np.sum(onsets >= np.log10((255* 0.68)**2)) )
        ax.set_xlabel(r'onsets')
        ax.set_ylabel(r'heights')
        
        ax.set_title(r'StyleGan manifold $d=$'+str( latent_dim ) )
        
        plt.savefig(os.path.join(save_path,str('theory_cluster_mean')+tag+'.pdf') )
        
        d_hat_NF, d_std_NF = 0, 0
    
    return d_hat_NF, d_std_NF