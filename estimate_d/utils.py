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

# sum of two sigmoid for fitting F(alpha)
def double_sigmoid(x, a1,b1,c1, a2,b2,c2):
    return a1 / (1 + np.exp(-c1 * (x-b1) ) ) + a2 / (1 + np.exp(-c2 * (x- (b1 + b2**2)) ) )

# theory prediction of singular values evolution, with additional flexibility by adding beta
def theory_fit(x, alpha, beta): # , beta)
    return -0.5 * np.log(alpha**2 + x) + np.log(beta**2)

# estimating decay onsets according to theory prediction of singular values evolution
def estimate_sigmoid(slopes,datadim,save_path,plot_title,tag,plot=True):
    t_max = slopes.max()
    t_min = slopes.min()

    alphas = np.linspace(t_min,t_max+1 ,1000)  # 1000 is precision, defines coarse graining of F(\alpha)
    counts = np.zeros(len(alphas))             # F(\alpha)

    i = -1
    for alpha in alphas:
        i += 1
        counts[i] = np.sum(slopes<alpha)

    # to get a good sigmoid fit, we artifically add data left and right of F(\alpha) to truly caputre the flatness of F(\alpha)
    # alternatively, we could also simply adjust t_max and t_min
    KK = 500
    JUMP_left = 10 #5
    JUMP_right = 10 #5
    # updated alphas and counts
    alphas_sigmoid = np.concatenate([np.linspace(t_min-JUMP_left,t_min,KK),alphas,np.linspace(t_max + 0.1 ,t_max+JUMP_right,KK)])
    counts_sigmoid = np.concatenate([np.zeros(KK),counts,np.ones(KK)*datadim])

    # intiual conditions
    bounds = ([0,-np.inf,0,0,0,0],[datadim,np.inf,np.inf,datadim,np.inf,np.inf])
    p0 = (1,t_min,1,1,np.sqrt(t_max-t_min),1) # forces the second sigmoidal to be after first

    sigmoid_param , pcov_d = curve_fit(double_sigmoid, alphas_sigmoid, counts_sigmoid,maxfev = 100000, method="trf",bounds=bounds,p0=p0)
    d_hat = sigmoid_param[3] # estimate is a_2 (the height of the second sigmoidal)

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

        plt.savefig(os.path.join(save_path,str('F_alpha_plot_')+tag+'.pdf') )

    return sigmoid_param[3]

# main function for estimating the ID using the ID-NF method
# requires: sing_values ... the singular values as functions of sigmas
#           sigmas ... the sigmas used for inflation
#           datadim ... data dimension
# following options are available:
#           mode ... vector or image data
#           latent_dim ... number of smallest singular values to plot red (onyl for illustration purposes and if true ID is known)
#           plot ... plotting the counting function F(\alpha) and its estimate
#                    (slow for very high dimensions)
#           save_path ... path for saving generatiede figures
#           tag ... custom tag added to file name of figure
#           plot_title ...title of plot
def ID_NF_estimator(sing_values,sigmas,datadim,mode='vector',latent_dim=2,plot=False, save_path = r'path_to_save', tag='random', plot_title = 'some title'):
    n_sigmas = len(sigmas)

    #check if correct shape
    #import pdb; pdb.set_trace()
    assert(sing_values.shape == (n_sigmas,datadim))

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
        # in case of no intrinsic noise, off-manifold singular values will decline immediately such that decay onset might
        # not reflect to decay; therefore, we add an artificial sigma measurment with the same value as the true last sigma
        # singular value
        sigmas = np.insert(sigmas,0,sigmas[0] * 10**(-1))
        sing_values = np.insert(sing_values,0,sing_values[0,:],axis=0)

        # onsets of decay
        onsets = np.zeros(datadim)
        for d in range(datadim):
            y_ = sing_values[:,d]
            #filling out missing data - potential for improvement :-)
            idx_0 = np.where(y_==0)[0]
            for idx in idx_0:
                y_[idx] = 0.5 * (y_[idx+1] +y_[idx-1])
                print('some missing data for d=',d)

            # fitting theory prediction to selected singular value
            bounds = ([0,0],[np.inf,np.inf]) # initial value for parameter search
            params_d, pcov_d = curve_fit(theory_fit, sigmas, np.log(y_),bounds=bounds,maxfev = 100000)
            a, b = params_d

            onsets[d] = np.log10(a)
            if a**2 < sigmas[0]: #onsets before sigma0 can be set w.l.o.g. to sigma[0]
                onsets[d] = np.log10(sigmas[0])

            if a**2 > sigmas[-1]: #onsets after sigma0 can be set w.l.o.g. to sigma[last]
                onsets[d] = np.log10(sigmas[-1])

        #import pdb; pdb.set_trace()
        # fits 2 sigmoidals and reads out the estimate ID
        d_hat_NF = estimate_sigmoid(onsets,datadim,save_path,plot_title,tag,plot=plot)

    elif mode == 'image':
        d_hat = 0
        onsets, heights = np.zeros(datadim), np.zeros(datadim)

        # coloring the smalles latent_dim singular values
        colors  = np.empty(datadim, dtype=str)
        for i in range(datadim):
           colors[i] = "b"

        for d in range(datadim):
            y_ = sing_values[:,d]
            params_d, pcov_d = curve_fit(theory_fit, sigmas, np.log(y_),maxfev = 50000)
            a, b  = params_d
            onsets[d]  =  np.log10(a**2)
            heights[d] = 1 * (np.log10(b**2) - 0.5 * np.log10(a**2) ) # heights not necessary

        colors[- latent_dim:] = "red"
        df = np.transpose(np.array([onsets,heights])) # dataframe

        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
        ax.scatter(df[:,0],df[:,1], c=colors )
        # plot dashed line at sigma max (as described in main paper)
        ax.vlines(np.log10( (255* 0.68)**2 ), -3.4, 1, colors='black',linestyles='dashed')

        ax.set_xlabel(r'onsets')
        ax.set_ylabel(r'heights')
        ax.set_title(plot_title)

        plt.savefig(os.path.join(save_path,'heights_onset_'+tag+'.pdf') )

        d_hat_NF =  np.sum(onsets >= np.log10((255* 0.68)**2)) # counting onsets >= sigma max

    return d_hat_NF
