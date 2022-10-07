

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

def linear_fit(x, alpha, beta): 
    return alpha*x + beta

def exp_fit(x, alpha, beta,gamma):
    return alpha - np.exp( (x-gamma) * beta)   

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
    
    # def double_sigmoid(x, a1,a2,b1,b2,c1,c2): #a1+a2**2
    #     return a1 / (1 + np.exp(-(c1)**2 * (b1- x) ) ) # + (a1+a2**2) / (1 + np.exp(- (c2)**2 *(b2 - x) ) )  #+ (a1+a2**2)
    KK = 500
    JUMP_left = 10 #5
    JUMP_right = 10 #5
    
    bounds = ([0,-np.inf,0,0,0,0],[datadim,np.inf,np.inf,datadim,np.inf,np.inf])
    p0 = (1,t_min,1,1,np.sqrt(t_max-t_min),1)  #[1:-2]  -2
     
    alphas_sigmoid = np.concatenate([np.linspace(t_min-JUMP_left,t_min,KK),alphas,np.linspace(t_max + 0.1 ,t_max+JUMP_right,KK)])
    counts_sigmoid = np.concatenate([np.zeros(KK),counts,np.ones(KK)*datadim])

    sigmoid_param , pcov_d = curve_fit(double_sigmoid, alphas_sigmoid, counts_sigmoid,maxfev = 100000, method="trf",bounds=bounds,p0=p0) 
    
    d_hat = sigmoid_param[3]
            
    #######################
    #######################
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
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title   
        ###################         

        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
        ax.plot(alphas,counts) 
        # ax.plot(alphas,np.ones(len(alphas))*int(datadim/2))
        ax.plot(alphas,counts,label= r'$F(\alpha)$')    
        ax.plot(alphas,double_sigmoid(alphas,*sigmoid_param), linestyle='dashed',label=r'$\hat{F}(\alpha)$') 
    
        yint = range(0, math.ceil(datadim)+1,40)
        matplotlib.pyplot.yticks(yint)
        
        # ax.set_title(str(args.latent_distribution) + ' on ' + str(args.dataset) + r' $ a_2 =$'+str( np.round(d_hat,2) ) ) 
        # ax.set_title( plot_title ) #int(datadim) 
        ax.set_xlabel(r'$\log  \ \alpha^2$')
        ax.set_ylabel(r'# singular values')
        
        ax.set_title(r'$ a_2 =$'+str( np.round(d_hat,2) ) )
    
        # ax.legend()
        # ax.legend(loc='upper left')
        
        ax.plot(alphas, datadim*np.ones(len(alphas)), c='black', linestyle='dashed')
        #ax.vlines(-6, 200, datadim, colors='black',linestyles='dashed')
        #ax.text(-5.8, 280, r'$\hat{d} =$' + str(200))
        
        plt.savefig(os.path.join(save_path,str('sing_values_plot_')+tag+'.pdf') )
    
    return sigmoid_param[3] 
    
def estimate_d(sing_values,sigmas,batch_size,datadim,mode='slope',n_start=0,latent_dim=2,plot=False, mean=False, save_path = r'D:\PROJECTS\estimate_intrinsic_dim_d\data\d_sphere\utils_plots', tag='random', plot_title = 'some title'):
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
        
    if mode == 'slope':
        
        if mean:
            # import pdb; pdb.set_trace()
            
            sing_values_mean = sing_values #np.mean(sing_values,axis=1)
            
            slopes = np.zeros(datadim)
                
            for d in range(datadim):
                y_ = sing_values_mean[:,d]
                #fill out missing data
                idx_0 = np.where(y_==0)[0]
                for idx in idx_0:
                    y_[idx] = 0.5 * (y_[idx+1] +y_[idx-1])
                    print('some missing data for d=',d)
                
                bounds = ([0,0],[np.inf,np.inf])
                params_d, pcov_d = curve_fit(theory_fit, sigmas[n_start:n_sigmas], np.log(y_),bounds=bounds,maxfev = 100000)
                a, b = params_d
                
                # import pdb
                # pdb.set_trace()
                if a <= np.min(sigmas[n_start:n_sigmas]):
                    slopes[d] = np.min(sigmas[n_start:n_sigmas])
                else: 
                    slopes[d] = np.log(a)  ## a instead of a**2 because of -0.5 in theory fit
                 
                # print(params_d)
                # onsets[d]  = np.log(a**2) #params_d[2] + 1 / params_d[1]
                # heights[d] = b - 0.5 * np.log(a**2) #params_d[0]
                
                # popt_d, pcov_d = curve_fit(linear_fit,np.log(sigmas[n_start:n_sigmas]), np.log(y_), maxfev=10000)
                # slopes[d] = popt_d[0]
            
            d_hat_NF = estimate_sigmoid(slopes,datadim,save_path,plot_title,tag,plot=plot)
            d_std_NF = 0
            
        else:   
            d_hat_batch = np.zeros(batch_size)
            for sample in range(batch_size): #batch_size
                slopes = np.zeros(datadim)
                
                #import pdb; pdb.set_trace()
                print('estimate d for sample ',sample)
                for d in range(datadim):
                    y_ = sing_values[:,sample,d] #np.log np.log
                    
                    #fill out missing data
                    idx_0 = np.where(y_==0)[0]
                    for idx in idx_0:
                        y_[idx] = 0.5 * (y_[idx+1] +y_[idx-1])
                        print('some missing data for d=',d)
                    import pdb; pdb.set_trace()        
                    bounds = ([0,0],[np.inf,np.inf])
                    
                    params_d, pcov_d = curve_fit(theory_fit,sigmas[n_start:n_sigmas], np.log(y_), maxfev=100000) #  sigmas, mean_d
                    a, b = params_d
                    
                    if a <= np.min(sigmas[n_start:n_sigmas]):
                        slopes[d] = np.min(sigmas[n_start:n_sigmas])
                    else: 
                        slopes[d] = np.log(a) 
                        
                    # print(popt_d)
                    # slopes[d] = popt_d[0]
                d_hat = estimate_sigmoid(slopes,datadim,save_path,plot_title,tag,plot=plot)
                d_hat_batch[sample] = d_hat #estimate_sigmoid(slopes,datadim,save_path,plot_title,tag,plot=plot)
                print('---d_hat ',d_hat) 
                import pdb; pdb.set_trace()
             
            d_hat_NF = np.mean(d_hat_batch)
            d_std_NF = np.std(d_hat_batch)
            
            idx_stick = np.array([    1,   2,   4,   8,   9,  10,  11,  16,  17,  19,  21,  25,  26,
                                     29,  32,  33,  36,  37,  38,  42,  46,  47,  50,  53,  55,  56,
                                     57,  58,  59,  62,  63,  64,  68,  69,  71,  76,  83,  84,  87,
                                     89,  90,  91,  92,  95,  96,  97, 104, 106, 108, 109, 110, 111,
                                    113, 116, 117, 119, 120, 123, 125, 129, 130, 131, 134, 136, 137,
                                    139, 140, 141, 142, 143, 145, 146, 147, 150, 153, 154, 156, 157,
                                    159, 161, 163, 164, 167, 168, 169, 173, 175, 177, 178, 183, 185,
                                    186, 189, 190, 191, 192, 194, 198])
            
            import pdb; pdb.set_trace()
        # np.save(os.path.join(save_path, 'd_hat.npy'),d_hat_NF)
        # np.save(os.path.join(save_path, 'd_hat_std.npy'),d_std_NF)    

    elif mode == 'cluster':
        d_hat = 0
        onsets, heights = np.zeros(datadim), np.zeros(datadim)
        
        colors  = np.empty(datadim, dtype=str)
        for i in range(datadim):
           colors[i] = "b"
        # colors = np.zeros(datadim)
        if mean:
            # import pdb; pdb.set_trace()
            sing_values_mean = np.mean(sing_values,axis=1)
            

            for d in range(datadim):
                y_ = sing_values_mean[:,d]           
                params_d, pcov_d = curve_fit(theory_fit, sigmas, np.log(y_),maxfev = 100000) #note that sigmas is actually sigmas^2... bad notation
                a,b = params_d
                # print(params_d)
                onsets[d]  =  np.log10(a**2) #params_d[2] + 1 / params_d[1]
                heights[d] = 1 * (np.log10(b**2) - 0.5 * np.log10(a**2) )#params_d[0]
            
                       
                if onsets[d] < np.log10(sigmas[0]): #onsets before sigma0 can be set w.l.o.g. to sigma[0]
                    onsets[d] = np.log10(sigmas[0])

                if onsets[d] > np.log10(sigmas[-1]): #onsets after sigma0 can be set w.l.o.g. to sigma[last]
                    onsets[d] = np.log10(sigmas[-1])
                
            colors[- latent_dim:] = "red" #0.5
            df = np.transpose(np.array([onsets,heights]))
            
            fig = plt.figure(figsize=(10,10))       
            ax = fig.add_subplot(111)
            ax.scatter(df[:,0],df[:,1], c=colors ) #kmeans.labels_.astype(float))
            
            ax.vlines(np.log10( (255* 0.68)**2 ), -3.4, 1, colors='black',linestyles='dashed')
            
            print( 'alphas > boundardy ',np.sum(onsets >= np.log10((255* 0.68)**2)) )
            ax.set_xlabel(r'onsets')
            ax.set_ylabel(r'heights')
            
            ax.set_title(r'StyleGan manifold $d=$'+str( latent_dim ) )
            
            plt.savefig(os.path.join(save_path,str('theory_cluster_mean')+tag+'.pdf') )
            
            d_hat_NF, d_std_NF = np.sum(onsets >= np.log10((255* 0.68)**2)), 0
        else:
            for b in range(batch_size):
                sing_values_mean = sing_values[:,b,:]

                for d in range(datadim):
                    y_ = sing_values_mean[:,d]           
                    params_d, pcov_d = curve_fit(theory_fit, sigmas, np.log(y_),maxfev = 100000)
                    a,b  = params_d
                    # print(params_d)
                    onsets[d]  = np.log(a**2) #params_d[2] + 1 / params_d[1]
                    heights[d] = np.log(b**2) - 0.5 * np.log(a**2) #params_d[0]
                    
                    if onsets[d] < np.log10(sigmas[0]): #onsets before sigma0 can be set w.l.o.g. to sigma[0]
                        onsets[d] = np.log10(sigmas[0])

                    if onsets[d] > np.log10(sigmas[-1]): #onsets after sigma0 can be set w.l.o.g. to sigma[last]
                        onsets[d] = np.log10(sigmas[-1])
                    
                    
                colors[- latent_dim:] = 1
                df = np.transpose(np.array([onsets,heights]))
                fig = plt.figure(figsize=(10,10))       
                ax = fig.add_subplot(111)
                ax.scatter(df[:,0],df[:,1], c=colors ) #kmeans.labels_.astype(float))
                
                ax.set_xlabel(r'onsets')
                ax.set_ylabel(r'heights')
                
                plt.savefig(os.path.join(save_path,str('theory_cluster_mean_batch_')+str(b)+'.pdf') )
                
                d_hat_NF, d_std_NF = 0, 0
            
        # else:       
        #     for sample in range(batch_size): #(batch_size #batch_size
        #         # bounds = ([-np.inf,0,-np.inf],[np.inf,np.inf,np.inf])
                
                
        #         for d in range(datadim):
        #             mean_d = sing_values[:,sample,d] #3(sing_values_mean[:,d])
        #             # params_d, pcov_d = curve_fit(exp_fit, np.log(sigmas), np.log(mean_d),maxfev = 50000,bounds=bounds)
        #             params_d, pcov_d = curve_fit(theory_fit, sigmas, np.log(mean_d),maxfev = 50000) #,bounds=bounds)
                    
        #             a,b  = params_d
        #             # print(params_d)
        #             onsets[d]  = np.log(a**2) #params_d[2] + 1 / params_d[1]
        #             heights[d] = b - 0.5 * np.log(a**2) #params_d[0]
                
        #         kmeans = KMeans(n_clusters= 2)
        #         df = np.transpose(np.array([onsets,heights]))
        #         label = kmeans.fit_predict(df)
                
        #         fig = plt.figure(figsize=(20,10))       
        #         ax = fig.add_subplot(122)
        #         ax.scatter(df[:,0],df[:,1],label = label, c=kmeans.labels_.astype(float))
                
        #         ax.set_xlabel(r'onsets')
        #         ax.set_ylabel(r'heights')
                
        #         plt.savefig(os.path.join(save_path,str('theory_cluster_')+tag+str(sample)+'.pdf') )
            # import pdb; pdb.set_trace()


    # print('d_hat = ',d_hat / batch_size)    

    return d_hat_NF, d_std_NF
    #plt.savefig(os.path.join(r'D:\PROJECTS\estimate_intrinsic_dim_d\data\d_sphere\10_K_imgs',str('sphere_sing_values_plot')+str(datadim)+'.pdf') )

def NEWestimate_d(sing_values,sigmas,batch_size,datadim,mode='slope',n_start=0,latent_dim=2,plot=False, mean=False, save_path = r'D:\PROJECTS\estimate_intrinsic_dim_d\data\d_sphere\utils_plots', tag='random', plot_title = 'some title'):
    if sing_values.shape[0] == batch_size and sing_values.shape[2] == datadim:
        sing_values = np.swapaxes(sing_values,0,1)
        
    onsets, heights = np.zeros(datadim), np.zeros(datadim)
    
    if mean:
        sing_values_mean = np.mean(sing_values,axis=1)
        for d in range(datadim):
            y_ = sing_values_mean[:,d]
            
            # import pdb; pdb.set_trace() 
            params_d, pcov_d = curve_fit(theory_fit, sigmas, np.log(y_),maxfev = 100000)
            alpha,beta  = params_d
            # alpha, beta  = params_d, 1
            
            onsets[d]  = np.log(alpha**2) #params_d[2] + 1 / params_d[1]
            heights[d] = np.log(beta**2) - 0.5 * np.log(alpha**2) #params_d[0]
           
            if onsets[d] < np.log(sigmas[0]): #onsets before sigma0 can be set w.l.o.g. to sigma[0]
                onsets[d] = np.log(sigmas[0])

            if onsets[d] > np.log(sigmas[-1]): #onsets after sigma0 can be set w.l.o.g. to sigma[last]
                onsets[d] = np.log(sigmas[-1])
                    
        sort_slopes = np.sort(onsets)
        dist = np.zeros(datadim-1)   
        for i in range(datadim-1):
            dist[i] = np.abs(sort_slopes[i+1] - sort_slopes[i])
        d_hats = datadim - (np.argmax(dist)+1)
        
        fig = plt.figure(figsize=(20,10))
        ax = fig.add_subplot(111)
        ax.hist(onsets)
        
    else:
        d_hats = np.zeros(batch_size)
        for bb in range(batch_size):
            sing_values_mean = sing_values[:,bb,:]
            
            # fig = plt.figure(figsize=(20,10))
            
            # # import pdb; pdb.set_trace()
            
            # ax = fig.add_subplot(111)
            # for d in range(datadim):
            #     # sing_values_d = sing_values_batch[:,bb,d]
            #     sing_values_d = sing_values_mean[:,d]
            #     ax.plot(sigmas,sing_values_d) #,c=colors[n],label=labels[n]
           
            # plt.yscale('log') #, nonposy='clip')
            # plt.xscale('log') #, nonposx='clip')
            
            # # ax.set_title(r'Style Gan 2d: $\lambda^{i}(\sigma^2)$' ) #int(datadim) 
            # ax.set_xlabel(r'inflation noise variance $\sigma^2$')
            # ax.set_ylabel(r'singular value $\lambda$')
            # # plt.show()
            # plt.savefig(os.path.join(r'D:\PROJECTS\estimate_intrinsic_dim_d\data\toy_examples\plots\supp_plots\swiss_correlated',str(bb)+'.pdf'))
            
            
            for d in range(datadim):
                
                y_ = sing_values_mean[:,d]    
                
                # idx_0 = np.where(y_==0)[0]
                # for idx in idx_0:
                #     y_[idx] = 0.5 * (y_[idx+1] +y_[idx-1])
                #     print('some missing data for d=',d)
                    
                        
                # import pdb; pdb.set_trace() 
                params_d, pcov_d = curve_fit(theory_fit, sigmas, np.log(y_),maxfev = 100000)
                alpha,beta  = params_d
                # alpha, beta  = params_d, 1
                
                
                # print(params_d)
                onsets[d]  = np.log(alpha**2) #params_d[2] + 1 / params_d[1]
                
                if onsets[d] < np.log(sigmas[0]): #onsets before sigma0 can be set w.l.o.g. to sigma[0]
                    onsets[d] = np.log(sigmas[0]) # no intrinsic noise

                if onsets[d] > np.log(sigmas[-1]): #onsets after sigma0 can be set w.l.o.g. to sigma[last]
                    onsets[d] = np.log(sigmas[-1]) #if sigma is not large enough to see onset
                    
                heights[d] = np.log(beta**2) - 0.5 * np.log(alpha**2) #params_d[0]
               
                
                #fig = plt.figure(figsize=(20,10))
                #ax = fig.add_subplot(111)
                #ax.hist(onsets)
                
            sort_slopes = np.sort(onsets)
            dist = np.zeros(datadim-1)   
            for i in range(datadim-1):
                dist[i] = np.abs(sort_slopes[i+1] - sort_slopes[i])
            
            
            
            # import pdb; pdb.set_trace()
            d_hats[bb] = datadim - (np.argmax(dist)+1)
            
            print('d_hats[b]', d_hats[bb])
    #import pdb; pdb.set_trace()
    return np.mean(d_hats), np.std(d_hats)
        
    
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