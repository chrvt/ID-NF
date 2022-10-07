import numpy as np
import scipy
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import os
import time
# TWO-NN METHOD FOR ESTIMATING INTRINSIC DIMENSIONALITY
# Facco, E., d’Errico, M., Rodriguez, A., & Laio, A. (2017).
# Estimating the intrinsic dimension of datasets by a minimal neighborhood information.
# Scientific reports, 7(1), 12140.


# Implementation by Jason M. Manley, jmanley@rockefeller.edu
# June 2019


def sample(n,data_dim,latent_dim):
    x_ = np.random.randn(n,int(latent_dim)).astype('f')  #self._transform_z_to_x(theta,phi,mode='numpy')
    x = normalize(x_,axis=1)
    
    for i in range( int(data_dim - latent_dim) ):
        dz =   np.zeros([n,1]).astype('f')
        x = np.concatenate([x,dz], axis=1) 
    return x


def estimate_id(X,datadim=0, plot=False, X_is_dist=False, save_path=save_path):
    # INPUT:
    #   X = Nxp matrix of N p-dimensional samples (when X_is_dist is False)
    #   plot = Boolean flag of whether to plot fit
    #   X_is_dist = Boolean flag of whether X is an NxN distance metric instead
    #
    # OUTPUT:
    #   d = TWO-NN estimate of intrinsic dimensionality

    N = X.shape[0]

    if X_is_dist:
        dist = X
    else:
        # COMPUTE PAIRWISE DISTANCES FOR EACH POINT IN THE DATASET
        dist = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(X, metric='euclidean'))

    # FOR EACH POINT, COMPUTE mu_i = r_2 / r_1,
    # where r_1 and r_2 are first and second shortest distances
    mu = np.zeros(N)

    for i in range(N):
        sort_idx = np.argsort(dist[i,:])
        mu[i] = dist[i,sort_idx[2]] / dist[i,sort_idx[1]]

    # COMPUTE EMPIRICAL CUMULATE
    sort_idx = np.argsort(mu)
    Femp     = np.arange(N)/N

    # FIT (log(mu_i), -log(1-F(mu_i))) WITH A STRAIGHT LINE THROUGH ORIGIN
    lr = LinearRegression(fit_intercept=False)
    lr.fit(np.log(mu[sort_idx]).reshape(-1,1), -np.log(1-Femp).reshape(-1,1))

    d = lr.coef_[0][0] # extract slope + 1 

    if plot:
        # PLOT FIT THAT ESTIMATES INTRINSIC DIMENSION
        s=plt.scatter(np.log(mu[sort_idx]), -np.log(1-Femp), c='r', label='data')
        p=plt.plot(np.log(mu[sort_idx]), lr.predict(np.log(mu[sort_idx]).reshape(-1,1)), c='k', label='linear fit')
        plt.xlabel('$\log(\mu_i)$'); plt.ylabel('$-\log(1-F_{emp}(\mu_i))$')
        plt.title('ID = ' + str(np.round(d, 3)))
        # plt.show()
        # plt.legend()
        plt.savefig(os.path.join(save_path,str('twoNN_')+str(datadim)+'.pdf') )


    return d 

batch_size = 10 #200

data_dims = [40,100,200,300,400] #[20,40,60,80,100, 120,140,160,200, 300 ,400] # [80,140,200,300,400] 
d_estimates = np.zeros([len(data_dims),batch_size])
# d_std = np.zeros(len(data_dims))
count_d = -1
for datadim in data_dims:
    count_d += 1
    for k in range(batch_size): 
        # print('data dim ',datadim)
        # t = time.time()  
        X = sample(10000,int(datadim),int(datadim/2))
        d = estimate_id(X,datadim=int(datadim),plot=True)
        # elapsed = time.time() - t
        # print('Time needed to evaluate model samples: %s sec',elapsed)
        # print('estimate ',d)
        d_estimates[count_d,k] = d
    print('finished d=',datadim)

d_mean = np.mean(d_estimates,axis=1)
d_std = np.std(d_estimates,axis=1)

save_path = r'save_path' 
np.save(os.path_join(save_path,'twoNN_mean.npy'),d_mean)
np.save(os.path_join(save_path,'twoNN_std.npy'),d_std)
