# ID-NF

*Christian Horvat and Jean-Pascal Pfister 2022*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![NeurIPS](http://img.shields.io/badge/NeurIPS-2021-8B6DA0.svg)](https://proceedings.neurips.cc/paper/2021/hash/4c07fe24771249c343e70c32289c1192-Abstract.html)

This is the official repository of the paper "Estimating the intrinsic dimensionality with Normalizing Flows". To reproduce our results from the paper or to use the method for your own data, follow the instructions below. 

### Methology
The ID-NF method estimates the ID by analyzing how the singular values of the flow's Jacobian change depending on the inflation noise $\sigma^2$. At least 3 NFs with different $\sigma^2$ need to be trained. For a detailed description of the method, we refer to the original paper.

### Using ID-NF for your own data
For instructions for how to train NFs on [images](images) or [vector data](vectors_data), see the corresponding README.md descriptions within the folders. Once $N$ NFs are trained for $\sigma_1,\dots,\sigma_D$ and the singular values are calculated on $x_{1}^*,\dots,x_{K}^*$, the ID can be estimated using the estimate_d function in [estimate_d/utils.py](estimate_d/utils.py), see the documentation of that function for details. We provide a dummy code which can serve as a blueprint for your data [estimate_d/estimate_d.py](estimate_d/estimate_d.py).

### Structure of the repository
In [estimate_d/toy_experiements](estimate_d/toy_experiements) and [estimate_d/OOD_experiements](estimate_d/OOD_experiements) we provide code used for the toy experiments and OOD_experiments.

### Acknoledgement
M-flow, inflation/deflation


### README FOR IMAGE OR VECTOR DATA
To train your own data, you need to write your own dataset simulator and include it in [experiments/datasets](experiments/datasets). In case of vector valued data, the simulator must have the following methods:

+ latent_dist ... returning the latent distribution to sample from
+ manifold ... returning the name of the manifold
+ is_image ... determining if dataset is image
+ data_dim ... returning the embedding space dimension
+ latent_dim ... returning the latent space dimension
+ def sample_and_noise ... creating an array of manifold data and noise
+ def _draw_z  ... drawing a latent variable
+ create_noise ... noise added to data while training
+ _transform_z_to_x ... mapping from latent to data space
+ sample ... returning only manifold data

See e.g. [experiments/datasets/sphere_simulator.py](experiments/datasets/sphere_simulator.py) as orientation.

In case of image data, you need to add another class to the image simulator [experiments/datasets/images.py](experiments/datasets/images.py). The following init values need to be set:
+ resolution ... sepcifying the resolution 64 for 64x64x3 images
+ scale_factor ... scaling the resolution, i.e. 0.5 will change an 64x64x3 image into 32x32x3 image
+ noise_type ... noise type for dequantization before training (e.g. uniform or Gaussian)
+ sig2 ... magnitude of dequantization noise

Data from a gdrive folder can be automatically downloaded when specifying the url in the init setting, see e.g. "class FFHQStyleGAN2DLoader(BaseImageLoader)". Otherwise, the data can be manually added into [experiments/data/samples](experiments/data/samples) within a folder named by the dataset name. The training set must be saved as "train.npy" with shape (N,c,res,res) where N is the number of samples, c the number of channels, and  res the reoslution. The test set must be saved as "test.npy" with the same shape as the training set.

In case you have problems with setting up your data, do not hesitate to contact us.


