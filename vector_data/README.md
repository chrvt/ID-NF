### Instructions for vector data

To train your own vector data, write your own dataset simulator and include it in [experiments/datasets](experiments/datasets). 
The simulator must have the following methods:

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

In case you have problems with setting up your data, do not hesitate to contact us.
