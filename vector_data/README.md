### Instructions for vector data

To estimte the ID of your vector dataset, use 

1. [my_vector_data_cluster.py](my_vector_data_cluster.py) which trains N NFs and calculates the singular values on K samples, 
2. [estimate_d/estimate_d.py](estimate_d/estimate_d.py) which estimates the ID given the singular value evolution.

## Details for 1.
+ your data with name 'data_name' must be stored in "/data/data_name". The folder must contain a "train.npy" and "val.npy" numpy arrays of shape [n_samples,D] where n_samples reflects the number of training/ validation samples and D is the data dimensionality
+ the parameters model yielding the best performance on the validation set will be used for calculating the singular values
+ use the [estimate_d_train.sh](estimate_d_train.sh) shell script to train the NFs 
+ the following arguments must be specified:
  - sig2_min ... min. noise magnitude
  - sig2_max ... max. noise magnitude
  - n_sigmas ... number of noise levels; important: number must align with number of arrays in [estimate_d_train.sh](estimate_d_train.sh) 
  - dataset ... name of your dataset
  
  Block Neural Autoregressive Flows parameters
  - n_hidden ...number of hidden layers 
  - hidden_dim ... hidden dimension (must be multicative of args.data_dim)
  
  Training parameters
  - N_epochs ... number of epochs
  - batch_size ... number of samples used for 1 gradient step
  Evaluation
  - ID_samples ... number of samples to estimate ID on (must be <= batch size, otherwise evaluation part of the script must be modified accordingly)

## Details for 2.
+ Follow the instructions in the file [estimate_d/estimate_d.py](estimate_d/estimate_d.py)

In case you have problems with setting up your data, do not hesitate to contact us.
