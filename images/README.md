### README FOR IMAGE DATA

## Celeba, Stylegan 
The configurations for the models and hyperparameter settings used in the paper can be found in [experiments/configs][experiments/configs]. If you want to reproduce our results on the StyleGan2d, StyleGan64d or CelebA dataset, run [cluster/train_flow_gan2d.sh](cluster/train_flow_gan2d.sh),  [cluster/train_flow_gan64d.sh](cluster/train_flow_gan64d.sh), or [cluster/train_flow_celeba.sh](cluster/train_flow_celeba.sh). 

## Your image data
To train your image data, you need to add another class to the image simulator [experiments/datasets/images.py](experiments/datasets/images.py). 
The following init values need to be set:

+ resolution ... sepcifying the resolution 64 for 64x64x3 images
+ scale_factor ... scaling the resolution, i.e. 0.5 will change an 64x64x3 image into 32x32x3 image
+ noise_type ... noise type for dequantization before training (e.g. uniform or Gaussian)
+ sig2 ... magnitude of dequantization noise (in case of gaussian noise)

Data from a gdrive folder can be automatically downloaded when specifying the url in the init setting, see e.g. "class FFHQStyleGAN2DLoader(BaseImageLoader)" in  [experiments/datasets/images.py](experiments/datasets/images.py).

Otherwise, the data can be manually added into [experiments/data/samples](experiments/data/samples) within a folder named by the dataset name. The training set must be saved as "train.npy" with shape (N,c,res,res) where N is the number of samples, c the number of channels, and  res the reoslution. The test set must be saved as "test.npy" with the same shape as the training set.

In case you have problems with setting up your data, do not hesitate to contact us.

Once your data generation is settled, you can train N=3 NFs and calculate the singular values on K samples using [cluster/train_flow_your_data.sh](cluster/train_flow_your_data.sh). We use the same architecture as for the StyleGan64d - please adopt accordingly.
