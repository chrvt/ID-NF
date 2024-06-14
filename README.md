# ID-NF

*Christian Horvat and Jean-Pascal Pfister 2022*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![NeurIPS](http://img.shields.io/badge/NeurIPS-2022-8B6DA0.svg)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/4f918fa3a7c38b2d9b8b484bcc433334-Abstract-Conference.html)

This is the official repository of the paper "Estimating the intrinsic dimensionality with Normalizing Flows". To reproduce our results from the paper or to use the method for your own data, follow the instructions below. 

### Methology
The ID-NF method estimates the ID by analyzing how the singular values of the flow's Jacobian change depending on the inflation noise $\sigma^2$. At least 3 NFs with different $\sigma^2$ need to be trained. For a detailed description of the method, we refer to the original paper.

### Using ID-NF for your own data
To estimate the ID of your own data, you need to train $N$ NFs and calculate the singular values on $K$ training examples. Any NF implementation can be used. For instructions for using the same NFs as in the original paper, see the README.md descriptions within the folders [images](images) or [vector data](vectors_data), respectively. Once $N$ NFs are trained for $\sigma_1,\dots,\sigma_D$ and the singular values are calculated on $x_{1},\dots,x_{K}$, the ID can be estimated using the estimate_d function in [estimate_d/utils.py](estimate_d/utils.py), see the documentation of that function for details. We provide a dummy code which can serve as a blueprint for your data [estimate_d/estimate_d.py](estimate_d/estimate_d.py).

### Reproducibility
In the paper, we tested our method on various toy and real data. For the image experiments on CelebA and on the Style gan image manifolds, please see instructions in [images](images). Once the singular values are given, the out-of-distribution experiments can be reproduced using [estimate_d/OOD_experiements](estimate_d/OOD_experiements).

For the manifolds and distributions displayed in Table 1, see [vector_data/toy_experiments](vector_data/toy_experiments). There you will find also the instructions for reproducing the experiments on the lolipop dataset and on S(D/2). 

### Acknowledgement
For image data, we used the [Manifold Flow repository](https://github.com/johannbrehmer/manifold-flow) which is based on the [Neural Spline code base](https://github.com/bayesiains/nsf). However, note that we use only the standard NF mode of Manifold Flows (Why? For convenience, I used to work with the code-base before). For vector data, we used the [Inflation-Deflation repository](https://github.com/chrvt/Inflation-Deflation) which is based on this [Block Neural Autoregressive Flow implementation](https://github.com/kamenbliznashki/normalizing_flows).

