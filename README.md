# ID-NF

*Christian Horvat and Jean-Pascal Pfister 2022*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![NeurIPS](http://img.shields.io/badge/NeurIPS-2021-8B6DA0.svg)](https://proceedings.neurips.cc/paper/2021/hash/4c07fe24771249c343e70c32289c1192-Abstract.html)

This is the official repository of the paper "Estimating the intrinsic dimensionality with Normalizing Flows". To reproduce our results from the paper or to use the method for your own data, follow the instructions below. 

### Methology
The ID-NF method estimates the ID by analyzing how the singular values of the flow's Jacobian change depending on the inflation noise $\sigma^2$. At least 3 NFs with different $\sigma^2$ need to be trained. For a detailed description of the method, we refer to the original paper.

### Using ID-NF for your own data
For instructions for how to train NFs on [images](images) or [vector data](vectors_data), see the corresponding README.md descriptions within the folders. Once $N$ NFs are trained for $\sigma_1,\dots,\sigma_D$ and the singular values are calculated on $x_{1}^{*},\dots,x_{K}^{*}$, the ID can be estimated using the estimate_d function in [estimate_d/utils.py](estimate_d/utils.py), see the documentation of that function for details. We provide a dummy code which can serve as a blueprint for your data [estimate_d/estimate_d.py](estimate_d/estimate_d.py).

### Structure of the repository
In [estimate_d/toy_experiements](estimate_d/toy_experiements) and [estimate_d/OOD_experiements](estimate_d/OOD_experiements) we provide code used for the toy experiments and OOD_experiments.

### Acknoledgement
M-flow, inflation/deflation


