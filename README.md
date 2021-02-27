
:warning: :warning: :warning: Experimental - **PLEASE BE CAREFUL**. Intended for Research purposes ONLY. :warning: :warning: :warning:

This repository contains code and data of the paper **GANDaLF: GAN for Data-Limited Fingerprinting**, to be appeared in **Privacy Enhancing Technologies Symposium (PETS) 2021**.
[Read the Paper (pre-print)](https://petsymposium.org/2021/files/papers/issue2/popets-2021-0029.pdf).

Code of this paper is written using Python3 and TensorFlow. 

### Dataset description

- AWF(awf1.npz, tor_open_400000w.npz): awf1.npz consists of 100 sites x 2498 instances. We downloaded tor_100w_2500tr.npz from [AWF github](https://github.com/DistriNet/DLWF) and balanced the instance count per site. 

- DF(df_data.npz, df_all.npz): Each set consists of 95 sites x 1000 instances and each trace consists of packet direction info.

- DF(df_slow.npz, df_fast.npz): Based on the DF set (where each trace file (named "site-instance_index") is formatted in <time_stamp>\t<packet_dir>), using circuit labels, 0-39, we first split the entire set into 40 circuit groups based on instance indices by [circuit_label*25 + i for i in range(25)] for each site. Then, based on the mean page loading time of each group, we select 4 fast circuit groups and 4 slow circuit groups, which we used as testing sets in wfi/circuit/wfi-circuit-fast.py and wfi/circuit/wfi-circuit-slow.py, respectively.

The following is [circuit_label, mean_loading_time] of these circuit groups.
```sh                                
4 slow groups: [[29, 29.311899313174017], [1, 29.026589783014725], [9, 28.93622463908884], [6, 28.92861685090533]]
4 fast groups: [[16, 27.838683811790737], [27, 27.880973149853084], [8, 27.951370530079572], [26, 27.981911256391193]]
```                          

- GDLF(gdlf25_ipd.npz, gdlf_ow_ipd.npz, gdlf25_ow_old_ipd.npz): We collected 25 sites x 96 subpages x 39 instances (gdlf25_ipd.npz) and 70000 x 1 instances (gdlf_ow_ipd.npz). For each of them, we extracted IPD sequences. gdlf25_ow_old_ipd.npz was collected around 3 months prior to gdlf_ow_ipd.npz. The entire set consisting of each trace file formatted in <time_stamp>\t<packet_size> can be provided upon request.

### Dependencies, Required Hardware (HW), and Software (SW) packages

#### HW requirements

 - RAM : > 100G (It will use 75-95G memory)
 
 - GPU Memory : 16G (It will use 15.7-8G memory)
 
 (We used 504G RAM and Tesla P100 GPU with 16G memory)

#### Development env setup

We used python3, cuda10, cudnn7.6.4

Install venv:

```sh
python3 -m venv ~/gdlf_env
```

Install prerequisite:

```sh
source ~/gdlf_env/bin/activate
pip install -r requirements.txt
deactivate
```

We used Tensorflow 1.13.0 and this version generated some warning messages that (type, 1) is deprecated and The graph couldn't be sorted in topological order. 

If you downgrade Tensorflow to 1.10.0, you can get rid of these messages.

Create data and model folders:

```sh
mkdir ~/datasets
mkdir ~/ssl_saved_model
```

But if you want to use your preferred locations rather than the home directory, use the options (i.e., --data_root and --model_root) to pass these locations when executing codes.

## Usage

Note that since the code randomly selects a small labeled set and a testing set for each iteration, std is relatively greater. Thus, we recommend to use at least five iterations and compute the mean value of their results especially for a lower number of instances (i.e., 5-20 instances). Highly recommend to use 10 iterations to evaluate GANDaLF. 

Clone github repository:

```sh
git clone https://github.com/traffic-analysis/gandalf.git
```

Then, you are ready to execute the following codes.

The following codes first preprocess the input data including building labeled and unlabeled data as well as splitting them into training and testing sets. Then, they build the network to have the generator and discriminator followed by building the loss functions for the discriminator and generator. Note that the discriminator is trained using both supervised loss (i.e., categorical cross entropy) and unsupervised loss while the generator is trained using the feature matching loss computing MAE (or MSE) between fake and real traces. Finally both models are trained using two optimizers (i.e., Adam) and the trained model is evaluated using the testing set. For OW, we provide separate evaluation code. For more details regarding the code execution and data, refer to the following links: 

### WF-I closed-world experiments ([wfi/cw](https://github.com/traffic-analysis/gandalf/tree/main/wfi#cw-experiments-using-510205090-instances))

### WF-I open-world experiments ([wfi/ow](https://github.com/traffic-analysis/gandalf/tree/main/wfi#ow-training-using-20-instances))

### WF-S closed-world experiments ([wfs/cw](https://github.com/traffic-analysis/gandalf/tree/main/wfs#cw-experiments-using-510205090-instances))

### WF-S open-world experiments ([wfs/ow](https://github.com/traffic-analysis/gandalf/tree/main/wfs#ow-training-using-90-instances))

## Acknowledgments and References

We extends [this implementation](https://medium.com/@jos.vandewolfshaar/semi-supervised-learning-with-gans-23255865d0a4) to incorporate 1D CNN architecture and different input pipelining to afford AWF dataset much larger than MNIST.

This implementation is based on the following two techniques:

1. Feature Matching Loss by Salimans et al. Improved techniques for training gans. In Advances in Neural Information ProcessingSystems (NeurIPS), pages 2234–2242, 2016

2. Manifold Regularization by Lecouat et al. Semi-supervised learning with gans: Revisiting manifold regularization.arXiv preprint arXiv:1805.08957, 2018.

### Funding Acknowledgements
This work was funded by the National Science Foundation under Grants No. 1722743, 1816851, 1433736, and 1815757.

