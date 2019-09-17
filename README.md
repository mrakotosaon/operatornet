# OperatorNet
This is our implementation of OperatorNet, a network that reconstructs shapes from shape difference operators.


![OperatorNet](https://raw.githubusercontent.com/mrakotosaon/operatornet/master/images/operatornet.png "OperatorNet")

This code was written by Ruqi Huang and Marie-Julie Rakotosaona.

## Prerequisites
* CUDA and CuDNN (changing the code to run on CPU should require few changes)
* Python 2.7
* Tensorflow

## Setup
Install required python packages, if they are not already installed:
``` bash
pip install numpy
pip install plyfile
```

Clone this repository:
``` bash
git clone https://github.com/mrakotosaon/operatornet.git
cd operatornet
```
## Shape difference operators

To generate demo shape difference matrices: run demo_compute_shape_diff.m with Matlab.

## Models

Download pretrained models:
``` bash
cd models
python download_models.py
```

 ## Data

A demo dataset can be found here: https://nuage.lix.polytechnique.fr/index.php/s/BqiX5rcWszkKT9N

It contains shape differences and labels as Matlab matrices. This dataset is a simplified version of the one used in the paper.


Download pretrained models:
``` bash
cd Data
python download_demo_data.py
```


## Training
To train OperatorNet with the default settings and demo data:
``` bash
python train.py
```

## Test
To test OperatorNet with the default settings and demo shapes:
``` bash
python test.py
```
Two shapes get reconstructed and inteprolated.

## Citation
If you use our work, please cite our paper.
```
@article{huang2019operatornet,
  title={OperatorNet: Recovering 3D Shapes From Difference Operators},
  author={Huang, Ruqi and Rakotosaona, Marie-Julie and Achlioptas, Panos and Guibas, Leonidas and Ovsjanikov, Maks},
  journal={ICCV},
  year={2019}
}
```
