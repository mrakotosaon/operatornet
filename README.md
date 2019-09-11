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
```


Clone this repository:
``` bash
git clone https://github.com/mrakotosaon/operatornet.git
cd operatornet
```


Download pretrained models:
``` bash
cd models
python download_models.py
```

 ## Data

Our data can be found here:

It contains the following files:



## Training
To train OperatorNet with the default settings:
``` bash
python train_operatornet.py
```

## Citation
If you use our work, please cite our paper.
```
@article{huang2019operatornet,
  title={OperatorNet: Recovering 3D Shapes From Difference Operators},
  author={Huang, Ruqi and Rakotosaona, Marie-Julie and Achlioptas, Panos and Guibas, Leonidas and Ovsjanikov, Maks},
  journal={arXiv preprint arXiv:1904.10754},
  year={2019}
}
```
