<div>
    <h1 style="text-align: center;">Deep Learning with Keras and Tensorflow</h1>
    <img style="text-align: left" src="https://blog.keras.io/img/keras-tensorflow-logo.jpg" width="30%" />
    <img style="text-align: left" src="imgs/conference_logo.png" width="30%" alt="Conference Logo" />
<div>
<br>

# Author: Valerio Maggio

### _PostDoc Data Scientist @ FBK/MPBA_

## Contacts:

<table style="border: 0px; display: inline-table">
    <tbody>
        <tr style="border: 0px;">
            <td style="border: 0px;">
                <img src="imgs/twitter_small.png" style="display: inline-block;" /> 
                <a href="http://twitter.com/leriomaggio" target="_blank">@leriomaggio</a>
            </td>
            <td style="border: 0px;">
                <img src="imgs/gplus_small.png" style="display: inline-block;" /> 
                <a href="http://plus.google.com/+ValerioMaggio" target="_blank">+ValerioMaggio</a>
            </td>
        </tr>
        <tr style="border: 0px;">
            <td style="border: 0px;">
                <img src="imgs/linkedin_small.png" style="display: inline-block;" /> 
                <a href="it.linkedin.com/in/valeriomaggio" target="_blank">valeriomaggio</a>
            </td>
            <td style="border: 0px;">
                <img src="imgs/gmail_small.png" style="display: inline-block;" /> 
                <a href="mailto:vmaggio@fbk.eu">vmaggio_at_fbk_dot_eu</a>
            </td>
       </tr>
  </tbody>
</table>


```shell

git clone -b pydatait 
https://github.com/leriomaggio/deep-learning-keras-tensorflow.git
```

---

# Outline at a glance

- **Warmup**

- **Part I**: **Introduction**

    - Intro to ANN
        - naive pure-Python implementation
        - fast forward, sgd, backprop
        
    - Intro to Tensorflow 
        - Model + SGD with Tensorflow
            
    - Introduction to Keras
        - Overview and main features
            - Tensorflow backend
            - Theano backend
            - Keras Backend
            - Overview of the main layers
        - Multi-Layer Perceptron and Fully Connected
            - Examples with `keras.models.Sequential` and `Dense`
            - HandsOn: FC with keras
    
    - Extra Material:
        - Intro to Theano
        - Alternative ANN implementation for MNIST
    
            
- **Break**

- **Part II**: **Supervised Learning and Convolutional Neural Nets**
    
    - Intro: Focus on Image Classification

    - Intro to ConvNets
        - meaning of convolutional filters
            - examples from ImageNet    
        - Meaning of dimensions of Conv filters (through an exmple of ConvNet) 
        - Visualising ConvNets
        - HandsOn: ConvNet with keras 

    - Advanced CNN
        - Dropout
        - MaxPooling
        - Batch Normalisation
        
    - Famous Models in Keras
        (ref: `keras.applications`)
            - VGG16
            - VGG19
            - ResNet50
            - Inception v3
        - Transfer Learning
        - HandsOn: Fine tuning a network on new dataset 
        
- **Part III**: **Unsupervised Learning**

    - AutoEncoders (`5 mins`)
    - word2vec & doc2vec (gensim) & `keras.datasets`
        - `Embedding`
        - word2vec and CNN
    - Exercises

- **Part IV**: **Advanced Materials**
    - RNN and LSTM (`10 mins`)
        -  RNN, LSTM, GRU  
    - Example of RNN and LSTM with Text
    - HandsOn: IMDB
    - **Multi-Input/Multi-Output Network Topologies**

- **Wrap up and Conclusions**

---

# Requirements

This tutorial requires the following packages:

- Python version 3.5
    - Python 3.4 should be fine as well
    - likely Python 2.7 would be also fine, but *who knows*? :P
    
- `numpy` version 1.10 or later: http://www.numpy.org/
- `scipy` version 0.16 or later: http://www.scipy.org/
- `matplotlib` version 1.4 or later: http://matplotlib.org/
- `pandas` version 0.16 or later: http://pandas.pydata.org
- `scikit-learn` version 0.15 or later: http://scikit-learn.org
- `keras` version 2.0 or later: http://keras.io
- `tensorflow` version 1.0 or later: https://www.tensorflow.org
- `ipython`/`jupyter` version 4.0 or later, with notebook support

(Optional but recommended):

- `pyyaml`
- `hdf5` and `h5py` (required if you use model saving/loading functions in keras)
- **NVIDIA cuDNN** if you have NVIDIA GPUs on your machines.
    [https://developer.nvidia.com/rdp/cudnn-download]()

The easiest way to get (most) these is to use an all-in-one installer such as [Anaconda](http://www.continuum.io/downloads) from Continuum. These are available for multiple architectures.

---

### Python Version

I'm currently running this tutorial with **Python 3** on **Anaconda**


```python
!python --version
```

    Python 3.5.2

---	
	
## Setting the Environment

In this repository, files to re-create virtual env with `conda` are provided for Linux and OSX systems, 
namely `deep-learning.yml` and `deep-learning-osx.yml`, respectively.

To re-create the virtual environments (on Linux, for example):

```shell
conda env create -f deep-learning.yml
```

For OSX, just change the filename, accordingly.

### Installing Tensorflow

To date `tensorflow` comes in two different packages, namely `tensorflow` and `tensorflow-gpu`, whether you want to install 
the framework with CPU-only or GPU support, respectively.

For this reason, `tensorflow` has **not** been included in the conda envs and has to be installed separately.

#### Tensorflow for CPU only:

```shell
pip install tensorflow
```

#### Tensorflow with GPU support:

```shell
pip install tensorflow-gpu
```

**Note**: NVIDIA Drivers and CuDNN **must** be installed and configured before hand. Please refer to the official 
[Tensorflow documentation](https://www.tensorflow.org/install/) for further details.


#### Important Note:

All the code provided+ in this tutorial can run even if `tensorflow` is **not** installed, and so using `theano` as the (default) backend!

___**This** is exactly the power of Keras!___

Therefore, installing `tensorflow` is **not** stricly required!

+: Apart from the **1.2 Introduction to Tensorflow** tutorial, of course.

### Configure Keras with tensorflow

By default, Keras is configured with `theano` as backend. 

If you want to use `tensorflow` instead, these are the simple steps to follow:

1) Create the `keras.json` (if it does not exist):

```shell
touch $HOME/.keras/keras.json
```

2) Copy the following content into the file:

```
{
    "epsilon": 1e-07,
    "backend": "tensorflow",
    "floatx": "float32",
    "image_data_format": "channels_last"
}
```

3) Verify it is properly configured:

```python
!cat ~/.keras/keras.json
```

    {
    	"epsilon": 1e-07,
    	"backend": "tensorflow",
    	"floatx": "float32",
    	"image_data_format": "channels_last"
    }

---

# Test if everything is up&running

## 1. Check import


```python
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
```


```python
import keras
```

    Using TensorFlow backend.


## 2. Check installed Versions


```python
import numpy
print('numpy:', numpy.__version__)

import scipy
print('scipy:', scipy.__version__)

import matplotlib
print('matplotlib:', matplotlib.__version__)

import IPython
print('iPython:', IPython.__version__)

import sklearn
print('scikit-learn:', sklearn.__version__)
```

    numpy: 1.11.1
    scipy: 0.18.0
    matplotlib: 1.5.2
    iPython: 5.1.0
    scikit-learn: 0.18



```python
import keras
print('keras: ', keras.__version__)

# optional
import theano
print('Theano: ', theano.__version__)

import tensorflow as tf
print('Tensorflow: ', tf.__version__)
```

    keras:  2.0.2
    Theano:  0.9.0
    Tensorflow:  1.0.1


<br>
<h1 style="text-align: center;">If everything worked till down here, you're ready to start!</h1>
