<div>
    <h1 style="text-align: center;">Ten Steps to Keras</h1>
<div>
    <img style="text-align: left" src="https://blog.keras.io/img/keras-tensorflow-logo.jpg" width="30%" />
</div>
<div>
    <img style="text-align: left" src="imgs/conference_logo.png" width="30%" alt="Conference Logo" />
</div>
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
                <img src="imgs/gmail_small.png" style="display: inline-block;" /> 
                <a href="mailto:vmaggio@fbk.eu">vmaggio_at_fbk_dot_eu</a>
            </td>
       </tr>
  </tbody>
</table>

## Get the Materials

<img src="imgs/github.jpg" />

```

git clone https://github.com/leriomaggio/deep-learning-keras-tensorflow.git
git checkout tags/euroscipy2017
```

---

# Outline (in ten-ish notebooks)

1. _Multi-layer Fully Connected Networks (and the `backends`)_
2. _Hidden Layers features and Embeddings_
3. _Convolutional Networks_
4. _Hyperparameter Tuning_
5. _Cutsom Layers_
6. _Deep CNN and Residual Networks_
7. _Transfer Learning and Fine Tuning_
8. _Recursive Neural Networks_
9. _AutoEncoders_
10. _Multi-Modal Networks_

---

# Requirements

This tutorial requires the following packages:

- Python version 3.5
    - Python 3.4 should be fine as well
    - likely Python 2.7 would be also fine, but *who knows*? :P
    

- `numpy` version >= 1.12: http://www.numpy.org/
- `scipy` version >= 0.19: http://www.scipy.org/
- `matplotlib` version >= 2.0: http://matplotlib.org/
- `pandas` version >= 0.19: http://pandas.pydata.org
- `scikit-learn` version >= 0.18: http://scikit-learn.org
- `keras` version >= 2.0: http://keras.io
- `tensorflow` version 1.2: https://www.tensorflow.org
- `ipython`/`jupyter` version >= 6.0, with notebook support

(Optional but recommended):

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

    Python 3.5.4

---	
	
## Setting the Environment

The repository provides a `keras-tutorial.yml` file to simply re-create the Anaconda Python Environment, using `conda` [1]. 

To re-create the virtual environments:

```shell
conda env create -f keras-tutorial.yml
```

A new `keras-tutorial` conda environment will be created.  To activate the environment: 

```sheell
source activate keras-tutorial
```

[1]: _Note_:  The conda environment creation  has been tested on both Linux and OSX platforms. Therefore, hopefully, it should also work on Windows !-)

## Notes about Enabling GPU support for Theano and TensorFlow

#### Prerequisites: 

To enable GPU support for `theano` and `tensorflow`, it is mandatorily required that NVIDIA Drivers and CuDNN are **already** installed and configured 
before hand (having GPU cards physically installed in your hardware configuration is assumed and took for granted!). 

Please refer to the official  [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) documentation for further details.

#### Theano Configuration

##### Preamble

- `theano` package is assumed to be already installed, as it is provided inside the Anaconda Virtual Environment.
- To date, Theano only supports `cuDNN 5.1`. No support for `cuDNN 6` or `7` is still available. Therefore, be sure to download and install the proper version.

##### Configuring Theano

```shell
echo "[global]
device = cuda0
floatX = float32

[lib]
cnmem = 1.0" > ~/.theanorc
```

#### TensorFlow Configuration

To date, `tensorflow` is available in two different packages, namely `tensorflow` and `tensorflow-gpu`, whether you want to install 
the framework with CPU-only or GPU support, respectively.

For this reason,  if you want to enable GPU support for `tensorflow`, please be sure that the `keras-tutorial.yml` file has been properly **modified** to 
include `tensorflow-gpu==1.2.1` package (instead of the default `tensorflow==1.2.1`).

### Configure Keras with TensorFlow

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
>>> import keras
Using TensorFlow backend.
```

# Test if everything is up&running

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

    numpy: 1.12.1
    scipy: 0.19.1
    matplotlib: 2.0.2
    iPython: 6.1.0
    scikit-learn: 0.19.0



```python
import keras
print('keras: ', keras.__version__)

# optional
import theano
print('Theano: ', theano.__version__)

import tensorflow as tf
print('TensorFlow: ', tf.__version__)
```

    keras:  2.0.8
    Theano:  0.9.0
    TensorFlow:  1.2.1


<br>
<h1 style="text-align: center;">If everything worked till down here, you're ready to start!</h1>
