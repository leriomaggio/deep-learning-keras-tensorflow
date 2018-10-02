<div>
    <h1 style="text-align: center;">Deep Learning with Keras and Tensorflow</h1>
    <img style="text-align: left" src="https://blog.keras.io/img/keras-tensorflow-logo.jpg" width="15%" />
<div>
<br>

<img src="http://forge.fiware.org/plugins/mediawiki/wiki/fiware/images/thumb/4/46/FBK-Logo.png/707px-FBK-Logo.png"
    width="15%" title="Fondazione Bruno Kessler" alt="FBK Logo" />

<img src="https://mpba.fbk.eu/sites/mpba.fbk.eu/themes/fbkunit/logo-en.png" title="MPBA"
     width="30%" alt="MPBA Logo" />

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

git clone https://github.com/leriomaggio/deep-learning-keras-tensorflow.git -b fbk
```

---

# Outline at a glance

- **Part I**: **Artificial Neural Networks and Frameworks**

- **Part II**: **Supervised Learning**

- **Part III**: **Unsupervised Learning**

- **Part IV**: **Recurrent Neural Networks**

- **Part V**: **Generative Adversarial Networks**

- **Part VI**: **Extra**:  
   - Custom Layers in Keras
   - Multi modal Network Topologies with Keras
   - Multi-GPU Models
   - Distributed Training

---

# Requirements

This tutorial requires the following packages:

- Python version 3.6
    - Python 3.4+ should be fine as well
    - likely Python 2.7 would be also fine, but *who knows*? :P
- `numpy`: http://www.numpy.org/
- `scipy`: http://www.scipy.org/
- `matplotlib`: http://matplotlib.org/
- `pandas`: http://pandas.pydata.org
- `scikit-learn` : http://scikit-learn.org
- `keras`: http://keras.io
- `tensorflow`: https://www.tensorflow.org
- `jupyter` & `notebook`: http://jupyter.org

(Optional but recommended):

- `pyyaml`
- `hdf5` and `h5py` (required if you use model saving/loading
    functions in keras)
- **NVIDIA cuDNN** if you have NVIDIA GPUs on your machines.
    [https://developer.nvidia.com/rdp/cudnn-download]()

The easiest way to get (most of) these is to use an all-in-one installer
such as [Anaconda](https://www.anaconda.com/download/) from Continuum,
which is available for multiple computer platforms.

---

### Python Version

I'm currently running this tutorial with **Python 3** on **Anaconda**


```python
!python --version
```

    Python 3.6.6

---

# Setting the Environment

In this repository, files to install the required packages are provided.
The first step to setup the environment is to create a
Python [Virtual Environment](https://docs.python.org/3.6/tutorial/venv.html).

Whether you are using [Anaconda](https://www.anaconda.com/download/)
Python Distribution or the Standard
Python framework (from [python.org](https://www.python.org/downloads/)),
reported below are the instructions for the two cases, respectively.

## (a) Conda Environment

The repository includes a `conda-environment.yml` file that is necessary
to re-create the Conda environment required for the tutorial.

To re-create the virtual environments:

```shell
$ conda env create -f conda-environment.yml
```

## (b) `pyenv` & `virtualenv`

### 1. Installing `pyenv`

`pyenv` is a new package that lets you easily switch between multiple
versions of Python.
It's simple, unobtrusive, and follows the UNIX tradition of single-purpose
tools that do one thing well.

To **setup** `pyenv`, please follow the instructions available on
project [GitHub Repository](https://github.com/pyenv/pyenv)
depending on the specific platform and operating system.

There is a `pyenv` plugin named `pyenv-virtualenv` which comes with various
features to help `pyenv` users to manage virtual environments created by
`virtualenv` or Anaconda.

### 2. Installing `pyenv-virtualenv`

I'd recommend to install `pyenv-virtualenv` as reported in
the official [GitHub Repository](https://github.com/pyenv/pyenv-virtualenv).

### 3. Setting up the virtual environment

Once `pyenv` and `pyenv-virtualenv` have been correctly installed and
configured, these are the necessary set of instructions to execute to
set up the virtual environment for this tutorial:

```shell

$ pyenv install 3.6.6
$ pyenv virtualenv 3.6.6 dl-keras-tf
$ pyenv activate dl-keras-tf
$ pip install -r requirements.txt

```

### Note on installing TensorFlow

To date `tensorflow` comes in two different packages, namely `tensorflow`
and `tensorflow-gpu`, whether you want to install
the framework with CPU-only or CPU/GPU support, respectively.

All the requirements files (namely `requirements.txt` and
`conda-environment.yml`) consider the `tensorflow` package, and so
with CPU-only support.
If you want to enable GPU support, please consider to change the used
requirements file, accordingly:

#### Tensorflow for CPU only:

```shell
tensorflow
```

#### Tensorflow with GPU support:

```shell
tensorflow-gpu
```

**Note**: To effectively enable the GPU computing and support with
TensorFlow, NVIDIA Drivers and CuDNN **must** be installed and configured
before hand. Please refer to the official
[Tensorflow documentation](https://www.tensorflow.org/install/)
for further details.

## Configure Keras with TensorFlow

In this tutorial, we are going to use **Keras** with **TensorFlow**
backend.

To do so, the following configuration steps are required:

a) Create the `keras.json` (if it does not exist):

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

    numpy: 1.15.2
    scipy: 1.1.0
    matplotlib: 3.0.0
    iPython: 7.0.1
    scikit-learn: 0.20.0



```python
import keras
print('Keras: ', keras.__version__)

import tensorflow as tf
print('Tensorflow: ', tf.__version__)
```

    Keras:  2.2.3
    Tensorflow:  1.11.0


<br>
<h1 style="text-align: center;">If everything worked till down here, you're ready to start!</h1>
