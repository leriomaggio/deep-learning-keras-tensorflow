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
                <a href="http://twitter.com/leriomaggio" target="\_blank">@leriomaggio</a>
            </td>
        </tr>
        <tr style="border: 0px;">
            <td style="border: 0px;">
                <img src="imgs/linkedin_small.png" style="display: inline-block;" />
                <a href="it.linkedin.com/in/valeriomaggio" target="\_blank">valeriomaggio</a>
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
which is available for multiple computer platforms, namely Linux,
Windows, and OSX.

---

### Python Version

I'm currently running this tutorial with **Python 3** on **Anaconda**


```shell
$ python --version
Python 3.6.6
```

---

# Accessing the materials

If you want to access the materials, you have several options:

## Jupyter Notebook

All the materials in this tutorial are provided as a collection of
Jupyter Notebooks.
if don't know **what is** a Jupyter notebook, here is a good
reference for a quick introduction:
[Jupyter Notebook Beginner Guide](https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/what_is_jupyter.html).

On the other hand, if you also want to know (_and you should_) **what is NOT**
a Jupyter notebook - *spoiler alert:* **it is NOT an IDE** -
here is a very nice reference:

&rightarrow; [I Don't like Notebooks,](https://twitter.com/joelgrus/status/1033035196428378113)
by _Joel Grus_ @ JupyterCon 2018.

If you already have all the environment setup on your machine,
all you need to do is to run the Jupyter notebook server:

```shell
$ jupyter notebook
```

Alternatively, I suggest you to try the new **Jupyter Lab** environment:
```shell
$ jupyter lab
```

## Binder

(Consider this option only if your WiFi is stable)

If you don't want the hassle of setting up all the environment and
libraries on your machine, or simply you want to avoid doing
"_too much computation_" on your old-fashioned hardware setup,
I strongly suggest you to use the **Binder** service.

The primary goal of Binder is to turn a GitHub repo into a collection of
interactive Jupyter notebooks

To start using Binder, just click on the button below:
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/leriomaggio/deep-learning-keras-tensorflow/fbk)

## Google Colaboratory

[Colaboratory](https://colab.research.google.com/) is a free Jupyter
notebook environment that
requires no setup and runs entirely in the Google cloud.
Moreover, **GPU** and **TPU** runtime environments are available,
and completely for free.
[Here](https://colab.research.google.com/notebooks/welcome.ipynb)
is an overview of the main features offered by Colaboratory.

To run this tutorial using Colaboratory, my suggestion is
to (1) clone the repository on your local machine,
(2) upload the folder into your Google Drive folder,
(3) Open each notebook in Colab using Python3+GPU
running environment.

To start using Colaboratory, just click on the button below:
[![Colab](https://img.shields.io/badge/launch-colaboratory-yellow.svg)](https://colab.research.google.com/)

---

# Setting the Environment

In this repository, files to install the required packages are provided.
The first step to setup the environment is to create a
Python [Virtual Environment](https://docs.python.org/3.6/tutorial/venv.html).

Whether you are using [Anaconda](https://www.anaconda.com/download/)
Python Distribution or the **Standard
Python framework** (from [python.org](https://www.python.org/downloads/)),
below are reported the instructions for the two cases, respectively.

## (a) Conda Environment

This repository includes a `conda-environment.yml` file that is necessary
to re-create the Conda virtual environment.

To re-create the virtual environments:

```shell
$ conda env create -f conda-environment.yml
```

Then, to **activate** the virtual environment:

```shell
$ conda activate dl-keras-tf
```

## (b) `pyenv` & `virtualenv`

On the other hand, if you don't want to install (yet) another Python
distribution on your machine, or you prefer not to use the full-stack Anaconda
Python, I strongly suggest to give a try to the new `pyenv` project.

### 1. Setup `pyenv`

`pyenv` is a new package that lets you easily switch between multiple
versions of Python.
It is simple, unobtrusive, and follows the UNIX tradition of single-purpose
tools that do one thing well.

To **setup** `pyenv`, please follow the instructions reported on the
[GitHub Repository](https://github.com/pyenv/pyenv) of the project,
according to the specific platform and operating system.

There exists a `pyenv` plugin named `pyenv-virtualenv` which comes with various
features to help `pyenv` users to manage virtual environments created by
`virtualenv` or Anaconda.

### 2. Installing `pyenv-virtualenv`

I would recommend to install `pyenv-virtualenv` as reported in
the official
[documentation](https://github.com/pyenv/pyenv-virtualenv/blob/master/README.md).

### 3. Setting up the virtual environment

Once `pyenv` and `pyenv-virtualenv` have been correctly installed and
configured, these are the instructions to
set up the virtual environment for this tutorial:

```shell
$ pyenv install 3.6.6  # downloads and enables Python 3.6
$ pyenv virtualenv 3.6.6 dl-keras-tf  # create virtual env using Py3.6
$ pyenv activate dl-keras-tf  # activate the environment
$ pip install -r requirements.txt  # install requirements

```

### Configure Keras with TensorFlow

In this tutorial, we are going to use **Keras** with **TensorFlow**
backend.

To do so, the following configuration steps are required:

a) Create the `keras.json` (if it does not exist):

```shell
$ touch $HOME/.keras/keras.json
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

```shell
$ cat ~/.keras/keras.json
{
	"epsilon": 1e-07,
	"backend": "tensorflow",
	"floatx": "float32",
	"image_data_format": "channels_last"
}
```

### Notes on TensorFlow and GPU Computing

By default, the requirements files refer to the `tensorflow` package,
which corresponds to the TensorFlow library with **CPU-only** support.
This is because this (hardware) configuration is the _less demanding_
and the most _general purpose_, namely only **CPU** computing is
assumed.

**However** If you have a GPU, please refer to the specific
guide to enable the GPU computing environment to work with
the Deep Learning frameworks.

&rightarrow; [SETUP GPU and FRAMEWORKS](./SETUP_GPU_AND_FRAMEWORKS.md)

---

## Test if everything is up&running

### 1. Check import


```Python
>>> import numpy as np
>>> import scipy as sp
>>> import pandas as pd
>>> import matplotlib.pyplot as plt
>>> import sklearn
>>> import keras
Using TensorFlow backend.
```

### 2. Check installed Versions


```Python
>>> import numpy
>>> print('numpy:', numpy.__version__)
>>> import scipy
>>> print('scipy:', scipy.__version__)
>>> import matplotlib
>>> print('matplotlib:', matplotlib.__version__)
>>> import sklearn
>>> print('scikit-learn:', sklearn.__version__)
```
```
    numpy: 1.15.2
    scipy: 1.1.0
    matplotlib: 3.0.0
    scikit-learn: 0.20.0
```

```Python
>>> import keras
>>> print('Keras: ', keras.__version__)

>>> import tensorflow as tf
>>> print('TensorFlow: ', tf.__version__)
```
```
    Keras:  2.2.4
    Tensorflow:  1.11.0
```

<br>
<h2 style="text-align: center;">If everything worked till down here, you're ready to start!</h2>

# Loop back and Cross References

_Some mentions and reference this repository has gotten along the way_

- [FloydHub](https://www.floydhub.com):
[https://github.com/floydhub/deep-learning-keras-tensorflow]()

- Big Data: Distributed Data Management and Scalable Analytics
(INFOH515) - Université Libre de Bruxelles:
[https://github.com/Yannael/BigDataAnalytics_INFOH515]()

- Mentioned in
[Learning Deep Learning with Keras](http://p.migdal.pl/2017/04/30/teaching-deep-learning.html)
by Piotr Migdał

- Mention in
[Awesome Deep Learning](https://github.com/VinodPathak/Awesome-DeepLearning/blob/master/Keras.md)
