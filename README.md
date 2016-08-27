
<div>
    <h1 style="text-align: center;">Deep Learning with Keras</h1>
    <img style="text-align: left" src="imgs/keras-logo-small.jpg" width="10%" />
<div>

<div>
    <h2 style="text-align: center;">Tutorial @ EuroScipy 2016</h2>
    <img style="text-align: left" src="imgs/euroscipy_2016_logo.png" width="40%" />
</div>    

##### Yam Peleg,  Valerio Maggio

# Goal of this Tutorial

- **Introduce** main features of Keras
- **Learn** how simple and Pythonic is doing Deep Learning with Keras
- **Understand** how easy is to do basic and *advanced* DL models in Keras;
    - **Examples and Hand-on Excerises** along the way.

## Source

https://github.com/leriomaggio/deep-learning-keras-euroscipy2016/

---

# (Tentative) Schedule 

## Attention: Spoilers Warning!


- **Setup** (`10 mins`)

- **Part I**: **Introduction** (`~65 mins`)

    - Intro to ANN (`~20 mins`)
        - naive pure-Python implementation
        - fast forward, sgd, backprop
        
    - Intro to Theano (`15 mins`)
        - Model + SGD with Theano
        
    - Introduction to Keras (`30 mins`)
        - Overview and main features
            - Theano backend
            - Tensorflow backend
        - Multi-Layer Perceptron and Fully Connected
            - Examples with `keras.models.Sequential` and `Dense`
            - HandsOn: MLP with keras
            
- **Coffe Break** (`30 mins`)

- **Part II**: **Supervised Learning and Convolutional Neural Nets** (`~45 mins`)
    
    - Intro: Focus on Image Classification (`5 mins`)

    - Intro to CNN (`25 mins`)
        - meaning of convolutional filters
            - examples from ImageNet    
        - Meaning of dimensions of Conv filters (through an exmple of ConvNet) 
        - Visualising ConvNets
        - HandsOn: ConvNet with keras 

    - Advanced CNN (`10 mins`)
        - Dropout
        - MaxPooling
        - Batch Normalisation
        
    - Famous Models in Keras (likely moved somewhere else) (`10 mins`)
        (ref: https://github.com/fchollet/deep-learning-models)
            - VGG16
            - VGG19
            - ResNet50
            - Inception v3
        - HandsOn: Fine tuning a network on new dataset 
        
- **Part III**: **Unsupervised Learning** (`10 mins`)

    - AutoEncoders (`5 mins`)
    - word2vec & doc2vec (gensim) & `keras.datasets` (`5 mins`)
        - `Embedding`
        - word2vec and CNN
    - Exercises

- **Part IV**: **Advanced Materials** (`20 mins`)
    - RNN and LSTM (`10 mins`)
        -  RNN, LSTM, GRU  
    - Example of RNN and LSTM with Text (`~10 mins`) -- *Tentative*
    - HandsOn: IMDB

- **Wrap up and Conclusions** (`5 mins`)

---

# Requirements

This tutorial requires the following packages:

- Python version 3.4+ 
    - likely Python 2.7 would be fine, but *who knows*? :P
- `numpy` version 1.10 or later: http://www.numpy.org/
- `scipy` version 0.16 or later: http://www.scipy.org/
- `matplotlib` version 1.4 or later: http://matplotlib.org/
- `pandas` version 0.16 or later: http://pandas.pydata.org
- `scikit-learn` version 0.15 or later: http://scikit-learn.org
- `keras` version 1.0 or later: http://keras.io
- `theano` version 0.8 or later: http://deeplearning.net/software/theano/
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


# How to set up your environment

The quickest and simplest way to setup the environment is to use [conda](https://store.continuum.io) environment manager. 

We provide in the materials a `deep-learning.yml` that is complete and **ready to use** to set up your virtual environment with conda.


```python
!cat deep-learning.yml
```

    name: deep-learning
    channels:
    - conda-forge
    - defaults
    dependencies:
    - accelerate=2.3.0=np111py35_3
    - accelerate_cudalib=2.0=0
    - bokeh=0.12.1=py35_0
    - cffi=1.6.0=py35_0
    - backports.shutil_get_terminal_size=1.0.0=py35_0
    - blas=1.1=openblas
    - ca-certificates=2016.8.2=3
    - cairo=1.12.18=8
    - certifi=2016.8.2=py35_0
    - cycler=0.10.0=py35_0
    - cython=0.24.1=py35_0
    - decorator=4.0.10=py35_0
    - entrypoints=0.2.2=py35_0
    - fontconfig=2.11.1=3
    - freetype=2.6.3=1
    - gettext=0.19.7=1
    - glib=2.48.0=4
    - h5py=2.6.0=np111py35_6
    - harfbuzz=1.0.6=0
    - hdf5=1.8.17=2
    - icu=56.1=4
    - ipykernel=4.3.1=py35_1
    - ipython=5.1.0=py35_0
    - ipywidgets=5.2.2=py35_0
    - jinja2=2.8=py35_1
    - jpeg=9b=0
    - jsonschema=2.5.1=py35_0
    - jupyter_client=4.3.0=py35_0
    - jupyter_console=5.0.0=py35_0
    - jupyter_core=4.1.1=py35_1
    - libffi=3.2.1=2
    - libiconv=1.14=3
    - libpng=1.6.24=0
    - libsodium=1.0.10=0
    - libtiff=4.0.6=6
    - libxml2=2.9.4=0
    - markupsafe=0.23=py35_0
    - matplotlib=1.5.2=np111py35_6
    - mistune=0.7.3=py35_0
    - nbconvert=4.2.0=py35_0
    - nbformat=4.0.1=py35_0
    - ncurses=5.9=8
    - nose=1.3.7=py35_1
    - notebook=4.2.2=py35_0
    - numpy=1.11.1=py35_blas_openblas_201
    - openblas=0.2.18=4
    - openssl=1.0.2h=2
    - pandas=0.18.1=np111py35_1
    - pango=1.40.1=0
    - path.py=8.2.1=py35_0
    - pcre=8.38=1
    - pexpect=4.2.0=py35_1
    - pickleshare=0.7.3=py35_0
    - pip=8.1.2=py35_0
    - pixman=0.32.6=0
    - prompt_toolkit=1.0.6=py35_0
    - protobuf=3.0.0b3=py35_1
    - ptyprocess=0.5.1=py35_0
    - pygments=2.1.3=py35_1
    - pyparsing=2.1.7=py35_0
    - python=3.5.2=2
    - python-dateutil=2.5.3=py35_0
    - pytz=2016.6.1=py35_0
    - pyyaml=3.11=py35_0
    - pyzmq=15.4.0=py35_0
    - qt=4.8.7=0
    - qtconsole=4.2.1=py35_0
    - readline=6.2=0
    - requests=2.11.0=py35_0
    - scikit-learn=0.17.1=np111py35_blas_openblas_201
    - scipy=0.18.0=np111py35_blas_openblas_201
    - setuptools=25.1.6=py35_0
    - simplegeneric=0.8.1=py35_0
    - sip=4.18=py35_0
    - six=1.10.0=py35_0
    - sqlite=3.13.0=1
    - terminado=0.6=py35_0
    - tk=8.5.19=0
    - tornado=4.4.1=py35_1
    - traitlets=4.2.2=py35_0
    - wcwidth=0.1.7=py35_0
    - wheel=0.29.0=py35_0
    - widgetsnbextension=1.2.6=py35_3
    - xz=5.2.2=0
    - yaml=0.1.6=0
    - zeromq=4.1.5=0
    - zlib=1.2.8=3
    - cudatoolkit=7.5=0
    - ipython_genutils=0.1.0=py35_0
    - jupyter=1.0.0=py35_3
    - libgfortran=3.0.0=1
    - llvmlite=0.11.0=py35_0
    - mkl=11.3.3=0
    - mkl-service=1.1.2=py35_2
    - numba=0.26.0=np111py35_0
    - pycparser=2.14=py35_1
    - pyqt=4.11.4=py35_4
    - snakeviz=0.4.1=py35_0
    - pip:
      - backports.shutil-get-terminal-size==1.0.0
      - certifi==2016.8.2
      - cycler==0.10.0
      - cython==0.24.1
      - decorator==4.0.10
      - h5py==2.6.0
      - ipykernel==4.3.1
      - ipython==5.1.0
      - ipython-genutils==0.1.0
      - ipywidgets==5.2.2
      - jinja2==2.8
      - jsonschema==2.5.1
      - jupyter-client==4.3.0
      - jupyter-console==5.0.0
      - jupyter-core==4.1.1
      - keras==1.0.7
      - mako==1.0.4
      - markupsafe==0.23
      - matplotlib==1.5.2
      - mistune==0.7.3
      - nbconvert==4.2.0
      - nbformat==4.0.1
      - nose==1.3.7
      - notebook==4.2.2
      - numpy==1.11.1
      - pandas==0.18.1
      - path.py==8.2.1
      - pexpect==4.2.0
      - pickleshare==0.7.3
      - pip==8.1.2
      - prompt-toolkit==1.0.6
      - protobuf==3.0.0b2
      - ptyprocess==0.5.1
      - pygments==2.1.3
      - pygpu==0.2.1
      - pyparsing==2.1.7
      - python-dateutil==2.5.3
      - pytz==2016.6.1
      - pyyaml==3.11
      - pyzmq==15.4.0
      - qtconsole==4.2.1
      - requests==2.11.0
      - scikit-learn==0.17.1
      - scipy==0.18.0
      - setuptools==25.1.4
      - simplegeneric==0.8.1
      - six==1.10.0
      - tensorflow==0.10.0rc0
      - terminado==0.6
      - theano==0.8.2
      - tornado==4.4.1
      - traitlets==4.2.2
      - wcwidth==0.1.7
      - wheel==0.29.0
      - widgetsnbextension==1.2.6
    prefix: /home/valerio/anaconda3/envs/deep-learning
    


# Recreate the Conda Environment

#### A. Create the Environment

```
conda env create -f deep-learning.yml  # this file is for Linux channels.
```

If you're using a **Mac OSX**, we also provided in the repo the conda file 
that is compatible with `osx-channels`:

```
conda env create -f deep-learning-osx.yml  # this file is for OSX channels.
```

#### B. Activate the new `deep-learning` Environment

```
source activate deep-learning
```

## Optionals

### 1. Enabling Conda-Forge

It is strongly suggested to enable [**conda forge**](https://conda-forge.github.io/) in your Anaconda installation.

**Conda-Forge** is a github organisation containing repositories of conda recipies.

To add `conda-forge` as an additional anaconda channel it is just required to type:

```shell
conda config --add channels conda-forge
```

### 2. Configure Theano

1) Create the `theanorc` file:

```shell
touch $HOME/.theanorc
```

2) Copy the following content into the file:

```
[global]
floatX = float32
device = gpu  # switch to cpu if no GPU is available on your machine

[nvcc]
fastmath = True

[lib]
cnmem=.90
```

**More on [theano documentation](http://theano.readthedocs.io/en/latest/library/config.html)**

### 3. Installing Tensorflow as backend 

```shell
# Ubuntu/Linux 64-bit, GPU enabled, Python 3.5
# Requires CUDA toolkit 7.5 and CuDNN v4. For other versions, see "Install from sources" below.
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.10.0rc0-cp35-cp35m-linux_x86_64.whl

pip install --ignore-installed --upgrade $TF_BINARY_URL
```

**More on [tensorflow documentation](https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html)**

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

    Using Theano backend.
    Using gpu device 0: GeForce GTX 760 (CNMeM is enabled with initial size: 90.0% of memory, cuDNN 4007)


## 2. Check installeded Versions


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
    scikit-learn: 0.17.1



```python
import keras
print('keras: ', keras.__version__)

import theano
print('Theano: ', theano.__version__)

# optional
import tensorflow as tf
print('Tensorflow: ', tf.__version__)
```

    keras:  1.0.7
    Theano:  0.8.2
    Tensorflow:  0.10.0rc0


<br>
<h1 style="text-align: center;">If everything worked till down here, you're ready to start!</h1>

---


# Consulting Material

You have two options to go through the material presented in this tutorial:

* Read (and execute) the material as **iPython/Jupyter** notebooks
* (just) read the material as (HTML) slides

In the first case, all you need to do is just execute `ipython notebook` (or `jupyter notebook`) depending on the version of `iPython` you have installed on your machine

(`jupyter` command works in case you have `iPython 4.0.x` installed)

In the second case, you may simply convert the provided notebooks in `HTML` slides and see them into your browser
thanks to `nbconvert`.

Thus, move to the folder where notebooks are stored and execute the following command:

    jupyter nbconvert --to slides ./*.ipynb --post serve
    
   
(Please substitute `jupyter` with `ipython` in the previous command if you have `iPython 3.x` installed on your machine)

## In case...

..you wanna do **both** (interactive and executable slides), I highly suggest to install the terrific `RISE` ipython notebook extension: [https://github.com/damianavila/RISE](https://github.com/damianavila/RISE)
