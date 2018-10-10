# Installing Deep Learning frameworks with GPU Support

This document is meant to be a (non-comprehensive) guide to
setup all the libraries needed to enable GPU computing on
**NVIDIA** cards, as well as installing the different
Deep Learning frameworks we will be using in this tutorial.

For a quick and lazy jump to the sections of interest,
please find the links below:

<a name="toc"></a>
## TOC:

- [Setup NVIDIA GPU](#nvidia)
- [Installing TensorFlow](#tf)
- [Installing Theano](#th)
- [Installing CNTK](#cntk)

---

<a name="nvidia"></a>
## Setup NVIDIA Dependencies and Libraries (on Linux OS)

This section is devoted to explain how to setup libraries
and dependencies provided by NVIDIA to enable Deep learning
frameworks on our GPU.

#### Step 0. CUDA Compute Capability

Before moving on, and download all the packages required, it is
paramount to verify that your GPU will **ever** support
any deep learning framework.
For this sake, NVIDIA published an (almost) comprehensive
list of GPU cards, along with their corresponding
**Compute Capability**.
This list is available
[**here**](https://developer.nvidia.com/cuda-gpus) on the NVIDIA Developer
Website.

Please checkout this list, and make sure that your GPU card has a
compute capability of at least **3.0**.

**IF** this is the case, please register to get an account
on the NVIDIA Developer [Website](https://developer.nvidia.com).

[top](#toc)

#### Step 1. Installing CUDA toolkit

The first NVIDIA software to setup is the **CUDA Toolkit**.
To do so, NVIDIA prepared specific repositories and instructions
for the different (hardware) platforms and operating systems.

Please download the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads),
accordingly.

For example, here is how to install **CUDA Toolkit 9.0**
(_not the very latest, but the one automatically supported by the latest
    TensorFlow 1.11 release_) on **Ubuntu 16.04**:

```shell

# Add NVIDIA package repository
$ wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.2.148-1_amd64.deb
$ sudo dpkg -i cuda-repo-ubuntu1604_9.2.148-1_amd64.deb
$ sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
$ sudo apt update

# Install CUDA and tools. Include optional NCCL 2.x
$ sudo apt-get -y install cuda-9.2 cuda-libraries-9-2
```

#### Step 2. Installing CuDNN library

The NVIDIA CUDA _Deep Neural Network library_ (cuDNN) is a GPU-accelerated
library of primitives for deep neural networks.
cuDNN provides highly tuned implementations for standard routines such as
forward and backward convolution, pooling, normalisation, and activation
layers.
cuDNN is part of the NVIDIA Deep Learning SDK and is **must-have** to enable
any Deep Learning framework.

To enable CuDNN support, please download, and install the package according to
the platform and operating system:
[CuDNN Download Page](https://developer.nvidia.com/rdp/cudnn-archive).

For example, here is how to install (_from source_) CuDNN on **Ubuntu 16.04**:

```shell
$ tar -zxf cudnn.tgz
$ cd cuda
$ sudo cp lib64/* /usr/local/cuda/lib64/
$ sudo cp include/* /usr/local/cuda/include/
```

#### Step 3. Installing NCCL library

The NVIDIA Collective Communications Library (NCCL) implements
multi-GPU and multi-node collective communication primitives that are
performance optimised for NVIDIA GPUs.
NCCL provides routines such as `all-gather`, `all-reduce`, `broadcast`,
`reduce`, `reduce-scatter`, that are optimised to achieve high bandwidth over
`PCIe` and `NVLink` high-speed interconnect.

To enable NCCL support, please download and install the `nccl` package
according to the platform, and the operating system:
[NCCL Download Page](https://developer.nvidia.com/nccl/nccl-download).

For example, here is how to install (_from source_) NCCL on **Ubuntu 16.04**:

```shell
$ tar xvf nccl_2.3.5-2+cuda9.2_x86_64.txz
$ cd nccl_2.3.5-2+cuda9.2_x86_64/
$ sudo cp ./NCCL-SLA.txt /usr/local/cuda/
$ sudo cp lib/* /usr/local/cuda/lib64/
$ sudo cp include/* /usr/local/cuda/include/
$ sudo cp -r lib/ /usr/local/cuda/
$ cd ..
```

#### Step 4. Export ENVIRONMENT Variables

To enable a system-wide support of CUDA Toolkit
(including shell commands), the following
environment variables have to be defined (e.g. on BASH):

```shell
$ export CUDA_HOME=/usr/local/cuda
$ export LD_LIBRARY_PATH=$CUDA_HOME/extras/CUPTI/lib64:$CUDA_HOME/lib64:/usr/local/lib:$LD_LIBRARY_PATH
$ export PATH=$CUDA_HOME/bin:$PATH
```

To avoid repeating the same instructions every time, those commands
may be saved into the `.bashrc` file on Linux (alt. `.bash_profile` on OSX):

```shell
# Write to Bashrc
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/extras/CUPTI/lib64:$CUDA_HOME/lib64:/usr/local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
```

#### Step 5. Test installation

If **all** the previous steps have been successfully completed,
the following commands may help you in checking if everything is _really_
up&running on your system:

```shell
# Check NVCC and CUDA Toolkit version
$ nvcc --version
```
You should get something like:
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2018 NVIDIA Corporation
Cuda compilation tools, release 9.2, V9.2.130
```

To get the version of CuDNN available on your system:
```shell
$ cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
```

You should get something like:
```
#define CUDNN_MAJOR 7
#define CUDNN_MINOR 2
#define CUDNN_PATCHLEVEL 1
```

[Back to top](#toc)

<a name="tf"></a>
## Installing TensorFlow (w/ GPU support)

To date `tensorflow` comes in two different packages, namely `tensorflow`
and `tensorflow-gpu`, whether you want to install
the framework with CPU-only or CPU/GPU support, respectively.

All the requirements files included in this repository
(namely `requirements.txt` and
`conda-environment.yml`) consider the `tensorflow` package, and so
with CPU-only support - assuming a more general and broader
audience.

**NOTE**: To effectively enable TensorFlow with GPU support,
NVIDIA Drivers and CuDNN **must** be installed and configured
beforehand.
Please refer to the official
[Tensorflow documentation](https://www.tensorflow.org/install/)
for further details, and/or the [Setup NVIDIA](#nvidia) section
for details.

Once all the dependencies and libraries have been installed,
TensorFlow can be installed either via `pip` or
directly from source.

```shell
pip install --ignore-installed tensorflow-gpu
```

#### Installing from source:

To install `tensorflow` from source, please refer to the
official documentation:
[Installing TF from Source](https://www.tensorflow.org/install/source)

**Note**: I would suggest installing TensorFlow directly from
source **only** in case the default `tf` requirements are not met on
your specific machine.

For instance, considering the latest **TensorFlow 1.11** release,
the pre-compiled Python wheels (w/ GPU support) have been
created considering the following settings:

```
- NVIDIA® GPU drivers — CUDA 9.0 requires 384.x or higher.
- CUDA® Toolkit — TensorFlow supports CUDA 9.0.
- cuDNN SDK (>= 7.2)
- NCCL 2.2 for multiple GPU support.
- Python 2.7, 3.3., 3.6
```

#### [TEST] Verify that TensorFlow has GPU support

```Python
>>> from tensorflow.python.client import device_lib
>>>
>>> def get_available_gpus():
...    local_device_protos = device_lib.list_local_devices()
...    return [x.name for x in local_device_protos if x.device_type == 'GPU']
...
>>> get_available_gpus()
['/device:GPU:0', '/device:GPU:1']
```

[Back to top](#toc)

<a name="th"></a>
## Installing Theano

The installation of Theano framework is optional and only needed to run
the introductory notebook(s) specifically using it.

If you want to install `theano` you can:

```shell

$ pip install theano

```

or if you are using Anaconda Python - and so `conda`:

```shell

$ conda install theano

```

#### Enabling GPU support with Theano

**NOTE**: Read this section **only** if after _pip installing_ `theano`,
it raises error in enabling the GPU support!

Since version `0.9` Theano introduced the
[`libgpuarray`](http://deeplearning.net/software/libgpuarray)
in the stable release (it was previously only available in the _development_ version).

The goal of `libgpuarray` is (_from the documentation_) make a common
GPU `ndarray` (_n dimensions array_) that can be reused by all projects
that is as future proof as possible, while keeping it easy to use for
simple need/quick test.

Here are some useful tips (hopefully) I came up with to properly install
and configure `theano` on (Ubuntu) Linux with **GPU** support:

1) [If you're using Anaconda] `conda install theano pygpu` should be just fine!

**However** in some cases I noted that

Sometimes it is suggested to install `pygpu` using the `conda-forge` channel:

`conda install -c conda-forge pygpu`

2) [Works with both Anaconda Python or Official CPython]

* Install `libgpuarray` from source: [Step-by-step install
`libgpuarray`
user library](http://deeplearning.net/software/libgpuarray/installation.html#step-by-step-install-user-library)

* Then, install `pygpu` from source: (in the same source folder)
`python setup.py build && python setup.py install`

* `pip install theano`.

#### Configure Theano:

After **Theano has been installed**, the `~/.theanorc` file
should be modified as reported below:

```
echo "[global]
device = cuda
floatX = float32

[lib]
cnmem = 1.0" > ~/.theanorc
```

***Note*** It might be needed to install `mpi4py` package
to enable MPI support with `theano` and `pygpu`.
To do so, it should be enough to just pip-installing it:

```shell
$ pip install mpi4py
```

#### [TEST] Verify that Theano has GPU support

```Python
>>> import theano
>>> print(theano.config.device)
cuda0
```

[Back to top](#toc)

<a name="cntk"></a>
## Installing CNTK

Similarly to `theano`, installing and configuring the *Microsoft Cognitive
Toolkit* (CNTK) is **optional** and only required whether you would
be interested in (a) switching to different Keras backend, and verify
that the same implementation works seemingly among multiple frameworks;
(b) running the only introductory notebook specifically focused on
showcasing the main features of CNTK.

That said, the very first important disclaimer I have to report is that
to date there is **no** available version of CNTK for Mac OSX.
So, if you have a Mac, the only option for you to run CNTK on
your machine (_to date_) is using a **Docker** container:
[https://docs.microsoft.com/en-us/cognitive-toolkit/CNTK-Docker-Containers]()

#### Installing CNTK via `pip`

CNTK comes with multiple wheels files available for several Python2 and
Python 3 versions, with both CPU and GPU support.
Similarly to TensorFlow, CNTK packages come in **two** different
flavours, depending on whether the GPU support should be enabled or not.

Therefore, to install CNTK with CPU-only support:

```shell
$ pip install cntk
```

To install CNTK with GPU support:
```shell
$ pip install cntk-gpus
```

The complete list of wheels files installable via `pip` is
available
[here](https://docs.microsoft.com/en-us/cognitive-toolkit/Setup-Linux-Python?tabs=cntkpy26#2-install-from-wheel-files).

**Note** Wheel files are very big, so a stable (Wi-FI) connection is
highly recommended when installing CNTK via `pip`.

#### Tips&Tricks and third-party libraries

In this section, I report the multiple Tips&Tricks I have discovered
along the way installing CNTK on my Linux server machine, aiming at
enabling the full set of features offered by the latest **CNTK 2.6**:

##### Open-MPI

It is highly recommend installing the Open MPI library.
Current CNTK Open MPI version requirement is at least `1.10`.

Please, check whether you have older version installations on your system and
if you do, either uninstall them or ensure (via, e.g. symbolic links) that
CNTK build procedure is using the required version.

Get the installation source:
```shell
$ wget https://www.open-mpi.org/software/ompi/v1.10/downloads/openmpi-1.10.3.tar.gz
```

Unpack, build and install Open MPI (to ``/usr/local/mpi` in this example):
```shell
$ tar -xzvf ./openmpi-1.10.3.tar.gz
$ cd openmpi-1.10.3
$ ./configure --prefix=/usr/local/mpi
$ make -j all
$ sudo make install
```

Finally, add the following environment variable to your current session and
your `.bashrc` profile:

```shell
export PATH=/usr/local/mpi/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/mpi/lib:$LD_LIBRARY_PATH
```

##### Open CV

You need to install OpenCV for `CNTK 2.3+`, if you want to build any of the
the following:
- CNTK Image Reader
- CNTK Image Writer - required to use Tensorboard's Image feature.

To enable Open CV, get the package and install it, using the
following commands:

```shell
$ wget https://github.com/Itseez/opencv/archive/3.1.0.zip
$ unzip 3.1.0.zip
$ cd opencv-3.1.0
$ mkdir release
$ cd release
$ cmake -D WITH_CUDA=OFF -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local/opencv-3.1.0 ..
$ make all
$ sudo make install
```

#### FIX `libstdc++.so.6`

If you encounter the following error when importing `cntk` (i.e. `import cntk`):
```shell
ImportError: <PATH>/libstdc++.so.6: version 'CXXABI_1.3.8' not found
```

here is how I fixed it: [GitHub Issue](https://github.com/Microsoft/CNTK/issues/2909)

(This solution consider using the `dl-keras-tf` as Conda Environment)

```shell
$ conda install libgcc
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/anaconda3/envs/dl-keras-tf/lib
$ ln -sf ~/anaconda3/envs/dl-keras-tf/lib/libfabric/libfabric.so.1 ~/anaconda3/envs/dl-keras-tf/lib
```

Importing CNTK now it should work:
```Python
>>> import cntk
```

#### [TEST] Verify that CNTK has GPU support

```Python
>>> from cntk.device import all_devices
>>> all_devices()
(GPU[0] GeForce GTX 1080Ti, GPU[1] GeForce GTX 1080Ti, CPU)
```


[Back to top](#toc)
