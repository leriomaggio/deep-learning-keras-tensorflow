# Outline (Draft)

- Part I: Introduction

    - Intro to ANN 
        - (naive pure-Python implementation from `pybrain`)
        - fast forward
        - sgd + backprop
    - Intro to Theano
        - Model + SGD with Theano
    - Introduction to Keras
        - Overview and main features
            - Theano backend
            - Tensorflow backend

- Part II: Supervised Learning + Keras Internals
    - Intro: Focus on Image Classification
    - Multi-Layer Perceptron and Fully Connected
        - Examples with `keras.models.Sequential` and `Dense`
    - Intro to CNN
        - meaning of convolutional filters
            - examples from ImageNet
        - Meaning of dimensions of Conv filters
    - Advanced CNN
        -  Dropout and MaxPooling
    - Famous ANN in Keras (likely moved somewhere else)
        - VGG16
        - VGG19
        - LaNet
        - Inception/GoogleNet
        - 
    
- Part III: Unsupervised Learning + Keras Internals
    - AutoEncoders
    - word2vec & doc2vec (gensim) + `keras.dataset` (i.e. `keras.dataset.imdb`)   

- Part IV: Advanced Materials
    - RNN (LSTM)
        -  RNN + CNN
    - Time Distributed Convolution 