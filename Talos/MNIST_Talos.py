'''
This is implementation of Talos for Hyperparameter optimization for MNIST

This implementation only works for Random and linear search. Bayesian Optimization not yet implemented.
Additionally, further errors with running optimizations with non integer training data.
Do not recommend the use of Talos for hyperparameter optimization
'''


import numpy as np
#import sherpa
np.random.seed(123)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Conv2D
from keras.utils import np_utils
from keras.datasets import mnist

import sys


import hyperio as hy
import pandas as pd

import talos as ta

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

def _model(x_train, y_train, x_val, y_val, params):
    model = Sequential()

    model.add(Convolution2D(filters=params['filters'], kernel_size=(3, 3), activation='relu', input_shape=(1, 28, 28),
                            data_format='channels_first'))

    model.add(Convolution2D(filters=params['filters'], kernel_size=(3, 3,), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(params['dropout_1']))

    model.add(Flatten())
    model.add(Dense(params['layers_1'], activation='relu'))
    model.add(Dropout(params['dropout_2']))
    model.add(Dense(params['layers_2'], activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    # metrics=['accuracy'])

    out = model.fit(x_train, y_train, batch_size=params['batch_size'], epochs=10, verbose=1,
                    validation_data=[X_test, Y_test])

    return out, model


p = {'filters': (20, 55, 1),
     'dropout_1': (0.05, 0.30, 0.05),
     'layers_1': (100, 158, 4),
     'dropout_2': (0.35, 0.60, 0.05),
     'layers_2': (5, 20, 2),
     'batch_size': (20, 40, 2),
     }

'''
Will throw error that value passed to parameter 'shape' has DataType float32 not in list of allowed values: int32,int64
Only use for models that 

'''
h = ta.Scan(X_train, Y_train, params=p, model=_model, grid_downsample=0.5, dataset_name='MNIST', experiment_no='1')

print(h.combinations)

