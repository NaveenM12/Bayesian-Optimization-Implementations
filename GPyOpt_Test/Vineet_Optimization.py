# coding: utf-8

# # Bayesian Optimization on Keras

# ### MNIST training on Keras with Bayesian optimization
# * This notebook runs MNIST training on Keras using Bayesian optimization to find the best hyper parameters.
# * The MNIST model here is just a simple one with one input layer, one hidden layer and one output layer, without convolution.
# * Hyperparameters of the model include the followings:
# * - number of convolutional layers in first layer
# * - dropout rate of first layer
# * - number of convolutional layers in second layer
# * - dropout rate of second layer
# * - number of units in the Dense Layer
# * - dropout rate of the third layer
# * - batch size
# * - epochs
# * I used GPy and GPyOpt to run Bayesian optimization.


# #### Import libraries

# In[1]:
from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, BatchNormalization, Dense
from keras.models import Sequential

import GPy, GPyOpt
import numpy as np
import pandas as pds
import random

from keras.datasets import mnist
from keras.metrics import categorical_crossentropy
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

import keras

import os


# #### Define MNIST model
# * includes data loading function, training function, fit function and evaluation function

# In[2]:


# MNIST class
class Vineet_Optimization():

    def __init__(self, l1_conv_layers=32, l1_dropout=0.25,
                 l2_conv_layers=64,
                 l2_dropout=0.25,
                 num_units =512 ,
                 l3_dropout=0.5,
                 batch_size=32,):
                 #epochs=5):
        self.l1_conv_layers = l1_conv_layers
        self.l1_dropout = l1_dropout
        self.l2_conv_layers = l2_conv_layers
        self.l2_dropout = l2_dropout
        self.num_units = num_units
        self.l3_dropout = l3_dropout
        self.batch_size = batch_size
        self.epochs = 10
        #self.validation_split = validation_split
        self.num_classes = 10
        self.num_predictions = 20
        self.data_augmentation = True
        self.save_dir = os.path.join(os.getcwd(), 'saved_models')
        self.model_name = 'keras_cifar10_trained_model.h5'

        self.__x_train, self.__x_test, self.__y_train, self.__y_test = self.cifar10_data()

        self.__model = self._model()

    # load mnist data from keras dataset
    def cifar10_data(self):

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        return x_train, x_test, y_train, y_test

    def _model(self):
        model = Sequential()
        model.add(Conv2D(self.l1_conv_layers, (3, 3), padding='same',
                         input_shape=self.__x_train.shape[1:]))
        model.add(Activation('relu'))
        model.add(Conv2D(self.l1_conv_layers, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(self.l1_dropout))

        model.add(Conv2D(self.l2_conv_layers, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(self.l2_conv_layers, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(self.l1_dropout))

        model.add(Flatten())
        model.add(Dense(self.num_units))
        model.add(Activation('relu'))
        model.add(Dropout(self.l3_dropout))
        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))

        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])

        return model

    # fit mnist model
    def _fit(self):

        if not self.data_augmentation:
            print('Not using data augmentation.')
            self.__model.fit(self.__x_train, self.__y_train,
                             batch_size=self.batch_size,
                             epochs=self.epochs,
                             validation_data=(self.__x_test, self.__y_test),
                             shuffle=True)
        else:
            print('Using real-time data augmentation.')
            # This will do preprocessing and realtime data augmentation:
            datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                zca_epsilon=1e-06,  # epsilon for ZCA whitening
                rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                # randomly shift images horizontally (fraction of total width)
                width_shift_range=0.1,
                # randomly shift images vertically (fraction of total height)
                height_shift_range=0.1,
                shear_range=0.,  # set range for random shear
                zoom_range=0.,  # set range for random zoom
                channel_shift_range=0.,  # set range for random channel shifts
                # set mode for filling points outside the input boundaries
                fill_mode='nearest',
                cval=0.,  # value used for fill_mode = "constant"
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False,  # randomly flip images
                # set rescaling factor (applied before any other transformation)
                rescale=None,
                # set function that will be applied on each input
                preprocessing_function=None,
                # image data format, either "channels_first" or "channels_last"
                data_format=None,
                # fraction of images reserved for validation (strictly between 0 and 1)
                validation_split=0.0)

            datagen.fit(self.__x_train)
            self.__model.fit_generator(datagen.flow(self.__x_train, self.__y_train,
                                             batch_size=self.batch_size),
                                epochs=self.epochs,
                                validation_data=(self.__x_test, self.__y_test),
                                workers=4)



        '''
        self.__model.fit(self.__x_train, self.__y_train,
                         batch_size=self.batch_size,
                         epochs=self.epochs,
                         verbose=0,
                         validation_split=self.validation_split,
                         callbacks=[early_stopping])
                         '''

    # evaluate mnist model
    def _evaluate(self):
        self._fit()

        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        model_path = os.path.join(self.save_dir, self.model_name)
        self.__model.save(model_path)
        print('Saved trained model at %s ' % model_path)

        evaluation = self.__model.evaluate(self.__x_test, self.__y_test, batch_size=self.batch_size, verbose=1)
        return evaluation


# #### Runner function for the MNIST model

# In[3]:


# function to run mnist class
def run_(l1_conv_layers=32, l1_dropout=0.25,
                 l2_conv_layers=64,
                 l2_dropout=0.25,
                 num_units =512 ,
                 l3_dropout=0.5,
                 batch_size=32,):
                 #epochs=5):
    _optimize = Vineet_Optimization(l1_conv_layers=l1_conv_layers, l1_dropout=l1_dropout,
                                 l2_conv_layers=l2_conv_layers, l2_dropout=l2_dropout,
                                 num_units=num_units, l3_dropout=l3_dropout,
                                 batch_size=batch_size,) #epochs=epochs)
    _evaluation = _optimize._evaluate()
    return _evaluation

# bounds for hyper-parameters in  model
# the bounds dict should be in order of continuous type and then discrete type
bounds = [{'name': 'l1_dropout', 'type': 'continuous', 'domain': (0.0, 0.3)},
          {'name': 'l2_dropout', 'type': 'continuous', 'domain': (0.0, 0.3)},
          {'name': 'l3_dropout', 'type': 'continuous', 'domain': (0.0, 0.6)},
          {'name': 'l1_conv_layers', 'type': 'discrete', 'domain': (10, 16, 32, 64, 128)},
          {'name': 'l2_conv_layers', 'type': 'discrete', 'domain': (32, 64, 86, 104, 128)},
          {'name': 'num_units', 'type': 'discrete', 'domain': (400, 450, 512, 560, 600)},
          {'name': 'batch_size', 'type': 'discrete', 'domain': (10, 32, 64, 100)},]
         # {'name': 'epochs', 'type': 'discrete', 'domain': (5, 10, 20)}]


# #### Bayesian Optimization

# In[5]:


# function to optimize mnist model
def f(x):
    print(x)
    evaluation = run_(
        l1_dropout=float(x[:, 0]),
        l2_dropout=float(x[:, 1]),
        l3_dropout=float(x[:, 2]),
        l1_conv_layers=int(x[:, 3]),
        l2_conv_layers=int(x[:, 4]),
        num_units=int(x[:, 5]),
        batch_size=int(x[:, 6]),)
       # epochs=int(x[:, 7]))
    print("LOSS:\t{0} \t ACCURACY:\t{1}".format(evaluation[0], evaluation[1]))
    print(evaluation)
    return evaluation[0]


# #### Optimizer instance

# In[6]:


# optimizer
opt_mnist = GPyOpt.methods.BayesianOptimization(f=f, domain=bounds)

# #### Running optimization

# In[7]:


# optimize mnist model
opt_mnist.run_optimization(max_iter=10)

# #### The output

# In[20]:


# print optimized mnist model
print("""
Optimized Parameters:
\t{0}:\t{1}
\t{2}:\t{3}
\t{4}:\t{5}
\t{6}:\t{7}
\t{8}:\t{9}
\t{10}:\t{11}
\t{12}:\t{13}
""".format(bounds[0]["name"],opt_mnist.x_opt[0],
           bounds[1]["name"],opt_mnist.x_opt[1],
           bounds[2]["name"],opt_mnist.x_opt[2],
           bounds[3]["name"],opt_mnist.x_opt[3],
           bounds[4]["name"],opt_mnist.x_opt[4],
           bounds[5]["name"],opt_mnist.x_opt[5],
           bounds[6]["name"],opt_mnist.x_opt[6]))
print("optimized loss: {0}".format(opt_mnist.fx_opt))


# In[21]:


opt_mnist.x_opt
