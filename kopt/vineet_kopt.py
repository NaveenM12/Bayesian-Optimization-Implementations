from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
import keras.layers as kl
from keras.optimizers import Adam
# kopt and hyoperot imports
from kopt import CompileFN, KMongoTrials, test_fn
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

import numpy as np
#import sherpa
np.random.seed(123)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Conv2D
from keras.utils import np_utils
from keras.datasets import mnist
from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os

batch_size = 32
num_classes = 10
epochs = 100
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# 1. define the data function returning training, (validation, test) data
def data():
    return (x_train, y_train), (x_test, y_test)


# 2. Define the model function returning a compiled Keras model
def model(train_data, filters= 32, dropout_1= 0.25, filters_2= 64,
          layers= 512, dropout_2= 0.5, lr=0.0001):

    model = Sequential()
    model.add(Conv2D(filters, (3, 3), padding='same',
                     input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(filters, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_1))

    model.add(Conv2D(filters_2, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters_2, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_1))

    model.add(Flatten())
    model.add(Dense(layers))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_2))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=lr, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])


# Specify the optimization metrics
db_name="cpu_gpu"
exp_name="test1"
objective = CompileFN(db_name, exp_name,
                      data_fn=data,
                      model_fn=model,
                      loss_metric="loss", # which metric to optimize for
                      loss_metric_mode="min",  # try to maximize the metric
                      valid_split=.2, # use 20% of the training data for the validation set
                      save_model='best', # checkpoint the best model
                      save_results=True, # save the results as .json (in addition to mongoDB)
                      save_dir="/Users/naveenmirapuri/PycharmProjects/kopt/saved_models")  # place to store the models

# define the hyper-parameter ranges
# see https://github.com/hyperopt/hyperopt/wiki/FMin for more info
hyper_params = {
    "data": {
    },
    "model": {
        "filters": hp.choice("m_filters", (1, 16)),
        "dropout_1": hp.choice("m_dropout_1", (0.1, 0.3)),
        "layers": hp.choice("m_layers", (100, 150)),
        "dropout_2": hp.choice("m_dropout_2", (0.35, 0.6)),
    },
    "fit": {
        "batch_size":32,
        "epochs": 20,
        "verbose": 1
    }
}

# test model training, on a small subset for one epoch
test_fn(objective, hyper_params)

# run hyper-parameter optimization sequentially (without any database)
trials = Trials()
best = fmin(objective, hyper_params, trials=trials, algo=tpe.suggest, max_evals=2)

# run hyper-parameter optimization in parallel (saving the results to MonogoDB)
# Follow the hyperopt guide:
# https://github.com/hyperopt/hyperopt/wiki/Parallelizing-Evaluations-During-Search-via-MongoDB
# KMongoTrials extends hyperopt.MongoTrials with convenience methods
trials = KMongoTrials(db_name, exp_name,
                      ip="localhost",
	              port=22334)
best = fmin(objective, hyper_params, trials=trials, algo=tpe.suggest, max_evals=2)