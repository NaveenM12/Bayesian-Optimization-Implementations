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

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

# 1. define the data function returning training, (validation, test) data
def data():
    return (X_train, Y_train), (X_test, Y_test)


# 2. Define the model function returning a compiled Keras model
def model(train_data, filters= 32, dropout_1= 0.25, layers= 128, dropout_2= 0.5):

    model = Sequential()

    model.add(Convolution2D(filters=(int)(filters), kernel_size=(3, 3), activation='relu', input_shape=(1, 28, 28),
                            data_format='channels_first'))

    model.add(Convolution2D(filters=(int)(filters), kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_1))

    model.add(Flatten())
    model.add(Dense((int)(layers), activation='relu'))
    model.add(Dropout(dropout_2))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['loss'])


# Specify the optimization metrics
db_name="mnist"
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
        "x": X_train,
        "y": Y_train,
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