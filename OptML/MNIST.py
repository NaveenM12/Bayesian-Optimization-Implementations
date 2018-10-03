import numpy as np

np.random.seed(123)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Conv2D
from keras.utils import np_utils
from keras.datasets import mnist
from optml.bayesian_optimizer import BayesianOptimizer
import sklearn

import optml



(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)


model = Sequential()

model.add(Convolution2D(filters= 32, kernel_size=(3,3), activation='relu', input_shape=(1, 28, 28), data_format='channels_first'))

model.add(Convolution2D(filters=32, kernel_size=(3, 3,), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


params = [optml.Parameter(name='filters', param_type='integer', lower=20, upper=50),
          optml.Parameter(name='units', param_type='integer', lower=100, upper=150)]

def clf_score(y_true,y_pred):
    mse = ((y_true - y_pred) ** 2).mean(axis=None)
    cce = sklearn.metrics.log_loss(y_true, y_pred)
    return cce


bayesOpt = BayesianOptimizer(model=model,
                             hyperparams=params,
                             eval_func=clf_score)

bayes_best_params, bayes_best_model = bayesOpt.fit(X_train=X_train, y_train=y_train, n_iters=50)