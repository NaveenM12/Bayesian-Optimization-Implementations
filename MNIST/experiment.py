from __future__ import division
from __future__ import print_function
import numpy as np

#np.random.seed(123)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Conv2D
from keras.utils import np_utils
from past.utils import old_div
import keras.datasets.mnist as mnist
import math


def calc_loss(filters, num_units):
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

    print('model created')
    model.add(Convolution2D(filters=filters, kernel_size=(3, 3), activation='relu', input_shape=(1, 28, 28),
                            data_format='channels_first'))

    model.add(Convolution2D(filters=filters, kernel_size=(3, 3,), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(num_units, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', )
    # metrics=['accuracy'])
    print('compiled, now fitting...')

    model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=1)
    # Epoch 1/10
    # 7744/60000 [==>...........................] - ETA: 96s - loss: 0.5806 - acc: 0.8164

    print('fitted, now evaluating...')

    score = model.evaluate(X_test, Y_test, verbose=0)

    print('Result = %f' % score)
    return score


# Write a function like this called 'main'
def main(job_id, params):
    print('Anything printed here will end up in the output directory for job #%d' % job_id)
    print(params)
    return calc_loss(int(params['filters']), int(params['num_units']))
