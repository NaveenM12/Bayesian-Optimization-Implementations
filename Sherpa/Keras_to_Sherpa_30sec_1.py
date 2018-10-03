import numpy as np
import sherpa.sherpa1.algorithms
import sherpa.sherpa1.core
np.random.seed(123)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Conv2D
from keras.utils import np_utils
from keras.datasets import mnist

    # Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Convert 1-dimensional class arrays to 10-dimensional class matrices
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)


parameters = [
              sherpa.sherpa1.core.Choice('num_conv_filters', [22, 32, 42]),
              #sherpa.Continuous('num_conv_filters', [25, 35]),
              sherpa.sherpa1.core.Choice('num_rows_conv_kernel', [1, 3, 5]),
              sherpa.sherpa1.core.Choice('num_cols_conv_kernel', [1, 3, 5]),
              sherpa.sherpa1.core.Choice('num_conv_filters2', [22, 32, 42]),
              sherpa.sherpa1.core.Choice('num_rows_conv_kernel2', [1, 3, 5]),
              sherpa.sherpa1.core.Choice('num_cols_conv_kernel2', [1, 3, 5]),
              sherpa.sherpa1.core.Choice('pooling_filter_w', [1,2, 5]),
              sherpa.sherpa1.core.Choice('pooling_filter_l', [1, 2, 5]),
              sherpa.sherpa1.core.Choice('num_units', [100, 128, 150]),
              ]


alg = sherpa.sherpa1.algorithms.BayesianOptimization(max_num_trials=2)

study = sherpa.sherpa1.core.Study(parameters=parameters,
                     algorithm=alg,
                     lower_is_better=True)

for trial in study:
    model = Sequential()

    model.add(Conv2D(trial.parameters['num_conv_filters'],
                     (trial.parameters['num_rows_conv_kernel'], trial.parameters['num_cols_conv_kernel']),
                     activation='relu', input_shape = ((int)(1), (int)(28), (int)(28)), data_format='channels_first'))
    # model.add(Convolution2D(32, (3,3), activation='relu', input_shape=(28, 28, 1), data_format='channels_last'))

    model.add(Convolution2D(trial.parameters['num_conv_filters2'],
                            (trial.parameters['num_rows_conv_kernel2'], trial.parameters['num_cols_conv_kernel2']),
                            activation='relu'))
    model.add(MaxPooling2D(pool_size=(trial.parameters['pooling_filter_w'], trial.parameters['pooling_filter_l'])))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(trial.parameters['num_units'], activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(X_train, Y_train, epochs=1, batch_size=32,
              callbacks=[study.keras_callback(trial, objective_name='loss',)]
              )
    study.finalize(trial, status='COMPLETED')



'''
# Epoch 1/10
# 7744/60000 [==>...........................] - ETA: 96s - loss: 0.5806 - acc: 0.8164


score = model.evaluate(X_test, Y_test, verbose=0)

print(score)



parameters = [sherpa1.Discrete('num_units', [50, 200])]
alg = sherpa1.algorithms.BayesianOptimization(max_num_trials=50)

study = sherpa1.Study(parameters=parameters,
                     algorithm=alg,
                     lower_is_better=True)

for trial in study:
    model = Sequential()
    model.add(Dense(units=trial.parameters['num_units'],
                    activation='relu', input_dim=100))
    model.add(Dense(units=10, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=1,
              callbacks=[study.keras_callback(trial, objective_name='val_loss')])
    study.finalize(trial)
    
    
    # Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()


#print (X_train.shape)
# (60000, 28, 28)

#from matplotlib import pyplot as plt
#plt.imshow(X_train[0])

X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

print (X_train.shape)
# (60000, 1, 28, 28)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


# print (y_train.shape)
# (60000,)


# print (y_train[:10])
# [5 0 4 1 9 2 1 3 1 4]

# Convert 1-dimensional class arrays to 10-dimensional class matrices
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

#print(Y_train.shape)
# (60000,)



'''