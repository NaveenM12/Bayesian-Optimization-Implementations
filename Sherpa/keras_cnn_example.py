import numpy as np
import sherpa
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
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

#print(Y_train.shape)
# (60000,)


model = Sequential()

model.add(Convolution2D(filters= 32, kernel_size=(3,3), activation='relu', input_shape=(1, 28, 28), data_format='channels_first'))

model.add(Convolution2D(filters=32, kernel_size=(3, 3,), activation='relu'))
'''

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(1, 28, 28), data_format='channels_first'))
#model.add(Convolution2D(32, (3,3), activation='relu', input_shape=(28, 28, 1), data_format='channels_last'))


#print(model.output_shape)
# (None, 32, 26, 26)

model.add(Convolution2D(32, (3, 3,), activation='relu'))
'''


model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='adam',)
             # metrics=['accuracy'])


model.fit(x= X_train, y=Y_train, batch_size=32, epochs=10, verbose=1)
# Epoch 1/10
# 7744/60000 [==>...........................] - ETA: 96s - loss: 0.5806 - acc: 0.8164


score = model.evaluate(X_test, Y_test, verbose=0)

print(score)

'''

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

'''