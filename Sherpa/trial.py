
import sherpa.sherpa1.database

client = sherpa.sherpa1.database.Client()
trial = client.get_trial()


import numpy as np
#np.random.seed(123)  # for reproducibility
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

# Convert 1-dimensional class arrays to 10-dimensional class matrices
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)



model = Sequential()
model.add(Conv2D(32, (3,3),
                 activation='relu', input_shape=(1, 28, 28), data_format='channels_first'))
#model.add(Convolution2D(32, (3,3), activation='relu', input_shape=(28, 28, 1), data_format='channels_last'))

model.add(Convolution2D(32, (3,3),
                        activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(units=(int)(trial.parameters['num_units']), activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


callbacks = [client.keras_send_metrics(trial=trial, objective_name='loss',context_names=['acc'])]

model.fit(X_train, Y_train, epochs=2, batch_size=32, callbacks=callbacks)



'''
#num_iterations = 3
#for i in range(num_iterations):

sherpa.sherpa1.database.Client.keras_send_metrics(trial=trial, iterationobjective_name='loss',
                                                      context_names=['num_conv_filters', 'num_rows_conv_kernel',
                                                                     'num_cols_conv_kernel', 'num_conv_filters2',
                                                                     'num_rows_conv_kernel2', 'num_cols_conv_kernel2',
                                                                     'pooling_filter_w', 'pooling_filter_l',
                                                                     'num_units', ])
#client.add_trial(trial)

'''



