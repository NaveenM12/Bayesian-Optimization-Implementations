import numpy as np
from numpy import *
import tensorflow
import keras
import sherpa.sherpa1.database

client = sherpa.sherpa1.database.Client()
trial = client.get_trial()

from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(units=trial.parameters['num_units'], activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

callbacks = [client.keras_send_metrics(trial, objective_name='val_loss',
             context_names=['val_acc'])]

x_train = array([1,2,3])
y_train = array([1,2,3])

model.fit(np.x_train, y_train, epochs=5, batch_size=32, callbacks=callbacks)



