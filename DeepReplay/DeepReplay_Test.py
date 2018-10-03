

from deepreplay.callbacks import ReplayData
from deepreplay.datasets.parabola import load_data

X, y = load_data()

replaydata = ReplayData(X, y, filename='hyperparams_in_action.h5', group_name='part1')

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.initializers import glorot_normal, normal

model = Sequential()
model.add(Dense(input_dim=2,
                units=2,
                activation='sigmoid',
                kernel_initializer=glorot_normal(seed=42),
                name='hidden'))
model.add(Dense(units=1,
                activation='sigmoid',
                kernel_initializer=normal(seed=42),
                name='output'))

model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.05), metrics=['acc'])

model.fit(X, y, epochs=150, batch_size=16, callbacks=[replaydata])

from deepreplay.replay import Replay

replay = Replay(replay_filename='hyperparams_in_action.h5', group_name='part1')

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(5, 5))

fs = replay.build_feature_space(ax, layer_name='hidden')

fs.plot(epoch=60).savefig('feature_space_epoch60.png', dpi=120)
fs.animate().save('feature_space_animation.mp4', dpi=120, fps=5)


