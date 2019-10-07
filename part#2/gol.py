import h5py
import numpy as np
import tqdm

class GameOfLife:
    @staticmethod
    def gen_random_state(width: int, height: int):
        state = (np.random.rand(width, height) * 2).astype(np.int32)
        return state

    @staticmethod
    def next_state(state):
        l_neib = np.roll(state, 1, 0)
        r_neib = np.roll(state, -1, 0)
        u_neib = np.roll(state, 1, 1)
        d_neib = np.roll(state, -1, 1)
        ul_neib = np.roll(l_neib, 1, 1)
        dl_neib = np.roll(l_neib, -1, 1)
        ur_neib = np.roll(r_neib, 1, 1)
        dr_neib = np.roll(r_neib, -1, 1)

        neibs = l_neib + r_neib + u_neib + d_neib + ul_neib + dl_neib + ur_neib + dr_neib
        next_state = np.copy(state)
        next_state[(neibs < 2) | (neibs > 3)] = 0
        next_state[neibs == 3] = 1

        return next_state


n_samples = 10000
width = 20
height = 30

try:
    data_file = h5py.File(f"dataset_{width}x{height}x{n_samples}.h5", 'r')
    x_train = data_file["x_train"][:]
    y_train = data_file["y_train"][:]
    data_file.close()
except OSError:
    print("Generate x_train")
    x_train = []
    for _ in tqdm.trange(n_samples):
        x_train.append(GameOfLife.gen_random_state(width, height))

    x_train = np.array(x_train)

    print("Generate y_train")
    y_train = np.zeros_like(x_train)
    for i, x in tqdm.tqdm(enumerate(x_train), total=len(x_train)):
        y_train[i] = GameOfLife.next_state(x)

    data_file = h5py.File(f"dataset_{width}x{height}x{n_samples}.h5", 'w')
    data_file.create_dataset("x_train", data=x_train)
    data_file.create_dataset("y_train", data=y_train)
    data_file.close()

print(f"Dataset shape: {x_train.shape}")


def get_data(width, height, n_samples):
    x = []
    for _ in tqdm.trange(n_samples):
        x.append(GameOfLife.gen_random_state(width, height))
    x = np.array(x)
    y = np.zeros_like(x)
    for i, j in tqdm.tqdm(enumerate(x), total=len(x)):
        y[i] = GameOfLife.next_state(j)
    return x, y

def read_data(file_path):
    with h5py.File(file_path, 'r') as f:
        x_train = np.array(f['x_train'])
        y_train = np.array(f['y_train'])
        iterations_count = len(x_train)
        width = len(x_train[0])
        height = len(x_train[0][0])
        return x_train, y_train

x_train, y_train = read_data(f"dataset_{width}x{height}x{n_samples}.h5")

import keras.backend as K
from keras.layers import Dense, LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Bidirectional, BatchNormalization
from keras.models import Sequential

def rnn_model(width, height):
    model = Sequential()
    model.add(Dense(1200, input_shape=(width, height), activation='linear'))
    model.add(Bidirectional(LSTM(units = 1200,
                                 input_shape=(width, height),
                                 activation='relu',
                                 dropout=0.25,
                                 recurrent_dropout=0.25,
                                 return_sequences=True,
                                 recurrent_initializer='random_uniform',
                                 unroll=True),
                            merge_mode='mul'))
    model.add(BatchNormalization())
    model.add(Dense(width * height, activation='relu'))
    model.add(Dense(height, activation='sigmoid'))
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


rnn = rnn_model(20, 30)

stopping = EarlyStopping(monitor = 'val_loss',
                          min_delta = 0,
                          patience = 3,
                          verbose = 1,
                          restore_best_weights = True)
checkpoint = ModelCheckpoint(filepath = 'checkpoint.hd5',
                             monitor = 'val_loss',
                             mode = 'min',
                             save_best_only = True,
                             verbose = 1)

rnn.fit(x_train, y_train, epochs = 10, verbose = 1, validation_split = 0.1, callbacks=[stopping, checkpoint])


x_test, y_test = get_data(width, height, n_samples)
pred = rnn.predict(x_test)
correct = 0

for i in tqdm.trange(len(x_test)):
    tmp = (y_test[i] == pred[i].round().astype(int))
    count = 0
    for v in tmp:
        count += v.sum()
    count /= (width * height)
    correct += count

print('\nPrediction accuracy: ', correct / n_samples)
