from tensorflow.python.keras import Model, Sequential
from tensorflow.python.keras.layers import Embedding, Dense, Conv1D, GlobalMaxPooling1D, Concatenate, Dropout

import to_dataset

# path_train = "D:/work/program/python/tf/minist/data/dataset/mnist.train.tfrecord"
path_test = "D:/work/program/python/tf/minist/data/dataset/mnist.test.tfrecord"
# dataset_train = to_dataset.read_dataset(path_train)
dataset_test = to_dataset.read_dataset(path_test)

model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(to_dataset.image_size,)))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(dataset_test, batch_size=to_dataset.batch_size, epochs=2, steps_per_epoch=to_dataset.steps_per_epoch)
