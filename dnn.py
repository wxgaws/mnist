from tensorflow.python.keras import Model, Sequential
from tensorflow.python.keras.layers import Embedding, Dense, Conv1D, GlobalMaxPooling1D, Concatenate, Dropout
import os
#
import to_dataset

path_train = os.path.join(to_dataset.dir_main, "data/dataset/mnist.train.tfrecord")
path_test = os.path.join(to_dataset.dir_main, "data/dataset/mnist.test.tfrecord")
dataset_train = to_dataset.read_dataset(path_train)
dataset_test = to_dataset.read_dataset(path_test)

print(to_dataset.dir_main)
print(path_train)
print(os.path.join("D:/work/program/python/tf/mnist", "./data/dataset/mnist.train.tfrecord"))

model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(to_dataset.image_size,)))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(dataset_train, batch_size=to_dataset.batch_size, epochs=2, steps_per_epoch=to_dataset.steps_per_epoch_train)
# model.fit(dataset_test, batch_size=to_dataset.batch_size, epochs=2, steps_per_epoch=to_dataset.steps_per_epoch_test)
