import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

image_size = 784
shuffle_size = 100
batch_size = 20
num_test_example = 10000
steps_per_epoch = int(num_test_example / batch_size)


def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature


def create_float_feature(values):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature


def to_dataset(path_in, path_out_train, path_out_test):
    mnist = input_data.read_data_sets(path_in, dtype=tf.uint8, one_hot=True)
    print("finish load zip")

    # 训练数据
    images_train = mnist.train.images.astype(np.int)
    labels_train = mnist.train.labels.astype(np.int)
    pixels_train = images_train.shape[1]
    num_examples_train = mnist.train.num_examples
    _to_dataset(path_out_train, num_examples_train, images_train, labels_train, pixels_train)

    # 测试数据
    images_test = mnist.test.images.astype(np.int)
    labels_test = mnist.test.labels.astype(np.int)
    pixels_test = images_test.shape[1]
    num_examples_test = mnist.test.num_examples
    _to_dataset(path_out_test, num_examples_test, images_test, labels_test, pixels_test)


def _to_dataset(path_out, num_examples, images, labels, pixels):
    writer = tf.python_io.TFRecordWriter(path_out)
    for index in range(num_examples):
        if (index % 1000 == 0): print("**_to_dataset(): index=%d" % index)
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'pixels': create_int_feature([pixels]),
                'label': create_int_feature([np.argmax(labels[index])]),
                'images': create_int_feature(images[index])
            }
        ))
        writer.write(example.SerializeToString())

    writer.close()
    print("**_to_dataset(): finish")


def read_dataset(path):
    dataset = tf.data.TFRecordDataset(path)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=shuffle_size)
    dataset = dataset.map(_decode_record)
    dataset = dataset.batch(batch_size=batch_size)

    return dataset


def _decode_record(record):
    name_to_features = {
        "pixels": tf.FixedLenFeature([1], tf.int64),
        "images": tf.FixedLenFeature([image_size], tf.int64),
        "label": tf.FixedLenFeature([1], tf.int64),
    }
    example = tf.parse_single_example(record, name_to_features)
    images = tf.cast(example["images"], tf.float32) / 255.
    label = tf.one_hot(example["label"][0], depth=10, dtype=tf.float32)

    return images, label


if __name__ == '__main__':
    dir = "D:/work/program/python/tf/minist"
    path_in = "./data/zip"
    path_out_train = dir + "/data/dataset/mnist.train.tfrecord"
    path_out_test = dir + "/data/dataset/mnist.test.tfrecord"
    to_dataset(path_in, path_out_train, path_out_test)
