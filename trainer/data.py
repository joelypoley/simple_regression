from __future__ import division
from __future__ import print_function

import tensorflow as tf

FEATURE_NAME = "x"
LABEL_NAME = "y"


def parse_fn(example_proto):
    features = {
        "x": tf.FixedLenFeature((), tf.float32, default_value=0.0),
        "y": tf.FixedLenFeature((), tf.float32, default_value=0.0),
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    label = parsed_features["y"]
    del parsed_features["y"]

    return parsed_features, label


def get_input_fn(file_path, shuffle=False, batch_size=1, repeat=1):

    def input_fn():
        # TODO: make the tfrecords sharded by default and use tf.data.Dataset.list_files.
        dataset = tf.data.TFRecordDataset(file_path).map(parse_fn)
        dataset = dataset.cache()
        if shuffle:
            dataset = dataset.shuffle(buffer_size=128)

        dataset = dataset.repeat(repeat)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=1)
        return dataset

    return input_fn


def json_serving_input_fn():
    receiver_tensors = {FEATURE_NAME: tf.placeholder(shape=[None], dtype=tf.float32)}
    features = {
        key: tf.expand_dims(tensor, -1) for key, tensor in receiver_tensors.items()
    }
    return tf.estimator.export.ServingInputReceiver(
        features=features, receiver_tensors=receiver_tensors
    )
