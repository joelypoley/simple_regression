from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import random

import tensorflow as tf

FEATURE_NAME = "x"
LABEL_NAME = "y"


def initialise_hyper_params(args_parser):
    args_parser.add_argument("--csv", required=True)
    args_parser.add_argument("--train-tfrecord", required=True)
    args_parser.add_argument("--eval-tfrecord", required=True)
    args_parser.add_argument("--validation-tfrecord", required=True)

    return args_parser.parse_args()


def csv_row_to_example(row):
    x_raw, y_raw = row.split(",")
    x, y = float(x_raw.strip()), float(y_raw.strip())
    feature_dict = {
        FEATURE_NAME: tf.train.Feature(float_list=tf.train.FloatList(value=[x])),
        LABEL_NAME: tf.train.Feature(float_list=tf.train.FloatList(value=[y])),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature_dict))


def main():
    csv_path = os.path.relpath(FLAGS.csv)
    train_tfrecord_path = os.path.relpath(FLAGS.train_tfrecord)
    eval_tfrecord_path = os.path.relpath(FLAGS.eval_tfrecord)
    validation_tfrecord_path = os.path.relpath(FLAGS.validation_tfrecord)
    with open(csv_path, "r") as csv:
        next(csv)  # Skip heading row.
        train_writer = tf.python_io.TFRecordWriter(train_tfrecord_path)
        eval_writer = tf.python_io.TFRecordWriter(eval_tfrecord_path)
        validation_writer = tf.python_io.TFRecordWriter(validation_tfrecord_path)
        for row in csv:
            example = csv_row_to_example(row)
            rand = random.random()
            if rand < 0.8:
                train_writer.write(example.SerializeToString())
            elif 0.8 <= rand < 0.9:
                eval_writer.write(example.SerializeToString())
            else:
                validation_writer.write(example.SerializeToString())

        train_writer.close()
        eval_writer.close()
        validation_writer.close()


args_parser = argparse.ArgumentParser()
FLAGS = initialise_hyper_params(args_parser)

if __name__ == "__main__":
    main()
