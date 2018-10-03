from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf


def initialise_flags(args_parser):
    args_parser.add_argument("--tfrecord", required=True)
    args_parser.add_argument("--head-only", type=bool, default=False)

    return args_parser.parse_args()


def main():
    tfrecord_path = os.path.relpath(FLAGS.tfrecord)
    num_printed = 0
    total_num = 0
    for example in tf.python_io.tf_record_iterator(tfrecord_path):
        if not (FLAGS.head_only and num_printed >= 3):
            print(tf.train.Example.FromString(example))
            print("----------------------------------------------")
            num_printed += 1

        total_num += 1

    print("Total number of records = ", total_num)


args_parser = argparse.ArgumentParser()
FLAGS = initialise_flags(args_parser)

if __name__ == "__main__":
    main()
