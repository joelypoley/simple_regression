from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import random

import tensorflow as tf

GRADIENT = 2


def initialise_hyper_params(args_parser):
    args_parser.add_argument("--num-training-examples", type=int, default=1000)
    args_parser.add_argument("--output-csv", required=True)

    return args_parser.parse_args()


def main():
    csv_path = os.path.relpath(FLAGS.output_csv)
    with open(csv_path, "w") as csv:
        csv.write("x, y\n")
        for _ in xrange(FLAGS.num_training_examples):
            x = random.randrange(1000)
            y = GRADIENT * x + random.random()
            csv.write("{}, {}\n".format(x, y))


args_parser = argparse.ArgumentParser()
FLAGS = initialise_hyper_params(args_parser)

if __name__ == "__main__":
    main()
