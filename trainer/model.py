from __future__ import division
from __future__ import print_function

import tensorflow as tf

import task
import data

FEATURE_NAME = "x"


def get_feature_columns():
    return [tf.feature_column.numeric_column(FEATURE_NAME)]


def linear_model_fn(features, labels, mode, params):
    input_layer = tf.feature_column.input_layer(features, params["feature_columns"])
    logits = tf.layers.dense(input_layer, 1, activation=None)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {"predictions": logits}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = tf.losses.mean_squared_error(tf.squeeze(labels), logits)
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss)

    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = params["optimizer"]
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def create_estimator():
    estimator = tf.estimator.Estimator(
        model_fn=linear_model_fn,
        params={
            "feature_columns": get_feature_columns(),
            "optimizer": tf.train.AdamOptimizer(task.FLAGS.learning_rate),
        },
        model_dir=task.FLAGS.job_dir,
    )
    return estimator
