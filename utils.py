import os
import shutil

import tensorflow as tf
from tensorflow.contrib.framework import add_arg_scope


def delete_existing(path):
    if os.path.exists(path):
        shutil.rmtree(path)


def save_model(saver, M, model_dir, global_step):
    path = saver.save(M.sess, os.path.join(model_dir, 'model'), global_step=global_step)
    print(f"Saving model to {path}")


@add_arg_scope
def leaky_relu(x, a=0.2, name=None):
    with tf.name_scope(name, 'leaky_relu'):
        return tf.maximum(x, a * x)


@add_arg_scope
def relu(x, name=None):
    return tf.nn.relu(x, name=name)


@add_arg_scope
def tanh(x, name=None):
    return tf.nn.tanh(x, name=name)


@add_arg_scope
def sigmoid(x, name=None):
    return tf.nn.sigmoid(x, name=name)
