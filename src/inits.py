import tensorflow as tf
import numpy as np


def uniform(shape, scale=0.05, name=None):
    """Uniform init."""
    initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    rtn = tf.Variable(initial, name=name)
    # initial = tf.random_uniform((1, shape[1]), minval=-init_range,
    #                             maxval=init_range,
    #                             dtype=tf.float32)
    # rtn = tf.tile(initial, [shape[0], 1])
    # rtn = tf.Variable(rtn, name=name)
    return rtn

def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def ones(shape, name=None):
    """All ones."""
    initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)