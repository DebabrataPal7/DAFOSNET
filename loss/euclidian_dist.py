#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
import numpy as np

#calculate euclidian distance between two 2d tensors
def calc_euclidian_dists2(x, y):
  # x : (n,d)
  # y : (m,d)
    n = x.shape[0]
    m = y.shape[0]
    x = tf.cast(tf.tile(tf.expand_dims(x, 1), [1, m, 1]),dtype=tf.float64)
    y = tf.cast(tf.tile(tf.expand_dims(y, 0), [n, 1, 1]),dtype=tf.float64)
    return tf.reduce_mean(tf.math.pow(x - y, 2), 2)