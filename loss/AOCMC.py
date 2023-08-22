#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf

#==========================AOCMC Loss==========================
def anti_open_close_mode_collapse_loss(sl, sh , zl, zh, epsilon=1e-8):
    cos_sl_sh = tf.reduce_sum(sl * sh, axis=-1) / (tf.norm(sl, axis=-1) * tf.norm(sh, axis=-1) + epsilon)
    cos_zl_zh = tf.reduce_sum(zl * zh, axis=-1) / (tf.norm(zl, axis=-1) * tf.norm(zh, axis=-1) + epsilon)

    numerator = 1 - cos_zl_zh + epsilon
    denominator = 1 - cos_sl_sh + epsilon
    LAOCMC = 1 + tf.math.log(numerator/denominator)
    
    loss = LAOCMC
    return tf.reduce_mean(loss)