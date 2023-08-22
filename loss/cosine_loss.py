#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf

#==========================Cosine Loss==========================
#  AOL Regularizer
def cosine_loss(s1,s2,z1,z2):
    s1=tf.nn.l2_normalize(s1,dim=1)
    s2=tf.nn.l2_normalize(s2,dim=1)
    z1=tf.nn.l2_normalize(z1,dim=1)
    z2=tf.nn.l2_normalize(z2,dim=1)
    cos_s=s1*s2
    cos_z=z1*z2
    loss = (1+tf.reduce_sum(cos_z,axis=1))*(tf.math.maximum(0.0000001, tf.reduce_sum(cos_s,axis=1)))
    return tf.reduce_mean(loss)