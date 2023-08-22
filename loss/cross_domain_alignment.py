#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
import numpy as np

#==========================P Loss==========================
#calculate P_loss
def calc_weighted_gamma_dists(z_prototypes_s_combined,z_prototypes_d_combined,Gamma_D1,Gamma_D2,Beta_D1,Beta_D2):
  prot_bn_dists = []
  for i in range(len(Gamma_D1)):
    for j in range(len(z_prototypes_s_combined)):
      d = ( tf.reduce_mean(z_prototypes_s_combined[j])*tf.cast(tf.reduce_mean(Gamma_D1[i]),tf.float64) + tf.cast(tf.reduce_mean(Beta_D1[i]),tf.float64) ) - ( tf.reduce_mean(z_prototypes_d_combined[j])*tf.cast(tf.reduce_mean(Gamma_D2[i]),tf.float64) + tf.cast(tf.reduce_mean(Beta_D2[i]),tf.float64) )
      prot_bn_dists.append(d)
  return tf.reduce_mean(prot_bn_dists)