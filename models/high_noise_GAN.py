#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
from tensorflow.keras import layers

#==================================================High noise GAN========================================================
# GAN2 : to generate pseudo-unknown to augment query samples: Input high noise variance in SOURCE DOMAIN
#Generator (High)
generator_input_size=10+8
input_feature = layers.Input(shape=(generator_input_size,))
layer_h_g1 = layers.Dense(32,activation='relu')(input_feature)
layer_h_g2 = layers.Dense(48,activation='relu')(layer_h_g1)
layer_h_g3 = layers.Dense(64,activation='relu')(layer_h_g2)
generator_nn_high_s = tf.keras.Model(inputs=input_feature,outputs=layer_h_g3, name='generator_high_source')
generator_nn_high_s.summary()

# Discriminator (High)
input_feature = layers.Input(shape=(74,))
layer_h_d1 = layers.Dense(48,activation='relu')(input_feature)
layer_h_d2 = layers.Dense(32,activation='relu')(layer_h_d1)
layer_h_d3 = layers.Dense(16,activation='relu')(layer_h_d2)
layer_h_d4 = layers.Dense(8,activation='relu')(layer_h_d3)
layer_h_d5 = layers.Dense(1)(layer_h_d4)
dis_nn_high_s = tf.keras.Model(inputs=input_feature,outputs=layer_h_d5, name='discriminator_high_source')
# dis_nn_high_s.summary()