#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
from tensorflow.keras import layers

#==============================================================Low noise GAN========================================================
#GAN1 : to generate pseudo-known samples: Input low noise variance for SOURCE DOMAIN

# Generator (Low) SOURCE DOMAIN
generator_input_size=10+8#dababrata: input dimesnions for gans?? 28 is from feature output from episodes
input_feature = layers.Input(shape=(generator_input_size,))
layer_low_g1 = layers.Dense(32,activation='relu')(input_feature)
layer_low_g2 = layers.Dense(48,activation='relu')(layer_low_g1)
layer_low_g3 = layers.Dense(64,activation='relu')(layer_low_g2)
generator_nn_low_s = tf.keras.Model(inputs=input_feature,outputs=layer_low_g3, name='generator_low_source')
generator_nn_low_s.summary()

#Discriminator (Low) SOURCE DOMAIN
input_feature = layers.Input(shape=(74,))
layer_l_d1 = layers.Dense(48,activation='relu')(input_feature)
layer_l_d2 = layers.Dense(32,activation='relu')(layer_l_d1)
layer_l_d3 = layers.Dense(16,activation='relu')(layer_l_d2)
layer_l_d4 = layers.Dense(8,activation='relu')(layer_l_d3)
layer_l_d5 = layers.Dense(1)(layer_l_d4)
dis_nn_low_s = tf.keras.Model(inputs=input_feature,outputs=layer_l_d5, name='discriminator_low_source')
# dis_nn_low_s.summary()
