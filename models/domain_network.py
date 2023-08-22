#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
from tensorflow.keras import layers

#============================Domain network====================
input_Feature = layers.Input(shape = (3,))
encoded_L1 = layers.Dense(128, activation='relu',kernel_regularizer=tf.keras.regularizers.L1(0.01),activity_regularizer=tf.keras.regularizers.L2(0.001))(input_Feature)
encoded_L2 = layers.Dense(64, activation='relu',kernel_regularizer=tf.keras.regularizers.L1(0.01),activity_regularizer=tf.keras.regularizers.L2(0.001))(encoded_L1)
encoded_L3 = layers.Dense(32, activation='relu',kernel_regularizer=tf.keras.regularizers.L1(0.01),activity_regularizer=tf.keras.regularizers.L2(0.001))(encoded_L2)
encoded_L4 = layers.Dense(16, activation='relu',kernel_regularizer=tf.keras.regularizers.L1(0.01),activity_regularizer=tf.keras.regularizers.L2(0.001))(encoded_L3)
encoded_L5 = layers.Dense(8, activation='relu',kernel_regularizer=tf.keras.regularizers.L1(0.01),activity_regularizer=tf.keras.regularizers.L2(0.001))(encoded_L4)
decoded = layers.Dense(2, activation='softmax')(encoded_L5)
domain_nn = tf.keras.Model(inputs=input_Feature,outputs=decoded,name='DomainNN')
# domain_nn.summary()