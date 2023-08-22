#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
from tensorflow.keras import layers

#Inherited from ResNet18 conventional batch norm layers are replaced with DSBN (Domain Specific Batch Norm)
Model_input = tf.keras.Input(shape=(224, 224, 3), batch_size=None)
X = layers.ZeroPadding2D((1, 1))(Model_input)
Conv1 = layers.Conv2D(256, (3, 3), activation='relu')(X)
BN1 = layers.BatchNormalization(name='BN1')(Conv1)
Maxpool1 = layers.MaxPooling2D()(BN1)

Usampled_Maxpool1 = layers.UpSampling2D()(Maxpool1)

Skipped_1_2 = layers.Add()([Usampled_Maxpool1, Conv1])
Activation_skipped_1_2 = layers.Activation('relu')(Skipped_1_2)

Conv2 = layers.Conv2D(128, (3, 3), activation='relu', strides=(2, 2), padding="same")(Activation_skipped_1_2)
BN2 = layers.BatchNormalization(name='BN2')(Conv2)
Maxpool2 = layers.MaxPooling2D()(BN2)

Usampled_Maxpool2 = layers.UpSampling2D()(Maxpool2)

Skipped_2_3 = layers.Add()([Usampled_Maxpool2, Conv2])
Activation_skipped_2_3 = layers.Activation('relu')(Skipped_2_3)

# conv block 3
Conv3 = layers.Conv2D(32, (3, 3), activation='relu',strides=(3, 3))(Activation_skipped_2_3)
subsection3_D1 = layers.Lambda(lambda x: x[:36, :, :, :])(Conv3)
subsection3_D2 = layers.Lambda(lambda x: x[36:, :, :, :])(Conv3)
# (Domain Specific Batch Norm)
BN3_D1 = layers.BatchNormalization(name='BN3_D1')(subsection3_D1)
BN3_D2 = layers.BatchNormalization(name='BN3_D2')(subsection3_D2)
Concat3 = layers.Concatenate(axis=0)([BN3_D1, BN3_D2])
BN3 = layers.BatchNormalization()(Concat3)
Maxpool3 = layers.MaxPooling2D((3,3))(BN3)

Flatten_l = layers.Flatten()(Maxpool3)

Dense1 = layers.Dense(1024, activation='relu')(Flatten_l)
BN4 = layers.BatchNormalization(name='BN4')(Dense1)
Dense2 = layers.Dense(512, activation='relu')(BN4)
BN5 = layers.BatchNormalization(name='BN5')(Dense2)
Dense3 = layers.Dense(128, activation='relu')(BN5)
BN6 = layers.BatchNormalization(name='BN6')(Dense3)
Dense4 = layers.Dense(64, activation='relu')(BN6)

CDFSOSR_Model = tf.keras.Model(inputs=Model_input, outputs=Dense4, name='CDFSOSR_Model')
# CDFSOSR_Model.summary()


