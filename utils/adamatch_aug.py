#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
import tensorflow as tf
from keras_cv.layers import RandAugment

#===================================Adamatch Augmentation===================================

RESIZE_TO=224
# Initialize `RandAugment` object with 2 layers of
# augmentation transforms and strength of 5.
augmenter = RandAugment(value_range=(0, 255), augmentations_per_image=2, magnitude=0.5)

def weak_augment(image, source=True):
    if image.dtype != tf.float64:
        image = tf.cast(image, tf.float64)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_crop(image, (RESIZE_TO, RESIZE_TO, 3))
    return image

def strong_augment(image, source=True):
    if image.dtype != tf.float64:
        image = tf.cast(image, tf.float64)
    image = augmenter(image)
    return image

def concat_images_to_same_class(tensor1,tensor2,num_classes,Ss1,Ss2):
    '''This function takes in two tensors where each array is of the form [[class1],[class2],[class3]....] and returns a single tensor of the form [[class1,class2,class3]....] by assigning images beloning to the same class from both tensors to the new tensor'''
    tensor1_domain1 = tensor1[:num_classes*Ss1,:,:,:]
    tensor2_domain1 = tensor2[:num_classes*Ss2,:,:,:]
    tensor_domain1 = tf.concat((tensor1_domain1, tensor2_domain1),axis=0)
    tensor1_domain2 = tensor1[num_classes*Ss1:,:,:,:]
    tensor2_domain2 = tensor2[num_classes*Ss2:,:,:,:]
    tensor_domain2 = tf.concat((tensor1_domain2, tensor2_domain2),axis=0)
    new_tensor = tf.concat((tensor_domain1,tensor_domain2),axis=0)
    return new_tensor

def adamatch_aug(tsupport_patches, support_labels, CS, Ss, Sd, support_dom_labels):
        '''return original + strong + weak augmented samples Ss_aug = Ss*3 Sd_aug = Sd*3'''
        #patches
        tsupport_patches_w = tf.convert_to_tensor(list(map(strong_augment,tsupport_patches)), dtype=tf.float64)
        tsupport_patches_s = tf.convert_to_tensor(list(map(weak_augment,tsupport_patches)), dtype=tf.float64)

        #concat weak and strong augmented samples
        tsupport_concat_w_s = concat_images_to_same_class(tsupport_patches_w, tsupport_patches_s, CS, Ss, Ss)
        #concat weak and strong augmented samples with original samples
        tsupport_patches_concat_original_w_s = concat_images_to_same_class(tsupport_patches, tsupport_concat_w_s, CS, Ss, Ss*2)

        Ss_aug=Ss*3
        Sd_aug=Sd*3
        #labels
        support_labels_source = support_labels[:CS*Ss]
        support_labels_target = support_labels[CS*Ss:CS*Ss+CS*Sd]

        support_dom_labels_source = support_dom_labels[:CS*Ss]
        support_dom_labels_target = support_dom_labels[CS*Ss:CS*Ss+CS*Sd]

        support_labels_aug = support_labels_source*3 + support_labels_target*3
        support_dom_labels_aug = support_dom_labels_source*3 + support_dom_labels_target*3

        return tsupport_patches_concat_original_w_s, support_labels_aug, Ss_aug, Sd_aug, support_dom_labels_aug