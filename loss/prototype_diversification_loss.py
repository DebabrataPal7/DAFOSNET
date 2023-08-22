#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
import numpy as np

#==========================Prototype Diversification Loss==========================
#calculate prototype diversification loss (triplet loss)
def calc_triplet_dists(Proto, query, query_labels, Alpha=0.5):  # [3,64], [90,64], [90,3]

    nbprototypes = Proto.shape[0]    # 3
    nbqueries = query.shape[0]             # 90
    Triplet = []

    for i in range(nbprototypes):
        pos_ind = np.where(query_labels[:,i]==1)   # indexes of Positive queries for i th prototype
        pos_ind = list(pos_ind[0])            # indexes of Positive queries for i th prototype
        Neg_ind = list(set(np.arange(nbqueries)).difference(set(pos_ind)))  # indexes of Negative queries for i th prototype
        Anchor = tf.expand_dims(Proto[i], 0)     # [1, 64]       # ith Prototype / Anchor

        Positive_d = []
        for j in range(len(pos_ind)):   # 15
            Pos = tf.expand_dims(query[pos_ind[j]], 0)   # [1, 64]
            Pos_dist = tf.reduce_mean(tf.math.pow(Anchor-Pos, 2), 1)   # scalar
            Positive_d.append(Pos_dist)                    # List
        Positive_d = tf.reduce_mean(Positive_d) # scalar

        Negative_d = []
        for j in range(len(Neg_ind)):
            Neg = tf.expand_dims(query[Neg_ind[j]], 0)   # [1, 64]
            Neg_dist = tf.reduce_mean(tf.math.pow(Anchor-Neg, 2), 1)   # scalar
            Negative_d.append(Neg_dist)                     # List
        Negative_d = tf.reduce_mean(Negative_d) # scalar

        P_N = Positive_d-Negative_d
        print(P_N)
        Triplet.append(P_N)            #list

    P_N_Proto_mean = tf.reduce_mean(Triplet)    # p_dist - n_dist  # averaged for all Prototypes   # scalar
    return tf.math.maximum(P_N_Proto_mean + Alpha, 0)  # scalar