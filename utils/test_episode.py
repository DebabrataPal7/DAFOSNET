#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
import tensorflow as tf
import numpy as np

def test_episode(source_domain, target_domain, CS, CQ, Ss, Sd, Qsk, Qdk, Qsu, Qdu, class_labels):
    # Select CS classes from source_domain
    support_classes = list(class_labels[:CS])
    # Select Ss samples from the selected classes in source domain as part of the support set
    tsupport_patches_d, support_labels_d = [], []
    # Select Sd samples from the selected classes in target domain as part of the support set
    tsupport_patches_t, support_labels_t = [], []
    # Select Qsk images from the previously selected class in (1a) which were NOT SELECTED EARLIER from domain 1
    known_query_patches_1, known_query_labels_1 = [], []
    # Select the same images (Qdk=Qsk) in (2a) for the same previously selected class in (1a) from domain 2
    known_query_patches_2, known_query_labels_2 = [], []
    if Ss>0:
      for x in range(len(support_classes)):
          sran_indices = np.random.choice(source_domain[x].shape[0], Ss, replace=False)
          support_patches = source_domain[x][sran_indices,:,:,:]
          tsupport_patches_d.extend(support_patches)
          for i in range(Ss):
              support_labels_d.append(x)

          unselected_indices = [j for j in list(range(source_domain[x].shape[0])) if j not in sran_indices] # to make sure there is no overlap between source and query images
          qran_indices_known = np.random.choice(unselected_indices, Qsk, replace=False)
          query_patches_1 = source_domain[x][qran_indices_known,:,:,:]
          known_query_patches_1.extend(query_patches_1)
          for i in range(Qsk):
              known_query_labels_1.append(x)
              
          if Sd>0:
            sran_indices = np.random.choice(target_domain[x].shape[0], Sd, replace=False)
            support_patches = target_domain[x][sran_indices,:,:,:]
            tsupport_patches_t.extend(support_patches)
            for i in range(Sd):
              support_labels_t.append(x)

            unselected_indices = [j for j in list(range(target_domain[x].shape[0])) if j not in sran_indices] # to make sure there is no overlap between source and query images
            qran_indices_known = np.random.choice(unselected_indices, Qdk, replace=False)
            query_patches_2 = target_domain[x][qran_indices_known,:,:,:]
            known_query_patches_2.extend(query_patches_2)
            for i in range(Qsk):
                known_query_labels_2.append(x)

    # 1c and 1d together form the support set
    tsupport_patches = tsupport_patches_d + tsupport_patches_t     # They are python arrays hence adding, not tf.tensors
    support_labels = support_labels_d + support_labels_t

    # 2a and 2b together form the known classes query set
    tquery_patches = known_query_patches_1 + known_query_patches_2      # They are python arrays hence adding, not tf.tensors
    query_labels = known_query_labels_1 + known_query_labels_2

    # Select in other CQ classes which was NOT SELECTED EARLIER in (1a) from domain 1 as a part of unknown query set
    other_classes = [c for c in class_labels if c not in support_classes]
    selected_classes = list(np.random.choice(other_classes, CQ, replace=False))

    # Randomly select Qsu images from the selected classes (CQ) in (2d) from domain 1 as a part of unknown query set
    unknown_query_patches_1, unknown_query_labels_1 = [], []
    # Randomly select Qdu images from the selected classes (CQ) in (2d) from domain 2 as a part of unknown query set
    unknown_query_patches_2, unknown_query_labels_2 = [], []
    if Qsu>0:
      for x in range(len(selected_classes)):
          qran_indices_unknown = np.random.choice(source_domain[x].shape[0], Qsu, replace=False)
          query_patches_1 = source_domain[x][qran_indices_unknown,:,:,:]
          unknown_query_patches_1.extend(query_patches_1)
          for i in range(Qsu):
              unknown_query_labels_1.append(x)
          if Qdu>0:
            query_patches_2 = target_domain[x][qran_indices_unknown,:,:,:]
            unknown_query_patches_2.extend(query_patches_2)
            for i in range(Qdu):
                unknown_query_labels_2.append(x)

    # 2e and 2f together form the unknown classes query set
    unknown_query_patches = unknown_query_patches_1 + unknown_query_patches_2  # They are python arrays hence adding, not tf.tensors
    unknown_query_labels = unknown_query_labels_1 + unknown_query_labels_2

    # Concatenate the known and unknown query sets
    tquery_patches += unknown_query_patches
    query_labels += unknown_query_labels

    # Convert the lists to tensors
    tquery_patches = tf.convert_to_tensor(tquery_patches, dtype=tf.float64)/255.0  # Normalize         # (Qsk+Qdk)*CS + (Qsu+Qdu)*CQ) ,image_height,image_width,channels)
    tsupport_patches = tf.convert_to_tensor(tsupport_patches, dtype=tf.float64)/255.0  # Normalize     #( (CS*Ss+CS*Sd),image_height,image_width,channels)

    return tquery_patches, tsupport_patches, query_labels, support_labels, support_classes