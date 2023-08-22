#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
import tensorflow as tf
import numpy as np

#====================================Data Loader====================================:
def new_episode(source_domain, target_domain, CS, CQ, Ss, Sd, Qsk, Qdk, Qsu, Qdu, class_labels):
    # Select CS(source) + CS(target) support classes from the universal set
    support_classes = list(np.random.choice(class_labels, CS*2, replace=False))
    # Select CS classes from source_domain
    source_support_classes = list(np.random.choice(support_classes, CS, replace=False))
    # Select Ss samples from the selected classes in source domain as part of the support set
    tsupport_patches_d, support_labels_d, support_dom_labels_d = [], [], []
    # Select Sd samples from the selected classes in target domain as part of the support set
    tsupport_patches_t, support_labels_t, support_dom_labels_t = [], [], []
    # Select Qsk images from the previously selected class in (1a) which were NOT SELECTED EARLIER from domain 1
    known_query_patches_1, known_query_labels_1, known_query_dom_labels_1 = [], [], []
    # Select the same images (Qdk=Qsk) in (2a) for the same previously selected class in (1a) from domain 2
    known_query_patches_2, known_query_labels_2, known_query_dom_labels_2 = [], [], []

    if Ss>0:
      for x in source_support_classes:
          sran_indices = np.random.choice(source_domain[x].shape[0], Ss, replace=False)
          support_patches = source_domain[x][sran_indices,:,:,:]
          tsupport_patches_d.extend(support_patches)
          support_dom_labels_d.extend([0]*len(sran_indices))
          for i in range(Ss):
              support_labels_d.append(x)

          unselected_indices = [j for j in list(range(source_domain[x].shape[0])) if j not in sran_indices] #to make sure there is no overlap between source and query images
          qran_indices_known = np.random.choice(unselected_indices, Qsk, replace=False)
          query_patches_1 = source_domain[x][qran_indices_known,:,:,:]
          known_query_patches_1.extend(query_patches_1)
          known_query_dom_labels_1.extend([0]*len(qran_indices_known))
          for i in range(Qsk):
              known_query_labels_1.append(x)

    target_support_classes =  [c for c in support_classes if c not in source_support_classes]
    if Sd>0:
      for x in target_support_classes:
        sran_indices = np.random.choice(target_domain[x].shape[0], Sd, replace=False)
        support_patches = target_domain[x][sran_indices,:,:,:]
        support_dom_labels_t.extend([1]*len(sran_indices))
        tsupport_patches_t.extend(support_patches)
        for i in range(Sd):
          support_labels_t.append(x)

        unselected_indices = [j for j in list(range(target_domain[x].shape[0])) if j not in sran_indices] #to make sure there is no overlap between source and query images
        qran_indices_known = np.random.choice(unselected_indices, Qdk, replace=False)
        query_patches_2 = target_domain[x][qran_indices_known,:,:,:]
        known_query_patches_2.extend(query_patches_2)
        known_query_dom_labels_2.extend([1]*len(qran_indices_known))
        for i in range(Qsk):
            known_query_labels_2.append(x)

    # 1c and 1d together form the support set
    tsupport_patches = tsupport_patches_d + tsupport_patches_t     # They are python arrays hence adding, not tf.tensors
    support_labels = support_labels_d + support_labels_t
    support_dom_labels = support_dom_labels_d + support_dom_labels_t

    # 2a and 2b together form the known classes query set
    tquery_patches = known_query_patches_1 + known_query_patches_2      # They are python arrays hence adding, not tf.tensors
    query_labels = known_query_labels_1 + known_query_labels_2
    query_dom_labels = known_query_dom_labels_1 + known_query_dom_labels_2

    # Select in other CQ classes which was NOT SELECTED EARLIER in (1a) from domain 1 as a part of unknown query set
    other_classes = [c for c in class_labels if c not in support_classes]
    # print('Unselected classes:', other_classes)
    unknown_classes = list(np.random.choice(other_classes, CQ*2, replace=False))
    source_unknown_classes = list(np.random.choice(unknown_classes, CQ, replace=False))
    # Randomly select Qsu images from the selected classes (CQ) in (2d) from domain 1 as a part of unknown query set
    unknown_query_patches_1, unknown_query_labels_1, unknown_query_labels_dom_1 = [], [], []
    # Randomly select Qdu images from the selected classes (CQ) in (2d) from domain 2 as a part of unknown query set
    unknown_query_patches_2, unknown_query_labels_2, unknown_query_labels_dom_2 = [], [], []
    if Qsu>0:
      for x in source_unknown_classes:
          qran_indices_unknown = np.random.choice(source_domain[x].shape[0], Qsu, replace=False)
          query_patches_1 = source_domain[x][qran_indices_unknown,:,:,:]
          unknown_query_patches_1.extend(query_patches_1)
          unknown_query_labels_dom_1.extend([0]*len(qran_indices_unknown))
          for i in range(Qsu):
              unknown_query_labels_1.append(x)

    source_unknown_and_support_classes = source_unknown_classes + support_classes
    target_unknown_classes = [c for c in class_labels if c not in source_unknown_and_support_classes]

    if Qdu>0:
      for x in target_unknown_classes:
        query_patches_2 = target_domain[x][qran_indices_unknown,:,:,:]
        unknown_query_patches_2.extend(query_patches_2)
        unknown_query_labels_dom_2.extend([1]*len(qran_indices_unknown))
        for i in range(Qdu):
            unknown_query_labels_2.append(x)

    # 2e and 2f together form the unknown classes query set
    unknown_query_patches = unknown_query_patches_1 + unknown_query_patches_2  # They are python arrays hence adding, not tf.tensors
    unknown_query_labels = unknown_query_labels_1 + unknown_query_labels_2
    unknown_query_labels_dom = unknown_query_labels_dom_1 + unknown_query_labels_dom_2

    # Concatenate the known and unknown query sets
    tquery_patches += unknown_query_patches
    query_labels += unknown_query_labels
    query_dom_labels += unknown_query_labels_dom

    # Convert the lists to tensors
    #rearrange support classes in following format [:CS]-->source [CS:]-->target
    support_classes_rearranged = source_support_classes + target_support_classes
    tquery_patches = tf.convert_to_tensor(tquery_patches, dtype=tf.float64)/255.0  # Normalize         # (Qsk+Qdk)*CS + (Qsu+Qdu)*CQ) ,image_height,image_width,channels)
    tsupport_patches = tf.convert_to_tensor(tsupport_patches, dtype=tf.float64)/255.0  # Normalize     #( (CS*Ss+CS*Sd),image_height,image_width,channels)

    return tquery_patches, tsupport_patches, query_labels, support_labels, support_classes_rearranged, query_dom_labels, support_dom_labels