#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
import numpy as np

#Optimizers for GANs
from optimizers.gan_optimizers import gan_reptile
from optimizers.gan_optimizers import gan_reptile_high
#loss
from loss.cross_domain_alignment import calc_weighted_gamma_dists
from loss.prototype_diversification_loss import calc_triplet_dists
#compute distances
from loss.euclidian_dist import calc_euclidian_dists2
#batch norm parameters
from utils.get_gamma_beta_from_layer import get_gamma_beta
#confusion matrix
from sklearn.metrics import confusion_matrix
#harmonic mean
from scipy.stats import hmean



#===============================================================Proto Train function===========================================================
def proto_train(support_images_tensor,query_images_tensor,support_labels_tensor,query_labels_tensor,support_classes,CS, CQ, Ss, Sd, Qsk, Qdk, Qsu, Qdu, query_dom_labels, support_dom_labels, CDFSOSR_Model, outlier_nn, domain_nn, generator_nn_low_s, dis_nn_low_s,generator_nn_high_s, dis_nn_high_s, ntimes, lr_schedule, optim_model_Outlier, optim_model_Domain, optim_d_low_s,optim_g_low_s, optim_d_high_s,optim_g_high_s):#3,3,4,1,5,5,5,5
    sembed = CDFSOSR_Model(support_images_tensor)                             # [x, 64]-Aug-> [3x, 64] (support)
    qembed = CDFSOSR_Model(query_images_tensor)                             # [p, 64] --> [p, 64] (query)

    qembed = tf.cast(qembed, dtype=tf.float64)
    #================================GAN PART==================================================
    # gan --> reptile
    #source domain+target domain
    ngan = 5
    for _ in range(ngan):
        sembed_low,vector = gan_reptile(sembed,support_labels_tensor,support_dom_labels,generator_nn_low_s,dis_nn_low_s,optim_d_low_s,optim_g_low_s,stdev=0.2,alpha1=0.003)
        sembed_high = gan_reptile_high(sembed,sembed_low,vector,support_labels_tensor,support_dom_labels,generator_nn_high_s,dis_nn_high_s,optim_d_high_s,optim_g_high_s,stdev=1.0,alpha1=0.003)

    #==========================================================================================
    sembed_low_s = []
    sembed_high_s = []
    sembed_s_labels = []
    sembed_s = []
    sembed_low_d = []
    sembed_high_d = []
    sembed_d_labels = []
    sembed_d = []
    for i in range(len(sembed)): #Support set is in the format [Ss1,Ss1,Ss1,Ss1,Sd1, Ss2,Ss2.....]
        if i < CS*Ss:
            sembed_low_s.append(sembed_low[i])
            sembed_high_s.append(sembed_high[i])
            sembed_s.append(sembed[i])
            sembed_s_labels.append(support_labels_tensor[i]) #source domain support images
        else:
            sembed_low_d.append(sembed_low[i])
            sembed_high_d.append(sembed_high[i])
            sembed_d.append(sembed[i])
            sembed_d_labels.append(support_labels_tensor[i]) #target domain support images

    sembed_s = tf.convert_to_tensor(sembed_s, dtype=tf.float64) #(CS*Ss, 64)
    sembed_d = tf.convert_to_tensor(sembed_d, dtype=tf.float64) #(CS*Sd, 64)
    sembed_low_s = tf.convert_to_tensor(sembed_low_s, dtype=tf.float64) #(CS*Ss, 64)
    sembed_low_d = tf.convert_to_tensor(sembed_low_d, dtype=tf.float64) #(CS*Sd, 64)
    sembed_high_s = tf.convert_to_tensor(sembed_high_s, dtype=tf.float64) #(CS*Ss, 64)
    sembed_high_d = tf.convert_to_tensor(sembed_high_d, dtype=tf.float64) #(CS*Sd, 64)
    sembed_s_labels = tf.convert_to_tensor(sembed_s_labels, dtype=tf.float64) #(CS*Ss, 64)
    sembed_d_labels = tf.convert_to_tensor(sembed_d_labels, dtype=tf.float64) #(CS*Sd, 64)

    # Calculate the number of elements in each of the 4 arrays
    num_elements_qsk = CS * Qsk
    num_elements_qdk = CS * Qdk
    num_elements_qsu = CQ * Qsu
    num_elements_qdu = CQ * Qdu

    # Slice the query images embeddings to get the 4 arrays for calculating triplet loss
    query_Qsk = qembed[:num_elements_qsk]
    query_Qdk = qembed[num_elements_qsk:num_elements_qsk+num_elements_qdk]
    query_Qsu = qembed[num_elements_qsk+num_elements_qdk:num_elements_qsk+num_elements_qdk+num_elements_qsu]
    query_Qdu = qembed[num_elements_qsk+num_elements_qdk+num_elements_qsu:]

    # Slice the query labels tensor
    query_Qsk_labels = query_labels_tensor[:num_elements_qsk]
    query_Qdk_labels = query_labels_tensor[num_elements_qsk:num_elements_qsk+num_elements_qdk]

    #=======================================Prototypes===================================================
    #support + low noise gan SOURCE DOMAIN
    z_proto_low_s = tf.reshape(sembed_low_s,[CS, Ss, sembed_low_s.shape[-1]])
    z_prototypes_s = tf.reshape(sembed_s,[CS, Ss, sembed_s.shape[-1]])           #weak + strong + original support images
    z_prototypes_s_combined = np.empty((CS,Ss*2,sembed_s.shape[-1]), dtype=np.float64)

    #combine the prototypes from original images and GAN output for SOURCE DOMAIN
    #Adder 1 source
    for i in range(z_prototypes_s_combined.shape[0]):
      z_prototypes_s_combined[i,:,:] = tf.concat((tf.cast(z_prototypes_s[i,:,:], dtype=tf.float64),tf.cast(z_proto_low_s[i,:,:], dtype=tf.float64)),axis=0) #[original source support + low noise GAN output for source support]

    # prototypes SOURCE DOMAIN
    z_prototypes_s_combined = tf.reshape(z_prototypes_s_combined, [CS,Ss*2, sembed_s.shape[-1]])
    z_prototypes_s_combined = tf.reduce_mean(z_prototypes_s_combined, axis=1)

    #support + low noise gan TARGET DOMAIN
    z_proto_low_d = tf.reshape(sembed_low_d,[CS, Sd, sembed_low_d.shape[-1]])
    z_prototypes_d = tf.reshape(sembed_d,[CS, Sd, sembed_d.shape[-1]])
    z_prototypes_d_combined = np.empty((CS,Sd*2,sembed_d.shape[-1]), dtype=np.float64)

    #combine the prototypes from original images and GAN output for SOURCE DOMAIN
    #Adder 1 target
    for i in range(z_prototypes_d_combined.shape[0]):
      z_prototypes_d_combined[i,:,:] = tf.concat((tf.cast(z_prototypes_d[i,:,:], dtype=tf.float64),tf.cast(z_proto_low_d[i,:,:], dtype=tf.float64)),axis=0) #[original target support + low noise GAN output for target support]

    # prototypes TARGET DOMAIN
    z_prototypes_d_combined = tf.reshape(z_prototypes_d_combined, [CS,Sd*2, sembed_d.shape[-1]])
    z_prototypes_d_combined = tf.reduce_mean(z_prototypes_d_combined, axis=1)

    #=======================================Adder 2===================================================
    #Adder 2 SOURCE DOMAIN   Just consider (unknown query and high noise augmented) not known query
    qembedU_high_s = tf.convert_to_tensor(tf.concat((sembed_high_s,query_Qsu),axis=0),dtype=tf.float64)
    #Adder 2 TARGET DOMAIN
    qembedU_high_d = tf.convert_to_tensor(tf.concat((sembed_high_d,query_Qdu),axis=0),dtype=tf.float64)
    #=======================================Distances===================================================
    #euclidian distances:
    dists_s_k = calc_euclidian_dists2(query_Qsk, z_prototypes_s_combined) #z_prototypes_s_combined with known query samples concatenated with z_prototypes_s_combined with unknown query samples
    dists_d_k = calc_euclidian_dists2(query_Qdk, z_prototypes_d_combined)
    dists_s_u = calc_euclidian_dists2(qembedU_high_s, z_prototypes_s_combined) #z_prototypes_s_combined with known query samples concatenated with z_prototypes_s_combined with unknown query samples
    dists_d_u = calc_euclidian_dists2(qembedU_high_d, z_prototypes_d_combined)

    #for domain net
    dists_s = tf.concat((dists_s_k,dists_s_u),axis=0)
    dists_d = tf.concat((dists_d_k,dists_d_u),axis=0)

    #concatanate distances
    dists = tf.concat((dists_s,dists_d),axis=0) #[CS*(Ss+Qsk+Qsu)+CS*(Sd+Qdk+Qdu), CS]

    log_p_y = tf.nn.log_softmax(-dists,axis=-1) ##[CS*(Ss+Qsk+Qsu)+CS*(Sd+Qdk+Qdu), CS]

    #for outlier net
    dists_inlier = tf.concat((dists_s_k,dists_d_k),axis=0)
    dists_outlier = tf.concat((dists_s_u,dists_d_u),axis=0)
    #concatanate distances
    dists_o = tf.concat((dists_outlier,dists_inlier),axis=0)

    #=====================================Domain + Outlier separate Networks==============================
    #two types of labels: 1) outlier prediction and 2)domain prediction
    #two types of labels: 1) outlier prediction and 2)domain prediction
    #outlier prediction labels
    outliers_source_o = [0]*len(qembedU_high_s)#1-->inlier 0-->outlier
    outliers_target_o = [0]*len(qembedU_high_d)
    inliers_source_o = [1]*len(query_Qsk)
    inliers_target_o = [1]*len(query_Qdk)
    y_outlier = outliers_source_o + outliers_target_o + inliers_source_o + inliers_target_o

    #domain prediction labels:
    inliers_source_d = [0]*len(query_Qsk) #0-->source 1-->target
    outliers_source_d = [0]*len(qembedU_high_s)
    inliers_target_d = [1]*len(query_Qdk)
    outliers_target_d = [1]*len(qembedU_high_d)
    y_domain = inliers_source_d + outliers_source_d + inliers_target_d + outliers_target_d

    for i in range(ntimes):
      with tf.GradientTape() as Outlier_tape:
        outlier_pred = outlier_nn(dists_o)
        outlier_pred = tf.squeeze(outlier_pred)
        loss_outlier = tf.keras.losses.SparseCategoricalCrossentropy()(y_outlier, outlier_pred)#0,1

      gradients_out = Outlier_tape.gradient(loss_outlier, outlier_nn.trainable_variables)
      # Compute the learning rate using the schedule
      learning_rate = lr_schedule(optim_model_Outlier.iterations)
      optim_model_Outlier.learning_rate = learning_rate
      optim_model_Outlier.apply_gradients(zip(gradients_out, outlier_nn.trainable_variables))

    for i in range(ntimes):
      with tf.GradientTape() as Domain_tape:
        domain_pred = domain_nn(dists)
        domain_pred = tf.squeeze(domain_pred)
        loss_domain = tf.keras.losses.SparseCategoricalCrossentropy()(y_domain, domain_pred)#0,1

      gradients_dom = Domain_tape.gradient(loss_domain, domain_nn.trainable_variables)
      # Compute the learning rate using the schedule
      learning_rate = lr_schedule(optim_model_Domain.iterations)
      optim_model_Domain.learning_rate = learning_rate
      optim_model_Domain.apply_gradients(zip(gradients_dom, domain_nn.trainable_variables))

    #======================================Accuracy===================================================
    #accuracy calculation
    outlier_pred = outlier_nn(dists_o)
    domain_pred = domain_nn(dists)

    outlier_index = tf.squeeze(tf.argmax(outlier_pred,axis=-1))

    #check the dimesions here before using this for calculating accuracy later
    predictions = tf.nn.softmax(-dists_o, axis=-1) #[0,0,1] [0,0,0] ....

    #return the index of the maximum value
    pred_index = tf.squeeze(tf.argmax(predictions,axis=-1))# for all query samples

    true_query_labels_source_k = query_Qsk_labels #true labels for known class source domain
    qembedU_high_s_labels = [-1]*len(qembedU_high_s) #unknown class labels so it does not matter what we assign as label as long as it is not present in support classes
    true_query_labels_target_k = query_Qdk_labels #true labels for known class target domain
    qembedU_high_d_labels = [-1]*len(qembedU_high_d) #unknown class labels so it does not matter what we assign as label as long as it is not present in support classes

    #source unknown + target unknown + target known + source known ==> in dists_o calculation for reference
    true_query_labels_s = qembedU_high_s_labels + true_query_labels_source_k#source out + source in
    true_query_labels_d = qembedU_high_d_labels + true_query_labels_target_k#target out + target in

    #source
    #source unknown + target unknown + target known + source known ==> in dists_o calculation for reference
    pred_index_s_out = pred_index[:len(outliers_source_o)]
    print(len(outliers_source_o)+len(outliers_target_o),":",len(outliers_source_o)+len(outliers_target_o)+len(inliers_source_o))
    pred_index_s_in = pred_index[len(outliers_source_o)+len(outliers_target_o):len(outliers_source_o)+len(outliers_target_o)+len(inliers_source_o)]
    pred_index_s = tf.concat((pred_index_s_out,pred_index_s_in),axis=0) #source out + source in
    #target
    pred_index_d_out = pred_index[len(outliers_source_o):len(outliers_source_o)+len(outliers_target_o)]
    pred_index_d_in = pred_index[len(outliers_source_o)+len(outliers_target_o)+len(inliers_source_o):]
    pred_index_d = tf.concat((pred_index_d_out,pred_index_d_in),axis=0) #target out + target in

    #source
    #source unknown + target unknown + target known + source known ==> in dists_o calculation for reference
    outlier_index_s_out = outlier_index[:len(outliers_source_o)]
    outlier_index_s_in = outlier_index[len(outliers_source_o)+len(outliers_target_o):len(outliers_source_o)+len(outliers_target_o)+len(inliers_source_o)]
    outlier_index_s = tf.concat((outlier_index_s_out, outlier_index_s_in),axis=0) #source out + source in
    #target
    outlier_index_d_out = outlier_index[len(outliers_source_o):len(outliers_source_o)+len(outliers_target_o)]
    outlier_index_d_in = outlier_index[len(outliers_source_o)+len(outliers_target_o)+len(inliers_source_o):]
    outlier_index_d = tf.concat((outlier_index_d_out,outlier_index_d_in),axis=0) #target out + target in

    #source unknown + target unknown + target known + source known ==> in dists_o calculation for reference
    predictions_s_out = predictions[:len(outliers_source_o),:]
    predictions_s_in = predictions[len(outliers_source_o)+len(outliers_target_o):len(outliers_source_o)+len(outliers_target_o)+len(inliers_source_o),:]
    predictions_s = tf.concat((predictions_s_out, predictions_s_in),axis=0) #source out + source in
    #target
    predictions_d_out = predictions[len(outliers_source_o):len(outliers_source_o)+len(outliers_target_o),:]
    predictions_d_in = predictions[len(outliers_source_o)+len(outliers_target_o)+len(inliers_source_o):,:]
    predictions_d = tf.concat((predictions_d_out, predictions_d_in),axis=0) #target out + target in
    #rearranged predictions ==>source+target
    predictions_rearranged = tf.concat((predictions_s, predictions_d),axis=0)

    correct_pred_s = 0
    outlier_s = 0
    correct_pred_d = 0
    outlier_d = 0

    for i in range(len(true_query_labels_s)) : #known and unknown samples from SOURCE DOMAIN
      #if class is from support set
      if true_query_labels_s[i] in support_classes : #known class
        if outlier_index_s[i] == 1 :
          x = support_classes.index(true_query_labels_s[i])
          if x == pred_index_s[i] :
            correct_pred_s += 1
      #if it is open set class
      else : #unknown class
          if outlier_index_s[i] == 0 :
            outlier_s = outlier_s + 1

    #only for query samples
    accuracy_s = correct_pred_s/(len(true_query_labels_source_k)) #only known samples from SOURCE DOMAIN
    outlier_det_acc_s = outlier_s/(len(qembedU_high_s_labels)) #only unknown samples

    for i in range(len(true_query_labels_d)) : #known and unknown samples from TARGET DOMAIN
      #if class is from support set
      if true_query_labels_d[i] in support_classes : #known class
        if outlier_index_d[i] == 1 :
          x = support_classes.index(true_query_labels_d[i])
          if x == pred_index_d[i] :
            correct_pred_d += 1
      #if it is open set class
      else : #unknown class
          if outlier_index_d[i] == 0 :
            outlier_d = outlier_d + 1

    #only for query samples
    accuracy_d = correct_pred_d/(len(true_query_labels_target_k)) #only known samples from TARGET DOMAIN
    outlier_det_acc_d = outlier_d/(len(qembedU_high_d_labels)) #only unknown samples

    accuracy = hmean([accuracy_s,accuracy_d])
    outlier_det_acc = hmean([outlier_det_acc_s,outlier_det_acc_d])

    y_pred_c_s = np.zeros((len(true_query_labels_source_k))) #all quries known from SOURCE
    for i in range(len(true_query_labels_source_k)) :
        if outlier_index_s_in[i] == 1 :
          y_pred_c_s[i] = pred_index_s_in[i]
        else :
          y_pred_c_s[i] = -1
    cm1 = confusion_matrix(true_query_labels_source_k,y_pred_c_s)
    FP = cm1.sum(axis=0) - np.diag(cm1)
    FN = cm1.sum(axis=1) - np.diag(cm1)
    TP = np.diag(cm1)
    TN = cm1.sum() - (FP + FN + TP)
    closed_oa_s = (sum(TP)+sum(TN))/(sum(TP)+sum(TN)+sum(FP)+sum(FN))

    y_pred_c_d = np.zeros((len(true_query_labels_target_k))) #all quries known from TARGET
    for i in range(len(true_query_labels_target_k)) :
        if outlier_index_d_in[i] == 1 :
          y_pred_c_d[i] = pred_index_d_in[i]
        else :
          y_pred_c_d[i] = -1 #Make sure 0 is not in support labels
    cm2 = confusion_matrix(true_query_labels_target_k,y_pred_c_d)
    FP = cm2.sum(axis=0) - np.diag(cm2)
    FN = cm2.sum(axis=1) - np.diag(cm2)
    TP = np.diag(cm2)
    TN = cm2.sum() - (FP + FN + TP)
    closed_oa_d = (sum(TP)+sum(TN))/(sum(TP)+sum(TN)+sum(FP)+sum(FN))

    closed_oa_cm = hmean([closed_oa_s, closed_oa_d])

    #open oa
    y_pred = np.zeros((len(true_query_labels_s))) #all quries known+unknown from SOURCE
    for i in range(len(true_query_labels_s)) :
        if outlier_index_s[i] == 1 :
          y_pred[i] = pred_index_s[i]
        else :
          y_pred[i] = -1 #Make sure 0 is not in support labels
    cm2 = confusion_matrix(true_query_labels_s,y_pred)
    FP = cm2.sum(axis=0) - np.diag(cm2)
    FN = cm2.sum(axis=1) - np.diag(cm2)
    TP = np.diag(cm2)
    TN = cm2.sum() - (FP + FN + TP)
    open_oa_s = (sum(TP)+sum(TN))/(sum(TP)+sum(TN)+sum(FP)+sum(FN))

    y_pred = np.zeros((len(true_query_labels_d))) #all quries known+unknown from TARGET
    for i in range(len(true_query_labels_d)) :
        if outlier_index[i] == 1 :
          y_pred[i] = pred_index_d[i]
        else :
          y_pred[i] = -1 #Make sure 0 is not in support labels
    cm2 = confusion_matrix(pred_index_d,y_pred)
    FP = cm2.sum(axis=0) - np.diag(cm2)
    FN = cm2.sum(axis=1) - np.diag(cm2)
    TP = np.diag(cm2)
    TN = cm2.sum() - (FP + FN + TP)
    open_oa_d = (sum(TP)+sum(TN))/(sum(TP)+sum(TN)+sum(FP)+sum(FN))

    open_oa = hmean([open_oa_s,open_oa_d])

    one_hot_labels_s = np.asarray(np.zeros((len(true_query_labels_s),CS)),dtype=np.float64)  # (*, 3)
    one_hot_labels_d = np.asarray(np.zeros((len(true_query_labels_d),CS)),dtype=np.float64)  # (*, 3)
    #map source query labels irrespective of class to [0. 0. 0.],[0. 0. 1.],[0. 1. 0.][1. 0. 0.
    for i in range(len(true_query_labels_s)) :
      if true_query_labels_s[i] in support_classes :
            x = support_classes[:CS].index(true_query_labels_s[i])
            one_hot_labels_s[i][x] = 1.                                      # [[0., 0., 1.], [0., 0., 0.], ... (*,3)
            print(i,x)
    #map target query labels irrespective of class to [0. 0. 0.],[0. 0. 1.],[0. 1. 0.][1. 0. 0.]
    for i in range(len(true_query_labels_d)) :
      if true_query_labels_d[i] in support_classes :
            x = support_classes[CS:].index(true_query_labels_d[i])
            one_hot_labels_d[i][x] = 1.                                      # [[0., 0., 1.], [0., 0., 0.], ... (*,3)

    #domain prediction accuracy
    correct_domain=0
    dom_bin = tf.argmax(domain_pred,axis=-1)
    for i,j in zip(y_domain,dom_bin):
      if i==j:
        correct_domain+=1
    domain_acc = correct_domain/(len(dom_bin))

    #FE loss
    quad_dis_s =  calc_triplet_dists(z_prototypes_s_combined, tf.concat((qembedU_high_s,query_Qsk),axis=0), one_hot_labels_s)
    quad_dis_d =  calc_triplet_dists(z_prototypes_d_combined, tf.concat((qembedU_high_d,query_Qdk),axis=0), one_hot_labels_d)
    #add P_loss
    Gamma_D1,Gamma_D2,Beta_D1,Beta_D2 = get_gamma_beta(CDFSOSR_Model)

    cross_domain_align_loss = calc_weighted_gamma_dists(z_prototypes_s_combined,z_prototypes_d_combined,Gamma_D1,Gamma_D2,Beta_D1,Beta_D2)

    cec_loss = -tf.reduce_mean((tf.reduce_sum(tf.multiply(tf.cast(tf.concat((one_hot_labels_s,one_hot_labels_d),axis=0),dtype=tf.float64), predictions_rearranged), axis=-1)))

    ## Choose appropriate weighting scalars for the loss acccording to your dataset
    loss = tf.cast(0.4*100*cec_loss, tf.float64) + tf.cast(0.3*loss_outlier, tf.float64) + tf.cast(0.05*loss_domain, tf.float64) + tf.cast(0.075*quad_dis_s, tf.float64) + tf.cast(0.125*quad_dis_d, tf.float64) + tf.cast(0.05*1000*cross_domain_align_loss, tf.float64)

    return loss, accuracy, closed_oa_cm, outlier_det_acc, open_oa, domain_acc   # scalar, scalar