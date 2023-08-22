#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
import numpy as np

#feature extractor
from models.Feature_extractor import CDFSOSR_Model
#outlier prediction network
from models.outlier_network import outlier_nn
#domain prediction network
from models.domain_network import domain_nn
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

def proto_test(support_images_tensor,query_images_tensor,support_labels_tensor,query_labels_tensor,support_classes,CS, CQ, Ss, Sd, Qsk, Qdk, Qsu, Qdu):#3,3,4,1,5,5,5,5    

    sembed = CDFSOSR_Model(support_images_tensor)                             # [x, 64]-No Aug-> [x, 64] (support)
    qembed = CDFSOSR_Model(query_images_tensor)                             # [p, 64] --> [p, 64] (query)

    qembed = tf.cast(qembed, dtype=tf.float64)

    sembed_s = []
    sembed_s_labels = []
    sembed_d = []
    sembed_d_labels = []
    for i in range(len(sembed)): #Support set is in the format [Ss1,Ss1,Ss1,Ss1,Sd1, Ss2,Ss2.....]
        if i % (Ss + Sd) < Ss:
            sembed_s.append(sembed[i])
            sembed_s_labels.append(support_labels_tensor[i]) #source domain support images
        else:
            sembed_d.append(sembed[i])
            sembed_d_labels.append(support_labels_tensor[i]) #target domain support images

    sembed_s = tf.convert_to_tensor(sembed_s, dtype=tf.float64) #(Ss*CS, 64)
    sembed_d = tf.convert_to_tensor(sembed_d, dtype=tf.float64) #(Sd*CS, 64)

    #==========================================================================================
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
    #support SOURCE DOMAIN

    z_prototypes_s = tf.reshape(sembed_s,[CS, Ss, sembed_s.shape[-1]])
    # prototypes SOURCE DOMAIN
    z_prototypes_s = tf.reduce_mean(z_prototypes_s, axis=1)

    #TARGET DOMAIN
    z_prototypes_d = tf.reshape(sembed_d,[CS, Sd, sembed_d.shape[-1]])
    # prototypes TARGET DOMAIN
    z_prototypes_d = tf.reduce_mean(z_prototypes_d , axis=1)

    #=======================================Adder 2===================================================
    #Adder 2 SOURCE DOMAIN   Just consider (unknown query) not known query
    query_Qsu = tf.convert_to_tensor(query_Qsu,dtype=tf.float64)
    #Adder 2 TARGET DOMAIN
    query_Qdu= tf.convert_to_tensor(query_Qdu,dtype=tf.float64)
    #=======================================Distances===================================================
    #euclidian distances:
    dists_s_k = calc_euclidian_dists2(query_Qsk, z_prototypes_s)
    dists_d_k = calc_euclidian_dists2(query_Qdk, z_prototypes_d)
    dists_s_u = calc_euclidian_dists2(query_Qsu, z_prototypes_s)
    dists_d_u = calc_euclidian_dists2(query_Qdu, z_prototypes_d)

    #for domain net
    dists_s = tf.concat((dists_s_k,dists_s_u),axis=0)
    dists_d = tf.concat((dists_d_k,dists_d_u),axis=0)
    #concatanate distances        
    dists = tf.concat((dists_s,dists_d),axis=0) #[CS*(Ss+Qsk+Qsu)+CS*(Sd+Qdk+Qdu), CS]
    
    log_p_y = tf.nn.log_softmax(-dists,axis=-1) #[CS*(Ss+Qsk+Qsu)+CS*(Sd+Qdk+Qdu), CS

    #for outlier net
    dists_inlier = tf.concat((dists_s_k,dists_d_k),axis=0)
    dists_outlier = tf.concat((dists_s_u,dists_d_u),axis=0)
    #concatanate distances 
    dists_o = tf.concat((dists_outlier,dists_inlier),axis=0)

    #=================================================================================================
    #outlier prediction labels
    outliers_source_o = [0]*len(query_Qsu)#1-->inlier 0-->outlier
    outliers_target_o = [0]*len(query_Qdu)
    inliers_source_o = [1]*len(query_Qsk)
    inliers_target_o = [1]*len(query_Qdk)
    y_outlier = outliers_source_o + outliers_target_o + inliers_source_o + inliers_target_o

    #domain prediction labels:
    inliers_source_d = [0]*len(query_Qsk) #0-->source 1-->target
    outliers_source_d = [0]*len(query_Qsu)
    inliers_target_d = [1]*len(query_Qdk)
    outliers_target_d = [1]*len(query_Qdu)
    y_domain = inliers_source_d + outliers_source_d + inliers_target_d + outliers_target_d

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
    query_Qsu_labels = [-1]*len(query_Qsu) #unknown class labels so it does not matter what we assign as label as long as it is not present in support classes
    true_query_labels_target_k = query_Qdk_labels #true labels for known class target domain
    query_Qdu_labels = [-1]*len(query_Qdu) #unknown class labels so it does not matter what we assign as label as long as it is not present in support classes
    
    #source unknown + target unknown + target known + source known ==> in dists_o calculation for reference
    true_query_labels_s = query_Qsu_labels + true_query_labels_source_k#source out + source in
    true_query_labels_d = query_Qdu_labels + true_query_labels_target_k#target out + target in

    #source
    #source unknown + target unknown + target known + source known ==> in dists_o calculation for reference
    pred_index_s_out = pred_index[:len(outliers_source_o)]
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

    for i in range(len(true_query_labels_s)) : #SOURCE DOMAIN 
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
    outlier_det_acc_s = outlier_s/(len(query_Qsu_labels)) #only unknown samples

    for i in range(len(true_query_labels_d)) : #TARGET DOMAIN 
      #if class is from support set
      if true_query_labels_d[i] in support_classes : #known class
        if outlier_index_d[i] == 1 :
          x = support_classes.index(true_query_labels_d[i])
          if x == pred_index_d[i] :
            correct_pred_d += 1
      #if it is open set class
      else : #unknown class
          if outlier_index_d[i] == 0:
            outlier_d = outlier_d + 1
            
    #only for query samples
    accuracy_d = correct_pred_d/(len(true_query_labels_target_k)) #only known samples from TARGET DOMAIN     
    outlier_det_acc_d = outlier_d/(len(query_Qdu_labels)) #only unknown samples

    accuracy = hmean([accuracy_s,accuracy_d])
    print(outlier_det_acc_s,outlier_det_acc_d)
    outlier_det_acc = hmean([outlier_det_acc_s,outlier_det_acc_d])

    y_pred_c_s = np.zeros((len(true_query_labels_source_k))) #all quries known from SOURCE
    for i in range(len(true_query_labels_source_k)) :
        if outlier_index_s_in[i] == 1 :
          y_pred_c_s[i] = pred_index_s_in[i]
          print(y_pred_c_s[i],true_query_labels_source_k[i])
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
          print(y_pred_c_d[i],true_query_labels_target_k[i])
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
    y_pred_o_s = np.zeros((len(true_query_labels_s))) #all quries known+unknown from SOURCE
    for i in range(len(true_query_labels_s)) :
        if outlier_index_s[i] == 1 :
          y_pred_o_s[i] = pred_index_s[i]
        else :
          y_pred_o_s[i] = -1 #Make sure 0 is not in support labels
    cm3 = confusion_matrix(true_query_labels_s, y_pred_o_s)
    FP = cm3.sum(axis=0) - np.diag(cm3)
    FN = cm3.sum(axis=1) - np.diag(cm3)
    TP = np.diag(cm3)
    TN = cm3.sum() - (FP + FN + TP)
    open_oa_s = (sum(TP)+sum(TN))/(sum(TP)+sum(TN)+sum(FP)+sum(FN))

    y_pred_o_d = np.zeros((len(true_query_labels_d))) #all quries known+unknown from TARGET
    for i in range(len(true_query_labels_d)) :
        if outlier_index[i] == 1 :
          y_pred_o_d[i] = pred_index_d[i]
        else :
          y_pred_o_d[i] = -1 #Make sure 0 is not in support labels
    cm4 = confusion_matrix(pred_index_d,y_pred_o_d)
    FP = cm4.sum(axis=0) - np.diag(cm4)
    FN = cm4.sum(axis=1) - np.diag(cm4)
    TP = np.diag(cm4)
    TN = cm4.sum() - (FP + FN + TP)
    open_oa_d = (sum(TP)+sum(TN))/(sum(TP)+sum(TN)+sum(FP)+sum(FN))

    open_oa = hmean([open_oa_s,open_oa_d])

    #domain prediction accuracy
    correct_domain=0
    dom_bin = tf.argmax(domain_pred,axis=-1)
    for i,j in zip(y_domain,dom_bin):
      if i==j:
        correct_domain+=1
    domain_acc = correct_domain/(len(dom_bin))

    return accuracy, closed_oa_cm, outlier_det_acc, open_oa, domain_acc   # scalar, scalar