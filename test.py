#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
import tensorflow as tf
import tensorflow.keras.backend as K
import os
import json
import numpy as np
import datetime
from sklearn.metrics import  confusion_matrix
from scipy.stats import hmean

#save image batches to numpy files for faster processing
from data.save_data_to_numpy import save_data
#feature extractor
from models.Feature_extractor import CDFSOSR_Model
#outlier prediction network
from models.outlier_network import outlier_nn
#domain prediction network
from models.domain_network import domain_nn
#low noise GANs
from models.low_noise_GAN import generator_nn_low_s
from models.low_noise_GAN import dis_nn_low_s
#high noise GANs
from models.high_noise_GAN import generator_nn_high_s
from models.high_noise_GAN import dis_nn_high_s
#custom metrics (Variance)
from utils.variance_metric import ComputeIterationVaraince
#save accuracy values
from utils.save_accuracy_values import save_accuracy_values
#proto train
from utils.proto_test import proto_test
#new train episode
from utils.test_episode import test_episode

with open('./config.json') as json_file:
    config = json.load(json_file)

base_dir = config['base_dir']
save_path_source = config['save_path_source']
save_path_target = config['save_path_target']
num_classes = config['num_classes']#train_classes+test_classes
num_images_per_class = config['num_images_per_class']

# Hyperparameters for Support and Query images
CS=config['CS']
CQ=config['CQ']   #Not necessarily equal
Ss=config['Ss']
Sd=config['Sd']
Qsk=config['Qsk']
Qdk=config['Qdk']#N
Qsu=config['Qsu']
Qdu=config['Qdu']#X

# Test Classes
test_classes = config['test_classes']

#embedding dimensions
emb_dim = config['emb_dim']

#load the saved arrays into the memory
source_domain = np.load(f'{save_path_source}/source_{num_classes}.npy')
target_domain = np.load(f'{save_path_target}/target_{num_classes}.npy')

# Numeric class labels
class_labels = []

for i in range(len(source_domain)):
  class_labels.append(i)

print("source_domain shape: ", source_domain.shape)
print("target_domain shape: ", target_domain.shape)
print("class_labels: ", class_labels)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

# Specify the directory where the existing checkpoint is saved
existing_checkpoint_dir = config["current_checkpoint_dir"]
existing_checkpoint_prefix = os.path.join(existing_checkpoint_dir, "ckpt")
# Load the existing checkpoint variables into the current model
existing_checkpoint = tf.train.Checkpoint(
                                 CDFSOSR_Model = CDFSOSR_Model,
                                 generator_nn_low_s=generator_nn_low_s, generator_nn_high_s=generator_nn_high_s, dis_nn_low_s=dis_nn_low_s, dis_nn_high_s=dis_nn_high_s,
                                 outlier_nn=outlier_nn,
                                 domain_nn=domain_nn)
existing_checkpoint.restore(tf.train.latest_checkpoint(existing_checkpoint_dir))

# Metrics to gather
test_acc = tf.metrics.Mean(name='test_accuracy')
test_openoa = tf.metrics.Mean(name='test_openoa')
test_closedoa_cm = tf.metrics.Mean(name='test_closedoa_cm')
test_outlier_acc = tf.metrics.Mean(name='test_outlier_acc')
test_domain_acc = tf.metrics.Mean(name='test_domain_acc')

def test_step(tsupport_patches,tquery_patches,support_labels,query_labels,support_classes,CS, CQ, Ss, Sd, Qsk, Qdk, Qsu, Qdu):
    accuracy, closed_oa_cm, outlier_det_acc, openoa, domain_acc  = proto_test(tsupport_patches,tquery_patches,support_labels,query_labels,support_classes,CS, CQ, Ss, Sd, Qsk, Qdk, Qsu, Qdu)
    test_acc(accuracy)
    test_closedoa_cm(closed_oa_cm)
    test_openoa(openoa)
    test_outlier_acc(outlier_det_acc)
    test_domain_acc(domain_acc)

for epoch in range(config["test_epochs"]): # 80 train + 80 tune + 100 train + 160 tune + 40 train
    test_acc.reset_states()
    test_closedoa_cm.reset_states()
    test_openoa.reset_states()
    test_outlier_acc.reset_states()
    test_domain_acc.reset_states()
    for epi in range(10):
        tquery_patches, tsupport_patches, query_labels, support_labels, support_classes = test_episode(source_domain[-test_classes:,:,:,:,:], target_domain[-test_classes:,:,:,:,:], CS, CQ, Ss, Sd, Qsk, Qdk, Qsu, Qdu, class_labels[-test_classes:])
        test_step(tsupport_patches,tquery_patches,support_labels,query_labels,support_classes,CS, CQ, Ss, Sd, Qsk, Qdk, Qsu, Qdu)
        print('query_labels:', query_labels)
    with test_summary_writer.as_default():
        #tf.summary.scalar('accuracy', test_acc.result(), step=epoch)
        tf.summary.scalar('closedoa_cm', test_closedoa_cm.result(), step=epoch)
        tf.summary.scalar('openoa',test_openoa.result(), step=epoch)
        tf.summary.scalar('outlier_acc',test_outlier_acc.result(), step=epoch)
        tf.summary.scalar('domain_acc',test_domain_acc.result(), step=epoch)

    template = 'Epoch {}, Test Closed OA CM: {:.2f}, Test Open OA: {:.2f}, Test Outlier Det. Acc: {:.2f}, Test Domain Det. Acc: {:.2f}'
    print(template.format(epoch+1,test_closedoa_cm.result()*100, test_openoa.result()*100,test_outlier_acc.result()*100,test_domain_acc.result()*100))
