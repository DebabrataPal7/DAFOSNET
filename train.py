#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
import tensorflow as tf
import tensorflow.keras.backend as K
import os
import json
import numpy as np
import datetime
from sklearn.metrics import confusion_matrix
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
from utils.proto_train import proto_train
#new train episode
from utils.train_episode import new_episode
#adamatch augmentation
# from utils.adamatch_aug import adamatch_aug

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

# Train Class split
train_classes = config['train_classes']

#embedding dimensions
emb_dim = config['emb_dim']

#bool to check if we need to save the input images to a numpy file
pre_process=config['pre_process']

#make numpy arrays of the image files
if pre_process == "True":
    save_data(base_dir, save_path_source, save_path_target, num_classes, num_images_per_class)
else:
    print("Not saving input images into Numpy files")

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

#====================================================================Optimizers================================================================
lr = config['lr']
initial_learning_rate = config['initial_learning_rate']
decay_rate = config['decay_rate']
decay_steps = config['decay_steps']

# Create the learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    staircase=True)

optim_d_low_s=tf.keras.optimizers.legacy.Adam(lr)
optim_g_low_s=tf.keras.optimizers.legacy.Adam(lr)
optim_d_high_s=tf.keras.optimizers.legacy.Adam(lr)
optim_g_high_s=tf.keras.optimizers.legacy.Adam(lr)
optim_model_Outlier = tf.keras.optimizers.legacy.Adam(initial_learning_rate)
optim_model_Domain = tf.keras.optimizers.legacy.Adam(initial_learning_rate)
optim2 = tf.keras.optimizers.legacy.Adam(lr)##Multitask Outlier network
optim3 = tf.keras.optimizers.legacy.Adam(initial_learning_rate)##FE
#====================================================================Train checkpoint================================================================
checkpoint_dir = config["existing_checkpoint_dir"]
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

# Define a variable to keep track of the current epoch
epoch_var = tf.Variable(0, dtype=tf.int64, name="epoch")

checkpoint = tf.train.Checkpoint(epoch_var,
                                 optim3=optim3, optim2=optim2,
                                 optim_d_low_s=optim_d_low_s,optim_g_low_s=optim_g_low_s, optim_d_high_s=optim_d_high_s,optim_g_high_s=optim_g_high_s,
                                 optim_model_Outlier = optim_model_Outlier,
                                 optim_model_Domain = optim_model_Domain,
                                 CDFSOSR_Model = CDFSOSR_Model,
                                 generator_nn_low_s=generator_nn_low_s, generator_nn_high_s=generator_nn_high_s, dis_nn_low_s=dis_nn_low_s, dis_nn_high_s=dis_nn_high_s,
                                 outlier_nn=outlier_nn,
                                 domain_nn=domain_nn)
ckpt_manager = tf.train.CheckpointManager(checkpoint,checkpoint_dir, max_to_keep=5)

#===============================================================Train Step========================================================
# Metrics to gather
train_loss = tf.metrics.Mean(name='train_loss')
train_acc = tf.metrics.Mean(name='train_accuracy')
train_closedoa_cm = tf.metrics.Mean(name='train_closedoa_cm')
train_openoa = tf.metrics.Mean(name='train_openoa')
train_outlier_acc = tf.metrics.Mean(name='train_outlier_acc')
train_domain_acc = tf.metrics.Mean(name='train_domain_acc')
train_closedoa_cm_variance = ComputeIterationVaraince(name="train_closedoa_cm_variance")
train_openoa_variance = ComputeIterationVaraince(name="train_openoa_variance")

ntimes=config['ntimes']

def train_step(tsupport_patches,tquery_patches,support_labels,query_labels,support_classes,CS, CQ, Ss, Sd, Qsk, Qdk, Qsu, Qdu, query_dom_labels, support_dom_labels):
    # Forward & update gradients
    with tf.GradientTape() as tape:
        loss, accuracy, closed_oa_cm, outlier_det_acc, openoa, domain_acc  = proto_train(tsupport_patches,tquery_patches,support_labels,query_labels,support_classes,CS, CQ, Ss, Sd, Qsk, Qdk, Qsu, Qdu, query_dom_labels, support_dom_labels, CDFSOSR_Model, outlier_nn, domain_nn, generator_nn_low_s, dis_nn_low_s,generator_nn_high_s, dis_nn_high_s, ntimes, lr_schedule, optim_model_Outlier, optim_model_Domain, optim_d_low_s,optim_g_low_s, optim_d_high_s,optim_g_high_s)
    gradients = tape.gradient(loss, CDFSOSR_Model.trainable_variables)
    # Compute the learning rate using the schedule
    learning_rate = lr_schedule(optim3.iterations)
    optim3.learning_rate = learning_rate
    optim3.apply_gradients(zip(gradients, CDFSOSR_Model.trainable_variables))
    train_loss(loss)
    train_acc(accuracy)
    train_closedoa_cm(closed_oa_cm)
    train_openoa(openoa)
    train_outlier_acc(outlier_det_acc)
    train_domain_acc(domain_acc)
    train_closedoa_cm_variance.add(closed_oa_cm)
    train_openoa_variance.add(openoa)
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

#===============================================================Train Loop========================================================
start_epoch = 0
if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
    # restoring the latest checkpoint in checkpoint_path
    checkpoint.restore(ckpt_manager.latest_checkpoint)
    print("Restarted from: ", start_epoch)
else:
    print("Training from scratch...")

for epoch in range(start_epoch, config["train_epochs"]): # 80 train + 80 tune + 100 train + 160 tune + 40 train
    train_loss.reset_states()
    train_acc.reset_states()
    train_closedoa_cm.reset_states()
    train_openoa.reset_states()
    train_outlier_acc.reset_states()
    train_domain_acc.reset_states()
    train_closedoa_cm_variance.reset_states()
    train_openoa_variance.reset_states()
    epoch_var.assign_add(1)
    for epi in range(10):
        tquery_patches, tsupport_patches, query_labels, support_labels, support_classes, query_dom_labels, support_dom_labels = new_episode(source_domain[:train_classes,:,:,:,:], target_domain[:train_classes,:,:,:,:], CS, CQ, Ss, Sd, Qsk, Qdk, Qsu, Qdu, class_labels[:train_classes])
        # tsupport_patches_aug, support_labels_aug, Ss_aug, Sd_aug, support_dom_labels_aug = adamatch_aug(tsupport_patches, support_labels, CS, Ss, Sd, support_dom_labels)
        train_step(tsupport_patches,tquery_patches,support_labels,query_labels,support_classes,CS, CQ, Ss, Sd, Qsk, Qdk, Qsu, Qdu, query_dom_labels, support_dom_labels)
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', train_loss.result(), step=epoch)
        tf.summary.scalar('accuracy', train_acc.result(), step=epoch)
        tf.summary.scalar('closedoa_cm', train_closedoa_cm.result(), step=epoch)
        tf.summary.scalar('openoa', train_openoa.result(), step=epoch)
        tf.summary.scalar('outlier_det_acc', train_outlier_acc.result(), step=epoch)
        tf.summary.scalar('domain_det_acc', train_domain_acc.result(), step=epoch)

    template = 'Epoch {}, Train Loss: {:.2f}, Train Accuracy: {:.2f}, Train Closed OA: {:.2f},Train Open OA: {:.2f}, Train Outlier Det. Acc: {:.2f}, Train Domain Det. Acc: {:.2f}'
    print(template.format(epoch+1,train_loss.result(),train_acc.result()*100,train_closedoa_cm.result()*100,train_openoa.result()*100,train_outlier_acc.result()*100,train_domain_acc.result()*100))
    #save mean accuracy values per epoch
    save_accuracy_values(train_closedoa_cm.result(), train_openoa.result(), 'accuracy_values.npy')
    #save variance for accuracys
    save_accuracy_values(train_closedoa_cm_variance.compute_variance(), train_openoa_variance.compute_variance(), 'accuracy_variance_values.npy')

    if epoch % config["save_every"] == 0 and epoch != 0 :
      print("Checkpoint saved at: ", checkpoint_dir)
      checkpoint.save(file_prefix = checkpoint_prefix)