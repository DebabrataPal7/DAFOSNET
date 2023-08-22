#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from loss.cosine_loss import cosine_loss
from loss.AOCMC import anti_open_close_mode_collapse_loss

#==========================Functions to optimize GAN==========================
#Function to optimize low noise GAN (GAN 1)
def gan_reptile(sembed,ep_class_labels,ep_domain_labels,generator,dis,optim_d,optim_g,stdev,alpha1):
    sembed = tf.cast(sembed,dtype=tf.float64)
    ep_class_dom_labels = []
    uniques_len = len(set(ep_class_labels))
    for i in range(len(ep_domain_labels)):
        if ep_domain_labels[i] == 0:
            #class [0,1,2]
            ep_class_dom_labels.append(ep_class_labels[i])
        else:
            #other domain same class as pseudo classes [3,4,5...]
            ep_class_dom_labels.append(ep_class_labels[i]+uniques_len)
    batch_size=sembed.shape[0]
    one_hot_labels=tf.one_hot(ep_class_dom_labels, depth=10, axis=-1)
    one_hot_labels=tf.cast(tf.reshape(one_hot_labels,(batch_size,10)),dtype=tf.float64)

    one_hot_labels=tf.one_hot(ep_class_labels, depth=10, axis=-1)#
    one_hot_labels=tf.cast(tf.reshape(one_hot_labels,(batch_size,10)),dtype=tf.float64)

    vector=tf.cast(tf.random.normal(shape=(batch_size,8), stddev=stdev),dtype=tf.float64)
    latent=tf.concat([vector,one_hot_labels], axis=1)

    # loss_function
    loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # generators_low
    generated_emb=tf.cast(generator(latent),dtype=tf.float64)

    fake_embs=tf.concat([generated_emb, one_hot_labels], axis=1)  #[,74]
    real_embs=tf.concat([sembed, one_hot_labels], axis=1)
    combined_embs=tf.concat([fake_embs, real_embs], axis=0)
    labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)

    # First-order meta-Learning - Reptile
    # Discriminator update
    old_vars_gen=generator.get_weights()
    old_vars_dis=dis.get_weights()
    with tf.GradientTape() as dis_tape:
        predictions=dis(combined_embs)
        d_loss=loss_fn(labels,predictions)
    grads=dis_tape.gradient(d_loss,dis.trainable_variables)
    optim_d.apply_gradients(zip(grads, dis.trainable_variables))
    new_vars_dis=dis.get_weights()

    for var in range(len(new_vars_dis)):
        new_vars_dis[var]=tf.cast(old_vars_dis[var],dtype=tf.float64) + tf.cast(((new_vars_dis[var]-old_vars_dis[var])*alpha1),dtype=tf.float64)
    dis.set_weights(new_vars_dis)

    # Generator
    mis_labels=tf.zeros((batch_size,1))
    old_gen=generator.get_weights()
    with tf.GradientTape() as gen_tape:
        vector=tf.cast(tf.random.normal(shape=(batch_size,8), stddev=0.2),dtype=tf.float64)
        latent=tf.concat([vector,one_hot_labels], axis=1)
        fake_emb=tf.cast(generator(latent),dtype=tf.float64)
        fake_emb_and_labels=tf.concat([fake_emb, one_hot_labels], axis=-1)
        predictions= tf.cast(dis(fake_emb_and_labels),dtype=tf.float64)
        g_loss=loss_fn(mis_labels,predictions)
    g_grads=gen_tape.gradient(g_loss, generator.trainable_variables)
    optim_g.apply_gradients(zip(g_grads, generator.trainable_variables))
    new_gen=generator.get_weights()

    for var in range(len(new_gen)):
        new_gen[var]= tf.cast(old_gen[var],dtype=tf.float64) +  tf.cast(((new_gen[var]-old_gen[var])* alpha1),dtype=tf.float64)
    generator.set_weights(new_gen)

    vector=tf.cast(tf.random.normal(shape=(batch_size,8), stddev=stdev),dtype=tf.float64)
    latent=tf.concat([vector,one_hot_labels], axis=1)
    sembed_gen=tf.cast(generator(latent),dtype=tf.float64)
    return sembed_gen,vector

# Function to optimize high noise GAN (GAN 2)
def gan_reptile_high(sembed,sembed_low,z1,ep_class_labels,ep_domain_labels,generator,dis,optim_d,optim_g,stdev,alpha1):
    sembed = tf.cast(sembed,dtype=tf.float64)
    ep_class_dom_labels = []
    uniques_len = len(set(ep_class_labels))
    for i in range(len(ep_domain_labels)):
        if ep_domain_labels[i] == 0:
            #class [0,1,2]
            ep_class_dom_labels.append(ep_class_labels[i])
        else:
            #other domain same class as pseudo classes [3,4,5...]
            ep_class_dom_labels.append(ep_class_labels[i]+uniques_len)
    batch_size=sembed.shape[0]
    one_hot_labels=tf.one_hot(ep_class_dom_labels, depth=10, axis=-1)
    one_hot_labels=tf.cast(tf.reshape(one_hot_labels,(batch_size,10)),dtype=tf.float64)

    vector=tf.cast(tf.random.normal(shape=(batch_size,8), stddev=stdev),dtype=tf.float64)
    latent=tf.concat([vector,one_hot_labels], axis=1)

    # loss_function
    loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # generators_High
    generated_emb=tf.cast(generator(latent),dtype=tf.float64)
    # AOL regulaizer loss
    c_lossd=cosine_loss(generated_emb,sembed_low,vector,z1)
    # AOCMC regulaizer loss
    # l_AOCMC = anti_open_close_mode_collapse_loss(sembed_low,generated_emb,z1,vector)

    fake_embs=tf.concat([generated_emb, one_hot_labels], axis=1)#[,74]
    real_embs=tf.concat([sembed, one_hot_labels], axis=1)
    combined_embs=tf.concat([fake_embs, real_embs], axis=0)
    labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)

    # First-order meta-Learning - Reptile
    # discriminator
    old_vars_gen=generator.get_weights()
    old_vars_dis=dis.get_weights()
    with tf.GradientTape() as dis_tape:
        predictions=tf.cast(dis(combined_embs),dtype=tf.float64)
        d_loss=loss_fn(labels,predictions) + c_lossd
    grads=dis_tape.gradient(d_loss,dis.trainable_variables)
    optim_d.apply_gradients(zip(grads, dis.trainable_variables))
    new_vars_dis=dis.get_weights()

    for var in range(len(new_vars_dis)):
        new_vars_dis[var]=old_vars_dis[var] + ((new_vars_dis[var]-old_vars_dis[var])*alpha1)
    dis.set_weights(new_vars_dis)

    # generator
    mis_labels=tf.zeros((batch_size,1))
    old_gen=generator.get_weights()
    with tf.GradientTape() as gen_tape:
        vector=tf.cast(tf.random.normal(shape=(batch_size,8), stddev=0.2),dtype=tf.float64)
        latent=tf.concat([vector,one_hot_labels], axis=1)
        fake_emb=tf.cast(generator(latent),dtype=tf.float64)
        c_lossg=cosine_loss(fake_emb,sembed_low,vector,z1)
        fake_emb_and_labels=tf.concat([fake_emb, one_hot_labels], axis=-1)
        predictions=dis(fake_emb_and_labels)
        g_loss=tf.cast(loss_fn(mis_labels,predictions),dtype=tf.float64) + tf.cast(c_lossg,dtype=tf.float64)
    g_grads=gen_tape.gradient(g_loss, generator.trainable_variables)
    optim_g.apply_gradients(zip(g_grads, generator.trainable_variables))
    new_gen=generator.get_weights()
    for var in range(len(new_gen)):
        new_gen[var]= tf.cast(old_gen[var],dtype=tf.float64) + tf.cast(((new_gen[var]-old_gen[var])* alpha1),dtype=tf.float64)
    generator.set_weights(new_gen)

    vector=tf.cast(tf.random.normal(shape=(batch_size,8), stddev=stdev),dtype=tf.float64)
    latent=tf.concat([vector,one_hot_labels], axis=1)
    sembed_gen=tf.cast(generator(latent),dtype=tf.float64)
    return sembed_gen
