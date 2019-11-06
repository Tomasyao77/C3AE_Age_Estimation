#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
import sys
sys.path.append('..')
from config import cfg

def l1_loss(preds, labels):#preds:(,160)? labels:(,10)
    preds = tf.reshape(preds, (-1,))
    labels = tf.reshape(labels, (-1,))
    l1_loss = tf.abs(preds - labels)
    # _, k_idx = tf.nn.top_k(l1_loss, tf.cast(tf.reduce_prod(tf.shape(l1_loss)) * cfg.ohem_ratio, tf.int32))
    # _, k_idx = tf.nn.top_k(l1_loss, tf.cast(cfg.batch_size * cfg.ohem_ratio, tf.int32))
    # _, k_idx = tf.nn.top_k(l1_loss, tf.cast(960 * cfg.ohem_ratio, tf.int32))
    # loss = tf.gather(l1_loss, k_idx)
    return tf.reduce_mean(l1_loss)

def kl_loss(preds, labels, l1):
    h_pq = -(tf.reduce_sum(tf.reduce_sum(labels * tf.log(preds + 1e-10), axis=1)))
    h_p = -tf.reduce_sum(labels * tf.log(labels + 1e-10))
    loss = h_pq - h_p + l1
    # kl_loss = tf.reduce_sum(labels * tf.log(preds + 1e-10), axis=1)
    # _, k_idx = tf.nn.top_k(kl_loss, tf.cast(tf.reduce_prod(tf.shape(kl_loss)) * cfg.ohem_ratio, tf.int32))
    # _, k_idx = tf.nn.top_k(kl_loss, tf.cast(cfg.batch_size * cfg.ohem_ratio, tf.int32))
    # _, k_idx = tf.nn.top_k(kl_loss, tf.cast(960 * cfg.ohem_ratio, tf.int32))
    # loss = tf.gather(kl_loss, k_idx)
    # return -tf.reduce_mean(loss) + l1
    return -tf.reduce_mean(loss)
