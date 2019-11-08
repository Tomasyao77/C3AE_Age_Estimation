#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import tensorflow as tf
import sys

sys.path.append('..')
from models.run_net import C3AENet
from prepare_data.gen_data_batch import gen_data_batch
from config import cfg
import os
import re
import tensorflow.contrib.slim as slim
import util.smtp as smtp
import time

gpu_list = np.arange(cfg.train.num_gpus)
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(i) for i in gpu_list)


def get_variables_to_restore(include_vars=[], exclude_global_pool=False):
    variables_to_restore = []
    for var in slim.get_model_variables():
        if exclude_global_pool and 'global_pool' in var.op.name:
            # print(var)
            continue
        variables_to_restore.append(var)
    for var in slim.get_variables_to_restore(include=include_vars):
        if exclude_global_pool and 'global_pool' in var.op.name:
            # print(var)
            continue
        variables_to_restore.append(var)
    return variables_to_restore


def val():
    is_training = False  # 验证val

    # data pipeline
    imgs, age_labels, age_vectors = gen_data_batch(cfg.val.tf_records, cfg.batch_size * cfg.train.num_gpus)
    # imgs, age_labels, age_vectors = gen_data_batch(cfg.train_tfrecords_path, cfg.batch_size * cfg.train.num_gpus)
    imgs = tf.reshape(imgs, (-1, imgs.get_shape()[2], imgs.get_shape()[3], imgs.get_shape()[4]))

    model = C3AENet(imgs, age_labels, age_vectors, is_training)
    loss = model.compute_ae()

    # GPU config
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Create a saver
    saver = tf.train.Saver()
    ckpt_dir = cfg.ckpt_path

    # 加载检查点状态，这里会获取最新训练好的模型
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print(ckpt.model_checkpoint_path)
        # 加载模型和训练好的参数
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("加载模型成功：" + ckpt.model_checkpoint_path)

    # saver.restore(sess, ckpt_dir + "C3AEDet-" + str(119)) # 无效
    # saver = tf.train.import_meta_graph('../ckpt/C3AEDet-62.meta')  # 加载模型结构
    # saver.restore(sess, tf.train.latest_checkpoint('./ckpt'))  # 只需要指定目录就可以恢复所有变量信息

    # init
    # 不能加这个!这个是训练时初始化权重等变量的，验证时直接恢复模型和参数了!
    #sess.run(tf.global_variables_initializer())

    # running
    min_val_loss = 1000
    mean_loss = []
    # print(int(cfg.val.num_samples / cfg.batch_size) + 1)#123 = 15629/128
    for i in range(1, int(cfg.val.num_samples / cfg.batch_size) + 1):
        loss_ = sess.run(loss)
        mean_loss.append(loss_)
        if min_val_loss > loss_:
            min_val_loss = loss_
        print('No.', i, ' batch, loss_val:', loss_)

    mean_loss = tf.reduce_mean(mean_loss)
    print("min_val_loss:" + str(min_val_loss))
    print("mean_val_loss:" + str(sess.run(mean_loss)))

if __name__ == '__main__':
    val()
