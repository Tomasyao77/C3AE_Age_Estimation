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


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
          tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
          List of pairs of (gradient, variable) where the gradient has been averaged
        across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def train(finetune):
    is_training = True
    print("开始训练时间：")
    start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print(start_time)

    # data pipeline
    imgs, age_labels, age_vectors = gen_data_batch(cfg.train.tf_records, cfg.batch_size * cfg.train.num_gpus)
    # 验证数据，不能这样写
    # imgs_val, age_labels_val, age_vectors_val = gen_data_batch(cfg.val.tf_records, cfg.batch_size * cfg.train.num_gpus)
    # imgs_val = tf.reshape(imgs_val, (-1, imgs_val.get_shape()[2], imgs_val.get_shape()[3], imgs_val.get_shape()[4]))
    # # 占位符
    # input_x = tf.placeHolder()

    print("!!!imgs.get_shape()!!!")
    print(imgs.get_shape())  # (?, 3, 64, 64, 3)
    imgs = tf.reshape(imgs, (-1, imgs.get_shape()[2], imgs.get_shape()[3], imgs.get_shape()[4]))
    print(imgs.get_shape())  # (?, 64, 64, 3)
    imgs_split = tf.split(imgs, cfg.train.num_gpus)
    age_labels_split = tf.split(age_labels, cfg.train.num_gpus)
    age_vectors_split = tf.split(age_vectors, cfg.train.num_gpus)

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0.), trainable=False)
    # lr = tf.train.piecewise_constant(global_step, cfg.train.lr_steps, cfg.train.learning_rate)
    # optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    learn_rate_decay_step = int(
        cfg.train.num_samples / cfg.batch_size / cfg.train.num_gpus * cfg.train.learn_rate_decay_epoch)
    # 36469/128/1*2 = 570 step(每两个epoch)
    # 学习率指数衰减
    learning_rate = tf.train.exponential_decay(cfg.train.learn_rate, global_step, learn_rate_decay_step,
                                               cfg.train.learn_rate_decay, staircase=True)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # Calculate the gradients for each model tower.
    # 计算每个显卡的梯度，这里只有一个显卡
    tower_grads = []
    with tf.variable_scope(tf.get_variable_scope()):
        for i in range(cfg.train.num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % (cfg.train.tower, i)) as scope:
                    model = C3AENet(imgs_split[i], age_labels_split[i], age_vectors_split[i], is_training)
                    loss = model.compute_loss()
                    tf.get_variable_scope().reuse_variables()
                    grads = optimizer.compute_gradients(loss)
                    tower_grads.append(grads)
                    if i == 0:
                        current_loss = loss
                        update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                        # print(tf.GraphKeys.UPDATE_OPS)

    grads = average_gradients(tower_grads)  # 多显卡计算梯度汇总
    with tf.control_dependencies(update_op):
        apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)
        train_op = tf.group(apply_gradient_op, *update_op)

    # GPU config
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Create a saver
    saver = tf.train.Saver(max_to_keep=1000)
    ckpt_dir = cfg.ckpt_path

    # init
    sess.run(tf.global_variables_initializer())
    # 微调，使用预训练模型
    if finetune:
        checkpoint = '../ckpt/pre-imdb-wiki-ckpt/'

        # variables_to_restore = slim.get_variables_to_restore()
        # init_assign_op, init_feed_dict = slim.assign_from_checkpoint(checkpoint, variables_to_restore, ignore_missing_vars=True)
        # sess.run(init_assign_op, init_feed_dict)

        # variables_to_restore = get_variables_to_restore(exclude_global_pool=True)
        # init_assign_op, init_feed_dict = slim.assign_from_checkpoint(checkpoint, variables_to_restore,
        #                                                              ignore_missing_vars=True)
        # sess.run(init_assign_op, init_feed_dict)

        #恢复模型参数
        # 加载检查点状态，这里会获取最新训练好的模型
        ckpt = tf.train.get_checkpoint_state(checkpoint)
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt.model_checkpoint_path)
            # 加载模型和训练好的参数
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("加载模型成功：" + ckpt.model_checkpoint_path)

    # running
    cnt_epoch = 0
    min_train_loss = 1000
    mean_train_loss = []
    # 160 epochs
    for i in range(1, int(cfg.train.num_samples / cfg.train.num_gpus / cfg.batch_size) * cfg.epochs + 1):
        _, loss_, lr_ = sess.run([train_op, current_loss, learning_rate])
        if i % 20 == 0:
            print('No.', i, ' batch, loss:', loss_, ' lr:', lr_)
        if int(i) % int(cfg.train.num_samples / cfg.train.num_gpus / cfg.batch_size) == 0:
            cnt_epoch += 1
            mean_train_loss.append(loss_)
            if min_train_loss > loss_:
                min_train_loss = loss_
            print('No.', cnt_epoch, ' epoch, loss:', loss_, ' lr:', lr_)
            # 计算验证集的误差
            # print('No.', cnt_epoch, ' epoch, loss_val:', sess.run(loss_val))
            saver.save(sess, ckpt_dir + 'C3AEDet', global_step=cnt_epoch, write_meta_graph=True)

    print("结束训练时间：")
    end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print(end_time)
    # 发邮件
    smtp.main(dict_={"共训练epochs: ": cnt_epoch,
                     "训练耗时: ": smtp.date_gap(start_time, end_time),
                     "最低train_loss: ": min_train_loss,
                     "平均train_loss: ": sess.run(tf.reduce_mean(mean_train_loss)),
                     "数据集: ": cfg.train.tf_records})


if __name__ == '__main__':
    train(finetune=False)
