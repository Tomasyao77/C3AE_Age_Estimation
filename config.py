from easydict import EasyDict as edict
import numpy as np
import tensorflow as tf
import os

cfg = edict()

cfg.batch_size = 50
cfg.ckpt_path = '../ckpt/'
# cfg.ckpt_path = '../ckpt_l1kl/'

# training options
cfg.train = edict()
cfg.train.tf_records = '../tf_records/train.records'

cfg.train.ignore_thresh = .5
cfg.train.ratio = 0.8
cfg.train.momentum = 0.9
cfg.train.bn_training = True
cfg.train.weight_decay = 0.00001 # 0.00004
cfg.train.learning_rate = [1e-3, 1e-4, 1e-5]
cfg.train.max_batches = 30000 # 64000
cfg.train.lr_steps = [10000., 20000.]
cfg.train.lr_scales = [.1, .1]
cfg.train.num_gpus = 1
cfg.train.tower = 'tower'

cfg.train.learn_rate = 0.001
cfg.train.learn_rate_decay = 0.9
cfg.train.learn_rate_decay_epoch = 2
cfg.train.num_samples = 36469
cfg.epochs = 160
cfg.PRINT_LAYER_LOG = True
cfg.ohem_ratio = 1.0
cfg.use_se_module = False

# validate options
cfg.val = edict()
cfg.val.num_samples = 10420
cfg.val.tf_records = '../tf_records/val.records'

# test options
cfg.test = edict()
cfg.val.num_samples = 5210
cfg.test.tf_records = '../tf_records/test.records'