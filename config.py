from easydict import EasyDict as edict
import numpy as np
import tensorflow as tf
import os

cfg = edict()

cfg.batch_size = 50
# cfg.ckpt_path = '../ckpt/pre-imdb-wiki-ckpt/'
cfg.ckpt_path = '../ckpt/ckpt-morph2/'

# training options
cfg.train = edict()
cfg.train.txt = "/media/gisdom/data11/tomasyao/workspace/pycharm_ws/mypython/dataset/morph2_split/test_morph2_align.txt"
cfg.train.tf_records = '../tf_records/morph2_align/train.records'
# cfg.train.tf_records = '../data/dataset/imdb_wiki_crop/imdb_wiki_crop_train.records'

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
cfg.train.num_samples = 41613 #36469 #363856
cfg.epochs = 160
cfg.PRINT_LAYER_LOG = True
cfg.ohem_ratio = 1.0
cfg.use_se_module = False

# validate options
cfg.val = edict()
cfg.val.num_samples = 10420
cfg.val.tf_records = '../tf_records/morph2_align/val.records'

# test options
cfg.test = edict()
cfg.val.num_samples = 10404 #5210
cfg.test.txt = "/media/gisdom/data11/tomasyao/workspace/pycharm_ws/mypython/dataset/morph2_split/train_morph2_align.txt"
cfg.test.tf_records = '../tf_records/morph2_align/test.records'