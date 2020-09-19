# -*- coding: utf-8 -*-
# 
# # Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
import os
import io
from io import StringIO
import time
import argparse
import functools
import errno
import scipy
import scipy.io
import requests
import zipfile
import random
import datetime
#
from functools import partial
from importlib import import_module
#
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
#
import numpy as np
from numpy import *
#
import math
from math import floor, log2
from random import random
from pylab import *
from IPython.core.display import display
import PIL
from PIL import Image
PIL.Image.MAX_IMAGE_PIXELS = 933120000
#
import scipy.ndimage as pyimg
import cv2
import imageio
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt 
import matplotlib.image as mgimg
import matplotlib.animation as anim
mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False
#
import shutil
import gdown
#
import sys
#
import tensorflow as tf 
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D, UpSampling2D, Conv2D
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.utils import conv_utils
#
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import add
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.models import clone_model
from tensorflow.keras.models import model_from_json
#
from absl import app
from absl import flags
from absl import logging
#
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#
print(f'|===> {tf.__version__}')
#
if 1: # get base.py from github
    cwd = os.getcwd()
    base_path = os.path.join(cwd, 'base.py')
    if not os.path.exists(base_path):

        base_file = 'base.py'
        urlfolder = 'https://raw.githubusercontent.com/sifbuilder/pylon/master/'
        url = f'{urlfolder}{base_file}'

        print(f"|===> nnimg: get base file \n \
            urlfolder: {urlfolder} \n \
            url: {url} \n \
            base_path: {base_path} \n \
        ")

        tofile = tf.keras.utils.get_file(f'{base_path}', origin=url, extract=True)

    else:
        print(f"|===> base in cwd {cwd}")
#
#
#   IMPS
#
#
# check if base.Onpyon is defined
try:
    var = Onpyon()
except NameError:
    sys.path.append('../')  # if called from eon, modules are in parallel folder
    sys.path.append('./')  #  if called from dnns, modules are in folder
    from base import *
#
onutil = Onutil()
onplot = Onplot()
onformat = Onformat()
onfile = Onfile()
onvid = Onvid()
onimg = Onimg()
ondata = Ondata()
onset = Onset()
onrecord = Onrecord()
ontree = Ontree()
#
onvgg = Onvgg()
onlllyas = Onlllyas()
oncuda = Oncuda()
onrosa = Onrosa()
onmoono = Onmoono()
#
ontf = Ontf()
#
#
#   CONTEXT
#
#
#
def getap():
    # https://github.com/moono/stylegan2-tf-2.x/tree/master/stylegan2
    cp = {
        "primecmd": 'nntrain',    
                
        "MNAME": "moono",      
        "AUTHOR": "moono",      
        "PROJECT": "stylegan2",      
        "GITPOD": "stylegan2-tf-2.x",      
        "DATASET": "ffhq",        
    
        "GDRIVE": 1,            # mount gdrive: gdata, gwork    
        "TRAINDO": 1,      
        "MAKEGIF": 1,      
        "RUNGIF": 0,      
        "CLEARTMP": 0,      
        "REGET": 0,             # get again data 
        "ING": 1,               # ckpt in gwork
        "MODITEM": "",          # will look into module
        "RESETCODE": False,
        "LOCALDATA": False,
        "LOCALMODELS": False,
        "LOCALLAB": True,
        "grel_infix": '../..',            # relative path to content 
        "net_prefix": '//enas/hdrive',     
        "gdrive_prefix": '/content/drive/My Drive',     
        "gcloud_prefix": '/content',     

    }

    local_prefix = os.path.abspath('')
    try:
        local_prefix = os.path.dirname(os.path.realpath(__file__)) # script dir
    except:
        pass
    cp["local_prefix"] = local_prefix

    hp = {
        "verbose": False,
        "visual": True,
    }

    ap = {}
    for key in cp.keys():
        ap[key] = cp[key]
    for key in hp.keys():
        ap[key] = hp[key]

    return ap
#
def getxp(cp):

    yp={}
    xp={}
    for key in cp.keys():
        xp[key] = cp[key]

    tree = ontree.tree(cp)
    for key in tree.keys():
        xp[key] = tree[key]

    for key in yp.keys():
        xp[key] = yp[key]
   
    return xp
#
#
#   NETS
#
#
class Dense(tf.keras.layers.Layer):
    def __init__(self, fmaps, gain, lrmul, **kwargs):
        super(Dense, self).__init__(**kwargs)
        self.fmaps = fmaps
        self.gain = gain
        self.lrmul = lrmul

    def build(self, input_shape):
        assert len(input_shape) == 2 or len(input_shape) == 4
        fan_in = np.prod(input_shape[1:])
        weight_shape = [fan_in, self.fmaps]
        init_std, runtime_coef = onmoono.compute_runtime_coef(weight_shape, self.gain, self.lrmul)

        self.runtime_coef = runtime_coef

        w_init = tf.random.normal(shape=weight_shape, mean=0.0, stddev=init_std)
        self.w = tf.Variable(w_init, name='w', trainable=True)

    def call(self, inputs, training=None, mask=None):
        weight = self.runtime_coef * self.w
        x = tf.keras.layers.Flatten()(inputs)
        x = tf.matmul(x, weight)
        return x

    def get_config(self):
        config = super(Dense, self).get_config()
        config.update({
            'fmaps': self.fmaps,
            'gain': self.gain,
            'lrmul': self.lrmul,
        })
        return config
#
class Bias(tf.keras.layers.Layer):
    def __init__(self, lrmul, **kwargs):
        super(Bias, self).__init__(**kwargs)
        self.lrmul = lrmul

    def build(self, input_shape):
        assert len(input_shape) == 2 or len(input_shape) == 4

        self.len2 = True if len(input_shape) == 2 else False

        b_init = tf.zeros(shape=(input_shape[1],), dtype=tf.dtypes.float32)
        self.b = tf.Variable(b_init, name='b', trainable=True)

    def call(self, inputs, training=None, mask=None):
        b = self.lrmul * self.b

        if self.len2:
            x = inputs + b
        else:
            x = inputs + tf.reshape(b, [1, -1, 1, 1])
        return x

    def get_config(self):
        config = super(Bias, self).get_config()
        config.update({
            'lrmul': self.lrmul,
        })
        return config
#
class LeakyReLU(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(LeakyReLU, self).__init__(**kwargs)
        self.alpha = 0.2
        self.gain = np.sqrt(2)

        self.act = tf.keras.layers.LeakyReLU(alpha=self.alpha)

    def call(self, inputs, training=None, mask=None):
        x = self.act(inputs)
        x *= self.gain
        return x

    def get_config(self):
        config = super(LeakyReLU, self).get_config()
        config.update({
            'alpha': self.alpha,
            'gain': self.gain,
        })
        return config
#
class LabelEmbedding(tf.keras.layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super(LabelEmbedding, self).__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        weight_shape = [input_shape[1], self.embed_dim]
        # tf 1.15 mean(0.0), std(1.0) default value of tf.initializers.random_normal()
        w_init = tf.random.normal(shape=weight_shape, mean=0.0, stddev=1.0)
        self.w = tf.Variable(w_init, name='w', trainable=True)

    def call(self, inputs, training=None, mask=None):
        x = tf.matmul(inputs, self.w)
        return x

    def get_config(self):
        config = super(LabelEmbedding, self).get_config()
        config.update({
            'embed_dim': self.embed_dim,
        })
        return config
#
class Noise(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Noise, self).__init__(**kwargs)

    def build(self, input_shape):
        self.noise_strength = tf.Variable(initial_value=0.0, dtype=tf.dtypes.float32, trainable=True, name='w')

    def call(self, x, training=None, mask=None):
        x_shape = tf.shape(x)
        noise = tf.random.normal(shape=(x_shape[0], 1, x_shape[2], x_shape[3]), dtype=tf.dtypes.float32)

        x += noise * self.noise_strength
        return x
#
class MinibatchStd(tf.keras.layers.Layer):
    def __init__(self, group_size, num_new_features, **kwargs):
        super(MinibatchStd, self).__init__(**kwargs)
        self.group_size = group_size
        self.num_new_features = num_new_features

    def call(self, x, training=None, mask=None):
        group_size = tf.minimum(self.group_size, tf.shape(x)[0])
        s = tf.shape(x)

        y = tf.reshape(x, [group_size, -1, self.num_new_features, s[1] // self.num_new_features, s[2], s[3]])
        y = tf.cast(y, tf.float32)
        y -= tf.reduce_mean(y, axis=0, keepdims=True)
        y = tf.reduce_mean(tf.square(y), axis=0)
        y = tf.sqrt(y + 1e-8)
        y = tf.reduce_mean(y, axis=[2, 3, 4], keepdims=True)
        y = tf.reduce_mean(y, axis=[2])
        y = tf.cast(y, x.dtype)
        y = tf.tile(y, [group_size, 1, s[2], s[3]])
        return tf.concat([x, y], axis=1)
#
class FusedModConv(tf.keras.layers.Layer):
    def __init__(self, fmaps, kernel, gain, lrmul, style_fmaps, demodulate, up, down, resample_kernel, **kwargs):
        super(FusedModConv, self).__init__(**kwargs)
        assert not (up and down)
        self.fmaps = fmaps
        self.kernel = kernel
        self.gain = gain
        self.lrmul = lrmul
        self.style_fmaps = style_fmaps
        self.demodulate = demodulate
        self.up = up
        self.down = down
        self.factor = 2
        if resample_kernel is None:
            resample_kernel = [1] * self.factor
        self.k = onmoono.setup_resample_kernel(k=resample_kernel)

        self.mod_dense = Dense(self.style_fmaps, gain=1.0, lrmul=1.0, name='mod_dense')
        self.mod_bias = Bias(lrmul=1.0, name='mod_bias')

    def build(self, input_shape):
        x_shape, w_shape = input_shape[0], input_shape[1]
        weight_shape = [self.kernel, self.kernel, x_shape[1], self.fmaps]
        init_std, runtime_coef = onmoono.compute_runtime_coef(weight_shape, self.gain, self.lrmul)

        self.runtime_coef = runtime_coef

        # [kkIO]
        w_init = tf.random.normal(shape=weight_shape, mean=0.0, stddev=init_std)
        self.w = tf.Variable(w_init, name='w', trainable=True)

    def scale_conv_weights(self, w):
        # convolution kernel weights for fused conv
        weight = self.runtime_coef * self.w     # [kkIO]
        weight = weight[np.newaxis]             # [BkkIO]

        # modulation
        style = self.mod_dense(w)                                   # [BI]
        style = self.mod_bias(style) + 1.0                          # [BI]
        weight *= style[:, np.newaxis, np.newaxis, :, np.newaxis]   # [BkkIO]

        # demodulation
        if self.demodulate:
            d = tf.math.rsqrt(tf.reduce_sum(tf.square(weight), axis=[1, 2, 3]) + 1e-8)  # [BO]
            weight *= d[:, np.newaxis, np.newaxis, np.newaxis, :]                       # [BkkIO]

        # weight: reshape, prepare for fused operation
        new_weight_shape = [tf.shape(weight)[1], tf.shape(weight)[2], tf.shape(weight)[3], -1]      # [kkI(BO)]
        weight = tf.transpose(weight, [1, 2, 3, 0, 4])                                              # [kkIBO]
        weight = tf.reshape(weight, shape=new_weight_shape)                                         # [kkI(BO)]
        return weight

    def call(self, inputs, training=None, mask=None):
        x, w = inputs
        height, width = tf.shape(x)[2], tf.shape(x)[3]

        # prepare convolution kernel weights
        weight = self.scale_conv_weights(w)

        # prepare inputs: reshape minibatch to convolution groups
        x = tf.reshape(x, [1, -1, height, width])

        if self.up:
            x = onmoono.upsample_conv_2d(x, self.k, weight, self.factor, self.gain)
        elif self.down:
            x = onmoono.conv_downsample_2d(x, self.k, weight, self.factor, self.gain)
        else:
            x = tf.nn.conv2d(x, weight, data_format='NCHW', strides=[1, 1, 1, 1], padding='SAME')

        # x: reshape back
        x = tf.reshape(x, [-1, self.fmaps, tf.shape(x)[2], tf.shape(x)[3]])
        return x

    def get_config(self):
        config = super(FusedModConv, self).get_config()
        config.update({
            'fmaps': self.fmaps,
            'kernel': self.kernel,
            'gain': self.gain,
            'lrmul': self.lrmul,
            'style_fmaps': self.style_fmaps,
            'demodulate': self.demodulate,
            'up': self.up,
            'down': self.down,
            'factor': self.factor,
            'k': self.k,
        })
        return config
#
class ResizeConv2D(tf.keras.layers.Layer):
    def __init__(self, fmaps, kernel, gain, lrmul, up, down, resample_kernel, **kwargs):
        super(ResizeConv2D, self).__init__(**kwargs)
        self.fmaps = fmaps
        self.kernel = kernel
        self.gain = gain
        self.lrmul = lrmul
        self.up = up
        self.down = down
        self.factor = 2
        if resample_kernel is None:
            resample_kernel = [1] * self.factor
        self.k = onmoono.setup_resample_kernel(k=resample_kernel)

    def build(self, input_shape):
        assert len(input_shape) == 4
        weight_shape = [self.kernel, self.kernel, input_shape[1], self.fmaps]
        init_std, runtime_coef = onmoono.compute_runtime_coef(weight_shape, self.gain, self.lrmul)

        self.runtime_coef = runtime_coef

        # [kernel, kernel, fmaps_in, fmaps_out]
        w_init = tf.random_normal_initializer(mean=0.0, stddev=init_std)
        self.w = tf.Variable(initial_value=w_init(shape=weight_shape, dtype='float32'), trainable=True, name='w')

    def call(self, inputs, training=None, mask=None):
        x = inputs
        weight = self.runtime_coef * self.w

        if self.up:
            x = onmoono.upsample_conv_2d(x, self.k, weight, self.factor, self.gain)
        elif self.down:
            x = onmoono.conv_downsample_2d(x, self.k, weight, self.factor, self.gain)
        else:
            x = tf.nn.conv2d(x, weight, data_format='NCHW', strides=[1, 1, 1, 1], padding='SAME')
        return x

    def get_config(self):
        config = super(ResizeConv2D, self).get_config()
        config.update({
            'fmaps': self.fmaps,
            'kernel': self.kernel,
            'gain': self.gain,
            'lrmul': self.lrmul,
            'up': self.up,
            'down': self.down,
            'factor': self.factor,
            'k': self.k,
        })
        return config
#
#
#   losses.py
#    -> https://github.com/moono/stylegan2-tf-2.x/blob/master/losses.py <-
#
#
#
def g_logistic_non_saturating(generator, discriminator, z, labels):
    # forward pass
    fake_images = generator([z, labels], training=True)
    fake_scores = discriminator([fake_images, labels], training=True)

    # gan loss
    g_loss = tf.math.softplus(-fake_scores)
    # g_loss = tf.reduce_mean(g_loss)
    return g_loss
#
def pl_reg(fake_images, w_broadcasted, pl_mean, pl_decay=0.01):
    print(f'|--->  _e_ pl_reg')
    h, w = fake_images.shape[2], fake_images.shape[3]

    # Compute |J*y|.
    with tf.GradientTape() as pl_tape:
        pl_tape.watch(w_broadcasted)
        pl_noise = tf.random.normal(tf.shape(fake_images), mean=0.0, stddev=1.0, dtype=tf.float32) / np.sqrt(h * w)

    pl_grads = pl_tape.gradient(tf.reduce_sum(fake_images * pl_noise), w_broadcasted)
    print(f'|--->  _e_ pl_reg pl_grads   {pl_grads}')
    print(f'|--->   pl_reg pl_mean   {pl_mean}')
    pl_lengths = tf.math.sqrt(tf.reduce_mean(tf.reduce_sum(tf.math.square(pl_grads), axis=2), axis=1))

    # Track exponential moving average of |J*y|.
    pl_mean_val = pl_mean + pl_decay * (tf.reduce_mean(pl_lengths) - pl_mean)
    pl_mean.assign(pl_mean_val)

    # Calculate (|J*y|-a)^2.
    pl_penalty = tf.square(pl_lengths - pl_mean)
    return pl_penalty
#
def d_logistic(generator, discriminator, z, labels, real_images):
    # forward pass
    fake_images = generator([z, labels], training=True)
    real_scores = discriminator([real_images, labels], training=True)
    fake_scores = discriminator([fake_images, labels], training=True)

    # gan loss
    d_loss = tf.math.softplus(fake_scores)
    d_loss += tf.math.softplus(-real_scores)
    return d_loss
#
def r1_reg(discriminator, labels, real_images):
    # simple GP
    with tf.GradientTape() as r1_tape:
        r1_tape.watch(real_images)
        real_loss = tf.reduce_sum(discriminator([real_images, labels], training=True))

    real_grads = r1_tape.gradient(real_loss, real_images)
    r1_penalty = tf.reduce_sum(tf.math.square(real_grads), axis=[1, 2, 3])
    r1_penalty = tf.expand_dims(r1_penalty, axis=1)
    return r1_penalty
#
#
#
#   losses.py (strategy)
#
def d_logistic(real_images, generator, discriminator, z_dim, labels_dim):
    print(f'|---> d_logistic')
    batch_size = tf.shape(real_images)[0]
    z = tf.random.normal(shape=[batch_size, z_dim], dtype=tf.float32)
    labels = tf.random.normal(shape=[batch_size, labels_dim], dtype=tf.float32)

    # forward pass
    fake_images = generator([z, labels], training=True)
    real_scores = discriminator([real_images, labels], training=True)
    fake_scores = discriminator([fake_images, labels], training=True)

    # gan loss
    d_loss = tf.math.softplus(fake_scores)
    d_loss += tf.math.softplus(-real_scores)
    return d_loss
#
#
def d_logistic_r1_reg(real_images, generator, discriminator, z_dim, labels_dim):
    print(f'|---> d_logistic_r1_reg')
    batch_size = tf.shape(real_images)[0]
    z = tf.random.normal(shape=[batch_size, z_dim], dtype=tf.float32)
    labels = tf.random.normal(shape=[batch_size, labels_dim], dtype=tf.float32)

    # forward pass
    fake_images = generator([z, labels], training=True)
    real_scores = discriminator([real_images, labels], training=True)
    fake_scores = discriminator([fake_images, labels], training=True)

    # gan loss
    d_loss = tf.math.softplus(fake_scores)
    d_loss += tf.math.softplus(-real_scores)

    # gradient penalty
    with tf.GradientTape() as r1_tape:
        r1_tape.watch([real_images, labels])
        real_loss = tf.reduce_sum(discriminator([real_images, labels], training=True))

    real_grads = r1_tape.gradient(real_loss, real_images)
    r1_penalty = tf.reduce_sum(tf.math.square(real_grads), axis=[1, 2, 3])
    r1_penalty = tf.expand_dims(r1_penalty, axis=1)
    return d_loss, r1_penalty
#
#
def g_logistic_non_saturating(real_images, generator, discriminator, z_dim, labels_dim):
    print(f'|---> g_logistic_non_saturating')
    batch_size = tf.shape(real_images)[0]
    z = tf.random.normal(shape=[batch_size, z_dim], dtype=tf.float32)
    labels = tf.random.normal(shape=[batch_size, labels_dim], dtype=tf.float32)

    # forward pass
    fake_images = generator([z, labels], training=True)
    fake_scores = discriminator([fake_images, labels], training=True)

    # gan loss
    g_loss = tf.math.softplus(-fake_scores)
    return g_loss
#
#
def g_logistic_ns_pathreg(real_images, generator, discriminator, z_dim, labels_dim,
                          pl_mean, pl_minibatch_shrink, pl_denorm, pl_decay):
    print(f'|---> g_logistic_ns_pathreg')
    batch_size = tf.shape(real_images)[0]
    print(f'|---> g_logistic_ns_pathreg batch_size {batch_size}')
    print(f'|---> g_logistic_ns_pathreg z_dim {z_dim}')
    print(f'|---> g_logistic_ns_pathreg labels_dim {labels_dim}')
    z = tf.random.normal(shape=[batch_size, z_dim], dtype=tf.float32)
    labels = tf.random.normal(shape=[batch_size, labels_dim], dtype=tf.float32)

    pl_minibatch = tf.maximum(1, tf.math.floordiv(batch_size, pl_minibatch_shrink))
    pl_z = tf.random.normal(shape=[pl_minibatch, z_dim], dtype=tf.float32)
    pl_labels = tf.random.normal(shape=[pl_minibatch, labels_dim], dtype=tf.float32)

    # forward pass
    fake_images, w_broadcasted = generator([z, labels], ret_w_broadcasted=True, training=True)
    fake_scores = discriminator([fake_images, labels], training=True)
    g_loss = tf.math.softplus(-fake_scores)

    # Evaluate the regularization term using a smaller minibatch to conserve memory.
    with tf.GradientTape() as pl_tape:
        pl_tape.watch([pl_z, pl_labels])
        pl_fake_images, pl_w_broadcasted = generator([pl_z, pl_labels], ret_w_broadcasted=True, training=True)

        pl_noise = tf.random.normal(tf.shape(pl_fake_images)) * pl_denorm
        pl_noise_applied = tf.reduce_sum(pl_fake_images * pl_noise)

    pl_grads = pl_tape.gradient(pl_noise_applied, pl_w_broadcasted)
    #pl_lengths = tf.math.sqrt(tf.reduce_mean(tf.reduce_sum(tf.math.square(pl_grads), axis=2), axis=1))

    # ***************************************************************************
    pl_lengths = 0.0                                # _e_ ***********************
    if pl_grads:
        pl_lengths = tf.math.sqrt(tf.reduce_mean(tf.reduce_sum(tf.math.square(pl_grads), axis=2), axis=1))
    # ***************************************************************************
    print(f"|...>  pl_lengths: {pl_lengths}")


    # Track exponential moving average of |J*y|.
    pl_mean_val = pl_mean + pl_decay * (tf.reduce_mean(pl_lengths) - pl_mean)
    pl_mean.assign(pl_mean_val)

    # Calculate (|J*y|-a)^2.
    pl_penalty = tf.square(pl_lengths - pl_mean)
    return g_loss, pl_penalty
#
#
#
#
#    -> https://github.com/moono/stylegan2-tf-2.x/tree/master/stylegan2/generator.py <-
#
#
class ToRGB(tf.keras.layers.Layer):
    def __init__(self, in_ch, **kwargs):
        super(ToRGB, self).__init__(**kwargs)
        self.in_ch = in_ch
        self.conv = FusedModConv(fmaps=3, kernel=1, gain=1.0, lrmul=1.0, style_fmaps=self.in_ch,
                                 demodulate=False, up=False, down=False, resample_kernel=None, name='conv')
        self.apply_bias = Bias(lrmul=1.0, name='bias')

    def call(self, inputs, training=None, mask=None):
        x, w = inputs
        assert x.shape[1] == self.in_ch

        x = self.conv([x, w])
        x = self.apply_bias(x)
        return x

    def get_config(self):
        config = super(ToRGB, self).get_config()
        config.update({
            'in_ch': self.in_ch,
        })
        return config
#
class Mapping(tf.keras.layers.Layer):
    def __init__(self, w_dim, labels_dim, n_mapping, **kwargs):
        super(Mapping, self).__init__(**kwargs)
        self.w_dim = w_dim
        self.labels_dim = labels_dim
        self.n_mapping = n_mapping
        self.gain = 1.0
        self.lrmul = 0.01

        if self.labels_dim > 0:
            self.labels_embedding = LabelEmbedding(embed_dim=self.w_dim, name='labels_embedding')

        self.normalize = tf.keras.layers.Lambda(lambda x: x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + 1e-8))

        self.dense_layers = list()
        self.bias_layers = list()
        self.act_layers = list()
        for ii in range(self.n_mapping):
            self.dense_layers.append(Dense(w_dim, gain=self.gain, lrmul=self.lrmul, name='dense_{:d}'.format(ii)))
            self.bias_layers.append(Bias(lrmul=self.lrmul, name='bias_{:d}'.format(ii)))
            self.act_layers.append(LeakyReLU(name='lrelu_{:d}'.format(ii)))

    def call(self, inputs, training=None, mask=None):
        latents, labels = inputs
        x = latents

        # embed label if any
        if self.labels_dim > 0:
            y = self.labels_embedding(labels)
            x = tf.concat([x, y], axis=1)

        # normalize inputs
        x = self.normalize(x)

        # apply mapping blocks
        for dense, apply_bias, leaky_relu in zip(self.dense_layers, self.bias_layers, self.act_layers):
            x = dense(x)
            x = apply_bias(x)
            x = leaky_relu(x)

        return x

    def get_config(self):
        config = super(Mapping, self).get_config()
        config.update({
            'w_dim': self.w_dim,
            'labels_dim': self.labels_dim,
            'n_mapping': self.n_mapping,
            'gain': self.gain,
            'lrmul': self.lrmul,
        })
        return config
#
class SynthesisConstBlock(tf.keras.layers.Layer):
    def __init__(self, fmaps, res, **kwargs):
        super(SynthesisConstBlock, self).__init__(**kwargs)
        assert res == 4
        self.res = res
        self.fmaps = fmaps
        self.gain = 1.0
        self.lrmul = 1.0

        # conv block
        self.conv = FusedModConv(fmaps=self.fmaps, kernel=3, gain=self.gain, lrmul=self.lrmul, style_fmaps=self.fmaps,
                                 demodulate=True, up=False, down=False, resample_kernel=[1, 3, 3, 1], name='conv')
        self.apply_noise = Noise(name='noise')
        self.apply_bias = Bias(lrmul=self.lrmul, name='bias')
        self.leaky_relu = LeakyReLU(name='lrelu')

    def build(self, input_shape):
        # starting const variable
        # tf 1.15 mean(0.0), std(1.0) default value of tf.initializers.random_normal()
        const_init = tf.random.normal(shape=(1, self.fmaps, self.res, self.res), mean=0.0, stddev=1.0)
        self.const = tf.Variable(const_init, name='const', trainable=True)

    def call(self, inputs, training=None, mask=None):
        w0 = inputs
        batch_size = tf.shape(w0)[0]

        # const block
        x = tf.tile(self.const, [batch_size, 1, 1, 1])

        # conv block
        x = self.conv([x, w0])
        x = self.apply_noise(x)
        x = self.apply_bias(x)
        x = self.leaky_relu(x)
        return x

    def get_config(self):
        config = super(SynthesisConstBlock, self).get_config()
        config.update({
            'res': self.res,
            'fmaps': self.fmaps,
            'gain': self.gain,
            'lrmul': self.lrmul,
        })
        return config
#
class SynthesisBlock(tf.keras.layers.Layer):
    def __init__(self, in_ch, fmaps, res, **kwargs):
        super(SynthesisBlock, self).__init__(**kwargs)
        self.in_ch = in_ch
        self.fmaps = fmaps
        self.res = res
        self.gain = 1.0
        self.lrmul = 1.0

        # conv0 up
        self.conv_0 = FusedModConv(fmaps=self.fmaps, kernel=3, gain=self.gain, lrmul=self.lrmul, style_fmaps=self.in_ch,
                                   demodulate=True, up=True, down=False, resample_kernel=[1, 3, 3, 1], name='conv_0')
        self.apply_noise_0 = Noise(name='noise_0')
        self.apply_bias_0 = Bias(lrmul=self.lrmul, name='bias_0')
        self.leaky_relu_0 = LeakyReLU(name='lrelu_0')

        # conv block
        self.conv_1 = FusedModConv(fmaps=self.fmaps, kernel=3, gain=self.gain, lrmul=self.lrmul, style_fmaps=self.fmaps,
                                   demodulate=True, up=False, down=False, resample_kernel=[1, 3, 3, 1], name='conv_1')
        self.apply_noise_1 = Noise(name='noise_1')
        self.apply_bias_1 = Bias(lrmul=self.lrmul, name='bias_1')
        self.leaky_relu_1 = LeakyReLU(name='lrelu_1')

    def call(self, inputs, training=None, mask=None):
        x, w0, w1 = inputs

        # conv0 up
        x = self.conv_0([x, w0])
        x = self.apply_noise_0(x)
        x = self.apply_bias_0(x)
        x = self.leaky_relu_0(x)

        # conv block
        x = self.conv_1([x, w1])
        x = self.apply_noise_1(x)
        x = self.apply_bias_1(x)
        x = self.leaky_relu_1(x)
        return x

    def get_config(self):
        config = super(SynthesisBlock, self).get_config()
        config.update({
            'in_ch': self.in_ch,
            'res': self.res,
            'fmaps': self.fmaps,
            'gain': self.gain,
            'lrmul': self.lrmul,
        })
        return config
#
class Synthesis(tf.keras.layers.Layer):
    def __init__(self, resolutions, featuremaps, name, **kwargs):
        super(Synthesis, self).__init__(name=name, **kwargs)
        self.resolutions = resolutions
        self.featuremaps = featuremaps
        self.k = onmoono.setup_resample_kernel(k=[1, 3, 3, 1])

        # initial layer
        res, n_f = resolutions[0], featuremaps[0]
        self.initial_block = SynthesisConstBlock(fmaps=n_f, res=res, name='{:d}x{:d}/const'.format(res, res))
        self.initial_torgb = ToRGB(in_ch=n_f, name='{:d}x{:d}/ToRGB'.format(res, res))

        # stack generator block with lerp block
        prev_n_f = n_f
        self.blocks = list()
        self.torgbs = list()
        for res, n_f in zip(self.resolutions[1:], self.featuremaps[1:]):
            self.blocks.append(SynthesisBlock(in_ch=prev_n_f, fmaps=n_f, res=res,
                                              name='{:d}x{:d}/block'.format(res, res)))
            self.torgbs.append(ToRGB(in_ch=n_f, name='{:d}x{:d}/ToRGB'.format(res, res)))
            prev_n_f = n_f

    def call(self, inputs, training=None, mask=None):
        w_broadcasted = inputs

        # initial layer
        w0, w1 = w_broadcasted[:, 0], w_broadcasted[:, 1]
        x = self.initial_block(w0)
        y = self.initial_torgb([x, w1])

        layer_index = 1
        for block, torgb in zip(self.blocks, self.torgbs):
            w0 = w_broadcasted[:, layer_index]
            w1 = w_broadcasted[:, layer_index + 1]
            w2 = w_broadcasted[:, layer_index + 2]

            x = block([x, w0, w1])
            y = onmoono.upsample_2d(y, self.k, factor=2, gain=1.0)
            y = y + torgb([x, w2])

            layer_index += 2

        images_out = y
        return images_out

    def get_config(self):
        config = super(Synthesis, self).get_config()
        config.update({
            'resolutions': self.resolutions,
            'featuremaps': self.featuremaps,
            'k': self.k,
        })
        return config
#
class Generator(tf.keras.Model):
    def __init__(self, g_params, **kwargs):
        super(Generator, self).__init__(**kwargs)

        self.z_dim = g_params['z_dim']
        self.w_dim = g_params['w_dim']
        self.labels_dim = g_params['labels_dim']
        self.n_mapping = g_params['n_mapping']
        self.resolutions = g_params['resolutions']
        self.featuremaps = g_params['featuremaps']
        self.w_ema_decay = g_params['w_ema_decay']
        self.style_mixing_prob = g_params['style_mixing_prob']

        self.n_broadcast = len(self.resolutions) * 2
        self.mixing_layer_indices = np.arange(self.n_broadcast)[np.newaxis, :, np.newaxis]

        self.g_mapping = Mapping(self.w_dim, self.labels_dim, self.n_mapping, name='g_mapping')
        self.broadcast = tf.keras.layers.Lambda(lambda x: tf.tile(x[:, np.newaxis], [1, self.n_broadcast, 1]))
        self.synthesis = Synthesis(self.resolutions, self.featuremaps, name='g_synthesis')

    def build(self, input_shape):
        # w_avg
        self.w_avg = tf.Variable(tf.zeros(shape=[self.w_dim], dtype=tf.dtypes.float32), name='w_avg', trainable=False)

    def set_as_moving_average_of(self, src_net, beta=0.99, beta_nontrainable=0.0):
        def split_first_name(name):
            splitted = name.split('/')
            new_name = '/'.join(splitted[1:])
            return new_name

        for cw in self.trainable_weights:
            cw_name = split_first_name(cw.name)
            for sw in src_net.trainable_weights:
                sw_name = split_first_name(sw.name)
                if cw_name == sw_name:
                    assert sw.shape == cw.shape
                    cw.assign(onmoono.lerp(sw, cw, beta))
                    break

        for cw in self.non_trainable_weights:
            cw_name = split_first_name(cw.name)
            for sw in src_net.non_trainable_weights:
                sw_name = split_first_name(sw.name)
                if cw_name == sw_name:
                    assert sw.shape == cw.shape
                    cw.assign(onmoono.lerp(sw, cw, beta_nontrainable))
                    break
        return

    def update_moving_average_of_w(self, w_broadcasted):
        # compute average of current w
        batch_avg = tf.reduce_mean(w_broadcasted[:, 0], axis=0)

        # compute moving average of w and update(assign) w_avg
        self.w_avg.assign(onmoono.lerp(batch_avg, self.w_avg, self.w_ema_decay))
        return

    def style_mixing_regularization(self, latents1, labels, w_broadcasted1):
        # get another w and broadcast it
        latents2 = tf.random.normal(shape=tf.shape(latents1), dtype=tf.dtypes.float32)
        dlatents2 = self.g_mapping([latents2, labels])
        w_broadcasted2 = self.broadcast(dlatents2)

        # find mixing limit index
        if tf.random.uniform([], 0.0, 1.0) < self.style_mixing_prob:
            mixing_cutoff_index = tf.random.uniform([], 1, self.n_broadcast, dtype=tf.dtypes.int32)
        else:
            mixing_cutoff_index = tf.constant(self.n_broadcast, dtype=tf.dtypes.int32)

        # mix it
        mixed_w_broadcasted = tf.where(
            condition=tf.broadcast_to(self.mixing_layer_indices < mixing_cutoff_index, tf.shape(w_broadcasted1)),
            x=w_broadcasted1,
            y=w_broadcasted2)
        return mixed_w_broadcasted

    def truncation_trick(self, w_broadcasted, truncation_cutoff, truncation_psi):
        ones = np.ones_like(self.mixing_layer_indices, dtype=np.float32)
        if truncation_cutoff is None:
            truncation_coefs = ones * truncation_psi
        else:
            truncation_coefs = ones
            for index in range(self.n_broadcast):
                if index < truncation_cutoff:
                    truncation_coefs[:, index, :] = truncation_psi

        truncated_w_broadcasted = onmoono.lerp(self.w_avg, w_broadcasted, truncation_coefs)
        return truncated_w_broadcasted

    @tf.function
    def call(self, inputs, truncation_cutoff=None, truncation_psi=1.0, 
        ret_w_broadcasted=True, 
        training=None, mask=None):

        latents, labels = inputs

        dlatents = self.g_mapping([latents, labels])
        w_broadcasted = self.broadcast(dlatents)

        if training:
            self.update_moving_average_of_w(w_broadcasted)
            w_broadcasted = self.style_mixing_regularization(latents, labels, w_broadcasted)

        if not training:
            w_broadcasted = self.truncation_trick(w_broadcasted, truncation_cutoff, truncation_psi)

        image_out = self.synthesis(w_broadcasted)
        return image_out, w_broadcasted

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)

        # shape_latents, shape_labels = input_shape
        return input_shape[0][0], 3, self.resolutions[-1], self.resolutions[-1]

    @tf.function
    def serve(self, latents, labels, truncation_psi):
        dlatents = self.g_mapping([latents, labels])
        w_broadcasted = self.broadcast(dlatents)
        w_broadcasted = self.truncation_trick(w_broadcasted, truncation_cutoff=None, truncation_psi=truncation_psi)
        image_out = self.synthesis(w_broadcasted)

        image_out.set_shape([None, 3, self.resolutions[-1], self.resolutions[-1]])
        return image_out
#
#
#    -> https://github.com/moono/stylegan2-tf-2.x/tree/master/stylegan2/discriminator.py <-
#
#
class FromRGB(tf.keras.layers.Layer):
    def __init__(self, fmaps, **kwargs):
        super(FromRGB, self).__init__(**kwargs)
        self.fmaps = fmaps

        self.conv = ResizeConv2D(fmaps=self.fmaps, kernel=1, gain=1.0, lrmul=1.0,
                                 up=False, down=False, resample_kernel=None, name='conv')
        self.apply_bias = Bias(lrmul=1.0, name='bias')
        self.leaky_relu = LeakyReLU(name='lrelu')

    def call(self, inputs, training=None, mask=None):
        y = self.conv(inputs)
        y = self.apply_bias(y)
        y = self.leaky_relu(y)
        return y
#
class DiscriminatorBlock(tf.keras.layers.Layer):
    def __init__(self, n_f0, n_f1, res, **kwargs):
        super(DiscriminatorBlock, self).__init__(**kwargs)
        self.gain = 1.0
        self.lrmul = 1.0
        self.n_f0 = n_f0
        self.n_f1 = n_f1
        self.res = res
        self.resnet_scale = 1. / np.sqrt(2.)

        # conv_0
        self.conv_0 = ResizeConv2D(fmaps=self.n_f0, kernel=3, gain=self.gain, lrmul=self.lrmul,
                                   up=False, down=False, resample_kernel=None, name='conv_0')
        self.apply_bias_0 = Bias(self.lrmul, name='bias_0')
        self.leaky_relu_0 = LeakyReLU(name='lrelu_0')

        # conv_1 down
        self.conv_1 = ResizeConv2D(fmaps=self.n_f1, kernel=3, gain=self.gain, lrmul=self.lrmul,
                                   up=False, down=True, resample_kernel=[1, 3, 3, 1], name='conv_1')
        self.apply_bias_1 = Bias(self.lrmul, name='bias_1')
        self.leaky_relu_1 = LeakyReLU(name='lrelu_1')

        # resnet skip
        self.conv_skip = ResizeConv2D(fmaps=self.n_f1, kernel=1, gain=self.gain, lrmul=self.lrmul,
                                      up=False, down=True, resample_kernel=[1, 3, 3, 1], name='skip')

    def call(self, inputs, training=None, mask=None):
        x = inputs
        residual = x

        # conv0
        x = self.conv_0(x)
        x = self.apply_bias_0(x)
        x = self.leaky_relu_0(x)

        # conv1 down
        x = self.conv_1(x)
        x = self.apply_bias_1(x)
        x = self.leaky_relu_1(x)

        # resnet skip
        residual = self.conv_skip(residual)
        x = (x + residual) * self.resnet_scale
        return x
#
class DiscriminatorLastBlock(tf.keras.layers.Layer):
    def __init__(self, n_f0, n_f1, res, **kwargs):
        super(DiscriminatorLastBlock, self).__init__(**kwargs)
        self.gain = 1.0
        self.lrmul = 1.0
        self.n_f0 = n_f0
        self.n_f1 = n_f1
        self.res = res

        self.minibatch_std = MinibatchStd(group_size=4, num_new_features=1, name='minibatchstd')

        # conv_0
        self.conv_0 = ResizeConv2D(fmaps=self.n_f0, kernel=3, gain=self.gain, lrmul=self.lrmul,
                                   up=False, down=False, resample_kernel=None, name='conv_0')
        self.apply_bias_0 = Bias(self.lrmul, name='bias_0')
        self.leaky_relu_0 = LeakyReLU(name='lrelu_0')

        # dense_1
        self.dense_1 = Dense(self.n_f1, gain=self.gain, lrmul=self.lrmul, name='dense_1')
        self.apply_bias_1 = Bias(self.lrmul, name='bias_1')
        self.leaky_relu_1 = LeakyReLU(name='lrelu_1')

    def call(self, x, training=None, mask=None):
        x = self.minibatch_std(x)

        # conv_0
        x = self.conv_0(x)
        x = self.apply_bias_0(x)
        x = self.leaky_relu_0(x)

        # dense_1
        x = self.dense_1(x)
        x = self.apply_bias_1(x)
        x = self.leaky_relu_1(x)
        return x
#
class Discriminator(tf.keras.Model):
    def __init__(self, d_params, **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        # discriminator's (resolutions and featuremaps) are reversed against generator's
        self.labels_dim = d_params['labels_dim']
        self.r_resolutions = d_params['resolutions'][::-1]
        self.r_featuremaps = d_params['featuremaps'][::-1]

        # stack discriminator blocks
        res0, n_f0 = self.r_resolutions[0], self.r_featuremaps[0]
        self.initial_fromrgb = FromRGB(fmaps=n_f0, name='{:d}x{:d}/FromRGB'.format(res0, res0))
        self.blocks = list()
        for index, (res0, n_f0) in enumerate(zip(self.r_resolutions[:-1], self.r_featuremaps[:-1])):
            n_f1 = self.r_featuremaps[index + 1]
            self.blocks.append(DiscriminatorBlock(n_f0=n_f0, n_f1=n_f1, res=res0, name='{:d}x{:d}'.format(res0, res0)))

        # set last discriminator block
        res = self.r_resolutions[-1]
        n_f0, n_f1 = self.r_featuremaps[-2], self.r_featuremaps[-1]
        self.last_block = DiscriminatorLastBlock(n_f0, n_f1, res, name='{:d}x{:d}'.format(res, res))

        # set last dense layer
        self.last_dense = Dense(max(self.labels_dim, 1), gain=1.0, lrmul=1.0, name='last_dense')
        self.last_bias = Bias(lrmul=1.0, name='last_bias')

    def call(self, inputs, training=None, mask=None):
        images, labels = inputs

        x = self.initial_fromrgb(images)
        for block in self.blocks:
            x = block(x)

        x = self.last_block(x)
        x = self.last_dense(x)
        x = self.last_bias(x)

        if self.labels_dim > 0:
            x = tf.reduce_sum(x * labels, axis=1, keepdims=True)
        scores_out = x
        return scores_out
#
#
#    -> https://github.com/moono/stylegan2-tf-2.x/tree/master/stylegan2/train_advanced.py <-
#
#
class StyleGAN2(object):
    def __init__(self, t_params, name):
        # _e_
        self.cur_tf_ver = t_params['cur_tf_ver']
        self.use_tf_function = t_params['use_tf_function']
        self.use_custom_cuda = t_params['use_custom_cuda']

        self.model_base_dir = t_params['model_base_dir']
        self.tfrecord_dir = t_params['tfrecord_dir']
        self.shuffle_buffer_size = t_params['shuffle_buffer_size']
        self.g_params = t_params['g_params']
        self.d_params = t_params['d_params']
        self.g_opt = t_params['g_opt']
        self.d_opt = t_params['d_opt']
        self.batch_size = t_params['batch_size']
        self.n_total_image = t_params['n_total_image']
        self.n_samples = min(t_params['batch_size'], t_params['n_samples'])
        self.lazy_regularization = t_params['lazy_regularization']

        self.global_batch_size = t_params['batch_size']
        self.global_batch_scaler = 1.0 / float(self.global_batch_size)

        self.r1_gamma = 10.0
        # self.r2_gamma = 0.0
        self.max_steps = int(np.ceil(self.n_total_image / self.batch_size))
        self.out_res = self.g_params['resolutions'][-1]
        self.log_template = 'step {}: elapsed: {:.2f}s, d_loss: {:.3f}, g_loss: {:.3f}, r1_reg: {:.3f}, pl_reg: {:.3f}'
        self.print_step = 10
        self.save_step = 100
        self.image_summary_step = 100
        self.reached_max_steps = False

        # setup optimizer params
        self.g_opt = self.set_optimizer_params(self.g_opt)
        self.d_opt = self.set_optimizer_params(self.d_opt)
        self.pl_mean = tf.Variable(initial_value=0.0, name='pl_mean', trainable=False)
        self.pl_decay = 0.01
        self.pl_weight = 1.0
        self.pl_denorm = 1.0 / np.sqrt(self.out_res * self.out_res)

        # Trainer
        self.pl_minibatch_shrink = 2
        self.pl_weight = float(self.pl_minibatch_shrink)

        ## grab dataset
        #print('|...> StyleGAN2 Trainer setting datasets')
        #self.dataset = folder_to_dataset(
        #        self.tfrecord_dir, 
        #        self.out_res, 
        #        self.shuffle_buffer_size,
        #        self.batch_size, 
        #        epochs=None)

        # create models
        print('|...> StyleGAN2 Trainer  Create models')
        #self.generator = Generator(self.g_params)
        #self.discriminator = Discriminator(self.d_params)
        #self.g_clone = Generator(self.g_params)

        # create model: model and optimizer must be created under `strategy.scope`
        #self.discriminator, self.generator, self.g_clone = initiate_models(self.g_params,
        #                                                                   self.d_params,
        #                                                                   self.use_custom_cuda)


        self.discriminator = load_discriminator(self.d_params, ckpt_dir=None, custom_cuda=self.use_custom_cuda)
        self.generator = load_generator(g_params=self.g_params, is_g_clone=False, ckpt_dir=None, custom_cuda=self.use_custom_cuda)
        self.g_clone = load_generator(g_params=self.g_params, is_g_clone=True, ckpt_dir=None, custom_cuda=self.use_custom_cuda)

        # set initial g_clone weights same as generator
        self.g_clone.set_weights(self.generator.get_weights())


        self.d_optimizer = tf.keras.optimizers.Adam(self.d_opt['learning_rate'],
                                                    beta_1=self.d_opt['beta1'],
                                                    beta_2=self.d_opt['beta2'],
                                                    epsilon=self.d_opt['epsilon'])
        self.g_optimizer = tf.keras.optimizers.Adam(self.g_opt['learning_rate'],
                                                    beta_1=self.g_opt['beta1'],
                                                    beta_2=self.g_opt['beta2'],
                                                    epsilon=self.g_opt['epsilon'])

        # finalize model (build)
        test_latent = np.ones((1, self.g_params['z_dim']), dtype=np.float32)
        test_labels = np.ones((1, self.g_params['labels_dim']), dtype=np.float32)
        test_images = np.ones((1, 3, self.out_res, self.out_res), dtype=np.float32)
        _, __ = self.generator([test_latent, test_labels], training=False)
        _ = self.discriminator([test_images, test_labels], training=False)
        _, __ = self.g_clone([test_latent, test_labels], training=False)
        print(f'|...> StyleGAN2 Copying g_clone')
        self.g_clone.set_weights(self.generator.get_weights())

        # setup saving locations (object based savings)
        self.ckpt_dir = os.path.join(self.model_base_dir, name)
        self.ckpt = tf.train.Checkpoint(d_optimizer=self.d_optimizer,
                                        g_optimizer=self.g_optimizer,
                                        discriminator=self.discriminator,
                                        generator=self.generator,
                                        g_clone=self.g_clone)
        self.manager = tf.train.CheckpointManager(self.ckpt, self.ckpt_dir, max_to_keep=2)

        # try to restore
        self.ckpt.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            print(f'|...> StyleGAN2 Restored from {self.manager.latest_checkpoint}')

            # check if already trained in this resolution
            restored_step = self.g_optimizer.iterations.numpy()
            if restored_step >= self.max_steps:
                print('Already reached max steps {}/{}'.format(restored_step, self.max_steps))
                self.reached_max_steps = True
                return
        else:
            print(f'|...> StyleGAN2 Not restoring checkpoint')

    def set_optimizer_params(self, params):
        if self.lazy_regularization:
            mb_ratio = params['reg_interval'] / (params['reg_interval'] + 1)
            params['learning_rate'] = params['learning_rate'] * mb_ratio
            params['beta1'] = params['beta1'] ** mb_ratio
            params['beta2'] = params['beta2'] ** mb_ratio
        return params

    @staticmethod
    def update_optimizer_params(params):
        params_copy = params.copy()
        mb_ratio = params_copy['reg_interval'] / (params_copy['reg_interval'] + 1)
        params_copy['learning_rate'] = params_copy['learning_rate'] * mb_ratio
        params_copy['beta1'] = params_copy['beta1'] ** mb_ratio
        params_copy['beta2'] = params_copy['beta2'] ** mb_ratio
        return params_copy

    @tf.function
    def d_train_step(self, z, real_images, labels):
        with tf.GradientTape() as d_tape:
            # forward pass
            fake_images, _ = self.generator([z, labels], training=True)
            real_scores = self.discriminator([real_images, labels], training=True)
            fake_scores = self.discriminator([fake_images, labels], training=True)

            # gan loss
            d_loss = tf.math.softplus(fake_scores)
            d_loss += tf.math.softplus(-real_scores)
            d_loss = tf.reduce_mean(d_loss)
        d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
        return d_loss

    @tf.function
    def d_reg_train_step(self, z, real_images, labels):
        with tf.GradientTape() as d_tape:
            # forward pass
            fake_images, _ = self.generator([z, labels], training=True)
            real_scores = self.discriminator([real_images, labels], training=True)
            fake_scores = self.discriminator([fake_images, labels], training=True)

            # gan loss
            d_loss = tf.math.softplus(fake_scores)
            d_loss += tf.math.softplus(-real_scores)

            # simple GP
            with tf.GradientTape() as p_tape:
                p_tape.watch(real_images)
                real_loss = tf.reduce_sum(self.discriminator([real_images, labels], training=True))

            real_grads = p_tape.gradient(real_loss, real_images)
            r1_penalty = tf.reduce_sum(tf.math.square(real_grads), axis=[1, 2, 3])
            r1_penalty = tf.expand_dims(r1_penalty, axis=1)
            r1_penalty = r1_penalty * self.d_opt['reg_interval']

            # combine
            d_loss += r1_penalty * (0.5 * self.r1_gamma)
            d_loss = tf.reduce_mean(d_loss)

        d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
        return d_loss, tf.reduce_mean(r1_penalty)

    @tf.function
    def g_train_step(self, z, labels):
        with tf.GradientTape() as g_tape:
            # forward pass
            fake_images, _ = self.generator([z, labels], training=True)
            fake_scores = self.discriminator([fake_images, labels], training=True)

            # gan loss
            g_loss = tf.math.softplus(-fake_scores)
            g_loss = tf.reduce_mean(g_loss)

        g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        return g_loss

    def g_reg_pl(self, z, labels):    # _e_

        ## path length regularization
        ## Compute |J*y|
        with tf.GradientTape() as pl_tape:
            pl_tape.watch([z, labels])

            pl_fake_images, pl_w_broadcasted = self.generator([z, labels], training=True)

            pl_noise = tf.random.normal(tf.shape(pl_fake_images), mean=0.0, stddev=1.0, dtype=tf.float32) * self.pl_denorm
            pl_noise_added = tf.reduce_sum(pl_fake_images * pl_noise)

            ##  pl_noise_added:  tf.Tensor(2.973158, shape=(), dtype=float32)
            ##  pl_w_broadcasted:   tf.Tensor([[[-0.25851128  0.28737155  1.8203238  ... -0.5302262  -0.00633283

        pl_grads = pl_tape.gradient(pl_noise_added, pl_w_broadcasted) # None
        print(f"\n|****> g_reg_pl pl_grads: {pl_grads} \n")
        return pl_grads

    # _e_
    def g_reg_train_step(self, dist_inputs):
        real_images = dist_inputs

        with tf.GradientTape() as g_tape:
            # compute losses
            g_gan_loss, pl_penalty = g_logistic_ns_pathreg(real_images, self.generator, self.discriminator,
                                                           self.g_params['z_dim'], self.g_params['labels_dim'],
                                                           self.pl_mean, self.pl_minibatch_shrink, self.pl_denorm,
                                                           self.pl_decay)
            pl_penalty = pl_penalty * self.pl_weight * self.g_opt['reg_interval']

            # scale loss
            pl_penalty = tf.reduce_sum(pl_penalty) * self.global_batch_scaler
            g_gan_loss = tf.reduce_sum(g_gan_loss) * self.global_batch_scaler

            # combine
            g_loss = g_gan_loss + pl_penalty

        g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        return g_loss, g_gan_loss, pl_penalty

    # @tf.function
    def __g_reg_train_step(self, z, labels):
        print(f'|...>       g_reg_train_step z   {np.shape(z)}')
        print(f'|...>       g_reg_train_step labels   {np.shape(labels)}')
        print(f'|...>       g_reg_train_step self.pl_mean   {self.pl_mean}')
        with tf.GradientTape() as g_tape:
            # forward pass
            fake_images, w_broadcasted = self.generator([z, labels], training=True)
            fake_scores = self.discriminator([fake_images, labels], training=True)

            # gan loss
            g_loss = tf.math.softplus(-fake_scores)

            pl_grads = self.g_reg_pl(z, labels)
            print(f"|...>  pl_grads: {pl_grads}")

            # ***************************************************************************
            pl_lengths = 0.0                                # _e_ ***********************
            if pl_grads:
                pl_lengths = tf.math.sqrt(tf.reduce_mean(tf.reduce_sum(tf.math.square(pl_grads), axis=2), axis=1))
            # ***************************************************************************
            print(f"|...>  pl_lengths: {pl_lengths}")

            # Track exponential moving average of |J*y|.
            pl_mean_val = self.pl_mean + self.pl_decay * (tf.reduce_mean(pl_lengths) - self.pl_mean)
            print(f"|...>  pl_mean_val: {pl_mean_val}")
            self.pl_mean.assign(pl_mean_val)

            # Calculate (|J*y|-a)^2.
            pl_penalty = tf.square(pl_lengths - self.pl_mean)
            print(f"|...>  pl_penalty: {pl_penalty}")


            # compute
            _pl_reg = pl_penalty * self.pl_weight
            print(f"|...>  _pl_reg: {_pl_reg}")

            # combine
            g_loss += _pl_reg
            g_loss = tf.reduce_mean(g_loss)

        g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)

        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        div = tf.reduce_mean(_pl_reg)

        print(f"|...>  g_loss: {g_loss}")
        print(f"|...>  div: {div}")
        return g_loss, div

    def train(self, dataset, strategy=None):
        if self.reached_max_steps:
            return

        # start actual training
        print(f'|---> train up to {self.max_steps} steps')

        # setup tensorboards
        train_summary_writer = tf.summary.create_file_writer(self.ckpt_dir)

        # loss metrics
        metric_g_loss = tf.keras.metrics.Mean('g_loss', dtype=tf.float32)
        metric_d_loss = tf.keras.metrics.Mean('d_loss', dtype=tf.float32)
        metric_r1_reg = tf.keras.metrics.Mean('r1_reg', dtype=tf.float32)
        metric_pl_reg = tf.keras.metrics.Mean('pl_reg', dtype=tf.float32)

        # start training
        losses = {'g_loss': 0.0, 'd_loss': 0.0, 'r1_reg': 0.0, 'pl_reg': 0.0}
        t_start = time.time()

        for real_images in dataset:
            # preprocess inputs
            z = tf.random.normal(shape=[tf.shape(real_images)[0], self.g_params['z_dim']], dtype=tf.dtypes.float32)
            real_images = onmoono.preprocess_fit_train_image(real_images, self.out_res)
            labels = tf.ones((tf.shape(real_images)[0], self.g_params['labels_dim']), dtype=tf.dtypes.float32)

            # get current step
            step = self.g_optimizer.iterations.numpy()

            # d train step
            if step % self.d_opt['reg_interval'] == 0:
                d_loss, r1_reg = self.d_reg_train_step(z, real_images, labels)

                # update values for printing
                losses['d_loss'] = d_loss.numpy()
                losses['r1_reg'] = r1_reg.numpy()

                # update metrics
                metric_d_loss(d_loss)
                metric_r1_reg(r1_reg)
            else:
                d_loss = self.d_train_step(z, real_images, labels)

                # update values for printing
                losses['d_loss'] = d_loss.numpy()
                losses['r1_reg'] = 0.0

                # update metrics
                metric_d_loss(d_loss)

            # update g_clone
            self.g_clone.set_as_moving_average_of(self.generator)
         
            # g train step
            if step % self.g_opt['reg_interval'] == 0:
                # _e_
                #g_loss, pl_reg = self.g_reg_train_step(z, labels)
                # g_loss, g_gan_loss, pl_penalty
                g_loss, pl_reg, pl_penalty = self.g_reg_train_step(real_images)

                # update values for printing
                losses['g_loss'] = g_loss.numpy()
                losses['pl_reg'] = pl_reg.numpy()

                # update metrics
                metric_g_loss(g_loss)
                metric_pl_reg(pl_reg)
            else:
                g_loss = self.g_train_step(z, labels)

                # update values for printing
                losses['g_loss'] = g_loss.numpy()
                losses['pl_reg'] = 0.0

                # update metrics
                metric_g_loss(g_loss)

            # save to tensorboard
            with train_summary_writer.as_default():
                tf.summary.scalar('g_loss', metric_g_loss.result(), step=step)
                tf.summary.scalar('d_loss', metric_d_loss.result(), step=step)
                tf.summary.scalar('r1_reg', metric_r1_reg.result(), step=step)
                tf.summary.scalar('pl_reg', metric_pl_reg.result(), step=step)
                tf.summary.histogram('w_avg', self.generator.w_avg, step=step)

            # save every self.save_step
            if step % self.save_step == 0:
                self.manager.save(checkpoint_number=step)

            # save every self.image_summary_step
            if step % self.image_summary_step == 0:
                # add summary image
                summary_image = self.sample_images_tensorboard(real_images)
                with train_summary_writer.as_default():
                    tf.summary.image('images', summary_image, step=step)

            # print every self.print_steps
            if step % self.print_step == 0:
                elapsed = time.time() - t_start
                print(self.log_template.format(step, elapsed,
                        losses['d_loss'], losses['g_loss'], losses['r1_reg'], losses['pl_reg']))

                # reset timer
                t_start = time.time()

            # check exit status
            if step >= self.max_steps:
                break

        # get current step
        step = self.g_optimizer.iterations.numpy()
        elapsed = time.time() - t_start
        print(self.log_template.format(step, elapsed,
                        losses['d_loss'], losses['g_loss'], losses['r1_reg'], losses['pl_reg']))

        # save last checkpoint
        self.manager.save(checkpoint_number=step)
        return

    def sample_images_tensorboard(self, real_images):
        # prepare inputs
        reals = real_images[:self.n_samples, :, :, :]
        latents = tf.random.normal(shape=(self.n_samples, self.g_params['z_dim']), dtype=tf.dtypes.float32)
        dummy_labels = tf.ones((self.n_samples, self.g_params['labels_dim']), dtype=tf.dtypes.float32)

        # run networks
        fake_images_00, _ = self.g_clone([latents, dummy_labels], truncation_psi=0.0, training=False)
        fake_images_05, _ = self.g_clone([latents, dummy_labels], truncation_psi=0.5, training=False)
        fake_images_07, _ = self.g_clone([latents, dummy_labels], truncation_psi=0.7, training=False)
        fake_images_10, _ = self.g_clone([latents, dummy_labels], truncation_psi=1.0, training=False)

        # merge on batch dimension: [5 * n_samples, 3, out_res, out_res]
        out = tf.concat([reals, fake_images_00, fake_images_05, fake_images_07, fake_images_10], axis=0)

        # prepare for image saving: [5 * n_samples, out_res, out_res, 3]
        out = onmoono.postprocess_images(out)

        # resize to save disk spaces: [5 * n_samples, size, size, 3]
        size = min(self.out_res, 256)
        out = tf.image.resize(out, size=[size, size])

        # make single image and add batch dimension for tensorboard: [1, 5 * size, n_samples * size, 3]
        out = onmoono.merge_batch_images(out, size, rows=5, cols=self.n_samples)
        out = np.expand_dims(out, axis=0)
        return out

#def initiate_models(g_params, d_params, use_custom_cuda):
#    discriminator = load_discriminator(d_params, ckpt_dir=None, custom_cuda=use_custom_cuda)
#    generator = load_generator(g_params=g_params, is_g_clone=False, ckpt_dir=None, custom_cuda=use_custom_cuda)
#    g_clone = load_generator(g_params=g_params, is_g_clone=True, ckpt_dir=None, custom_cuda=use_custom_cuda)

#    # set initial g_clone weights same as generator
#    g_clone.set_weights(generator.get_weights())
#    return discriminator, generator, g_clone

class Trainer(object):
    def __init__(self, t_params, name):
        self.cur_tf_ver = t_params['cur_tf_ver']
        self.use_tf_function = t_params['use_tf_function']
        self.use_custom_cuda = t_params['use_custom_cuda']
        self.model_base_dir = t_params['model_base_dir']
        self.global_batch_size = t_params['batch_size']
        self.n_total_image = t_params['n_total_image']
        self.max_steps = int(np.ceil(self.n_total_image / self.global_batch_size))
        self.n_samples = min(t_params['batch_size'], t_params['n_samples'])
        self.train_res = t_params['train_res']
        self.print_step = 10
        self.save_step = 100
        self.image_summary_step = 100
        self.reached_max_steps = False
        self.log_template = '{:s}, {:s}, {:s}'.format(
            'step {}: elapsed: {:.2f}s, d_loss: {:.3f}, g_loss: {:.3f}',
            'd_gan_loss: {:.3f}, g_gan_loss: {:.3f}',
            'r1_penalty: {:.3f}, pl_penalty: {:.3f}')

        # copy network params
        self.g_params = t_params['g_params']
        self.d_params = t_params['d_params']

        # set optimizer params
        self.global_batch_scaler = 1.0 / float(self.global_batch_size)
        self.r1_gamma = 10.0
        self.g_opt = self.update_optimizer_params(t_params['g_opt'])
        self.d_opt = self.update_optimizer_params(t_params['d_opt'])
        self.pl_minibatch_shrink = 2
        self.pl_weight = float(self.pl_minibatch_shrink)
        self.pl_denorm = tf.math.rsqrt(float(self.train_res) * float(self.train_res))
        self.pl_decay = 0.01
        self.pl_mean = tf.Variable(initial_value=0.0, name='pl_mean', trainable=False,
                                   synchronization=tf.VariableSynchronization.ON_READ,
                                   aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)

        # create model: model and optimizer must be created under `strategy.scope`
        #self.discriminator, self.generator, self.g_clone = initiate_models(self.g_params,
        #                                                                   self.d_params,
        #                                                                   self.use_custom_cuda)


        self.discriminator = load_discriminator(self.d_params, ckpt_dir=None, custom_cuda=self.use_custom_cuda)
        self.generator = load_generator(g_params=self.g_params, is_g_clone=False, ckpt_dir=None, custom_cuda=self.use_custom_cuda)
        self.g_clone = load_generator(g_params=self.g_params, is_g_clone=True, ckpt_dir=None, custom_cuda=self.use_custom_cuda)


        # set optimizers
        self.d_optimizer = tf.keras.optimizers.Adam(self.d_opt['learning_rate'],
                                                    beta_1=self.d_opt['beta1'],
                                                    beta_2=self.d_opt['beta2'],
                                                    epsilon=self.d_opt['epsilon'])
        self.g_optimizer = tf.keras.optimizers.Adam(self.g_opt['learning_rate'],
                                                    beta_1=self.g_opt['beta1'],
                                                    beta_2=self.g_opt['beta2'],
                                                    epsilon=self.g_opt['epsilon'])

        # setup saving locations (object based savings)
        self.ckpt_dir = os.path.join(self.model_base_dir, name)
        self.ckpt = tf.train.Checkpoint(d_optimizer=self.d_optimizer,
                                        g_optimizer=self.g_optimizer,
                                        discriminator=self.discriminator,
                                        generator=self.generator,
                                        g_clone=self.g_clone,
                                        pl_mean=self.pl_mean)
        self.manager = tf.train.CheckpointManager(self.ckpt, self.ckpt_dir, max_to_keep=2)

        # try to restore
        self.ckpt.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            print('Restored from {}'.format(self.manager.latest_checkpoint))

            # check if already trained in this resolution
            restored_step = self.g_optimizer.iterations.numpy()
            if restored_step >= self.max_steps:
                print('Already reached max steps {}/{}'.format(restored_step, self.max_steps))
                self.reached_max_steps = True
                return
        else:
            print('Not restoring from saved checkpoint')

    @staticmethod
    def update_optimizer_params(params):
        params_copy = params.copy()
        mb_ratio = params_copy['reg_interval'] / (params_copy['reg_interval'] + 1)
        params_copy['learning_rate'] = params_copy['learning_rate'] * mb_ratio
        params_copy['beta1'] = params_copy['beta1'] ** mb_ratio
        params_copy['beta2'] = params_copy['beta2'] ** mb_ratio
        return params_copy

    def d_train_step(self, dist_inputs):
        real_images = dist_inputs[0]

        with tf.GradientTape() as d_tape:
            # compute losses
            d_loss = d_logistic(real_images, self.generator, self.discriminator,
                                self.g_params['z_dim'], self.g_params['labels_dim'])

            # scale loss
            d_loss = tf.reduce_sum(d_loss) * self.global_batch_scaler

        d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
        return d_loss

    def d_train_step_reg(self, dist_inputs):
        real_images = dist_inputs[0]

        with tf.GradientTape() as d_tape:
            # compute losses
            d_gan_loss, r1_penalty = d_logistic_r1_reg(real_images, self.generator, self.discriminator,
                                                       self.g_params['z_dim'], self.d_params['labels_dim'])
            r1_penalty = r1_penalty * (0.5 * self.r1_gamma) * self.d_opt['reg_interval']

            # scale losses
            r1_penalty = tf.reduce_sum(r1_penalty) * self.global_batch_scaler
            d_gan_loss = tf.reduce_sum(d_gan_loss) * self.global_batch_scaler

            # combine
            d_loss = d_gan_loss + r1_penalty

        d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
        return d_loss, d_gan_loss, r1_penalty

    def g_train_step(self, dist_inputs):
        real_images = dist_inputs[0]

        with tf.GradientTape() as g_tape:
            # compute losses
            g_loss = g_logistic_non_saturating(real_images, self.generator, self.discriminator,
                                               self.g_params['z_dim'], self.g_params['labels_dim'])

            # scale loss
            g_loss = tf.reduce_sum(g_loss) * self.global_batch_scaler

        g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        return g_loss

    def g_train_step_reg(self, dist_inputs):
        real_images = dist_inputs[0]

        with tf.GradientTape() as g_tape:
            # compute losses
            g_gan_loss, pl_penalty = g_logistic_ns_pathreg(real_images, self.generator, self.discriminator,
                                                           self.g_params['z_dim'], self.g_params['labels_dim'],
                                                           self.pl_mean, self.pl_minibatch_shrink, self.pl_denorm,
                                                           self.pl_decay)
            pl_penalty = pl_penalty * self.pl_weight * self.g_opt['reg_interval']

            # scale loss
            pl_penalty = tf.reduce_sum(pl_penalty) * self.global_batch_scaler
            g_gan_loss = tf.reduce_sum(g_gan_loss) * self.global_batch_scaler

            # combine
            g_loss = g_gan_loss + pl_penalty

        g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        return g_loss, g_gan_loss, pl_penalty

    def train(self, dist_datasets, strategy):
        def dist_d_train_step(inputs):
            if self.cur_tf_ver == '2.0.0':
                per_replica_losses = strategy.experimental_run_v2(fn=self.d_train_step, args=(inputs,))
            else:
                per_replica_losses = strategy.run(fn=self.d_train_step, args=(inputs,))

            mean_d_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
            return mean_d_loss

        def dist_d_train_step_reg(inputs):
            if self.cur_tf_ver == '2.0.0':
                per_replica_losses = strategy.experimental_run_v2(fn=self.d_train_step_reg, args=(inputs,))
            else:
                per_replica_losses = strategy.run(fn=self.d_train_step_reg, args=(inputs,))
            mean_d_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses[0], axis=None)
            mean_d_gan_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses[1], axis=None)
            mean_r1_penalty = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses[2], axis=None)
            return mean_d_loss, mean_d_gan_loss, mean_r1_penalty

        def dist_g_train_step(inputs):
            if self.cur_tf_ver == '2.0.0':
                per_replica_losses = strategy.experimental_run_v2(fn=self.g_train_step, args=(inputs,))
            else:
                per_replica_losses = strategy.run(fn=self.g_train_step, args=(inputs,))
            mean_g_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
            return mean_g_loss

        def dist_g_train_step_reg(inputs):
            if self.cur_tf_ver == '2.0.0':
                per_replica_losses = strategy.experimental_run_v2(fn=self.g_train_step_reg, args=(inputs,))
            else:
                per_replica_losses = strategy.run(fn=self.g_train_step_reg, args=(inputs,))
            mean_g_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses[0], axis=None)
            mean_g_gan_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses[1], axis=None)
            mean_pl_penalty = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses[2], axis=None)
            return mean_g_loss, mean_g_gan_loss, mean_pl_penalty

        def dist_gen_samples(dist_inputs):
            if self.cur_tf_ver == '2.0.0':
                per_replica_samples = strategy.experimental_run_v2(self.gen_samples, args=(dist_inputs,))
            else:
                per_replica_samples = strategy.run(self.gen_samples, args=(dist_inputs,))
            return per_replica_samples

        # wrap with tf.function
        if self.use_tf_function:
            dist_d_train_step = tf.function(dist_d_train_step)
            dist_g_train_step = tf.function(dist_g_train_step)
            dist_d_train_step_reg = tf.function(dist_d_train_step_reg)
            dist_g_train_step_reg = tf.function(dist_g_train_step_reg)
            dist_gen_samples = tf.function(dist_gen_samples)

        if self.reached_max_steps:
            return

        # start actual training
        print(f'|===> Start Training')

        # setup tensorboards
        train_summary_writer = tf.summary.create_file_writer(self.ckpt_dir)

        # loss metrics
        metric_d_loss = tf.keras.metrics.Mean('d_loss', dtype=tf.float32)
        metric_g_loss = tf.keras.metrics.Mean('g_loss', dtype=tf.float32)
        metric_d_gan_loss = tf.keras.metrics.Mean('d_gan_loss', dtype=tf.float32)
        metric_g_gan_loss = tf.keras.metrics.Mean('g_gan_loss', dtype=tf.float32)
        metric_r1_penalty = tf.keras.metrics.Mean('r1_penalty', dtype=tf.float32)
        metric_pl_penalty = tf.keras.metrics.Mean('pl_penalty', dtype=tf.float32)

        # start training
        zero = tf.constant(0.0, dtype=tf.float32)
        print(f'|...> train max_steps: {self.max_steps}')
        t_start = time.time()
        for real_images in dist_datasets:
            step = self.g_optimizer.iterations.numpy()
            print(f'|...> train step: {step}')

            # d train step
            print(f'|...> d train step')
            if (step + 1) % self.d_opt['reg_interval'] == 0:
                d_loss, d_gan_loss, r1_penalty = dist_d_train_step_reg((real_images, ))
            else:
                d_loss = dist_d_train_step((real_images, ))
                d_gan_loss = d_loss
                r1_penalty = zero

            # g train step
            print(f'|...> g train step')
            if (step + 1) % self.g_opt['reg_interval'] == 0:
                g_loss, g_gan_loss, pl_penalty = dist_g_train_step_reg((real_images, ))
            else:
                g_loss = dist_g_train_step((real_images,))
                g_gan_loss = g_loss
                pl_penalty = zero

            # update g_clone
            self.g_clone.set_as_moving_average_of(self.generator)

            # update metrics
            metric_d_loss(d_loss)
            metric_g_loss(g_loss)
            metric_d_gan_loss(d_gan_loss)
            metric_g_gan_loss(g_gan_loss)
            metric_r1_penalty(r1_penalty)
            metric_pl_penalty(pl_penalty)

            # get current step
            step = self.g_optimizer.iterations.numpy()

            # save to tensorboard
            with train_summary_writer.as_default():
                tf.summary.scalar('d_loss', metric_d_loss.result(), step=step)
                tf.summary.scalar('g_loss', metric_g_loss.result(), step=step)
                tf.summary.scalar('d_gan_loss', metric_d_gan_loss.result(), step=step)
                tf.summary.scalar('g_gan_loss', metric_g_gan_loss.result(), step=step)
                tf.summary.scalar('r1_penalty', metric_r1_penalty.result(), step=step)
                tf.summary.scalar('pl_penalty', metric_pl_penalty.result(), step=step)

            # save every self.save_step
            if step % self.save_step == 0:
                self.manager.save(checkpoint_number=step)

            # save every self.image_summary_step
            if step % self.image_summary_step == 0:
                # add summary image
                test_z = tf.random.normal(shape=(self.n_samples, self.g_params['z_dim']), dtype=tf.dtypes.float32)
                test_labels = tf.ones((self.n_samples, self.g_params['labels_dim']), dtype=tf.dtypes.float32)
                summary_image = dist_gen_samples((test_z, test_labels))

                # convert to tensor image
                summary_image = self.convert_per_replica_image(summary_image, strategy)

                with train_summary_writer.as_default():
                    tf.summary.image('images', summary_image, step=step)

            # print every self.print_steps
            if step % self.print_step == 0:
                elapsed = time.time() - t_start
                print(self.log_template.format(step, elapsed, d_loss.numpy(), g_loss.numpy(),
                                               d_gan_loss.numpy(), g_gan_loss.numpy(),
                                               r1_penalty.numpy(), pl_penalty.numpy()))

                # reset timer
                t_start = time.time()

            # check exit status
            if step >= self.max_steps:
                break

        # save last checkpoint
        step = self.g_optimizer.iterations.numpy()
        self.manager.save(checkpoint_number=step)
        return

    def gen_samples(self, inputs):
        test_z, test_labels = inputs

        # run networks
        fake_images_05 = self.g_clone([test_z, test_labels], truncation_psi=0.5, training=False)
        fake_images_07 = self.g_clone([test_z, test_labels], truncation_psi=0.7, training=False)

        # merge on batch dimension: [n_samples, 3, out_res, 2 * out_res]
        final_image = tf.concat([fake_images_05, fake_images_07], axis=2)
        return final_image

    @staticmethod
    def convert_per_replica_image(nchw_per_replica_images, strategy):
        as_tensor = tf.concat(strategy.experimental_local_results(nchw_per_replica_images), axis=0)
        as_tensor = tf.transpose(as_tensor, perm=[0, 2, 3, 1])
        as_tensor = (tf.clip_by_value(as_tensor, -1.0, 1.0) + 1.0) * 127.5
        as_tensor = tf.cast(as_tensor, tf.uint8)
        return as_tensor        
#
#
#   FUNS
#
#
def filter_resolutions_featuremaps(resolutions, featuremaps, res):
    index = resolutions.index(res)
    filtered_resolutions = resolutions[:index + 1]
    filtered_featuremaps = featuremaps[:index + 1]
    return filtered_resolutions, filtered_featuremaps
#
#
#    -> https://github.com/moono/stylegan2-tf-2.x/tree/master/stylegan2/dataset_ffhq.py <-
#
def parse_tfrecord_tf(record):
    # n_samples = 70000
    features = tf.io.parse_single_example(record, features={
        'shape': tf.io.FixedLenFeature([3], tf.dtypes.int64),
        'data': tf.io.FixedLenFeature([], tf.dtypes.string)
    })

    # [0 ~ 255] uint8
    images = tf.io.decode_raw(features['data'], tf.dtypes.uint8)
    images = tf.reshape(images, features['shape'])

    # [0.0 ~ 255.0] float32
    images = tf.cast(images, tf.dtypes.float32)
    return images
#
def folder_to_dataset(tfrecord_base_dir, 
        res, buffer_size, batch_size, epochs=None, prefix='tfrecord'):

    fn_index = int(np.log2(res))
    recordidx = str(fn_index).zfill(2)
    tfrecord_fn = os.path.join(tfrecord_base_dir, f'{prefix}-r{recordidx}.tfrecords')

    with tf.device('/cpu:0'):
        dataset = tf.data.TFRecordDataset(tfrecord_fn)
        dataset = dataset.map(map_func=parse_tfrecord_tf, num_parallel_calls=8)
        dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.repeat(epochs)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset
#
#
#   FUNS l4rz
#
#
import multiprocessing
#
#
def resize_and_convert(img, size, resample, quality=100):
    img = trans_fn.resize(img, size, resample)
    img = trans_fn.center_crop(img, size)
    buffer = io.BytesIO()
    img.save(buffer, format="jpeg", quality=quality)
    val = buffer.getvalue()

    return val
#
#
def resize_multiple(img, sizes=(128, 256, 512, 1024), resample=Image.LANCZOS, quality=100):
    imgs = []

    for size in sizes:
        imgs.append(resize_and_convert(img, size, resample, quality))

    return imgs
#
#
def resize_worker(img_file, sizes, resample):
    i, file = img_file
    try:
        img = Image.open(file)
        img = img.convert("RGB")
    except:
        print(file, "truncated")
    out = resize_multiple(img, sizes=sizes, resample=resample)

    return i, out
#
# 
def prepare(env, dataset, n_worker, sizes=(128, 256, 512, 1024), resample=Image.LANCZOS):
    resize_fn = partial(resize_worker, sizes=sizes, resample=resample)

    files = sorted(dataset.imgs, key=lambda x: x[0])
    files = [(i, file) for i, (file, label) in enumerate(files)]
    total = 0

    with multiprocessing.Pool(n_worker) as pool:
        for i, imgs in tqdm(pool.imap_unordered(resize_fn, files)):
            for size, img in zip(sizes, imgs):
                key = f"{size}-{str(i).zfill(5)}".encode("utf-8")

                with env.begin(write=True) as txn:
                    txn.put(key, img)

            total += 1

        with env.begin(write=True) as txn:
            txn.put("length".encode("utf-8"), str(total).encode("utf-8"))
#
#
#   https://github.com/NVlabs/stylegan2/blob/master/pretrained_networks.py
#
def show_gen():
    print(f'|===> nngen: show generator model and layers \n \
            cwd: {os.getcwd()} \n \
    ')

    batch_size = 4
    g_params_with_label = {
        'z_dim': 512,
        'w_dim': 512,
        'labels_dim': 0,
        'n_mapping': 8,
        'resolutions': [4, 8, 16, 32, 64, 128, 256, 512, 1024],
        'featuremaps': [512, 512, 512, 512, 512, 256, 128, 64, 32],
        'w_ema_decay': 0.995,
        'style_mixing_prob': 0.9,
        'truncation_psi': 0.5,
        'truncation_cutoff': None,
    }

    test_z = np.ones((batch_size, g_params_with_label['z_dim']), dtype=np.float32)
    test_y = np.ones((batch_size, g_params_with_label['labels_dim']), dtype=np.float32)

    generator = Generator(g_params_with_label)
    fake_images1, _ = generator([test_z, test_y], training=True)
    fake_images2, _ = generator([test_z, test_y], training=False)
    generator.summary()

    print(f'|...> fake_images1.shape: {fake_images1.shape}')

    for v in generator.variables:
        print(f'|...> generator.variables: {v.name}: {v.shape}')
#
#
def show_disc():

    print(f'|===> nndisr: : show discriminator model and layers \n \
            cwd: {os.getcwd()}  \n \
    ')

    batch_size = 4
    d_params_with_label = {
        'labels_dim': 0,
        'resolutions': [4, 8, 16, 32, 64, 128, 256, 512, 1024],
        'featuremaps': [512, 512, 512, 512, 512, 256, 128, 64, 32],
    }

    input_res = d_params_with_label['resolutions'][-1]
    test_images = np.ones((batch_size, 3, input_res, input_res), dtype=np.float32)
    test_labels = np.ones((batch_size, d_params_with_label['labels_dim']), dtype=np.float32)

    discriminator = Discriminator(d_params_with_label)
    scores1 = discriminator([test_images, test_labels], training=True)
    scores2 = discriminator([test_images, test_labels], training=False)
    discriminator.summary()

    print(f'|...> scores1.shape: {scores1.shape}')
    print(f'|...> scores2.shape: {scores2.shape}')

    print()
    for v in discriminator.variables:
        print(f'|...> discriminator.variables: {v.name}: {v.shape}')
#
# https://github.com/moono/stylegan2-tf-2.x/dataset_ffhq.py
# https://github.com/moono/stylegan2-tf-2.x/blob/master/dataset_ffhq.py
# n_samples = 70000
def parse_tfrecord_tf(record):
    features = tf.io.parse_single_example(record, features={
        'shape': tf.io.FixedLenFeature([3], tf.dtypes.int64),
        'data': tf.io.FixedLenFeature([], tf.dtypes.string)
    })

    # [0 ~ 255] uint8 -> [-1.0 ~ 1.0] float32
    image = tf.io.decode_raw(features['data'], tf.dtypes.uint8)
    image = tf.reshape(image, features['shape'])
    image = tf.transpose(image, perm=[1, 2, 0])
    image = tf.image.random_flip_left_right(image)
    image = tf.cast(image, tf.dtypes.float32)
    image = image / 127.5 - 1.0
    image = tf.transpose(image, perm=[2, 0, 1])
    return image
#
#
# https://github.com/moono/stylegan2-tf-2.x/blob/master/load_models.py
def load_generator(g_params=None, is_g_clone=False, ckpt_dir=None, custom_cuda=True):
    # _e_
    #if custom_cuda:
    #    from stylegan2.generator import Generator
    #else:
    #    from stylegan2_ref.generator import Generator

    if g_params is None:
        g_params = {
            'z_dim': 512,
            'w_dim': 512,
            'labels_dim': 0,
            'n_mapping': 8,
            'resolutions': [4, 8, 16, 32, 64, 128, 256, 512, 1024],
            'featuremaps': [512, 512, 512, 512, 512, 256, 128, 64, 32],
        }

    test_latent = tf.ones((1, g_params['z_dim']), dtype=tf.float32)
    test_labels = tf.ones((1, g_params['labels_dim']), dtype=tf.float32)

    # build generator model
    generator = Generator(g_params)
    _ = generator([test_latent, test_labels])

    if ckpt_dir is not None:
        if is_g_clone:
            ckpt = tf.train.Checkpoint(g_clone=generator)
        else:
            ckpt = tf.train.Checkpoint(generator=generator)
        manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=1)
        ckpt.restore(manager.latest_checkpoint).expect_partial()
        if manager.latest_checkpoint:
            print(f'Generator restored from {manager.latest_checkpoint}')
    return generator
#
#
def load_discriminator(d_params=None, ckpt_dir=None, custom_cuda=True):
    # _e_
    #if custom_cuda:
    #    from stylegan2.discriminator import Discriminator
    #else:
    #    from stylegan2_ref.discriminator import Discriminator

    if d_params is None:
        d_params = {
            'labels_dim': 0,
            'resolutions': [4, 8, 16, 32, 64, 128, 256, 512, 1024],
            'featuremaps': [512, 512, 512, 512, 512, 256, 128, 64, 32],
        }

    res = d_params['resolutions'][-1]
    test_images = tf.ones((1, 3, res, res), dtype=tf.float32)
    test_labels = tf.ones((1, d_params['labels_dim']), dtype=tf.float32)

    # build discriminator model
    discriminator = Discriminator(d_params)
    _ = discriminator([test_images, test_labels])

    if ckpt_dir is not None:
        ckpt = tf.train.Checkpoint(discriminator=discriminator)
        manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=1)
        ckpt.restore(manager.latest_checkpoint).expect_partial()
        if manager.latest_checkpoint:
            print('Discriminator restored from {}'.format(manager.latest_checkpoint))
    return discriminator
#
#https://github.com/moono/stylegan2-tf-2.x/blob/master/inference_from_official_weights.py
#
#
def handle_mapping(w_name):
    def extract_info(name):
        splitted = name.split('/')
        index = splitted.index('g_mapping')
        indicator = splitted[index + 1]
        val = indicator.split('_')[-1]
        return val

    level = extract_info(w_name)
    if 'w' in w_name:
        official_var_name = 'G_mapping_1/Dense{}/weight'.format(level)
    else:
        official_var_name = 'G_mapping_1/Dense{}/bias'.format(level)
    return official_var_name
#
#
def handle_synthesis(w_name):
    def extract_info(name):
        splitted = name.split('/')
        index = splitted.index('g_synthesis')
        indicator1 = splitted[index + 1]
        indicator2 = splitted[index + 2]
        r = indicator1.split('x')[1]
        d = indicator2
        return r, d

    def to_rgb_layer(name, r):
        if 'conv/w:0' in name:
            o_name = 'G_synthesis_1/{}x{}/ToRGB/weight'.format(r, r)
        elif 'mod_dense/w:0' in name:
            o_name = 'G_synthesis_1/{}x{}/ToRGB/mod_weight'.format(r, r)
        elif 'mod_bias/b:0' in name:
            o_name = 'G_synthesis_1/{}x{}/ToRGB/mod_bias'.format(r, r)
        else:   # if 'bias/b:0' in name:
            o_name = 'G_synthesis_1/{}x{}/ToRGB/bias'.format(r, r)
        return o_name

    def handle_block_layer(name, r):
        if 'conv_0/w:0' in name:
            o_name = 'G_synthesis_1/{}x{}/Conv0_up/weight'.format(r, r)
        elif 'conv_0/mod_dense/w:0' in name:
            o_name = 'G_synthesis_1/{}x{}/Conv0_up/mod_weight'.format(r, r)
        elif 'conv_0/mod_bias/b:0' in name:
            o_name = 'G_synthesis_1/{}x{}/Conv0_up/mod_bias'.format(r, r)
        elif 'noise_0/w:0' in name:
            o_name = 'G_synthesis_1/{}x{}/Conv0_up/noise_strength'.format(r, r)
        elif 'bias_0/b:0' in name:
            o_name = 'G_synthesis_1/{}x{}/Conv0_up/bias'.format(r, r)
        elif 'conv_1/w:0' in name:
            o_name = 'G_synthesis_1/{}x{}/Conv1/weight'.format(r, r)
        elif 'conv_1/mod_dense/w:0' in name:
            o_name = 'G_synthesis_1/{}x{}/Conv1/mod_weight'.format(r, r)
        elif 'conv_1/mod_bias/b:0' in name:
            o_name = 'G_synthesis_1/{}x{}/Conv1/mod_bias'.format(r, r)
        elif 'noise_1/w:0' in name:
            o_name = 'G_synthesis_1/{}x{}/Conv1/noise_strength'.format(r, r)
        else:   # if 'bias_1/b:0' in name:
            o_name = 'G_synthesis_1/{}x{}/Conv1/bias'.format(r, r)
        return o_name

    def handle_const_layer(name):
        if 'const:0' in name:
            o_name = 'G_synthesis_1/4x4/Const/const'
        elif 'conv/w:0' in name:
            o_name = 'G_synthesis_1/4x4/Conv/weight'
        elif 'mod_dense/w:0' in name:
            o_name = 'G_synthesis_1/4x4/Conv/mod_weight'
        elif 'mod_bias/b:0' in name:
            o_name = 'G_synthesis_1/4x4/Conv/mod_bias'
        elif 'noise/w:0' in name:
            o_name = 'G_synthesis_1/4x4/Conv/noise_strength'
        else:
            o_name = 'G_synthesis_1/4x4/Conv/bias'
        return o_name

    res, divider = extract_info(w_name)
    if divider == 'ToRGB':
        official_var_name = to_rgb_layer(w_name, res)
    elif divider == 'block':
        official_var_name = handle_block_layer(w_name, res)
    else:   # const
        official_var_name = handle_const_layer(w_name)
    return official_var_name
#
#
def variable_name_mapper(g):
    name_mapper = dict()
    for w in g.weights:
        w_name, w_shape = w.name, w.shape

        # mapping layer
        if 'g_mapping' in w_name:
            official_var_name = handle_mapping(w_name)
        elif 'g_synthesis' in w_name:
            official_var_name = handle_synthesis(w_name)
        else:
            official_var_name = 'Gs/dlatent_avg'
            pass    # w_avg

        name_mapper[official_var_name] = w
    return name_mapper
#
#
def check_shape(name_mapper, official_vars):
    for official_name, v in name_mapper.items():
        official_shape = [s for n, s in official_vars if n == official_name][0]

        if official_shape == v.shape:
            print('{}: shape matches'.format(official_name))
        else:
            # print(f'Official: {official_name} -> {official_shape}')
            # print(f'Current: {v.name} -> {v.shape}')
            raise ValueError('{}: wrong shape'.format(official_name))
    return
#
#
def test_generator(ckpt_dir, use_custom_cuda):
    g_clone = load_generator(g_params=None, is_g_clone=True, ckpt_dir=ckpt_dir, custom_cuda=use_custom_cuda)

    # test
    seed = 6600
    rnd = np.random.RandomState(seed)
    latents = rnd.randn(1, g_clone.z_dim)
    labels = rnd.randn(1, g_clone.labels_dim)
    latents = latents.astype(np.float32)
    labels = labels.astype(np.float32)
    image_out = g_clone([latents, labels], training=False, truncation_psi=0.5)
    image_out = onmoono.postprocess_images(image_out)
    image_out = image_out.numpy()

    out_fn = f'seed{seed}-restored-cuda.png' if use_custom_cuda else f'seed{seed}-restored-ref.png'
    Image.fromarray(image_out[0], 'RGB').save(out_fn)
    return
#
#
def inference_from_each_other(ckpt_dir_base, ckpt_from='cuda', inference_from='ref'):
    assert ckpt_from != inference_from

    # set proper path and variables
    use_custom_cuda = True if inference_from == 'cuda' else False
    ckpt_dir = os.path.join(ckpt_dir_base, 'cuda') if ckpt_from == 'cuda' else os.path.join(ckpt_dir_base, 'ref')

    # create generator
    g_clone = load_generator(g_params=None, is_g_clone=True, ckpt_dir=ckpt_dir, custom_cuda=use_custom_cuda)

    # test
    seed = 6600
    rnd = np.random.RandomState(seed)
    latents = rnd.randn(1, g_clone.z_dim)
    labels = rnd.randn(1, g_clone.labels_dim)
    latents = latents.astype(np.float32)
    labels = labels.astype(np.float32)
    image_out = g_clone([latents, labels], training=False, truncation_psi=0.5)
    image_out = onmoono.postprocess_images(image_out)
    image_out = image_out.numpy()

    out_fn = f'seed{seed}-restored-from-{ckpt_from}-to-{inference_from}.png'
    Image.fromarray(image_out[0], 'RGB').save(out_fn)
    return
#
#save_path = saver.save(tf.get_default_session(), "./model.ckpt")
#
#
#
#   CMDS
#
#
#https://colab.research.google.com/drive/1s2XPNMwf6HDhrJ1FMwlW1jl-eQ2-_tlk?usp=sharing#scrollTo=7YFk46FLM9qo
# Network blending in StyleGAN
#https://colab.research.google.com/drive/1tputbmA9EaXs9HL9iO21g7xN7jz_Xrko?usp=sharing#scrollTo=2fCG-AZ69ttu
#https://twitter.com/Buntworthy/status/1295445259971899393

    #imgset = datasets.ImageFolder(args.data_dir)

    #with lmdb.open(args.out, map_size=1024 ** 4, readahead=False) as env:
    #    prepare(env, imgset, args.n_worker, sizes=sizes, resample=resample)

    #!wget https://upload.wikimedia.org/wikipedia/commons/6/6d/Shinz%C5%8D_Abe_Official.jpg -O raw/example.jpg

    # Convert weight from official checkpoints

    ## use my copy of the blended model to save Doron's download bandwidth
    ## get the original here https://mega.nz/folder/OtllzJwa#C947mCCdEfMCRTWnDcs4qw
    #blended_url = "https://drive.google.com/uc?id=1H73TfV5gQ9ot7slSed_l-lim9X7pMRiU" 
    #ffhq_url = "http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-ffhq-config-f.pkl"

    #_, _, Gs_blended = pretrained_networks.load_networks(blended_url)
    #_, _, Gs = pretrained_networks.load_networks(ffhq_url)
    #!python align_images.py raw aligned
    #!python project_images.py --num-steps 500 aligned generated
    #import numpy as np
    #from PIL import Image
    #import dnnlib
    #import dnnlib.tflib as tflib
    #from pathlib import Path

    #latent_dir = Path("generated")
    #latents = latent_dir.glob("*.npy")
    #for latent_file in latents:
    #  latent = np.load(latent_file)
    #  latent = np.expand_dims(latent,axis=0)
    #  synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=False), minibatch_size=8)
    #  images = Gs_blended.components.synthesis.run(latent, randomize_noise=False, **synthesis_kwargs)
    #  Image.fromarray(images.transpose((0,2,3,1))[0], 'RGB').save(latent_file.parent / (f"{latent_file.stem}-toon.jpg"))

    #from IPython.display import Image 
    #embedded = Image(filename="generated/example_01.png", width=256)
    #display(embedded)
    #tooned = Image(filename="generated/example_01-toon.jpg", width=256)
    #display(tooned)  

    # Generate samples
    # python generate.py --sample N_FACES --pics N_PICS --ckpt PATH_CHECKPOINT
    # Project images to latent spaces
    # python projector.py --ckpt [CHECKPOINT] --size [GENERATOR_OUTPUT_SIZE] FILE1 FILE2 ...
#  
#
#
# https://github.com/moono/stylegan2-tf-2.x
#   [1] weight modulation / demodulation
#   [2] skip architecture
#   [3] resnet architecture
#   [4] path regularization
#   [5] lazy regularization
#   [6] single GPU training
#   [7] inference from official Generator weights
#
#
#
#   nnrun
#
def nnrun(args, kwargs):
    # https://github.com/moono/stylegan2-tf-2.x
    # run git scrits

    args = onutil.pargs(vars(args))
    xp = getxp(vars(args))
    args = onutil.pargs(xp)
    onutil.ddict(vars(args), 'args')

    print(f'|===> nnrun: {args.MNAME}:{args.PROJECT} \n \
        data: {args.DATASET}) \n \
        code: {args.GITPOD} \n \
    ')#

    if 1: # tree
        print(f'|===> nnrun: tree \n \
            cwd:  {os.getcwd()}, \n \
            args.dataorg_dir: {args.dataorg_dir}, \n \
        ')

    if 1: # git
        print(f'|===> nnrun: git ')
        onutil.get_git(args.AUTHOR, args.GITPOD, args.proj_dir)

    if 1: # config
        print(f'|===> nnrun: config \n \
        ')

    if 1: # run py command
        cmd = f"python inference.py"
        print("cmd %s" %cmd)
        os.system(cmd)
#
#
#   nntrain
#
def nntrain(args, kwargs):
    # https://github.com/moono/stylegan2-tf-2.x/blob/master/dataset_ffhq.py

    if 0:
        args = onutil.pargs(vars(args))
        args.PROJECT = 'ffhq'
        args.DATASET = 'ffhq'
        args.train_res = 256 # 
        xp = getxp(vars(args))
        args = onutil.pargs(xp)
        onutil.ddict(vars(args), 'args')
    elif 1:
        args = onutil.pargs(vars(args))
        args.PROJECT = 'gsfc'
        args.DATASET = 'gsfc'
        args.train_res = 32 # fn_index = int(np.log2(res)) - res=32 => 5 # 256 #         
        xp = getxp(vars(args))
        args = onutil.pargs(xp)
        onutil.ddict(vars(args), 'args')

    print(f'|===> nnrun: {args.MNAME}:{args.PROJECT} \n \
        data: {args.DATASET}) \n \
        code: {args.GITPOD} \n \
    ')

    if 1: # tree
        args.raw_dir = os.path.join(args.proj_dir, "raw")
        args.aligned_dir = os.path.join(args.proj_dir, "aligned")
        args.generated_dir = os.path.join(args.proj_dir, "generated")

        args.tfrecord_dir = os.path.join(args.gdata, f'{args.DATASET}_records')
        
        args.ckpt_dir=os.path.join(args.proj_dir, 'ckpts_dir')
        args.model_base_dir=os.path.join(args.proj_dir, 'models')

        os.makedirs(args.raw_dir, exist_ok=True)
        os.makedirs(args.aligned_dir, exist_ok=True)
        os.makedirs(args.generated_dir, exist_ok=True)

        os.makedirs(args.model_base_dir, exist_ok=True)
        os.makedirs(args.tfrecord_dir, exist_ok=True)        
        os.makedirs(args.data_dir, exist_ok=True)    

    print(f'\n |===> nntrain: tree \n \
        cwd:  {os.getcwd()}, \n \
        args.dataorg_dir: {args.dataorg_dir}, \n \
        args.results_dir: {args.results_dir}, \n \
        args.models_dir:  {args.models_dir}, \n \
        args.ckpt_dir:  {args.ckpt_dir}, \n \
        args.data_dir:    {args.data_dir}, \n \
        args.tfrecord_dir:    {args.tfrecord_dir}, \n \
        args.raw_dir:       {args.raw_dir}, \n \
        args.aligned_dir:   {args.aligned_dir}, \n \
        args.generated_dir: {args.generated_dir}, \n \
    \n ')

    if 1: # git
        print(f'|===> nnrun: git ')
        onutil.get_git(args.AUTHOR, args.GITPOD, args.proj_dir)

    if 1: # config
        args.clip = (256, 256) # images will be clipped to clip (from bigger img)
        args.ps = (256, 256) # imgs will be resized to ps
        args.im_size = 256
        args.qslices = 1 # 10 # number of slices per image
        args.eps = 1.0 # 0.9 
        args.size = "128,256,512,1024"
        args.n_worker = 8
        args.resample = "bilinear"
        args.resample_map = {"lanczos": Image.LANCZOS, "bilinear": Image.BILINEAR}
        args.resample = args.resample_map[args.resample]

        args.n_samples = 70000
        args.epochs = 1

        args.allow_memory_growth = True
        args.debug_split_gpu = False
        args.use_tf_function = True
        args.use_custom_cuda = True

        # check tensorflow version
        cur_tf_ver = ontf.check_tf_version()

        if args.allow_memory_growth:
            ontf.allow_memory_growth()
        if args.debug_split_gpu:
            ontf.split_gpu_for_testing(mem_in_gb=4.5)

        # network params

        args.shuffle_buffer_size = 1000
        args.batch_size = 4
        args.batch_size_per_replica = 4

        print(f'|===> nntrain: config \n \
                args.n_samples: {args.n_samples}, \n \
                args.train_res: {args.train_res}, \n \
                args.batch_size: {args.batch_size}, \n \
                args.epochs: {args.epochs}, \n \
        ')

        train_res = args.train_res

        resolutions = [  4,   8,  16,  32,  64, 128, 256, 512, 1024]
        featuremaps = [512, 512, 512, 512, 512, 256, 128,  64,   32]
        train_resolutions, train_featuremaps = filter_resolutions_featuremaps(resolutions, featuremaps, train_res)
        g_params = {
            'z_dim': 512,
            'w_dim': 512,
            'labels_dim': 0,
            'n_mapping': 8,
            'resolutions': train_resolutions,
            'featuremaps': train_featuremaps,
            # _e_            
            'w_ema_decay': 0.995,
            'style_mixing_prob': 0.9,
            'truncation_psi': 0.5,
            'truncation_cutoff': None,                    
        }
        d_params = {
            'labels_dim': 0,
            'resolutions': train_resolutions,
            'featuremaps': train_featuremaps,
        }

        # training parameters
        training_parameters = {
            # global params
            'cur_tf_ver': cur_tf_ver,
            'use_tf_function': args.use_tf_function,
            'use_custom_cuda': args.use_custom_cuda,

            # **args,
            'model_base_dir': args.model_base_dir,
            'tfrecord_dir': args.tfrecord_dir,
            'shuffle_buffer_size': args.shuffle_buffer_size,

            # network params
            'g_params': g_params,
            'd_params': d_params,

            # training params
            'g_opt': {'learning_rate': 0.002, 'beta1': 0.0, 'beta2': 0.99, 'epsilon': 1e-08, 'reg_interval': 4},
            'd_opt': {'learning_rate': 0.002, 'beta1': 0.0, 'beta2': 0.99, 'epsilon': 1e-08, 'reg_interval': 16},
            'batch_size': args.batch_size,
            'n_total_image': 25000000,
            'n_samples': args.n_samples,
            'lazy_regularization': True,

            #
            'train_res': args.train_res,
        }

        onutil.ddict(training_parameters, 'training_parameters')

    if 1: # org => data (formdata)

        print(f'|====> nndata formed images \n \
            from {args.dataorg_dir} into {args.data_dir} \n \
            args.clip: {args.clip} \n \
            qslices: {args.qslices} \n \
            ps: {args.ps} \n \
            eps: {args.eps} \n \
        ')

        qinfiles = onfile.qfiles(args.dataorg_dir, ['*.jpg', '*.png'])
        qinslices = qinfiles * args.qslices
        qoutfiles = onfile.qfiles(args.data_dir, ['*.jpg', '*.png'])
        if qinslices > qoutfiles:

            if onutil.isempty(args.data_dir):
                print(f"args.data_dir {args.data_dir} not empty !!!")

            imgs = ondata.folder_to_formed_pils(
                args.dataorg_dir, 
                ps = args.ps, 
                qslices = args.qslices, 
                eps = args.eps)
            onfile.pils_to_folder(imgs, args.data_dir)

        else:

            print(f'|... no org to data files')

        print(f"|... nndata formed images \n \
            {onfile.qfiles(args.data_dir)} files in {args.data_dir} \n \
        ")

    if 1: # data => tfrecords

        print(f"|===> nndata generate tfrecords \n \
            from {args.data_dir} into {args.tfrecord_dir} \n \
            {onfile.qfiles(args.tfrecord_dir)} files in {args.tfrecord_dir} \n \
        ")
        if onutil.isempty(args.tfrecord_dir):
            print(f"|...> args.tfrecord_dir {args.tfrecord_dir} not empty. SKIP")
        else:
            onrecord.folder_to_tfrecords(args.data_dir, args.tfrecord_dir)

    if 1: # get dataset from tfrecords

        sizes = [int(s.strip()) for s in args.size.split(",")]
        print(f'|===> probe dataset of image sizes: {", ".join(str(s) for s in sizes)} \n \
             convert images to jpeg and pre-resize it \n \
             args.tfrecord_dir: {args.tfrecord_dir} \n \
        ')

         #dataset = get_ffhq_dataset(args.tfrecord_dir, 
         #  args.train_res, batch_size=global_batch_size, epochs=None)

        #dataset = folder_to_dataset(args.tfrecord_dir, 
        #    args.train_res, args.batch_size, args.epochs)

        args.out_res = g_params['resolutions'][-1]
        dataset = folder_to_dataset(args.tfrecord_dir, 
            args.out_res, args.shuffle_buffer_size, args.batch_size, args.epochs)

    if 0: # probe dataset from tfrecords

        for real_images in dataset.take(args.batch_size):
            print(real_images.shape)  # => (1, 3, 32, 32) ...

        q_imgs_to_show = 4
        print(f'|===> dataset probe with {q_imgs_to_show} imgs')

        for real_images in dataset.take(q_imgs_to_show): # show imgs in tfrecord
            # real_images: [batch_size, 3, res, res] (-1.0 ~ 1.0) float32

            images = real_images.numpy()
            images = np.transpose(images, axes=(0, 2, 3, 1))
            images = (images + 1.0) * 127.5
            images = images.astype(np.uint8)
            image = Image.fromarray(images[0])
            image.show()

    if 0: # build generator model
        print(f'|===> nnrun build generator model')
        generator = Generator(g_params)
        _ = generator([test_latent, test_labels])

        if args.ckpt_dir is not None:
            print(f'|...> Generator get checkpoint from {args.ckpt_dir}')
            is_g_clone = 1
            if is_g_clone:
                ckpt = tf.train.Checkpoint(g_clone=generator)
            else:
                ckpt = tf.train.Checkpoint(generator=generator)
            manager = tf.train.CheckpointManager(ckpt, args.ckpt_dir, max_to_keep=1)
            ckpt.restore(manager.latest_checkpoint).expect_partial()
            if manager.latest_checkpoint:
                print(f'|...> Generator restored from {manager.latest_checkpoint}')

    if 0: # trainer
        pass

    elif 1: # train strat
        print(f'|===> trainer')

        # prepare distribute strategy
        strategy = tf.distribute.MirroredStrategy()
        global_batch_size = args.batch_size_per_replica * strategy.num_replicas_in_sync

        print(f'|...> global_batch_size {global_batch_size}')   # 4

        with strategy.scope():
            # distribute dataset
            dist_dataset = strategy.experimental_distribute_dataset(dataset)

            name=f'stylegan2-{args.PROJECT}-{args.train_res}x{args.train_res}'
            trainer = Trainer(training_parameters, name=name)
            trainer.train(dist_dataset, strategy)
    
    elif 0: # train prestat
        print(f'|===> train')


        name=f'stylegan2-{args.PROJECT}-{args.train_res}x{args.train_res}'
        trainer = StyleGAN2(training_parameters, name=name)
        trainer.train(dataset)

    #   https://github.com/eladrich/pixel2style2pixel
    if 0: # inference - apply the trained model on a set of images
        args.exp_dir=args.results_dir    # path to experiment
        args.checkpoint_path=os.path.join(args.models_dir, 'thismodel.h5')
        args.data_path=args.data_dir
        args.test_batch_size=4
        args.test_workers=4
        args.couple_outputs=1

    if 0: # style_mixing - segmentation-to-image experiment
        args.exp_dir=args.results_dir    # path to experiment
        args.checkpoint_path=os.path.join(args.models_dir, 'thismodel.h5')
        args.data_path=args.data_dir
        args.test_batch_size=4
        args.test_workers=4
        args.n_images=25
        args.n_outputs_to_generate=5
        args.latent_mask=8,9,10,11,12,13,14,15,16,17

#
#
#
#   MAIN
#
#
def main():

    parser = argparse.ArgumentParser(description='Run "python %(prog)s <subcommand> --help" for subcommand help.')

    onutil.dodrive()
    ap = getap()
    for p in ap:
        cls = type(ap[p])
        parser.add_argument('--'+p, type=cls, default=ap[p])

    cmds = [key for key in globals() if key.startswith("nn")]
    primecmd = ap["primecmd"]
        
    # ---------------------------------------------------------------
    #   add subparsers
    #
    subparsers = parser.add_subparsers(help='subcommands', dest='command') # command - subparser
    for cmd in cmds:
        subparser = subparsers.add_parser(cmd, help='cmd')  # add subcommands
    
    subparsers_actions = [action for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)] # retrieve subparsers from parser
  
    for subparsers_action in subparsers_actions:  # add common       
        for choice, subparser in subparsers_action.choices.items(): # get all subparsers and print help
            for p in {}:  # subcommand args dict
                cls = type(ap[p])
                subparser.add_argument('--'+p, type=cls, default=ap[p])

    # get args to pass to nn cmds
    if onutil.incolab():
        args = parser.parse_args('') #  defaults as args
    else:
        args = parser.parse_args() #  parse_arguments()

    kwargs = vars(args)
    subcmd = kwargs.pop('command')      

    if subcmd is None:
        print (f"Missing subcommand. set to default {primecmd}")
        subcmd = primecmd
    
    for name in cmds:
        if (subcmd == name):
            print(f'|===> call {name}')
            globals()[name](args, kwargs) # pass args to nn cmd
#
#
#
# python base/base.py nninfo
if __name__ == "__main__":
    print("|===>", __name__)
    main()
