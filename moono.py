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
#   FUNS
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
#
#   CONTEXT
#
#
#   *******************
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
#
#    -> https://github.com/moono/stylegan2-tf-2.x/tree/master/stylegan2/losses.py <-
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
    h, w = fake_images.shape[2], fake_images.shape[3]

    # Compute |J*y|.
    with tf.GradientTape() as pl_tape:
        pl_tape.watch(w_broadcasted)
        pl_noise = tf.random.normal(tf.shape(fake_images), mean=0.0, stddev=1.0, dtype=tf.float32) / np.sqrt(h * w)

    pl_grads = pl_tape.gradient(tf.reduce_sum(fake_images * pl_noise), w_broadcasted)
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
    def call(self, inputs, truncation_cutoff=None, truncation_psi=1.0, training=None, mask=None):
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
class Trainer(object):
    def __init__(self, t_params, name):
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

        # grab dataset
        print('|... Trainer setting datasets')
        self.dataset = get_ffhq_dataset(self.tfrecord_dir, self.out_res, self.shuffle_buffer_size,
                                        self.batch_size, epochs=None)

        # create models
        print('|... Trainer  Create models')
        self.generator = Generator(self.g_params)
        self.discriminator = Discriminator(self.d_params)
        self.d_optimizer = tf.keras.optimizers.Adam(self.d_opt['learning_rate'],
                                                    beta_1=self.d_opt['beta1'],
                                                    beta_2=self.d_opt['beta2'],
                                                    epsilon=self.d_opt['epsilon'])
        self.g_optimizer = tf.keras.optimizers.Adam(self.g_opt['learning_rate'],
                                                    beta_1=self.g_opt['beta1'],
                                                    beta_2=self.g_opt['beta2'],
                                                    epsilon=self.g_opt['epsilon'])
        self.g_clone = Generator(self.g_params)

        # finalize model (build)
        test_latent = np.ones((1, self.g_params['z_dim']), dtype=np.float32)
        test_labels = np.ones((1, self.g_params['labels_dim']), dtype=np.float32)
        test_images = np.ones((1, 3, self.out_res, self.out_res), dtype=np.float32)
        _, __ = self.generator([test_latent, test_labels], training=False)
        _ = self.discriminator([test_images, test_labels], training=False)
        _, __ = self.g_clone([test_latent, test_labels], training=False)
        print('Copying g_clone')
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
            print('Restored from {}'.format(self.manager.latest_checkpoint))

            # check if already trained in this resolution
            restored_step = self.g_optimizer.iterations.numpy()
            if restored_step >= self.max_steps:
                print('Already reached max steps {}/{}'.format(restored_step, self.max_steps))
                self.reached_max_steps = True
                return
        else:
            print('Not restoring from saved checkpoint')

    def set_optimizer_params(self, params):
        if self.lazy_regularization:
            mb_ratio = params['reg_interval'] / (params['reg_interval'] + 1)
            params['learning_rate'] = params['learning_rate'] * mb_ratio
            params['beta1'] = params['beta1'] ** mb_ratio
            params['beta2'] = params['beta2'] ** mb_ratio
        return params

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

    # @tf.function
    def g_reg_train_step(self, z, labels):
        with tf.GradientTape() as g_tape:
            # forward pass
            fake_images, _ = self.generator([z, labels], training=True)
            fake_scores = self.discriminator([fake_images, labels], training=True)

            # gan loss
            g_loss = tf.math.softplus(-fake_scores)

            # path length regularization
            # Compute |J*y|.
            with tf.GradientTape() as pl_tape:
                fake_images, w_broadcasted = self.generator([z, labels], training=True)

                pl_noise = tf.random.normal(tf.shape(fake_images), mean=0.0, stddev=1.0, dtype=tf.float32) * self.pl_denorm
                pl_noise_added = tf.reduce_sum(fake_images * pl_noise)
            
            print("|... g_reg_train_step ***************************************>")
            #  pl_noise_added:  tf.Tensor(2.973158, shape=(), dtype=float32)
            #  w_broadcasted:   tf.Tensor([[[-0.25851128  0.28737155  1.8203238  ... -0.5302262  -0.00633283
            pl_grads = pl_tape.gradient(pl_noise_added, w_broadcasted) # None
            print(f"|... g_reg_train_step: pl_grads {pl_grads}")

            # ***************************************************************************
            pl_lengths = 0.0                                # _e_ ***********************
            if pl_grads:
                pl_lengths = tf.math.sqrt(tf.reduce_mean(tf.reduce_sum(tf.math.square(pl_grads), axis=2), axis=1))
            # ***************************************************************************

            # Track exponential moving average of |J*y|.
            pl_mean_val = self.pl_mean + self.pl_decay * (tf.reduce_mean(pl_lengths) - self.pl_mean)
            self.pl_mean.assign(pl_mean_val)

            # Calculate (|J*y|-a)^2.
            pl_penalty = tf.square(pl_lengths - self.pl_mean)

            # pl_penalty = pl_reg(fake_images, w_broadcasted, self.pl_mean)
            # print(f"*** pl_penalty {pl_penalty}")

            # compute
            _pl_reg = pl_penalty * self.pl_weight
            print(f"*** _pl_reg {_pl_reg}")

            print("|... g_reg_train_step  <***********************************")

            # combine
            g_loss += _pl_reg
            g_loss = tf.reduce_mean(g_loss)

        g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        return g_loss, tf.reduce_mean(_pl_reg)

    def train(self):
        if self.reached_max_steps:
            return

        # start actual training
        print('Start Training')

        # setup tensorboards
        train_summary_writer = tf.summary.create_file_writer(self.ckpt_dir)

        # loss metrics
        metric_g_loss = tf.keras.metrics.Mean('g_loss', dtype=tf.float32)
        metric_d_loss = tf.keras.metrics.Mean('d_loss', dtype=tf.float32)
        metric_r1_reg = tf.keras.metrics.Mean('r1_reg', dtype=tf.float32)
        metric_pl_reg = tf.keras.metrics.Mean('pl_reg', dtype=tf.float32)

        # start training
        print('max_steps: {}'.format(self.max_steps))
        losses = {'g_loss': 0.0, 'd_loss': 0.0, 'r1_reg': 0.0, 'pl_reg': 0.0}
        t_start = time.time()

        for real_images in self.dataset:
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
                g_loss, pl_reg = self.g_reg_train_step(z, labels)

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
def get_ffhq_dataset(tfrecord_base_dir, res, buffer_size, batch_size, epochs=None):
    fn_index = int(np.log2(res))
    tfrecord_fn = os.path.join(tfrecord_base_dir, 'ffhq-r{:02d}.tfrecords'.format(fn_index))

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
#   CMDS
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
def nnutils(args, kwargs):

    args = onutil.pargs(vars(args))
    onutil.ddict(vars(args), 'args')

    print(f'|===> nnutils:  \n \
            cwd: {os.getcwd()} \n \
    ')

    if 1: # git
        onutil.get_git(args.AUTHOR, args.GITPOD, args.proj_dir)

    if 1: # rgb to nba
        print(f'|===> adjust_dynamic_range')
        out_dtype = tf.dtypes.float32
        range_in = (0.0, 255.0)
        range_out = (-1.0, 1.0)
        # images = np.ones((2, 2), dtype=np.float32) * 255.0
        images1 = np.ones((2, 2), dtype=np.float32) * 0.0
        images2 = np.ones((2, 2), dtype=np.float32) * 127.5
        images3 = np.ones((2, 2), dtype=np.float32) * 255.0
        adjusted1 = onmoono.adjust_dynamic_range(images1, range_in, range_out, out_dtype)
        adjusted2 = onmoono.adjust_dynamic_range(images2, range_in, range_out, out_dtype)
        adjusted3 = onmoono.adjust_dynamic_range(images3, range_in, range_out, out_dtype)
        print(f'|.... adjust_dynamic_range 1: {adjusted1}')
        print(f'|.... adjust_dynamic_range 2: {adjusted2}')
        print(f'|.... adjust_dynamic_range 3: {adjusted3}')
    
    if 1: # nba to rgb
        out_dtype = tf.dtypes.uint8
        range_in = (-1.0, 1.0)
        range_out = (0.0, 255.0)
        images1 = np.ones((2, 2), dtype=np.float32) * -1.0
        images2 = np.ones((2, 2), dtype=np.float32) * 0.0
        images3 = np.ones((2, 2), dtype=np.float32) * 1.0
        adjusted1 = onmoono.adjust_dynamic_range(images1, range_in, range_out, out_dtype)
        adjusted2 = onmoono.adjust_dynamic_range(images2, range_in, range_out, out_dtype)
        adjusted3 = onmoono.adjust_dynamic_range(images3, range_in, range_out, out_dtype)
        print(f'|.... adjust_dynamic_range 1: {adjusted1}')
        print(f'|.... adjust_dynamic_range 2: {adjusted2}')
        print(f'|.... adjust_dynamic_range 3: {adjusted3}')

    if 1: # merge
        batch_size = 8
        res = 128
        fake_images = np.ones(shape=(batch_size, res, res, 3), dtype=np.uint8)
        canvas = onmoono.merge_batch_images(fake_images, res, rows=4, cols=2)
        print(f'|.... canvas shape {np.shape(canvas)}')
#
def nngen(args, kwargs):

    args = onutil.pargs(vars(args))
    onutil.ddict(vars(args), 'args')

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

    print(f'|... fake_images1.shape: {fake_images1.shape}')

    for v in generator.variables:
        print(f'|... generator.variables: {v.name}: {v.shape}')
#
def nndisc(args, kwargs):

    args = onutil.pargs(vars(args))
    onutil.ddict(vars(args), 'args')

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

    print(f'|... scores1.shape: {scores1.shape}')
    print(f'|... scores2.shape: {scores2.shape}')

    print()
    for v in discriminator.variables:
        print(f'|... discriminator.variables: {v.name}: {v.shape}')
#
def nntrain(args, kwargs):
    # https://github.com/moono/stylegan2-tf-2.x/blob/master/dataset_ffhq.py

    args = onutil.pargs(vars(args))
    onutil.ddict(vars(args), 'args')

    if 1: # tree
        args.tfrecord_dir = os.path.join(args.gdata, 'ffhq')
        args.model_base_dir=os.path.join(args.proto_dir, 'models')

        os.makedirs(args.model_base_dir, exist_ok=True)

    print(f'|===> nntrain: train \n \
            args.tfrecord_dir: {args.tfrecord_dir}, \n \
            args.model_base_dir: {args.model_base_dir}, \n \
    ')

    if 1: # test tfrecords
        n_samples = 70000
        res = 32
        batch_size = 4
        epochs = 1

        dataset = get_ffhq_dataset(args.tfrecord_dir, res, batch_size, epochs)
        for real_images in dataset.take(batch_size):
            print(real_images.shape)  # => (1, 3, 32, 32) ...

    if 1: # network params
        args.train_res=32
        args.shuffle_buffer_size=1000
        args.batch_size=4

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
            'w_ema_decay': 0.995,
            'style_mixing_prob': 0.9,
        }
        d_params = {
            'labels_dim': 0,
            'resolutions': train_resolutions,
            'featuremaps': train_featuremaps,
        }

        # training parameters
        training_parameters = {
            # global params
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
            'n_samples': 4,
            'lazy_regularization': True,
        }

    onutil.ddict(training_parameters, 'training_parameters')

    if 1: # train
        trainer = Trainer(
            training_parameters, 
            name='stylegan2-ffhq'
        )
        trainer.train()
#
def nninfo(args, kwargs):
	args = onutil.pargs(vars(args))
	onutil.ddict(vars(args), 'args')
	print(f"|===> nninfo: {args.MNAME} {args.AUTHOR} {args.PROJECT} {args.GITPOD} {args.DATASET} \n \
		nnutils: try rgb2nba, nba2rgb, merge batch images  \n \
		nngen: generate image  \n \
		nndisc: discriminate image  \n \
		nnffhq: test tfrecord input funcion \n \
		nntrain: train model with g_params, d_params \n \
	")
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
