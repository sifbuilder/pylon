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
oncuda = Oncuda()
onrosa = Onrosa()
onplot = Onplot()
onformat = Onformat()
onfile = Onfile()
onvid = Onvid()
ondata = Ondata()
onset = Onset()
onrecord = Onrecord()
ontree = Ontree()
onlllyas = Onlllyas()
onmoono = Onmoono()
#
#
#   CONTEXT
#
#
def getap():
    cp = {
        "primecmd": 'nntrain',    
                
        "MNAME": "rosasalberto",      
        "AUTHOR": "rosasalberto",      
        "PROJECT": "rosalberto",      
        "GITPOD": "StyleGAN2-TensorFlow-2.x",      
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

    yp={
    }
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
#
#   NETS
#
#
#
#
#   utils/weights_map
#
# import numpy as np
#
# weights      
file_id = [
    '1afMN3e_6UuTTPDL63WHaA0Fb9EQrZceE', 
    '1Av4p3JNWkmWsJx6s9Hq32JvlNWSW4kgk',
    '1LCyHcKtqNs8NirJeSPq3fKAUZ-MLt_-8', 
    '1pEAeJTK_ZPfkUvV7VFDHq0O4tKYsslkX', 
    '16_QtKS2w9-40uxGZ02h3OtFhnJgNgdi_'
    ]
name = ['ffhq.npy', 'car.npy', 'cat.npy', 'church.npy', 'horse.npy']
weights = {
    "ffhq": '1afMN3e_6UuTTPDL63WHaA0Fb9EQrZceE',
    "car": '1Av4p3JNWkmWsJx6s9Hq32JvlNWSW4kgk',
    "cat": '1LCyHcKtqNs8NirJeSPq3fKAUZ-MLt_-8',
    "church": '1pEAeJTK_ZPfkUvV7VFDHq0O4tKYsslkX',
    "horse": '16_QtKS2w9-40uxGZ02h3OtFhnJgNgdi_',   
}
#
resolutions = [  4,   8,  16,  32,  64, 128, 256, 512, 1024]
featuremaps = [512, 512, 512, 512, 512, 256, 128,  64,   32]
#
available_weights = ['ffhq', 'car', 'cat', 'church', 'horse']
weights_stylegan2_dir = 'weights/'
#
mapping_weights = [ 'Dense0/weight', 'Dense0/bias',
                    'Dense1/weight', 'Dense1/bias',
                    'Dense2/weight', 'Dense2/bias',
                    'Dense3/weight', 'Dense3/bias',
                    'Dense4/weight', 'Dense4/bias',
                    'Dense5/weight', 'Dense5/bias',
                    'Dense6/weight', 'Dense6/bias',
                    'Dense7/weight', 'Dense7/bias']
#
def get_synthesis_name_weights(resolution):
    synthesis_weights = ['4x4/Const/const',
                         '4x4/Conv/noise_strength',
                         '4x4/Conv/bias',
                         '4x4/Conv/mod_bias',
                         '4x4/Conv/mod_weight',
                         '4x4/Conv/weight',
                         '4x4/ToRGB/bias',
                         '4x4/ToRGB/mod_bias',
                         '4x4/ToRGB/mod_weight',
                         '4x4/ToRGB/weight']

    for res in range(3,int(np.log2(resolution)) + 1):
        name = '{}x{}/'.format(2**res, 2**res)
        for up in ['Conv0_up/', 'Conv1/', 'ToRGB/']:
            for var in ['noise_strength', 'bias', 'mod_bias', 'mod_weight', 'weight']:
                if up == 'ToRGB/' and var == 'noise_strength':
                    continue
                synthesis_weights.append(name+up+var)
                
    return synthesis_weights
#
synthesis_weights_1024 = get_synthesis_name_weights(1024)
synthesis_weights_512 = get_synthesis_name_weights(512)
synthesis_weights_256 = get_synthesis_name_weights(256)
#
discriminator_weights_1024 = ['disc_4x4/Conv/bias',
                            'disc_1024x1024/FromRGB/bias',
                            'disc_1024x1024/FromRGB/weight',
                            'disc_1024x1024/Conv0/bias',
                            'disc_1024x1024/Conv1_down/bias',
                            'disc_1024x1024/Conv0/weight',
                            'disc_1024x1024/Conv1_down/weight',
                            'disc_1024x1024/Skip/weight',
                            'disc_512x512/Conv0/bias',
                            'disc_512x512/Conv1_down/bias',
                            'disc_512x512/Conv0/weight',
                            'disc_512x512/Conv1_down/weight',
                            'disc_512x512/Skip/weight',
                            'disc_256x256/Conv0/bias',
                            'disc_256x256/Conv1_down/bias',
                            'disc_256x256/Conv0/weight',
                            'disc_256x256/Conv1_down/weight',
                            'disc_256x256/Skip/weight',
                            'disc_128x128/Conv0/bias',
                            'disc_128x128/Conv1_down/bias',
                            'disc_128x128/Conv0/weight',
                            'disc_128x128/Conv1_down/weight',
                            'disc_128x128/Skip/weight',
                            'disc_64x64/Conv0/bias',
                            'disc_64x64/Conv1_down/bias',
                            'disc_64x64/Conv0/weight',
                            'disc_64x64/Conv1_down/weight',
                            'disc_64x64/Skip/weight',
                            'disc_32x32/Conv0/bias',
                            'disc_32x32/Conv1_down/bias',
                            'disc_32x32/Conv0/weight',
                            'disc_32x32/Conv1_down/weight',
                            'disc_32x32/Skip/weight',
                            'disc_16x16/Conv0/bias',
                            'disc_16x16/Conv1_down/bias',
                            'disc_16x16/Conv0/weight',
                            'disc_16x16/Conv1_down/weight',
                            'disc_16x16/Skip/weight',
                            'disc_8x8/Conv0/bias',
                            'disc_8x8/Conv1_down/bias',
                            'disc_8x8/Conv0/weight',
                            'disc_8x8/Conv1_down/weight',
                            'disc_8x8/Skip/weight',
                            'disc_4x4/Conv/weight',
                            'disc_4x4/Dense0/weight',
                            'disc_4x4/Dense0/bias',
                            'disc_Output/weight',
                            'disc_Output/bias']
#
discriminator_weights_512 = ['disc_4x4/Conv/bias',
                            'disc_512x512/FromRGB/bias',
                            'disc_512x512/FromRGB/weight',
                            'disc_512x512/Conv0/bias',
                            'disc_512x512/Conv1_down/bias',
                            'disc_512x512/Conv0/weight',
                            'disc_512x512/Conv1_down/weight',
                            'disc_512x512/Skip/weight',
                            'disc_256x256/Conv0/bias',
                            'disc_256x256/Conv1_down/bias',
                            'disc_256x256/Conv0/weight',
                            'disc_256x256/Conv1_down/weight',
                            'disc_256x256/Skip/weight',
                            'disc_128x128/Conv0/bias',
                            'disc_128x128/Conv1_down/bias',
                            'disc_128x128/Conv0/weight',
                            'disc_128x128/Conv1_down/weight',
                            'disc_128x128/Skip/weight',
                            'disc_64x64/Conv0/bias',
                            'disc_64x64/Conv1_down/bias',
                            'disc_64x64/Conv0/weight',
                            'disc_64x64/Conv1_down/weight',
                            'disc_64x64/Skip/weight',
                            'disc_32x32/Conv0/bias',
                            'disc_32x32/Conv1_down/bias',
                            'disc_32x32/Conv0/weight',
                            'disc_32x32/Conv1_down/weight',
                            'disc_32x32/Skip/weight',
                            'disc_16x16/Conv0/bias',
                            'disc_16x16/Conv1_down/bias',
                            'disc_16x16/Conv0/weight',
                            'disc_16x16/Conv1_down/weight',
                            'disc_16x16/Skip/weight',
                            'disc_8x8/Conv0/bias',
                            'disc_8x8/Conv1_down/bias',
                            'disc_8x8/Conv0/weight',
                            'disc_8x8/Conv1_down/weight',
                            'disc_8x8/Skip/weight',
                            'disc_4x4/Conv/weight',
                            'disc_4x4/Dense0/weight',
                            'disc_4x4/Dense0/bias',
                            'disc_Output/weight',
                            'disc_Output/bias']
#
discriminator_weights_256 =  ['disc_4x4/Conv/bias',
                            'disc_256x256/FromRGB/bias',
                            'disc_256x256/FromRGB/weight',
                            'disc_256x256/Conv0/bias',
                            'disc_256x256/Conv1_down/bias',
                            'disc_256x256/Conv0/weight',
                            'disc_256x256/Conv1_down/weight',
                            'disc_256x256/Skip/weight',
                            'disc_128x128/Conv0/bias',
                            'disc_128x128/Conv1_down/bias',
                            'disc_128x128/Conv0/weight',
                            'disc_128x128/Conv1_down/weight',
                            'disc_128x128/Skip/weight',
                            'disc_64x64/Conv0/bias',
                            'disc_64x64/Conv1_down/bias',
                            'disc_64x64/Conv0/weight',
                            'disc_64x64/Conv1_down/weight',
                            'disc_64x64/Skip/weight',
                            'disc_32x32/Conv0/bias',
                            'disc_32x32/Conv1_down/bias',
                            'disc_32x32/Conv0/weight',
                            'disc_32x32/Conv1_down/weight',
                            'disc_32x32/Skip/weight',
                            'disc_16x16/Conv0/bias',
                            'disc_16x16/Conv1_down/bias',
                            'disc_16x16/Conv0/weight',
                            'disc_16x16/Conv1_down/weight',
                            'disc_16x16/Skip/weight',
                            'disc_8x8/Conv0/bias',
                            'disc_8x8/Conv1_down/bias',
                            'disc_8x8/Conv0/weight',
                            'disc_8x8/Conv1_down/weight',
                            'disc_8x8/Skip/weight',
                            'disc_4x4/Conv/weight',
                            'disc_4x4/Dense0/weight',
                            'disc_4x4/Dense0/bias',
                            'disc_Output/weight',
                            'disc_Output/bias']
#
synthesis_weights = {
    'ffhq' : synthesis_weights_1024,
    'car' : synthesis_weights_512,
    'cat' : synthesis_weights_256,
    'horse' : synthesis_weights_256,
    'church' : synthesis_weights_256
    }
#
discriminator_weights = {
    'ffhq' : discriminator_weights_1024,
    'car' : discriminator_weights_512,
    'cat' : discriminator_weights_256,
    'horse' : discriminator_weights_256,
    'church' : discriminator_weights_256
    }
#
#
#
#   NETS
#
#
# from utils.utils_stylegan2 import get_weight_initializer_runtime_coef
# from dnnlib.ops.upfirdn_2d import upsample_conv_2d, conv_downsample_2d
#
#   layers/block_layer.py
#
class BlockLayer(tf.keras.layers.Layer):
    """
    StyleGan2 discriminator Block layer
    """
    def __init__(self, res, impl='cuda', gpu=False, **kwargs):
        
        super(BlockLayer, self).__init__(**kwargs)
        
        self.res = res
        self.impl = impl
        self.gpu = gpu
        self.resample_kernel = [1, 3, 3, 1]
        
    def build(self, input_shape):
        
        self.conv2d_0 = Conv2DLayer(fmaps=onrosa.nf(self.res-1), kernel=3, 
                                    impl=self.impl, gpu=self.gpu, name='Conv0')
        
        self.bias_0 = self.add_weight(name='Conv0/bias', shape=(onrosa.nf(self.res-1),), 
                                      initializer=tf.random_normal_initializer(0, 1), trainable=True)
        
        self.conv2d_1_down = Conv2DLayer(fmaps=onrosa.nf(self.res-2), kernel=3, down=True, 
                                         resample_kernel=self.resample_kernel, 
                                         impl=self.impl, gpu=self.gpu, name='Conv1_down')
        
        self.bias_1_down = self.add_weight(name='Conv1_down/bias', shape=(onrosa.nf(self.res-2),), 
                                           initializer=tf.random_normal_initializer(0, 1), trainable=True)
        
        self.conv2d_skip = Conv2DLayer(fmaps=onrosa.nf(self.res-2), kernel=1, down=True, 
                                       resample_kernel=self.resample_kernel, 
                                       impl=self.impl, gpu=self.gpu, name='Skip')
        
    def call(self, x):
        t = x
        
        x = self.conv2d_0(x)
        x += tf.reshape(self.bias_0, [-1 if i == 1 else 1 for i in range(x.shape.rank)])
        x = tf.math.multiply(tf.nn.leaky_relu(x, 0.2), tf.math.sqrt(2.))
        
        x = self.conv2d_1_down(x)
        x += tf.reshape(self.bias_1_down, [-1 if i == 1 else 1 for i in range(x.shape.rank)])
        x = tf.math.multiply(tf.nn.leaky_relu(x, 0.2), tf.math.sqrt(2.))
 
        t = self.conv2d_skip(t)
        x = (x + t) * (1 / np.sqrt(2))
        
        return x
#
#
#   layers/conv_2d_layer.py
#
class Conv2DLayer(tf.keras.layers.Layer):
    """
    StyleGan2 discriminator convolutional layer
    """
    def __init__(self, fmaps, kernel, up=False, down=False, 
                               demodulate=True, resample_kernel=None, gain=1, use_wscale=True, lrmul=1, 
                               impl='cuda', gpu=True, **kwargs):
                
        super(Conv2DLayer, self).__init__(**kwargs)
        
        self.fmaps = fmaps
        self.kernel = kernel
        
        self.up = up
        self.down = down
        self.resample_kernel = resample_kernel
        self.gain = gain
        self.use_wscale = use_wscale
        self.lrmul = lrmul
        self.impl = impl
        self.gpu = gpu
        
    def build(self, input_shape):
        
        self.init_std_w, self.runtime_coef_w = onrosa.get_weight_initializer_runtime_coef(shape=[self.kernel, self.kernel, input_shape[1], self.fmaps],
                                    gain=self.gain, use_wscale=self.use_wscale, lrmul=self.lrmul)
        
        self.weight = self.add_weight(name='weight', shape=(self.kernel,self.kernel, input_shape[1], self.fmaps), 
                                    initializer=tf.random_normal_initializer(0, self.init_std_w), trainable=True)
    
    def call(self, x):
        
        # multiply weight per runtime_coef
        w = tf.math.multiply(self.weight, self.runtime_coef_w)
        
        # Convolution with optional up/downsampling.
        if self.up:
            if self.gpu:
                x = oncuda.upsample_conv_2d(x, tf.cast(w, x.dtype), data_format='NCHW', k=self.resample_kernel, impl=self.impl)
            else:
                x = tf.transpose(x, [0, 2, 3, 1])
                x = oncuda.upsample_conv_2d(x, tf.cast(w, x.dtype), data_format='NHWC', k=self.resample_kernel, impl=self.impl, gpu=False)
                x = tf.transpose(x, [0, 3, 1, 2]) 
        elif self.down:
            if self.gpu:
                x = oncuda.conv_downsample_2d(x, tf.cast(w, x.dtype), data_format='NCHW', k=self.resample_kernel, impl=self.impl)
            else:
                x = tf.transpose(x, [0, 2, 3, 1])
                x = oncuda.conv_downsample_2d(x, tf.cast(w, x.dtype), data_format='NHWC', k=self.resample_kernel, impl=self.impl, gpu=False)
                x = tf.transpose(x, [0, 3, 1, 2]) 
        else:
            if self.gpu:
                x = tf.nn.conv2d(x, tf.cast(w, x.dtype), data_format='NCHW', strides=[1,1,1,1], padding='SAME')
            else:
                x = tf.transpose(x, [0, 2, 3, 1])
                x = tf.nn.conv2d(x, tf.cast(w, x.dtype), data_format='NHWC', strides=[1,1,1,1], padding='SAME')
                x = tf.transpose(x, [0, 3, 1, 2])  
        return x
#
#
#   layers/dense_layer.py
#
class DenseLayer(tf.keras.layers.Layer):
    """
    StyleGan2 Dense layer, including weights multiplication per runtime coef, and bias multiplication per lrmul
    """
    def __init__(self, fmaps, lrmul=1, **kwargs):
        
        super(DenseLayer, self).__init__(**kwargs)
        
        self.fmaps = fmaps
        self.lrmul = lrmul
        
    def build(self, input_shape):
        
        init_std, self.runtime_coef = onrosa.get_weight_initializer_runtime_coef(shape=[input_shape[1], self.fmaps], 
                                                                              gain=1, use_wscale=True, lrmul=self.lrmul)
        
        self.dense_weight = self.add_weight(name='weight', shape=(input_shape[1],self.fmaps),
                                            initializer=tf.random_normal_initializer(0,init_std), trainable=True)
        self.dense_bias = self.add_weight(name='bias', shape=(self.fmaps,),
                                          initializer=tf.random_normal_initializer(0,init_std), trainable=True)
        
    def call(self, x):
        
        x = tf.matmul(x, tf.math.multiply(self.dense_weight, self.runtime_coef)) 
        x += tf.reshape(tf.math.multiply(self.dense_bias, self.lrmul), [-1 if i == 1 else 1 for i in range(x.shape.rank)])
        
        return x
#
#
#   layers/from_rgb_layer.py
#
class FromRgbLayer(tf.keras.layers.Layer):
    """
    StyleGan2 discriminator From RGB layer
    """
    def __init__(self, fmaps, impl='cuda', gpu=True, **kwargs):
        
        super(FromRgbLayer, self).__init__(**kwargs)
        
        self.fmaps = fmaps
        self.impl = impl
        self.gpu = gpu
        
    def build(self, input_shape):
        
        self.conv2d_rgb = Conv2DLayer(fmaps=self.fmaps, kernel=1, impl=self.impl, gpu=self.gpu, name='FromRGB')
        
        self.rgb_bias = self.add_weight(name='FromRGB/bias', shape=(self.fmaps,), 
                                        initializer=tf.random_normal_initializer(0,1), trainable=True)
        
    def call(self, x, y):
        
        t = self.conv2d_rgb(y)

        #add bias and lrelu activation
        t += tf.reshape(self.rgb_bias, [-1 if i == 1 else 1 for i in range(t.shape.rank)])
        t = tf.math.multiply(tf.nn.leaky_relu(t, 0.2), tf.math.sqrt(2.))
        
        return t if x is None else x + t
#
#
#   layers/mini_batch_std_layer.py
#
class MinibatchStdLayer(tf.keras.layers.Layer):
    """
    StyleGan2 discriminator MinibatchStdLayer
    """
    def __init__(self, group_size=4, num_new_features=1, **kwargs):
        
        super(MinibatchStdLayer, self).__init__(**kwargs)
        
        self.group_size = group_size
        self.num_new_features = num_new_features
        
    def call(self, x):
        
        group_size = tf.minimum(self.group_size, x.shape[0])     # Minibatch must be divisible by (or smaller than) group_size.
        s = x.shape                                             # [NCHW]  Input shape.
        y = tf.reshape(x, [group_size, -1, self.num_new_features, s[1]//self.num_new_features, s[2], s[3]])   # [GMncHW] Split minibatch into M groups of size G. Split channels into n channel groups c.
        y = tf.cast(y, tf.float32)                              # [GMncHW] Cast to FP32.
        y -= tf.reduce_mean(y, axis=0, keepdims=True)           # [GMncHW] Subtract mean over group.
        y = tf.reduce_mean(tf.square(y), axis=0)                # [MncHW]  Calc variance over group.
        y = tf.sqrt(y + 1e-8)                                   # [MncHW]  Calc stddev over group.
        y = tf.reduce_mean(y, axis=[2,3,4], keepdims=True)      # [Mn111]  Take average over fmaps and pixels.
        y = tf.reduce_mean(y, axis=[2])                         # [Mn11] Split channels into c channel groups
        y = tf.cast(y, x.dtype)                                 # [Mn11]  Cast back to original data type.
        y = tf.tile(y, [group_size, 1, s[2], s[3]])             # [NnHW]  Replicate over group and pixels.
    
        return tf.concat([x, y], axis=1)
#
#
#   layers/modulated_conv_2d_layer.py
#
class ModulatedConv2DLayer(tf.keras.layers.Layer):
    """
    StyleGan2 generator modulated convolution layer
    """
    def __init__(self, fmaps, kernel, up=False, down=False, 
                               demodulate=True, resample_kernel=None, gain=1, use_wscale=True, lrmul=1, 
                               fused_modconv=True, impl='cuda', gpu=True, **kwargs):
                
        super(ModulatedConv2DLayer, self).__init__(**kwargs)
        
        self.fmaps = fmaps
        self.kernel = kernel
        
        self.up = up
        self.down = down
        self.demodulate = demodulate
        self.resample_kernel = resample_kernel
        self.gain = gain
        self.use_wscale = use_wscale
        self.lrmul = lrmul
        self.fused_modconv = fused_modconv
        self.latent_size = 512
        self.impl = impl
        self.gpu = gpu
        
    def build(self, input_shape):
        
        self.init_std_w, self.runtime_coef_w = onrosa.get_weight_initializer_runtime_coef(shape=[self.kernel, self.kernel, input_shape[1], self.fmaps],
                gain=self.gain, use_wscale=self.use_wscale, lrmul=self.lrmul)
        
        self.init_std_s, self.runtime_coef_s = onrosa.get_weight_initializer_runtime_coef(shape=[self.latent_size, input_shape[1]],
                gain=self.gain, use_wscale=self.use_wscale, lrmul=self.lrmul)
        
        self.mod_bias = self.add_weight(name='mod_bias', shape=(input_shape[1],), 
                initializer=tf.random_normal_initializer(0, self.init_std_s), trainable=True)
        self.mod_weight = self.add_weight(name='mod_weight', shape=(self.latent_size, input_shape[1]), 
                initializer=tf.random_normal_initializer(0, self.init_std_s), trainable=True)
        self.weight = self.add_weight(name='weight', shape=(self.kernel,self.kernel, input_shape[1], self.fmaps), 
                initializer=tf.random_normal_initializer(0, self.init_std_w), trainable=True)
    
    def call(self, x, dlatent_vect):
        
        # multiply weight per runtime_coef
        w = tf.math.multiply(self.weight, self.runtime_coef_w)
        ww = w[np.newaxis] # [BkkIO] Introduce minibatch dimension.
        
        # Modulate.
        s = tf.matmul(dlatent_vect, tf.math.multiply(self.mod_weight, self.runtime_coef_s)) # [BI] Transform incoming W to style.
        s = tf.nn.bias_add(s, tf.math.multiply(self.mod_bias, self.lrmul)) + 1 # [BI] Add bias (initially 1).

        ww = tf.math.multiply(ww, tf.cast(s[:, np.newaxis, np.newaxis, :, np.newaxis], w.dtype)) # [BkkIO] Scale input feature maps.
        
        # Demodulate.
        if self.demodulate:
            d = tf.math.rsqrt(tf.reduce_sum(tf.square(ww), axis=[1,2,3]) + 1e-8) # [BO] Scaling factor.
            ww = tf.math.multiply(ww, d[:, np.newaxis, np.newaxis, np.newaxis, :]) # [BkkIO] Scale output feature maps.
        
        # Reshape/scale input.
        if self.fused_modconv:
            x = tf.reshape(x, [1, -1, x.shape[2], x.shape[3]]) # Fused => reshape minibatch to convolution groups.
            w = tf.reshape(tf.transpose(ww, [1, 2, 3, 0, 4]), [ww.shape[1], ww.shape[2], ww.shape[3], -1])
        else:
            x = tf.math.multiply(x, tf.cast(s[:, :, np.newaxis, np.newaxis], x.dtype)) # [BIhw] Not fused => scale input activations.
        
        # Convolution with optional up/downsampling.
        if self.up:
            if self.gpu:
                x = oncuda.upsample_conv_2d(x, tf.cast(w, x.dtype), data_format='NCHW', k=self.resample_kernel, impl=self.impl)
            else:
                x = tf.transpose(x, [0, 2, 3, 1])
                x = oncuda.upsample_conv_2d(x, tf.cast(w, x.dtype), data_format='NHWC', k=self.resample_kernel, impl=self.impl, gpu=False)
                x = tf.transpose(x, [0, 3, 1, 2]) 
        elif self.down:
            if self.gpu:
                x = oncuda.conv_downsample_2d(x, tf.cast(w, x.dtype), data_format='NCHW', k=self.resample_kernel, impl=self.impl)
            else:
                x = tf.transpose(x, [0, 2, 3, 1])
                x = oncuda.conv_downsample_2d(x, tf.cast(w, x.dtype), data_format='NHWC', k=self.resample_kernel, impl=self.impl, gpu=False)
                x = tf.transpose(x, [0, 3, 1, 2]) 
        else:
            if self.gpu:
                x = tf.nn.conv2d(x, tf.cast(w, x.dtype), data_format='NCHW', strides=[1,1,1,1], padding='SAME')
            else:
                x = tf.transpose(x, [0, 2, 3, 1])
                x = tf.nn.conv2d(x, tf.cast(w, x.dtype), data_format='NHWC', strides=[1,1,1,1], padding='SAME')
                x = tf.transpose(x, [0, 3, 1, 2])  
                  
        # Reshape/scale output.
        if self.fused_modconv:
            x = tf.reshape(x, [-1, self.fmaps, x.shape[2], x.shape[3]]) # Fused => reshape convolution groups back to minibatch.
        elif self.demodulate:
            x = tf.math.multiply(x, tf.cast(d[:, :, np.newaxis, np.newaxis], x.dtype)) # [BOhw] Not fused => scale output activations.
        return x
#
#
#   layers/synthesis_main_layer.py
#
class SynthesisMainLayer(tf.keras.layers.Layer):
    """
    StyleGan2 synthesis network main layer
    """
    def __init__(self, fmaps, up=False, impl='cuda', gpu=True,**kwargs):
        
        super(SynthesisMainLayer, self).__init__(**kwargs)
        
        self.fmaps = fmaps
        self.up = up
        self.impl = impl
        self.gpu = gpu
        
        self.resample_kernel = [1,3,3,1]
        self.kernel = 3
        
        if self.up:
            self.l_name = 'Conv0_up'
        else:
            self.l_name = 'Conv1'
        
    def build(self, input_shape):
        
        self.noise_strength = self.add_weight(name=self.l_name+'/noise_strength', shape=[], initializer=tf.initializers.zeros(), trainable=True)
        self.bias = self.add_weight(name=self.l_name+'/bias', shape=(self.fmaps,), initializer=tf.random_normal_initializer(0,1), trainable=True)
        
        self.mod_conv2d_layer = ModulatedConv2DLayer(fmaps=self.fmaps, kernel=self.kernel, 
                                                up=self.up, resample_kernel=self.resample_kernel,
                                                 impl=self.impl, gpu=self.gpu, name=self.l_name)
        
    def call(self, x, dlatent_vect):
                
        x = self.mod_conv2d_layer(x, dlatent_vect)
        
        #randomize noise
        noise = tf.random.normal([tf.shape(x)[0], 1, x.shape[2], x.shape[3]], dtype=x.dtype)
        
        # adding noise to layer
        x += tf.math.multiply(noise, tf.cast(self.noise_strength, x.dtype))
        
        # adding bias and lrelu activation
        x += tf.reshape(self.bias, [-1 if i == 1 else 1 for i in range(x.shape.rank)])
        x = tf.math.multiply(tf.nn.leaky_relu(x, 0.2), tf.math.sqrt(2.))
        
        return x
#
#
#   layers/to_rgb_layer.py
#
class ToRgbLayer(tf.keras.layers.Layer):
    """
    StyleGan2 generator To RGB layer
    """
    def __init__(self, impl='cuda', gpu=True,**kwargs):
        
        super(ToRgbLayer, self).__init__(**kwargs)
        
        self.impl = impl
        self.gpu = gpu
        
    def build(self, input_shape):
        
        self.mod_conv2d_rgb = ModulatedConv2DLayer(fmaps=3, kernel=1, demodulate=False, 
                                              impl=self.impl, gpu=self.gpu, name='ToRGB')
        
        self.rgb_bias = self.add_weight(name='ToRGB/bias', shape=(3,), 
                                        initializer=tf.random_normal_initializer(0, 1), trainable=True)
        
    def call(self, x, dlatent_vect, y):
        
        u = self.mod_conv2d_rgb(x, dlatent_vect)
        t = u + tf.reshape(self.rgb_bias, [-1 if i == 1 else 1 for i in range(x.shape.rank)])
        
        return t if y is None else y + t
#
#
#   stylegan2_discriminator.py
#
class StyleGan2Discriminator(tf.keras.layers.Layer):
    """
    StyleGan2 discriminator config f for tensorflow 2.x 
    """
    def __init__(self, resolution=1024, weights=None, impl='cuda', gpu=True, **kwargs):
        """
        Parameters
        ----------
        resolution : int, optional
            Resolution output of the synthesis network, will be parsed 
            to the floor integer power of 2. 
            The default is 1024.
        weights : string, optional
            weights name in weights dir to be loaded. The default is None.
        impl : str, optional
            Wether to run some convolutions in custom tensorflow 
            operations or cuda operations. 'ref' and 'cuda' available.
            The default is 'cuda'.
        gpu : boolean, optional
            Wether to use gpu. The default is True.
        """
        super(StyleGan2Discriminator, self).__init__(**kwargs)
    
        self.gpu = gpu
        self.impl = impl
    
        self.resolution = resolution
        if weights is not None: self.__adjust_resolution(weights)
        self.resolution_log2 = int(np.log2(resolution))
        
        # load weights
        if weights is not None:
            _ = self(tf.zeros(shape=(1, 3, self.resolution, self.resolution)))
            self.__load_weights(weights)
             
    def build(self, input_shape):
        
        self.mini_btch_std_layer = MinibatchStdLayer()
        self.from_rgb = FromRgbLayer(fmaps=onrosa.nf(self.resolution_log2-1), 
                                     name='{}x{}'.format(self.resolution, self.resolution),
                                     impl=self.impl, gpu=self.gpu)
        
        for res in range(self.resolution_log2, 2, -1):
            res_str = str(2**res)
            setattr(self, 'block_{}_{}'.format(res_str, res_str), 
                    BlockLayer(res=res, name='{}x{}'.format(res_str, res_str), 
                               impl=self.impl, gpu=self.gpu))
        
        #last layers
        self.conv_4_4 = Conv2DLayer(fmaps=onrosa.nf(1), kernel=3, impl=self.impl, 
                                    gpu=self.gpu, name='4x4/Conv')
        self.conv_4_4_bias = self.add_weight(name='4x4/Conv/bias', shape=(512,),
                                             initializer=tf.random_normal_initializer(0,1), trainable=True)
        self.dense_4_4 = DenseLayer(fmaps=512, name='4x4/Dense0')
        self.dense_output = DenseLayer(fmaps=1, name='Output')
    
    def _call(self, y):
        """
        Parameters
        ----------
        y : tensor of the image/s to evaluate. shape [batch, channel, height, width]
        Returns
        -------
        output of the discriminator. 
        """
        y = tf.cast(y, 'float32')
        x = None
        
        for res in range(self.resolution_log2, 2, -1):
            if  res == self.resolution_log2:
                x = self.from_rgb(x, y)
            x = getattr(self, 'block_{}_{}'.format(2**res, 2**res))(x)

        # minibatch std dev
        x = self.mini_btch_std_layer(x)
        
        # last convolution layer
        x = self.conv_4_4(x)
        x += tf.reshape(self.conv_4_4_bias, [-1 if i == 1 else 1 for i in range(x.shape.rank)])
        x = tf.math.multiply(tf.nn.leaky_relu(x, 0.2), tf.math.sqrt(2.))
        
        x = tf.reshape(x, [-1, np.prod([d for d in x.shape[1:]])])
        # dense layer
        x = self.dense_4_4(x)
        x = tf.math.multiply(tf.nn.leaky_relu(x, 0.2), tf.math.sqrt(2.))
        # output layer
        x = self.dense_output(x)
        
        return tf.identity(x, name='scores_out')
    
    def call(self, inputs, training=False):

        y, labels = inputs

        y = tf.cast(y, 'float32')
        x = None
        
        for res in range(self.resolution_log2, 2, -1):
            if  res == self.resolution_log2:
                x = self.from_rgb(x, y)
            x = getattr(self, 'block_{}_{}'.format(2**res, 2**res))(x)

        #minibatch std dev
        x = self.mini_btch_std_layer(x)
        
        #last convolution layer
        x = self.conv_4_4(x)
        x += tf.reshape(self.conv_4_4_bias, [-1 if i == 1 else 1 for i in range(x.shape.rank)])
        x = tf.math.multiply(tf.nn.leaky_relu(x, 0.2), tf.math.sqrt(2.))
        
        x = tf.reshape(x, [-1, np.prod([d for d in x.shape[1:]])])
        # dense layer
        x = self.dense_4_4(x)
        x = tf.math.multiply(tf.nn.leaky_relu(x, 0.2), tf.math.sqrt(2.))
        #output layer
        x = self.dense_output(x)
        
        return tf.identity(x, name='scores_out')
    
    def __adjust_resolution(self, weights_name):
        """
        Adjust resolution of the synthesis network output. 
        
        Parameters
        ----------
        weights_name : name of the weights
        """
        if  weights_name == 'ffhq': 
            self.resolution = 1024
        elif weights_name == 'car': 
            self.resolution = 512
        elif weights_name in ['cat', 'church', 'horse']: 
            self.resolution = 256
    
    def __load_weights(self, weights_name):
        """
        Load pretrained weights, stored as a dict with numpy arrays.
        Parameters
        ----------
        weights_name : name of the weights
        """
        
        if (weights_name in available_weights) and type(weights_name) == str:
            data = np.load(weights_stylegan2_dir + weights_name + '.npy', allow_pickle=True)[()]
            
            weights_discriminator = [data.get(key) for key in discriminator_weights[weights_name]]
            self.set_weights(weights_discriminator)
            
            print("Loaded {} discriminator weights!".format(weights_name))
        else:
            print('Cannot load the specified weights')
#
#
class MappingNetwork(tf.keras.layers.Layer):
    """
    StyleGan2 generator mapping network, from z to dlatents for tensorflow 2.x
    """
    def __init__(self, **kwargs):
        
        super(MappingNetwork, self).__init__(**kwargs)
        
        self.dlatent_size = 512
        self.dlatent_vector = 18
        self.mapping_layers = 8
        self.lrmul = 0.01
        
    def build(self, input_shape):

        self.weights_dict = {}
        for i in range(self.mapping_layers):
            setattr(self, 'Dense{}'.format(i), DenseLayer(fmaps=512, lrmul=self.lrmul, name='Dense{}'.format(i)))
    
        self.g_mapping_broadcast = tf.keras.layers.RepeatVector(self.dlatent_vector)
            
    def call(self, z):
        
        z = tf.cast(z, 'float32')
        
        # Normalize inputs
        scale = tf.math.rsqrt(tf.reduce_mean(tf.square(z), axis=1, keepdims=True) + 1e-8)
        x = tf.math.multiply(z, scale)
        
        # Mapping
        for i in range(self.mapping_layers):
        
            x = getattr(self, 'Dense{}'.format(i))(x)
            x = tf.math.multiply(tf.nn.leaky_relu(x, 0.2), tf.math.sqrt(2.))
        
        # Broadcasting
        dlatents = self.g_mapping_broadcast(x)
        
        return dlatents
#
class SynthesisNetwork(tf.keras.layers.Layer):
    """
    StyleGan2 generator synthesis network from dlatents to img tensor for tensorflow 2.x
    """
    def __init__(self, resolution=1024, impl='cuda', gpu=True, **kwargs):
        """
        Parameters
        ----------
        resolution : int, optional
            Resolution output of the synthesis network, will be parsed to the floor integer power of 2. 
            The default is 1024.
        impl : str, optional
            Wether to run some convolutions in custom tensorflow operations or cuda operations. 'ref' and 'cuda' available.
            The default is 'cuda'.
        gpu : boolean, optional
            Wether to use gpu. The default is True.
        """
        super(SynthesisNetwork, self).__init__(**kwargs)
        
        self.impl = impl
        self.gpu = gpu
        self.resolution = resolution
        
        self.resolution_log2 = int(np.log2(self.resolution))
        self.resample_kernel = [1, 3, 3, 1]

    def build(self, input_shape):
        
        #constant layer
        self.const_4_4 = self.add_weight(name='4x4/Const/const', shape=(1, 512, 4, 4), 
                                        initializer=tf.random_normal_initializer(0, 1), trainable=True)
        #early layer 4x4
        self.layer_4_4 = SynthesisMainLayer(fmaps=onrosa.nf(1), impl=self.impl, gpu=self.gpu, name='4x4')
        self.torgb_4_4 = ToRgbLayer(impl=self.impl, gpu=self.gpu, name='4x4')
        #main layers
        for res in range(3, self.resolution_log2 + 1):
            res_str = str(2**res)
            setattr(self, 'layer_{}_{}_up'.format(res_str, res_str), 
                    SynthesisMainLayer(fmaps=onrosa.nf(res-1), impl=self.impl, gpu=self.gpu, up=True, name='{}x{}'.format(res_str, res_str)))
            setattr(self, 'layer_{}_{}'.format(res_str, res_str), 
                    SynthesisMainLayer(fmaps=onrosa.nf(res-1), impl=self.impl, gpu=self.gpu, name='{}x{}'.format(res_str, res_str)))
            setattr(self, 'torgb_{}_{}'.format(res_str, res_str), 
                    ToRgbLayer(impl=self.impl, gpu=self.gpu, name='{}x{}'.format(res_str, res_str)))
        
    def call(self, dlatents_in):
        
        dlatents_in = tf.cast(dlatents_in, 'float32')
        y = None
        
        # Early layers
        x = tf.tile(tf.cast(self.const_4_4, 'float32'), [tf.shape(dlatents_in)[0], 1, 1, 1])
        x = self.layer_4_4(x, dlatents_in[:, 0])
        y = self.torgb_4_4(x, dlatents_in[:, 1], y)
                
        # Main layers
        for res in range(3, self.resolution_log2 + 1):
            x = getattr(self, 'layer_{}_{}_up'.format(2**res, 2**res))(x, dlatents_in[:, res*2-5])
            x = getattr(self, 'layer_{}_{}'.format(2**res, 2**res))(x, dlatents_in[:, res*2-4])
            y = oncuda.upsample_2d(y, k=self.resample_kernel, impl=self.impl, gpu=self.gpu)
            y = getattr(self, 'torgb_{}_{}'.format(2**res, 2**res))(x, dlatents_in[:, res*2-3], y)
                    
        images_out = y
        return tf.identity(images_out, name='images_out')
#	
#
#
#   stylegan2_generator.py
#
# from utils.weights_map import available_weights, synthesis_weights, mapping_weights, weights_stylegan2_dir
class StyleGan2Generator(tf.keras.layers.Layer):
    """
    StyleGan2 generator config f for tensorflow 2.x
    """
    def __init__(self, resolution=1024, weights=None, impl='cuda', gpu=True, **kwargs):
        """
        Parameters
        ----------
        resolution : int, optional
            Resolution output of the synthesis network, will be parsed 
            to the floor integer power of 2. 
            The default is 1024.
        weights : string, optional
            weights name in weights dir to be loaded. The default is None.
        impl : str, optional
            Wether to run some convolutions in custom tensorflow operations 
            or cuda operations. 'ref' and 'cuda' available.
            The default is 'cuda'.
        gpu : boolean, optional
            Wether to use gpu. The default is True.
        """
        super(StyleGan2Generator, self).__init__(**kwargs)
        
        self.resolution = resolution
        if weights is not None: self.__adjust_resolution(weights)

        self.mapping_network = MappingNetwork(name='Mapping_network')
        self.synthesis_network = SynthesisNetwork(resolution=self.resolution, impl=impl, 
                                                  gpu=gpu, name='Synthesis_network')
        
        # load weights
        if weights is not None:
            #we run the network to define it, not the most efficient thing to do...
            _ = self([tf.zeros(shape=(1, 512)), None])
            self.__load_weights(weights)
        
    def _call(self, z):
        """
        Parameters
        ----------
        z : tensor, latent vector of shape [batch, 512]
        Returns
        -------
        img : tensor, image generated by the generator of shape  [batch, channel, height, width]
        """
        dlatents = self.mapping_network(z)
        img = self.synthesis_network(dlatents)

        return img
    
    # _e_
    # add 
    def call(self, inputs, truncation_cutoff=None, truncation_psi=1.0, training=None, mask=None):

        # weights_name = 'ffhq'
        # w_average = np.load(f'weights/{weights_name}_dlatent_avg.npy')
        # self.n_broadcast = len(self.resolutions) * 2        
        # self.broadcast = tf.keras.layers.Lambda(lambda x: tf.tile(x[:, np.newaxis], [1, self.n_broadcast, 1]))

        dlatents, labels = inputs

        dlatents = self.mapping_network(dlatents) # z
        img = self.synthesis_network(dlatents)

        # if training:
        #     self.update_moving_average_of_w(w_broadcasted)
        #     w_broadcasted = self.style_mixing_regularization(dlatents, labels, w_broadcasted)

        # if not training:
        #     w_broadcasted = self.truncation_trick(w_broadcasted, truncation_cutoff, truncation_psi)

        # # adjusting dlatents depending on truncation psi, if truncatio_psi = 1, no adjust
        # dlatents = w_average + (dlatents - w_average) * truncation_psi 
        # # running synthesis network
        # out = self.synthesis_network(dlatents)

        # #converting image/s to uint8
        # img = convert_images_to_uint8(out, nchw_to_nhwc=True, uint8_cast=True)

        # # image_out = self.synthesis(w_broadcasted)
        # # return image_out, w_broadcasted

        return img   

    def __adjust_resolution(self, weights_name):
        """
        Adjust resolution of the synthesis network output. 
        
        Parameters
        ----------
        weights_name : name of the weights
        Returns
        -------
        None.
        """
        if  weights_name == 'ffhq': 
            self.resolution = 1024
        elif weights_name == 'car': 
            self.resolution = 512
        elif weights_name in ['cat', 'church', 'horse']: 
            self.resolution = 256
    
    def __load_weights(self, weights_name):
        """
        Load pretrained weights, stored as a dict with numpy arrays.
        Parameters
        ----------
        weights_name : name of the weights
        Returns
        -------
        None.
        """
        
        if (weights_name in available_weights) and type(weights_name) == str:
            data = np.load(weights_stylegan2_dir + weights_name + '.npy', allow_pickle=True)[()]
            
            weights_mapping = [data.get(key) for key in mapping_weights]
            weights_synthesis = [data.get(key) for key in synthesis_weights[weights_name]]
            
            self.mapping_network.set_weights(weights_mapping)
            self.synthesis_network.set_weights(weights_synthesis)
            
            print("Loaded {} generator weights!".format(weights_name))
        else:
            raise Exception('Cannot load {} weights'.format(weights_name))
#
#
#
#   stylegan2.py
#
class StyleGan2(tf.keras.Model):
    """ 
    StyleGan2 config f for tensorflow 2.x 
    """
    def __init__(self, 
        resolution=1024, 
        weights=None, 
        impl='cuda', 
        gpu=True, 
  
        # ---------
        lr = 0.0001, 
        decay = 0.00001, 
        silent = True,

        latent_size = 512,
        im_size = 256,

        batch_size = 4, # 16

        mixed_prob = 0.9,
    
        steps = 1, 
        ncyc = None, 

        qsteps = 1000001,

        name='sol',
        results_dir = "results",
        models_dir = "models",
        logs_dir = "logs",
        ckpt_dir = "checkpoint",
        ckpt_prefix = 'ckpt',

        dataorg_dir = "images",
        data_dir = "data",
        dataset_dir = "dataset",
        tfrecord_dir = "tfrecords",

        update = 0,
        images = [],
        segments = [],
        verbose = True,   

        training_parameters = {},
        # ---------
        **kwargs
        ):

        super(StyleGan2, self).__init__(**kwargs)

        # --------------

        self.results_dir = results_dir
        self.models_dir = models_dir
        self.logs_dir = logs_dir

        self.data_dir = data_dir
        self.dataset_dir = dataset_dir
        self.tfrecord_dir = tfrecord_dir

        self.latent_size = latent_size
        self.im_size = im_size
        self.batch_size = batch_size
        self.update = update
        self.images = images
        self.segments = segments
        self.verbose = verbose

        self.mixed_prob = 0.9
    
        self.im_size = im_size
        print("StyleGan:im_size: %d " %self.im_size)    

        self.n_layers = int(log2(im_size) - 1)
        print("StyleGan:n_layers: %d " %self.n_layers)

        self.steps = steps
        self.qsteps = qsteps                                # to train steps
        print("StyleGan:qsteps: will train up to %d steps" %self.qsteps)

        t_params = training_parameters

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

        self.r1_gamma = 10.0
        # self.r2_gamma = 0.0
        self.max_steps = int(np.ceil(self.n_total_image / self.batch_size))
        self.out_res = self.g_params['resolutions'][-1]
        self.log_template = 'step {}: elapsed: {:.2f}s, d_loss: {:.3f}, g_loss: {:.3f}, r1_penalty: {:.3f}'
        self.print_step = 10
        self.save_step = 100
        self.image_summary_step = 100
        self.reached_max_steps = False        
        # --------------

        self.resolution = resolution

        if weights is not None:
            self.__adjust_resolution(weights)
        print(f'resolution: {self.resolution}')            

        print('Create models')            
        self.G = StyleGan2Generator(resolution=self.resolution, weights=weights, 
                        impl=impl, gpu=gpu, name='Generator')
        self.D = StyleGan2Discriminator(resolution=self.resolution, weights=weights, 
                        impl=impl, gpu=gpu, name='Discriminator')
        

        learning_rate = 0.01
        beta1 = 0.99
        beta2 = 0.99
        epsilon = 1e-1
        self.DMO = tf.keras.optimizers.Adam(learning_rate,
                                        beta_1=beta1,
                                        beta_2=beta2,
                                        epsilon=epsilon)
        self.GMO = tf.keras.optimizers.Adam(learning_rate,
                                        beta_1=beta1,
                                        beta_2=beta2,
                                        epsilon=epsilon)
        self.g_clone = StyleGan2Generator(resolution=self.resolution, 
                weights=weights, 
                impl=impl, gpu=gpu, name='Generator')

        self.M = MappingNetwork(name='Mapping_network')
        self.S = SynthesisNetwork(resolution=self.resolution, impl=impl, 
                                        gpu=gpu, name='Synthesis_network')

        self.ckpt_dir = ckpt_dir
        self.ckpt_prefix = ckpt_prefix
        self.ckpt = tf.train.Checkpoint(
            GMO=self.GMO,
            DMO=self.DMO,
            G=self.G,
            D=self.D,
             g_clone=self.g_clone)

        self.manager = tf.train.CheckpointManager(self.ckpt, self.ckpt_dir, max_to_keep=2)

        self.restore_checkpoint()
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

        print("|... StyleGan2 inited")

    def call(self, latent_vector):
        """
        Parameters
        ----------
        latent_vector : latent vector z of size [batch, 512].
        Returns
        -------
        score : output of the discriminator. 
        """
        img = self.G([latent_vector, None]) # _e_
        score = self.D([img, None])

        return score    

    def noise(self, n):
        latent_size = self.latent_size

        return np.random.normal(0.0, 1.0, size = [n, latent_size]).astype('float32')

    def noiseList(self, n):
        noise = self.noise
        n_layers = self.n_layers

        return [noise(n)] * (n_layers)

    def mixedList(self, n):
        noise = self.noise
        n_layers = self.n_layers

        tt = int(np.random.uniform(()) * n_layers)
        p1 = [noise(n)] * tt
        p2 = [noise(n)] * (n_layers - tt)
        return p1 + [] + p2

    def nImage(self, n):
        im_size = self.im_size
        return np.random.uniform(0.0, 1.0, size = [n, im_size, im_size, 1]).astype('float32')

    def __adjust_resolution(self, weights_name):
        """
        Adjust resolution of the synthesis network output. 
        
        Parameters
        ----------
        weights_name : name of the weights
        """
        if  weights_name == 'ffhq': 
            self.resolution = 1024
        elif weights_name == 'car': 
            self.resolution = 512
        elif weights_name in ['cat', 'church', 'horse']: 
            self.resolution = 256

    @tf.function            
    def train_step(self, real_images):
        if 0:
            print(f"|===> train_step \n \
                    real_images: {np.shape(real_images)} {type(real_images)} \n \
            ")

        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:

            z = tf.random.normal(shape=[tf.shape(real_images)[0], self.g_params['z_dim']], dtype=tf.dtypes.float32)
            real_images = onmoono.preprocess_fit_train_image(real_images, self.out_res)
            labels = tf.ones((tf.shape(real_images)[0], self.g_params['labels_dim']), dtype=tf.dtypes.float32)

            fake_images = self.G([z, labels], training=True)
            real_scores = self.D([real_images, labels], training=True)
            fake_scores = self.D([fake_images, labels], training=True)
            
            divergence = tf.math.softplus(fake_scores)  # log ( 1 + exp( D(fakes) ) ) # alt tf.math.relu
            divergence += tf.math.softplus(-real_scores) # log ( 1 - exp( D(reals) ) )  # alt tf.math.relu
            divergence = tf.keras.backend.mean(divergence)

            d_loss = tf.reduce_mean(divergence)
            g_loss = tf.math.softplus(-fake_scores)
            g_loss = tf.reduce_mean(g_loss)

        g_gradients = g_tape.gradient(g_loss, self.G.trainable_variables)
        self.GMO.apply_gradients(zip(g_gradients, self.G.trainable_variables))

        d_gradients = d_tape.gradient(d_loss, self.D.trainable_variables)
        self.DMO.apply_gradients(zip(d_gradients, self.D.trainable_variables))

        return d_loss, g_loss

    def fit(self):

        print(f"|===> fit \n \
                self.steps: {self.steps} \n \
                self.qsteps: {self.qsteps} \n \
                self.tfrecord_dir: {self.tfrecord_dir} \n \
                self.out_res: {self.out_res} \n \
                self.shuffle_buffer_size: {self.shuffle_buffer_size} \n \
                self.batch_size: {self.batch_size} \n \
                self.g_params: {self.g_params} \n \
            \n ")

        n_iterations = 100

        # if self.reached_max_steps:
        #     return

        # tensorboards
        summary_writer = tf.summary.create_file_writer(
            os.path.join(self.logs_dir, "fit" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

        # loss metrics
        metric_g_loss = tf.keras.metrics.Mean('g_loss', dtype=tf.float32)
        metric_d_loss = tf.keras.metrics.Mean('d_loss', dtype=tf.float32)
        metric_r1_reg = tf.keras.metrics.Mean('r1_reg', dtype=tf.float32)
        metric_pl_reg = tf.keras.metrics.Mean('pl_reg', dtype=tf.float32)

        # start training
        # print('max_steps: {}'.format(self.max_steps))
        losses = {'g_loss': 0.0, 'd_loss': 0.0, 'r1_reg': 0.0, 'pl_reg': 0.0}
        t_start = time.time()

        self.dataset = onrecord.get_dataset(
            self.tfrecord_dir,
            self.out_res,
            self.shuffle_buffer_size,
            self.batch_size,
            epochs=None
        )

        for real_images in self.dataset: # (4, 3, 256, 256)

            z = tf.random.normal(shape=[tf.shape(real_images)[0], self.g_params['z_dim']], dtype=tf.dtypes.float32)
            real_images = onmoono.preprocess_fit_train_image(real_images, self.out_res)
            labels = tf.ones((tf.shape(real_images)[0], self.g_params['labels_dim']), dtype=tf.dtypes.float32)

            # get current step
            step = self.GMO.iterations.numpy()

            print('.', end='')
            if (step+1) % 100 == 0: 
                print(f"|---> fit \n {step}: {np.shape(self.dataset)} {type(self.dataset)} \n ")

            self.train_step(real_images)

            # 	save checkpoint
            
            if (step + 1) % n_iterations == 0:
                print(f'|... saving (checkpoint) the model every \
                    {n_iterations} steps to {self.ckpt_prefix}')
                self.ckpt.save(file_prefix = self.ckpt_prefix)
                #self.manager.save(checkpoint_number=step)				

                print ('Time taken for step {} is {} sec\n'.format(step + 1, time.time()-t_start))

        self.ckpt.save(file_prefix = self.ckpt_prefix)
        # self.manager.save(checkpoint_number=step)

    def restore_checkpoint(self, max_to_keep=5):

        print(f'|===> model.restore_checkpoint \n \
            self.ckpt: {self.ckpt} \n \
            self.ckpt_dir: {self.ckpt_dir} \n \
            max_to_keep: {max_to_keep} \n \
        ')

        self.ckpt_manager = ckpt_manager = tf.train.CheckpointManager(
            self.ckpt, 
            self.ckpt_dir, 
            max_to_keep=max_to_keep
        )
        # if a checkpoint exists, restore the latest checkpoint.
        if ckpt_manager.latest_checkpoint:
            self.ckpt.restore(ckpt_manager.latest_checkpoint)
            print(f'\n |...> model.restore_checkpoint checkpoint restored !!! \n')
        else:
            print(f'|...> model.restore_checkpoint checkpoint not found')
#
#
#   CMDS
#
#
#
#
#   nntrain
#
def nntrain(args, kwargs):
    # ERROR TF COMPILE
    
    args = onutil.pargs(vars(args))
    args.PROJECT = 'rosalberto'
    args.DATASET = 'spaceone'
    xp = getxp(vars(args))
    args = onutil.pargs(xp)
    onutil.ddict(vars(args), 'args')

    print(f"|===> nntrain:  \n \
        cwd: {os.getcwd()} \n \
        AUTHOR: {args.AUTHOR} \n \
        PROJECT: {args.PROJECT} \n \
        proj_dir: {args.proj_dir} \n \
        verbose: {args.verbose} \n \
    ")

    if 1: # config
        args.clip = (256, 256)
        args.qslices = 10
        args.eps = 0.9
        args.impl = 'cuda' # 'ref' if cuda is not available in your machine
        args.gpu = True # False if tensorflow cpu is used   		
        args.weights_name = 'ffhq' # face model trained by Nvidia

        args.shuffle_buffer_size = 10 # 1000 # _e_
        args.latent_size = 512
        args.train_res = 256   # real images _e_
        args.im_size = 256
        args.batch_size = 4 # 12, # 16
        args.mixed_prob = 0.9
        args.qsteps = 1001 # 1000001,   #   up to step

    print(f'|===> nndata config:  \n \
        args.clip: {args.clip} \n \
        args.qslices: {args.qslices} \n \
        args.eps: {args.eps} \n \
        args.impl: {args.impl} \n \
        args.gpu: {args.gpu} \n \
        args.weights_name: {args.weights_name} \n \
        \n \
        args.train_res: {args.train_res} \n \
        args.shuffle_buffer_size: {args.shuffle_buffer_size} \n \
        args.latent_size: {args.latent_size} \n \
        args.im_size: {args.im_size}\n \
        args.batch_size: {args.batch_size}\n \
        args.mixed_prob: {args.mixed_prob}\n \
        args.qsteps: {args.qsteps}\n \
    ')

    if 1: # tree

        args.model_base_dir = args.models_dir
        args.tfrecord_dir = args.records_dir
        args.proj_dir = os.path.join(args.proj_dir, 'code')

        # args.tfrecord_dir = os.path.join(args.gdata, 'ffhq')
        # args.tfrecord_dir = args.dataset_dir

        args.results_dir = os.path.join(args.proto_dir, 'results')

        args.ckpt_dir = args.ckpt_dir
        args.ckpt_prefix = os.path.join(args.ckpt_dir, "ckpt")

        os.makedirs(args.data_dir, exist_ok=True)    
        os.makedirs(args.proj_dir, exist_ok=True)    
        os.makedirs(args.dataset_dir, exist_ok=True)    
        os.makedirs(args.tfrecord_dir, exist_ok=True)     
        os.makedirs(args.results_dir, exist_ok=True)     
        os.makedirs(args.ckpt_dir, exist_ok=True)     

        weights_dir = os.path.join(args.proj_dir, 'weights')
        weights_path = os.path.join(weights_dir, args.weights_name + '.npy')

        os.makedirs(weights_dir, exist_ok=True)

    print(f'|===> nntrain tree:  \n \
        cwd: {os.getcwd()} \n \
        args.proj_dir: {args.proj_dir} \n \
        args.model_base_dir: {args.model_base_dir} \n \
        args.tfrecord_dir: {args.tfrecord_dir} \n \
        args.ckpt_dir: {args.ckpt_dir} \n \
        args.ckpt_prefix: {args.ckpt_prefix} \n \
        args.models_dir: {args.models_dir} \n \
        args.dataorg_dir: {args.dataorg_dir} \n \
        args.data_dir: {args.data_dir} \n \
        args.dataset_dir: {args.dataset_dir} \n \
    ')

    if 1: # git

        onutil.get_git(args.AUTHOR, args.GITPOD, args.proj_dir)

    if 1: # org => data (formdata)

        print(f'|====> nndata formed images \n \
            from {args.dataorg_dir} into {args.data_dir} \n \
            args.clip: {args.clip} \n \
            qslices: {args.qslices} \n \
            eps: {args.eps} \n \
        ')

        qinfiles = onfile.qfiles(args.dataorg_dir, ['*.jpg', '*.png'])
        qinslices = qinfiles * args.qslices
        qoutfiles = onfile.qfiles(args.data_dir, ['*.jpg', '*.png'])
        if qinslices > qoutfiles:

            if onutil.isempty(args.data_dir):
                print(f"args.data_dir {args.data_dir} not empty !!!")

            imgs = ondata.folder_to_formed_pils(args.dataorg_dir, 
                clip = args.clip, qslices = args.qslices, eps = args.eps)
            onfile.pils_to_folder(imgs, args.data_dir)

        else:

            print(f'|... no org to data files')

        print(f"|... nndata formed images \n \
            {onfile.qfiles(args.data_dir)} files in {args.data_dir} \n \
        ")

    if 1: # data => tfrecords

        print(f"|===> nndata generate tfrecords \n \
            from {args.data_dir} into {args.tfrecord_dir} \n \
        ")
        if onutil.isempty(args.tfrecord_dir):
            print(f"args.tfrecord_dir {args.tfrecord_dir} not empty !!!")
        onrecord.folder_to_tfrecords(args.data_dir, args.tfrecord_dir)

        print(f"|===> nndata tfrecords \n \
            {onfile.qfiles(args.tfrecord_dir)} files in {args.tfrecord_dir} \n \
        ")  

    if 1: # net params

        train_res = args.train_res

        train_resolutions, train_featuremaps = onmoono.filter_resolutions_featuremaps(resolutions, featuremaps, train_res)
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

        vs = vars(args)
        training_parameters = {
            # global params
            **vs,

            # network params
            'g_params': g_params,
            'd_params': d_params,

            # training params
            'g_opt': {'learning_rate': 0.002, 'beta1': 0.0, 'beta2': 0.99, 'epsilon': 1e-08},
            'd_opt': {'learning_rate': 0.002, 'beta1': 0.0, 'beta2': 0.99, 'epsilon': 1e-08},
            'batch_size': vs['batch_size'],
            'n_total_image': 25000000,
            'n_samples': 4,
        }

        # print(f"training_parameters {training_parameters}")
        onutil.ddict(training_parameters, "training_parameters")

    if 1: # model

        model = StyleGan2(        
            resolution=args.train_res, 
            weights=None, 
            impl=args.impl, 
            gpu=args.gpu, 

            latent_size = args.latent_size,
            im_size = args.im_size,
            batch_size = args.batch_size, # 12, # 16
            mixed_prob = args.mixed_prob,
            qsteps = args.qsteps, # 1000001,   #   up to step

            name = args.MNAME,
            results_dir = args.results_dir,
            models_dir = args.models_dir,
            ckpt_dir = args.ckpt_dir,
            ckpt_prefix = args.ckpt_prefix,
            dataorg_dir = args.dataorg_dir,
            logs_dir = args.logs_dir,
            data_dir = args.data_dir,
            dataset_dir = args.dataset_dir,
            tfrecord_dir = args.tfrecord_dir,

            training_parameters = training_parameters,
        )

    if 0: # fit

        # trainer = Trainer(training_parameters, name='stylegan2-ffhq')
        # trainer.train()    
        model.fit()

    if 0: # weights

        if (not os.path.exists(args.weights_path)):
            print(f"could not find weights file {args.weights_path}")

            weights_id = weights[args.weights_name]

            # output = weight_dir + name[i]
            output = args.weights_path
            url = f"https://drive.google.com/uc?id={weights_id}"
            print(f"gdown {url} to {output}")

            gdown.download(url, output, quiet=False) 

        else:
            print(f"weights file {weights_path} found")

    if 1: # generator

        print("instantiate generator network")
        #generator = StyleGan2Generator(weights=args.weights_name, 
        #	impl=args.impl, gpu=args.gpu)
        generator = model.G

        print("load w average")
        wavg_path = os.path.join(args.proj_dir, "weights", 
            f"{args.weights_name}_dlatent_avg.npy")
        w_average = np.load(wavg_path)

        print("generate and plot images not using truncation")
        onplot.generate_and_plot_images(generator, seed=96, w_avg=w_average)

        print("generate and plot images using truncation 0.5")
        onplot.generate_and_plot_images(generator, seed=96, w_avg=w_average, truncation_psi=0.5)

    if 1: # generate images 	- creating random latent vector

        import tqdm 

        seed = 96
        rnd = np.random.RandomState(seed)
        z = rnd.randn(4, 512).astype('float32')

        # running network
        out = generator([z, None]) # _e_

        #converting image to uint8
        out_image = onrosa.convert_images_to_uint8(out, nchw_to_nhwc=True, uint8_cast=True)

        onfile.generate_and_save_images(out_image.numpy(), outdir = args.results_dir)

        n_images = 500

        for i in tqdm(range(n_images)):
            onfile.generate_and_save_images(out_image.numpy(), i, plot_fig=False, outdir = args.results_dir)
            
            #moving randomly in the latent space z
            seed = i
            rnd = np.random.RandomState(seed)
            
            #mofying slightly latent vector and generating new images
            z += rnd.randn(4, 512).astype('float32') / 40
            out = generator([z, None]) # _e_
            out_image = onrosa.convert_images_to_uint8(out, nchw_to_nhwc=True, uint8_cast=True)

    if 0: # save gif

        anim_file = os.path.join(args.results_dir, 'ffhq_latent.gif')

        with imageio.get_writer(anim_file, mode='I') as writer:
            filenames = glob.glob(os.path.join(args.results_dir, 'image_at_iter*.png'))
            filenames = sorted(filenames)
            for i,filename in enumerate(filenames):
                if i % 8 != 0:
                    continue
                image = imageio.imread(filename)
                writer.append_data(image)
            image = imageio.imread(filename)
            writer.append_data(image)

    if 0: # project
        args.dataorg_dir=os.path.join(args.dataorg_dir, "")

        args.dataset_dir=os.path.join(args.proj_dir, "dataset")   
        args.models_dir=os.path.join(args.proj_dir, "Models")   
        args.results_dir=os.path.join(args.proj_dir, "Results")   
        args.ani_dir=os.path.join(args.proj_dir, "ani")   

        os.makedirs(args.data_dir, exist_ok=True) # deduped formed
        os.makedirs(args.dataset_dir, exist_ok=True) # npy folder
        os.makedirs(args.models_dir, exist_ok=True)
        os.makedirs(args.results_dir, exist_ok=True)
        os.makedirs(args.tmp_dir, exist_ok=True) # name raws
        os.makedirs(args.ani_dir, exist_ok=True) # anis

        numSteps = 1000
        initialLearningRate = 0.1
        initialNoiseFactor = 0.05
        verbose = False

        tfrecord_dir = args.dataset_dir
        out_res =  256
        shuffle_buffer_size = 10
        batch_size = 4
        
        print(f"|===> nnproj\n \
            args.proj_dir: {args.proj_dir} \n \
            args.data_dir: {args.data_dir} \n \
            args.dataset_dir: {args.dataset_dir} \n \
            tfrecord_dir: {tfrecord_dir} \n \
        ")

        dataset = onmoono.get_dataset(
            tfrecord_dir,
            out_res,
            shuffle_buffer_size,
            batch_size,
            epochs=None
        )
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
