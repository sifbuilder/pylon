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
ondata = Ondata()
onset = Onset()
ontree = Ontree()
onrecord = Onrecord()
onrolux = Onrolux()
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
#
def getap():
    cp = {
        "primecmd": 'nntrain',

        "MNAME": "matchue",
        "AUTHOR": "manicman1999",
        "GITPOD": "StyleGan2-Tensorflow-2.0",
        "PROJECT": "space",
        "DATASET": "space",
    
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
        "batch_size": 1,
        "img_width": 256,
        "img_height": 256,
        "buffer_size": 1000,
        "input_channels": 3,
        "output_channels": 3,
        "epochs": 100,
        "n_iterations": 10, # iters for snapshot
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
# Convolutional layers.
#
class Conv2DMod(tf.keras.layers.Layer):

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 dilation_rate=1,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 demod=True,
                 **kwargs):

         
        super(Conv2DMod, self).__init__(**kwargs) # super(Conv2DMod, self).__init__() # 
      
        self.filters = filters
        self.rank = 2
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 2, 'dilation_rate')
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.demod = demod
        self.input_spec = [InputSpec(ndim = 4),
                            InputSpec(ndim = 2)]

    def build(self, input_shape):
        channel_axis = -1
        if input_shape[0][channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[0][channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        if input_shape[1][-1] != input_dim:
            raise ValueError('The last dimension of modulation input should be equal to input dimension.')

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        # Set input spec.
        self.input_spec = [InputSpec(ndim=4, axes={channel_axis: input_dim}),
                            InputSpec(ndim=2)]
        self.built = True

    def call(self, inputs):
        #To channels last
        x = tf.transpose(inputs[0], [0, 3, 1, 2])

        #Get weight and bias modulations
        #Make sure w's shape is compatible with self.kernel
        w = K.expand_dims(K.expand_dims(K.expand_dims(inputs[1], axis = 1), axis = 1), axis = -1)

        #Add minibatch layer to weights
        wo = K.expand_dims(self.kernel, axis = 0)

        #Modulate
        weights = wo * (w+1)

        #Demodulate
        if self.demod:
            d = K.sqrt(K.sum(K.square(weights), axis=[1,2,3], keepdims = True) + 1e-8)
            weights = weights / d

        #Reshape/scale input
        x = tf.reshape(x, [1, -1, x.shape[2], x.shape[3]]) # Fused => reshape minibatch to convolution groups.
        w = tf.reshape(tf.transpose(weights, [1, 2, 3, 0, 4]), [weights.shape[1], weights.shape[2], weights.shape[3], -1])

        x = tf.nn.conv2d(x, w,
                strides=self.strides,
                padding="SAME",
                data_format="NCHW")

        # Reshape/scale output.
        x = tf.reshape(x, [-1, self.filters, x.shape[2], x.shape[3]]) # Fused => reshape convolution groups back to minibatch.
        x = tf.transpose(x, [0, 2, 3, 1])

        return x

    def compute_output_shape(self, input_shape):
        item = input_shape[0][1:-1]
        new_item = []
        for i in range(len(item)):
            new_dim = conv_utils.conv_output_length(
                item[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            new_item.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.filters,)

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.dilation_rate,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'demod': self.demod
        }
        base_config = super(Conv2DMod, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
#
#
class StyleGan2(object):

    def __init__(self, 
        lr = 0.0001, 
        decay = 0.00001,

        im_size = 256,
        latent_size = 512,
        BATCH_SIZE = 4, # 16

        cha = 24,

        mixed_prob = 0.9,
   
        steps = 1, 
        ncyc = None, 

        qsteps = 1000001,

        n_cycle = 1000,
        n_save = 500,
        n_ref = 10000,

        name='sol',
        results_dir = "results",
        models_dir = "models",
        dataorg_dir = "images",
        data_dir = "data",
        dataset_dir = "dataset",
        ani_dir = "ani",

        update = 0,
        images = [],
        segments = [],
        verbose = True,    
        silent = True,

    ):
        print(" ------------- matchue ------------- ")
        self.name = name

        self.results_dir = results_dir
        self.models_dir = models_dir
        self.data_dir = data_dir
        self.dataset_dir = dataset_dir
        self.ani_dir = ani_dir

        self.latent_size = latent_size
        self.im_size = im_size
        self.BATCH_SIZE = BATCH_SIZE
        self.update = update
        self.images = images
        self.segments = segments
        self.verbose = verbose

        self.mixed_prob = 0.9        
        self.steps = steps

        self.qsteps = qsteps       
        if 0:                         # to train steps
            print("StyleGan:qsteps: will train up to %d steps" %self.qsteps)

        self.n_cycle = n_cycle  # will evaluate model each
        self.n_save = n_save    # will save model each
        self.n_ref = n_ref   # will identify model each

        self.im_size = im_size
        self.latent_size = latent_size
        self.BATCH_SIZE = BATCH_SIZE

        self.cha = cha
 
        self.mixed_prob = 0.9
    
        self.n_layers = int(log2(im_size) - 1)
        if 0:
            print("matchue.Gan.n_layers %d" %self.n_layers)

        #Models
        self.D = None
        self.S = None
        self.G = None

        self.GE = None
        self.SE = None

        self.DM = None
        self.AM = None

        #Config
        self.LR = lr

        self.beta = 0.999

        #Init Models
        self.discriminator()
        self.stylemapper()
        self.generator()

        self.GMO = Adam(lr = self.LR, beta_1 = 0, beta_2 = 0.999)
        self.DMO = Adam(lr = self.LR, beta_1 = 0, beta_2 = 0.999)

        self.GE = clone_model(self.G)
        self.GE.set_weights(self.G.get_weights())

        self.SE = clone_model(self.S)
        self.SE.set_weights(self.S.get_weights())

        self.GenModel()
        self.GenModelA()

        # --------------
        self.ncyc = ncyc
        if self.ncyc is not None:
            print(f"ncyc {self.ncyc} specified as arg")
        else:
            lastSavedCyc = self.getRecordNumber()
            if lastSavedCyc is not None:
                print()
                print("StyleGan2:lastSavedCyc: %d" %lastSavedCyc)
                print()
            self.ncyc = lastSavedCyc

        if self.ncyc is not None:
            nstep = (self.ncyc + 1) * self.n_cycle + 1 # cyc 0 save will jump to n_cycle steps
            self.steps = nstep
            print(f"StyleGan2:load MODEL {self.ncyc} for steps: {nstep}")
            self.load(self.ncyc)
        else:
            print("StyleGan2:ncyc is None, will not load model")

        if 0:
            print("StyleGan2:steps %d" %self.steps)

        #Set up variables
        self.lastblip = time.process_time()

        self.silent = silent

        self.ones = np.ones((BATCH_SIZE, 1), dtype=np.float32)
        self.zeros = np.zeros((BATCH_SIZE, 1), dtype=np.float32)
        self.nones = -self.ones

        self.pl_mean = 0
        self.av = np.zeros([44]) # _e_

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

    @staticmethod
    def hinge_d(y_true, y_pred):
        return K.mean(K.relu(1.0 + (y_true * y_pred)))

    @staticmethod
    def w_loss(y_true, y_pred):
        return K.mean(y_true * y_pred)

    #Lambdas
    # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Lambda?version=stable
    @staticmethod
    def crop_to_fit(x):
        height = x[1].shape[1]
        width = x[1].shape[2]

        return x[0][:, :height, :width, :]

    @staticmethod
    def upsample(x):
        return K.resize_images(x,2,2,"channels_last",interpolation='bilinear')

    @staticmethod
    def upsample_to_size(x, im_size=1024):
        y = int(im_size / x.shape[2]) # _e_
        x = K.resize_images(x, y, y, "channels_last",interpolation='bilinear')
        return x

    #Blocks
    def g_block(self, inp, istyle, inoise, fil, u = True):
        crop_to_fit = self.crop_to_fit
        to_rgb = self.to_rgb
        upsample = self.upsample

        if u:
            #Custom upsampling because of clone_model issue
            out = Lambda(upsample, output_shape=[None, inp.shape[2] * 2, inp.shape[2] * 2, None])(inp)
        else:
            out = Activation('linear')(inp)

        rgb_style = Dense(fil, kernel_initializer = VarianceScaling(200/out.shape[2]))(istyle)
        style = Dense(inp.shape[-1], kernel_initializer = 'he_uniform')(istyle)
        delta = Lambda(crop_to_fit)([inoise, out])
        d = Dense(fil, kernel_initializer = 'zeros')(delta)

        out = Conv2DMod(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_uniform')([out, style])
        out = add([out, d])
        out = LeakyReLU(0.2)(out)

        style = Dense(fil, kernel_initializer = 'he_uniform')(istyle)
        d = Dense(fil, kernel_initializer = 'zeros')(delta)

        out = Conv2DMod(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_uniform')([out, style])
        out = add([out, d])
        out = LeakyReLU(0.2)(out)

        return out, to_rgb(out, rgb_style)

    def d_block(self, inp, fil, p = True):
        res = Conv2D(fil, 1, kernel_initializer = 'he_uniform')(inp)
        out = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_uniform')(inp)
        out = LeakyReLU(0.2)(out)
        out = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_uniform')(out)
        out = LeakyReLU(0.2)(out)

        out = add([res, out])

        if p:
            out = AveragePooling2D()(out)

        return out

    def to_rgb(self, inp, style):

        im_size = self.im_size

        upsample_to_size = self.upsample_to_size

        size = inp.shape[2]
        x = Conv2DMod(3, 1, kernel_initializer = VarianceScaling(200/size), demod = False)([inp, style])
        return Lambda(upsample_to_size, output_shape=[None, im_size, im_size, None], arguments= {'im_size': im_size})(x)

    def from_rgb(self, inp, conc = None):
        
        im_size = self.im_size

        fil = int(im_size * 4 / inp.shape[2])
        z = AveragePooling2D()(inp)
        x = Conv2D(fil, 1, kernel_initializer = 'he_uniform')(z)
        if conc is not None:
            x = concatenate([x, conc])
        return x, z

    def discriminator(self):

        im_size = self.im_size
        cha = self.cha

        if self.D:
            return self.D

        inp = Input(shape = [im_size, im_size, 3])
        x = self.d_block(inp, 1 * cha)   #128
        x = self.d_block(x, 2 * cha)   #64
        x = self.d_block(x, 4 * cha)   #32
        x = self.d_block(x, 6 * cha)  #16
        x = self.d_block(x, 8 * cha)  #8
        x = self.d_block(x, 16 * cha)  #4
        x = self.d_block(x, 32 * cha, p = False)  #4
        x = Flatten()(x)
        x = Dense(1, kernel_initializer = 'he_uniform')(x)

        self.D = Model(inputs = inp, outputs = x)

        return self.D

    def stylemapper(self):
        print(" ------------- matchue stylemapper start ------------- ")

        if self.S:
            return self.S

        latent_size = self.latent_size

        self.S = Sequential()

        self.S.add(Dense(512, input_shape = [latent_size]))
        self.S.add(LeakyReLU(0.2))
        self.S.add(Dense(512))
        self.S.add(LeakyReLU(0.2))
        self.S.add(Dense(512))
        self.S.add(LeakyReLU(0.2))
        self.S.add(Dense(512))
        self.S.add(LeakyReLU(0.2))

        return self.S

    def generator(self):
        print(" ------------- matchue generator start ------------- ")

        latent_size = self.latent_size
        cha = self.cha
        n_layers = self.n_layers
        im_size = self.im_size

        if self.G:
            return self.G

        # # === Style Mapping ===

        # self.S = Sequential()

        # self.S.add(Dense(512, input_shape = [latent_size]))
        # self.S.add(LeakyReLU(0.2))
        # self.S.add(Dense(512))
        # self.S.add(LeakyReLU(0.2))
        # self.S.add(Dense(512))
        # self.S.add(LeakyReLU(0.2))
        # self.S.add(Dense(512))
        # self.S.add(LeakyReLU(0.2))

        # === Generator ===

        #Inputs
        inp_style = []

        for i in range(n_layers):
            inp_style.append(Input([512]))

        inp_noise = Input([im_size, im_size, 1])

        #Latent
        x = Lambda(lambda x: x[:, :1] * 0 + 1)(inp_style[0])

        outs = []

        #Actual Model
        x = Dense(4*4*4*cha, activation = 'relu', kernel_initializer = 'random_normal')(x)
        x = Reshape([4, 4, 4*cha])(x)

        x, r = self.g_block(x, inp_style[0], inp_noise, 32 * cha, u = False)  #4
        outs.append(r)

        x, r = self.g_block(x, inp_style[1], inp_noise, 16 * cha)  #8
        outs.append(r)

        x, r = self.g_block(x, inp_style[2], inp_noise, 8 * cha)  #16
        outs.append(r)

        x, r = self.g_block(x, inp_style[3], inp_noise, 6 * cha)  #32
        outs.append(r)

        x, r = self.g_block(x, inp_style[4], inp_noise, 4 * cha)   #64
        outs.append(r)

        x, r = self.g_block(x, inp_style[5], inp_noise, 2 * cha)   #128
        outs.append(r)

        x, r = self.g_block(x, inp_style[6], inp_noise, 1 * cha)   #256
        outs.append(r)

        x = add(outs)

        x = Lambda(lambda y: y/2 + 0.5)(x) #Use values centered around 0, but normalize to [0, 1], providing better initialization

        self.G = Model(inputs = inp_style + [inp_noise], outputs = x)

        return self.G

    def GenModel(self):

        n_layers = self.n_layers
        im_size = self.im_size
        latent_size = self.latent_size

        # Generator Model for Evaluation
        if 0:
            print("GM")
            print("GM:n_layers", n_layers)
            print("GM:im_size", im_size)
            print("GM:latent_size", latent_size)

        inp_style = []
        style = []

        for i in range(n_layers):
            inp_style.append(Input([latent_size]))
            style.append(self.S(inp_style[-1])) # add S layers

        inp_noise = Input([im_size, im_size, 1])

        if 0:
            print("GM:shape style (added S layers)", np.shape(style)) # (7,)
            print("GM:shape style plus noise", np.shape(style + [inp_noise])) # (8,)

        if 0:
            self.G.summary()

        gf = self.G(style + [inp_noise])

        if 0:
            print("GM:shape style + noise", np.shape(inp_style + [inp_noise])) # style + noise (8,)
            # A None dimension in a shape tuple means that the network will be able to accept inputs of any dimension
            print("GM:shape generated style + noise", np.shape(gf)) # (None, 256, 256, 3)

        self.GM = Model(inputs = inp_style + [inp_noise], outputs = gf)

        return self.GM

    def GenModelA(self):

        n_layers = self.n_layers
        im_size = self.im_size
        latent_size = self.latent_size

        #Parameter Averaged Generator Model

        inp_style = []
        style = []

        for i in range(n_layers):
            inp_style.append(Input([latent_size]))
            style.append(self.SE(inp_style[-1]))

        inp_noise = Input([im_size, im_size, 1])

        gf = self.GE(style + [inp_noise])

        self.GMA = Model(inputs = inp_style + [inp_noise], outputs = gf)

        return self.GMA

    def EMA(self):
        #Parameter Averaging
        for i in range(len(self.G.layers)):
            up_weight = self.G.layers[i].get_weights()
            old_weight = self.GE.layers[i].get_weights()
            new_weight = []
            for j in range(len(up_weight)):
                new_weight.append(old_weight[j] * self.beta + (1-self.beta) * up_weight[j])
            self.GE.layers[i].set_weights(new_weight)

        for i in range(len(self.S.layers)):
            up_weight = self.S.layers[i].get_weights()
            old_weight = self.SE.layers[i].get_weights()
            new_weight = []
            for j in range(len(up_weight)):
                new_weight.append(old_weight[j] * self.beta + (1-self.beta) * up_weight[j])
            self.SE.layers[i].set_weights(new_weight)

    def MAinit(self):
        #Reset Parameter Averaging
        self.GE.set_weights(self.G.get_weights())
        self.SE.set_weights(self.S.get_weights())

    def get_batch(self, num, flip = True, verbose=False):

        npyfolder = self.dataset_dir

        # load self.images from npy if beyond limit
        if self.update > self.images.shape[0]:
            self.load_from_npy(npyfolder)
            # self.update = 0

        self.update = self.update + num

        idx = np.random.randint(0, self.images.shape[0] - 1, num)
        out = []

        for i in idx:
            out.append(self.images[i])
            if flip and np.random.uniform(0.0, 1.0) < 0.5:  # flip as arg
                out[-1] = np.flip(out[-1], 1)

        return np.array(out).astype('float32') / 255.0

    def fit(self):

        im_size = self.im_size
        latent_size = self.latent_size
        mixed_prob = self.mixed_prob
        noiseList = self.noiseList
        mixedList = self.mixedList
        nImage = self.nImage
        BATCH_SIZE = self.BATCH_SIZE

        reststeps = self.qsteps - self.steps

        self.load_from_npy(self.dataset_dir)

        n_cycle = self.n_cycle
        n_save = self.n_save
        n_ref = self.n_ref

        firststep = self.steps

        print(f"|===> fit:  \n \
            steps: {self.steps} \n \
            qsteps: {self.qsteps} \n \
            reststeps: {reststeps} \n \
            BATCH_SIZE: {BATCH_SIZE} \n \
            n_layers: {self.n_layers} \n \
            latent_size: {self.latent_size} \n \
            im_size: {self.im_size} \n \
            self.data_dir: {self.data_dir} \n \
            n_cycle: {n_cycle} \n \
            n_save: {n_save} \n \
            n_ref: {n_ref} \n \
            verbose: {self.verbose} \n \
        ")

        while self.steps < self.qsteps:

            #   train alternating - per layer
            styletype = ''
            if np.random.uniform(()) < mixed_prob:
                styletype = 'mixedList'
                style = mixedList(BATCH_SIZE)
            else:
                # style = np.random.normal(0.0, 1.0, size = [BATCH_SIZE, latent_size]).astype('float32')
                styletype = 'noiseList'
                style = noiseList(BATCH_SIZE)

            #   batch of real images
            images = self.get_batch(BATCH_SIZE, flip = True, verbose=True)

            #   
            noises = np.random.uniform(0.0, 1.0, size = [BATCH_SIZE, im_size, im_size, 1]).astype('float32')

            if self.steps - firststep < 6: # doc first steps

                print(f"|===> step: {self.steps - firststep}:  \n \
                    styletype: {styletype} \n \
                    style: {np.shape(style)} {type(style)} \n \
                    images: {np.shape(images)} {type(images)} \n \
                    noises: {np.shape(noises)} {type(noises)} \n \
                ")

            apply_gradient_penalty = self.steps % 2 == 0 or self.steps < 10000
            apply_path_penalty = self.steps % 16 == 0 # penalties every 16 steps

            # ===========
            # train step
            # d_loss, g_loss, divergence, pl_lengths
            a, b, c, d = self.train_step(images, style, noises, 
                        apply_gradient_penalty, 
                        apply_path_penalty)

            # ===========

            #Adjust path length penalty mean
            #d = pl_mean when no penalty is applied
            if self.pl_mean == 0:
                self.pl_mean = np.mean(d)
            self.pl_mean = 0.99*self.pl_mean + 0.01*np.mean(d)

            # GAN EMA if steps > 20000 mod 10
            if self.steps % 10 == 0 and self.steps > 20000:
                self.EMA() # Parameter Averaging

            # GAN MAinit if 
            if self.steps <= 25000 and self.steps % 1000 == 2:
                self.MAinit() # Reset Parameter Averaging

            if np.isnan(a):
                print("NaN Value Error.")
                exit()

            if self.steps % 100 == 0:
                if self.verbose:
                    print(f'|===> fit:  \n \
                        steps:  {str(self.steps)}\n \
                        steps to save model:  {n_save - (self.steps % n_save)}\n \
                        steps to increase cycle:  {n_cycle - (self.steps % n_cycle)}\n \
                        steps to evaluate:  {n_cycle - (self.steps % n_cycle)}\n \
                    ')     

            if self.steps % 100 == 0:
                if self.verbose:
                    print(f"|===> fit:  \n \
                        D loss:     {np.array(a)}\n \
                        G loss:     {np.array(b)}\n \
                        path length:{self.pl_mean}\n \
                    ")                       

                s = round((time.process_time() - self.lastblip), 4)
                self.lastblip = time.process_time()

                steps_per_second = 100 / s
                steps_per_minute = steps_per_second * 60
                steps_per_hour = steps_per_minute * 60

                min1k = floor(1000/steps_per_minute)
                sec1k = floor(1000/steps_per_second) % 60
                steps_left = self.qsteps - self.steps + 1e-7
                hours_left = steps_left // steps_per_hour
                minutes_left = (steps_left // steps_per_minute) % 60

                if self.verbose:
                    print(f'|===> fit:  \n \
                        StyleGan2 Steps/Second:  {str(round(steps_per_second, 2))}\n \
                        StyleGan2 1k Steps:  {str(min1k) + ":" + str(sec1k)} \n \
                        StyleGan2 Till Completion: {str(int(hours_left)) + "h" + str(int(minutes_left)) + "m"} \n \
                    ')       

                # save model
                if self.steps % n_save == 0:
                    _num = int(floor(self.steps / n_ref))
                    print(f"|===> fit:  \n \
                        step :  {self.steps}\n \
                        save model :  {_num}\n \
                    ")                          
                    self.save(_num)

                # increase cycle
                if self.steps % n_cycle == 0:
                    self.ncyc = floor(self.steps / n_cycle)
                    print(f"|===> fit:  \n \
                        self.steps:  {self.steps}\n \
                        n_cycle:  {n_cycle}\n \
                        StyleGan2.train cycle:  {self.ncyc}\n \
                    ")                     

                # Model Evaluate
                if self.steps % n_cycle == 0:
                    print(f"|===> fit:  \n \
                        evaluate step:  {self.steps}\n \
                    ")                        
                    self.evaluate(self.steps)

            onutil.printProgressBar(self.steps % 100, 99, decimals = 0, fill=">")

            self.steps = self.steps + 1

    @tf.function
    def train_step(self, real_images, style, noise, perform_gp = True, perform_pl = False):

        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:

            w_space = []
            pl_lengths = self.pl_mean   # _e_
            for i in range(len(style)):
                w_space.append(self.S(style[i]))

            fake_images = self.G(w_space + [noise])
            real_scores = self.D(real_images, training=True)
            fake_scores = self.D(fake_images, training=True)

            divergence = tf.math.softplus(fake_scores)  # log ( 1 + exp( D(fakes) ) ) # alt tf.math.relu
            divergence += tf.math.softplus(-real_scores) # log ( 1 - exp( D(reals) ) )  # alt tf.math.relu
            divergence = tf.keras.backend.mean(divergence)

            d_loss = divergence
            with tf.GradientTape() as p_tape:
                p_tape.watch(real_images)
                real_loss = tf.reduce_sum(self.D(real_images, training=True))

            if perform_gp: # gradient penalty - Penalize the gradient norm  
                real_grads = p_tape.gradient(real_loss, real_images)
                r1_penalty = tf.reduce_sum(tf.math.square(real_grads), # ||grad||^2
                    axis=np.arange(0, len(np.shape(real_grads)))) # [0, 1, 2, 3]
                self.r1_gamma = 10.0 # weight
                r1_penalty = r1_penalty * (0.5 * self.r1_gamma) # (weight / 2) * ||grad||^2
                d_loss += r1_penalty

            d_loss = tf.reduce_mean(d_loss)

            g_loss = tf.math.softplus(-fake_scores)

            if perform_pl: # path length penalty - slightly adjust W space
                w_space_2 = []
                for i in range(len(style)):
                    std = 0.1 / (tf.keras.backend.std(w_space[i], axis = 0, keepdims = True) + 1e-8)
                    w_space_2.append(w_space[i] + tf.keras.backend.random_normal(tf.shape(w_space[i])) / (std + 1e-8))

                # Generate from slightly adjusted W space
                pl_images = self.G(w_space_2 + [noise])

                # Get distance after adjustment (path length)
                delta_g = tf.keras.backend.mean(tf.keras.backend.square(pl_images - fake_images), axis = [1, 2, 3])
                pl_lengths = delta_g

                if self.pl_mean > 0:
                    g_loss += tf.keras.backend.mean(tf.keras.backend.square(pl_lengths - self.pl_mean))

            g_loss = tf.reduce_mean(g_loss)

        g_gradients = g_tape.gradient(g_loss, self.GM.trainable_variables)
        self.GMO.apply_gradients(zip(g_gradients, self.GM.trainable_variables))

        d_gradients = d_tape.gradient(d_loss, self.D.trainable_variables)
        self.DMO.apply_gradients(zip(d_gradients, self.D.trainable_variables))

        return d_loss, g_loss, divergence, pl_lengths

    def load_from_npy(self, folder):
        # for dirpath, dirnames, filenames in os.walk("data/" + folder + "-npy-" + str(self.im_size)):
        for dirpath, dirnames, filenames in os.walk(folder):
            for filename in [f for f in filenames if f.endswith(".npy")]:
                self.segments.append(os.path.join(dirpath, filename))
        self.load_segment()

    def load_segment(self):
        segment_num =np.random.randint(0, len(self.segments))
        self.images = np.load(self.segments[segment_num])
        self.update = 0

    #
    def evaluate(self, trunc = 1.0, outImage = True, num = None, qsegs = 1, frames=1, verbose=False):

        results_dir = self.results_dir
        latent_size = self.latent_size
        noise = self.noise      # np.random.normal(0.0, 1.0, size = [n, self.latent_size]).astype('float32')
        nImage = self.nImage    # np.random.uniform(0.0, 1.0, size = [n, self.im_size, self.im_size, 1]).astype('float32')
        BATCH_SIZE = self.BATCH_SIZE
        n_layers = self.n_layers
        noise = self.noise
        noiseList = self.noiseList
        im_size = self.im_size
        GM = self.GM
        GMA = self.GMA

        if (not num == None):
            ncyc = num
        elif (not self.ncyc == None):
            ncyc = self.ncyc
        else:
            ncyc =0

        qcells = qsegs * qsegs
        nnoisedata = [np.random.random(size = [qcells, latent_size]).astype('float32')] * (n_layers)
        nimagedata = np.random.uniform(low = 0, high = 1, size = [qcells, im_size, im_size, 1]).astype('float32')
        trunc = np.ones([qcells, 1]) * trunc

        # will save
        img_path = os.path.join(results_dir, "i{:04d}".format(int(ncyc)) + ".png")
        ema_path = os.path.join(results_dir, "i{:04d}".format(int(ncyc)) + "-ema.png")
        mr_path = os.path.join(results_dir,  "i{:04d}".format(int(ncyc)) + "-mr.png")

        # GM  predict
        seed = nnoisedata + [nimagedata] #
        if self.verbose:
            print(f"|===> evaluate:  \n \
                get samples: {qcells} * {latent_size} normal distribution, style map, average \n \
                nnoisedata:  {np.shape(nnoisedata)} {type(nnoisedata)}\n \
                [nimagedata]:  {np.shape([nimagedata])} {type([nimagedata])}\n \
                trunc:  {np.shape(trunc)} {type(trunc)}\n \
                save predict:  {img_path} \n \
                save ema:  {ema_path} \n \
                save mixed:  {mr_path} \n \
            ")                           

        generated_images = GM.predict(seed, batch_size = BATCH_SIZE)

        r = []

        for i in range(0, qcells, qsegs):
            r.append(np.concatenate(generated_images[i:i+qsegs], axis = 1))

        c1 = np.concatenate(r, axis = 0)
        c1 = np.clip(c1, 0.0, 1.0)
        x = Image.fromarray(np.uint8(c1*255))

        x.save(img_path)

        # GMA predict     
        generated_images = GMA.predict(nnoisedata + [nimagedata, trunc], batch_size = BATCH_SIZE)

        r = []

        for i in range(0, qcells, qsegs):
            r.append(np.concatenate(generated_images[i:i+qsegs], axis = 1))

        c1 = np.concatenate(r, axis = 0)
        c1 = np.clip(c1, 0.0, 1.0)

        x = Image.fromarray(np.uint8(c1*255))

        x.save(ema_path)

        #Mixing Regularities
        nn = noise(qsegs)
        nnoisedata = np.tile(nn, (qsegs, 1))
        nimagedata = np.repeat(nn, qsegs, axis = 0)
        tt = int(n_layers / 2)

        p1 = [nnoisedata] * tt
        p2 = [nimagedata] * (n_layers - tt)

        latent = p1 + [] + p2

        # GMA predict
        generated_images = GMA.predict(latent + [nImage(qcells), trunc], batch_size = BATCH_SIZE)

        r = []

        for i in range(0, qcells, qsegs):
            r.append(np.concatenate(generated_images[i:i+qsegs], axis = 0))

        c1 = np.concatenate(r, axis = 1)
        c1 = np.clip(c1, 0.0, 1.0)

        x = Image.fromarray(np.uint8(c1*255))
        x.save(mr_path)

        return x
    
    #
    def generateTruncated(self, trunc = 0.7, outImage = False, num = 0, qsegs=1):

        # n1 = model.noiseList(64)
        # n2 = model.nImage(64)
        # n1, noi = n2

        results_dir = self.results_dir
        BATCH_SIZE = self.BATCH_SIZE
        noise = self.noise
        noiseList = self.noiseList
        nImage = self.nImage
        GE = self.GE
        S = self.S # Style Mapping
        pl_mean = self.pl_mean # 0
        # av = self.av # np.zeros([44])
        ncyc = self.ncyc
        latent_size = self.latent_size
        n_layers = self.n_layers
        im_size = self.im_size
        qcells = qsegs * qsegs  # will predict images
            
        # Get W's (latents') center of mass
        dlatent_avg_start = np.random.normal(0.0, 1.0, size = [qcells, latent_size]).astype('float32')
        dlatent_avg_predict = S.predict(dlatent_avg_start, batch_size = BATCH_SIZE)
        dlatent_avg_mean = np.mean(dlatent_avg_predict, axis = 0)
        dlatent_avg = np.expand_dims(dlatent_avg_mean, axis = 0)

        vary = np.random.normal(0.0, 1.0, size = [qcells, latent_size]).astype('float32')

        dlatent_imgs = []
        styles = [vary] * (n_layers)
        for i in range(len(styles)):
            dlatent = S.predict(styles[i])
            dl = trunc * (dlatent - dlatent_avg) + dlatent_avg # trunc * dlatent
            dlatent_imgs.append(dl)

        noises = np.random.uniform(0.0, 1.0, size = [qcells, im_size, im_size, 1]).astype('float32')
        seeds = dlatent_imgs + [noises]
        z_images = GE.predict(seeds, batch_size = BATCH_SIZE)

        if self.verbose:
            print(f"|===> generateTruncated:  \n \
                [noises]: {np.shape([noises])} {type([noises])} \n \
                dlatent_imgs: {np.shape(dlatent_imgs)} {type(dlatent_imgs)} \n \
                trunc get {qcells} * {latent_size} normal distribution samples, style map, average\n \
            ")    

        if outImage:
            imgpath = os.path.join(results_dir, f"t{str(num)}.png")
            if self.verbose:
                print(f"|===> generateTruncated:  \n \
                    save:  {imgpath}\n \
                ")                   
            onformat.imgs_to_tiling(z_images, imgpath, qsegs)

        return z_images

    def generateWalk(self, trunc = 0.7, outImage = True, num = 0, qsegs=1, fps = 2, maxTime = 3, ):

        results_dir = self.results_dir
        ani_dir = self.ani_dir
        BATCH_SIZE = self.BATCH_SIZE
        noise = self.noise
        noiseList = self.noiseList
        nImage = self.nImage
        GE = self.GE
        S = self.S # Style Mapping
        pl_mean = self.pl_mean # 0
        # av = self.av # np.zeros([44])
        ncyc = self.ncyc
        latent_size = self.latent_size
        n_layers = self.n_layers
        im_size = self.im_size
        qcells = qsegs * qsegs  # will predict images

        # video settings
        frameCount = 0
        time = 0
        nframes = int( maxTime*fps )
        print(f"walk get {qcells} * {latent_size} normal distribution samples, style map, average")

        # Get W's (latents') center of mass
        dlatent_avg_start = np.random.normal(0.0, 1.0, size = [qcells, latent_size]).astype('float32')
        dlatent_avg_predict = S.predict(dlatent_avg_start, batch_size = BATCH_SIZE)
        dlatent_avg_mean = np.mean(dlatent_avg_predict, axis = 0)
        dlatent_avg = np.expand_dims(dlatent_avg_mean, axis = 0)

        seed_start = np.random.normal(1, 1, (qcells, latent_size))
        latentSpeed = np.random.normal(2, 1, (nframes, latent_size))
        varys = np.random.normal(1, 1, (nframes, qcells, latent_size))        

        noises = np.random.uniform(0.0, 1.0, size = [qcells, im_size, im_size, 1]).astype('float32')  # (1, 9, 512, 512, 1)

        print(f"|===> generateWalk:  \n \
            nframes: {nframes} \n \
            qcells: {qcells} \n \
            latent_size: {latent_size} \n \
            noises: {type([noises])}{np.shape([noises])} \n \
        ")    

        imgs = []
        for i in range(nframes): 
            time = i/nframes
            if 0:
                print(f"{i} -------------------------{time} {maxTime} {time/maxTime}")
            for k in range(qcells):
                
                for j in range(latent_size): # _e_

                    idelta = np.sin( 2*np.pi*(time/maxTime) * latentSpeed[i][j] ) 
                    if 0:
                        print(f"{i}:{j}:{k}: {seed_start[k][j]} - {time/maxTime} - {latentSpeed[i][j]} - {idelta}")
                    varys[i][k][j] = seed_start[k][j] + idelta
                    

            vary = varys[i] # <class 'numpy.ndarray'> (9, 512) (segs, ldims)

            dlatent_imgs = []  # (8, 9, 512)
            styles = [vary] * (n_layers)
            for n in range(len(styles)):
                dlatent = S.predict(styles[n])
                dl = trunc * (dlatent - dlatent_avg) + dlatent_avg # trunc * dlatent
                dlatent_imgs.append(dl)

            seeds = dlatent_imgs + [noises]
            z_images = GE.predict(seeds, batch_size = BATCH_SIZE) # (1, 512, 512, 3)

            if outImage:
                imgpath = os.path.join(ani_dir, f"frame{str(i)}.png")
                onformat.imgs_to_tiling(z_images, imgpath, qsegs)

            imgs.append(z_images)

        return imgs

    def getRecordNumber(self):

        import re

        models_dir = self.models_dir

        num = None
        nums = []
        num_re = re.compile(r'^gen_(\d+).h5$')

        if self.verbose:
            print("getRecordNumber in models_dir", models_dir)
        
        for file in os.listdir(models_dir):
            m = num_re.search(file)
            if 0:
                print("getRecordNumber", file, m)            
            if m:
                num = int(m.group(1))
                nums.append(num)     

        if nums:
            num = max(nums)

        return num  # None or int

    def saveModel(self, model, name, num):
        
        models_dir = self.models_dir

        json = model.to_json()
        with open(models_dir + "/"+name+".json", "w") as json_file:
            json_file.write(json)

        model.save_weights(models_dir + "/"+name+"_"+str(int(num))+".h5")

    def loadModel(self, name, num):
        
        models_dir = self.models_dir
        file_path = models_dir + "/"+name+".json"

        file = open(file_path, 'r')
        json = file.read()
        file.close()

        mod = model_from_json(json, custom_objects = {'Conv2DMod': Conv2DMod})
        weights_path = models_dir + "/"+name+"_"+str(num)+".h5"
        if 0:
            print(f"loadModel weights_path: {weights_path}")
        mod.load_weights(weights_path)
        return mod

    def save(self, num): #Save JSON and Weights into /Models/

        if self.verbose:
            print(f"StyleGan.save num {num}")

        self.saveModel(self.S, "sty", num)
        self.saveModel(self.G, "gen", num)
        self.saveModel(self.D, "dis", num)

        self.saveModel(self.GE, "genMA", num)
        self.saveModel(self.SE, "styMA", num)

    def load(self, num): #Load JSON and Weights from /Models/

        if self.verbose:
            print("StyleGan.load num %d" %num)

        #Load Models
        self.D = self.loadModel("dis", num)
        self.S = self.loadModel("sty", num)
        self.G = self.loadModel("gen", num)

        self.GE = self.loadModel("genMA", num)
        self.SE = self.loadModel("styMA", num)

        self.GenModel()
        self.GenModelA()
#
#
#   CMDS
#
#
#
#   nnviv
#
def nnviv(args, kwargs):

    args = onutil.pargs(vars(args))

    args.PROJECT = 'StyleGan2-Tensorflow-2.0'
    args.DATASET = 'space'
    xp = getxp(vars(args))
    args = onutil.pargs(xp)

    onutil.ddict(vars(args), 'args')

    if 1: # git
        onutil.get_git(args.AUTHOR, args.GITPOD, args.proj_dir)

    if 1: # tree
        args.data_dir = os.path.join(args.proj_dir, "data")
        args.models_dir = os.path.join(args.proj_dir, "Models")
        args.results_dir = os.path.join(args.proj_dir, "Results")

        os.makedirs(args.data_dir, exist_ok=True)
        os.makedirs(args.models_dir, exist_ok=True)
        os.makedirs(args.results_dir, exist_ok=True)

        args.dataorg_dir=os.path.join(args.gdata, "spaceone")
        args.data_dir = os.path.join(args.data_dir, 'Earth')
        os.makedirs(args.data_dir, exist_ok=True)    

    print(f"|===> nnviv:  \n \
        cwd:  {os.getcwd()}, \n \
        args.dataorg_dir: {args.dataorg_dir}, \n \
        args.results_dir: {args.results_dir}, \n \
        args.models_dir:  {args.models_dir}, \n \
        args.data_dir:    {args.data_dir}, \n \
        ")

    if 1: # copy models
        earth_model_src_dir = os.path.join(args.gdata, "LandscapesBig")
        onfile.copyfolder(earth_model_src_dir, args.models_dir)

    if 0: # copy images
        clip = (256, 256)
        imgs = ondata.folder_to_formed_pils(args.dataorg_dir, clip = clip, qslices = 10, eps = 0.5)
        onfile.pils_to_folder(imgs, args.data_dir)

    if 1: # run script
        cmd = f'python stylegan_two.py'
        print(f'cmd: {cmd}')
        os.system(cmd)
#
#
#   nnearth
#
def nnearth(args, kwargs):

    args = onutil.pargs(vars(args))
    onutil.ddict(vars(args), 'args')

    args.models_dir = os.path.join(args.gmodel, 'LandscapesBig')

    clip = (256, 256)
    latent_size = 512
    im_size = 256
    BATCH_SIZE = 8
    mixed_prob = 0.9
    qsteps = 20001

    n_cycle = 500  # will evaluate model each
    n_save = int(n_cycle / 1)    # will save model each
    n_ref = n_cycle   # will identify model each

    print(f"|===> nnearth:  \n \
                cwd: {os.getcwd()}, \n \
                results_dir: {args.results_dir}, \n \
                models_dir:  {args.models_dir}, \n \
                dataorg_dir: {args.dataorg_dir}, \n \
                data_dir:    {args.data_dir}, \n \
                dataset_dir: {args.dataset_dir}, \n \
                latent_size: {latent_size}, \n \
                im_size: {im_size}, \n \
                BATCH_SIZE: {BATCH_SIZE}, \n \
                mixed_prob: {mixed_prob}, \n \
                qsteps: {qsteps}, \n \
            \n ")

    model = StyleGan2(lr = 0.0001, silent = False,

        latent_size = latent_size,
        im_size = im_size,
        BATCH_SIZE = BATCH_SIZE, # 16
        mixed_prob = mixed_prob,
        qsteps = qsteps,   #   up to step

        n_cycle = n_cycle,
        n_save = n_save,
        n_ref = n_ref,

        results_dir = args.results_dir,
        models_dir = args.models_dir,
        dataorg_dir = args.dataorg_dir,
        data_dir = args.data_dir,
        dataset_dir = args.dataset_dir,
    )

    if 1:
        model.evaluate(trunc = 0.7, qsegs = 1)
#
#   nndata
#
def nndata(args, kwargs):

    args = onutil.pargs(vars(args))

    args.PROJECT = 'space'
    args.DATASET = 'space'
    xp = getxp(vars(args))
    args = onutil.pargs(xp)

    onutil.ddict(vars(args), 'args')

    print(f"|===> nndata \n \
                cwd: {os.getcwd()} \n \
        \n ")

    zfill = 4

    tile = (int(512*2), int(512*2))
    clip = (512, 512)
    im_size = 512

    latent_size = 512
    BATCH_SIZE = 8
    mixed_prob = 0.9
    qsteps = 200001

    if 1: # clear code folder
        onfile.clearfolder(args.proj_dir, inkey=args.PROJECT)

    if 0: # clear domain folder
        onfile.clearfolder(args.proto_dir, inkey=args.MNAME)

    if 1: # tree
        args.dataorg_dir=os.path.join(args.dataorg_dir, "")

        # args.data_dir=os.path.join(args.gdata, args.DATASET + '_' + str(clip[0]) + '_' + str(clip[1]))   
        args.dataset_dir=os.path.join(args.proj_dir, "dataset")   
        args.models_dir=os.path.join(args.proj_dir, "Models")   
        args.results_dir=os.path.join(args.proj_dir, "Results")  

        tmp1_dir =  os.path.join(args.proto_dir, "tmp1")  # named
        tmp2_dir =  os.path.join(args.proto_dir, "tmp2")  # formed
        tmp3_dir =  os.path.join(args.proto_dir, "tmp3")  # dedup

        os.makedirs(args.data_dir, exist_ok=True) # deduped formed
        os.makedirs(args.dataset_dir, exist_ok=True) # npy folder
        os.makedirs(args.models_dir, exist_ok=True)
        os.makedirs(args.results_dir, exist_ok=True)

        os.makedirs(args.tmp_dir, exist_ok=True)

        os.makedirs(tmp1_dir, exist_ok=True)
        os.makedirs(tmp2_dir, exist_ok=True)
        os.makedirs(tmp3_dir, exist_ok=True)

    print(f"|===> nndata \n \
                dataorg (raw) => tmp (raw) => data (formed) => dataset (npy) \n \
                cwd:  {os.getcwd()}, \n \
                args.dataorg_dir: {args.dataorg_dir} \n \
                args.data_dir: {args.data_dir} \n \
                args.dataset_dir: {args.dataset_dir} \n \
                args.models_dir: {args.models_dir} \n \
                args.results_dir: {args.results_dir} \n \
        \n ")

    print(f"|===> nndata:  \n \
                clip: {clip} \n \
                latent_size: {latent_size} \n \
                im_size: {im_size} \n \
                BATCH_SIZE: {BATCH_SIZE} \n \
                mixed_prob: {mixed_prob} \n \
                qsteps: {qsteps} \n \
        \n ")

    if 0: #     copy models tp models_dir
        print(f"copy models from {earth_model_src_dir} to {args.models_dir}")
        earth_model_src_dir = os.path.join(args.gdata, "LandscapesBig")
        onfile.copyfolder(earth_model_src_dir, args.models_dir)

    if 0: #     plot dataorg sample
        imgs = onfile.folder_to_pils(args.dataorg_dir, n=1)
        img = imgs[0]
        print(f"img {type(img)} {np.shape(img)}")
        onplot.pil_show_pil(img)

    if 0: #     dataorg to tmp1_dir (raws to named)
        print(f"nndata {onfile.qfiles(args.dataorg_dir)} raws in {args.dataorg_dir}")
        onfile.folder_to_named_files(args.dataorg_dir, tmp1_dir, n=-1) # _e_
        # onfile.folder_to_named_files(args.dataorg_dir, tmp1_dir, n=18) # _e_

    if 1: #     tmp1_dir to tmp2_dir (raws  to  formed)
        
        paths = onfile.folder_to_paths(tmp1_dir)
        print(f"{len(paths)} raws in {tmp1_dir} to {tmp2_dir}")

        if 1:  # get form imgs

            for ipath,imgpath in enumerate(paths):

                basename = os.path.basename(imgpath)
                name = os.path.splitext(basename)[0]
                ext = os.path.splitext(basename)[1]
                namecode = name[len("img"):]
                if 0:
                    print(f"nndata imgpath: {imgpath}  {basename} {name} {ext} {namecode}")

                newidx=0
                with Image.open(imgpath) as img:

                    if 1: # scale
                        img = ondata.pil_scale(img, 1)

                    if 1: # tile
                        _newimgs = ondata.img_to_tiled_pils(img, tile = tile)
                        maxtiles = 9
                        if len(_newimgs) > maxtiles:
                            ids = np.random.choice(len(_newimgs), maxtiles)
                            selectimgs = []
                            for i in ids:
                                selectimgs.append(_newimgs[i])
        
                            print(f"{ipath}/{len(paths)}: tiles found {len(selectimgs)} in {np.shape(img)} {imgpath}. add {len(selectimgs[1:])}")
                            for _img in selectimgs:
                                _newimg = ondata.pil_resize(_img, ps = clip)

                                newpath = os.path.join( tmp2_dir, name + str(newidx).zfill(zfill) + ".jpg")
                                _newimg.save(newpath, 'JPEG', quality=90)                                 
                                newidx += 1
                        else:
        
                            if (len(_newimgs)) > 0:
                                print(f"{ipath}/{len(paths)}: tiles found {len(_newimgs)} in {np.shape(img)} {imgpath}. add {len(_newimgs[1:])}")
                                for _img in _newimgs:
                                    _newimg = ondata.pil_resize(_img, ps = clip)

                                    newpath = os.path.join( tmp2_dir, name + str(newidx).zfill(zfill) + ".jpg")
                                    _newimg.save(newpath, 'JPEG', quality=90)                                 
                                    newidx += 1              
                            
                    if 1: # resize
                        _newimg = ondata.pil_resize(img, ps = clip)

                        newpath = os.path.join( tmp2_dir, name + str(newidx).zfill(zfill) + ".jpg")
                        _newimg.save(newpath, 'JPEG', quality=90)                                 
                        newidx += 1    

                    if 1: # purge file from data_dir until empty
                        os.remove(imgpath)

    if 1: #     tmp2_dir => dedup => tmp3_dir / excluded
            #   tmp2_dir =>       => data_dir
        pairs = []            
        args.process_type = 'exclude'
        args.file_extension = 'jpg'
        args.avg_match = 1.0
        args.absolute = False
        args.output_folder = tmp2_dir # will save dups to args.output_folder/excluded
        pairs = onfile.folder_to_cv_name_pairs(tmp2_dir, args) # get from 

        deduppairs = onset.exclude(pairs, args) 
        print(f"nndata: got {len(deduppairs)} dedups")

        if not len(deduppairs) == onfile.qfiles(args.data_dir):
            args.output_folder = args.data_dir  # imgs => output_folder (data_dir)
            print(f"nndata: will save undups to {args.output_folder}")
            onfile.cv_name_pairs_to_folder(deduppairs, args)

    if 1: #     data folder  ==>  dataset npy
        onrecord.folder_to_npy(args.data_dir, args.dataset_dir, im_size = clip[0], mss = (1024 ** 3))
        imgs = onrecord.npys_folder_to_rgbs(args.dataset_dir)
        print(f"nndata: got imgs: {np.shape(imgs)}")
#
#   nntrain
#
def nntrain(args, kwargs):

    args = onutil.pargs(vars(args))

    args.PROJECT = 'space'
    args.DATASET = 'space'
    
    xp = getxp(vars(args))
    args = onutil.pargs(xp)

    onutil.ddict(vars(args), 'args')

    print(f"|===> nntrain \n \
                cwd: {os.getcwd()} \n \
        \n ")

    zfill = 4

    tile = (int(512*2), int(512*2))
    clip = (512, 512)
    qslices = 4
    eps = 0.9

    latent_size = 512
    im_size = 512
    BATCH_SIZE = 4
    mixed_prob = 0.9
    qsteps = 200001

    n_cycle = 500               # will evaluate model each
    n_save = int(n_cycle / 1)   # will save model each
    n_ref = n_cycle             # will identify model each

    if 1: # tree
        args.dataorg_dir=os.path.join(args.dataorg_dir, "")

        # args.data_dir=os.path.join(args.gdata, args.DATASET + '_' + str(clip[0]) + '_' + str(clip[1]))   
        args.dataset_dir=os.path.join(args.proj_dir, "dataset")   
        args.models_dir=os.path.join(args.proj_dir, "Models")   
        args.results_dir=os.path.join(args.proj_dir, "Results")   

        os.makedirs(args.data_dir, exist_ok=True) # deduped formed
        os.makedirs(args.dataset_dir, exist_ok=True) # npy folder
        os.makedirs(args.models_dir, exist_ok=True)
        os.makedirs(args.results_dir, exist_ok=True)
        os.makedirs(args.tmp_dir, exist_ok=True) # name raws

    print(f"|===> nntrain \n \
                dataorg (raw) => tmp (raw) => data (formed) => dataset (npy) \n \
                cwd:  {os.getcwd()}, \n \
                args.dataorg_dir: {args.dataorg_dir} \n \
                args.data_dir: {args.data_dir} \n \
                args.dataset_dir: {args.dataset_dir} \n \
                args.models_dir: {args.models_dir} \n \
                args.results_dir: {args.results_dir} \n \
        \n ")

    print(f"|===> nntrain:  \n \
                clip: {clip} \n \
                latent_size: {latent_size} \n \
                im_size: {im_size} \n \
                BATCH_SIZE: {BATCH_SIZE} \n \
                mixed_prob: {mixed_prob} \n \
                qsteps: {qsteps} \n \
                n_cycle: {n_cycle} \n \
                n_save: {n_save} \n \
                n_ref: {n_ref} \n \
        \n ")

    if 1: #     get model

        model = StyleGan2(lr = 0.0001, silent = False,

            latent_size = latent_size,
            im_size = im_size,
            BATCH_SIZE = BATCH_SIZE,
            mixed_prob = mixed_prob,
            qsteps = qsteps,

            n_cycle = n_cycle,
            n_save = n_save,
            n_ref = n_ref,

            name = args.MNAME,
            results_dir = args.results_dir,
            models_dir = args.models_dir,
            dataorg_dir = args.dataorg_dir,
            data_dir = args.data_dir,
            dataset_dir = args.dataset_dir,
        )

    if 1: # train
        model.fit()

    if 1: # evaluate
        img = model.evaluate()
        onplot.pil_show_pil(img)
#
#   nntrunc
#
def nntrunc(args, kwargs):

    args = onutil.pargs(vars(args))
    onutil.ddict(vars(args), 'args')

    print(f"|===> nntrunc:  \n \
    ")

    latent_size = 512
    im_size = 512
    BATCH_SIZE = 4
    mixed_prob = 0.9
    qsteps = 200001

    n_cycle = 500               # will evaluate model each
    n_save = int(n_cycle / 1)   # will save model each
    n_ref = n_cycle             # will identify model each

    if 1: # tree
        args.dataorg_dir=os.path.join(args.dataorg_dir, "")

        # args.data_dir=os.path.join(args.gdata, args.DATASET + '_' + str(clip[0]) + '_' + str(clip[1]))   
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

    if 1: #     get model

        model = StyleGan2(lr = 0.0001, silent = False,

            latent_size = latent_size,
            im_size = im_size,
            BATCH_SIZE = BATCH_SIZE,
            mixed_prob = mixed_prob,
            qsteps = qsteps,

            n_cycle = n_cycle,
            n_save = n_save,
            n_ref = n_ref,

            name = args.MNAME,
            results_dir = args.results_dir,
            models_dir = args.models_dir,
            dataorg_dir = args.dataorg_dir,
            data_dir = args.data_dir,
            dataset_dir = args.dataset_dir,
            ani_dir = args.ani_dir,
            verbose = True,
        )

    if 1:  # truncate
        model.generateTruncated(trunc = 0.5, outImage = True, num = 0, qsegs=1)

    if 1:  # frames truncate
        fps = 6 # 30
        maxTime = 5 # 30 # seconds
        nframes = int( maxTime*fps )               
        for i in range(nframes):
            imgs = model.generateTruncated(trunc = i / 50, outImage = True, num = i, qsegs=2)

    if 1:  # ani create
        onvid.folder_to_vid(args.results_dir, os.path.join(args.results_dir, "ani.gif"))

    if 1:  # ani show
        onvid.vid_show(os.path.join(args.results_dir, "ani.gif"))
#
#   nnwalk
#
def nnwalk(args, kwargs):

    args = onutil.pargs(vars(args))
    onutil.ddict(vars(args), 'args')

    print(f"|===> nnwalk:  \n \
    ")

    latent_size = 512
    im_size = 512
    BATCH_SIZE = 4
    mixed_prob = 0.9
    qsteps = 200001

    n_cycle = 500               # will evaluate model each
    n_save = int(n_cycle / 1)   # will save model each
    n_ref = n_cycle             # will identify model each

    if 1: # tree
        args.dataorg_dir=os.path.join(args.dataorg_dir, "")

        # args.data_dir=os.path.join(args.gdata, args.DATASET + '_' + str(clip[0]) + '_' + str(clip[1]))   
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

    if 1: #     get model

        model = StyleGan2(lr = 0.0001, silent = False,

            latent_size = latent_size,
            im_size = im_size,
            BATCH_SIZE = BATCH_SIZE,
            mixed_prob = mixed_prob,
            qsteps = qsteps,

            n_cycle = n_cycle,
            n_save = n_save,
            n_ref = n_ref,

            name = args.MNAME,
            results_dir = args.results_dir,
            models_dir = args.models_dir,
            dataorg_dir = args.dataorg_dir,
            data_dir = args.data_dir,
            dataset_dir = args.dataset_dir,
            ani_dir = args.ani_dir,
            verbose = True,
        )
        
    if 1: # create imgs
        print(f"nnwalk create imgs in {args.ani_dir}")
        imgs = model.generateWalk(qsegs=1, fps=20,maxTime=6,)
        print(f"nnwalk got {len(imgs)} imgs")

    if 1: # create gif
        dstpath = os.path.join(args.ani_dir, 'latent.gif')
        ext='png'
        onvid.folder_to_gif(args.ani_dir, dstpath, ext)
        onvid.vid_show(dstpath)

    # # creating random latent vector
    # seed = 96
    # rnd = np.random.RandomState(seed)
    # z = rnd.randn(4, 512).astype('float32')
    # print(f"z: {np.shape(z)}")

    # # # running network
    # # # out = generator([z, None]) # _e_
    # # out = model.G(z) # _e_
    # # print(f"out: {np.shape(out)}")

    # #converting image to uint8
    # out_image = onrosa.convert_images_to_uint8(out, nchw_to_nhwc=True, uint8_cast=True)

    # onfile.generate_and_save_images(out_image.numpy(), 0)

    # n_images = 500

    # for i in tqdm(range(n_images)):
    #     onfile.generate_and_save_images(out_image.numpy(), i, plot_fig=False)
        
    #     #moving randomly in the latent space z
    #     seed = i
    #     rnd = np.random.RandomState(seed)
        
    #     #mofying slightly latent vector and generating new images
    #     z += rnd.randn(4, 512).astype('float32') / 40
    #     out = generator([z, None]) # _e_
    #     out_image = onrosa.convert_images_to_uint8(out, nchw_to_nhwc=True, uint8_cast=True)
#
#   nnproj
#
def nnproj(args, kwargs):

    args = onutil.pargs(vars(args))

    args.PROJECT = 'ffhq'
    args.DATASET = 'ffhq'
    
    xp = getxp(vars(args))
    args = onutil.pargs(xp)

    onutil.ddict(vars(args), 'args')

    print(f"|===> nnproj\n \
                cwd: {os.getcwd()} \n \
        \n ")

    if 1: # tree
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

    if 1: #     get model
        latent_size = 512
        im_size = 512
        BATCH_SIZE = 4
        mixed_prob = 0.9
        qsteps = 200001

        n_cycle = 500               # will evaluate model each
        n_save = int(n_cycle / 1)   # will save model each
        n_ref = n_cycle             # will identify model each

        model = StyleGan2(lr = 0.0001, silent = False,

            latent_size = latent_size,
            im_size = im_size,
            BATCH_SIZE = BATCH_SIZE,
            mixed_prob = mixed_prob,
            qsteps = qsteps,

            n_cycle = n_cycle,
            n_save = n_save,
            n_ref = n_ref,

            name = args.MNAME,
            results_dir = args.results_dir,
            models_dir = args.models_dir,
            dataorg_dir = args.dataorg_dir,
            data_dir = args.data_dir,
            dataset_dir = args.dataset_dir,
        )

    if 1:
        dataset = DatasetLoader(
            path_dir = args.data_dir, 
            resolution = 1024, 
            batch_size = args.batch_size, 
            cache_file=True
        )
        for i in range(2):
            print(i)
            imgs = dataset.get_batch() # (4, 3, 256, 256)
            print(f"{i} {np.shape(imgs)}")

    #   stylegan2-ffhq-config-f.pkl	StyleGAN2 for FFHQ dataset at 1024×1024
    #   vgg16_zhang_perceptual.pkl	Standard LPIPS metric to estimate perceptual similarity.

    # print('Projecting image "%s"...' % os.path.basename(src_file))
    # images, _labels = dataset_obj.get_minibatch_np(1)

    prj = Projector(
        verbose=True,
        im_size=512,
    )
    prj._info()
    prj.set_network(model.G)
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
