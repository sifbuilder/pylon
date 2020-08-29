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
        
from functools import partial
from importlib import import_module

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import numpy as np
from numpy import *

import math
from math import floor, log2
from random import random
from pylab import *

# import IPython.display as display
from IPython.core.display import display

import PIL
from PIL import Image
PIL.Image.MAX_IMAGE_PIXELS = 933120000

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

import shutil
import gdown

import sys


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

from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import add
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.models import clone_model
from tensorflow.keras.models import model_from_json

tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print(f'|---> {tf.__version__}')

#   ******************
#   FUNS
#
#   ******************

try:
    # check if base.Onpyon is defined
    var = Onpyon()

except NameError:
    # Onpyon not defined

    sys.path.append('../')  # if called from eon, modules are in parallel folder
    sys.path.append('./')  #  if called from dnns, modules are in folder

    from base import *

onutil = Onutil()
onplot = Onplot()
onformat = Onformat()
onfile = Onfile()
onvid = Onvid()
ondata = Ondata()
onset = Onset()
onrecord = Onrecord()
ontree = Ontree()
onlllyas = Onlllyas()

#   *******************
#   CONTEXT
#
#   *******************

def getap():
    cp = {
        "primecmd": 'nndanboo', # 'nncrys', #  

        "MNAME": "pix2pix",
        "AUTHOR": "tensorflow2",
        "PROJECT": "facades",   # python pix2pix/pix2pix.py --PROJECT=facades --DATASET=facades nnfacades
        "GITPOD": "facades",
        "DATASET": "facades",

        "GDRIVE": 1,            # mount gdrive: gdata, gwork    
        "TRAINDO": 1,      
        "MAKEGIF": 1,      
        "RUNGIF": 0,      
        "CLEARTMP": 0,      
        "REGET": 0,             # get again data 
        "ING": 1,               # ckpt in gwork
        "MODITEM": "",          # will look into module
        "RESETCODE": 0,
        "LOCALDATA": 0,
        "LOCALMODELS": 0,
        "LOCALLAB": 1,
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
        "verbose": 1, # [0,n]
        "visual": 1, # [0,n]

        # train
        "batch_size": 1,
        "img_width": 256,
        "img_height": 256,
        "buffer_size": 1000,
        "input_channels": 3,
        "output_channels": 3,
        "max_epochs": 200,
        "n_iterations": 10, # iters for snapshot

        # dataset.py args
        "input_folder": './input/',
        "output_folder": './output/',
        "keep_folder": 0,
        "process_type": 'resize',
        "blur_type": "", # ["","gaussian","median"]
        "blur_amount": 1,
        "max_size": 256,
        "height": 256,
        "width": 256,
        "shift_y": 0,
        "v_align": 'center',
        "h_align": 'center',
        "shift_x": 0,
        "scale": 2.0,
        "direction": 'AtoB',
        "border_type": 'stretch',
        "border_color": '255,255,255',
        "mirror": 0,
        "rotate": 0,
        "file_extension": 'png',

        # var
        "name": 0, # use counter
        "keep_name": 0, # _e_
        "numbered": 1, # _e_
        "zfill": 4, # zfill name counter

    }

    ap = {}

    for key in cp.keys():
        ap[key] = cp[key]
    for key in hp.keys():
        ap[key] = hp[key]
    return ap


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


#   ******************
#   FUNS SEGMENT
#
#   https://github.com/lllyasviel/DanbooRegion/blob/master/code/segment.py

def go_vector(x):
    return x[None, :, :, :]


def go_flipped_vector(x):
    a = go_vector(x)
    b = np.fliplr(go_vector(np.fliplr(x))) # numpy.fliplr(m: marray_like) -> fndarray
    c = np.flipud(go_vector(np.flipud(x)))
    d = np.flipud(np.fliplr(go_vector(np.flipud(np.fliplr(x)))))
    return (a + b + c + d) / 4.0


def go_transposed_vector(x):
    a = go_flipped_vector(x)
    b = np.transpose(go_flipped_vector(np.transpose(x, [1, 0, 2])), [1, 0, 2])
    return (a + b) / 2.0


def get_fill(image):
    labeled_array, num_features = label(image / 255)
    filled_area = onlllyas.find_all(labeled_array)
    return filled_area


def up_fill(fills, cur_fill_map):
    new_fillmap = cur_fill_map.copy()
    padded_fillmap = np.pad(cur_fill_map, [[1, 1], [1, 1]], 'constant', constant_values=0)
    max_id = np.max(cur_fill_map)
    for item in fills:
        points0 = padded_fillmap[(item[0] + 1, item[1] + 0)]
        points1 = padded_fillmap[(item[0] + 1, item[1] + 2)]
        points2 = padded_fillmap[(item[0] + 0, item[1] + 1)]
        points3 = padded_fillmap[(item[0] + 2, item[1] + 1)]
        all_points = np.concatenate([points0, points1, points2, points3], axis=0)
        pointsets, pointcounts = np.unique(all_points[all_points > 0], return_counts=True)
        if len(pointsets) == 1 and item[0].shape[0] < 128:
            new_fillmap[item] = pointsets[0]
        else:
            max_id += 1
            new_fillmap[item] = max_id
    return new_fillmap


#   ******************
#   NETS
#
#   ******************

class GAN(object):

    def __init__(self, 
    
            models_dir = './',
            logs_dir = './',
            ckptidx = None,
            ckpt_dir = './',
            ckpt_prefix = 'ckpt-',
            results_dir = './results',
            input_shape = [256,256,3],
            output_shape = [256,256,3],

    ):
        print(f'|---> sg2pix2pix.GAN')

        self.input_shape = input_shape
        self.output_shape = output_shape

        self.models_dir = models_dir
        self.logs_dir = logs_dir
        self.results_dir = results_dir
        self.ckptidx = ckptidx
        self.ckpt_dir = ckpt_dir
        self.ckpt_prefix = ckpt_prefix

        self.generator = self.Generator(input_shape=input_shape)
        self.discriminator = self.Discriminator(output_shape=output_shape)   

        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        self.step = tf.Variable(0)

        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.generator,
            discriminator=self.discriminator,
            step=self.step
        )



        self.restore_checkpoint()

    # Build the Generator
    #     The architecture of generator is a modified U-Net.
    #     Each block in the encoder is (Conv -> Batchnorm -> Leaky ReLU)
    #     Each block in the decoder is (Transposed Conv -> Batchnorm -> Dropout(applied to the first 3 blocks) -> ReLU)
    #     There are skip connections between the encoder and decoder (as in U-Net)

    def Generator(self, input_shape=[256,256,3]):
        print(f'|---> Generator instance with input_shape: {input_shape}')
        inputs = tf.keras.layers.Input(input_shape)

        down_stack = [
            ondata.downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
            ondata.downsample(128, 4), # (bs, 64, 64, 128)
            ondata.downsample(256, 4), # (bs, 32, 32, 256)
            ondata.downsample(512, 4), # (bs, 16, 16, 512)
            ondata.downsample(512, 4), # (bs, 8, 8, 512)
            ondata.downsample(512, 4), # (bs, 4, 4, 512)
            ondata.downsample(512, 4), # (bs, 2, 2, 512)
            ondata.downsample(512, 4), # (bs, 1, 1, 512)
        ]

        up_stack = [
            ondata.upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
            ondata.upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
            ondata.upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
            ondata.upsample(512, 4), # (bs, 16, 16, 1024)
            ondata.upsample(256, 4), # (bs, 32, 32, 512)
            ondata.upsample(128, 4), # (bs, 64, 64, 256)
            ondata.upsample(64, 4), # (bs, 128, 128, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(
            3, # output_channels, # filters
            4,  # kernel_size
            strides=2,
            padding='same',
            kernel_initializer=initializer,
            activation='tanh') # (bs, 256, 256, 3)

        # concat = tf.keras.layers.Concatenate()

        # inputs = tf.keras.layers.Input(input_shape=[None, None, 3])
        x = inputs

        # Downsampling through the model
        skips = []
        for down in down_stack:
                x = down(x)
                skips.append(x)

        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
                x = up(x)
                x = tf.keras.layers.Concatenate()([x, skip])

        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x)


    # Discriminator
    #     The Discriminator is a PatchGAN.
    #     Each block in the discriminator is (Conv -> BatchNorm -> Leaky ReLU)
    #     The shape of the output after the last layer is (batch_size, 30, 30, 1)
    #     Each 30x30 patch of the output classifies a 70x70 portion of the input image (such an architecture is called a PatchGAN).
    #     Discriminator receives 2 inputs.
    #         Input image and the target image, which it should classify as real.
    #         Input image and the generated image (output of generator), which it should classify as fake.
    #         We concatenate these 2 inputs together in the code (tf.concat([inp, tar], axis=-1))
    def Discriminator(self, output_shape):
            initializer = tf.random_normal_initializer(0., 0.02)

            inp = tf.keras.layers.Input(shape=output_shape, name='output_image')
            tar = tf.keras.layers.Input(shape=output_shape, name='target_image')

            x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, input_channels*2)

            down1 = ondata.downsample(64, 4, False)(x) # (bs, 128, 128, 64)
            down2 = ondata.downsample(128, 4)(down1) # (bs, 64, 64, 128)
            down3 = ondata.downsample(256, 4)(down2) # (bs, 32, 32, 256)

            zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
            conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                            kernel_initializer=initializer,
                            use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

            batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
            leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
            zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)
            last = tf.keras.layers.Conv2D(1, 4, strides=1,
                            kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

            return tf.keras.Model(inputs=[inp, tar], outputs=last)

    # discriminator loss
    #     The discriminator loss function takes 2 inputs; real images, generated images
    #     real_loss is a sigmoid cross entropy loss of the real images and an array of ones(since these are the real images)
    #     generated_loss is a sigmoid cross entropy loss of the generated images and an array of zeros(since these are the fake images)
    #     Then the total_loss is the sum of real_loss and the generated_loss

    def discriminator_loss(self, disc_real_output, disc_generated_output,
            loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    ):
        real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss

    # generator loss
    #     It is a sigmoid cross entropy loss of the generated images and an array of ones.
    #     The paper also includes L1 loss which is MAE (mean absolute error) between the generated image and the target image.
    #     This allows the generated image to become structurally similar to the target image.
    #     The formula to calculate the total generator loss = gan_loss + LAMBDA * l1_loss, where LAMBDA = 100. This value was decided by the authors of the paper.
    def generator_loss(self, disc_generated_output, gen_output, target, lda = 100, 
            loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    ):
        LAMBDA = lda # LAMBDA = 100

        gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

        # mean absolute error
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

        total_gen_loss = gan_loss + (LAMBDA * l1_loss)

        return total_gen_loss, gan_loss, l1_loss
            
    # @tf.function
    def train_step(self, input_image, target, epoch, summary_writer, args=None):

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)

            disc_real_output = self.discriminator([input_image, target], training=True)
            disc_generated_output = self.discriminator([input_image, gen_output], training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator_loss(disc_generated_output, gen_output, target)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss, self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_variables))

        with summary_writer.as_default():
            tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
            tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
            tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
            tf.summary.scalar('disc_loss', disc_loss, step=epoch)    


    def fit(self, train_ds, test_ds=None, args = None, ):

        max_epochs = args.max_epochs
        n_iterations  = args.n_iterations
        print(f'|---> fit \n \
            max_epochs: {max_epochs} \n \
            n_iterations: {n_iterations} \n \
            test_ds: {1 if test_ds else 0} \n \
        ')

        # tensorboard
        summary_writer = tf.summary.create_file_writer(
            os.path.join(self.logs_dir, "fit" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

        for epoch in range(max_epochs):
            start = time.time()

            # 	STEP per batches in dataset
            for n, (input_image, target) in train_ds.enumerate():
                ''' {np.shape(input_image)}" (1, 512, 512, 3) '''
                print('.', end='')
                if (n+1) % 100 == 0:
                    print() # cr after 100 interations
                self.train_step(input_image, target, epoch, summary_writer, args)

            print()
          
            # 	save step (epoch)

            self.step.assign_add(1)

            # 	save checkpoint

            if (epoch + 1) % n_iterations == 1:
                print(f'|... saving (checkpoint) the model every {n_iterations} max_epochs to {self.ckpt_prefix}')
                file_prefix = os.path.join(self.ckpt_dir, self.ckpt_prefix)
                self.checkpoint.save(file_prefix = file_prefix)
 
            #   save generated images           

            if 0 and test_ds and (epoch + 1) % n_iterations == 1:
                print(f'|... savings images every for epoch {epoch}')
                self.generate_images(test_ds, epoch, args)

            print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))

        file_prefix = os.path.join(self.ckpt_dir, self.ckpt_prefix)
        self.checkpoint.save(file_prefix = file_prefix)


    def generate_images(self, test_dataset, epoch, args=None ):

        generator = self.generator
        results_dir = self.results_dir
        zfill = args.zfill

        print(f"|===> generate_images \n \
            results_dir: {results_dir} \n \
            zfill: {zfill} \n \
        ")

        idx = 0
        for n, (img_input, img_target) in test_dataset.enumerate():
            img_prediction = generator(img_input, training=True) # _e_
            display_list = [
                onformat.nnba_to_rgb(img_input),
                onformat.nnba_to_rgb(img_target), 
                onformat.nnba_to_rgb(img_prediction)
            ]
            titles = ['Input Image', 'Ground Truth', 'Predicted Image']
            if args.visual > 1:
                    print(f'generated image {idx} in epoch {epoch}')
                    onplot.pil_show_rgbs(display_list, scale=1, rows=1)   
    
            filename = f'{str(idx).zfill(zfill)}_{str(epoch).zfill(zfill)}.png'
            if args.verbose > 2:
                print(f'|===> generate_images save \n \
                    n: {n} \n \
                    idx: {idx} \n \
                    results_dir: {results_dir} \n \
                    filename: {filename} \n \
                ')            
            save_image_path = os.path.join(results_dir, filename)
            onfile.rgbs_to_file(display_list, scale=1, rows=1, save_path=save_image_path)
            idx += 1
        
    def restore_checkpoint(self, ckptidx=None, max_to_keep=5):

        print(f'|===> model.restore_checkpoint \n \
            self.checkpoint: {self.checkpoint} \n \
            self.ckptidx: {self.ckptidx} \n \
            self.ckpt_dir: {self.ckpt_dir} \n \
            self.ckpt_prefix = {self.ckpt_prefix} \n \
            max_to_keep: {max_to_keep} \n \
        ')

        self.ckpt_manager = ckpt_manager = tf.train.CheckpointManager(
            self.checkpoint, 
            self.ckpt_dir, 
            max_to_keep=max_to_keep
        )
        # if a checkpoint exists, restore the latest checkpoint.
        

        # self.ckptidx: {None => last, ckptidx-n => n, ckptidx--n => none}
        if self.ckptidx == None:
            fromcheckpoint = ckpt_manager.latest_checkpoint
            self.ckptidx = fromcheckpoint.split('-')[-1]
        elif int(self.ckptidx) < 0:
            fromcheckpoint = None
            self.ckptidx = fromcheckpoint
        elif int(self.ckptidx) >= 0:
            fromcheckpoint = os.path.join(self.ckpt_dir, f'{self.ckpt_prefix}{self.ckptidx}')
            self.ckptidx = self.ckptidx
        print(f'\n |...> model.restore_checkpoint fet ckptidx {self.ckptidx} \n')

        if fromcheckpoint:
            self.checkpoint.restore(fromcheckpoint)
            print(f'\n |...> model.restore_checkpoint checkpoint restored from {fromcheckpoint} !!! \n')
            return fromcheckpoint
        else:
            print(f'|...> model.restore_checkpoint checkpoint not found')
            return None

#   ******************
#   FUNS PRJ
#
#   ******************

# ---------- 11
def path_to_dataset_11(pth, patt, 
        batch_size=None, height=None, width=None, buffer_size=None):
    
    print(f'|---> path_to_dataset_11:   \n \
        pth: {pth} \n \
        patt: {patt} \n \
        height: {height} \n \
        width: {width} \n \
        buffer_size: {buffer_size} \n \
        batch_size: {batch_size} \n \
    ')
    lti_one = (lambda x: path_to_pair(x, height, width))

    c = tf.data.Dataset.list_files(os.path.join(pth, patt))

    # each element in the dataset is a file with two imgs
    dataset = c.map(lambda x: lti_one(x), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)

    if 1: # list batches
        for n, (inp, re) in dataset.enumerate():
            ''' {np.shape(input_image)}" (n, 512, 512, 3) '''
            if n < 2:
                print(f'|... path_to_dataset_11 batch {n}: {np.shape(inp)} {np.shape(re)}')

    return dataset


def path_to_pair(path, height=None, width=None, exotor=1.2):
    print(f"|---> path_to_pair 11: {path}")	
    (img1, img2) = path_to_decoded(path)

    (img1, img2) = imgs_process([img1, img2], height, width, exotor)	

    return (img1, img2)


def path_to_decoded(path):
    print(f'|---> path_to_decoded: {path}')
    img = tf.io.read_file(path) # => dtype=string
    if 0: print(f'|... img: {type(img)} {np.shape(img)}')    
    img = tf.image.decode_jpeg(img) # => shape=(256, 512, 3), dtype=uint8)
    w = tf.shape(img)[1]
    w = w // 2
    img1 = img[:, :w, :] # real comes left 
    img2 = img[:, w:, :] 
    return (img2, img1)

# Random jittering:
# Resize an image to bigger height and width
# Randomly crop to the target size
# Randomly flip the image horizontally
# the image is resized to 286 x 286 and then randomly cropped to 256 x 256

def probe_dataset_11(dataset, dst_dir):
    print(f'|---> probe_dataset_11')			

    i = 0
    for sample_inp, sample_re in dataset.take(3):
        inp, re = sample_inp, sample_re

        plt.figure()
        plt.imshow(inp[0] * 0.5 + 0.5)
        print(f'|... save to {i}_example_input.png')
        plt.savefig(os.path.join(dst_dir, f'{i}_example_input.png'))

        plt.figure()
        plt.imshow(re[0] * 0.5 + 0.5)
        print(f'|... save to {i}_example_target.png')        
        plt.savefig(os.path.join(dst_dir, f'{i}_example_target.png'))

        i += 1


# ---------- 22
def paths_to_dataset_22(pths, patts, 
        batch_size=None, height=None, width=None, buffer_size=None,
        exotor=1.2):

    print(f'|---> paths_to_dataset_22:   \n \
        pths: {pths} \n \
        patts: {patts} \n \
        height: {height} \n \
        width: {width} \n \
        buffer_size: {buffer_size} \n \
        batch_size: {batch_size} \n \
    ')
    lti_two = (lambda x,y: paths_to_pair(x, y, height, width, exotor))

    a = tf.data.Dataset.list_files(os.path.join(pths[0], patts[0]), shuffle=False)
    b = tf.data.Dataset.list_files(os.path.join(pths[1], patts[1]), shuffle=False)
    c = tf.data.Dataset.zip((a, b))

    # each element in the dataset is a list of two imgs
    dataset = c.map(lambda x,y: lti_two(x,y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)

    if 1: # list batches
        for n, (inp, re) in dataset.enumerate():
            ''' {np.shape(input_image)}" (n, 512, 512, 3) '''
            if n< 2:
                print(f'|... paths_to_dataset_22 batch {n}: {np.shape(inp)} {np.shape(re)}')

    return dataset

# ----------

def paths_to_pair(path1, path2, height=256, width=256, exotor=1.0):
    print(f'|---> paths_to_pair (22): {path1} {path2}')
    (img1, img2) = paths_to_decoded([path1, path2])

    (img1, img2) = imgs_process([img1, img2], height, width, exotor)

    return (img1, img2)


def path_process(path, height=256, width=256, exotor=1.0, flip=0):
    print(f'|---> path_process')

    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img)
    img = tf.cast(img, tf.float32)

    img = img_process(img, height, width, exotor, flip)

    return img

def img_process(img, height=256, width=256, exotor=1.0, flip=0):
    print(f'|---> img_process')

    img = img_resize_with_tf(img, int(exotor * height), int(exotor * width))
    img = img_crop(img, height, width)
    if flip > 0:
        img = img_random_flip(img)
    img = onformat.rgb_to_nba(img)
    img = img[np.newaxis,:,:,:]

    return img


def imgs_process(imgs, height=256, width=256, exotor=1.2):
    print(f'|---> imgs_process')

    (img1, img2) = imgs_resize(imgs, int(exotor * height), int(exotor * width))
    (img1, img2) = imgs_crop([img1, img2], height, width)
    (img1, img2) = imgs_random_flip([img1, img2])
    (img1, img2) = rgbs_to_nbas([img1, img2])

    return (img1, img2)

def paths_to_decoded(paths):
    print(f'|---> paths_to_decoded: {paths}')    
    imgs = []
    for path in paths:
        img = tf.io.read_file(path) # => dtype=string
        if 0: print(f'|... img: {type(img)} {np.shape(img)}')
        img = tf.image.decode_jpeg(img) # => shape=(256, 512, 3), dtype=uint8)
        img = tf.cast(img, tf.float32)
        imgs.append(img)
    return imgs

def imgs_resize(imgs, height, width):
    print(f'|---> imgs_resize: {np.shape(imgs)}')	
    _imgs = []
    for i,img in enumerate(imgs):
        print(f'|... imgs_resize {i} {np.shape(img)}')
        img = tf.image.resize(img, [height, width],
            method=tf.image.ResizeMethod.AREA)
        _imgs.append(img)		
    return _imgs

def rgbs_to_nbas(imgs):
    _imgs = []
    for img in imgs:
        img = onformat.rgb_to_nba(img)
        _imgs.append(img)
    return _imgs

def imgs_crop(imgs, height, width):
    print(f'|---> imgs_crop: {np.shape(imgs)}')	
    stacked_image = tf.stack(imgs, axis=0)
    b = len(imgs)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[b, height, width, 3])
    return cropped_image[0], cropped_image[1] # _e_

def img_crop(img, height, width):
    imgs = [img]
    print(f'|---> img_crop: {np.shape(imgs)}')	
    stacked_image = tf.stack(imgs, axis=0)
    b = len(imgs)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[b, height, width, 3])
    return cropped_image[0] # _e_

def imgs_random_flip(imgs):
    print(f'|---> imgs_random_flip')	
    _imgs = []
    if tf.random.uniform(()) > 0.5: # random mirroring
        for item in imgs:
            item = tf.image.flip_left_right(item)
            _imgs.append(item)
    return imgs

def img_random_flip(img):
    imgs = [img]
    print(f'|---> img_random_flip')	
    _imgs = []
    if tf.random.uniform(()) > 0.5: # random mirroring
        for item in imgs:
            item = tf.image.flip_left_right(item)
            _imgs.append(item)
    return imgs[0]

def pair_resize(img1, img2, height, width):
    print(f'|---> pair_resize: ')		
    img1 = tf.image.resize(img1, [height, width],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    img2 = tf.image.resize(img2, [height, width],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return img1, img2

def img_resize_with_tf(img, height, width):
    print(f'|---> img_resize_with_tf {np.shape(img)} ')			
    img = tf.image.resize(img, [height, width],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return img

def img_crop_random(img, height, width, ):
    shape = np.shape(img)
    print(f'|---> img_crop_random {shape}, {height}, {width}')	
    img = tf.image.random_crop(img, size=[height, width, 3])
    print(f'|... img_crop_random {np.shape(img)}, {height}, {width}')	
    return img

def img_jitter_random(img, height, width):
    print(f'|---> img_jitter_random {np.shape(img)}')		
    img = img_crop_random(img, height, width)
    return img

# ----
def path_to_dataset_from_src(path, patt, 
        batch_size=None, height=None, width=None, buffer_size=None):
    print(f'|---> path_to_dataset_from_src ')

    if type(patt) == str:
        print('|...> get files from one source: input and real in one file')
        dataset = path_to_dataset_11(path, patt, 
            height=height, width=width, buffer_size=buffer_size, batch_size=batch_size)

    elif isinstance(patt, list): 
        print('|...> get files from two sources')
        dataset = paths_to_dataset_22(path, patt, 
            height=height, width=width, buffer_size=buffer_size, batch_size=batch_size)

    return dataset

# ----
# https://github.com/phillipi/pix2pix
def load_predefined_image_data_by_task_name(task_name,predefined_task_name_list=None,):
    print(f"|---> load_predefined_image_data_by_task_name: {task_name}")

    """Data from https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/
    View sample images here, https://github.com/yuanxiaosc/DeepNude-an-Image-to-Image-technology/tree/master/Pix2Pix"""

    if task_name in predefined_task_name_list:
            _URL = f'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/{task_name}.tar.gz'
            path_to_zip = tf.keras.utils.get_file('edges2shoes.tar.gz', origin=_URL, extract=True)
            PATH = os.path.join(os.path.dirname(path_to_zip), 'edges2shoes/')
            print(f"Store {task_name} raw data to {PATH}")
    else:
            raise ValueError(f"Predefined tasks do not include this {task_name} task!")
    return PATH

def download_and_processing_pix2pix_dataset(
        data_dir_or_predefined_task_name=None,
        predefined_task_name_list=None,
        args=None):
    print(f'|---> download_and_processing_pix2pix_dataset: {data_dir_or_predefined_task_name}')

    if data_dir_or_predefined_task_name in predefined_task_name_list:
            PATH = load_predefined_image_data_by_task_name(
                    data_dir_or_predefined_task_name,
                    predefined_task_name_list,
                    )
            print("|... prepare data from task_name")
    elif os.path.exists(data_dir_or_predefined_task_name):
            PATH = data_dir_or_predefined_task_name
            print("|... prepare data from data_dir")
    else:
            raise ValueError("Task_name error and data_dir does not exist!")


def getckptidx(ckptname): # ckpt-98, None, ckpt--1
    idx = None
    if ckptname:
        m = re.search(r'([a-zA-Z_]*)-(.*)', ckptname)
        if m and m.group(2):
            idx = m.group(2)
    return idx

#   ******************
#   CMDS
#
#   ******************

#   ******************
#   nnzip
#   
def nnzip(args, kwargs):

    args = onutil.pargs(vars(args))
    args.PROJECT = 'enzip'
    args.DATASET = 'facades'
    xp = getxp(vars(args))
    args = onutil.pargs(xp)
    onutil.ddict(vars(args), 'args')

    print(f"|---> nnzip: {args.PROJECT}:   \n ")

    if 1: # tree

        args.dataorg_train_dir = os.path.join(args.dataorg_dir, 'train')
        args.dataorg_test_dir = os.path.join(args.dataorg_dir, 'test')

        assert(os.path.exists(args.dataorg_dir))
        assert(os.path.exists(args.dataorg_train_dir))
        assert(os.path.exists(args.dataorg_test_dir))

        args.ckpt_dir = args.models_dir
        args.ckpt_prefix = os.path.join(args.ckpt_dir, "ckpt-")

        ''' train/test images in origin with pattern '''
        args.data_train_B_dir = os.path.join(args.data_dir, 'train_B')
        args.data_train_A_dir = os.path.join(args.data_dir, 'train_A')
        args.data_test_B_dir = os.path.join(args.data_dir, 'test_B')
        args.data_test_A_dir = os.path.join(args.data_dir, 'test_A')

        args.dataset_train_B_dir = os.path.join(args.dataset_dir, 'train_B')
        args.dataset_train_A_dir = os.path.join(args.dataset_dir, 'train_A')
        args.dataset_test_B_dir = os.path.join(args.dataset_dir, 'test_B')
        args.dataset_test_A_dir = os.path.join(args.dataset_dir, 'test_A')

        args.zipfile = os.path.join(args.dataorg_dir, f'{args.DATASET}.zip')

        os.makedirs(args.data_train_B_dir, exist_ok=True) 
        os.makedirs(args.data_train_A_dir, exist_ok=True) 
        os.makedirs(args.data_test_B_dir, exist_ok=True) 
        os.makedirs(args.data_test_A_dir, exist_ok=True) 

        os.makedirs(args.dataset_train_B_dir, exist_ok=True) 
        os.makedirs(args.dataset_train_A_dir, exist_ok=True) 
        os.makedirs(args.dataset_test_B_dir, exist_ok=True) 
        os.makedirs(args.dataset_test_A_dir, exist_ok=True) 

        os.makedirs(args.models_dir, exist_ok=True)
        os.makedirs(args.logs_dir, exist_ok=True)
        os.makedirs(args.tmp_dir, exist_ok=True)



    if args.verbose: print(f"|---> tree: {args.PROJECT}:   \n \
        args.PROJECT:    	  {args.PROJECT} \n \
        args.DATASET:    	  {args.DATASET} \n \
        cwd:     				{os.getcwd()} \n \
        args.dataorg_dir:     {args.dataorg_dir} \n \
        args.dataorg_train_dir: {args.dataorg_train_dir} \n \
        args.dataorg_test_dir: {args.dataorg_test_dir} \n \
        args.ckpt_dir:  {args.ckpt_dir} \n \
        args.logs_dir:        {args.logs_dir} \n \
        args.tmp_dir:        {args.tmp_dir} \n \
        args.zipfile:         {args.zipfile} \n \
        args.verbose:         {args.verbose}, \n \
        args.visual:          {args.visual}, \n \
    ")


    if 1: # config
        args.height = args.img_height
        args.width = args.img_width
        args.buffer_size = args.buffer_size
        args.batch_size = args.batch_size
        args.input_channels = args.input_channels
        args.input_shape = [args.height, args.width, args.input_channels]
        if 0:
            ''' copy org to same data folder'''
            args.patts = ['*in.png', '*re.png']
        else:
            ''' will separate images in data'''
            args.patts = ['*.png', '*.png']

    if args.verbose: print(f"|---> nnleonardo config:   \n \
        args.max_epochs:            {args.max_epochs}, \n \
        args.output_channels:	{args.output_channels} \n \
        args.height: 			{args.height} \n \
        args.width: 			{args.width} \n \
        args.input_channels: 	{args.input_channels} \n \
        args.buffer_size: 		{args.buffer_size} \n \
        args.batch_size: 		{args.batch_size} \n \
        args.input_shape: 		{args.input_shape} \n \
        args.patts:     		{args.patts}, \n \
    ")


    if 1: # remote data to local

        #args.DATASET = 'facades'
        dat = 'facades'
        dats=['facades']

        if 0: # create zip
            tarfolder = os.path.join(args.data_dir, dat)
            print(f'|===> create zip from {tarfolder} folder')
            tarfile = os.path.join(args.data_dir, 'test.gz')
            onutil.tenzip(tarfile, tarfolder)

        print(f'|===> download google folder by id')
        if 0: 
            URL='https://drive.google.com/drive/folders/'
            gid = '1vTa_y5RYvzk6BPDXQ7HrXfhp8BgjzoKv'
            onutil.gdownid(URL, gid, args.data_dir)

        print(f'|===> download from a set of files on remote server')

        print(f'|... chech that remote is url')

        dats=['cityscapes', 'night2day', 'edges2handbags', 'edges2shoes', 'facades', 'maps']
        url = 'http://efrosgans.eecs.berkeley.edu/pix2pix/datasets'
        #url = args.dataorg_dir
        zexts = ['.tar.gz', '.tar', '.tgz', '.zip']

        print(f'|... chech that remote is path')


        local_tar_path = None
        if 1:
            print(f'|... look for tar in local site {args.data_dir}')		
            print(len(dats), len(zexts))
            it1 = ((i, j) for i in range(len(dats)) for j in range(len(zexts)))
            for i, j in it1:
                dat_path = os.path.join(args.data_dir, f'{dats[i]}{zexts[j]}')
                print(f'|.... look for {dat_path}')
                if os.path.exists(dat_path):
                    local_tar_path = dat_path
                    print(f'|.... local {dat_path} already EXISTS !!! ')
                    break

        remote_tar_path = None
        if not local_tar_path:
            print(f'|... look for tar in remote site {args.dataorg_dir}')	
            print(len(dats), len(zexts))
            it2 = ((i, j) for i in range(len(dats)) for j in range(len(zexts)))
            for i, j in it2:
                dat_path = os.path.join(args.gdata, f'{dats[i]}{zexts[j]}')
                if os.path.exists(dat_path):
                    remote_tar_path = dat_path
                    print(f'|... remote {dat_path} already EXISTS !!! ')
                    break

        print(f'|... local_tar_path EXISTS : {local_tar_path}')
        print(f'|... remote_tar_path EXISTS : {remote_tar_path}')

        if remote_tar_path and not local_tar_path:
            print(f'|===> remote {remote_tar_path} to local {local_tar_path}')
            tarname = os.path.basename(os.path.normpath(remote_tar_path))
            print(tarname)
            local_tar_path = os.path.join(args.data_dir, tarname)
            print(f'|... copy {remote_tar_path} to {local_tar_path}')
            from shutil import copyfile
            copyfile(remote_tar_path, local_tar_path)


        if local_tar_path:
            print("|... unzip to data")
            tree_root = None
            print(f'|... unzip to data local_tar_path {local_tar_path}')
            # onutil.gunzip(file_basename, file_id, destination, results_dir)
            onutil.tunzip(None, local_tar_path, args.data_dir, tree_root)
    
        # check data folder
        untarfolder = os.path.join(args.data_dir, dat)
        print(f'|... untar folder {untarfolder} exists" {os.path.exists(untarfolder)}')


#   ******************
#   nndanboo
#
def nndanboo(args, kwargs):

    
    if onutil.incolab():
        args = onutil.pargs(vars(args))
        args.AUTHOR = 'lllyasviel'
        args.PROJECT = 'nndanboo'
        args.GITPOD = 'DanbooRegion'
        args.DATASET = 'DanbooRegion2020'
        xp = getxp(vars(args))
        args = onutil.pargs(xp)
        onutil.ddict(vars(args), 'args')
    else:
        args = onutil.pargs(vars(args))
        args.AUTHOR  = 'lllyasviel'
        args.PROJECT = 'danregion'
        args.GITPOD = 'DanbooRegion'
        args.DATASET = 'danregion' # 'DanbooRegion2020'
        xp = getxp(vars(args))
        args = onutil.pargs(xp)
        onutil.ddict(vars(args), 'args')        

    print(f"|---> nndanboo: {args.PROJECT}  \n ")

    if 1: # tree
        # [1] https://github.com/lllyasviel/DanbooRegion
        # [2] https://drive.google.com/drive/folders/1ihLt6P7UQRlaFtZUEclXkWC9grmEXEUK?usp=sharing

        assert(os.path.exists(args.dataorg_dir))

        args.dataorg_train_dir = os.path.join(args.dataorg_dir, 'train')
        args.dataorg_test_dir = os.path.join(args.dataorg_dir, 'test')

        args.download_dir = os.path.join(args.proj_dir, 'download')

        args.results_dir = os.path.join(args.proj_dir, 'results') # in project dir
        args.code_dir = os.path.join(args.proj_dir, 'code') # inside project dir


        args.data_train_dir = os.path.join(args.data_dir, 'train')
        args.data_test_dir = os.path.join(args.data_dir, 'test')
        args.data_val_dir = os.path.join(args.data_dir, 'val')

        if onutil.incolab():
            args.ckpt_dir = args.models_dir
        else:
            _glab = os.path.join(args.gdata, '../glab/', args.MNAME, args.PROJECT)
            args.ckpt_dir = os.path.normpath(os.path.join(_glab, 'Models'))

        args.ckpt_prefix = os.path.normpath(os.path.join(args.ckpt_dir, "ckpt-"))

        args.data_dir = os.path.join(args.data_dir, '') # in project dir
        args.data_train_pict_dir = os.path.join(args.data_train_dir, 'pict')
        args.data_train_draw_dir = os.path.join(args.data_train_dir, 'draw')

        args.data_test_pict_dir = os.path.join(args.data_test_dir, 'pict')
        args.data_test_draw_dir = os.path.join(args.data_test_dir, 'draw')
        args.data_test_predict_dir = os.path.join(args.data_test_dir, 'predict')


        print(f"|---> nndanboo tree:  \n \
        cwd: {os.getcwd()} \n \
        args.proto_dir: {args.proto_dir} \n \
        args.code_dir: {args.code_dir} \n \
        args.ckpt_dir: {args.ckpt_dir} \n \
        args.ckpt_prefix: {args.ckpt_prefix} \n \
        \n \
        args.dataorg_dir: {args.dataorg_dir}, {onfile.qfiles(args.dataorg_dir, '*.png')}\n \
        args.dataorg_train_dir: {args.dataorg_train_dir}, {onfile.qfiles(args.dataorg_train_dir, '*.png')}\n \
        args.dataorg_test_dir: {args.dataorg_test_dir}, {onfile.qfiles(args.dataorg_test_dir, '*.png')}\n \
        \n \
        args.data_dir: {args.data_dir}, \n \
        args.data_train_dir: {args.data_train_dir}, {onfile.qfiles(args.data_train_dir, '*.png')}\n \
        args.data_train_dir (images): {args.data_train_dir}, {onfile.qfiles(args.data_train_dir, '*.image.png')} \n \
        args.data_train_dir (regions): {args.data_train_dir}, {onfile.qfiles(args.data_train_dir, '*.region.png')} \n \
        args.data_train_dir (skels): {args.data_train_dir}, {onfile.qfiles(args.data_train_dir, '*.skeleton.png')} \n \
        args.data_test_dir: {args.data_test_dir}, {onfile.qfiles(args.data_test_dir, '*.png')}\n \
        args.data_val_dir: {args.data_val_dir}, {onfile.qfiles(args.data_val_dir, '*.png')}\n \
        ")


        args.dataorg_dir = os.path.join(args.dataorg_dir, '')
        if not os.path.exists(args.dataorg_dir):
            print(f'org data missing \n \
                the org data folder: {args.dataorg_dir} was not found !!! \n \
                set args.dataorg_dir if needed and \n \
                download the dataset following instructions in \n \
                    https://github.com/lllyasviel/DanbooRegion : \n \
                Download the dataset from:\n \
                    https://drive.google.com/drive/folders/1ihLt6P7UQRlaFtZUEclXkWC9grmEXEUK?usp=sharing\n \
                For linux then run: (Windows does not need this step.)\n \
                    cat DanbooRegion2020.zip.* > DanbooRegion2020.zip\n \
                    zip -FF DanbooRegion2020.zip --out DanbooRegion2020_FF.zip\n \
                    unzip DanbooRegion2020_FF.zip\n \
            ')
            exit()


        os.makedirs(args.proj_dir, exist_ok=True) 
        os.makedirs(args.code_dir, exist_ok=True) 
        os.makedirs(args.results_dir, exist_ok=True) 
        os.makedirs(args.download_dir, exist_ok=True) 
        os.makedirs(args.data_train_dir, exist_ok=True) 
        os.makedirs(args.data_train_pict_dir, exist_ok=True) 
        os.makedirs(args.data_train_draw_dir, exist_ok=True) 
        os.makedirs(args.data_test_dir, exist_ok=True) 
        os.makedirs(args.data_test_pict_dir, exist_ok=True) 
        os.makedirs(args.data_test_draw_dir, exist_ok=True) 
        os.makedirs(args.data_test_predict_dir, exist_ok=True) 


    if 1: # config

        args.ckpt_prefix = 'ckpt-'

        args.dim = 512
        args.show_size = 512
        args.height = 512
        args.width = 512

        args.max_size = None # control set resize

        args.buffer_size = 10000
        args.batch_size = 1
        args.max_epochs = 601
        args.gpu = 1 # _e_
        args.input_shape = [args.height, args.width, args.input_channels]		

    print(f"|---> nndanboo config:  \n \
        args.show_size: {args.show_size}, \n \
        args.batch_size: {args.batch_size}, \n \
        args.gpu: {args.gpu}, \n \
        args.height: {args.height}, \n \
        args.width: {args.width}, \n \
        args.buffer_size: {args.buffer_size}, \n \
        args.batch_size: {args.batch_size}, \n \
        args.input_shape: {args.input_shape}, \n \
    ")


    if 1: # git
        onutil.get_git(args.AUTHOR, args.GITPOD, args.code_dir)

    assert os.path.exists(args.code_dir), "code_dir not found"        
    os.chdir(args.code_dir) # _e_ not std

    if args.visual > 1: # visualize region image with image color map

        if 0:
            cmd = f"python visualize.py ./X.image.png ./X.region.png"
            print("cmd %s" %cmd)
            os.system(cmd)
        else:
            color_map = cv2.imread('./X.image.png') # _e_
            region_map = cv2.imread('./X.region.png')
            cv2.imshow('vis', onlllyas.vis(region_map, color_map))
            cv2.waitKey(0)


    if 1: # org to (train) data 

        print(f'|---> nndanregion: org => data train \n \
            \n \
            org image: {onfile.qfiles(args.dataorg_train_dir, "*.png")} \n \
            train: {onfile.qfiles(args.data_train_pict_dir, "*.png")} \n \
        ')

        args.keep_folder=True    
        args.process_type='resize'
        args.file_extension= 'png'
        args.name=1 # if 0: calculate name on index

        args.input_folder = args.dataorg_train_dir
        args.filepatt = f'.*{args.file_extension}'
        args.output_folder = args.data_train_pict_dir

        qinfiles = onfile.qfiles(args.input_folder, f'*{args.file_extension}')
        qoutfiles = onfile.qfiles(args.output_folder)

        print(f"|---> nndanregion train copy: {args.process_type} \n \
            {args.input_folder} (q: {qinfiles})  \n \
            \t ==> {args.output_folder} (q: {qoutfiles}) \n \
        ")

        # count max and exclude list _e_
        if  qinfiles > qoutfiles:
            print(f'|... processFolder from ({qinfiles}) onto ({qoutfiles}) files ')
            args.exclude = [os.path.basename(item) for item in onfile.path_to_paths(args.output_folder)]
            if args.verbose > 2: 
                print(args.exclude)
            onset.processFolder(args) # copy inputs to train
        else:
            print(f'|... train no processFolder !!!!. files already there ')


    if 1: # org to (test) data 

        print(f'|---> nndanregion org test to data q: \n \
            org image: {onfile.qfiles(args.dataorg_test_dir, "*.png")} \n \
            test re: {onfile.qfiles(args.data_test_pict_dir, "*.png")} \n \
        ')

        args.keep_folder=True    
        args.process_type='resize' # just copy
        args.file_extension= 'png'
        args.name=1
        args.zfill=4

        args.input_folder = args.dataorg_test_dir
        args.filepatt = f'.*{args.file_extension}'
        args.output_folder = args.data_test_pict_dir

        qinfiles = onfile.qfiles(args.input_folder, f'*{args.file_extension}')
        qoutfiles = onfile.qfiles(args.output_folder)

        print(f"|---> test copy: {args.process_type} \n \
            {args.input_folder} (q: {qinfiles})  \n \
            \t ==> {args.output_folder} (q: {qoutfiles}) \n \
        ")

        if  qinfiles > qoutfiles:
            args.exclude = [os.path.basename(item) for item in onfile.path_to_paths(args.output_folder)]
            onset.processFolder(args) # copy inputs to test
        else:
            print(f'|... test no processFolder !!!!. files already there ')



    if 0: # probe images to skeletons

        print(f'|---> skeletonize test png')

        from skimage.morphology import thin as skeletonize
        if 0:
            cmd = f"|---> python skeletonize.py ./region_test.png"
            print("cmd %s" %cmd)
            os.system(cmd)
        else:
            region_map = cv2.imread('./region_test.png')
            cv2.imshow('vis', onlllyas.get_skeleton(region_map, filterstrength=1.0))
            cv2.waitKey(0)



    if 1: # (train) data to skeletons

        print(f'|===> skeletonize_all')

        linpatts = ['*.region.png']
        fromdir = args.data_train_pict_dir
        input_region_train_paths = onfile.path_to_paths(fromdir, linpatts)
        train_paths = input_region_train_paths[0:len(input_region_train_paths)]
        qinfiles = len(train_paths)

        todir = args.data_train_draw_dir
        qoutfiles = onfile.qfiles(todir)
        
        filterstrengths = [5.0] # {1.0,5.0}

        print(f'|...>  skeletonize_all \n \
            from {fromdir} ({qinfiles}) \n \
            to {todir} ({qoutfiles}) \n \
            to filterstrengths: {filterstrengths} \n \
        ')

        for filterstrength in filterstrengths:

            _todir = f'{todir}' # {int(filterstrength)}
            os.makedirs(_todir, exist_ok=True)

            loutpatt = f'.skeleton'
            patt = f'*{loutpatt}*' # look for skels with strength
            qoutfiles = onfile.qfiles(_todir, patt)
            outprefixes = [os.path.basename(item).split('.')[0] for item in onfile.path_to_paths(_todir)]

            
            if qinfiles > qoutfiles: # files in outdir with loutpatt
                print(f'|...>  raw to data ({qinfiles}) to ({qoutfiles})')
                for i,path in enumerate(train_paths):

                    filename = os.path.basename(path)
                    ext = filename.split('.')[-1]	# maintain ext
                    prefix = filename.split('.')[0] # get left to dot

                    if prefix not in outprefixes: # exclude
                        out_path = os.path.join(_todir,f'{prefix}{loutpatt}.{ext}',)
                        region = cv2.imread(path)

                        if 0: print(f'|...> skeleton  region  {np.shape(region)} to {_todir}')
                        skeleton = onlllyas.get_skeleton(region, 
                            filterstrength=filterstrength
                        )

                        if 0: print(f'|...> skeleton  shape  {np.shape(skeleton)}')
                        print(f'|...> write   {out_path}   {str(i + 1)} /{qinfiles}')
                        skeleton = cv2.cvtColor(skeleton,cv2.COLOR_GRAY2RGB)					
                        cv2.imwrite(out_path, skeleton)
                    else:
                        print(f'prefix: {prefix} already in target')

            else:
                print(f'|... no process. out files {qoutfiles} in')

    

    if 0: # skeletom to regions show

        #skeleton_path = os.path.join(args.code_dir, 'skeleton_test.png')
        skeleton_path = os.path.join(args.code_dir, 'danskel.jpg')
        assert os.path.exists(skeleton_path), f"skeleton_path {skeleton_path} does not exist"
        skeleton_map = cv2.imread(skeleton_path)
        img = onlllyas.skeleton_to_regions(skeleton_map)
        if onutil.incolab():
            from google.colab.patches import cv2_imshow
            cv2_imshow(img)
            cv2.waitKey(0)            
        else:
            cv2.imshow('skel', img)
            cv2.waitKey(0)



    if 1: # model

        if args.verbose: print(f'|===> model:   \n \
            models_dir = {args.models_dir},\n \
            logs_dir = {args.logs_dir},\n \
            results_dir = {args.results_dir},\n \
            ckpt_dir = {args.ckpt_dir},\n \
            ckpt_prefix = {args.ckpt_prefix},\n \
            input_shape = {args.input_shape},\n \
            output_shape = {args.input_shape},\n \
        ')
        model = GAN(
            models_dir = args.models_dir,
            logs_dir = args.logs_dir,
            results_dir = args.results_dir,
            ckpt_dir = args.ckpt_dir,
            ckpt_prefix = args.ckpt_prefix,
            ckptidx = -1,   # ****
            input_shape = args.input_shape,
            output_shape = args.input_shape,
        )
    

    # 3. segment
    
    if 1:	# python segment.py ./emilia.jpg

        #path = os.path.join(args.code_dir, 'emilia.jpg')
        path = os.path.join(args.code_dir, 'danimage.jpg')   
        path = os.path.join(args.code_dir, 'danskel.jpg')           
        assert os.path.exists(path), f'{path} could not be found'

        basename = os.path.basename(path)
        print(f'|===> segment : {path}')

        #raw_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) # go_srcnn
        raw_img = cv2.imread(path) # go_srcnn
        raw_img = onlllyas.min_resize(raw_img, 512)
        raw_img = raw_img.clip(0, 255)
        raw_img = raw_img.astype(np.uint8)

        raw_img = raw_img[np.newaxis,:,:,:]
        raw_img = model.generator(raw_img, training=True)
        try:
            from google.colab.patches import cv2_imshow
            cv2_imshow(onformat.nnba_to_rgb(raw_img))
            cv2.waitKey(0)            
        except:
            cv2.imshow(basename, onformat.nnba_to_rgb(raw_img))
            cv2.waitKey(0)


        img_2048 = onlllyas.min_resize(raw_img, 2048)        

        #transposed = onlllyas.go_transposed_vector(onlllyas.mk_resize(raw_img, 64))
        #height = onlllyas.d_resize(transposed, img_2048.shape) * 255.0
        height = onlllyas.d_resize(go_transposed_vector(onlllyas.mk_resize(raw_img, 64)), img_2048.shape) * 255.0


        final_height = height.copy()

        height += (height - cv2.GaussianBlur(height, (0, 0), 3.0)) * 10.0
        height = height.clip(0, 255).astype(np.uint8)

        marker = height.copy()
        marker[marker > 135] = 255
        marker[marker < 255] = 0


        #print("*************************************** marker", marker)
        # scipy.ndimage.label:  non-zero values : features; zero values: background
        #fills = onlllyas.get_fill(marker / 255)

        def count_all(labeled_array, all_counts): # count labelled features
            #labeled_array:  [[[0 0 1]
            #                  [0 0 1]
            #                  [0 0 1]
            #                  ...
            M = labeled_array.shape[0] # 2048
            N = labeled_array.shape[1] # 2048
            print("count_all MN ", M, N)
            for x in range(M):
                for y in range(N):
                    i = labeled_array[x, y] - 1 # [-1 -1  0]
                    print(f'count_all {x},{y}: {i}')
                    if i > -1:
                        all_counts[i] = all_counts[i] + 1
            return


        def trace_all(labeled_array, xs, ys, cs):
            M = labeled_array.shape[0]
            N = labeled_array.shape[1]
            for x in range(M):
                for y in range(N):
                    current_label = labeled_array[x, y] - 1
                    if current_label > -1:
                        current_label_count = cs[current_label]
                        xs[current_label][current_label_count] = x
                        ys[current_label][current_label_count] = y
                        cs[current_label] = current_label_count + 1
            return

        def find_all(labeled_array):
            hist_size = int(np.max(labeled_array))
            if hist_size == 0:
                return []
            all_counts = [0 for _ in range(hist_size)]
            count_all(labeled_array, all_counts)
            xs = [np.zeros(shape=(item, ), dtype=np.uint32) for item in all_counts]
            ys = [np.zeros(shape=(item, ), dtype=np.uint32) for item in all_counts]
            cs = [0 for item in all_counts]
            trace_all(labeled_array, xs, ys, cs)
            filled_area = []
            for _ in range(hist_size):
                filled_area.append((xs[_], ys[_]))
            return filled_area

        labeled_array, num_features = label(marker / 255)
        fills = find_all(labeled_array)
        # *********************************************


        for fill in fills:
            if fill[0].shape[0] < 64:
                marker[fill] = 0
        filter = np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]],
            dtype=np.uint8)
        big_marker = cv2.erode(marker, filter, iterations=5)
        fills = onlllyas.get_fill(big_marker / 255)
        for fill in fills:
            if fill[0].shape[0] < 64:
                big_marker[fill] = 0
        big_marker = cv2.dilate(big_marker, filter, iterations=5)
        small_marker = marker.copy()
        small_marker[big_marker > 127] = 0
        fin_labels, nil = label(big_marker / 255)
        fin_labels = up_fill(onlllyas.get_fill(small_marker), fin_labels)
        water = cv2.watershed(img_2048.clip(0, 255).astype(np.uint8), fin_labels.astype(np.int32)) + 1
        water = onlllyas.thinning(water)
        all_region_indices = onlllyas.find_all(water)
        regions = np.zeros_like(img_2048, dtype=np.uint8)
        for region_indices in all_region_indices:
            regions[region_indices] = np.random.randint(low=0, high=255, size=(3,)).clip(0, 255).astype(np.uint8)
        result = np.zeros_like(img_2048, dtype=np.uint8)
        for region_indices in all_region_indices:
            result[region_indices] = np.median(img_2048[region_indices], axis=0)
        
        img = final_height.clip(0, 255).astype(np.uint8), regions.clip(0, 255).astype(np.uint8), result.clip(0, 255).astype(np.uint8)



        #skeleton, region, flatten = segment(img)
        #cv2.imwrite('./current_skeleton.png', skeleton)
        #cv2.imwrite('./current_region.png', region)
        #cv2.imwrite('./current_flatten.png', flatten)
        #print('./current_skeleton.png')
        #print('./current_region.png')
        #print('./current_flatten.png')
        #print('ok!')

        #img = path_process(path, height=512, width=512)
        #img = model.generator(img)	
        #img = onformat.nnba_to_rgb(img)

        try:
            from google.colab.patches import cv2_imshow
            cv2_imshow(img)
            cv2.waitKey(0)            
        except:
            cv2.imshow(basename, img)
            cv2.waitKey(0)


    # 3b. predict test images
    
    if 0:	# 
        
        paths = Onfile.folder_to_paths(args.data_test_pict_dir)


        idx = 0
        for n, path in enumerate(paths):
            basename = os.path.basename(os.path.normpath(path))
            print(f'|===> predict test images path {path}')

            #img = tf.io.read_file(path)
            #img = tf.image.decode_jpeg(img)
            #img = tf.cast(img, tf.float32)
            #img = tf.image.resize(img, [512, 512])
            #img = onformat.rgb_to_nba(img)
            #img = img[np.newaxis,:,:,:]

            img = path_process(path, height=512, width=512)

            img = model.generator(img, training=True) # _e_

            img = onformat.nnba_to_rgb(img)
            try:
                from google.colab.patches import cv2_imshow
                cv2_imshow(img)
                cv2.waitKey(0)            
            except:
                cv2.imshow(basename, img)
                cv2.waitKey(0)

            save_image_path = os.path.join(args.data_test_predict_dir, basename)
            print(f'|... save {save_image_path}')            
            onfile.rgbs_to_file([img], scale=1, rows=1, save_path=save_image_path)

  
    if 0: # train

        if 1: #  data => dataset (train) # paths_to_dataset_22

            #pths_train = [args.data_train_draw_dir, args.data_train_pict_dir]
            #patts = ['*.skeleton.png', '*.region.png']

            pths_train = [args.data_train_pict_dir, args.data_train_draw_dir]
            patts = ['*.region.png', '*.skeleton.png']

            print(f'|===> nndanregion train dataset:  \n \
                pths_train {pths_train} \n \
                train_draw ({onfile.qfiles(args.data_train_draw_dir, "*.png")}) \n \
                train_pict ({onfile.qfiles(args.data_train_pict_dir, "*.png")}) \n \
            ')

            train_dataset = paths_to_dataset_22(pths_train, patts, # BatchDataset
                height=args.height, width=args.width, 
                buffer_size=args.buffer_size, batch_size=args.batch_size)


        if 1: #  data => dataset (test) --- # paths_to_dataset_22

            print(f'|---> nndanregion datasets:  \n \
                test_draw ({onfile.qfiles(args.data_test_draw_dir, "*.png")}) \n \
                test_pict ({onfile.qfiles(args.data_test_pict_dir, "*.png")}) \n \
            ')

            if 0:
                pths_test = [args.data_train_draw_dir, args.data_train_pict_dir]
                patts = ['*.skeleton.png', '*.region.png']
            else:
                pths_test = [args.data_train_pict_dir, args.data_train_draw_dir]
                patts = ['*.region.png', '*.skeleton.png']

            test_dataset = paths_to_dataset_22(pths_test, patts,
                height=args.height, width=args.width, 
                buffer_size=args.buffer_size, batch_size=1) # 1 per batch in test dataset _e_


        if 0: # probe train dataset

            print(f'|===> probe nndanregion train dataset')

            for im,re in train_dataset.take(1):
                print(f'|... item shape in dataset: {np.shape(im)} {np.shape(re)}')

                img1 = onformat.nnba_to_rgb(im)
                img2 = onformat.nnba_to_rgb(re)

                display_list = [img1, img2]
                onplot.pil_show_rgbs(display_list, scale=1, rows=1) 


        print(f'|===> training loop:')
        model.fit(train_dataset, test_dataset, args)     # , summary_writer


    print('|===> end danboo')

#   ******************
#   nnart
#   ******************
def nnart(args, kwargs):
# https://github.com/memo/webcam-pix2pix-tensorflow

    args = onutil.pargs(vars(args))
    args.PROJECT = 'art'
    args.DATASET = 'danregion'
    xp = getxp(vars(args))
    args = onutil.pargs(xp)
    onutil.ddict(vars(args), 'args')

    print(f"|---> nnart: {args.PROJECT}:  \n ")

    if 1: # tree
        # ckpt_dir
        ckpt_dir = args.models_dir
        ckpt_prefix = 'ckpt-'

        # data
        args.data_train_dir = os.path.join(args.data_dir, 'train')
        args.data_test_dir = os.path.join(args.data_dir, 'test')

        # dataset
        args.dataset_train_B_dir = os.path.join(args.dataset_dir, 'train_B_dir')
        args.dataset_train_A_dir = os.path.join(args.dataset_dir, 'train_A_dir')
        args.dataset_test_B_dir = os.path.join(args.dataset_dir, 'test_B_dir')
        args.dataset_test_A_dir = os.path.join(args.dataset_dir, 'test_A_dir')

        os.makedirs(args.models_dir, exist_ok=True)
        os.makedirs(args.logs_dir, exist_ok=True)

        # raw data src
        os.makedirs(args.dataorg_dir, exist_ok=True)
        os.makedirs(args.data_dir, exist_ok=True)
        os.makedirs(args.data_train_dir, exist_ok=True) 
        os.makedirs(args.data_test_dir, exist_ok=True)	

        os.makedirs(args.dataset_dir, exist_ok=True)
        os.makedirs(args.dataset_train_B_dir, exist_ok=True) 
        os.makedirs(args.dataset_train_A_dir, exist_ok=True) 
        os.makedirs(args.dataset_test_B_dir, exist_ok=True) 
        os.makedirs(args.dataset_test_A_dir, exist_ok=True) 
        os.makedirs(args.records_dir, exist_ok=True)


    if args.verbose: print(f"|---> nnart tree: {args.PROJECT}:  \n \
    cwd: {os.getcwd()} \n \
    args.PROJECT:        {args.PROJECT} \n \
    args.DATASET:        {args.DATASET} \n \
    args.proj_dir:       {args.proj_dir}, data and code will be in args.proj_dir\n \
    args.logs_dir:       {args.logs_dir} {onfile.qfiles(args.models_dir, '*')} logs \n \
    args.records_dir:    {args.records_dir} :-: {onfile.qfiles(args.records_dir, '*')} files \n \
    args.models_dir (ckpts): {args.models_dir} :-: {onfile.qfiles(args.models_dir, '*.index')} \n \
    args.dataorg_dir:    {args.dataorg_dir} :-: {onfile.qfiles(args.dataorg_dir, '*.jpg')}  jpgs \n \
    \n \
    args.data_dir:       {args.data_dir}: {onfile.qfiles(args.data_dir, '*')} files \n \
    args.data_train_dir: {args.data_train_dir} :-: {onfile.qfiles(args.data_train_dir, '*.jpg')} \n \
    args.data_test_dir:  {args.data_test_dir} :-: {onfile.qfiles(args.data_test_dir, '*.jpg')} \n \
    \n \
    args.dataset_dir:    {args.dataset_dir} :-: {onfile.qfolders(args.dataset_dir)} folders\n \
    args.dataset_train_B_dir (re): {args.dataset_train_B_dir} :-: {onfile.qfiles(args.dataset_train_B_dir, '*.jpg')}  \n \
    args.dataset_train_A_dir (in): {args.dataset_train_A_dir} :-: {onfile.qfiles(args.dataset_train_A_dir, '*.jpg')}  \n \
    args.dataset_test_B_dir (re): {args.dataset_test_B_dir} :-: {onfile.qfiles(args.dataset_test_B_dir, '*.jpg')}  \n \
    args.dataset_test_A_dir (in): {args.dataset_test_A_dir} :-: {onfile.qfiles(args.dataset_test_A_dir, '*.jpg')}  \n \
    ")

    if 0: # clear tree

        print(f'|---> clear tree at {args.proj_dir}')
        onfile.clearfolder(args.proj_dir, inkey=args.PROJECT)

    if 1: # data params

        args.img_height = 512
        args.img_width = 512

        args.max_size=512

        args.height=args.img_height
        args.width=args.img_height
        args.input_shape = [args.height, args.width, args.input_channels]

    if args.verbose: print(f"|---> config  \n \
        args.height:         {args.height} \n \
        args.width:          {args.width} \n \
        args.input_channels: {args.input_channels} \n \
        args.buffer_size:    {args.buffer_size} \n \
        args.batch_size:     {args.batch_size} \n \
        args.input_shape:    {args.input_shape} \n \
    ")

    if 1: # raw to paths

        trainpc = 0.9
        testpc = 0.1
        forcecopy = 0

        input_paths = onfile.path_to_paths(args.dataorg_dir, patts=['*.jpg', '*.jpeg', '*.png'])
        qimgs = len(input_paths)
        q_train_imgs = int(qimgs * trainpc )
        q_test_imgs = int(qimgs * testpc )		

        train_paths = input_paths[0:q_train_imgs]
        test_paths = input_paths[q_train_imgs:qimgs]		

        print(f'|--->  raw to data - keep img file names \n \
        from {args.dataorg_dir}, {qimgs} images \n \
            to data_train_dir: {args.data_train_dir} {q_train_imgs} ({trainpc} %) \n \
            to data_test_dir:  {args.data_test_dir}  {q_test_imgs} ({testpc} %)\n \
        ')
        
        train_roots = [onutil.get_rootid(path) for path in train_paths]
        if all(root.isdigit() for root in train_roots):
            print(f'|---> sort numeric')
            train_paths = sorted(train_paths, key=lambda path: int(onutil.get_rootid(path)))
        else:
            print(f'|---> sort alphabetic')
            train_paths = sorted(train_paths)

        assert len(train_paths) > 0, f'train files missing'

        test_roots = [onutil.get_rootid(path) for path in test_paths]
        if all(root.isdigit() for root in test_roots):
            test_paths = sorted(test_paths, key=lambda path: int(onutil.get_rootid(path)))
        else:
            test_paths = sorted(test_paths)

        assert len(test_paths) >= 0, f'test files may be missing'


    if 0: # raw to formed data (dataorg_dir => data_train_dir, data_test_dir)
        for i,path in enumerate(train_paths):
            rootid = int(onutil.get_rootid(path))
            rootfilled = onutil.nameint(rootid)
            ext = os.path.splitext(path)[-1]

            out_path = os.path.join(args.data_train_dir, f'img{rootfilled}{ext}')

            print(f"|---> {i} {rootfilled} {path} {out_path}")
            img = ondata.path_to_formed_pair(path)
            img.save(os.path.join(out_path))

        for i,path in enumerate(test_paths):
            rootid = int(onutil.get_rootid(path))
            rootfilled = onutil.nameint(rootid)
            ext = os.path.splitext(path)[-1]

            out_path = os.path.join(args.data_test_dir, f'img{rootfilled}{ext}')

            print(f"{i} {rootfilled} {path} {out_path}")
            img = ondata.path_to_formed_pair(path)
            img.save(os.path.join(out_path))

        assert len(onfile.qfiles(args.data_train_dir)) > 0, f'train data files missing'
        assert len(onfile.qfiles(args.data_test_dir)) > 0, f'test data files missing'

    if 1: # model

        print('|---> gan model')
        gan = GAN(
            models_dir = args.models_dir,
            logs_dir = args.logs_dir,
            ckpt_dir = ckpt_dir,
            #ckpt_prefix = ckpt_prefix,
            input_shape = args.input_shape,
            output_shape = args.input_shape,
        )

    if 1: #  formed => datasets 11

        print(f'|===> to datasets 11')
        train_dataset = path_to_dataset_11(
            args.data_train_dir, 
            '*.png', 
            batch_size=4)
        test_dataset = path_to_dataset_11(
            args.data_test_dir, 
            '*.png', 
            batch_size=4)

        print(f'|---> train_dataset {type(train_dataset)}')
        print(f'|---> test_dataset {type(test_dataset)}')
    
    if 1: # train

        print('|===> run the training loop')
        gan.fit(train_dataset, test_dataset, args)     # , summary_writer

    print(f'|===> end nnart')


#   ******************
#   nncrys - 22
#   
def nncrys(args, kwargs):

    args = onutil.pargs(vars(args))
    args.PROJECT = 'crystals'
    args.DATASET = 'quimica_tech'
    args.LOCALLAB = 0
    xp = getxp(vars(args))
    args = onutil.pargs(xp)
    onutil.ddict(vars(args), 'args')

    print(f"|---> nncrys: {args.PROJECT}:  \n ")

    if 1: # tree
        assert(os.path.exists(args.dataorg_dir))

        args.ckpt_dir = args.models_dir

        args.data_train_dir = os.path.join(args.data_dir, 'train_dir')
        args.data_test_dir = os.path.join(args.data_dir, 'test_dir')

        args.results_dir = os.path.join(args.proj_dir, 'results') # in project dir

        args.dataset_train_B_dir = os.path.join(args.dataset_dir, 'train_B')
        args.dataset_train_A_dir = os.path.join(args.dataset_dir, 'train_A')
        args.dataset_test_B_dir = os.path.join(args.dataset_dir, 'test_B')
        args.dataset_test_A_dir = os.path.join(args.dataset_dir, 'test_A')

        os.makedirs(args.models_dir, exist_ok=True)
        os.makedirs(args.logs_dir, exist_ok=True)

        os.makedirs(args.data_train_dir, exist_ok=True) 
        os.makedirs(args.data_test_dir, exist_ok=True) 


        os.makedirs(args.dataset_train_B_dir, exist_ok=True) 
        os.makedirs(args.dataset_train_A_dir, exist_ok=True) 
        os.makedirs(args.dataset_test_B_dir, exist_ok=True) 
        os.makedirs(args.dataset_test_A_dir, exist_ok=True) 

        os.makedirs(args.models_dir, exist_ok=True)
        os.makedirs(args.logs_dir, exist_ok=True)
        os.makedirs(args.tmp_dir, exist_ok=True)
        os.makedirs(args.records_dir, exist_ok=True)
        os.makedirs(args.results_dir, exist_ok=True)


    if args.verbose: print(f"|---> nncrys tree: {args.PROJECT}:  \n \
    cwd: {os.getcwd()} \n \
    args.PROJECT:        {args.PROJECT} \n \
    args.DATASET:        {args.DATASET} \n \
    args.proj_dir:       {args.proj_dir} \n \
    args.dataorg_dir:    {args.dataorg_dir} :-: {onfile.qfiles(args.dataorg_dir, '*.jpg')}  jpgs \n \
    args.data_dir:       {args.data_dir} :-: {onfile.qfiles(args.data_dir, '*')} items \n \
    args.data_train_dir: {args.data_train_dir} :-: {onfile.qfiles(args.data_train_dir, '*.jpg')} jpgs \n \
    args.data_test_dir:  {args.data_test_dir} :-: {onfile.qfiles(args.data_test_dir, '*.jpg')} jpgs \n \
    args.dataset_dir:    {args.dataset_dir} :-: {onfile.qfolders(args.dataset_dir)} folders\n \
    args.logs_dir:       {args.logs_dir} {onfile.qfiles(args.models_dir, '*')} logs \n \
    args.results_dir:       {args.results_dir} {onfile.qfiles(args.results_dir, '*')} results \n \
    args.dataset_train_B_dir: {args.dataset_train_B_dir} :-: {onfile.qfiles(args.dataset_train_B_dir, '*.jpg')}  jpgs \n \
    args.dataset_train_A_dir: {args.dataset_train_A_dir} :-: {onfile.qfiles(args.dataset_train_A_dir, '*.jpg')}  jpgs \n \
    args.dataset_test_B_dir:  {args.dataset_test_B_dir} :-: {onfile.qfiles(args.dataset_test_B_dir, '*.jpg')}  jpgs \n \
    args.dataset_test_A_dir:  {args.dataset_test_A_dir} :-: {onfile.qfiles(args.dataset_test_A_dir, '*.jpg')}  jpgs \n \
    args.records_dir:    {args.records_dir} :-: {onfile.qfiles(args.records_dir, '*')} files \n \
    args.models_dir:     {args.models_dir} :-: {onfile.qfiles(args.models_dir, '*.index')} snaps \n \
    ")


    if 1: # config

        args.ckpt_prefix = 'ckpt-'

        args.img_height = 512
        args.img_width = 512

        args.max_size=512

        args.height=args.img_height
        args.width=args.img_height
        args.batch_size=1
        args.input_shape = [args.height, args.width, args.input_channels]

    if args.verbose: print(f'|--->  {args.PROJECT}: config  \n \
        args.height:         {args.height} \n \
        args.width:          {args.width} \n \
        args.max_size:       {args.max_size} \n \
        args.input_channels: {args.input_channels} \n \
        args.buffer_size:    {args.buffer_size} \n \
        args.batch_size:     {args.batch_size} \n \
        args.input_shape:    {args.input_shape} \n \
    ')


    if 0: # clear tree

        print(f"clear tree at {args.proj_dir}")
        onfile.clearfolder(args.proj_dir, inkey=args.PROJECT)


    if 0: # make gif out of results: all generatd imgs with patterns

        imgpatts = ['0012*.png']
        dstpath = os.path.join(args.results_dir, 'out.gif')

        nresults = args.results_dir
        srcdir = nresults # args.results_dir

        print(f"|===> tovid \n \
            args.results_dir: {args.results_dir} \n \
            nresults: {nresults} \n \
            srcdir: {srcdir} \n \
        ")
        onvid.folder_to_gif(srcdir, dstpath, patts=imgpatts)


    if 1: # 	model

        print(f"|===> get model from {args.models_dir} \n ")		
        model = GAN( 
            models_dir = args.models_dir,
            logs_dir = args.logs_dir,
            results_dir = args.results_dir,
            ckptidx = -1,
            ckpt_dir = args.ckpt_dir,
            ckpt_prefix = args.ckpt_prefix,
            input_shape = args.input_shape,
            output_shape = args.input_shape,
        )


    if 1: # raw images to data (dataorg_dir => data_train_dir, data_test_dir)

        trainpc = 0.9
        testpc = 0.1
    
        input_paths = onfile.folder_to_paths(args.dataorg_dir)
        q_from_imgs = len(input_paths)

        q_train_imgs = int(q_from_imgs * trainpc)
        q_test_imgs = int(q_from_imgs * testpc)

        train_paths = input_paths[0:q_train_imgs]
        test_paths = input_paths[0:q_test_imgs]

        q_from_train_imgs = len(train_paths)
        q_from_test_imgs = len(test_paths)

        q_to_train_imgs = onfile.qfiles(args.data_train_dir)
        q_to_test_imgs = onfile.qfiles(args.data_test_dir)

        print(f"|===>  org raw to data \n \
            {len(train_paths)} to {args.data_train_dir} with {q_to_train_imgs} \n \
            {len(test_paths)} to {args.data_test_dir} with {q_to_test_imgs} \n \
        ")

        # raw images to train data (args.dataorg_dir => args.data_train_dir)

        if not q_from_train_imgs == q_to_train_imgs:		
            onfile.paths_to_folder_with_cv(train_paths, args.data_train_dir)
        else:
            print(f"|... !!! train not copied. {args.data_train_dir} got them")

        # raw images to test data (args.dataorg_dir => args.data_test_dir)

        if not q_from_test_imgs == q_to_test_imgs:			
            onfile.paths_to_folder_with_cv(test_paths, args.data_test_dir)
        else:
            print(f"|... !!! test not  copied. {args.data_test_dir} got them")
        
        if 1:  # show raw images shapes
            nuas = onfile.folder_to_nuas(args.data_train_dir)
            n = np.random.randint(0, len(nuas))
            print(f"|... data train image {n} shape: {np.shape(nuas[n])}")

    if 1: # data to dataset B (data_dir => (train, test))

        args.keep_folder=True    
        args.process_type='crop_to_square'
        args.file_extension= 'jpg'
        args.name=1 # alpha name

        print(f"|===> data to dataset B (real) \n \
            {args.data_train_dir} to {args.dataset_train_B_dir}   \n \
            {args.data_test_dir}  to {args.dataset_test_B_dir} \n \
            args.process_type:  {args.process_type} \n \
            args.file_extension:{args.file_extension} \n \
            args.max_size:{args.max_size} \n \
        ")


        # dataset to train data

        args.input_folder = args.data_train_dir
        args.output_folder = args.dataset_train_B_dir

        q_from_train_imgs = onfile.qfiles(args.input_folder)
        q_to_train_imgs = onfile.qfiles(args.output_folder)

        print(f'|==> data to dataset train: {args.process_type} \n \
            {args.input_folder} ==> {args.output_folder} \n \
            q_from_train_imgs: {onfile.qfiles(args.input_folder)} \n \
            q_to_train_imgs: {onfile.qfiles(args.output_folder)} \n \
        ')

        if not q_from_train_imgs == q_to_train_imgs: # dir empty				
            onset.processFolder(args) # copy reals to train
        else:
            print(f"|... nothing copied. {args.output_folder} not empty")

        assert onfile.qfiles(args.output_folder) > 0, f're train files not found {args.output_folder}'

        # show train images shapes

        if 1:
            nuas = onfile.folder_to_nuas(args.dataset_train_B_dir)
            n = np.random.randint(0, len(nuas))
            print(f"|---> train image {n} shape: {np.shape(nuas[n])}")		

        # dataset to test data

        args.input_folder = args.data_test_dir
        args.output_folder = args.dataset_test_B_dir

        q_from_test_imgs = onfile.qfiles(args.input_folder)
        q_to_test_imgs = onfile.qfiles(args.output_folder)

        print(f'|==> data to dataset test: {args.process_type} \n \
            {args.input_folder} ==> {args.output_folder} \n \
            q_from_test_imgs: {onfile.qfiles(args.input_folder)} \n \
            q_to_test_imgs: {onfile.qfiles(args.output_folder)} \n \
        ')

        if not q_from_test_imgs == q_to_test_imgs: # dir empty				
            onset.processFolder(args) # copy reals to test
        else:
            print(f"|... nothing copied. {args.output_folder} not empty")


    if 1: # canny to dataset A ((train, test)B => (train, test)A)

        args.keep_folder=True    
        args.process_type='canny' # 'square' # 'canny-pix2pix'
        args.direction='B2A'

        args.file_extension='jpg'
        args.blur_type='gaussian'
        args.blur_amount=3
        args.name=1
        args.mirror=None
        args.rotate=None

        if args.verbose: print(f"|===>  canny B ==> dataset A (sketch):   \n \
            args.process_type:   {args.process_type} \n \
            args.direction:   	 {args.direction} \n \
            args.blur_type:   	 {args.blur_type} \n \
            args.blur_amount:    {args.blur_amount} \n \
            args.file_extension: {args.file_extension} \n \
            from {args.dataset_train_B_dir}  \n \
                to {args.dataset_train_A_dir} \n \
            from {args.dataset_test_B_dir}  \n \
                to {args.dataset_test_A_dir} \n \
        ")

        # data to dataset train B

        args.input_folder = args.dataset_train_B_dir
        args.output_folder = args.dataset_train_A_dir

        if not os.listdir(args.output_folder): # dir empty
            print(f"|---> train canny {args.process_type}: {args.input_folder}   \n \
                to {args.output_folder},  \n \
                from imgs q: {onfile.qfiles(args.input_folder)}	\n \
                to imgs q: {onfile.qfiles(args.output_folder)}	\n \
            ")
            onset.processFolder(args) # canny's to train

            #convert grey to rgb
            for filename in os.listdir(args.output_folder):

                file_path = os.path.join(args.output_folder, filename)
                img = cv2.imread(file_path)    

                canny = cv2.cvtColor(Onset.processCanny(img,args),cv2.COLOR_GRAY2RGB)

                if(args.file_extension == "png"):
                    new_file = os.path.splitext(filename)[0] + ".png"
                    cv2.imwrite(os.path.join(args.output_folder, new_file), canny, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                elif(args.file_extension == "jpg"):
                    new_file = os.path.splitext(filename)[0] + ".jpg"
                    cv2.imwrite(os.path.join(args.output_folder, new_file), canny, [cv2.IMWRITE_JPEG_QUALITY, 90])

        else:
            print(f"|---> nothing copied. {args.output_folder} not empty")


        # data to dataset test B

        args.input_folder = args.dataset_test_B_dir
        args.output_folder = args.dataset_test_A_dir

        if not os.listdir(args.output_folder): # dir  empty			
            print(f"|---> test canny {args.process_type}: {args.input_folder}   \n \
                to {args.output_folder},  \n \
                from imgs q: {onfile.qfiles(args.input_folder)}	\n \
                to imgs q: {onfile.qfiles(args.output_folder)}	\n \
            ")
            onset.processFolder(args) # canny's to test

            #convert grey to rgb
            for filename in os.listdir(args.output_folder):

                file_path = os.path.join(args.output_folder, filename)
                img = cv2.imread(file_path)    

                canny = cv2.cvtColor(Onset.processCanny(img,args),cv2.COLOR_GRAY2RGB)

                if(args.file_extension == "png"):
                    new_file = os.path.splitext(filename)[0] + ".png"
                    cv2.imwrite(os.path.join(args.output_folder, new_file), canny, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                elif(args.file_extension == "jpg"):
                    new_file = os.path.splitext(filename)[0] + ".jpg"
                    cv2.imwrite(os.path.join(args.output_folder, new_file), canny, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
        else:
            print(f"|---> nothing copied. {args.output_folder} not empty")


    if 1: #  data_train_dir => tf datasets 22

        print(f"|===> data to datasets 22")

        pths_train = [args.dataset_train_A_dir, args.dataset_train_B_dir, ]
        pths_test = [args.dataset_test_A_dir,args.dataset_test_B_dir, ]
        args.patts = ['*.jpg', '*.jpg']

        print(f"|--->  data to dataset A:   \n \
            train_dataset from {pths_train} with args.patts {args.patts} \n \
            test_dataset from {pths_test} with args.patts {args.patts} \n \
            height {args.height} \n \
            width {args.width} \n \
            buffer_size {args.buffer_size} \n \
            batch_size {args.height} \n \
        ")

        train_dataset = paths_to_dataset_22(
            pths_train, 
            args.patts, 
            height=args.height, width=args.width, buffer_size=args.buffer_size, batch_size=args.batch_size)
        
        test_dataset = paths_to_dataset_22(
            pths_test, 
            args.patts, 
            height=args.height, width=args.width, buffer_size=args.buffer_size, batch_size=args.batch_size)


    if 0: # probe train dataset

        print(f"|===> probe train dataset")

        iterator = train_dataset.take(1) # <class 'tensorflow.python.data.ops.dataset_ops.TakeDataset'>
        for elem in iterator:
            #itemlist = list(elem)
            example_input, example_target = elem
            if args.verbose:
                print(f'|---> probe train dataset {type(example_input)} {type(example_target)} ')
            img1 = onformat.nnba_to_rgb(example_input)
            img2 = onformat.nnba_to_rgb(example_target)
                      
            display_list = [img1, img2]
            onplot.pil_show_rgbs(display_list, scale=1, rows=1) 



    if 0: #   	data => tfrecords

        print(f"|===> data to tfrecords   \n \
        from: {args.dataset_train_A_dir} \n \
            to args.records_dir: {args.records_dir} \n \
        ")		
        # <class 'numpy.ndarray'> (512, 1024, 3)
        onrecord.folder_to_tfrecords(
            args.dataset_train_A_dir, 
            args.records_dir)


    if 1: # walk ckpt models

        maxitems = 40
        patts = ['*.index']
        paths = onfile.path_to_paths(model.ckpt_manager.directory, patts)

        print(f'|===> walk ckpt models \n \
            ckpt_manager: {model.ckpt_manager.directory} \n \
            ckpt_manager.latest_checkpoint: {model.ckpt_manager.latest_checkpoint} \n \
            ckpt_manager.checkpoints: {model.ckpt_manager.checkpoints} \n \
            ckpt_manager.directory: {model.ckpt_manager.directory} \n \
            patts: {patts} \n \
            paths: {len(paths)} \n \
        ')

        for path in paths:
            filename = os.path.basename(path)
            ckpt = filename.split('.')[0] # {prefix}-{ckptidx}.index => {prefix}-{ckptidx}
            ckptidx = getckptidx(ckpt)

        ckptidxs = []
        for path in paths:
            filename = os.path.basename(path)
            infix = filename.split('.')[0]

            ckptidx = getckptidx(infix)

            ckptidxs.append(int(ckptidx)) # if None
        ckptidxs = sorted(ckptidxs)
        qckptidxs = len(ckptidxs)


        mod = int(qckptidxs/maxitems)
        print(f'|===> walk ckpt models with mod {mod}')


        ckptidxs=[]
        for i,idx in enumerate(range(qckptidxs-1)):
            if i % mod == 1:
                ckptidxs.append(idx)
        ckptidxs.append(len(ckptidxs))
        

        for ckptidx in ckptidxs:
            print(f"|===> walk ckptidxs with ckpt: {ckpt}")
            model = GAN( 
                models_dir = args.models_dir,
                logs_dir = args.logs_dir,
                results_dir = args.results_dir,
                ckptidx = ckptidx,  # 1, None, -1
                ckpt_dir = args.ckpt_dir,
                ckpt_prefix = args.ckpt_prefix, # 'ckpt-'
                input_shape = args.input_shape,
                output_shape = args.input_shape,
            )
            onplot.plot_iter_grid(model, test_dataset, 1, 3, figsize = (6.4, 6.3), do=['save']) # do=['plot', 'save']


    if 1: # ckpt models to gif
        fromfolder = args.results_dir
        dstpath = os.path.join(args.results_dir, 'out.gif')
        patts= ['frame*']   

        print(f'|===>  gif of ckpt models \n \
            fromfolder: {fromfolder} \n \
            dstpath: {dstpath} \n \
            patts: {patts}  \n \
        ')

        onvid.folder_to_gif(fromfolder, dstpath, patts)
        #_folder_to_gif(fromfolder, dstpath, patts)


    if 0: #  	colab

        if onutil.incolab():
            print("|---> load tensorboard to monitor logs with colab")
            os.system(f"load_ext tensorboard")
            os.system(f"tensorboard --logdir {args.logs_dir}")
        else:
            print(f"|---> launch a separate tensorboard process to monitor logs with colab")


    if 0: # walk test_dataset

        print(f"|===> walk the test_dataset")
        onplot.plot_iter_grid(model, test_dataset, 3, 3, figsize = (6.4, 6.3), do=['plot', 'save'])


    if 0: # show train first

        print(f"|===> show first in train dataset")
        for train_input, train_target in train_dataset.take(1):
            prediction = model.generator(train_input, training=True) # _e_
            display_list = [
                onformat.nnba_to_rgb(train_input),
                onformat.nnba_to_rgb(train_target),
                onformat.nnba_to_rgb(prediction)
            ]
            onplot.pil_show_rgbs(display_list, scale=1, rows=1)   

    if 0: # show test first

        print(f"|===> show first in test dataset")
        for test_input, test_target in test_dataset.take(1):
            prediction = model.generator(test_input, training=True) # _e_
            display_list = [
                onformat.nnba_to_rgb(test_input),
                onformat.nnba_to_rgb(test_target),
                onformat.nnba_to_rgb(prediction)
            ]
            onplot.pil_show_rgbs(display_list, scale=1, rows=1)        

    if 0: # show all in dataset

        print(f"|===> show all in test dataset")
        for inp,tar in test_dataset.as_numpy_iterator():
            prediction = model.generator(inp, training=True) # _e_
            display_list = [
                onformat.nnba_to_rgb(inp),
                onformat.nnba_to_rgb(tar),
                onformat.nnba_to_rgb(prediction)
            ]
            onplot.pil_show_rgbs(display_list, scale=1, rows=1)   


    if 0: # train

        print(f"|===> train loop")

        model.fit(train_dataset, test_dataset, args)

    print(f'|===> end nncrys')

#   ******************
#   nnleonardo - 22
#   
def nnleonardo(args, kwargs):

    args = onutil.pargs(vars(args))
    args.PROJECT = 'leonardo'
    args.DATASET = 'leonardo'
    xp = getxp(vars(args))
    args = onutil.pargs(xp)
    onutil.ddict(vars(args), 'args')

    print(f"|---> nnleonardo: {args.PROJECT}:   \n ")

    if 1: # tree

        args.dataorg_train_dir = os.path.join(args.dataorg_dir, 'train')
        args.dataorg_test_dir = os.path.join(args.dataorg_dir, 'test')

        assert(os.path.exists(args.dataorg_dir))
        assert(os.path.exists(args.dataorg_train_dir))
        assert(os.path.exists(args.dataorg_test_dir))

        args.ckpt_dir = args.models_dir
        args.ckpt_prefix = os.path.join(args.ckpt_dir, "ckpt-")

        ''' train/test images in origin with pattern '''
        args.data_train_B_dir = os.path.join(args.data_dir, 'train_B')
        args.data_train_A_dir = os.path.join(args.data_dir, 'train_A')
        args.data_test_B_dir = os.path.join(args.data_dir, 'test_B')
        args.data_test_A_dir = os.path.join(args.data_dir, 'test_A')

        args.dataset_train_B_dir = os.path.join(args.dataset_dir, 'train_B')
        args.dataset_train_A_dir = os.path.join(args.dataset_dir, 'train_A')
        args.dataset_test_B_dir = os.path.join(args.dataset_dir, 'test_B')
        args.dataset_test_A_dir = os.path.join(args.dataset_dir, 'test_A')

        args.zipfile = os.path.join(args.dataorg_dir, f'{args.DATASET}.zip')

        os.makedirs(args.data_train_B_dir, exist_ok=True) 
        os.makedirs(args.data_train_A_dir, exist_ok=True) 
        os.makedirs(args.data_test_B_dir, exist_ok=True) 
        os.makedirs(args.data_test_A_dir, exist_ok=True) 

        os.makedirs(args.dataset_train_B_dir, exist_ok=True) 
        os.makedirs(args.dataset_train_A_dir, exist_ok=True) 
        os.makedirs(args.dataset_test_B_dir, exist_ok=True) 
        os.makedirs(args.dataset_test_A_dir, exist_ok=True) 

        os.makedirs(args.models_dir, exist_ok=True)
        os.makedirs(args.logs_dir, exist_ok=True)
        os.makedirs(args.tmp_dir, exist_ok=True)

    if args.verbose: print(f"|---> tree: {args.PROJECT}:   \n \
        args.PROJECT:    	  {args.PROJECT} \n \
        args.DATASET:    	  {args.DATASET} \n \
        args.dataorg_dir:     {args.dataorg_dir} \n \
        args.dataorg_train_dir: {args.dataorg_train_dir} \n \
        args.dataorg_test_dir: {args.dataorg_test_dir} \n \
        args.ckpt_dir:  {args.ckpt_dir} \n \
        args.logs_dir:        {args.logs_dir} \n \
        args.tmp_dir:        {args.tmp_dir} \n \
        args.zipfile:         {args.zipfile} \n \
        args.verbose:         {args.verbose}, \n \
        args.visual:          {args.visual}, \n \
    ")


    if 1: # config
        args.height = args.img_height
        args.width = args.img_width
        args.buffer_size = args.buffer_size
        args.batch_size = args.batch_size
        args.input_channels = args.input_channels
        args.input_shape = [args.height, args.width, args.input_channels]
        if 0:
            ''' copy org to same data folder'''
            args.patts = ['*in.png', '*re.png']
        else:
            ''' will separate images in data'''
            args.patts = ['*.png', '*.png']

    if args.verbose: print(f"|---> nnleonardo config:   \n \
        args.max_epochs:            {args.max_epochs}, \n \
        args.output_channels:	{args.output_channels} \n \
        args.height: 			{args.height} \n \
        args.width: 			{args.width} \n \
        args.input_channels: 	{args.input_channels} \n \
        args.buffer_size: 		{args.buffer_size} \n \
        args.batch_size: 		{args.batch_size} \n \
        args.input_shape: 		{args.input_shape} \n \
        args.patts:     		{args.patts}, \n \
    ")

    if args.visual > 1: # show ref images

        ref_pict_path = os.path.join(args.dataorg_train_dir, 'da02_re.png')
        ref_draw_path = os.path.join(args.dataorg_train_dir, 'da02_in.png')

        re = onfile.path_to_rgb(ref_pict_path)
        inp = onfile.path_to_rgb(ref_draw_path)

        down_model = ondata.downsample(3, 4)
        down_result = down_model(tf.expand_dims(tf.cast(inp, tf.float32), 0)) # _e_

        up_model = ondata.upsample(3, 4)
        up_result = up_model(down_result)

        print(f"|---> show ref images:   \n \
            ref_pict_path: {ref_pict_path} \n \
            ref_draw_path: {ref_draw_path} \n \
            down_result.shape: {down_result.shape} \n \
            up_result.shape: {up_result.shape} \n \
            args.input_shape: {args.input_shape} \n \
            re shape: {np.shape(re)} \n \
            inp shape: {np.shape(inp)} \n \
        ")

        print(f"|---> plot re, inp ...")
        onplot.pil_show_rgbs([re, inp])


    if 1: #  dataorg raw => data train 

        print(f"|---> raw to data")

        args.keep_folder=True    
        args.process_type='crop_to_square'
        args.file_extension= 'png'
        args.name=1
        args.zfill=0

        args.input_folder = args.dataorg_train_dir
        args.filepatt = f'.*in.{args.file_extension}'
        #args.filepatt = f'*' => re.error: nothing to repeat at position 0
        args.output_folder = args.data_train_A_dir

        qinfiles = onfile.qfiles(args.input_folder, f'*in.{args.file_extension}')
        qoutfiles = onfile.qfiles(args.output_folder)

        print(f"|---> nnleonardo train copy: {args.process_type}: \n \
            {args.input_folder} ==> {args.output_folder}, \n \
            qinfiles : {qinfiles}, \n \
            qoutfiles : {qoutfiles}, \n \
        ")


        if  qinfiles > qoutfiles:
            onset.processFolder(args) # copy inputs to train
        else:
            print(f'|... no processFolder !!!!. files already there ')


        args.filepatt = f'.*re.{args.file_extension}'
        args.output_folder = args.data_train_B_dir

        qinfiles = onfile.qfiles(args.input_folder, f'*in.{args.file_extension}')
        qoutfiles = onfile.qfiles(args.output_folder)

        print(f"|---> train copy: {args.process_type}: \n \
            {args.input_folder} ==> {args.output_folder}, \n \
            qinfiles : {qinfiles}, \n \
            qoutfiles : {qoutfiles}, \n \
        ")

        if  qinfiles > qoutfiles:
            onset.processFolder(args) # copy reals to train
        else:
            print(f'|... no processFolder !!!!. files already there ')


    if 1: #  dataorg_test_dir raw => data_test_dir (A/B) data
        args.keep_folder=True    
        args.process_type='crop_to_square'
        args.file_extension= 'png'
        args.name=1
        args.zfill=0

        args.input_folder = args.dataorg_test_dir
        args.filepatt = f'.*in.{args.file_extension}'
        args.output_folder = args.data_test_A_dir

        args.filepatt = f'.*re.{args.file_extension}'
        args.output_folder = args.data_train_B_dir

        print(f"|---> test copy: {args.process_type}: \n \
            {args.input_folder} ==> {args.output_folder}, \n \
            qinfiles : {qinfiles}, \n \
            qoutfiles : {qoutfiles}, \n \
        ")
        if not qinfiles == qoutfiles:
            onset.processFolder(args) # copy inputs to test
        else:
            print(f'|... no processFolder !!!!. files there ')

        args.filepatt = f'.*re.{args.file_extension}'
        args.output_folder = args.data_test_B_dir


        args.filepatt = f'.*re.{args.file_extension}'
        args.output_folder = args.data_train_B_dir

        print(f"|---> test copy: {args.process_type}:  \n \
            {args.input_folder} ==> {args.output_folder},\n \
            qinfiles : {qinfiles}, \n \
            qoutfiles : {qoutfiles}, \n \
        ")
        if not qinfiles == qoutfiles:
            onset.processFolder(args) # copy reals to test
        else:
            print(f'|... no processFolder !!!!. files there ')


    if 1: #  data to dataset (A/B) data
        print(f"|---> data to dataset")

        args.keep_folder=True    
        args.process_type='crop_to_square'
        args.file_extension= 'png'
        args.name=1
        args.zfill=0

        # data_train_dir raw => dataset_train_dir (A/B) data

        args.input_folder = args.data_train_A_dir
        args.filepatt = f'.*.{args.file_extension}'
        args.output_folder = args.dataset_train_A_dir

        qinfiles = onfile.qfiles(args.input_folder, f'*.{args.file_extension}')
        qoutfiles = onfile.qfiles(args.output_folder)

        print(f"|---> train copy: {args.process_type}:   \n \
            {args.input_folder} ==> {args.output_folder}, \n \
            qinfiles : {qinfiles}, \n \
            qoutfiles : {qoutfiles}, \n \
        ")
        if not qinfiles == qoutfiles:
            onset.processFolder(args) # copy inputs to train
        else:
            print(f'|... no processFolder !!!!. files there ')

        args.input_folder = args.data_train_B_dir
        args.filepatt = f'.*.{args.file_extension}'
        args.output_folder = args.dataset_train_B_dir

        qinfiles = onfile.qfiles(args.input_folder, f'*.{args.file_extension}')
        qoutfiles = onfile.qfiles(args.output_folder)

        print(f"|---> train copy: {args.process_type}:   \n \
            {args.input_folder} ==> {args.output_folder}, \n \
            qinfiles : {qinfiles}, \n \
            qoutfiles : {qoutfiles}, \n \
        ")
        if not qinfiles == qoutfiles:
            onset.processFolder(args) # copy reals to train
        else:
            print(f'|... no processFolder !!!!. files there ')

        # data_test_dir raw => dataset

        args.input_folder = args.data_test_A_dir
        args.filepatt = f'.*in.{args.file_extension}'
        args.output_folder = args.dataset_test_A_dir

        qinfiles = onfile.qfiles(args.input_folder, f'*in.{args.file_extension}')
        qoutfiles = onfile.qfiles(args.output_folder)

        print(f"|---> test copy: {args.process_type}:   \n \
            {args.input_folder} ==> {args.output_folder}, \n \
            qinfiles : {qinfiles}, \n \
            qoutfiles : {qoutfiles}, \n \
        ")
        if not qinfiles == qoutfiles:
            onset.processFolder(args) # copy inputs to test
        else:
            print(f'|... no processFolder !!!!. files there ')

        args.input_folder = args.data_test_B_dir
        args.filepatt = f'.*re.{args.file_extension}'
        args.output_folder = args.dataset_test_B_dir


        qinfiles = onfile.qfiles(args.input_folder, f'*re.{args.file_extension}')
        qoutfiles = onfile.qfiles(args.output_folder)

        print(f"|---> test copy: {args.process_type}:   \n \
            {args.input_folder} ==> {args.output_folder}, \n \
            qinfiles : {qinfiles}, \n \
            qoutfiles : {qoutfiles}, \n \
        ")
        if not qinfiles == qoutfiles:
            onset.processFolder(args) # copy reals to test
        else:
            print(f'|... no processFolder !!!!. files there ')

    if 1: #  data_train_dir => tf datasets

        print(f"|---> raw to datasets")

        pths_train = [args.data_train_A_dir, args.data_train_B_dir]
        train_dataset = paths_to_dataset_22(
            pths_train, 
            args.patts, 
            height=args.height, width=args.width, buffer_size=args.buffer_size, batch_size=args.batch_size)

        pths_test = [args.data_test_A_dir, args.data_test_B_dir]
        test_dataset = paths_to_dataset_22(
            pths_test, 
            args.patts, 
            height=args.height, width=args.width, buffer_size=args.buffer_size, batch_size=args.batch_size)
        

    if args.visual > 1: # probe train dataset

        print(f"|---> probe dataset")
        iter = train_dataset.take(1) # <class 'tensorflow.python.data.ops.dataset_ops.TakeDataset'>
        for i,elem in enumerate(iter):
            example_input, example_target = elem
            img1 = onformat.nba_to_rgb(example_input)
            img2 = onformat.nba_to_rgb(example_target)

            print(f'probe train dataset: {i} \n \
                input:{type(example_input)} \n \
                target: {type(example_target)}')
            display_list = [
                img1,
                img2,
            ]
            onplot.pil_show_rgbs(display_list, scale=1, rows=1) 


    if 1: # model
        print("|===> model")

        gan = GAN(
            models_dir = args.models_dir,
            logs_dir = args.logs_dir,
            ckpt_dir = args.ckpt_dir,
            ckpt_prefix = args.ckpt_prefix,
            input_shape = args.input_shape,
            output_shape = args.input_shape,
        )

    if 1: # colab

        if onutil.incolab():
            print("|---> load tensorboard to monitor logs with colab")
            os.system(f"load_ext tensorboard")
            os.system(f"tensorboard --logdir {args.logs_dir}")
        else:
            print("|---> launch a separate tensorboard process to monitor logs with colab")

    if args.visual: # try generator

        print("|===> probe generator with train image pair")

        img1 = os.path.join(args.data_dir, 'train_A/da09_in.png')
        img2 = os.path.join(args.data_dir, 'train_B/da09_re.png')

        imgs = paths_to_pair(img1, img2, args.img_height, args.img_width)

        print("|... img1", type(img1), np.shape(img1))
        print("|... img2", type(img2), np.shape(img2))

        img1 = tf.cast(imgs[0], tf.float32)[tf.newaxis,...]
        img2 = tf.cast(imgs[1], tf.float32)[tf.newaxis,...]

        prediction = gan.generator(img1, training=True) # _e_
        print("|... prediction", type(prediction), np.shape(prediction))

        img1 = onformat.nnba_to_rgb(img1)
        img2 = onformat.nnba_to_rgb(img2)
        img3 = onformat.nnba_to_rgb(prediction)

        display_list = [ img1, img2, img3 ]
        print("|... pil show rgbs")
        onplot.pil_show_rgbs(display_list, scale=1, rows=1)     



    if args.visual:  # demo dataset

        print("|===> probe train dataset prediction")
        for train_input, train_target in train_dataset.take(1):
                
            print("|--->.. train_input", type(train_input), np.shape(train_input))
            prediction = gan.generator(train_input, training=True) # _e_
            print("|--->.. prediction", type(prediction), np.shape(prediction))

            img1 = onformat.nnba_to_rgb(train_input)
            img2 = onformat.nnba_to_rgb(train_target) 
            img3 = onformat.nnba_to_rgb(prediction)

            display_list = [ img1, img2,  img3 ]
            onplot.pil_show_rgbs(display_list, scale=1, rows=1) 
                    

        if test_dataset:
            print("|---> probe test dataset prediction")
            for test_input, test_target in test_dataset.take(1):
                prediction = gan.generator(test_input, training=True) # _e_

                img1 = onformat.nnba_to_rgb(test_input)
                img2 = onformat.nnba_to_rgb(test_target) 
                img3 = onformat.nnba_to_rgb(prediction)

            display_list = [ img1, img2,  img3 ]
            onplot.pil_show_rgbs(display_list, scale=1, rows=1)        
    


    if 1: # train

        print("|===> training loop")
        gan.fit(train_dataset, test_dataset, args)     # , summary_writer

    print(f'|===> end nnleonardo')

#   ******************
#   nnfacades - 11
#   
def nnfacades(args, kwargs):

    args = onutil.pargs(vars(args))
    args.PROJECT = 'facades'
    args.DATASET = 'facades'
    xp = getxp(vars(args))
    args = onutil.pargs(xp)
    onutil.ddict(vars(args), 'args')

    print(f"|---> nnfacades: {args.PROJECT}:  \n ")

    if 1: # tree
        args.ckpt_dir = args.models_dir
        args.ckpt_prefix = os.path.join(args.ckpt_dir, "ckpt-")

        os.makedirs(args.models_dir, exist_ok=True)
        os.makedirs(args.logs_dir, exist_ok=True)

        assert(os.path.exists(args.dataorg_dir))

    if args.verbose: print(f"|---> {args.PROJECT} tree:   \n \
        args.ckpt_dir: {args.ckpt_dir} \n \
        args.logs_dir: {args.logs_dir} \n \
        args.output_channels: {args.output_channels} \n \
        args.batch_size: {args.batch_size}, \n \
        args.img_width: {args.img_width} \n \
        args.img_height: {args.img_height} \n \
        args.input_channels: {args.input_channels} \n \
        args.max_epochs: {args.max_epochs} \n \
    ")

    if 1: # config
        args.height = args.img_height
        args.width = args.img_width
        args.buffer_size = args.buffer_size
        args.batch_size = 1 # args.batch_size

        args.input_channels = args.input_channels
        args.input_shape = [args.height, args.width, args.input_channels]
        args.src_pattern = '*.jpg'  # superseed pat

    if args.verbose: print(f"|---> {args.PROJECT} config:   \n \
        cwd: {os.getcwd()} \n \
        args.PROJECT: {args.PROJECT} \n \
        args.height: {args.height} \n \
        args.width: {args.width} \n \
        args.input_channels: {args.input_channels} \n \
        args.buffer_size: {args.buffer_size} \n \
        args.batch_size: {args.batch_size} \n \
        args.input_shape: {args.input_shape}, \n \
        args.src_pattern: {args.src_pattern}, \n \
        args.verbose: {args.verbose}, \n \
        args.visual: {args.visual}, \n \
    ")

    if 1: # git
        onutil.get_git(args.AUTHOR, args.GITPOD, args.proj_dir)

    if args.visual: # ref images

        ref_img_path = os.path.join(args.dataorg_dir, 'train/100.jpg')        
        inp, re = path_to_decoded(ref_img_path)
        if args.visual : 
            onplot.pil_show_rgbs([re, inp])

        down_model = ondata.downsample(3, 4)
        down_result = down_model(tf.expand_dims(tf.cast(inp, tf.float32), 0)) # _e_
        up_model = ondata.upsample(3, 4)
        up_result = up_model(down_result)

        if args.verbose: print(f"|---> {args.PROJECT}: ref images:  \n \
            args.input_shape: {args.input_shape} \n \
            ref_img_path: {ref_img_path} \n \
            down_result.shape: {down_result.shape} \n \
            up_result.shape: {up_result.shape} \n \
        ")

    if 1: # get DATASETS from src

        path_train = os.path.join(args.dataorg_dir, 'train')
        path_test = os.path.join(args.dataorg_dir, 'val')

        train_dataset = path_to_dataset_11(path_train, '*.jpg',
            height=args.height,width=args.width,buffer_size=args.buffer_size,batch_size=args.batch_size)

        test_dataset = path_to_dataset_11(path_test, '*.jpg',
            height=args.height,width=args.width,buffer_size=args.buffer_size,batch_size=args.batch_size)


    if 1: # model
        print(f"|===> model")
        gan = GAN(
            models_dir = args.models_dir,
            logs_dir = args.logs_dir,
            ckpt_dir = args.ckpt_dir,
            ckpt_prefix = args.ckpt_prefix,
            input_shape = args.input_shape,
            output_shape = args.input_shape,
        )

    if 0: # # tensorboard
        if onutil.incolab():
            print("|---> load tensorboard to monitor logs with colab")
            os.system(f"load_ext tensorboard")
            os.system(f"tensorboard -- logdir {args.logs_dir}")
        else:
            print("launch a separate tensorboard process to monitor logs with colab")


    if args.visual: # 

        print(f"|---> show dataset train")
        for train_input, train_target in train_dataset.take(1):

            prediction = gan.generator(train_input, training=True) # _e_
            print("|... train_input", type(train_input), np.shape(train_input))
            print("|... train_target", type(train_target), np.shape(train_target))
            print("|... prediction", type(prediction), np.shape(prediction))
            
            img1 = onformat.nnba_to_rgb(train_input)
            img2 = onformat.nnba_to_rgb(train_target) 
            img3 = onformat.nnba_to_rgb(prediction)

            display_list = [img1, img2, img3]
            onplot.pil_show_rgbs(display_list, scale=1, rows=1)        

    if args.visual: 
        print(f"|---> show dataset test")
        for test_input, test_target in test_dataset.take(1):

            prediction = gan.generator(test_input, training=True) # _e_
            
            img1 = onformat.nnba_to_rgb(test_input)
            img2 = onformat.nnba_to_rgb(test_target) 
            img3 = onformat.nnba_to_rgb(prediction)

            display_list = [img1, img2, img3]
            onplot.pil_show_rgbs(display_list, scale=1, rows=1)        


    if 1: # train

        print(f"|===> training loop")
        gan.fit(train_dataset, test_dataset, args)     # , summary_writer

    print(f'|===> end nnfacades')

#   ******************
#   nngoya
#   
def nngoya(args, kwargs):

    args = onutil.pargs(vars(args))
    args.PROJECT = 'goya'
    args.DATASET = 'goya'
    xp = getxp(vars(args))
    args = onutil.pargs(xp)
    onutil.ddict(vars(args), 'args')

    print(f"|---> nngoya: {args.PROJECT}:  \n ")

    if 1: # data params
        args.src_pattern = '*.jpg'  # superseed pat
        height = args.img_height
        width = args.img_width
        buffer_size = args.buffer_size
        batch_size = args.batch_size
        input_channels = args.input_channels
        input_shape = [height, width, input_channels]

    if 1: # tree
        ckpt_dir = os.path.join(args.proto_dir, 'leonardo', 'Models')
        ckpt_prefix = os.path.join(ckpt_dir, "ckpt-")

        os.makedirs(args.data_dir, exist_ok=True)
        os.makedirs(args.models_dir, exist_ok=True)
        os.makedirs(args.logs_dir, exist_ok=True)

    if args.verbose: print(f"|---> nngoya tree:   \n \
        args.data_dir: {args.data_dir} \n \
        ckpt_dir: {ckpt_dir} \n \
        ckpt_prefix: {ckpt_prefix} \n \
        logs_dir: {args.logs_dir} \n \
    ")

    if args.verbose: print(f"|---> nngoya config:   \n \
        cwd: {os.getcwd()} \n \
        PROJECT: {args.PROJECT} \n \
        height: {height} \n \
        width: {width} \n \
        input_channels: {input_channels} \n \
        buffer_size: {buffer_size} \n \
        batch_size: {batch_size} \n \
        input_shape: {input_shape} \n \
    ")

    if 1: # model
        gan = GAN(
            models_dir = args.models_dir,
            logs_dir = args.logs_dir,
            ckpt_dir = ckpt_dir,
            ckpt_prefix = ckpt_prefix,
            input_shape = input_shape,
            output_shape = input_shape,
        )

    if 1: # try generator
        img1 = os.path.join(args.gdata, 'leonardo', 'train', 'da02_in.png')
        img1 = os.path.join(args.gdata, 'leonardo', 'test', 'da13_in.png')			
        img1 = os.path.join(args.gdata, 'Goya__Guerra', 'img0006.jpg')

        img1 = onfile.path_to_rgb(img1)
        img1 = img_jitter_random(img1, height, width)        
        img1 = onformat.rgb_to_nba(img1)
        img1 = tf.cast(img1, tf.float32)[tf.newaxis,...]

        prediction = gan.generator(img1, training=True) # _e_
        print(f"|---> prediction", type(prediction), np.shape(prediction))

        display_list = [
            onformat.nnba_to_rgb(img1),
            onformat.nnba_to_rgb(prediction)
        ]
        onplot.pil_show_rgbs(display_list, scale=1, rows=1)  

    print(f'|---> end nngoya')

def nninfo(args, kwargs):

    args = onutil.pargs(vars(args))
    onutil.ddict(vars(args), 'args')

    print(f"|---> pix2pix:  \n \
    nndanboo: implement https://github.com/lllyasviel/DanbooRegion \n \
    nncrys: train quimica_tech dataset\n \
    nnfacades: train facades dataset {args.dataorg_dir} \n \
    nnleonardo: train leonardo dataset {args.dataorg_dir} \n \
    nninfo: show commands \n \
    tree, data,  \n \
    ref: https://github.com/NVIDIA/pix2pixHD \n \
    ref: https://www.tensorflow.org/tutorials/generative/pix2pix \n \
    \n \
    citations:\n \
        InProceedings=DanbooRegion2020,\n \
        author=Lvmin Zhang, Yi JI, and Chunping Liu, \n \
        booktitle=European Conference on Computer Vision (ECCV), \n \
        title=DanbooRegion: An Illustration Region Dataset, \n \
        year=2020, \n \
    ")


#   ******************
#
#   MAIN
#
#   ******************
def main():

    parser = argparse.ArgumentParser(description='''Run 'python %(prog)s <subcommand> --help' for subcommand help.''')

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
            print("|---> call %s" %name)
            globals()[name](args, kwargs) # pass args to nn cmd

# ---------------------------------------------------------------
# python base/base.py nninfo
if __name__ == "__main__":
    print("|--->", __name__)
    main()

#---------------------------------------------------------------
