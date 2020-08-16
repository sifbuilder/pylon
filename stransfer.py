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

from absl import app
from absl import flags
from absl import logging

tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print(f'|===> {tf.__version__}')

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
onvgg = Onvgg()
onlllyas = Onlllyas()


#   *******************
#   CONTEXT
#
#   *******************

def getap():
	cp = {
        "primecmd": 'nnimg',        

        "MNAME": "stransfer",      
        "AUTHOR": "xueyangfu",      
        "PROJECT": "building",      
        "GITPOD": "neural-style-tf",      
        "DATASET": "styler",        
    
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


	tree = ontree.tree(cp)
	for key in tree.keys():
		cp[key] = tree[key]

	hp = {
        "verbose": 0,
        "visual": 1,
	}

	ap = {}
	for key in cp.keys():
		ap[key] = cp[key]
	for key in hp.keys():
		ap[key] = hp[key]
	return ap


def getxp(cp):

    yp = {
        # "model_weights": os.path.join(cp["gmodel"], 'vgg', 'vgg19_weights_tf_dim_ordering_tf_kernels.h5'),
        "model_weights": os.path.join(cp["gmodel"], 'vgg', 'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'), # vgg19
        "model_include_top": 0,
        "model_classes": 1000,

        "max_epochs": 100,
        "steps_per_epoch": 1,
        "print_iterations": 10, #  50
        "max_iterations": 100, # 1000

        # "optimizer": "Adam", # "SGD", # 
        "adam_learning_rate": 0.01,
        "adam_beta_1": 0.99,
        "adam_epsilon": 1e-1,

        "optimizer": "SGD", # "Adam", # "SGD", # 
        "sgd_learning_rate": 0.01, 
        "learning_rate": 1e0,
        "model_pooling": 'avg',

        "exp_decay": [100., 100, 0.96], # [initial_learning_rate, decay_steps, decay_rate]
        # "momentum": 0.0, 
        # "nesterov": 0


        "content_imgs_files": [],
        "content_imgs_dir": cp["data_dir"],
        "content_imgs_weights": [1.0],
        "content_layers": ['conv4_2'], # ['conv4_2']
        "content_layer_weights": [1e4],
        # "content_layers_weights": [1e4], 
        # "content_layers_weights": [1e2], 
        "content_layers_weights": [2.5 * 1e-8], 
        "content_layer_weights": [1.0],
        "content_loss_function": 1,
        "content_weight": 5e0,
        "content_frame_frmt": 'frame{}.jpg',

        "style_imgs_dir": cp["data_dir"],
        "style_scale": 1,
        "style_imgs_weights": [1.0],
        # "style_layers": ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'],
        "style_layers": ['conv1_1','conv2_1','conv3_1','conv4_1','conv5_1'],
        # "style_layers_weights": [1e-2],
        # "style_layers_weights": [1e2],  
        # "style_layers_weights": [1e-6 / 100],  
        # "style_layers_weights": [1e-2, 1e-2, 1e-2, 1e-2, 1e-2],
        "style_layers_weights": [0.2, 0.2, 0.2, 0.2, 0.2],
        "style_mask_imgs": [],
        "style_weight": 1e4,

        "max_size": 512,
        "input_shape": (224, 224, 3), # (512, 512, 3) # 
        "first_frame_iterations": 2000,
        "first_frame_type": 'content', # args.first_frame_type ['random', 'content', 'style'
        "init_image_type": "content", # 'content','style','init','random','prev','prev_warped'
        "init_frame_type": 'prev',
        "init_img_dir": cp["data_dir"],
        "init_image_type": "content",
        "first_frame_iterations": 2000,
        "init_img_type": 'content',
        "img_name": 'result',

        "style_mask": None,
        "original_colors": None,
        "color_convert_type": 'yuv',
        "color_convert_time": 'after',
        "optimizer": 'adam', # 'lbfgs' # 
        
        # "total_variation_weight": 1e-6, # 300,
        "total_variation_weight": 1e-3,

        # "video_input_dir": os.path.join(cp["proj_dir"], 'intput_vid'),        
        "video_input_dir": os.path.join(cp["data_dir"], ''),
        "video_output_dir": os.path.join(cp["results_dir"], 'output_vid'),
        "video_output_dir": os.path.join(cp["data_dir"], 'output_vid'),
        "video_dir": os.path.join(cp["lab"], ""),
        "video_frames_dir": os.path.join(cp["lab"], "NewYork"),
        "video_frames_dir": os.path.join(cp["results_dir"], 'frames'),
        "video_output_dir": os.path.join(cp["data_dir"], 'output_vid'),
        "video_file": 'videoframe.mp4',
        "video_file": 'Streets of New York City 4K video-vCdBIRtsL6o.f313.mp4',
        "video_file": 'portu.mp4',
        # "video_input_dir": os.path.join(cp["data_dir"], 'intput_vid'),
        "video_output_dir": os.path.join(cp["results_dir"], 'output_vid'),
        "video_dir": os.path.join(cp["lab"], ""),
        # "video_path": os.path.join(video_dir, video_file),
        "backward_optical_flow_frmt": 'backward_{}_{}.flo',
        "forward_optical_flow_frmt": 'forward_{}_{}.flo',
        "content_weights_frmt": 'reliable_{}_{}.txt',
        "video": 0,

        "frames_dir": os.path.join(cp["results_dir"], 'cromes/frames'),
        "frames_dir": os.path.join(cp["lab"], "NewYork"),
        "frame_iterations": 800,
        "start_frame": 1,
        "end_frame": 9,
        "frame": None,    

        "temporal_weight": 2e2,

        "img_output_dir": cp["results_dir"],
        "image_step_format": 'step{}.jpg',
        "image_epoch_format": 'epoch{}.jpg',
        "show_entry_imgs": 0,
        "show_set_imgs": 1,
        "show_fit_imgs": 1,
        "zfill": 4,
        "device": '/gpu:0',

    }
    xp={}
    for key in cp.keys():
        xp[key] = cp[key]

    tree = ontree.tree(cp)
    for key in tree.keys():
        xp[key] = tree[key]

    for key in yp.keys():

        print(key, yp[key])

        xp[key] = yp[key]
   
    return xp


#   ******************
#   FUNS VIDEO
#
#   ***************

def frame_cv2_rgb_vgg(frame, frames_dir, args):
    content_frame_frmt = args.content_frame_frmt
    max_size = args.max_size
    zfill = args.zfill
    frame_name = args.content_frame_frmt.format(str(frame).zfill(zfill))
    frame_img_path = os.path.join(frames_dir, frame_name)
    print("frame_img_path", frame_img_path)
    img = cv2.imread(frame_img_path, cv2.IMREAD_COLOR)
    img = onvgg.vgg_preprocess(img)
    # img = onvgg.vgg_deprocess(img)
    # cv2.imshow('img', img)
    # cv2.waitKey(2000)    
    return img
    
def get_Lucas_Kanade_Optical_Flow(path):

    cap = cv2.VideoCapture(path)

    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                        qualityLevel = 0.3,
                        minDistance = 7,
                        blockSize = 7 )

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0,255,(100,3))

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    while(1):
        ret,frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]

        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
        img = cv2.add(frame,mask)

        cv2.imshow('frame',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)

    cv2.destroyAllWindows()
    cap.release()



def get_dense_Optical_Flow(path):

    cap = cv2.VideoCapture(path)

    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255

    while(1):
        ret, frame2 = cap.read()
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

        cv2.imshow('frame2',rgb)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('opticalfb.png',frame2)
            cv2.imwrite('opticalhsv.png',rgb)
        prvs = next

    cap.release()
    cv2.destroyAllWindows()


def get_content_frame(frame, args):
        
    video_input_dir = args.video_input_dir
    content_frame_frmt = args.content_frame_frmt

    # content_frame_frmt = 'frame_{}.ppm' # args.content_frame_frmt
    zfill = 4 # args.zfill
    frame_name = content_frame_frmt.format(str(frame).zfill(zfill))
    # path_to_img = os.path.join(video_input_dir, frame_name)

    # print("get_content_frame path", path_to_img)    
    # img = read_image(path_to_img)

    path = os.path.join(args.video_input_dir, frame_name)
    img = onfile.path_to_tnua_with_tf(path, args)
    return img


#   *****************
#   get_content_img:  load an image and limit its maximum dimension in pixels
def get_content_image(content_img, content_imgs_dir='./', max_size=512):

    path = os.path.join(content_imgs_dir, content_img)
    # bgr image
    img = cv2.imread(path, cv2.IMREAD_COLOR)

    h, w, d = img.shape
    mx = max_size
    # resize if > max size
    if h > w and h > mx:
        w = (float(mx) / float(h)) * w
        img = cv2.resize(img, dsize=(int(w), mx), interpolation=cv2.INTER_AREA)
    if w > mx:
        h = (float(mx) / float(w)) * h
        img = cv2.resize(img, dsize=(mx, int(h)), interpolation=cv2.INTER_AREA)

    return img


def _get_style_images(content_img, args=None):
  _, ch, cw, cd = content_img.shape
  style_imgs = []
  for style_fn in args.style_imgs_files:
    path = os.path.join(args.style_imgs_dir, style_fn)
    # bgr image
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    check_image(img, path)
    img = img.astype(np.float32)
    img = cv2.resize(img, dsize=(cw, ch), interpolation=cv2.INTER_AREA)
    img = onvgg.vgg_preprocess(img)
    style_imgs.append(img)
  return style_imgs


def cv2_frame(cvi, refimg, scale):
    _, ch, cw, cd = refimg.shape #  (1, 256, 256, 3) [[[[-107.68   -90.779  -78.939]

    img = cvi
    img = img.astype(np.float32) # shape: (657, 1000, 3) [[[ 51.  68. 135.]

    sh, sw, sd = img.shape # shape: (657, 1000, 3)

    # use scale args to resize and tile image
    scaled_img = cv2.resize(img, dsize=(int(sw*scale), int(sh*scale)), 
        interpolation=cv2.INTER_AREA)
    ssh, ssw, ssd = scaled_img.shape # shape: (657, 1000, 3)

    if ssh > ch and ssw > cw:
        starty = int((ssh-ch)/2)
        startx = int((ssw-cw)/2)
        img = scaled_img[starty:starty+ch, startx:startx+cw]
    elif ssh > ch:
        starty = int((ssh-ch)/2)
        img = scaled_img[starty:starty+ch, 0:ssw]
    # if ssw != cw:   # scaled_style_width != content_width 1000 != 256 
    if ssw <= cw:   # scaled_style_width != content_width 1000 != 256 
        # cv2.copyMakeBorder(src, top, bottom, left, right, borderType, value)
        img = cv2.copyMakeBorder(img,0,0,0,(cw-ssw),cv2.BORDER_REFLECT)
    elif ssw > cw:
        startx = int((ssw-cw)/2)
        img = scaled_img[0:ssh, startx:startx+cw]
    # if ssh != ch:
    if ssh <= ch:
        img = cv2.copyMakeBorder(img,0,(ch-ssh),0,0,cv2.BORDER_REFLECT)
    # else:
    if ssh <= ch and ssw <= cw:
        img = cv2.copyMakeBorder(scaled_img,0,(ch-ssh),0,(cw-ssw),cv2.BORDER_REFLECT)
    return img


def get_style_cvis(content_img, args):
    style_imgs=[]
    style_imgs_dir=args.style_imgs_dir
    max_size=args.max_size
    style_scale =args.style_scale

    _, ch, cw, cd = content_img.shape #  (1, 256, 256, 3) [[[[-107.68   -90.779  -78.939]
    mx = max_size

    cvis = []
    for style_fn in args.style_imgs_files:
        path = os.path.join(style_imgs_dir, style_fn)
        # bgr image
        cvi = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2_frame(cvi, content_img, style_scale)

        # img = onvgg.vgg_preprocess(img)
        cvis.append(img)
    return cvis


def _get_init_image(init_img, init_img_dir='./', max_size=512):
    path = os.path.join(init_img_dir, init_img)
    # bgr image
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    check_file(img, path)

    h, w, d = img.shape
    mx = max_size
    # resize if > max size
    if h > w and h > mx:
        w = (float(mx) / float(h)) * w
        img = cv2.resize(img, dsize=(int(w), mx), interpolation=cv2.INTER_AREA)
    if w > mx:
        h = (float(mx) / float(w)) * h
        img = cv2.resize(img, dsize=(mx, int(h)), interpolation=cv2.INTER_AREA)

    return img

def get_init_image(init_type, content_img, style_imgs, frame=None, args=None):
    if init_type == 'content':
        return content_img
    elif init_type == 'style':
        return style_imgs[0]
    elif init_type == 'random':
        init_img = get_noise_image(args.noise_ratio, content_img)
        return init_img
    # only for video frames
    elif init_type == 'prev':
        init_img = get_prev_frame(frame, args)
        return init_img
    elif init_type == 'prev_warped':
        init_img = get_prev_warped_frame(frame, args)
        return init_img


def get_input_image(
        init_type= 'content', # {content,style,init,random,prev,prev_warped}
        content_img=None, 
        style_imgs=[], 
        init_img=None, 
        frame=None, 
        video_input_dir='./', 
        video_output_dir='./', 
        content_frame_frmt = 'frame_{}.ppm',
        zill=4,
        noise_ratio = 1.0,
    ):
    print(f"|---> get_input_image of type: {init_type} with frame: {frame}")

    if init_type == 'content':
        return content_img
    elif init_type == 'style':
        style_img = style_imgs[0]   # _e_
        return style_img
    elif init_type == 'init':
        return init_img
    elif init_type == 'random':
        init_img = get_noise_image(noise_ratio, content_img)
        return init_img
    # only for video frames
    elif init_type == 'prev':
        assert(not frame == None)
        init_img = get_prev_frame(
            frame, 
            video_output_dir,
            content_frame_frmt,
            zill,
        )
        return init_img
    elif init_type == 'prev_warped':
        init_img = get_prev_warped_frame(frame, video_input_dir)
        return init_img

def get_optimizer(loss):
    max_iterations = 1000 # args.max_iterations
    verbose = True # args.verbose
    learning_rate = 1e0 # args.learning_rate
    optimizer = 'adam' # args.optimizer

    print_iterations = print_iterations if verbose else 0
    if optimizer == 'lbfgs':
        optimizer = tf.contrib.opt.ScipyOptimizerInterface(
        loss, method='L-BFGS-B',
        options={'maxiter': max_iterations,
                    'disp': print_iterations})
    elif optimizer == 'adam':
        # optimizer = tf.train.AdamOptimizer(learning_rate)
        optimizer = tf.optimizers.Adam(learning_rate)
    return optimizer

def get_noise_image(noise_ratio, content_img, 
        seed = 0
    ):
    np.random.seed(seed)
    noise_img = np.random.uniform(-20., 20., content_img.shape).astype(np.float32)
    img = noise_ratio * noise_img + (1.-noise_ratio) * content_img
    return img

def get_mask_image(mask_img, width, height, 
        content_imgs_dir = './'
    ):
    path = os.path.join(content_imgs_dir, mask_img)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    check_file(img, path)
    img = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32)
    mx = np.amax(img)
    img /= mx
    return img


#   ****************
#   get_prev_frame: previously stylized frame
def get_prev_frame(frame, args=None):
    video_output_dir = args.video_input_dir # _e_ tbc
    content_frame_frmt = args.content_frame_frmt
    zfill = args.zfill
    
    prev_frame = frame - 1
    fn = content_frame_frmt.format(str(prev_frame).zfill(zfill))
    path = os.path.join(video_output_dir, fn)
    print(f"|---> get_prev_frame for frame: {frame} from {path}")

    # img = cv2.imread(path, cv2.IMREAD_COLOR)
    # check_file(img, path)

    img = onfile.path_cv_pil(path)
    img = onformat.pil_to_dnua(img)

    return img

def get_prev_warped_frame(frame, args=None):

    video_input_dir = args.video_input_dir
    backward_optical_flow_frmt = args.backward_optical_flow_frmt
    content_frame_frmt = args.content_frame_frmt

    prev_img = get_prev_frame(frame, args)
    print(f"|---> get_prev_frame in {video_input_dir} with {content_frame_frmt}")
    prev_frame = frame - 1
    # backwards flow: current frame -> previous frame
    fn = backward_optical_flow_frmt.format(str(frame), str(prev_frame))
    path = os.path.join(video_input_dir, fn)
    flow = read_flow_file(path)
    warped_img = warp_image(prev_img, flow).astype(np.float32)
    img = onvgg.vgg_preprocess(warped_img)
    return img

def get_content_weights(frame, prev_frame, args):
    video_input_dir=args.video_input_dir
    content_weights_frmt = args.content_weights_frmt

    forward_fn = content_weights_frmt.format(str(prev_frame), str(frame))
    backward_fn = content_weights_frmt.format(str(frame), str(prev_frame))
    forward_path = os.path.join(video_input_dir, forward_fn)
    backward_path = os.path.join(video_input_dir, backward_fn)
    forward_weights = read_weights_file(forward_path)
    backward_weights = read_weights_file(backward_path)
    return forward_weights #, backward_weights

def warp_image(src, flow):
    _, h, w = flow.shape
    flow_map = np.zeros(flow.shape, dtype=np.float32)
    for y in range(h):
        flow_map[1,y,:] = float(y) + flow[1,y,:]
    for x in range(w):
        flow_map[0,:,x] = float(x) + flow[0,:,x]
    # remap pixels to optical flow
    dst = cv2.remap(
        src, flow_map[0], flow_map[1], 
        interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
    return dst

def convert_to_original_colors(content_img, stylized_img, args=None):
    color_convert_type = args.color_convert_type

    content_img  = onvgg.vgg_deprocess(content_img)
    stylized_img = onvgg.vgg_deprocess(stylized_img)
    if color_convert_type == 'yuv':
        cvt_type = cv2.COLOR_BGR2YUV
        inv_cvt_type = cv2.COLOR_YUV2BGR
    elif color_convert_type == 'ycrcb':
        cvt_type = cv2.COLOR_BGR2YCR_CB
        inv_cvt_type = cv2.COLOR_YCR_CB2BGR
    elif color_convert_type == 'luv':
        cvt_type = cv2.COLOR_BGR2LUV
        inv_cvt_type = cv2.COLOR_LUV2BGR
    elif color_convert_type == 'lab':
        cvt_type = cv2.COLOR_BGR2LAB
        inv_cvt_type = cv2.COLOR_LAB2BGR
    content_cvt = cv2.cvtColor(content_img, cvt_type)
    stylized_cvt = cv2.cvtColor(stylized_img, cvt_type)
    c1, _, _ = cv2.split(stylized_cvt)
    _, c2, c3 = cv2.split(content_cvt)
    merged = cv2.merge((c1, c2, c3))
    dst = cv2.cvtColor(merged, inv_cvt_type).astype(np.float32)
    dst = onvgg.vgg_preprocess(dst)
    return dst


#   ******************
#   FUNCS LOG
#

def write_video_output(
        frame, 
        output_img, 
        video_output_dir= './',
        content_frame_frmt='frame{}.jpg',
        zfill = 4,
    ):
    print(f"|---> write_video_output frame {frame}")
    fn = content_frame_frmt.format(str(frame).zfill(zfill))
    path = os.path.join(video_output_dir, fn)
    save_cv2(path, output_img)


def write_image_output(
        output_img, 
        content_img, 
        style_imgs, 
        init_img, 
        img_output_dir='./',
        max_epochs = 1000,
        args=None,
    ):
    img_name = 'result'
    content_img_file = 'content.jpg'
    init_img_name = 'init.jpg'
    style_imgs_weights = [1.0]
    style_mask_imgs = None
    init_img_type = 'content'
    content_weight = 5e0
    style_weight = 1e4
    total_variation_weight = 1e-3
    content_layers = ['conv5_2']
    style_layers = ['conv1_1','conv2_1','conv3_1','conv4_1','conv5_1']
    optimizer = 'adam'
    max_size = 512

    out_dir = os.path.join(img_output_dir, str(max_epochs))
    os.makedirs(out_dir, exist_ok=True)

    img_path = os.path.join(out_dir, img_output_dir+'-'+str(max_epochs)+'.png')
    content_path = os.path.join(out_dir, content_img_file)
    init_path = os.path.join(out_dir, init_img_name)

    onfile.pil_to_file_with_cv(img_path, output_img)
    onfile.pil_to_file_with_cv(content_path, content_img)
    onfile.pil_to_file_with_cv(init_path, init_img)
    index = 0
    for style_img in style_imgs:
        path = os.path.join(out_dir, 'style_'+str(index)+'.png')
        onfile.pil_to_file_with_cv(path, style_img)
        index += 1
  
    # save the configuration settings
    out_file = os.path.join(out_dir, 'meta_data.txt')
    f = open(out_file, 'w')
    f.write('image_name: {}\n'.format(img_name))
    f.write('content: {}\n'.format(content_img))
    index = 0
    for style_img, weight in zip(style_imgs, style_imgs_weights):
        f.write('styles['+str(index)+']: {} * {}\n'.format(weight, style_img))
        index += 1
    index = 0
    if style_mask_imgs is not None:
        for mask in style_mask_imgs:
            f.write('style_masks['+str(index)+']: {}\n'.format(mask))
            index += 1
    f.write('init_type: {}\n'.format(init_img_type))
    f.write('content_weight: {}\n'.format(content_weight))
    f.write('style_weight: {}\n'.format(style_weight))
    f.write('total_variation_weight: {}\n'.format(total_variation_weight))
    f.write('content_layers: {}\n'.format(content_layers))
    f.write('style_layers: {}\n'.format(style_layers))
    f.write('optimizer_type: {}\n'.format(optimizer))
    f.write('max_epochs: {}\n'.format(max_epochs))
    f.write('max_image_size: {}\n'.format(max_size))
    f.close()


#   *****************
#   FUNCS MODEL
#
def read_weights_file(path):
    lines = open(path).readlines()
    header = list(map(int, lines[0].split(' ')))
    w = header[0]
    h = header[1]
    vals = np.zeros((h, w), dtype=np.float32)
    for i in range(1, len(lines)):
        line = lines[i].rstrip().split(' ')
        vals[i-1] = np.array(list(map(np.float32, line)))
        vals[i-1] = list(map(lambda x: 0. if x < 255. else 1., vals[i-1]))
    # expand to 3 channels
    weights = np.dstack([vals.astype(np.float32)] * 3)
    return weights


def get_content_weights(frame, prev_frame):

    content_weights_frmt = 'reliable_{}_{}.txt' # args.content_weights_frmt
    video_input_dir = './video_input' # args.video_input_dir

    forward_fn = content_weights_frmt.format(str(prev_frame), str(frame))
    backward_fn = content_weights_frmt.format(str(frame), str(prev_frame))
    forward_path = os.path.join(video_input_dir, forward_fn)
    backward_path = os.path.join(video_input_dir, backward_fn)
    forward_weights = read_weights_file(forward_path)
    backward_weights = read_weights_file(backward_path)
    return forward_weights #, backward_weights

def minimize_with_lbfgs(sess, net, optimizer, init_img,
        verbose=True,
    ):
    if verbose: print('\nMINIMIZING LOSS USING: L-BFGS OPTIMIZER')
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    sess.run(net['input'].assign(init_img))
    optimizer.minimize(sess)

def minimize_with_adam(sess, net, optimizer, init_img, loss,
        verbose=True,
        max_iterations=1000,
        print_iterations=50,
    ):
    if verbose: print('\nMINIMIZING LOSS USING: ADAM OPTIMIZER')
    train_op = optimizer.minimize(loss)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    sess.run(net['input'].assign(init_img))
    iterations = 0
    while (iterations < max_iterations):
        sess.run(train_op)
        if iterations % print_iterations == 0 and verbose:
            curr_loss = loss.eval()
            print("At iterate {}\tf=  {}".format(iterations, curr_loss))
        iterations += 1


#   ******************
#   NETS
#

'''
  'a neural algorithm for artistic style' loss functions
'''
def content_layer_loss(p, x, args=None):
  _, h, w, d = p.get_shape()
  M = h * w # h.value * w.value
  N = d # d.value
  if args.content_loss_function   == 1:
    K = 1. / (2. * N**0.5 * M**0.5)
  elif args.content_loss_function == 2:
    K = 1. / (N * M)
  elif args.content_loss_function == 3:  
    K = 1. / 2.
  loss = K * tf.reduce_sum(input_tensor=tf.pow((x - p), 2))
  return loss

def style_layer_loss(a, x):
  _, h, w, d = a.get_shape()
  M = h * w # h.value * w.value
  N = d # d.value
  A = gram_matrix(a, M, N)
  G = gram_matrix(x, M, N)
  loss = (1./(4 * N**2 * M**2)) * tf.reduce_sum(input_tensor=tf.pow((G - A), 2))
  return loss

def gram_matrix(x, area, depath):
  F = tf.reshape(x, (area, depath))
  G = tf.matmul(tf.transpose(a=F), F)
  return G

def mask_style_layer(a, x, mask_img):
  _, h, w, d = a.get_shape()
  # mask = get_mask_image(mask_img, w.value, h.value)
  mask = get_mask_image(mask_img, w, h)
  mask = tf.convert_to_tensor(value=mask)
  tensors = []
  # for _ in range(d.value): 
  for _ in range(d): 
    tensors.append(mask)
  mask = tf.stack(tensors, axis=2)
  mask = tf.stack(mask, axis=0)
  mask = tf.expand_dims(mask, 0)
  a = tf.multiply(a, mask)
  x = tf.multiply(x, mask)
  return a, x

def sum_masked_style_losses(net, style_imgs, args=None):
  total_style_loss = 0.
  weights = args.style_imgs_weights
  masks = args.style_mask_imgs
  for img, img_weight, img_mask in zip(style_imgs, weights, masks):
    net['input'].assign(img)
    style_loss = 0.
    for layer, weight in zip(args.style_layers, args.style_layers_weights):
      a = net[layer]
      print("sum_masked_style_losses a", a)
      x = net[layer]
      a = tf.convert_to_tensor(value=a)
      a, x = mask_style_layer(a, x, img_mask)
      style_loss += style_layer_loss(a, x) * weight
    style_loss /= float(len(args.style_layers))
    total_style_loss += (style_loss * img_weight)
  total_style_loss /= float(len(style_imgs))
  return total_style_loss

def sum_style_losses(net, combo, style_imgs, args=None):
    total_style_loss = 0.
    weights = args.style_imgs_weights


    for img, img_weight in zip(style_imgs, weights):
            # net['input'].assign(img)
            ps = net.extract_features([combo], args.style_layers)[0]   # net[layer]
            xs = net.extract_features([img], args.style_layers)[0]   # net[layer]            
            style_loss = 0.
            for layer, weight in zip(args.style_layers, args.style_layers_weights):
                p = ps[layer] # net[layer]
                x = xs[layer] # net[layer]
                # p = tf.convert_to_tensor(value=p)
                style_loss += style_layer_loss(p, x) * weight
            style_loss /= float(len(args.style_layers))
            total_style_loss += (style_loss * img_weight)
    total_style_loss /= float(len(style_imgs))
    return total_style_loss

def sum_content_losses(net, combo, content_img, args=None):
    # net['input'].assign(content_img)
    content_loss = 0.
    ps = net.extract_features([combo], args.content_layers)[0]   # net[layer]
    xs = net.extract_features([content_img], args.content_layers)[0]   # net[layer]

    for layer, weight in zip(args.content_layers, args.content_layer_weights):
            p = ps[layer]   # net[layer]
            x = xs[layer]   # net[layer]
            # p = tf.convert_to_tensor(value=p)
            content_loss += content_layer_loss(p, x, args) * weight
    content_loss /= float(len(args.content_layers))
    return content_loss

'''
  'artistic style transfer for videos' loss functions
'''
def temporal_loss(x, w, c):
  c = c[np.newaxis,:,:,:]
  D = float(x.size)
  loss = (1. / D) * tf.reduce_sum(input_tensor=c * tf.nn.l2_loss(x - w))
  loss = tf.cast(loss, tf.float32)
  return loss

def get_longterm_weights(i, j, args=None):
  c_sum = 0.
  for k in range(args.prev_frame_indices):
    if i - k > i - j:
      c_sum += get_content_weights(i, i - k, args)
  c = get_content_weights(i, i - j, args)
  c_max = tf.maximum(c - c_sum, 0.)
  return c_max

def sum_longterm_temporal_losses(sess, net, frame, input_img, args=None):
  x = sess.run(net['input'].assign(input_img))
  loss = 0.
  for j in range(args.prev_frame_indices):
    prev_frame = frame - j

    print("sum_longterm_temporal_losses", args.init_frame_type)
    w = input_img # _e_
    if args.init_frame_type == 'prev':
        w = get_prev_frame(frame, args) # _e_
    elif args.init_frame_type == 'prev_warped':
        w = get_prev_warped_frame(frame, args)

    # w = get_prev_warped_frame(frame, args)
    c = get_longterm_weights(frame, prev_frame)
    loss += temporal_loss(x, w, c)
  return loss

def sum_shortterm_temporal_losses(sess, net, frame, input_img, args=None):
  x = sess.run(net['input'].assign(input_img))
  prev_frame = frame - 1

  print("sum_longterm_temporal_losses init_frame_type", args.init_frame_type)
  w = input_img # _e_
  if args.init_frame_type == 'prev':
      w = get_prev_frame(frame, args) # _e_
  elif args.init_frame_type == 'prev_warped':
      w = get_prev_warped_frame(frame, args)

  #   w = get_prev_warped_frame(frame, args)

  c = get_content_weights(frame, prev_frame, args)
  loss = temporal_loss(x, w, c)
  return loss

'''
  utilities and i/o
'''
def read_image(path):
  # bgr image
  img = cv2.imread(path, cv2.IMREAD_COLOR)
  check_image(img, path)
  img = img.astype(np.float32)
  img = onvgg.vgg_preprocess(img)
  return img

def write_image(path, img):
  img = onvgg.vgg_deprocess(img)
  cv2.imwrite(path, img)


def preprocess_tensor(inputs):

    # Keras works with batches of images. 
    # So, the first dimension is used for the number of samples (or images) you have.
    # vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((3,1,1))
    VGG_BGR_MEAN = [103.939, 116.779, 123.68]
    vgg_bgr_mean_arr = np.array(VGG_BGR_MEAN)

    imgpre = inputs[0] # np.squeeze(inputs, axis = 0) # 

    imgpre = imgpre[...,::-1] # BGR => RGB
    imgpre = imgpre[np.newaxis,:,:,:] # shape (h, w, d) to (1, h, w, d)
    imgpre -= vgg_bgr_mean_arr.reshape((1,1,1,3))

    imgpre = tf.convert_to_tensor(imgpre)
    return imgpre

def preprocess_tensors(inputs):

    pretensors = []
    for item in inputs:
        preitem = preprocess_tensor(item)
        pretensors.append(preitem)

    return pretensors


def show_effect_a(path):
    from PIL import Image
    image = Image.open(path)
    image.show()

    import cv2
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if 0:
        cv2.imshow('img', img)
        cv2.waitKey(0)      

    img = onvgg.vgg_preprocess(img) # <class 'numpy.ndarray'> (1, 1654, 1654, 3)
    print(type(img))
    print(np.shape(img))

    rgb = tf.squeeze(img, axis=0)
    rgb = np.array(rgb, dtype=np.uint8)
    rgb = PIL.Image.fromarray(rgb)
    rgb.show()

    img = onvgg.vgg_deprocess(img)
    image = Image.fromarray(img)
    image.show() 



def read_flow_file(path):
  with open(path, 'rb') as f:
    # 4 bytes header
    header = struct.unpack('4s', f.read(4))[0]
    # 4 bytes width, height    
    w = struct.unpack('i', f.read(4))[0]
    h = struct.unpack('i', f.read(4))[0]   
    flow = np.ndarray((2, h, w), dtype=np.float32)
    for y in range(h):
      for x in range(w):
        flow[0,y,x] = struct.unpack('f', f.read(4))[0]
        flow[1,y,x] = struct.unpack('f', f.read(4))[0]
  return flow

def read_weights_file(path):
  lines = open(path).readlines()
  header = list(map(int, lines[0].split(' ')))
  w = header[0]
  h = header[1]
  vals = np.zeros((h, w), dtype=np.float32)
  for i in range(1, len(lines)):
    line = lines[i].rstrip().split(' ')
    vals[i-1] = np.array(list(map(np.float32, line)))
    vals[i-1] = list(map(lambda x: 0. if x < 255. else 1., vals[i-1]))
  # expand to 3 channels
  weights = np.dstack([vals.astype(np.float32)] * 3)
  return weights

def normalize(weights):
  denom = sum(weights)
  if denom > 0.:
    return [float(i) / denom for i in weights]
  else: return [0.] * len(weights)

def check_image(img, path):
  if img is None:
    raise OSError(errno.ENOENT, "No such file", path)

'''
  rendering -- where the magic happens
'''
def compute_loss(combo):
    loss = tf.zeros(shape=())

def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def build_vgg(
        input_shape = (224, 224, 3),
        model_classes = 1000,
        model_pooling=None,
        model_include_top = True,
        model_weights = None,
    ):

    inputs = tf.keras.layers.Input(shape = input_shape)

    # Block 1
    x = tf.keras.layers.Conv2D(64, (3,3),activation='relu',padding='same',name='conv1_1')(inputs)
    x = tf.keras.layers.Conv2D(64, (3,3),activation='relu',padding='same',name='conv1_2')(x)
    x = tf.keras.layers.MaxPooling2D((2,2), strides=(2,2), name='pool1')(x)

    # Block 2
    x = tf.keras.layers.Conv2D(128, (3,3),activation='relu',padding='same',name='conv2_1')(x)
    x = tf.keras.layers.Conv2D(128, (3,3),activation='relu',padding='same',name='conv2_2')(x)
    x = tf.keras.layers.MaxPooling2D((2,2), strides=(2,2), name='pool2')(x)

    # Block 3
    x = tf.keras.layers.Conv2D(256, (3,3),activation='relu',padding='same',name='conv3_1')(x)
    x = tf.keras.layers.Conv2D(256, (3,3),activation='relu',padding='same',name='conv3_2')(x)
    x = tf.keras.layers.Conv2D(256, (3,3),activation='relu',padding='same',name='conv3_3')(x)
    x = tf.keras.layers.Conv2D(256, (3,3),activation='relu',padding='same',name='conv3_4')(x)
    x = tf.keras.layers.MaxPooling2D((2,2), strides=(2,2), name='pool3')(x)

    # Block 4
    x = tf.keras.layers.Conv2D(512, (3,3),activation='relu',padding='same',name='conv4_1')(x)
    x = tf.keras.layers.Conv2D(512, (3,3),activation='relu',padding='same',name='conv4_2')(x)
    x = tf.keras.layers.Conv2D(512, (3,3),activation='relu',padding='same',name='conv4_3')(x)
    x = tf.keras.layers.Conv2D(512, (3,3),activation='relu',padding='same',name='conv4_4')(x)
    x = tf.keras.layers.MaxPooling2D((2,2), strides=(2,2), name='pool4')(x)

    # Block 5
    x = tf.keras.layers.Conv2D(512, (3,3),activation='relu',padding='same',name='conv5_1')(x)
    x = tf.keras.layers.Conv2D(512, (3,3),activation='relu',padding='same',name='conv5_2')(x)
    x = tf.keras.layers.Conv2D(512, (3,3),activation='relu',padding='same',name='cpmv5_3')(x)
    x = tf.keras.layers.Conv2D(512, (3,3),activation='relu',padding='same',name='conv5_4')(x)
    x = tf.keras.layers.MaxPooling2D((2,2), strides=(2,2), name='pool5')(x)

    if model_include_top:
        print("include classification block: flatten, fc1, fc2, predictions")
        # Classification block
        x = tf.keras.layers.Flatten(name='flatten')(x)
        x = tf.keras.layers.Dense(4096, activation='relu', name='fc1')(x)
        x = tf.keras.layers.Dense(4096, activation='relu', name='fc2')(x)
        x = tf.keras.layers.Dense(model_classes, activation='softmax', name='predictions')(x)
    else:
        if model_pooling == 'avg':
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
        elif model_pooling == 'max':
            x = tf.keras.layers.GlobalMaxPooling2D()(x)

    outputs = x

    vgg = tf.keras.Model(inputs, outputs)

    if model_weights:
        print("load model_weights")
        vgg.load_weights(model_weights)
    else:
        print("did not load model_weights")

    return vgg


class Styler(tf.keras.models.Model):

    def __init__(self, 
        input_shape = (224, 224, 3),        
        args=None, 
    ):

        super(Styler, self).__init__()

        content_layers = args.content_layers
        style_layers = args.style_layers

        model_weights = args.model_weights
        model_include_top = args.model_include_top
        model_classes = args.model_classes
        model_pooling = args.model_pooling
        
        self.num_content_layers = len(content_layers)
        self.num_style_layers = len(style_layers)

        # shap = np.shape(tf.squeeze(content_img, axis=0))
        self.imgshape = input_shape
        vggmodel = build_vgg(
            input_shape = self.imgshape,
            model_classes = model_classes,
            model_pooling = model_pooling,
            model_include_top = model_include_top,
            model_weights = model_weights,
        )

        layer_names = content_layers + style_layers
        print("Styler content_layers", content_layers)
        print("Styler style_layers", style_layers)
        print("Styler layers", layer_names)
        outputs = {name: vggmodel.get_layer(name).output for name in layer_names}        
        # outputs = [vggmodel.get_layer(name).output for name in layer_names]
        self.net = tf.keras.Model([vggmodel.input], outputs) 
        self.net.trainable = False        
        if 0:
            self.net.summary()

        self.optimizer = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

        # if not hasattr(self, "image"):  # Or set self.v to None in __init__ 
        #     image = tf.Variable(self.content_img) 
        # self.image = image


    def extract_features(self, imgs, layers=None):
        net = self.net
        imgshape = self.imgshape

        if len(imgshape) == 3:
            imgshape = imgshape[:-1]
        (width,height) = imgshape

        rei = []
        for i,sty_img in enumerate(imgs):
            img = imgs[i] # _e_

            preprocessed_input = onvgg.tnua_to_vgg(img, width=width, height=height)
            print(f'|===> extract_features \n \
                preprocessed_input {type(preprocessed_input)} {np.shape(preprocessed_input)} \n \
            ')
            outputs = net(preprocessed_input)

            features = {}
            if layers:
                for layer in layers:
                    features[layer]=outputs[layer]
            else:
                features = outputs

            rei.append(features)

        return rei            

    def call(self, image = None):
        print(f'|===> Styler call {type(image)}')
        img = self.image # <class 'tensorflow.python.ops.resource_variable_ops.ResourceVariable'>
        img = onformat.nua_to_pil(img)
        img.show()
 

    # @tf.function
    def train_step(self, image, content_img, style_imgs, args = None):
        net = self.net

        total_variation_weight = args.total_variation_weight
        style_weight = args.style_weight
        content_weight = args.content_weight
        video = args.video
        frame = args.frame
        temporal_weight = args.temporal_weight
        content_layer_weight = args.content_layer_weights
        style_imgs_weights = args.style_imgs_weights
        style_layers_weight = args.style_layers_weights

        with tf.GradientTape() as tape:
           
        #     # content loss
            L_content = sum_content_losses(self, image, content_img, args)

            # style loss
            if args.style_mask:
                L_style = sum_masked_style_losses(self, image, style_imgs, args)
            else:
                L_style = sum_style_losses(self, image, style_imgs, args)

            L_tv = tf.image.total_variation(image)

            # loss weights
            alpha = args.content_weight
            beta  = args.style_weight
            theta = args.total_variation_weight
            
            # total loss
            L_total  = 0.
            L_total += alpha * L_content
            L_total += beta  * L_style
            L_total += theta * L_tv

        grad = tape.gradient(L_total, image)
        self.optimizer.apply_gradients([(grad, image)])
        image.assign(clip_0_1(image))
        self.image = image


    def fit(self, input_img, content_img, style_imgs, args  = None): # frame = None,

        max_epochs = args.max_epochs
        steps_per_epoch = args.steps_per_epoch
        video = args.video
        verbose = args.verbose
        video_output_dir = args.video_output_dir
        img_output_dir = args.img_output_dir
        image_step_format = args.image_step_format
        image_epoch_format = args.image_epoch_format
        content_frame_frmt = args.content_frame_frmt
        zfill = args.zfill
        show_fit_imgs = args.show_fit_imgs

        print(f'|===> fit \n \
            max_epochs: {max_epochs} \n \
            steps_per_epoch: {steps_per_epoch} \n \
            input_img: {np.shape(input_img)} \n \
        ')
        train_step = self.train_step

        # # ============
   
        image = tf.Variable(input_img)   
        self.image = image

        # # ============

        start = time.time()

        step = 0
        if 0: print(f"[   .   ] display input image {step} image")
        img = onformat.nua_to_pil(image)
        img = ondata.pil_resize(img, (256,256))
        img.show()

        for epoch in range(max_epochs):
            for step_in_epoch in range(steps_per_epoch):
                step += 1

                # ==========================================
                train_step(image, content_img, style_imgs, args)
                # =========================================

                print(".", end='')

            if 1 and step % 100 == 0:
                if 0:
                    try:
                        # display image per epoch _e_
                        display.clear_output(wait=True)
                        print(f"[   .   ] display in step {step} image")
                        display.display(onformat.nua_to_pil(image))
                    except:
                        print("could not display epoch images")
                else:
                    print(f"[   .   ] show epoch {epoch} image")
                    img = onformat.nua_to_pil(image)
                    img.show()

            # write image per epoch
            img_name = image_epoch_format.format(str(epoch).zfill(zfill))        
            img_path = os.path.join(img_output_dir, img_name)
            onfile.pil_to_file_with_cv(img_path, img)

            write_image_output(
                output_img = onformat.nua_to_pil(image), 
                content_img = onformat.nua_to_pil(content_img), 
                style_imgs = onformat.dnuas_to_pils(style_imgs), 
                init_img = onformat.nua_to_pil(input_img),
                img_output_dir = img_output_dir, # results_dir _e_ *****
                max_epochs = max_epochs, # results_dir _e_ *****
                args = args,
            )


        end = time.time()
        print("Total time: {:.1f}".format(end-start))

#   ******************
#   CMDS
#
#   ****************
   

#   *******************
#   nnimg
#
def nnimg(args, kwargs):

    if 0:
        args.PROJECT = 'bomze' # https://github.com/tg-bomze/Style-Transfer-Collection
        args.DATASET = 'bomze'
        args.content_img_file = 'bomze-content.png'
        args.init_img_name = 'bomze-content.png'
        args.style_imgs_files = ['bomze-style.png']

    if 0:
        args.PROJECT = 'building'
        args.DATASET = 'building'
        args.content_img_file = 'IMG_4468.JPG'
        args.init_img_name = 'IMG_4468.JPG'
        args.style_imgs_files = ['EZNJVoTXsAgUh6M.jpg', 'EZsQWC1X0AQaxHs.jpg']

    if 0:
        args.PROJECT = 'labrador'
        args.DATASET = 'labrador'        
        args.content_img_file = 'YellowLabradorLooking_new.jpg'
        args.init_img_name = 'YellowLabradorLooking_new.jpg'
        args.style_imgs_files = ['starry-night.jpg']

    if 1:
        args.PROJECT = 'kandinsky'
        args.DATASET = 'Kandinsky'        
        args.content_img_file = 'YellowLabradorLooking_new.jpg'
        args.init_img_name = 'YellowLabradorLooking_new.jpg'
        args.style_imgs_files = ['Vassily_Kandinsky,_1913_-_Composition_7.jpg']

    if 0:
        args.PROJECT = 'lion'
        args.DATASET = 'lion'        
        args.content_img_file = 'lion.jpg'
        args.init_img_name = 'lion.jpg'
        args.style_imgs_files = ['a-hymn-to-the-shulamite-1982.jpg']


    args = onutil.pargs(vars(args))
    xp = getxp(vars(args))
    args = onutil.pargs(xp)    
    onutil.ddict(vars(args), 'args')

    if 1: # config

        args.max_epochs = 100
        args.steps_per_epoch = 10
        args.total_variation_weight = 0.001
        args.content_imgs_weights = [1.0]
        args.content_layers_weights = [2.5e-08]
        args.style_imgs_weights = [1.0]
        args.style_layers_weights = [0.01, 0.01, 0.01, 0.01, 0.01]

        #--
        args.content_loss_function = 1
        args.content_weight = 5e0
        args.style_layers_weights = [0.2, 0.2, 0.2, 0.2, 0.2]
        args.style_mask_imgs = []
        args.style_weight = 1e4
        args.total_variation_weight = 1e-3
        args.style_mask = None


        print(f"|===> nnimg: config \n \
        args.video: {args.video} : content images from frames \n \
        args.show_entry_imgs: {args.show_entry_imgs} \n \
        args.steps_per_epoch: {args.steps_per_epoch} \n \
        args.model_weights: {args.model_weights} \n \
        args.max_epochs: {args.max_epochs} \n \
        ")
    #--
    if 1: # tree

        args.style_layers_weights  = normalize(args.style_layers_weights)
        args.content_layer_weights = normalize(args.content_layer_weights)
        args.style_imgs_weights    = normalize(args.style_imgs_weights)

        content_base = os.path.splitext(args.content_img_file)[0]
        style_base = os.path.splitext(args.style_imgs_files[0])[0]
        args.output_folder = content_base + "_" + style_base
        args.img_output_dir = os.path.join(args.results_dir, 'cromes', args.output_folder)

        args.output_folder = content_base + "_" + style_base
        args.img_output_dir = os.path.join(args.results_dir, 'cromes', args.output_folder)

        print(f"|===> nnimg: tree \n \
        args.output_folder: {args.output_folder} \n \
        args.img_output_dir: {args.img_output_dir} \n \
        ")

        os.makedirs(args.data_dir, exist_ok=True)
        os.makedirs(args.content_imgs_dir, exist_ok=True) # data_dir

    if 1: # content image: (422, 512, 3) <class 'numpy.ndarray'>  [[[ 99 165 160]

        args.content_size = (512, 512)
        content_img_path = os.path.join(args.content_imgs_dir, args.content_img_file)
        assert os.path.exists(content_img_path), f"content image {content_img_path} does not exist"
        content_img = onfile.path_cv_pil(content_img_path)
        content_img = ondata.pil_resize(content_img, ps=args.content_size)
        content_img = onformat.pil_to_dnua(content_img)
        print(f"|===> nnimg: content \n \
            content_img_path: {content_img_path} \n \
            args.content_imgs_dir: {args.content_imgs_dir} \n \
            content_shape: {np.shape(content_img)} \n \
        ")
    
    if args.visual: 
        print(f'|---> vis content')
        onplot.pil_show_nua(content_img)
    if 0: # show content image
        content_img_path = os.path.join(args.content_imgs_dir, args.content_img_file)
        img = cv2.imread(content_img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (512, int((np.shape(img)[0]/np.shape(img)[1]) * 512)))
        img = onvgg.vgg_preprocess(img)
        img = onvgg.vgg_deprocess(img)
        cv2.imshow('img', img)
        cv2.waitKey(1000)    

    if 1: #   input image: (422, 512, 3) <class 'numpy.ndarray'> [[[ 99 165 160]
        init_path = os.path.join(args.init_img_dir, args.init_img_name)
        init_image = onfile.path_to_tnua_with_tf(init_path, args)
        print(f"|===> nnimg: init \n \
            init_path: {init_path} \n \
            args.content_imgs_dir: {args.content_imgs_dir} \n \
            init_shape: {np.shape(init_image)} \n \
        ")
        if args.visual:
            print(f'|---> vis init')
            onplot.pil_show_nua(init_image, "[   .   ] init_image")

    if 1: #   style images
        style_imgs = onfile.names_to_nuas_with_tf(args.style_imgs_files, args.style_imgs_dir, args,)
        print(f"|===> nnimg: styles \n \
            args.style_imgs_files: {args.style_imgs_files} \n \
            args.style_imgs_dir: {args.style_imgs_dir} \n \
            shapes: {[str(np.shape(style_imgs[i])) for i,img in enumerate(style_imgs)]} \n \
        ")
        if args.visual:
            print(f'|---> vis styles')
            onplot.pil_show_nuas(style_imgs, ["[   .   ] style_imgs"])

    if 1: # input image
        input_img = get_input_image(
            args.init_image_type, 
            content_img, 
            style_imgs,
            init_image,
            args.frame, 
            args.video_input_dir, 
            args.video_output_dir, 
            args.content_frame_frmt,
        )
        if args.visual:
            onplot.pil_show_nua(input_img, "[   .   ] input_img")
        if 1:   # imgs props
            print(f"|===> nnimg: input \n \
                input shape: {np.shape(input_img)} \n \
        ")        

    if 1: # model
        b,w,h,c = np.shape(input_img)
        input_shape = (w,h,c)

        print(f"|===> model \n \
            input_shape: {input_shape} \n \
        ")

        model = Styler( input_shape = input_shape, args = args, )

    if 1: # fit

        print(f"|===> fit   \n ")
        model.fit(input_img, content_img, style_imgs, args)


#   *******************
#   nnani
#
def nnani(args, kwargs):

    args = onutil.pargs(vars(args))
    args.PROJECT = 'portu'
    args.DATASET = 'stransfer'
    xp = getxp(vars(args))
    args = onutil.pargs(xp)
    onutil.ddict(vars(args), 'args')

    if 1: # config
        args.start_frame = 1
        args.end_frame = -1 # num_frames
        args.max_iterations = 10 # 1000
        args.first_frame_iterations = 20 # 2000
        args.video = True 
        args.content_frame_frmt = 'frame{}.jpg'
        args.style_imgs_files = ['starry-night.jpg']

    print(f"|===> nnani config \n \
        args.first_frame_iterations: {args.first_frame_iterations} \n \
        args.start_frame: {args.start_frame} \n \
        args.max_iterations: {args.max_iterations} \n \
        args.style_imgs_files = {args.style_imgs_files} \n \
    ")

    if 1: # tree

        extension=os.path.splitext(os.path.basename(args.video_file))[1]

        content_filename=os.path.splitext(os.path.basename(args.video_file))[0]
        content_filename=f"{content_filename}" # eg portu

        # input from org
        video_input_path = os.path.join(args.dataorg_dir, args.video_file)

        # frames, styled, output, results .. in code
        args.video_frames_dir=os.path.join(args.proj_dir, 'frames')
        args.video_styled_dir=os.path.join(args.proj_dir, 'styled')
        args.video_output_dir=os.path.join(args.proj_dir, 'output')
        img_output_dir=os.path.join(args.proj_dir, 'results')

        os.makedirs(args.video_frames_dir, exist_ok=True)
        os.makedirs(args.video_styled_dir, exist_ok=True)
        os.makedirs(args.video_output_dir, exist_ok=True)
        os.makedirs(img_output_dir, exist_ok=True)

        assert os.path.exists(video_input_path), f"input vide {video_input_path} does not exist"

    print(f"|===> nnani tree \n \
        cwd: {os.getcwd()} \n \
        args.video_file: {args.video_file} \n \
        content_filename: {content_filename} \n \
        video_input_path: {video_input_path} \n \
        args.video_frames_dir: {args.video_frames_dir} \n \
        args.video_styled_dir: {args.video_styled_dir} \n \
        args.video_output_dir: {args.video_output_dir} \n \
    ")


    if 1: # vid => frames
        print(f"|===> nnani vid to frames \n \
        video_input_path: {video_input_path} \n \
        args.video_frames_dir: {args.video_frames_dir} \n \
        ")
        onvid.vid_to_frames(video_input_path, args.video_frames_dir, target = 1)


    if 1: # video

        num_frames = len([name for name in os.listdir(args.video_frames_dir) if os.path.isfile(os.path.join(args.video_frames_dir, name))])
        args.start_frame = 1
        args.end_frame = num_frames
        args.first_frame_type = 'content'
        args.max_iterations = 10 # 1000
        args.first_frame_iterations = 20 # 2000
        args.init_frame_type = 'prev' # 'prev_warped'
        args.content_frame_frmt = 'frame{}.jpg'
        args.video_input_dir = args.video_output_dir # frames _e_

        found_q_frames = len([name for name in os.listdir(args.video_styled_dir) if os.path.isfile(os.path.join(args.video_styled_dir, name))])
        start_frame = found_q_frames + 1
    
        frame = args.start_frame
        frame_name = args.content_frame_frmt.format(str(frame).zfill(args.zfill))
        frame_img_path = os.path.join(args.video_frames_dir, frame_name) 

    print(f"|===> nnani video \n \
        cwd: {os.getcwd()} \n \
        num_frames: {num_frames} \n \
        video_input_path: {video_input_path} \n \
        img_output_dir: {args.img_output_dir} \n \
        content_filename: {content_filename} \n \
        extension: {extension} \n \
        style_imgs_dir: {args.style_imgs_dir} \n \
        style_imgs: {args.style_imgs_files} \n \
        video_file: {args.video_file} \n \
        video_output_dir: {args.video_output_dir} \n \
        video: {args.video} \n \
        args.video_frames_dir: {args.video_frames_dir} \n \
        end_frame: {args.end_frame} \n \
        input_shape: {args.input_shape} \n \
        content_frame_frmt: {args.content_frame_frmt} \n \
        frame: {frame} \n \
        frame_name: {frame_name} \n \
        frame_img_path: {frame_img_path} \n \
    ")


    if 1: # content
        content_frame = onfile.path_cv_pil(frame_img_path)
        content_frame = onformat.pil_to_dnua(content_frame)
        if 1 : onplot.pil_show_nua(content_frame)

    if 1: # style
        style_imgs = get_style_cvis(content_frame, args)
        style_imgs = onformat.cvis_to_pils(style_imgs)
        style_imgs = onformat.pils_to_dnuas(style_imgs)
        if 1 : onplot.pil_show_nuas(style_imgs)

    if 1: # input shape
        input_img = get_init_image(args.first_frame_type, content_frame, style_imgs, frame, args)
        b,w,h,c = np.shape(input_img)
        input_shape = (w,h,c)    

    print(f'|===> nnani  \n \
        cwd: {os.getcwd()} \n \
        video_input_path: {video_input_path} \n \
        img_output_dir: {args.img_output_dir} \n \
        content_filename: {content_filename} \n \
        content_frame_type: {type(content_frame)} \n \
        extension:  {extension} \n \
        style_imgs_dir: {args.style_imgs_dir} \n \
        style_imgs: {args.style_imgs_files} \n \
        q_style_imgs: {len(args.style_imgs_files)} \n \
        video_file: {args.video_file} \n \
        video_output_dir: {args.video_output_dir} \n \
        video:      {args.video} \n \
        args.video_frames_dir: {args.video_frames_dir} \n \
        start_frame: {args.start_frame} \n \
        end_frame:  {args.end_frame} \n \
        num_frames: {num_frames} \n \
        input_shape: {args.input_shape} \n \
        content_frame_frmt: {args.content_frame_frmt} \n \
    ')

    # _e_ !!!
    if 1: # model
        args.video_output_dir = args.video_styled_dir
        args.img_output_dir = img_output_dir

        print(f"|===> nnani model \n \
            cwd: {os.getcwd()} \n \
            args.video_output_dir: {args.video_output_dir} \n \
            args.img_output_dir: {args.img_output_dir} \n \
        ")

        model = Styler(
            input_shape = input_shape,          
            args = args,
        )

    if 1: # fit

        args.max_iterations = args.frame_iterations
        for frame in range(args.start_frame, args.end_frame+1):
            print("frame", frame)
            print('\n---- RENDERING VIDEO FRAME: {}/{} ----\n'.format(frame, args.end_frame))

            if frame == args.start_frame:
                input_img = get_init_image(args.first_frame_type, content_frame, style_imgs, frame, args)
            else:
                input_img = get_init_image(args.init_frame_type, content_frame, style_imgs, frame, args)
            if 1:
                onplot.pil_show_nua(input_img ,"init image")

            tick = time.time()

            model.fit( 
                input_img, content_frame, style_imgs,
                # frame = frame,
                args = args,
            )

            tock = time.time()
            print('Frame {} elapsed time: {}'.format(frame, tock - tick))    


#   ******************
#
#   nnvid
#
def nnvid(args, kwargs):

    args = onutil.pargs(vars(args))
    onutil.ddict(vars(args), 'args')

    if 1: # config
        args.start_frame = 1
        args.end_frame = -1 # num_frames
        args.max_iterations = 10 # 1000
        args.first_frame_iterations = 20 # 2000
        args.video = True 
        args.content_frame_frmt = 'frame{}.ppm'

    print(f"|===> nnvid config \n \
        cwd: {os.getcwd()} \n \
        args.start_frame: {args.start_frame} \n \
        args.end_frame: {args.end_frame} \n \
        args.max_iterations: {args.max_iterations} \n \
        args.first_frame_iterations: {args.first_frame_iterations} \n \
        args.content_frame_frmt: {args.content_frame_frmt} \n \
    ")

    if 1: # tree: video file, frames folder
        args.video_file = 'portu.mp4'
        args.video_input_dir = os.path.join(args.proto_dir, 'data')

        content_filename=os.path.splitext(os.path.basename(args.video_file))[0]
        extension=os.path.splitext(os.path.basename(args.video_file))[1]
        content_filename=f"{content_filename}" # portu

        video_input_path = os.path.join(args.data_dir, args.video_file)
        args.video_frames_dir=os.path.join(args.video_input_dir, content_filename, 'frames')
        args.video_styled_dir=os.path.join(args.video_input_dir, content_filename, 'styled')

        os.makedirs(args.video_input_dir, exist_ok=True)         
        os.makedirs(args.video_frames_dir, exist_ok=True)         
        os.makedirs(args.video_styled_dir, exist_ok=True)         

    print(f"|===> nnvid video tree \n \
        cwd: {os.getcwd()} \n \
        content_filename: {content_filename} \n \
        args.video_file: {args.video_file} \n \
        args.video_input_dir: {args.video_input_dir} \n \
        args.video_frames_dir: {args.video_frames_dir} \n \
        args.video_styled_dir: {args.video_styled_dir} \n \
        video_input_path: {video_input_path} \n \
    ")


    if 1: # vid => frames
        if 1:
            print(f"|===> nnani vid to frames \n \
                video_input_path: {video_input_path} \n \
                args.video_frames_dir: {args.video_frames_dir} \n \
            ")
            onvid.vid_to_frames(video_input_path, args.video_frames_dir, target = 1)
        else:
            frame_pattern = 'frame%04d.ppm'
            cmd = f'ffmpeg -v quiet -i "{video_input_path}" "{args.video_frames_dir}/{frame_pattern}"'
            print(f"cmd: {cmd}")
            os.system(cmd)


    if 1: # frames size
        ppm = os.path.join(args.video_frames_dir, 'frame0000.jpg')
        im = Image.open(ppm)
        width, height = im.size   
        max_size = max(width, height)

        print(f"|===> nnani frame size \n \
            max_size: {max_size} \n \
        ")


    if 1: # frames number
        qframes = len([name for name in os.listdir(args.video_frames_dir) 
            if os.path.isfile(os.path.join(args.video_frames_dir, name))])

        qstyled = len([name for name in os.listdir(args.video_styled_dir) 
            if os.path.isfile(os.path.join(args.video_styled_dir, name))])

        startFrame = qstyled + 1
        endFrame = qframes
        stepSize = 1
        frame_format = 'frame{}.ppm'

        print(f"|===> nnani frame \n \
            qframes: {qframes} \n \
            qstyled: {qstyled} \n \
            startFrame: {startFrame} \n \
            endFrame: {endFrame} \n \
            stepSize: {stepSize} \n \
            frame_format: {frame_format} \n \
        ")


    i=startFrame
    j=startFrame + stepSize
    while i < 2: # True:
        # fwdname = f"{videoFolder}/forward_{i}_{j}.flo"
        fwdname = f"{args.video_frames_dir}/forward_{i}_{j}.png"
        fwdname_horz = f"{args.video_frames_dir}/forward_horz_{i}_{j}.png"
        fwdname_vert = f"{args.video_frames_dir}/fwdname_vert_{i}_{j}.png"
        bckname = f"{args.video_frames_dir}/backward_{i}_{j}.flo"
        rlname = f"{args.video_frames_dir}/reliable_{i}_{j}.txt"      
        file1 = frame_format.format(str(i).zfill(4))
        file2 = frame_format.format(str(j).zfill(4))
        path1 = os.path.join(args.video_frames_dir, file1)
        path2 = os.path.join(args.video_frames_dir, file2)
        print(f"paths: {path1} {path2}")
        if os.path.exists(path1) and os.path.exists(path2):
            frame1 = cv2.imread(path1)
            print(f'|... {file1} {frame1.shape}')
            prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            hsv = np.zeros_like(frame1)        
            frame2 = cv2.imread(path2)

            # https://www.programcreek.com/python/example/89313/cv2.calcOpticalFlowFarneback
            # https://github.com/wolfd/ocular/tree/master/scripts
            next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
            prvs, next,None, 0.5, 3, 15, 3, 5, 1.2, 0)

            horz = cv2.normalize(flow[..., 0], None, 0, 255, cv2.NORM_MINMAX)
            vert = cv2.normalize(flow[..., 1], None, 0, 255, cv2.NORM_MINMAX)
            horz = horz.astype('uint8')
            vert = vert.astype('uint8')

            # Change here too
            if 0:
                cv2.imshow('Horizontal Component', horz)
                cv2.imshow('Vertical Component', vert)

            k = cv2.waitKey(0) & 0xff
            if k == ord('s'):  # Change here
                cv2.imwrite('opticalflow_horz.pgm', horz)
                cv2.imwrite('opticalflow_vert.pgm', vert)

            cv2.destroyAllWindows() 
            # print(f"save {fwdname}") 
            # cv2.imwrite(fwdname,rgb) # cv2.imwrite(os.path.join(folder,"dense_flow_" + str(pt) + ".png"),rgb)

            i += 1
            j += 1
        else:
            break

        
    print(f"|===> nnvid render stylized video \n \
        cwd: {cwd} \n \
        video: {args.video} \n \
        ")

    cmd = f'python neural_style.py --video \
      --video_input_dir "{args.video_frames_dir}" \
      --style_imgs_dir "{args.style_imgs_dir}" \
      --style_imgs {args.style_imgs_files[0]} \
      --end_frame {end_frame} \
      --max_size {max_size} \
      --verbose'
    print(cmd)


    # fit args

    frame = 1 # args.start_frame
    frame_name = frame_format.format(str(frame).zfill(args.zfill))
    frame_img_path = os.path.join(args.video_frames_dir, frame_name) 

    # content
    print(f"|===> nnvid content \n \
        args.video_frames_dir: {args.video_frames_dir} \n \
        frame: {frame} \n \
        frame_name: {frame_name} \n \
        frame_img_path: {frame_img_path} \n \
        ")

    content_frame = onfile.path_cv_pil(frame_img_path)
    content_frame = onformat.pil_to_dnua(content_frame)
    if 1 : onplot.pil_show_nua(content_frame)

    # style
    style_imgs = get_style_cvis(content_frame, args)
    style_imgs = cvis_to_pils(style_imgs)
    style_imgs = pils_to_dnuas(style_imgs)
    if 1 : onplot.pil_show_nuas(style_imgs)

    # input shape
    input_img = get_init_image(args.first_frame_type, content_frame, style_imgs, frame, args)
    b,w,h,c = np.shape(input_img)
    input_shape = (w,h,c)    

    print(f"|===> nnani shape \n \
    cwd: {cwd} \n \
    content_frame: {np.shape(content_frame)} \n \
    style[0]shape: {np.shape(style_imgs[0])} \n \
    input_img: {np.shape(input_img)} \n \
    input_shape: {input_shape} \n \
    ")


    args.video_output_dir = args.video_styled_dir
    args.content_frame_frmt = frame_format

    if 1: # model
        model = Styler(
            input_shape = input_shape,          
            args = args,
        )

    print(f"|===> nnani fit \n \
        cwd: {cwd} \n \
        args.start_frame: {args.start_frame} \n \
        args.end_frame: {args.end_frame} \n \
    ")

    args.max_iterations = args.frame_iterations
    for frame in range(start_frame, end_frame+1):
        print("frame", frame)
        print('\n---- RENDERING VIDEO FRAME: {}/{} ----\n'.format(frame, end_frame))

        if frame == start_frame:
            input_img = get_init_image(args.first_frame_type, content_frame, style_imgs, frame, args)
        else:
            input_img = get_init_image(args.init_frame_type, content_frame, style_imgs, frame, args)
        if 1:
            onplot.pil_show_nua(input_img ,"init image")

        tick = time.time()

        model.fit( 
            input_img, content_frame, style_imgs,
            # frame = frame,
            args = args,
        )

        tock = time.time()
        print('Frame {} elapsed time: {}'.format(frame, tock - tick))    


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
            print(f'|===> call {name}')
            globals()[name](args, kwargs) # pass args to nn cmd

#---------------------------------------------------------------
# python base/base.py nninfo
if __name__ == "__main__":
    print("|===>", __name__)
    main()

#---------------------------------------------------------------
