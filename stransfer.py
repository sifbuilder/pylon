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
onvgg = Onvgg()
onlllyas = Onlllyas()
#
#
#   CONTEXT
#
#
def getap():
    cp = {
        "primecmd": "nnani",        

        "MNAME": "stransfer",      
        "AUTHOR": "xueyangfu",      
        "PROJECT": "building",      
        "GITPOD": "neural-style-tf",      
        "DATASET": "styler",        
    
        "TRAIN": 1,              # train model

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
#
def getxp(cp):

    yp = {
        # "model_weights": os.path.join(cp["gmodel"], 'vgg', 'vgg19_weights_tf_dim_ordering_tf_kernels.h5'),
        "model_weights": os.path.join(cp["gmodel"], 'vgg', 'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'), # vgg19
        "model_include_top": 0,
        "model_classes": 1000,

        "max_epochs": 80, # 800, # 10,
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
        "frame_content_frmt": 'frame{}.jpg',

        "content_imgs_weights": [1.0],
        "content_layers_weights": [2.5e-08],
        "content_loss_function": 1,
        "content_weight": 5e0,
        "content_weights_frmt": 'reliable_{}_{}.txt',

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
        "style_weight": 1e4,
        "style_imgs_weights": [1.0],
        "style_layers_weights": [0.01, 0.01, 0.01, 0.01, 0.01],
        "style_layers_weights": [0.2, 0.2, 0.2, 0.2, 0.2],
        "style_weight": 1e4,
        "style_mask": 0,

        "max_size": 512,
        "input_shape": (224, 224, 3), # (512, 512, 3) # 
        "frame_first_iterations": 2000,
        "frame_first_type": 'content', # args.frame_first_type ['random', 'content', 'style'
        "init_image_type": "content", # 'content','style','init','random','prev','prev_warped'
        "frame_init_type": 'prev',
        "init_img_dir": cp["data_dir"],
        "init_image_type": "content",
        "frame_first_iterations": 2000,
        "init_img_type": 'content',
        "img_name": 'result',

        "original_colors": None,
        "color_convert_type": 'yuv',
        "color_convert_time": 'after',
        "optimizer": 'adam', # 'lbfgs' # 
        
        # "total_variation_weight": 1e-6, # 300,
        "total_variation_weight": 0.001,
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
        "video": 0,

        "frames_dir": os.path.join(cp["results_dir"], 'cromes/frames'),
        "frame_iterations": 800,
        "frame_end": 9,
        "frame": None,    
        "frame_start": 0,
        "frame_end": -1, # num_frames
        "frame_first_iterations": 20, # 2000
        "frame_init_type": 'prev', # 'prev_warped'
        "frame_content_frmt": 'frame{}.jpg',
        "frame_first_type": 'content',

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
#
#
#   FUNS VIDEO
#
#
def get_content_frame(frame, args):
    video_input_dir = args.video_input_dir
    frame_content_frmt = args.frame_content_frmt

    # frame_content_frmt = 'frame_{}.ppm' # args.frame_content_frmt
    zfill = 4 # args.zfill
    frame_name = frame_content_frmt.format(str(frame).zfill(zfill))
    # path_to_img = os.path.join(video_input_dir, frame_name)

    # print("get_content_frame path", path_to_img)    
    # img = read_image(path_to_img)

    path = os.path.join(args.video_input_dir, frame_name)
    img = onfile.path_to_tnua_tf(path, args)
    return img
#
def _get_style_images(content_img, args=None):
  _, ch, cw, cd = content_img.shape
  style_imgs = []
  for style_fn in args.style_imgs_files:
    path = os.path.join(args.style_imgs_dir, style_fn)
    # bgr image
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img.astype(np.float32)
    img = cv2.resize(img, dsize=(cw, ch), interpolation=cv2.INTER_AREA)
    img = onvgg.vgg_preprocess(img)
    style_imgs.append(img)
  return style_imgs
#
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
#
def get_style_cvis(content_img, args):
    if args.verbose >0:
        print(f'|---> get_style_cvis  \n \
            content_img: {args.style_imgs_files} \n \
            args.style_imgs_dir: {args.style_imgs_dir} \n \
        ')

    style_imgs=[]
    style_imgs_dir=args.style_imgs_dir
    max_size=args.max_size
    style_scale =args.style_scale

    _, ch, cw, cd = content_img.shape #  (1, 256, 256, 3) [[[[-107.68   -90.779  -78.939]
    mx = max_size

    cvis = []
    for style_fn in args.style_imgs_files:
        path = os.path.join(style_imgs_dir, style_fn)
        if args.verbose >0:
            print(f'|...> get_style_cvis path {path}')        
        # bgr image
        cvi = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2_frame(cvi, content_img, style_scale)

        # img = onvgg.vgg_preprocess(img)
        cvis.append(img)
    return cvis
#
def get_input_image(
        init_type= 'content', # {content,style,init,random,prev,prev_warped}
        content_img=None, 
        style_imgs=[], 
        init_img=None, 
        frame=None, 
        video_input_dir='./', 
        video_output_dir='./', 
        frame_content_frmt = 'frame_{}.ppm',
        zill=4,
        noise_ratio = 1.0, 
        args=None
    ):
    if args.verbose >0:
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
        #init_img = get_prev_frame(
        #    frame, 
        #    video_output_dir,
        #    frame_content_frmt,
        #    zill,
        #)
        init_img = get_prev_frame(
            frame, 
            args
        )        
        return init_img
    elif init_type == 'prev_warped':
        #init_img = get_prev_warped_frame(
        #    frame, 
        #    video_input_dir
        #)
        init_img = get_prev_warped_frame(
            frame, 
            args
        )        
        return init_img
#
def get_optimizer(loss):
    print_iterations = 100 # 
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
#
def get_noise_image(noise_ratio, content_img, 
        seed = 0
    ):
    np.random.seed(seed)
    noise_img = np.random.uniform(-20., 20., content_img.shape).astype(np.float32)
    img = noise_ratio * noise_img + (1.-noise_ratio) * content_img
    return img
#
def get_mask_image(mask_img, width, height, 
        content_imgs_dir = './',
        args = None
    ):
    path = os.path.join(content_imgs_dir, mask_img)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    #check_file(img, path)
    if args.verbose > 0:
        print(f'|---> get_mask_image {path}')
    img = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32)
    mx = np.amax(img)
    img /= mx
    return img
#
def get_prev_frame(frame, args=None):
    #   get_prev_frame: previously stylized frame    
    video_output_dir = args.video_input_dir # _e_ tbc
    frame_content_frmt = args.frame_content_frmt
    zfill = args.zfill
    
    prev_frame = frame - 1
    fn = frame_content_frmt.format(str(prev_frame).zfill(zfill))
    path = os.path.join(video_output_dir, fn)
    if args.verbose > 0:
        print(f"|---> get_prev_frame for frame: {frame} from {path}")

    # img = cv2.imread(path, cv2.IMREAD_COLOR)
    # check_file(img, path)

    img = onfile.path_to_pil_cv(path)
    img = onformat.pil_to_dnua(img)

    return img
#
def get_prev_warped_frame(frame, args=None):

    video_input_dir = args.video_input_dir
    backward_optical_flow_frmt = args.backward_optical_flow_frmt
    frame_content_frmt = args.frame_content_frmt

    prev_img = get_prev_frame(frame, args)
    if args.verbose > 0:
        print(f"|---> get_prev_frame in {video_input_dir} with {frame_content_frmt}")
    prev_frame = frame - 1
    # backwards flow: current frame -> previous frame
    fn = backward_optical_flow_frmt.format(str(frame), str(prev_frame))
    path = os.path.join(video_input_dir, fn)
    flow = read_flow_file(path)
    warped_img = warp_image(prev_img, flow).astype(np.float32)
    img = onvgg.vgg_preprocess(warped_img)
    return img
#
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
#
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
#
#
#   FUNCS LOG
#
def write_image_output(output_img, 
        content_img, style_imgs, input_img, 
        epoch=0,
        args=None,
    ):

    if args.verbose > 0:
        print(f'|---> write_image_output \n \
            to img_output_dir: {args.img_output_dir} \n \
            epoch: {epoch} \n \
        ')

    output_img = onformat.nua_to_pil(output_img)
    content_img = onformat.nua_to_pil(content_img)
    style_imgs = onformat.dnuas_to_pils(style_imgs)
    input_img = onformat.nua_to_pil(input_img)

    # save result image
    img_file = f'output_'+str(epoch).zfill(args.zfill)+'.png'
    img_path = os.path.join(args.img_output_dir, img_file)
    onfile.pil_to_file_cv(img_path, output_img)

    # save content image
    content_file = f'content.jpg'
    content_path = os.path.join(args.img_output_dir, content_file)
    onfile.pil_to_file_cv(content_path, content_img)

    # save input image
    init_file = f'init.jpg'
    init_path = os.path.join(args.img_output_dir, init_file)
    onfile.pil_to_file_cv(init_path, input_img)

    style_files = []
    style_paths = []

    # save style images
    for i, style_img in enumerate(style_imgs):
        style_files.append(f'style_{str(i)}.png')
        style_paths.append(os.path.join(args.img_output_dir, style_files[i]))
        path = style_paths[i]
        onfile.pil_to_file_cv(path, style_img)

    # save the configuration settings
    out_file = os.path.join(args.img_output_dir, 'meta_data.txt')
    f = open(out_file, 'w')
    f.write(f'image_name: {img_path}\n')
    f.write(f'content: {content_path}\n')

    index = 0
    for i, style_path in enumerate(style_paths):
        f.write(f'styles[{str(i)}]: {style_path}\n')

    if 'mask_imgs_files' in vars(args).keys():
        for i, style_mask_img in enumerate(args.mask_imgs_files):
            style_mask_img_path = os.path(args.style_imgs_dir, args.mask_imgs_files[i])
            f.write(f'style_masks[{str(i)}]: {style_mask_img_path}\n')        

    f.write(f'init_type: {args.init_img_type}\n')
    f.write(f'content_weight: {args.content_weight}\n')
    f.write(f'content_layers: {args.content_layers}\n')
    f.write(f'content_layers_weights: {args.content_layers_weights}\n')
    f.write(f'style_imgs_weights: {args.style_imgs_weights}\n')
    f.write(f'style_layers: {args.style_layers}\n')
    f.write(f'style_layers_weights: {args.style_layers_weights}\n')
    f.write(f'total_variation_weight: {args.total_variation_weight}\n')
    f.write(f'content_size: {args.content_size}\n')
    f.write(f'input_shape: {args.input_shape}\n')
    f.write(f'optimizer: {args.optimizer}\n')
    f.write(f'max_size: {args.max_size}\n')
    f.write(f'frame_iterations: {args.frame_iterations}\n')
    f.write(f'max_epochs: {args.max_epochs}\n')
    f.write(f'steps_per_epoch: {args.steps_per_epoch}\n')
    f.write(f'print_iterations: {args.print_iterations}\n')
    f.close()
#
#
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
#
def get_content_weights(frame, prev_frame, args=None):

    content_weights_frmt = 'reliable_{}_{}.txt' # args.content_weights_frmt
    video_input_dir = './video_input' # args.video_input_dir

    content_weights_frmt = args.content_weights_frmt
    video_input_dir=args.video_input_dir
    
    forward_fn = content_weights_frmt.format(str(prev_frame), str(frame))
    backward_fn = content_weights_frmt.format(str(frame), str(prev_frame))
    forward_path = os.path.join(video_input_dir, forward_fn)
    backward_path = os.path.join(video_input_dir, backward_fn)
    forward_weights = read_weights_file(forward_path)
    backward_weights = read_weights_file(backward_path)
    return forward_weights #, backward_weights
#
def minimize_with_lbfgs(sess, net, optimizer, init_img,
        verbose=True,
    ):
    if verbose: print('\nMINIMIZING LOSS USING: L-BFGS OPTIMIZER')
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    sess.run(net['input'].assign(init_img))
    optimizer.minimize(sess)
#
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
#
#
#   NETS
#
# a neural algorithm for artistic style' loss functions
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
#
def style_layer_loss(a, x):
    if 0:
        print(f'|---> style_layer_loss')      
    _, h, w, d = a.get_shape()
    M = h * w # h.value * w.value
    N = d # d.value
    A = gram_matrix(a, M, N)
    G = gram_matrix(x, M, N)
    loss = (1./(4 * N**2 * M**2)) * tf.reduce_sum(input_tensor=tf.pow((G - A), 2))
    return loss
#
def gram_matrix(x, area, depath):
    F = tf.reshape(x, (area, depath))
    G = tf.matmul(tf.transpose(a=F), F)
    return G
#
def mask_style_layer(a, x, mask_img, args=None):
    if args.verbose > 0:
        print(f'|---> mask_style_layer {mask_img}')    
    _, h, w, d = a.get_shape()
    #mask = get_mask_image(mask_img, w.value, h.value)
    mask = get_mask_image(mask_img, w, h,
        content_imgs_dir=args.style_imgs_dir,
        args=args) # _e_
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
#
def sum_masked_style_losses(net, combo, style_imgs, args=None):
    if args.verbose > 0:
        print(f'|---> sum_masked_style_losses')
    total_style_loss = 0.
    weights = args.style_imgs_weights

    masks = args.style_mask_imgs

    for img, img_weight, img_mask in zip(style_imgs, weights, masks):
        #net['input'].assign(img)
        ps = net.extract_features([combo], args.style_layers)[0]   # net[layer]
        xs = net.extract_features([img], args.style_layers)[0]   # net[layer]          
        style_loss = 0.
        for layer, weight in zip(args.style_layers, args.style_layers_weights):
            a = ps[layer] # net[layer]
            x = xs[layer] # net[layer]
            #a = tf.convert_to_tensor(value=a)
            a, x = mask_style_layer(a, x, img_mask, args)
            style_loss += style_layer_loss(a, x) * weight
        style_loss /= float(len(args.style_layers))
        total_style_loss += (style_loss * img_weight)
    total_style_loss /= float(len(style_imgs))
    return total_style_loss
#
def sum_style_losses(net, combo, style_imgs, args=None):
    if args.verbose > 0:
        print(f'|---> sum_style_losses')    
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
#
def sum_content_losses(net, combo, content_img, args=None):
    if args.verbose > 0:
        print(f'|---> sum_content_losses')     
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
#
# artistic style transfer for videos' loss functions
def temporal_loss(x, w, c):
  c = c[np.newaxis,:,:,:]
  D = float(x.size)
  loss = (1. / D) * tf.reduce_sum(input_tensor=c * tf.nn.l2_loss(x - w))
  loss = tf.cast(loss, tf.float32)
  return loss
#
def get_longterm_weights(i, j, args=None):
  c_sum = 0.
  for k in range(args.prev_frame_indices):
    if i - k > i - j:
      c_sum += get_content_weights(i, i - k, args)
  c = get_content_weights(i, i - j, args)
  c_max = tf.maximum(c - c_sum, 0.)
  return c_max
#
def sum_longterm_temporal_losses(sess, net, frame, input_img, args=None):
  x = sess.run(net['input'].assign(input_img))
  loss = 0.
  for j in range(args.prev_frame_indices):
    prev_frame = frame - j
    w = input_img # _e_
    if args.frame_init_type == 'prev':
        w = get_prev_frame(frame, args) # _e_
    elif args.frame_init_type == 'prev_warped':
        w = get_prev_warped_frame(frame, args)

    c = get_longterm_weights(frame, prev_frame)
    loss += temporal_loss(x, w, c)
  return loss
#
def sum_shortterm_temporal_losses(sess, net, frame, input_img, args=None):
  x = sess.run(net['input'].assign(input_img))
  prev_frame = frame - 1
  w = input_img # _e_
  if args.frame_init_type == 'prev':
      w = get_prev_frame(frame, args) # _e_
  elif args.frame_init_type == 'prev_warped':
      w = get_prev_warped_frame(frame, args)
  c = get_content_weights(frame, prev_frame, args) # _e_
  loss = temporal_loss(x, w, c)
  return loss
#
# utilities and i/o
def read_image(path):
  # bgr image
  img = cv2.imread(path, cv2.IMREAD_COLOR)
  check_image(img, path)
  img = img.astype(np.float32)
  img = onvgg.vgg_preprocess(img)
  return img
#
def write_image(path, img):
  img = onvgg.vgg_deprocess(img)
  cv2.imwrite(path, img)
#
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
#
def preprocess_tensors(inputs):
    pretensors = []
    for item in inputs:
        preitem = preprocess_tensor(item)
        pretensors.append(preitem)

    return pretensors
#
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
#
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
#
def normalize(weights):
    denom = sum(weights)
    if denom > 0.:
        return [float(i) / denom for i in weights]
    else: 
        return [0.] * len(weights)
#
def check_image(img, path):
    if img is None:
        raise OSError(errno.ENOENT, "No such file", path)
#
# rendering -- where the magic happens
def compute_loss(combo):
    loss = tf.zeros(shape=())
#
def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
#
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
        print("|...> load model_weights")
        vgg.load_weights(model_weights)
    else:
        print("|...> did not load model_weights")

    return vgg
#
#
class GAN(tf.keras.models.Model):

    def __init__(self, 
        input_shape = (224, 224, 3),        
        args=None, 
    ):

        super(GAN, self).__init__()

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
        if args.verbose > 0:
            print("|...> GAN content_layers", content_layers)
            print("|...> GAN style_layers", style_layers)
            print("|...> GAN layers", layer_names)
        outputs = {name: vggmodel.get_layer(name).output for name in layer_names}        
        # outputs = [vggmodel.get_layer(name).output for name in layer_names]
        self.net = tf.keras.Model([vggmodel.input], outputs) 
        self.net.trainable = False        
        if args.verbose > 0:
            self.net.summary()

        self.optimizer = tf.optimizers.Adam(
            learning_rate=0.01,
            beta_1=0.99, 
            epsilon=1e-1
        )

        #self.optimizer = tf.keras.optimizers.SGD(
        #    learning_rate=0.1, 
        #    momentum=0.9, 
        #    #nesterov=False, 
        #    #name='SGD'
        #)    

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
            if 0:
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
        print(f'|===> GAN call {type(image)}')
        img = self.image # <class 'tensorflow.python.ops.resource_variable_ops.ResourceVariable'>
        img = onformat.nua_to_pil(img)
        img.show()
    #
    #
    #   train_step
    #
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
    #
    #
    #   fit
    #
    def fit(self, input_img, content_img, style_imgs, 
            frame = None,
            args  = None
        ):

        max_epochs = args.max_epochs
        steps_per_epoch = args.steps_per_epoch
        video = args.video
        visual = args.visual
        verbose = args.verbose
        img_output_dir = args.img_output_dir
        video_styled_dir = args.video_styled_dir
        image_step_format = args.image_step_format
        image_epoch_format = args.image_epoch_format
        frame_content_frmt = args.frame_content_frmt
        zfill = args.zfill
        show_fit_imgs = args.show_fit_imgs

        train_step = self.train_step

        image = tf.Variable(input_img)   
        self.image = image

        print(f'|===> fit \n \
            max_epochs: {max_epochs} \n \
            steps_per_epoch: {steps_per_epoch} \n \
            train_step: {train_step} \n \
            input_img shape: {np.shape(input_img)} \n \
            content_img shape: {np.shape(content_img)} \n \
            style_imgs shape: {[np.shape(img) for img in style_imgs]} \n \
            combo_img shape: {type(self.image)} {np.shape(self.image)} \n \
            style_mask: {args.style_mask} \n \
        ')

        if visual > 1:
            print(f'|---> pil input_img')
            if 1: onplot.pil_show_nua(input_img)        

            print(f'|---> pil content_img')
            if 1: onplot.pil_show_nua(content_img)        

            print(f'|---> pil style_imgs')
            if 1: onplot.pil_show_nuas(style_imgs)   

            print(f'|---> pil combo_img')
            if 1: onplot.pil_show_nua(self.image)

            if args.style_mask_imgs:
                print(f'|---> pil combo_img')
                onplot.pil_show_nua(self.image)

        # # ============

        start = time.time()

        #if 0: print(f"[   .   ] display input image {step} image")
        #img = onformat.nua_to_pil(image)
        #img = ondata.pil_resize(img, (256,256))
        #img.show()

        for epoch in range(max_epochs):
            step = 0
            for step_in_epoch in range(steps_per_epoch):
                step += 1

                if 1 and step % args.print_iterations == 0:
                    if video:
                        print(f'|---> fit {frame}:{epoch}:{step}')
                    else:
                        print(f'|---> fit {epoch}:{step}')

                # ==========================================

                train_step(image, content_img, style_imgs, args)

                # =========================================

                print(".", end='')

            if 1 and step % args.print_iterations == 0: # show results each n steps
                if 0:
                    try:
                        # display image per epoch _e_
                        display.clear_output(wait=True)
                        print(f"[   .   ] display in step {step} image")
                        display.display(onformat.nua_to_pil(self.image))
                    except:
                        print("could not display epoch images")
                else:
                    print(f"[   .   ] show epoch {epoch} image")
                    img = onformat.nua_to_pil(self.image)
                    img.show()

            # write image per epoch
            img_name = image_epoch_format.format(str(epoch).zfill(zfill))        
            img_path = os.path.join(img_output_dir, img_name)
            #onfile.pil_to_file_cv(img_path, self.image)

            if 1: # write image per epoch
                write_image_output(
                    output_img = self.image, 
                    content_img = content_img, 
                    style_imgs = style_imgs, 
                    input_img = input_img,
                    epoch = epoch,
                    args = args,
                )

            if visual > 1:
                if 1:
                    print(f'|---> pil combo_img epoch: {epoch}')
                onplot.pil_show_nua(self.image)

        end = time.time()
        print()
        print("|...> Total time: {:.1f}".format(end-start))

        if video:
            frame_name = args.frame_content_frmt.format(str(frame).zfill(zfill))
            outpath_path = os.path.join(video_styled_dir, frame_name) # styled
            print(f"|...> fit save frame to {outpath_path} on epoch {epoch}")
            output_img = onformat.nua_to_pil(image)
            onfile.pil_to_file_cv(outpath_path, output_img)

#
#
#   CMDS
#
#
#
#
#   nnimg
#
#   stylize
#    sum_style_losses
#    sum_masked_style_losses args.style_mask
def nnimg(args, kwargs):

    if 0:
        args.PROJECT = 'bomze' # https://github.com/tg-bomze/Style-Transfer-Collection
        args.DATASET = 'bomze'
        args.content_img_file = 'bomze-content.png'
        args.content_size = (512, 512)        
        args.init_img_name = 'bomze-content.png'
        args.style_imgs_files = ['bomze-style.png']

    if 0:
        args.PROJECT = 'building'
        args.DATASET = 'building'
        args.content_img_file = 'IMG_4468.JPG'
        args.content_size = (512, 512)        
        args.init_img_name = 'IMG_4468.JPG'
        args.style_imgs_files = ['EZNJVoTXsAgUh6M.jpg', 'EZsQWC1X0AQaxHs.jpg']

    if 0:
        args.PROJECT = 'labrador'
        args.DATASET = 'labrador'        
        args.content_img_file = 'YellowLabradorLooking_new.jpg'
        args.content_size = (512, 512)        
        args.init_img_name = 'YellowLabradorLooking_new.jpg'
        args.style_imgs_files = ['starry-night.jpg']

    if 0:
        args.PROJECT = 'kandinsky'
        args.DATASET = 'Kandinsky'        
        args.content_img_file = 'YellowLabradorLooking_new.jpg'
        args.content_size = (512, 512)        
        args.init_img_name = 'YellowLabradorLooking_new.jpg'
        args.style_imgs_files = ['Vassily_Kandinsky,_1913_-_Composition_7.jpg']

    if 0:
        args.PROJECT = 'lion'
        args.DATASET = 'lion'        
        args.content_img_file = 'lion.jpg'
        args.content_size = (512, 512)        
        args.init_img_name = 'lion.jpg'
        args.style_imgs_files = ['a-hymn-to-the-shulamite-1982.jpg']

    if 1:
        args.PROJECT = 'lion'
        args.DATASET = 'lion'        
        args.content_img_file = 'lion.jpg'
        args.content_size = (512, 512)        
        args.init_img_name = 'lion.jpg'
        args.style_imgs_files = ['starry-night.jpg']

    if 0:
        args.PROJECT = 'madrid'
        args.DATASET = 'madrid'          
        args.content_img_file = 'madrid.JPG'
        args.content_size = (512, 512)
        args.init_img_name = 'madrid.JPG'
        args.style_imgs_files = ['starry-night.jpg']

        #args.style_mask_imgs_files = [onimg.name_to_maskname(item) for item in args.style_imgs_files]
        args.style_mask_imgs_files = ['madrid_mask.JPG' for item in args.style_imgs_files]
        #args.style_imgs_thress = [(100,cv2.THRESH_BINARY)]
        #args.style_imgs_thress = [(145,cv2.THRESH_BINARY)]
        #args.style_imgs_thress = [(38,cv2.THRESH_BINARY)]
        args.style_imgs_thress = [(38, cv2.THRESH_BINARY_INV)]

    if 0:
        args.PROJECT = 'portubridge'
        args.DATASET = 'portubridge'        
        args.content_img_file = 'portu_frame0000.jpg'
        args.content_size = (512, 512)        
        args.init_img_name = 'portu_frame0000.jpg'
        args.style_imgs_files = ['kandinsky.jpg']

    args = onutil.pargs(vars(args))
    xp = getxp(vars(args))
    args = onutil.pargs(xp)    
    onutil.ddict(vars(args), 'args')

    if 1: # config

        print(f"|===> nnimg: config \n \
            args.video: {args.video} : content images from frames \n \
            args.show_entry_imgs: {args.show_entry_imgs} \n \
            args.steps_per_epoch: {args.steps_per_epoch} \n \
            args.model_weights: {args.model_weights} \n \
            args.max_epochs: {args.max_epochs} \n \
        ")

    if 1: # tree

        args.code_dir = os.path.join(args.proto_dir, 'code') # up project dir
        #args.data_dir = os.path.join(args.proto_dir, 'data')

        args.video_input_path = os.path.join(args.dataorg_dir, args.video_file)
        args.video_frames_dir=os.path.join(args.proj_dir, 'frames')
        args.video_styled_dir=os.path.join(args.proj_dir, 'outstyled')
        args.video_output_dir=os.path.join(args.proj_dir, 'outvid')
        args.video_input_dir = args.video_output_dir # frames _e_

        args.style_imgs_weights = normalize(args.style_imgs_weights)
        args.style_imgs_dir = args.data_dir
        style_base = os.path.splitext(args.style_imgs_files[0])[0]
        args.style_layers_weights = normalize(args.style_layers_weights)

        args.content_layer_weights = normalize(args.content_layer_weights)
        args.content_imgs_dir = args.data_dir
        content_base = os.path.splitext(args.content_img_file)[0]

        args.output_folder = content_base + "_" + style_base
        args.img_output_dir = os.path.join(args.results_dir, 'cromes', args.output_folder)

        args.init_img_dir = args.data_dir

        print(f"|===> nnimg: tree \n \
            args.output_folder: {args.output_folder} \n \
            args.img_output_dir: {args.img_output_dir} \n \
            \n \
            args.proto_dir: {args.proto_dir} \n \
            args.data_dir: {args.data_dir} \n \
            args.content_imgs_dir: {args.content_imgs_dir} \n \
            args.init_img_dir: {args.init_img_dir} \n \
            args.style_imgs_dir: {args.style_imgs_dir} \n \
        ")

        os.makedirs(args.data_dir, exist_ok=True)
        os.makedirs(args.video_output_dir, exist_ok=True)
        os.makedirs(args.img_output_dir, exist_ok=True)
        os.makedirs(args.content_imgs_dir, exist_ok=True) # data_dir

    if 1: # content image: (422, 512, 3) <class 'numpy.ndarray'>  [[[ 99 165 160]

        content_img_path = os.path.join(args.content_imgs_dir, args.content_img_file)
        img_file = os.path.basename(content_img_path)

        if not os.path.exists(content_img_path):

            urlfolder = 'https://github.com/xueyangfu/neural-style-tf/raw/master/image_input/'
            url = f'{urlfolder}{img_file}'
            topath = os.path.join(args.content_imgs_dir, f'{img_file}')

            print(f"|===> nnimg: content file does not exist\n \
                content_img_path: {content_img_path} \n \
                urlfolder: {urlfolder} \n \
                url: {url} \n \
                topath: {topath} \n \
                args.content_imgs_dir: {args.content_imgs_dir} \n \
            ")

            tofile = tf.keras.utils.get_file(f'{topath}', origin=url, extract=True)

        assert os.path.exists(content_img_path), f"content image {content_img_path} does not exist"

        img = onfile.path_to_pil_cv(content_img_path)
        img = ondata.pil_resize(img, ps=args.content_size)
        img = onformat.pil_to_dnua(img)
        print(f"|===> nnimg: content \n \
            content_img_path: {content_img_path} \n \
            args.content_imgs_dir: {args.content_imgs_dir} \n \
            img shape: {np.shape(img)} \n \
        ")
        content_img = img

    if args.visual > 1: # show content image
        print(f'|...> pil content img')
        onplot.pil_show_nua(content_img) # non interrupt

    if args.visual > 1:# show content image from path
        print(f'|...> cv content img')
        content_img_path = os.path.join(args.content_imgs_dir, args.content_img_file)
        onplot.cv_path(content_img_path, size = 512, title='content img', wait=2000)

    if 1: #   input image: (422, 512, 3) <class 'numpy.ndarray'> [[[ 99 165 160]
        init_path = os.path.join(args.init_img_dir, args.init_img_name)
        init_image = onfile.path_to_tnua_tf(init_path, args)
        print(f"|===> nnimg: init \n \
            init_path: {init_path} \n \
            args.content_imgs_dir: {args.content_imgs_dir} \n \
            init_shape: {np.shape(init_image)} \n \
        ")

        if args.visual > 1:
            print(f'|...> vis init')
            onplot.pil_show_nua(init_image, "[   .   ] init_image")

    if 1: #   style images
        style_imgs_paths = onfile.names_to_paths(args.style_imgs_files, args.style_imgs_dir)

        for path in style_imgs_paths:
            print(f'path: {path}')
            if not os.path.exists(path):
                print(f'path: {path} DOES NOT EXIST')

                urlfolder = 'https://raw.github.com/xueyangfu/neural-style-tf/master/styles/'
                file_name = os.path.basename(path)
                url = f'{urlfolder}{file_name}'
                topath = os.path.join(args.style_imgs_dir, file_name)
                tofile = tf.keras.utils.get_file(f'{topath}', origin=url, extract=True)

        style_imgs = onfile.names_to_nuas_tf(args.style_imgs_files, args.style_imgs_dir, args)
        print(f"|===> nnimg: styles \n \
            args.style_imgs_files: {args.style_imgs_files} \n \
            args.style_imgs_dir: {args.style_imgs_dir} \n \
            shapes: {[str(np.shape(style_imgs[i])) for i,img in enumerate(style_imgs)]} \n \
        ")

        if args.visual > 1:
            print(f'|...> vis styles')
            onplot.pil_show_nuas(style_imgs, ["[   .   ] style_imgs"])

    if 1: #   mask style images

        b,h,w,c = np.shape(content_img)
        csize = (w,h)

        print(f'|===> nnimg: mask style images \n \
            masks are regions of the content(input) image impacted by style transfer \n \
            content image size: {csize} \n \
        ')

        args.style_mask = 0
        if 'style_mask_imgs_files' in vars(args).keys():
            if args.style_mask_imgs_files:

                for maskfile, thress in zip(args.style_mask_imgs_files, args.style_imgs_thress):
                    maskpath = os.path.join(args.style_imgs_dir, maskfile)

                    if os.path.exists(maskpath):

                        print(f'|... nnimg: mask file exists - assume ok, delete if not \n \
                            maskpath: {maskpath} \n \
                            maskfile: {maskfile} \n \
                            thress: {thress} \n \
                        ')
                        args.style_mask = 1

                    else:
   
                        unmaskfile = onimg.name_to_unmaskname(maskfile)
                        unmaskpath =  os.path.join(args.style_imgs_dir, unmaskfile)
                        print(f'|... nnimg: maskpath {maskpath} does NOT exist \n \
                            generate {maskpath} from {unmaskpath} \n \
                            with thress {thress} \n \
                            the mask has to be created from {unmaskpath} with same size than imput img: {csize} \n \
                        ')

    
                        style_img_to_mask = onfile.path_to_cvi(unmaskpath)
                        style_img_to_mask = cv2.resize(style_img_to_mask, dsize=csize, interpolation=cv2.INTER_AREA)

                        if args.visual > 1:
                            cv2.imshow('style_img_to_mask', style_img_to_mask)             
                            cv2.waitKey(0) & 0xFF is 27
                            cv2.destroyAllWindows()

                        style_mask = onimg.img_to_mask(
                                style_img_to_mask, outpath=maskpath,
                                height=None,width=None,
                                threshold=thress[0], # 77,
                                thresmode=thress[1], #cv2.THRESH_BINARY,
                                visual=1, verbose=1)                        

                        args.style_mask = 1

            args.style_mask = 1
            args.style_mask_imgs = args.style_mask_imgs_files

        if args.style_mask: # _e_
            style_mask_imgs = onfile.names_to_nuas_tf(args.style_mask_imgs_files, args.style_imgs_dir, args,)
            print(f"|===> nnimg: masks \n \
                args.style_mask_imgs_files: {args.style_mask_imgs_files} \n \
                args.style_imgs_dir: {args.style_imgs_dir} \n \
                shapes: {[str(np.shape(style_mask_imgs[i])) for i,img in enumerate(style_mask_imgs)]} \n \
            ")
            if args.visual > 0:
                print(f'|---> vis masks')
                onplot.pil_show_nuas(style_mask_imgs, ["[   .   ] style_mask_imgs"])

    if 0: #   mask content images

        print("|===> mask content images")
        
        name_base = os.path.splitext(args.content_img_file)[0]
        name_ext = os.path.splitext(args.content_img_file)[1]
        print("|...> name_base", name_base, name_ext)

        name_mask = f'{name_base}_mask{name_ext}'
        print("|...> name_mask", name_mask)

        (b,h,w,c) = np.shape(content_img)
        dim = (w, h)
        print("|...> content_size", dim)

        outpath = os.path.join(args.data_dir, name_mask)

        img = onformat.nua_to_pil(content_img)
        cvi = onformat.pil_to_cvi(img)

        content_img_mask = onimg.img_to_mask(
            cvi,outpath=outpath,
            height=h,width=w,
            threshold=38,
            thresmode=cv2.THRESH_BINARY
        )
        cvimasked = onimg.cvi_mask_to_cvi(cvi, content_img_mask, op=4)

        if args.visual > 1:
            cv2.imshow('content img', cvi)                 
            cv2.imshow('content mask', content_img_mask)            
            cv2.imshow('masked img', cvimasked)            
            cv2.waitKey(0) & 0xFF is 27
            cv2.destroyAllWindows()

    if 1: # input image
        print(f'|===> input shape  \n \
            cwd: {os.getcwd()} \n \
            args.video: {args.video} \n \
            args.init_image_type: {args.init_image_type} \n \
        ')          
        input_img = get_input_image( # (1, 512, 512, 3)
            args.init_image_type, 
            content_img, 
            style_imgs,
            init_image,
            args.frame, 
            args.video_input_dir, 
            args.video_output_dir, 
            args.frame_content_frmt,
            args=args
        )

        input_img = onformat.tnua_resize(input_img, args.max_size, args.max_size)
        if 0 and args.visual:
            onplot.pil_show_nua(input_img, "[   .   ] input_img")

    if 1: # model

        b,w,h,c = np.shape(input_img)
        input_shape = (w,h,c)

        print(f'|===> model \n \
            input_shape: {input_shape} \n \
        ')
        model = GAN( input_shape = input_shape, args = args, )

    if 1: # fit

        print(f'|===> fit input image \n \
            cwd: {os.getcwd()} \n \
            input_img shape: {np.shape(input_img)} \n \
            args.style_mask: {args.style_mask} \n \
        ')

        model.fit(
            input_img, content_img, style_imgs, 
            frame= None,
            args = args
        )

    if 0: # mask result
        epoch = 10
        basename = f'output_{int(epoch-1)}.png'
        img_output_path = os.path.join(args.img_output_dir, basename)

        print(f'|---> cv img_output_path img {img_output_path}')
        output = onfile.path_to_cvi(img_output_path)

        content_img_path = os.path.join(args.content_imgs_dir, args.content_img_file)
        content = onfile.path_to_cvi(content_img_path)
        content = onplot.cv_resize(content, dim=args.content_size)

        result1 = onimg.cvi_mask_to_cvi(content, content_img_mask, op=1)
        result2 = onimg.cvi_mask_to_cvi(output, content_img_mask, op=-1)
        #result = cv2.addWeighted(result1,0.0,result2,0.0,0)
        result = cv2.add(result1,result2)

        if args.visual > 1:
            cv2.imshow('result1 img', result1)                 
            cv2.imshow('result2 img', result2)                 
            cv2.imshow('result img', result)                 
            cv2.waitKey(0) & 0xFF is 27
            cv2.destroyAllWindows()
#
#
#   nnani
#
def nnani(args, kwargs):

    if 0:
        args = onutil.pargs(vars(args))
        args.PROJECT = 'newyork'
        args.DATASET = 'stransfer'
        args.content_size = (512, 512)   
        args.style_imgs_files = ['starry-night.jpg']        
        args.video_file = 'Streets of New York City 4K video-vCdBIRtsL6o.f313.mp4'

    if 1:
        args = onutil.pargs(vars(args))
        args.PROJECT = 'portu'
        args.DATASET = 'stransfer'
        args.content_size = (512, 512)   
        args.style_imgs_files = ['kandinsky.jpg']        
        args.video_file = 'portu.mp4'    

    args = onutil.pargs(vars(args))
    xp = getxp(vars(args))
    args = onutil.pargs(xp)    
    onutil.ddict(vars(args), 'args')
    #
    #
    #   config
    #
    if 1:
        args.frame_start = 0
        args.frame_end = -1 # num_frames
        args.video = True 

    if args.verbose > 0: print(f"|===> nnani config \n \
        args.video: {args.video} \n \
        args.visual: {args.visual} \n \
        args.frame_first_iterations: {args.frame_first_iterations} \n \
        args.frame_start: {args.frame_start} \n \
        args.max_iterations: {args.max_iterations} \n \
        args.style_imgs_files = {args.style_imgs_files} \n \
    ")
    #
    #
    #   tree
    #
    if 1:

        extension=os.path.splitext(os.path.basename(args.video_file))[1]

        args.code_dir = os.path.join(args.proto_dir, 'code') # up project dir

        args.video_input_path = os.path.join(args.dataorg_dir, args.video_file)
        args.video_frames_dir=os.path.join(args.proj_dir, 'frames')
        args.video_styled_dir=os.path.join(args.proj_dir, 'outstyled')
        args.video_output_dir=os.path.join(args.proj_dir, 'outvid')
        args.video_input_dir = args.video_frames_dir # frames _e_

        content_filename=os.path.splitext(os.path.basename(args.video_file))[0]
        content_filename=f"{content_filename}" # eg portu

        args.img_output_dir=os.path.join(args.proj_dir, 'outimgs')

        args.data_dir = os.path.join(args.proto_dir, 'data')
        args.content_imgs_dir = args.data_dir
        args.init_img_dir = args.data_dir
        args.style_imgs_dir = args.data_dir

        if args.verbose > 0: print(f'|===> nnani tree \n \
            cwd: {os.getcwd()} \n \
            args.video_file: {args.video_file} \n \
            content_filename: {content_filename} \n \
            args.proto_dir: {args.proto_dir} \n \
            args.code_dir: {args.code_dir} \n \
            args.video_input_path (input video): {args.video_input_path} \n \
            args.style_imgs_dir (style imgs dir): {args.style_imgs_dir} \n \
            args.video_frames_dir (raw frames dir): {args.video_frames_dir} \n \
            args.video_input_dir (raw frames dir): {args.video_input_dir} \n \
            args.video_styled_dir (styled frames dir - per frame): {args.video_styled_dir} \n \
            args.img_output_dir (fit control imgs dir - per epoch): {args.img_output_dir} \n \
            args.video_output_dir (styled vids dir): {args.video_output_dir} \n \
        ')

        os.makedirs(args.results_dir, exist_ok=True)
        os.makedirs(args.video_frames_dir, exist_ok=True)
        os.makedirs(args.video_styled_dir, exist_ok=True)
        os.makedirs(args.video_output_dir, exist_ok=True)
        os.makedirs(args.img_output_dir, exist_ok=True)

        if not args.video_input_path: # try
            urlfolder = 'https://github.com/sifbuilder/eodoes/blob/master/packages/eodoes-eodo-nnstransfer/media/'
            url = f'{urlfolder}{args.video_file}'
            topath = args.video_input_path # os.path.join(args.dataorg_dir, f'{args.video_file}')

            if args.verbose > 0: print(f"|===> nnani: video src does not exist\n \
                urlfolder: {urlfolder} \n \
                url: {url} \n \
                topath: {topath} \n \
            ")
            tofile = tf.keras.utils.get_file(f'{topath}', origin=url, extract=True)
        assert os.path.exists(args.video_input_path), f'input vide {args.video_input_path} does not exist'
    #
    #
    #   git
    #
    if 1:
        onutil.get_git(args.AUTHOR, args.GITPOD, args.code_dir)

    assert os.path.exists(args.code_dir), "code_dir not found"        
    os.chdir(args.code_dir) # _e_ not std
    #
    #
    #   extract vid => frames
    #
    if 1:
        if 1:
            print(f'|===> nnani vid to frames \n \
                args.video_input_path: {args.video_input_path} \n \
                args.video_frames_dir: {args.video_frames_dir} \n \
            ')
            onvid.vid_to_frames(args.video_input_path, args.video_frames_dir, target = 1)
        else:
            frame_pattern = 'frame%04d.ppm'
            cmd = f'ffmpeg -v quiet -i "{args.video_input_path}" "{args.video_frames_dir}/{frame_pattern}"'
            print(f"cmd: {cmd}")
            os.system(cmd)            
    #
    #
    #   frames size
    #
    if 0:
        ppm = os.path.join(args.video_frames_dir, 'frame0000.jpg')
        im = Image.open(ppm)
        width, height = im.size   
        max_size = max(width, height)

        print(f"|===> nnani frame size \n \
            max_size: {max_size} \n \
        ")
    #
    #
    #   video
    #
    if 1:

        num_frames = len([name for name in os.listdir(args.video_frames_dir) if os.path.isfile(os.path.join(args.video_frames_dir, name))])
        args.frame_end = num_frames

        found_q_frames = len([name for name in os.listdir(args.video_styled_dir) if os.path.isfile(os.path.join(args.video_styled_dir, name))])
        frame_start = found_q_frames + 1
    
        frame = args.frame_start
        args.frame_name = args.frame_content_frmt.format(str(frame).zfill(args.zfill))
        args.frame_img_path = os.path.join(args.video_frames_dir, args.frame_name) 

    print(f"|===> nnani video \n \
        cwd: {os.getcwd()} \n \
        num_frames: {num_frames} \n \
        args.video_input_path: {args.video_input_path} \n \
        args.img_output_dir: {args.img_output_dir} \n \
        content_filename: {content_filename} \n \
        extension: {extension} \n \
        args.style_imgs_dir: {args.style_imgs_dir} \n \
        args.style_imgs: {args.style_imgs_files} \n \
        args.video_file: {args.video_file} \n \
        args.video: {args.video} \n \
        args.frame_end: {args.frame_end} \n \
        args.input_shape: {args.input_shape} \n \
        args.frame_content_frmt: {args.frame_content_frmt} \n \
        frame: {frame} \n \
        args.frame_name: {args.frame_name} \n \
        args.frame_img_path: {args.frame_img_path} \n \
    ")
    #
    #
    #   content
    #
    if 1:
        print(f'|===> get content  \n \
            cwd: {os.getcwd()} \n \
            content_frame: {args.frame_img_path} \n \
        ')                
        img = onfile.path_to_pil_cv(args.frame_img_path)
        img = ondata.pil_resize(img, ps=args.content_size)
        img = onformat.pil_to_dnua(img)
        print(f'|===> nnani: content \n \
            args.frame_img_path: {args.frame_img_path} \n \
            args.content_imgs_dir: {args.content_imgs_dir} \n \
            img shape: {np.shape(img)} \n \
        ')
        content_frame = img
        if args.visual > 1:
            onplot.pil_show_nua(content_frame)
    #
    #
    #   style images
    #
    if 1:

        style_img_paths = [os.path.join(args.style_imgs_dir, item) for item in args.style_imgs_files]
        for style_img_path in style_img_paths:
            print(f'|... style_img_path {style_img_path}')

            if not os.path.exists(style_img_path):

                base_path = style_img_path
                base_file = os.path.basename(style_img_path)
                urlfolder = 'https://raw.githubusercontent.com/sifbuilder/eodoes/master/packages/eodoes-eodo-nnstransfer/media/'
                url = f'{urlfolder}{base_file}'

                print(f"|===> nnimg: get style base file \n \
                    urlfolder: {urlfolder} \n \
                    url: {url} \n \
                    base_path: {base_path} \n \
                ")

                tofile = tf.keras.utils.get_file(f'{base_path}', origin=url, extract=True)

            else:
                base_file = os.path.basename(style_img_path)
                print(f"|===> style file {base_file} already in {args.style_imgs_dir}")        

        style_imgs = onfile.names_to_nuas_tf(args.style_imgs_files, args.style_imgs_dir, args,)
        print(f'|===> nnani: styles \n \
            args.style_imgs_files: {args.style_imgs_files} \n \
            args.style_imgs_dir: {args.style_imgs_dir} \n \
            shapes: {[str(np.shape(style_imgs[i])) for i,img in enumerate(style_imgs)]} \n \
        ')
        if 0 and args.visual:
            print(f'|---> vis styles')
            onplot.pil_show_nuas(style_imgs, ['[   .   ] style_imgs'])
    #
    #
    #   input shape
    #
    if 1:
        print(f'|===> input shape  \n \
            cwd: {os.getcwd()} \n \
            args.video: {args.video} \n \
            args.frame_first_type: {args.frame_first_type} \n \
            frame: {frame} \n \
        ')                    
        input_img = get_input_image(
            args.frame_first_type, 
            content_frame, 
            style_imgs, 
            init_img=None, 
            frame=frame,  
            args=args
        )

    print(f'|===> nnani video config \n \
        cwd: {os.getcwd()} \n \
        content_filename: {content_filename} \n \
        args.frame_content_frmt: {args.frame_content_frmt} \n \
        content_frame shape: {np.shape(content_frame)} \n \
        args.style_imgs: ({len(args.style_imgs_files)}) {args.style_imgs_files} \n \
        style_imgs shapes: {[np.shape(img) for img in style_imgs]} \n \
        args.video_file: {args.video_file} \n \
        args.video_output_dir: {args.video_output_dir} \n \
        args.video_frames_dir: {args.video_frames_dir} \n \
        args.frame_start/frame_end: ({num_frames}) {args.frame_start} : {args.frame_end} \n \
        args.input_img shape: {np.shape(input_img)} \n \
    ')
    #
    #
    #   model
    #
    if 1:

        b,w,h,c = np.shape(input_img)
        input_shape = (w,h,c)   

        print(f"|===> nnani model \n \
            cwd: {os.getcwd()} \n \
            input_shape: {input_shape} \n \
            args.video_input_path: {args.video_input_path} \n \
            args.img_output_dir: {args.img_output_dir} \n \
        ")
        model = GAN(input_shape = input_shape, args = args,)
    #
    #
    #   train
    #
    if args.TRAIN > 0:

        print(f'|===> nnani fit \n \
            cwd: {os.getcwd()} \n \
            args.video_styled_dir: {args.video_styled_dir} \n \
            args.img_output_dir: {args.img_output_dir} \n \
        ')

        args.max_iterations = args.frame_iterations
        for frame in range(args.frame_start, args.frame_end+1):
            print(f'|...> RENDERING VIDEO FRAME ({args.frame_first_type}): {frame}/{args.frame_end} ----\n')

            if frame == args.frame_start:
                print(f"|...> frame_start input_img type: {type(input_img)}")
                input_img = get_input_image(
                    init_type=args.frame_first_type, 
                    content_img=content_frame, 
                    style_imgs=style_imgs, 
                    init_img=None, 
                    frame=frame,  
                    args=args
                )
            else:
                print(f"|...> other_frame input_img type: {type(input_img)}")
                input_img = get_input_image(  # (1, 1080, 1440, 3) <class 'numpy.ndarray'>
                    init_type=args.frame_init_type, #
                    content_img=content_frame, 
                    style_imgs=style_imgs, 
                    init_img=None, 
                    frame=frame,  
                    args=args
                )

            input_img = onimg.tf_resize_nua(input_img, args=args)

            print(f'|...> fit input image \n \
                cwd: {os.getcwd()} \n \
                input_img shape: {np.shape(input_img)} \n \
            ')

            model.fit( 
                input_img, content_frame, style_imgs,
                frame = frame,                
                args = args,
            )
    #
    #
    #   render stylized video
    #
    if 1:
        
        os.chdir(args.code_dir) # _e_ not std

        print(f'|===> nnani gen gif stylized video \n \
            cwd: {os.getcwd()} \n \
            video: {args.video} \n \
        ')

        cmd = f'|...> cmd: python neural_style.py --video \
        --video_input_dir "{args.video_input_dir}" \
        --style_imgs_dir "{args.style_imgs_dir}" \
        --style_imgs {args.style_imgs_files[0]} \
        --frame_end {args.frame_end} \
        --max_size {args.max_size} \
        --verbose'
        print(cmd)
        os.system(cmd)
    #
    #
    #   gen gif video
    #
    if 1:

        fps = 6
        maxTime = 9 # seconds
        frameCount = 0
        time = 0
        nframes = int( maxTime*fps )

        qsegs = 7
        qcells = qsegs * qsegs
        fps=10

        gif_output_path = os.path.join(args.video_output_dir, 'v.gif')

        outvidframes_dir = args.img_output_dir # args.video_frames_dir

        print(f'|===> nnani gen gif stylized video \n \
            from args.video_styled_dir: {args.video_styled_dir} \n \
            to video_output_path: {gif_output_path} \n \
        ')

        onvid.folder_to_gif(args.video_styled_dir, gif_output_path)
    #
    #
    #   gen mp4 video
    #           
    if 1:

        fps = 6
        maxTime = 9 # seconds
        frameCount = 0
        time = 0
        nframes = int( maxTime*fps )

        qsegs = 7
        qcells = qsegs * qsegs
        fps=10

        video_output_path = os.path.join(args.video_output_dir, 'v.mp4')

        outvidframes_dir = args.img_output_dir # args.video_frames_dir

        print(f'|===> nnani gen mp4 stylized video \n \
            from args.video_frames_dir: {args.video_styled_dir} \n \
            to video_output_path: {video_output_path} \n \
        ')

        onvid.frames_to_video(
            args.video_styled_dir, 
            video_output_path, 
            fps
        )
#
#
#   nninfo
#   
def nninfo(args, kwargs):

    args = onutil.pargs(vars(args))
    onutil.ddict(vars(args), 'args')

    print(f'|---> stransfer:  \n \
    ref: https://github.com/ProGamerGov/neural-style-pt \n \
    nnimg: style transform of img \n \
    nnani: style transform of vid \n \
    ')        
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
