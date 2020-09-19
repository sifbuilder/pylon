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
onvgg = Onvgg()
onlllyas = Onlllyas()
#
#
#   CONTEXT
#
#
def getap():
    cp = {
        "primecmd": 'nnmodel',

        "MNAME": "microsoft",
        "AUTHOR": "microsoft",
        "PROJECT": "msconfig",
        "GITPOD": "ConfigNet",
        "DATASET": "confignet",

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
#
#
#   FUNS
#
#
#
#
#   CMDS
#
#
#
#   nnmodel
#
def nnmodel(args, kwargs):

    args = onutil.pargs(vars(args))
    xp = getxp(vars(args))
    args = onutil.pargs(xp)    
    onutil.ddict(vars(args), 'args')

    print(f'|---> nnmodel: {args.PROJECT}:   \n \
        args.DATASET:    {args.DATASET} \n \
    ')

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

    if 0: # conda
        onutil.conda()
 
    if 1: # git
        onutil.get_git(args.AUTHOR, args.GITPOD, args.proj_dir)

    if 1: # tree

        args.models_dir = os.path.join(args.proj_dir, "models")

        os.makedirs(args.data_dir, exist_ok=True)
        os.makedirs(args.models_dir, exist_ok=True)

    if args.verbose: print(f"|===> tree: {args.PROJECT}:   \n \
        cwd:     				{os.getcwd()} \n \
        args.dataorg_dir: {args.dataorg_dir} \n \
        args.data_dir: {args.data_dir} \n \
        args.models_dir: {args.models_dir} \n \
        args.dataorg_dir:     {args.dataorg_dir} \n \
        args.ckpt_dir:  {args.ckpt_dir} \n \
        args.logs_dir:        {args.logs_dir} \n \
        args.tmp_dir:        {args.tmp_dir} \n \
        args.verbose:         {args.verbose}, \n \
        args.visual:          {args.visual}, \n \
    ")
    #
    #from confignet import ConfigNet, LatentGAN, FaceImageNormalizer
    #def process_images(image_path: str, resolution: int) -> List[np.ndarray]:
    #    '''Load the input images and normalize them'''
    #    if os.path.isfile(image_path):
    #        img = cv2.imread(image_path)
    #        img = FaceImageNormalizer.normalize_individual_image(img, (resolution, resolution))
    #        return [img]
    #    elif os.path.isdir(image_path):
    #        FaceImageNormalizer.normalize_dataset_dir(image_path, pre_normalize=True,
    #                                                output_image_shape=(resolution, resolution), write_done_file=False)
    #        normalized_image_dir = os.path.join(image_path, "normalized")
    #        image_paths = glob.glob(os.path.join(normalized_image_dir, "*.png"))
    #        max_images = 200
    #        image_paths = image_paths[:max_images]
    #        if len(image_paths) == 0:
    #            raise ValueError("No images in input directory")
    #        imgs = []
    #        for path in image_paths:
    #            imgs.append(cv2.imread(path))
    #        return imgs
    #    else:
    #        raise ValueError("Image path is neither directory nor file")
    #
    if 1: # models.zip
        os.chdir(args.proj_dir)
        if args.verbose: print(f"|===> models :   \n \
            cwd:     				{os.getcwd()} \n \
        ")
        import urllib.request
        local_tar_path = ''
        tree_root = None
        local_tar_path = os.path.join(args.proj_dir, 'models.zip')
        url = 'https://github.com/microsoft/ConfigNet/releases/download/v1.0.0/models.zip'
        if not os.path.exists(local_tar_path):
            print(f'|... urlretrieve: {url} to {local_tar_path}')
            filename, headers = urllib.request.urlretrieve(url, filename=local_tar_path)

        if os.listdir(args.models_dir) == 0:
            print(f'|... tunzip: {local_tar_path} to {args.proj_dir}') # tree_root: models 
            import zipfile
            with zipfile.ZipFile(local_tar_path, 'r') as zip_ref:
                zip_ref.extractall(args.proj_dir)

    # openface - https://github.com/microsoft/ConfigNet/tree/main/setup/download_deps.py
    setup_dir = os.path.join(args.proj_dir, 'setup')
    os.chdir(setup_dir)
    if args.verbose: print(f"|===> setup :   \n \
        cwd:     				{os.getcwd()} \n \
    ")

    # openface - https://github.com/microsoft/ConfigNet/tree/main/setup/download_models.py
    import subprocess
    OPENFACE = "https://github.com/TadasBaltrusaitis/OpenFace/releases/download/OpenFace_2.2.0/OpenFace_2.2.0_win_x64.zip"
    #include_dir_path = os.path.abspath(os.path.dirname(__file__))
    #cwd = os.getcwd()
    #os.chdir(include_dir_path)

    third_party_dir = os.path.join("..", "3rd_party")
    openface_filename = os.path.basename(OPENFACE)
    openface_dirname = os.path.splitext(openface_filename)[0]
    openface_local_path = os.path.join(third_party_dir, openface_dirname)

    if args.verbose: print(f"|---> openface:  \n \
        cwd:                 {os.getcwd()} \n \
        openface_dirname:    {openface_dirname} \n \
        openface_filename:   {openface_filename} \n \
        openface_local_path: {openface_local_path} \n \
    ")

    if not os.path.exists(openface_local_path):
        print('|... Downloading OpenFace')
        response = requests.get(OPENFACE)
        with open(openface_filename, "wb") as fp:
            fp.write(response.content)

        print("|... Extracting OpenFace")
        os.makedirs(third_party_dir, exist_ok=True)
        with zipfile.ZipFile(openface_filename, "r") as zip_ref:
            zip_ref.extractall(third_party_dir)

        print("|... Downloading patch experts")
        os.chdir(openface_local_path)
        subprocess.run([r'powershell.exe', '-ExecutionPolicy', 'Unrestricted', './download_models.ps1'], cwd=os.getcwd())
    # 
    os.chdir(args.proj_dir)   
    imgs_dir = os.path.join(args.proj_dir, 'assets')
    imgs_dir = os.path.join(args.gdata, 'threefaces', 'stock_photo1.jpg')
    cmd = f'python evaluation/confignet_demo.py --image_path "{imgs_dir}"'

    setup_dir = os.path.join(args.proj_dir, '')
    os.chdir(setup_dir)
    if args.verbose: print(f"|---> confignet_demo :   \n \
        cwd:  {os.getcwd()} \n \
        cmd:  {cmd} \n \
    ")
    os.system(cmd)
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
