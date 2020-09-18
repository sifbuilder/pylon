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
onimg = Onimg()
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
        "primecmd": 'nnjoin',

        "MNAME": "pylons",
        "AUTHOR": "sifbuilder",
        "PROJECT": "pylons",
        "GITOPOD": "pylons",
        "DATASET": "",     
    
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

        "video": 0,


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
#   FUNS 
#
#   ***************

def read_file(path):
    """Read contents of file at given path as bytes."""
    with open(path, 'rb') as f:
        return f.read()


def write_file(path, data):
    """Write data bytes to file at given path."""
    with open(path, 'wb') as f:
        f.write(data)


def folder_to_paths(folder, format=None):
    paths=[]
    if not format:
        format="*"
    paths = glob.glob(os.path.join(folder, format))
    return paths

def path_to_filename(path):
    name, _ = os.path.splitext(os.path.basename(path))
    return name

def name_to_basename(name):
    base, _ = os.path.splitext(name)
    return base

def name_to_extension(name):
    base, ext = os.path.splitext(name)
    return ext

def path_to_basename(path):
    base, _ = os.path.splitext(path_to_filename(path))
    return base

def path_to_extension(path):
    ext = os.path.splitext(path)[-1]
    return ext


#   ******************
#   CMDS
#
#   ****************

#   *******************
#   nnimg
#
def nnjoin(args, kwargs):

    args = onutil.pargs(vars(args))
    onutil.ddict(vars(args), 'args')

    if 1: # tree

        args.pyons_dir = args.local_prefix
        os.makedirs(args.pyons_dir, exist_ok=True)

        print(f"|===> pylons:{args.PROJECT}.nnjoin: create pylons and commit to git \n \
            cwd: {os.getcwd()} \n \
            args.pyons_dir: {args.pyons_dir} \n \
        ")


    paths = set(folder_to_paths(args.pyons_dir, '*.py'))
    paths = paths - set(folder_to_paths(args.pyons_dir, 'base*.py'))
    paths = paths - set(folder_to_paths(args.pyons_dir, 'pylon*.py'))
    paths = paths - set(folder_to_paths(args.pyons_dir, '_*.py'))

    base_path = os.path.join(args.pyons_dir, 'base.py')
    # base_data = read_file(base_path)

    print(f"|===> pylons: \n \
        base_path: {base_path} \n \
        paths: {paths} \n \
    ")

    import p2j
    #import nbformat
    #from nbformat.v4 import new_code_cell,new_notebook

    for path in paths:
        # file_data = read_file(path)
        # pyon_data = base_data + '\n\n' + file_data

        basename = path_to_basename(path)

        tmpextension = 'py' # path_to_extension(path)
        newextension = 'ipynb' # path_to_extension(path)

        tmpfilename = f'___{basename}.{tmpextension}' # _ prefix
        tmpfile_path = os.path.join(args.pyons_dir, tmpfilename)

        interfilename = f'___{basename}.{newextension}' # _ prefix
        interfile_path = os.path.join(args.pyons_dir, interfilename)

        newfilename = f'{basename}.{newextension}' # _ prefix
        newfile_path = os.path.join(args.pyons_dir, newfilename)
     
        print(f"|===> pylons: {path}\n \
            to path: {newfile_path} \n \
        ")        

        alllines = ''


        if 0: # do not add base
            with open(base_path) as infile:
                for line in infile:
                    alllines += line

        if 1: # add py script
            with open(path) as infile:
                for line in infile:
                    alllines += line

        prime = ''
        primeline = re.search('(["\']primecmd["\']: ["\'])([a-zA-Z_0-9]*)(["\'])', alllines)
        if primeline:
            print(f'|... primeline found: ex: "primecmd": "nntrain", ')
            print(f'|... prime is 2nd in primeline')
            prime = primeline[2]
            print(f'|... prime: {prime}')

            nndefs = re.findall('def (nn[a-zA-Z_0-9]*)', alllines, re.DOTALL)
            if nndefs:
                print(f'|... nndefs found: {nndefs}')

                for nndef in nndefs:
                    newfilename = f'{basename}-{nndef}.{newextension}'
                    newfile_path = os.path.join(args.pyons_dir, newfilename)

                    print(f"|... REPLACE {prime} with {nndef} into {newfile_path}")
                    alllines = re.sub('(["\']primecmd["\']: ["\'])([a-zA-Z_0-9]*)(["\'])', f'\\1{nndef}\\3', alllines)

                    newprimeline = re.search('(["\']primecmd["\']: ["\'])([a-zA-Z_0-9]*)(["\'])', alllines)
                    newprime = newprimeline[2]

                    print(f'|... newprime is: {newprime} ')

                    with open(tmpfile_path, 'w') as outfile:
                        outfile.write(alllines)

                    cmd = f'p2j {tmpfile_path}' # tmpfile_path => interfile_path
                    print(f"|---> convert to ipynb")
                    print(f"|---> cmd {cmd}")
                    os.system(cmd)

                    if os.path.exists(tmpfile_path):
                        os.remove(tmpfile_path)
                    if os.path.exists(newfile_path):
                        os.remove(newfile_path)
                    os.rename(interfile_path, newfile_path) # interfile_path => newfile_path

        else:
            with open(tmpfile_path, 'w') as outfile:
                outfile.write(alllines)

            cmd = f'p2j {tmpfile_path}' # tmpfile_path => interfile_path
            print(f"|---> convert to ipynb")
            print(f"|---> cmd {cmd}")
            os.system(cmd)

            if os.path.exists(tmpfile_path):
                os.remove(tmpfile_path)
            if os.path.exists(newfile_path):
                os.remove(newfile_path)
            os.rename(interfile_path, newfile_path) # interfile_path => newfile_path


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
