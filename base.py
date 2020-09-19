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
import re
import uuid
import hashlib
import tempfile
from typing import Any, List, Tuple, Union
import types

from functools import partial
import importlib
from importlib import import_module

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import numpy as np

import math
from math import floor, log2
from random import random

# import IPython.display as display
from IPython.core.display import display

import PIL
from PIL import Image
PIL.Image.MAX_IMAGE_PIXELS = 933120000

from mpl_toolkits.axes_grid1 import ImageGrid

import shutil

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

import gdown

#  https://github.com/lllyasviel/DanbooRegion
from numba import njit
from scipy.ndimage import label


import sys
sys.path.append('../')  # if called from eon, modules are in parallel folder
sys.path.append('./')  #  if called from dnns, modules are in folder

try:
    TF1
    print("set tf 1.x env")
    # %tensorflow_version 1.x
except:
    pass

# pip install -q -U tensorboard

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
from tensorflow.python.client import device_lib # pylint: disable=no-name-in-module

# python -c "import tensorflow as tf; print(tf.version.GIT_VERSION, tf.version.VERSION)"

# GLOBALS 

local_prefix = os.path.abspath('')
try:
    local_prefix = os.path.dirname(os.path.realpath(__file__)) # script dir
except:
    pass

# cuda_cache_path = os.path.join(os.path.dirname(__file__), '_cudacache')
cuda_cache_path = os.path.join(os.path.dirname(local_prefix), '_cudacache')
cuda_cache_version_tag = 'v1' # _e_
do_not_hash_included_headers = True # _e_ # Speed up compilation by assuming that headers included by the CUDA code never change. Unsafe!

compiler_bindir_search_path = [
    'C:/Program Files (x86)/Microsoft Visual Studio/2017/Community/VC/Tools/MSVC/14.14.26428/bin/Hostx64/x64',
    'C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Tools/MSVC/14.25.28610/bin/Hostx64/x64',
    'C:/Program Files (x86)/Microsoft Visual Studio 14.0/vc/bin',

    'C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Tools/MSVC/14.26.28801/bin/Hostx64/x64'
]

_plugin_cache = dict()

class Onpyon:

    @staticmethod
    def ver():
        return '0.0.0'

class Ontree:

    @staticmethod
    def tree(cp):

        # dataprefix
        if os.path.exists('/content/drive/My Drive'):  # collab with drive
            dataprefix = '/content/drive/My Drive'
        elif os.path.exists('/content'): # if /content exists, is collab
            dataprefix = '/content' 
        elif not cp["LOCALDATA"] and os.path.exists(cp['net_prefix']): # network
            dataprefix = cp['net_prefix'] 
        else:                               # local
            dataprefix = os.path.join(cp["local_prefix"], cp["grel_infix"], 'content') # dnns/../content/gdata/

        # modelsprefix
        if os.path.exists('/content/drive/My Drive'):  # collab with drive
            modelsprefix = '/content/drive/My Drive'
        elif os.path.exists('/content'): # if /content exists, is collab
            modelsprefix = '/content' 
        elif cp["LOCALMODELS"]:                               # local
            modelsprefix = os.path.join(cp["local_prefix"], cp["grel_infix"], 'content') # dnns/../content/gmodel/
        elif os.path.exists(cp['net_prefix']): # try net drive
                modelsprefix = cp['net_prefix'] 
        else:                               # local
            print(f'drive could not be found !!!!!!!!!!!!!! ')


        # labprefix
        if os.path.exists('/content/drive/My Drive'):  # collab with drive
            labprefix = '/content/drive/My Drive'
        elif os.path.exists('/content'): # if /content exists, is collab
            labprefix = '/content' 
        elif cp["LOCALLAB"]:                               # local
            labprefix = os.path.join(cp["local_prefix"], cp["grel_infix"], 'content') # dnns/../content/glab/
        elif os.path.exists(cp['net_prefix']): # try net drive
                labprefix = cp['net_prefix'] 
        else:                               # local
            print(f'drive could not be found !!!!!!!!!!!!!! ')

        gdata = dataprefix + '/gdata/'
        gmodel = modelsprefix + '/gmodel/'
        glab = labprefix + '/glab/'
        gadir = os.getcwd()
        gedir = os.path.join(cp["local_prefix"])


        Onutil.ddict(cp, 'cp')
        print(labprefix, glab)

        assert os.path.exists(gdata), f"gdata {gdata} does not exist"
        assert os.path.exists(gmodel), f"gmodel {gmodel} does not exist"
        assert os.path.exists(glab), f"glab {glab} does not exist"
        assert os.path.exists(gadir), f"gadir {gadir} does not exist"
        assert os.path.exists(gedir), f"gedir {gedir} does not exist"

        lab=os.path.normpath(os.path.join(glab, cp["MNAME"]))
        os.makedirs(lab, exist_ok=True)

        proto_dir=os.path.normpath(os.path.join(glab,cp["MNAME"]))
        os.makedirs(proto_dir, exist_ok=True)

        proj_dir = os.path.join(proto_dir, cp["PROJECT"])

        if cp["RESETCODE"] and os.path.exists(proj_dir):
            print("will remove tree %s" %proj_dir)
            try:
                shutil.rmtree(proj_dir)
            except:
                print("could not remove git tree")
                pass

        os.makedirs(proj_dir, exist_ok=True)
        os.chdir(proj_dir) # %cd $proj_dir
        cwd = os.getcwd()
        proj_dir = cwd


        tp = {

            "LOCAL": cp["local_prefix"], # ''
            "FROMPATH": cp["local_prefix"],
                    
            "gdata": gdata,
            "gmodel": gmodel,
            "glab": glab,
            "gadir": gadir,
            "gedir": gedir,
            "lab": lab,
            "proto_dir": proto_dir,
            "proj_dir": proj_dir,
            "cwd": cwd,

            "ani_dir": os.path.join(proj_dir, 'ani'),           # ani results gifs dir (.gif, .mp4) 
            "results_dir": os.path.join(proj_dir, "Results"),   # Results

            "tmp_dir": os.path.join(proj_dir, "tmp"),           # tmp
            "logs_dir": os.path.join(proj_dir, 'logs'),         # logs
            "trace_dir": os.path.join(proj_dir, 'trace'),       # trace

            "data_dir": os.path.join(proj_dir, "data"),         # data
            "train_dir": os.path.join(proj_dir, "data", "train"),  # train
            "test_dir": os.path.join(proj_dir, "data", "test"), # test
            
            "dataset_dir": os.path.join(proj_dir, "dataset"),   # dataset
            "records_dir": os.path.join(proj_dir, "records"),   # records

            "ckpt_dir": os.path.join(proj_dir, 'ckpt'),  # ckpt checkpoints place
            "weights_dir": os.path.join(proj_dir, 'weights'),   # weights dir (h5) 
            "models_dir": os.path.join(proj_dir, "Models"),     # Models

            "dataorg_dir": gdata + '/' + cp["DATASET"],   # dataorg

        }

        return tp

#   ******************
#   FUNS UTILS
#
#   ******************
class Onutil:

    @staticmethod
    def info():
        print("Onutil")

    @staticmethod
    def error(msg):
        print('Error: ' + msg)
        exit(1)

    @staticmethod
    def incolab():
        res = False
        if os.path.exists('/content'):
            res = True
        return res

    @staticmethod
    def conda():

        ## StyleGAN:
        cmd = f'pip install pillow numpy moviepy scipy opencv-python lmdb matplotlib'
        os.system(cmd)

        cmd = f'pip install cmake'
        os.system(cmd)

        cmd = f'pip install dlib'
        os.system(cmd)

        cmd = f'pip install ipython gdown'
        os.system(cmd)

        # ## lllyasviel
        cmd = f'pip install numba scikit-image scikit-learn'
        os.system(cmd)

        # ## 
        cmd = f'pip install cython h5py Pillow'
        os.system(cmd)

        cmd = f'pip install -U gradient'
        os.system(cmd)

        cmd = f'pip install windows-curses'
        os.system(cmd)

        # ## confignet https://github.com/microsoft/ConfigNet/blob/main/setup/requirements.txt
        #scipy==1.4.1
        #scikit-learn==0.20.0
        #tensorflow-gpu==2.1.0
        cmd = f'pip install azureml-sdk matplotlib numpy opencv-python pytest transformations'
        os.system(cmd)

    @staticmethod
    def pargs(cp):
        xp={}        
        for key in cp.keys():
            xp[key] = cp[key]

        ## ------------ _e_tbc ref stransfer.py
        tree = Ontree.tree(cp)
        for key in tree.keys():
            xp[key] = tree[key]


        parser = argparse.ArgumentParser(description='Run "python %(prog)s <subcommand> --help" for subcommand help.')
        for p in xp:
            cls = type(xp[p])
            parser.add_argument('--'+p, type=cls, default=xp[p])
        args = parser.parse_args('')
        return args

    @staticmethod
    def dodrive(domount=True):
        if domount:
            if os.path.exists('/content/'):
                if not os.path.exists('/content/drive'):
                    try:
                        from google.colab import drive # drive from colab
                        drive.mount('/content/drive', force_remount=True)
                    except:
                        print("|===> unable to mount drive")

    @staticmethod
    def check_file(file, path):
        if file is None:
            raise OSError(errno.ENOENT, "No such file", path)

    @staticmethod
    def get_git(AUTHOR, PROJECT, proj_dir='./'):
        try:
            # os.system("git clone https://github.com/rolux/stylegan2encoder %s" %proj_dir)
            cmd = f'git clone https://github.com/{AUTHOR}/{PROJECT}.git "{proj_dir}"'
            print("|... cmd %s" %cmd)
            os.system(cmd)
        except:
            cmd = f'git pull https://github.com/{AUTHOR}/{PROJECT}.git "{proj_dir}"'
            print("|... cmd %s" %cmd)
            os.system(cmd)

    @staticmethod
    def ddict(item, txt=":"):
        print(f"|===> {txt}")
        for i in item:
            print (f"  {i} => {item[i]}")

    @staticmethod
    def ditem(item, txt=":", val=True):
        print(txt)
        print(type(item))
        # print(np.shape(item))
        if val:
            print(item)

    @staticmethod
    def isempty(folder):
        empty = True
        if [f for f in os.listdir(folder) if not f.startswith('.')] == []:
            empty = False
        return empty

    @staticmethod
    def walkfolder(folder):
        if os.path.exists(folder):        
            for root, dirs, files in os.walk(folder, topdown=False):
                for name in files:
                    print(os.path.join(root, name))
                for name in dirs:
                    print(os.path.join(root, name))


    @staticmethod
    def nameint(i,zfill=4):
        return str(i).zfill(4)

    @staticmethod
    def pathsort(paths):
        roots = [Onutil.get_rootid(path) for path in paths]
        if all(root.isdigit() for root in roots):
            paths = sorted(paths, key=lambda path: int(Onutil.get_rootid(path)))
            print(f'|---> pathsort numeric q: {len(paths)}')
        else:
            paths = sorted(paths)
            print(f'|---> pathsort alphabetic q: {len(paths)}')
        return paths


    @staticmethod
    def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 50, fill = '█'):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '.' * (length - filledLength)
        print('\r %s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
        # Print New Line on Complete
        if iteration == total:
            print()
            print()

    @staticmethod
    def gunzip(file_basename, file_id, dst, results_dir):
        if not os.path.exists(dst):
            print(f'|... download {file_basename} from google drive to {dst}')
            gdown.download(file_id, dst)
        else:
            print(f'|... {file_basename} already in content {dst}')

        path_to_zip_file = dst
        directory_to_extract_to = results_dir
        print("will unzip %s to %s" %(path_to_zip_file, directory_to_extract_to))
        with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
            zip_ref.extractall(directory_to_extract_to)

    @staticmethod
    def gdownid(urlfolder, gfolderid, dst):
        import gdown
        print(f'|... gdownid {urlfolder} / {gfolderid} to {dst}')
        #urlfolder = 'https://drive.google.com/drive/folders/'
        #gfolderid = '1ihLt6P7UQRlaFtZUEclXkWC9grmEXEUK'
        #dst = 'DanbooRegion2020.zip.xxx' # 001 - 077
        url = f'{urlfolder}{gfolderid}'
        gdown.download(url, dst)		

    @staticmethod
    def gdownfile(link_name, dest):
        from google_drive_downloader import GoogleDriveDownloader as gdd
        print(f'|... gdownfile {link_name} to {dest}')		
        gdd.download_file_from_google_drive(file_id=link_name,dest_path=dest,unzip=True)

    @staticmethod
    def netget(urlpath, localpath, args=None):
        basedir = os.path.dirname(urlpath)
        baseurl = os.path.file(urlpath)
        localdir = os.path.dirname(localpath)
        localfile = os.path.file(localpath)
        tofile = tf.keras.utils.get_file(f'{localpath}', origin=urlpath, extract=True)

    @staticmethod
    def tenzip(output_filename, source_dir, arcname=None):
        import tarfile
        with tarfile.open(output_filename, "w:gz") as tar:
            tar.add(source_dir, arcname=arcname)

    @staticmethod
    def tunzip(url, tarpath, results_dir, tree_root=None):
        # tar xvzf car_devkit.tgz
        import tarfile
        #print(f'|---> tunzip {tarpath} to {results_dir}')
        #if not os.path.exists(tarpath): 
        #	cmd = f'wget -O "{tarpath}" "{url}"'
        #	print(f"cmd: {cmd}")
        #	os.system(cmd)
        #else:
        #	print(f"{tarpath} already exists")

        target_dir = None
        if tree_root:
            target_dir = os.path.join(results_dir, tree_root)

        if target_dir and os.path.exists(target_dir): 
            print(f'do nothing. target dir {target_dir} already exists')
        else:
            if tarpath.endswith("tar.gz"):
                print(f'|... tar.gz {tarpath} is tarfile {tarfile.is_tarfile(tarpath)}')
                tar = tarfile.open(tarpath, "r:")
                tar.extractall(results_dir)
                tar.close()
            elif tarpath.endswith("tar"):
                print(f'|... tar {tarpath} is tarfile {tarfile.is_tarfile(tarpath)}')
                tar = tarfile.open(tarpath, "r:")
                tar.extractall()
                tar.close()
            elif tarpath.endswith("tgz"):
                print(f'|... tgz {tarpath} is tarfile {tarfile.is_tarfile(tarpath)}')                
                tar = tarfile.open(tarpath, "r:gz")
                tar.extractall()
                tar.close()
            elif tarpath.endswith("zip"):
                print(f'|... tar {tarpath} is zip {tarfile.is_tarfile(tarpath)}')
                tar = tarfile.open(tarpath, "r:")
                tar.extractall()
                tar.close()                
            else:
                print(f'|... unsupported tarfile {tarpath}')


    @staticmethod
    def zipdir(folder, dstzip='Python.zip'):
        import zipfile

        zipf = zipfile.ZipFile(dstzip, 'w', zipfile.ZIP_DEFLATED)		
        # zipf is zipfile handle
        for root, dirs, files in os.walk(folder):
            for file in files:
                zipf.write(os.path.join(root, file))

        zipf.close()

    def mnormal(mu=0.0, sigma=1.0, size=None):
        return np.random.normal(loc = mu, scale = sigma, size = size).astype('float32')

    @staticmethod
    def muniform(low=0.0, high=1.0, size=None):
        return np.random.uniform(low = low, high = high, size = size).astype('float32')

    @staticmethod
    def mrandom(size=None):
        return np.random.random(size = size).astype('float32')

    @staticmethod
    def get_filename(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name

    @staticmethod
    def get_rootname(name):
        root, _ = os.path.splitext(name)
        return root

    @staticmethod
    def get_rootid(path):
        return Onutil.get_rootname(Onutil.get_filename(path))


# https://github.com/rolux/stylegan2encoder/dnnlib/util.py
class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

# https://github.com/rolux/stylegan2encoder/dnnlib/util.py
class Logger(object):
    """Redirect stderr to stdout, optionally print stdout to a file, and optionally force flushing on both stdout and the file."""

    def __init__(self, file_name: str = None, file_mode: str = "w", should_flush: bool = True):
        self.file = None

        if file_name is not None:
            self.file = open(file_name, file_mode)

        self.should_flush = should_flush
        self.stdout = sys.stdout
        self.stderr = sys.stderr

        sys.stdout = self
        sys.stderr = self

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def write(self, text: str) -> None:
        """Write text to stdout (and a file) and optionally flush."""
        if len(text) == 0: # workaround for a bug in VSCode debugger: sys.stdout.write(''); sys.stdout.flush() => crash
            return

        if self.file is not None:
            self.file.write(text)

        self.stdout.write(text)

        if self.should_flush:
            self.flush()

    def flush(self) -> None:
        """Flush written text to both stdout and a file, if open."""
        if self.file is not None:
            self.file.flush()

        self.stdout.flush()

    def close(self) -> None:
        """Flush, close possible files, and remove stdout/stderr mirroring."""
        self.flush()

        # if using multiple loggers, prevent closing in wrong order
        if sys.stdout is self:
            sys.stdout = self.stdout
        if sys.stderr is self:
            sys.stderr = self.stderr

        if self.file is not None:
            self.file.close()


#   ******************
#   FUNS PLOT
#
#   ******************
class Onplot:

    @staticmethod
    def info():
        print("Plot")

    @staticmethod   
    def display_pil(img):
        display.display(PIL.Image.fromarray(np.array(img)))

    @staticmethod
    def pil_show_pil(img, title=""):
        display(img) if Onutil.incolab() else img.show(title=title)
        return img
    
    @staticmethod    
    def pil_show_nua(img, title=""):
        img = np.asarray(img)
        if np.ndim(img)>3:
            assert img.shape[0] == 1
            rgb = img[0]        
        #img = np.clip(img, 0, 1).astype('float32')        
        img = Onformat.nua_to_pil(img)
        Onplot.pil_show_pil(img, title)

    @staticmethod
    def pil_show_nuas(nuas, scale=1, rows=1):
        rgbs=Onformat.nuas_to_rgbs(nuas)
        Onplot.pil_show_rgbs(rgbs, scale=1, rows=1)

    @staticmethod   
    def pil_show_nba(img, title=""):
        img = np.asarray(img)        
        img = Onformat.nba_to_nua(img)
        img = Onformat.nua_to_rgb(img)
        img = PIL.Image.fromarray(img)
        Onplot.pil_show_pil(img, title)
     
    @staticmethod   
    def pil_show_nbas(nbas, title=[]):
        for nba in nbas:
            Onplot.pil_show_nba(nba)

    @staticmethod    
    def pil_show_rgb(img, title=""):
        img = np.asarray(img)  
        if len(img.shape) > 3: img = tf.squeeze(img, axis=0)  # img = img[0,...]            
        img = Onformat.rgb_to_pil(img)
        Onplot.pil_show_pil(img, title)

    @staticmethod
    def pil_show_rgbs(rgbs, scale=1, rows=1):
        pils = []
        for img in rgbs:
            if len(img.shape) > 3: img = tf.squeeze(img, axis=0)  # img = img[0,...]
            pils.append(PIL.Image.fromarray(np.array(img, dtype=np.uint8)))
        w,h = pils[0].size
        w = int(w*scale)
        h = int(h*scale)
        height = rows*h
        cols = int(math.ceil(len(pils) / rows))
        width = cols*w
        pil = PIL.Image.new('RGBA', (width,height), 'white')
        for i,img in enumerate(pils):
            img = img.resize((w,h), PIL.Image.ANTIALIAS)
            pil.paste(img, (w*(i % cols), h*(i // cols))) 
        Onplot.pil_show_pil(pil)

    @staticmethod
    def pil_show_nba(img, title="img"):
        if len(img.shape) > 3:
            assert img.shape[0] == 1
            img = tf.squeeze(img, axis=0)  # img = img[0,...] # bnbt => nba    
        img = (img + 1.0)/2.0
        img = np.array(255 * img, dtype=np.uint8)
        img = PIL.Image.fromarray(img)
        img if Onutil.incolab() else img.show(title=title)
        return img

    @staticmethod
    def plot_pils_grid(imgs, rows=2, cols=2, figsize=(4, 4)):
        qtiles = cols * rows
        plt.figure(figsize=figsize)
        for i in range(qtiles):
            plt.subplot(rows, cols, i+1)
            plt.imshow(np.array(imgs[i])/255.0)
            plt.axis('off')
        plt.show()
 

    #https://stackoverflow.com/questions/53255432/saving-a-grid-of-heterogenous-images-in-python
    @staticmethod
    def plot_save_grid(ims, path=None, rows=None, cols=None, 
            figsize = (6,5.9),
            fill=1, showax=0,
            do =  ['plot']):
        if rows is None != cols is None:
            raise ValueError("Set either both rows and cols or neither.")

        print(f'|---> plot_save_grid {len(ims)} to {path}')

        plt.close() 

        if rows is None:
            rows = len(ims)
            cols = 1
        gridspec_kw = {'wspace': 0, 'hspace': 0} if fill else {}
        fig,axarr = plt.subplots(rows, cols, gridspec_kw=gridspec_kw, figsize=figsize)

        if fill:
            bleed = 0
            fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))

        for ax,im in zip(axarr.ravel(), ims):
            ax.imshow(im)
            if not showax:
                ax.set_axis_off()

        kwargs = {'pad_inches': .01} if fill else {}

        if 'save' in do:
            if path:
                fig.savefig(path, **kwargs)
            else:
                print(f'path must be defined')

        if 'plot' in do:
            plt.show()

    @staticmethod
    def plot_iter_grid(model, dataset, rows, cols, figsize = (10,9), do=['plot'], ext='jpg', prefix = 'frame'):
        rndvects = {}

        print(f'|--->  plot_iter_grid {do}')

        size = (model.input_shape[0], model.input_shape[1], model.input_shape[2])
        vary = np.random.normal(0.0, 1.0, size=size).astype('float32') # vary shape: (512, 512, 3)

        iterator = iter(dataset)
        
        ppi = 80
        ppt = 3 * ppi
        fy = int(cols * ppt / ppi)
        fx = int(rows * ppt / ppi)

        #fig = plt.figure(figsize=figsize)
        fig = plt.figure(figsize=(fy, fx))
        #plt.tight_layout(pad=0.2, w_pad=0.2, h_pad=1.0)

        if 1: # fill:
            bleed = 0
            fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))

        qtiles = cols * rows
        
        ckptidx = model.ckptidx
        ckptidx = Onutil.nameint(ckptidx)
        frame = f'{prefix}{ckptidx}.{ext}'
        path = os.path.join(model.results_dir, frame)

        images = []
        for i in range(rows):

            for j in range(cols):

                inp, rea = iterator.get_next()
                img = model.generator(inp, training=True)
                img = Onformat.nnba_to_rgb(img)

                images.append(img)


        Onplot.plot_save_grid(images, path, rows, cols,do=['plot', 'save'])



    @staticmethod   
    def plot_nuas(imgs=[], r=1, c=1, titles=[]):
        assert(not c == 0)
        qtiles = r * c
        qimgs = len(imgs)
        for i,img in enumerate(imgs):
            col = 1 + (i % c)   # subplot rows: 1 ..
            row = 1 + int(i/c)  # subplot cols: 1 ..
            print("plot_nuas x", i, r, c)
            plt.subplot(r, c, i + 1)
            idx =  i if i < qimgs else qimgs -1
            plt.imshow(imgs[idx]) 
            if (len(titles) > i):
                plt.title(titles[i])
            plt.axis('off')
        plt.show()
        plt.close()

    @staticmethod
    def plot_paths(img_paths=[], rows=1, cols=1):
        qtiles = rows * cols
        qimgs = len(img_paths)
        imgs = []
        for i in range(qtiles):
            idx =  i if i < qimgs else qimgs -1
            imgs.append(Image.open(img_paths[idx]))
        Onplot.plot_pils_grid(imgs, rows, cols)

    @staticmethod
    def plot_names(names=[], dir="./", rows=1, cols=1):
        imgpaths = []
        for name in names:
            imgpaths.append(os.path.join(dir, name))
        Onplot.plot_paths(imgpaths, rows, cols)     

    @staticmethod    
    def plot_pil(img, title=None):
        plt.imshow(img)
        if title:
            plt.title(title)
        plt.show()

    @staticmethod   
    def plot_nua(nua, title=None):
        plt.imshow(nua)
        if title:
            plt.title(title)
        plt.show()

    @staticmethod   
    def plot_dnua(tnua, title=None):
        nua = Onformat.dnua_to_nua(tnua)
        Onplot.plot_nua(nua, title)

    @staticmethod   
    def plot_rgb(img, title=None):
        rgb = np.array(img, dtype=np.uint8)
        plt.imshow(rgb)
        if title:
            plt.title(title)
        plt.show() 

    @staticmethod        
    def plot_grid_pils(imgs, rows=2, cols=2, figsize=(4, 4)):
        from mpl_toolkits.axes_grid1 import ImageGrid
        fig = plt.figure(figsize=figsize) # width, height in inches
        grid = ImageGrid(fig, 111,
                        nrows_ncols=(rows, cols),  # creates 2x2 grid of axes
                        axes_pad=0.1,  # pad between axes in inch.
                        )
        for ax, im in zip(grid, imgs): # Iterating over the grid returns the Axes.
            ax.imshow(im)
        plt.show()

    @staticmethod
    def plot_grid(im_list, grid_shape, scale=0.1, axes_pad=0.07):
    # https://gist.github.com/lebedov/7018889ba47668c64bcf96aee82caec0
        """
        Display the specified PIL images in a grid.
        Parameters
        ----------
        im_list : list of numpy.ndarray instances
            Bitmaps to display.
        grid_shape : tuple
            Grid shape.
        scale : float
            Scaling factor; 1 is 100%.
        axes_pad : float or (float, float)
            Padding between axes, in inches.
        """

        # Grid must be 2D:
        assert len(grid_shape) == 2

        # Make sure all images can fit in grid:
        assert np.prod(grid_shape) >= len(im_list)

        grid = ImageGrid(plt.gcf(), 111, grid_shape, axes_pad=axes_pad)
        for i, data in enumerate(im_list):

            # Scale image:
            im = PIL.Image.fromarray(data)
            thumb_shape = [int(scale*j) for j in im.size]
            im.thumbnail(thumb_shape, PIL.Image.ANTIALIAS)
            data_thumb = np.array(im)
            grid[i].plot_dnua(data_thumb)

            # Turn off axes:
            grid[i].axes.get_xaxis().set_visible(False)
            grid[i].axes.get_yaxis().set_visible(False)


    @staticmethod   
    def generate_and_plot_images(gen, seed, w_avg, truncation_psi=1):
        # https://github.com/rosasalberto/StyleGAN2-TensorFlow-2.x
        """ plot images from generator output """
        
        fig, ax = plt.subplots(1,3,figsize=(15,15))
        for i in range(3):
            
            # creating random latent vector
            rnd = np.random.RandomState(seed)
            z = rnd.randn(1, 512).astype('float32')

            # running mapping network
            dlatents = gen.mapping_network(z)
            # adjusting dlatents depending on truncation psi, if truncatio_psi = 1, no adjust
            dlatents = w_avg + (dlatents - w_avg) * truncation_psi 
            # running synthesis network
            out = gen.synthesis_network(dlatents)

            #converting image/s to uint8
            img = Onrosa.convert_images_to_uint8(out, nchw_to_nhwc=True, uint8_cast=True)

            #plotting images
            # ax[i].axis('off')
            # img_plot = ax[i].imshow(img.numpy()[0])
            print(f"plot img {i}: {type(img)} : {np.shape(img)}")
            Onplot.pil_show_rgb(img)

            seed += 1

    @staticmethod   
    def cv_tfr(src_dir,tfr_file):
        images = []
        tfr_files = sorted(glob.glob(os.path.join(src_dir, '*.tfrecords')))
        dataset = tf.data.TFRecordDataset(src_dir, compression_type=None, buffer_size=None, num_parallel_reads=None)
        tfrpath = os.path.join(src_dir, tfr_file)
        raw_image_dataset = tf.data.TFRecordDataset(tfrpath) # <TFRecordDatasetV2 shapes: (), types: tf.string>
        image_feature_description = {
            'shape': tf.io.FixedLenFeature([3], tf.int64),
            'data': tf.io.FixedLenFeature([], tf.string),
        }
        def _parse_image_function(example_proto):
            return tf.io.parse_single_example(example_proto, image_feature_description)        

        parsed_dataset = raw_image_dataset.map(_parse_image_function)
        for idx, dataitem in enumerate(parsed_dataset):
            tfshape = dataitem['shape']
            shape = tfshape.numpy()
            tfdata = dataitem['data']
            data = tfdata.numpy()
            size = (shape[1], shape[2])
            mode = 'RGB' if shape[0] == 3 else 'RGBA'
            img = np.array(data)
            channels = shape[0]
            width = shape[1]
            height = shape[2]
            bytes_needed = int(width * height * channels)  # tfr 9: 512x512x3
            bytesq = len(data) # tfr 9: 512x512x3
            b = np.frombuffer(data, dtype=np.uint8)
            img_arr =  b.reshape(shape[0], shape[1], shape[2]) # 3x3 b/w
            img_arr = img_arr.transpose([1, 2, 0])
            Onplot.cv_rgb(img_arr)

    @staticmethod   
    def cv_path(path, size = 512, title='img', wait=2000):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (size, int((np.shape(img)[0]/np.shape(img)[1]) * size)))
        img = Onvgg.vgg_preprocess(img)
        img = Onvgg.vgg_deprocess(img)
        cv2.imshow(title, img) # keep open per wait
        cv2.waitKey(wait)


    @staticmethod  
    def cv_resize(img, max_size= None, dim=None):
        if max_size:
            img = cv2.resize(img, (max_size, int((np.shape(img)[0]/np.shape(img)[1]) * max_size)))
        
        if dim:
            img = cv2.resize(img, (
                dim[0] if dim[0] > 0 else img.size[0],
                dim[1] if dim[1] > 0 else img.size[1]
                ), Image.ANTIALIAS)            

        return img


    def cv_rgb(rgb, wait=2000):
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow('img', rgb)
        if wait:
            cv2.waitKey(wait) 


    @staticmethod   
    def cv_nba(img, title="img", wait=0):
        img = Onformat.nnba_to_rgb(img)
        cv2.imshow(title, img)
        cv2.waitKey(0)

    @staticmethod   
    def cv_img(img, title="img", wait=0):
        if Onutil.incolab():
            from google.colab.patches import cv2_imshow
            cv2_imshow(img)
            cv2.waitKey(wait)
            cv2.destroyAllWindows()
        else:
            cv2.imshow(title, img)
            cv2.waitKey(wait)
            cv2.destroyAllWindows()


#   ******************
#   FUNS FORMAT
#
#   ******************
class Onformat:

    @staticmethod
    def cvt_to_rgb(cvt):
        img=np.array(cvt, dtype=np.float32)
        img = img[0]
        img = (img / 255 + 1.0)/2.0
        return img

    @staticmethod
    def pil_to_cvi(img):
        img = np.array(img) # [[[155  91  69]
        img = img[:, :, ::-1] # val: [[[ 69  91 155]
        return img

    @staticmethod
    def pil_to_nua(img):
        img = img.convert("RGB")
        img = np.asarray(img, dtype=np.float32) / 255    
        return img

    @staticmethod
    def nua_to_pil(nua):
        nua = nua*255
        nua = np.array(nua, dtype=np.uint8)
        if np.ndim(nua)>3:
            assert nua.shape[0] == 1
            nua = nua[0]
        return PIL.Image.fromarray(nua)


    @staticmethod
    def pil_to_rgb(img):
        img = img.convert("RGB")
        img = np.asarray(img)
        return img

    @staticmethod
    def nba_to_nnba(img):
        return img[np.newaxis,:,:,:]
    
    @staticmethod
    def tnua_to_nua(img):
        # np.array (by default) will make a copy of the object, 
        # while np.asarray will not unless necessary
        img = np.asarray(img) # img = np.array(img)
        return img

    @staticmethod
    def rgb_to_pil(img):
        img = PIL.Image.fromarray(np.array(img, dtype=np.uint8))
        return img

    @staticmethod
    def rgb_to_nba(img):
        # normalizing the images to [-1, 1]
        img = tf.cast(img, tf.float32) # _e_
        img = (img / 127.5) - 1
        return img
        
    @staticmethod
    def _rgbs_to_nbas(input_image, real_image):
        # normalizing the images to [-1, 1]
        input_image = tf.cast(input_image, tf.float32) # _e_
        real_image = tf.cast(real_image, tf.float32)   # _e_
        input_image = (input_image / 127.5) - 1
        real_image = (real_image / 127.5) - 1
        return input_image, real_image

    @staticmethod
    def rgbs_to_nbas(imgs):
        res = []
        for img in imgs:
            img = Onformat.rgb_to_nba(img)
            res.append(img)
        return res

    @staticmethod
    def rgbs_to_nuas_batch(imgs, num, flip = True):
        idx = np.random.randint(0, imgs.shape[0] - 1, num)
        out = []

        for i in idx:
            out.append(imgs[i])
            if flip and np.random.uniform(()) < 0.5:  # flip as arg
                out[-1] = np.flip(out[-1], 1)

        return np.array(out).astype('float32') / 255.0   

    @staticmethod
    def nua_to_rgb(img):
        img = img*255
        rgb = np.array(img, dtype=np.uint8)
        if np.ndim(img)>3:
            assert img.shape[0] == 1
            rgb = img[0]
        return rgb

    @staticmethod
    def nua_to_pil(img):
        img = Onformat.nua_to_rgb(img)
        img = PIL.Image.fromarray(img)
        return img

    @staticmethod
    def nua_to_rgb(nua):
        res = nua*255
        res = np.array(res, dtype=np.uint8)
        if np.ndim(res)>3:
            assert res.shape[0] == 1
            res = res[0]
        return res

    @staticmethod
    def nuas_to_rgbs(nuas):
        imgs=[]
        for nua in nuas:
            imgs.append(Onformat.nua_to_rgb(nua))
        return imgs

    @staticmethod
    def names_to_nnuas(img_names=[], img_dir = "./", max_size = None, img_nrows = None, img_ncols = None):
        _imgs = []
        for img_name in img_names:
            fil = os.path.join(img_dir, img_name)
            img = Onfile.path_to_nnua(fil, max_size, img_nrows, img_ncols)
            _imgs.append(img)
        return _imgs

    @staticmethod
    def nnba_to_rgb(img):
        img = img[0,...]
        img = (img + 1.0)/2.0
        img = np.array(255 * img, dtype=np.uint8)
        return img

    @staticmethod
    def nba_to_rgb(img):
        img = 255*(img + 1.0)/2.0
        return tf.cast(img, tf.uint8)

    @staticmethod
    def bgr_to_nua(img):
        img = (img / 255 + 1.0)/2.0
        return img

    @staticmethod
    def nba_to_nua(img): # [-1,1] => [0,255]
        img = (img + 1.0)/2.0
        return img

    @staticmethod
    def tnua_resize(image, width, height):
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, (width, height))
        # image = image[None, ...]
        return image

    @staticmethod
    def cvt_to_nua(cvt):
        img=np.array(cvt, dtype=np.float32)
        img = img[0] # squeeze dim
        img = (img / 255 + 1.0)/2.0
        return img

    @staticmethod
    def imgs_to_tiling(imgs, imgpath, qsegs=1, save=True):
        qtiles = qsegs * qsegs  # will predict images
        r = []

        for i in range(0, qtiles, qsegs): # for each row
            r.append(np.concatenate(imgs[i:i+qsegs], axis = 0)) # concat cols

        c1 = np.concatenate(r, axis = 1)
        c1 = np.clip(c1, 0.0, 1.0)

        x = Image.fromarray(np.uint8(c1*255))

        if save:
            x.save(imgpath)
        return x

    @staticmethod
    def dnua_to_nua(tnua):
        nua = tnua
        if len(tnua.shape) > 3:
            assert tnua.shape[0] == 1
            nua = tf.squeeze(tnua, axis=0)
        nua = np.asarray(nua)        
        return nua

    @staticmethod
    def dnuas_to_nuas(tnuas):
        nuas=[]
        for item in tnuas:
            nuas.append(Onformat.dnua_to_nua(item))
        return nuas

    @staticmethod
    def dnuas_to_pils(nuas):
        imgs=[]
        for nua in nuas:
            imgs.append(Onformat.nua_to_pil(nua))
        return imgs
    
    @staticmethod
    def rgb_to_bgr(img):
        img = np.array(img) # [[[155  91  69]
        img = img[:, :, ::-1] # val: [[[ 69  91 155]

        return img

    @staticmethod
    def rgb_to_dnua(img):
        img = img/255.0
        img = img[np.newaxis,:,:,:]
        return img

    @staticmethod
    def pil_to_dnua(img):
        img=np.array(img, dtype=np.float32)
        img = img/255.0
        img = img[np.newaxis,:,:,:]
        return img

    @staticmethod
    def pils_to_dnuas(pils):
        dnuas=[]
        for img in pils:
            nua = Onformat.pil_to_dnua(img)
            dnuas.append(nua)
        return dnuas

    @staticmethod
    def cvi_to_pil(img):
        img = img[...,::-1] # bgr => rgb
        img = Image.fromarray((img).astype(np.uint8))
        img = np.array(img)
        img = PIL.Image.fromarray(img)
        return img


    @staticmethod
    def cvis_to_pils(imgs):
        pils=[]
        for img in imgs:
            pil = Onformat.cvi_to_pil(img)
            pils.append(pil)
        return pils

    @staticmethod
    def bgr_cv2_rgb(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def path_to_nbt_with_tf(img, height=256, width=256):
        img = Onfile.path_to_rgb(img)
        img = tf.image.resize(img, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)        
        img = Onformat.rgb_to_nba(img)
        return img


#   ******************
#   FUNS FILE
#
#   ******************
class Onfile:

    @staticmethod
    def rgbs_to_file(rgbs, scale=1, rows=1, save_path='./img.png',):
        pils = []
        for img in rgbs:
            pils.append(PIL.Image.fromarray(np.array(img, dtype=np.uint8)))
        w,h = pils[0].size
        w = int(w*scale)
        h = int(h*scale)
        height = rows*h
        cols = int(math.ceil(len(pils) / rows))
        width = cols*w
        canvas = PIL.Image.new('RGBA', (width,height), 'white')
        for i,img in enumerate(pils):
            img = img.resize((w,h), PIL.Image.ANTIALIAS)
            canvas.paste(img, (w*(i % cols), h*(i // cols))) 
        canvas.save(save_path)

    @staticmethod
    def path_to_nba(img, height=256, width=256):
        img = Onfile.path_to_rgb(img)
        img = tf.image.resize(img, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        img = Onformat.rgb_to_nba(img)
        return img

    @staticmethod
    def path_to_tnua_with_cv(path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = img.astype(np.float32)

        img = img[...,::-1]
        # shape (h, w, d) to (1, h, w, d)
        img = img[np.newaxis,:,:,:]
        img -= np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
        return img

    @staticmethod
    def path_to_cvi(path,  max_size=None):
        # bgr image
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if max_size:
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


    @staticmethod
    def path_cv_pil(path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = img[...,::-1] # bgr => rgb
        img = np.array(Image.fromarray((img).astype(np.uint8)))
        img = PIL.Image.fromarray(img)
        return img

    @staticmethod
    def path_to_tvgg(img_path, tileimg=None, scale=None, max_size=None ):
        print("get init with cv2")
        img = Onfile.path_to_cvi(img_path,max_size=max_size)
        img = Onvgg.vgg_preprocess(img)    
        return img

    @staticmethod
    def path_to_nua(img_path, max_size = None, img_nrows = None, img_ncols = None):
        assert os.path.isfile(img_path), f"path_to_nnua {img_path} not found"

        # <class 'tensorflow.python.framework.ops.EagerTensor'> ()
        # tf.Tensor(b'\xff\xd8\xff\xe0\x00\x10JFIF
        img = tf.io.read_file(img_path)

        img = tf.image.decode_image(img, channels=3) # _e_
        img = tf.image.convert_image_dtype(img, tf.float32)

        shape = tf.cast(tf.shape(img)[:-1], tf.float32) # 
        long_dim = max(shape)

        scale = max_size / long_dim if max_size else 1
        new_shape = tf.cast(shape * scale, tf.int32)

        if (img_nrows and img_ncols): new_shape = tf.cast((img_nrows, img_ncols), tf.int32)
        img = tf.image.resize(img, new_shape)
        return img

    @staticmethod
    def path_to_nnua(img_path, max_size = None, img_nrows = None, img_ncols = None):
        img = Onfile.path_to_nua(img_path, max_size, img_nrows, img_ncols)
        img = img[tf.newaxis, :] # b, h, w, c
        return img

    @staticmethod
    def names_to_nuas_with_tf(imgs_names, img_dir, args=None):
        imgs = []
        for item in imgs_names:
            path = os.path.join(img_dir, item)
            img = Onfile.path_to_tnua_with_tf(path, args)
            imgs.append(img)
        return imgs

    @staticmethod
    def names_to_paths(imgs_names, img_dir):
        paths = []
        for item in imgs_names:
            path = os.path.join(img_dir, item)
            paths.append(path)
        return paths

    @staticmethod
    def folder_to_tnuas(folder, patt='*.jpg', max_size = None, img_nrows = None, img_ncols = None):
        paths = glob.glob(os.path.join(folder, patt))
        tnuas = []
        for path in paths:
            nua = Onfile.path_to_nnua(path, max_size, img_nrows, img_ncols)
            tnuas.append(nua)
        return tnuas

    @staticmethod
    def path_to_tnua_with_tf(path, args=None):
            
        max_size = args.max_size 
        assert os.path.isfile(path), f"path_to_tnua_with_tf img {path} not found"

        parts = tf.strings.split(path, os.sep)
        label = parts[-2]    

        # <class 'tensorflow.python.framework.ops.EagerTensor'>  () tf.Tensor(b'\xff\xd8\xff\xe0\x00\x10JFIF
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        shape = tf.cast(tf.shape(img)[:-1], tf.float32) #  
        long_dim = max(shape)
        scale = max_size / long_dim if max_size else 1
        new_shape = tf.cast(shape * scale, tf.int32)

        # <class 'tensorflow.python.framework.ops.EagerTensor'>  (336, 512, 3) tf.Tensor([[[0.60938966 0.36011618 0.27445853]    
        img = tf.image.resize(img, new_shape)
        img = img[tf.newaxis, :]

        return img

    @staticmethod
    def pil_to_file_with_cv(path, img):
        cv2.imwrite(path, Onformat.pil_to_cvi(img))

    @staticmethod
    def pil_to_file(path, img):
        img.save(path)

    @staticmethod
    def path_to_bgr(path): # _e_
        # bgr image 
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = img.astype(np.float32)
        # img = Onformat.vgg_preprocess(img)
        return img

    @staticmethod
    def names_to_pils(imgs,dst_dir= './',img_name_frmt='img{}.jpg',zfill = 4,):
        for i,img in enumerate(imgs):
            img_name = img_name_frmt.format(str(i).zfill(zfill))
            path = os.path.join(dst_dir, img_name)
            tft = img # <class 'tensorflow.python.framework.ops.EagerTensor'> (256, 256, 3)
            arr = img.numpy().astype(np.uint8) # <class 'numpy.ndarray'> (256, 256, 3)
            img = Image.fromarray(arr)
            img = img.convert("RGB")
            img.save(path)

    @staticmethod
    def path_cv2_dnua(path):  #  _e_
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = Onformat.vgg_preprocess(img)
        img = Onformat.vgg_deprocess(img)
        img = Onformat.pil_to_dnua(img)
        return img
    
    @staticmethod
    def cvs_to_folder(imgs, path, img_format='img{}.jpg', zfill=4):
        for i,img in enumerate(imgs):
            Onfile.cv_to_path(img, os.path.join(path, img_format.format(str(i).zfill(zfill))))

    @staticmethod
    def cv_to_path(img, path):
        saved = cv2.imwrite(path, img)
        return saved

    @staticmethod
    def cv_to_file(path, img):
        cv2.imwrite(path, Onformat.rgb_to_bgr(img))

    @staticmethod
    def path_to_cv(path, max_size=512):

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


    @staticmethod
    def paths_to_cvs(paths):
        imgs = []
        for path in paths:
            imgs.append(Onfile.path_to_cv(path))
        return imgs

    @staticmethod
    def paths_to_folder_with_cv(paths, folder, img_format='img{}.jpg', zfill=4):
        imgs = []
        for i,path in enumerate(paths):
            img = Onfile.path_to_cv(path)
            img_path = os.path.join(folder, img_format.format(str(i).zfill(zfill)))
            if 1:
                print(f'save {i} to {img_path}')
            Onfile.cv_to_file(img_path, img)
        return imgs

    @staticmethod
    def folder_to_cvs(path):
        paths = Onfile.folder_to_paths(path)
        imgs = Onfile.paths_to_cvs(paths)
        return imgs

    @staticmethod
    def folder_to_cv_name_pairs(path, args=None):
        pairs = []
        for root, subdirs, files in os.walk(path):
            if(args.verbose): print('folder_to_cv_name_pairs --\nroot = ' + root)

            for subdir in subdirs:
                if(args.verbose): print('\t- subdirectory ' + subdir)

            for filename in files:
                if not filename.startswith('.'):
                    file_path = os.path.join(root, filename)
                    if(args.verbose): print('\t- file %s (full path: %s)' % (filename, file_path))
                    
                    pairs.append([cv2.imread(file_path),filename])
        return pairs

    @staticmethod
    def cv_name_pairs_to_folder(pairs, args=None):
        folder = args.output_folder
        for pair in pairs:
            img = pair[0]
            filename = pair[1]
            if(args.file_extension == "png"):
                new_file = os.path.splitext(filename)[0] + ".png"
                cv2.imwrite(os.path.join(folder, new_file), img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            else:
                new_file = os.path.splitext(filename)[0] + ".jpg"
                cv2.imwrite(os.path.join(folder, new_file), img, [cv2.IMWRITE_JPEG_QUALITY, 90])

    @staticmethod
    def path_to_pil(path):
        img = Image.open(path)
        return img

    @staticmethod
    def folder_to_paths(path, 
            exts=['jpg', 'jpeg', 'png'], 
            n=-1
        ):
        import glob
        paths = [] # <class 'list'>
        for ext in exts:
            paths += glob.glob(os.path.join(path, f"*.{ext}"))
            if 0 < n and n < len(paths):
                break            
        return paths

    @staticmethod
    def folder_to_paths(path, 
            exts=['jpg', 'jpeg', 'png'], 
            n=-1
        ):
        paths=[]
        for root, dirs, files in os.walk(path):
            for name in files:
                if name.endswith(tuple(exts)):
                    target_path = os.path.join(path, name)
                    paths.append(target_path)
                    if 0 < n and n < len(paths):
                        break
                else:
                    pass # print('Only image')          
        return paths


    @staticmethod
    def copyfolder(input_folder, output_folder, deep=True):
        for filename in os.listdir(input_folder):
            input_path = os.path.join(input_folder, filename)
            try:
                if os.path.isfile(input_path) or os.path.islink(input_path):
                    output_path = os.path.join(output_folder, filename)
                    shutil.copyfile(input_path, output_path)
                elif os.path.isdir(input_path):
                    if deep:
                        Onfile.copyfolder(input_path, output_folder)
            except Exception as e:
                print(f'Failed to copy {input_path}. Reason: {e}')
                
    @staticmethod
    def clearfolder(folder, inkey=''):
        if inkey and inkey in folder:
            print(f"|===> clearfolder: {folder} has key {inkey}. will clear")
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))
        else:
            print(f"|===> clearfolder: {folder} has no key {inkey}")

    @staticmethod
    def qfolders(folder, topdown=False):
        q = 0
        if os.path.exists(folder):        
            for root, dirs, files in os.walk(folder, topdown=topdown):
                for name in dirs:
                    q += 1
        return q

    @staticmethod
    def qfiles(folder, patts=None):
        howmany = 0
        if not patts:
            howmany = len(os.listdir(folder))
        elif type(patts) is list: 
            paths = []
            howmany = len(paths)
            for patt in patts:
                infolder = os.path.join(folder, patt)
                pathsinfolder = glob.glob(infolder)
                paths = paths + pathsinfolder
                howmany = len(paths)		
        else:
            files = glob.glob(os.path.join(folder, patts))
            howmany = len(files)
        return howmany


    @staticmethod
    def path_to_paths(path, patts=None): # ['*.jpg', '*.jpeg', '*.png']
        paths = []

        if patts:
            for patt in patts:
                inpath = os.path.join(path, patt)
                new = glob.glob(inpath)
                paths = paths + new
        else:
            for entry in os.listdir(path):
                full_path = os.path.join(path, entry)
                if os.path.isfile(full_path):
                    paths.append(full_path)

        return paths

    @staticmethod
    def folder_to_pils(src_dir, n=-1):
        if 0: print(f"folder_to_pils: was asked to get {n} imgs from: {src_dir}")
        imgs = []
        import glob
        jpgs = glob.glob(os.path.join(src_dir, "*.jpg"))
        pngs = glob.glob(os.path.join(src_dir, "*.png"))
        fs = jpgs + pngs  # <class 'list'>
        for i,f in enumerate(fs):
            if (n<0 or i<n):
                img = Image.open(f)
                imgs.append(img)
        if 0: print(f"folder_to_pils: got {len(imgs)} imgs from: {src_dir}")            
        return imgs

    @staticmethod
    def pils_to_folder(imgs, dest_dir, img_format = 'img{}.jpg', zfill=4, verbose=False):
        for i, pil in enumerate(imgs):        
            img_name = img_format.format(str(i).zfill(zfill))                
            outpath = os.path.join(dest_dir, img_name)
            if 0: 
                print(f"save image {i} to {outpath}")
            pil.save(outpath, 'JPEG', quality=90) 


    @staticmethod
    def folder_to_named_files(src_dir, dst_dir, img_format= 'img{}.jpg', zfill= 4, n=-1, verbose=False):
        paths=[]
        imgs=[]
        idx = 0
        for root, dirs, files in os.walk(src_dir):
            for name in files:
                if name.endswith((".png", ".jpg")):
                    src_path = os.path.join(src_dir, name)

                    newname = img_format.format(str(idx).zfill(zfill))       
                    target_path = os.path.join(dst_dir, newname)

                    img = Image.open(src_path)
                    paths.append(target_path)
                    if verbose: 
                        print(f"save image {idx} to {target_path}")
                    img.save(target_path, 'JPEG', quality=90) 

                    if -1 < n and n < len(paths):
                        break
                    idx += 1
                else:
                    print('get only image files')          
        return paths

    @staticmethod
    def folder_to_rgbs(src_dir, qimgs=-1):
        if 0: print(f"folder_to_rgbs q:{qimgs} from: {src_dir}")
        imgs = Onfile.folder_to_pils(src_dir, qimgs)
        res=[]
        for img in imgs:
            img = np.asarray(img)
            res.append(img)
        return res

    @staticmethod
    def folder_to_nuas(src_dir, qimgs=-1):
        if 0: print(f"folder_to_nuas q:{qimgs} from: {src_dir}")
        imgs = Onfile.folder_to_rgbs(src_dir, qimgs)
        res=[]
        for img in imgs:
            img = img.astype(np.float32)
            img = img / 255
            res.append(img)
        return np.asarray(res)

    @staticmethod
    def tfts_to_files(imgs,dst_dir= './',img_name_frmt='img{}.jpg',zfill = 4,):
        for i,img in enumerate(imgs):
            img_name = img_name_frmt.format(str(i).zfill(zfill))
            path = os.path.join(dst_dir, img_name)
            tft = img # <class 'tensorflow.python.framework.ops.EagerTensor'> (256, 256, 3)
            arr = img.numpy().astype(np.uint8) # <class 'numpy.ndarray'> (256, 256, 3)
            img = Image.fromarray(arr)
            img = img.convert("RGB")
            img.save(path)

    @staticmethod
    def path_of_paired_to_rgbs(path):
        image = tf.io.read_file(path) # => dtype=string
        image = tf.image.decode_jpeg(image) # => shape=(256, 512, 3), dtype=uint8)
        w = tf.shape(image)[1]
        w = w // 2
        real_image = image[:, :w, :] # real comes left 
        input_image = image[:, w:, :] 
        return input_image, real_image

    @staticmethod
    def path_to_rgb(path):
        img = tf.io.read_file(path) # => dtype=string
        img = tf.image.decode_jpeg(img) # => shape=(256, 512, 3), dtype=uint8)
        return img

    @staticmethod
    def path_to_rgbt(path):
        return Onfile.path_to_rgb(path)

    @staticmethod
    def path_to_tnua_with_tf(path, args=None):
        max_size = args.max_size 
        assert os.path.isfile(path), f"path_to_tnua_with_tf img {path} not found"
        parts = tf.strings.split(path, os.sep)
        label = parts[-2]    

        # <class 'tensorflow.python.framework.ops.EagerTensor'>  () tf.Tensor(b'\xff\xd8\xff\xe0\x00\x10JFIF
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        shape = tf.cast(tf.shape(img)[:-1], tf.float32) #  
        long_dim = max(shape)
        scale = max_size / long_dim if max_size else 1
        new_shape = tf.cast(shape * scale, tf.int32)

        # <class 'tensorflow.python.framework.ops.EagerTensor'>  (336, 512, 3) tf.Tensor([[[0.60938966 0.36011618 0.27445853]    
        img = tf.image.resize(img, new_shape)
        img = img[tf.newaxis, :]

        return img

    @staticmethod
    def names_tf_tnuas(imgs_names, img_dir, args=None):
        imgs = []
        for item in imgs_names:
            path = os.path.join(img_dir, item)
            img = Onfile.path_to_tnua_with_tf(path, args)
            imgs.append(img)
        return imgs

    # https://github.com/rosasalberto/StyleGAN2-TensorFlow-2.x
    @staticmethod
    def generate_and_save_images(images, it, plot_fig=True, outdir='./'):
        plt.close() 
        fig = plt.figure(figsize=(9,9))

        for i in range(images.shape[0]):
            plt.subplot(2, 2, i+1)
            plt.imshow(images[i])
            plt.axis('off')

        # tight_layout minimizes the overlap between 2 sub-plots
        fig.tight_layout()
        imgoutname = 'image_at_iter_{:04d}.png'.format(it)
        imgoutpath = os.path.join(outdir, imgoutname)
        plt.savefig(imgoutpath)
        if plot_fig: plt.show()

    @staticmethod
    def rgb_cv2_file(rgb, path): # to jpg
        im_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, im_bgr)   


#   ******************
#   FUNS Onimg
#
#   ******************
class Onimg:


    @staticmethod
    def img_to_mask(img, outpath=None, 
                    height=None, width=None, 
                    threshold=76, maxVal=255, 
                    thresmode=cv2.THRESH_BINARY,
                    visual=0, verbose=0,
                ):
        #https://answers.opencv.org/question/228538/how-to-create-a-binary-mask-for-medical-images/

        if verbose > -1:
            print(f'|---> img_to_mask {np.shape(img)}')

        #cv2.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]])
        if height and width:
            (h1, w1) = (height, width)
            img = cv2.resize(img, (w1, h1))
        elif height:
            h0 = np.shape(img)[0] # 384
            w0 = np.shape(img)[1] # 512
            h1 = height
            w1 = int(w0  * (h1/h0))
            img = cv2.resize(img, (w1, h1))
        elif width:
            h0 = np.shape(img)[0] # 384
            w0 = np.shape(img)[1] # 512
            w1 = width
            h1 = int(h0  * (w1/w0))
            dim = (w1, h1)
            img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        else:
            pass

        image_contours = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)

        image_binary = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)

        for channel in range(img.shape[2]):
            ret, image_thresh = cv2.threshold(img[:, :, channel],
                                            threshold, maxVal,
                                            thresmode)

            contours = cv2.findContours(image_thresh, 1, 1)[0]   
            cv2.drawContours(image_contours,
                            contours, -1,
                            (255,255,255), 3)

        contours = cv2.findContours(image_contours, cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)[0]

        cv2.drawContours(image_binary, [max(contours, key = cv2.contourArea)],
                        -1, (255, 255, 255), -1)

        if outpath:
            outbasename = os.path.basename(outpath)
            outdirname = os.path.dirname(outpath)
            assert os.path.exists(outdirname), f"outdirname {outdirname} does not exist"
            cv2.imwrite(outpath, image_binary)

        if visual > 0:
            cv2.imshow('inimg', img)            
            cv2.imshow('outimg', image_binary)
            cv2.waitKey(0) & 0xFF is 27
            cv2.destroyAllWindows()

        return image_binary


    @staticmethod
    def name_to_maskname(name):

        name_base = os.path.splitext(name)[0]
        name_ext = os.path.splitext(name)[1]
        print("|...> name_base", name_base, name_ext)

        name_mask = f'{name_base}_mask{name_ext}'
        return name_mask
        

    @staticmethod
    def name_to_unmaskname(name):

        name_base = os.path.splitext(name)[0]
        name_ext = os.path.splitext(name)[1]
        print("|...> name_base", name_base, name_ext)

        basename_unmask = name_base.replace('_mask', '')
        name_unmask = f'{basename_unmask}{name_ext}'
        return name_unmask
        

    @staticmethod
    def path_to_maskpath(path):

        dirname = os.path.dirname(path)
        basename  =os.path.basename(path)

        maskname = Onimg.name_to_maskname(basename)
        maskpath = os.path.join(dirname, maskname)

        return maskpath

    @staticmethod
    def path_to_mask(inpath, outpath, height = None, width = None, ml=2*38, mh=255, visual=0):
        #https://answers.opencv.org/question/228538/how-to-create-a-binary-mask-for-medical-images/

        inbasename = os.path.basename(inpath)
        assert os.path.exists(inpath), f"inpath {inpath} does not exist"

        img = cv2.imread(inpath)
        image_binary = img_to_mask(img, outpath, height, width, ml, mh, visual)

        return image_binary


    @staticmethod
    def tf_resize_nua(nua, args=None):
            
        max_size = args.max_size 

        shape = np.shape(nua)
        (d,w,h,c) = shape
        dim = (w,h)
        long_dim = max(w,h)
        scale = max_size / long_dim if max_size else 1
        new_dim = (int(w * scale), int(h * scale))

        print(f"|===> tf_resize_nua \n \
            nua type: {type(nua)} \n \
            shape: {shape} \n \
            max_size: {max_size} \n \
            long_dim: {long_dim} \n \
            scale: {scale} \n \
            dim: {dim} \n \
            new_dim: {new_dim} \n \
        ")

        nua = tf.image.resize(nua, new_dim)
        #nua = nua[tf.newaxis, :]

        print(f"|... tf_resize_nua \n \
            nua type: {type(nua)} \n \
            shape: {np.shape(nua)} \n \
        ")


        return nua


    @staticmethod
    def cvi_mask_to_cvi(img, mask, op=2):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        
        if op < 0: # reverse mask
            mask = cv2.bitwise_not(mask)
            op = abs(op)

        if op == 1: # bitwise_and
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            img = cv2.bitwise_and(img, mask)

        elif op == 2: # bitwise_or
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            img = cv2.bitwise_or(img, mask)

        elif op == 3: # bitwise_xor
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            img = cv2.bitwise_xor(img, mask)

        elif op == 4: # bitwise_not
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            img = cv2.bitwise_not(img, mask)

        elif op == 5: # addWeighted
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            img = cv2.addWeighted(img, 0.5, mask, 0.5, 0)

        elif op == 6: # thresh

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (13,13), 0)
            thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,51,7)

            # Morph close
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

            # Find contours, sort for largest contour, draw contour
            cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            for c in cnts:
                cv2.drawContours(thresh, [c], -1, (36,255,12), 2)
                break
            img = thresh



        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img


#   ******************
#   FUNS ONVID
#
#   ******************
class Onvid:

    @staticmethod
    def vid_to_frames(video_path, frames_dir='./', frame_format= 'frame{}.jpg', zfill= 4, target = 1):

        video = cv2.VideoCapture(video_path)      

        success, frame = video.read() # Capture frame-by-frame

        currentFrame = 0
        counter = 0
        addframe = 0

        start = time.time()
        while(success):

            if currentFrame % target ==0:
                name = frames_dir + '/frame' + "{:04d}".format(currentFrame) + '.jpg'
                if 0: print (f'Creating... {currentFrame} {name}')
                cv2.imwrite(name, frame)

                success,frame = video.read()
                if 0: print (f'will create... {currentFrame}')
                currentFrame += 1
                counter = 0
                addframe += 1

            else:
                ret = video.grab()
                currentFrame += 1
                counter += 1

        end = time.time()
        seconds = end - start
        num_frames = currentFrame
        fps  = addframe / (seconds * target)
        print (f"|... Estimated frames per second : {int(round(fps))}")

        video.release() # When everything done, release the capture
        cv2.destroyAllWindows()

    @staticmethod
    def frames_to_video(inputpath, outputpath, fps):
        DWITH_FFMPEG="ON"    
        image_array = []
        files = [f for f in os.listdir(inputpath) if os.path.isfile(os.path.join(inputpath, f))]
        
        # files.sort(key = lambda x: int(x[5:-4]))

        if 1:
            for i in range(len(files)):
                img_path = os.path.join(inputpath,files[i] )
                print(f'|---> frames_to_video img_path: {img_path}')
                img = cv2.imread(img_path)
                size =  (img.shape[1],img.shape[0])
                img = cv2.resize(img,size)
                image_array.append(img)

            fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
            print(f'|---> frames_to_video outputpath: {outputpath}')            
            out = cv2.VideoWriter(outputpath, fourcc, fps, size) # VideoWriter(filename, cv2.CAP_OPENCV_MJPEG, ...)
            for i in range(len(image_array)):
                out.write(image_array[i])     
            out.release()

        else:
            with imageio.get_writer(outputpath, mode='I') as writer:
                for filename in files:
                    img_path = os.path.join(inputpath,filename)
                    image = imageio.imread(img_path)
                    writer.append_data(image)   

    @staticmethod
    def folder_to_movie(images, out_dir, out_name):
        temp_dir = 'frames%06d'%int(1000000*random.random())
        os.system('mkdir %s'%temp_dir)
        for idx in tqdm(range(len(images))):
            Image.fromarray(images[idx], 'RGB').save('%s/frame%05d.png' % (temp_dir, idx))
        cmd = 'ffmpeg -i %s/frame%05d.png -c:v libx264 -pix_fmt yuv420p %s/%s.mp4' % (temp_dir, out_dir, out_name)
        print(cmd)
        os.system(cmd)
        os.system('rm -rf %s'%temp_dir)

    @staticmethod
    def folder_to_vid(fromfolder, dstfile, ext='png', save=True, args=None):
        srcpath = fromfolder
        anifile = dstfile
        size=(640, 480)
        fps=20

        if 1: # args.verbose: 
            print(f'|--->  folder_to_vid  \n \
                srcpath:         {srcpath} \n \
                anifile:         {anifile} \n \
            ')            

        # anime.anigif(srcpath, anifile)
        fig = plt.figure()
        fig.set_size_inches(size[0] / 100, size[1] / 100)
        ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect=1)
        ax.set_xticks([])
        ax.set_yticks([])
        images = []

        imgnames = [img for img in os.listdir(srcpath) if img.endswith(ext)]
        for i,imgname in enumerate(imgnames):
            imgpath = os.path.join(srcpath, imgname)
            img = imageio.imread(imgpath)
            label=' '
            plt_im = plt.imshow(img, animated=True)
            plt_txt = plt.text(10, 310, label, color='black')
            images.append([plt_im, plt_txt])

        # animated_gif.save(anipath + "/ani.gif")
        # MovieWriter imagemagick unavailable; 
        # trying to use <class 'matplotlib.animation.PillowWriter'> instead
        if 1: #  args.verbose:
            print(f'|...>  folder_to_vid: call anim.ArtistAnimation  \n \
                images len:      {len(images)} \n \
            ')

        import matplotlib.animation as anim
        ani = anim.ArtistAnimation(fig, images, 
            interval=20, blit=True, repeat_delay=1000)

        if save:
            ani.save(dstfile, writer='imagemagick')  

        return ani

    @staticmethod
    def folder_to_gif(fromfolder, dstpath='./out.gif', patts=None):
        paths = Onfile.path_to_paths(fromfolder, patts)
        paths = sorted(paths)

        print(f'|---> folder_to_gif \n \
            fromfolder: {fromfolder} \n \
            dstpath: {dstpath} \n \
            patts: {patts} \n \
            paths: {len(paths)} \n \
        ')

        with imageio.get_writer(dstpath, mode='I') as writer:

            for i,filename in enumerate(paths):
                # if i % 8 != 0: continue
                img = imageio.imread(filename)
                writer.append_data(img)
            #image = imageio.imread(filename)
            #writer.append_data(image)

    @staticmethod
    def vid_show(path):
        print("show anigif")
        anim_file = path
        cap = cv2.VideoCapture(anim_file)
        ret, frame = cap.read()
        while(1):
            ret, frame = cap.read()
            if (not frame is None):
                cv2.imshow('frame',frame)
                if cv2.waitKey(1) & 0xFF == ord('q') or ret==False :
                    cap.release()
                    cv2.destroyAllWindows()
                    break
                cv2.imshow('frame',frame)

    @staticmethod
    def imgs_to_tiling(imgs, imgpath, qsegs=1, save=True):
        qcells = qsegs * qsegs  # will predict images
        r = []

        for i in range(0, qcells, qsegs): # for each row
            r.append(np.concatenate(imgs[i:i+qsegs], axis = 0)) # concat cols

        c1 = np.concatenate(r, axis = 1)
        c1 = np.clip(c1, 0.0, 1.0)

        x = Image.fromarray(np.uint8(c1*255))

        if save:
            x.save(imgpath)
        return x    

    # rolux
    @staticmethod
    def render_video(src_file, dst_dir, tmp_dir, num_frames, mode, size, fps, codec, bitrate):

        import PIL.Image
        import moviepy.editor

        def render_frame(t):
            frame = np.clip(np.ceil(t * fps), 1, num_frames)
            image = PIL.Image.open('%s/video/%08d.png' % (tmp_dir, frame))
            if mode == 1:
                canvas = image
            else:
                canvas = PIL.Image.new('RGB', (2 * src_size, src_size))
                canvas.paste(src_image, (0, 0))
                canvas.paste(image, (src_size, 0))
            if size != src_size:
                canvas = canvas.resize((mode * size, size), PIL.Image.LANCZOS)
            return np.array(canvas)

        src_image = PIL.Image.open(src_file)
        src_size = src_image.size[1]
        duration = num_frames / fps
        filename = os.path.join(dst_dir, os.path.basename(src_file)[:-4] + '.mp4')
        video_clip = moviepy.editor.VideoClip(render_frame, duration=duration)
        video_clip.write_videofile(filename, fps=fps, codec=codec, bitrate=bitrate)


    # dvschultz
    @staticmethod
    def create_mp4(
            src_dir, 
            dest_dir, 
            steppattern="*step*.png",
            tgtpattern="*target*.png",
            moviename = "movie.mp4",
        ):

        # Create video 
        import glob

        imgspattern = os.path.join(src_dir, steppattern)
        imgs = sorted(glob.glob(imgspattern))

        targetpattern = os.path.join(src_dir, tgtpattern)
        target_imgs = sorted(glob.glob(targetpattern))
        assert len(target_imgs) == 1, "More than one target found?"
        target_img = imageio.imread(target_imgs[0])
        
        moviepath = os.path.join(dest_dir, moviename)
        with imageio.get_writer(moviepath, mode='I') as writer:
            for filename in imgs:
                image = imageio.imread(filename)

                # Concatenate images with original target image
                w,h = image.shape[0:2]
                canvas = PIL.Image.new('RGBA', (w*2,h), 'white')
                canvas.paste(Image.fromarray(target_img), (0, 0))
                canvas.paste(Image.fromarray(image), (w, 0))

                writer.append_data(np.array(canvas))

    @staticmethod
    def vid_to_file(
            frame, 
            output_img, 
            video_output_dir= './',
            frame_content_frmt='frame{}.jpg',
            zfill = 4,
        ):
        print(f"|---> vid_to_file frame {frame}")
        fn = frame_content_frmt.format(str(frame).zfill(zfill))
        path = os.path.join(video_output_dir, fn)
        save_cv2(path, output_img)

    @staticmethod
    def frame_cv2_rgb_vgg(frame, frames_dir, args):
        frame_content_frmt = args.frame_content_frmt
        max_size = args.max_size
        zfill = args.zfill
        frame_name = args.frame_content_frmt.format(str(frame).zfill(zfill))
        frame_img_path = os.path.join(frames_dir, frame_name)
        print("frame_img_path", frame_img_path)
        img = cv2.imread(frame_img_path, cv2.IMREAD_COLOR)
        img = onvgg.vgg_preprocess(img)
        # img = onvgg.vgg_deprocess(img)
        # cv2.imshow('img', img)
        # cv2.waitKey(2000)    
        return img

    @staticmethod
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

    
    @staticmethod
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

#   ******************
#    Ondata
#   ******************
class Ondata:

    @staticmethod
    def pil_scale(img, scale=1):
        img.thumbnail((img.size[0]*scale, img.size[1]*scale), Image.LANCZOS) 
        return img

    @staticmethod
    def pil_resize(img, ps=(-1, -1)):
        img = img.resize((
            ps[0] if ps[0] > 0 else img.size[0],
            ps[1] if ps[1] > 0 else img.size[1]
            ), Image.ANTIALIAS)
        return img

    @staticmethod
    def resize_tf_hwc(image, width, height):
        # Resized images will be distorted if their original aspect ratio is not the same as size. 
        # To avoid distortions see tf.image.resize_with_pad
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, (width, height))
        return image

    @staticmethod
    def resize_tf_bhwc(image, width, height):
        Ondata.resize_tf_hwc(image, width, height)
    
    @staticmethod
    def crop_image(img,tol=0):
        # https://www.youtube.com/watch?v=weRmcRzMfUQ
        # https://codereview.stackexchange.com/questions/132914/crop-black-border-of-image-using-numpy
        # https://www.youtube.com/user/bustbright/videos
        # img is 2D or 3D image data
        # tol  is tolerance
        mask = img>tol
        if img.ndim==3:
            mask = mask.all(2)
        mask0,mask1 = mask.any(0),mask.any(1)
        return img[np.ix_(mask0,mask1)]

    @staticmethod
    def crop_img_only_outside(img,tol=0):
        # img is 2D or 3D image data
        # tol  is tolerance
        mask = img>tol
        if img.ndim==3:
            mask = mask.all(2)
        m,n = mask.shape
        mask0,mask1 = mask.any(0),mask.any(1)
        col_start,col_end = mask0.argmax(),n-mask0[::-1].argmax()
        row_start,row_end = mask1.argmax(),m-mask1[::-1].argmax()
        return img[row_start:row_end,col_start:col_end]

    @staticmethod
    def random_crop_with_tf(img, h, w):
        stacked_image = tf.stack([img], axis=0)
        cropped_image = tf.image.random_crop(stacked_image, size=[1, h, w, 3])

        return cropped_image[0]

    @staticmethod
    def resize_with_tf(img, height, width):
        img = tf.image.resize(img, [height, width],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return img

    @staticmethod
    def img_to_sqr_pils(img, qslices=1, eps = 0.5, clip = (512, 512)):
        newimgs = []
        img_data = np.array(list(img.getdata())).reshape( (img.size[1],img.size[0],-1) ) # y,x
        for n in range(qslices):
            rx = int(( ( img.size[0]-clip[0] ) / 2 ) + eps * np.random.randint( ( img.size[0]-clip[0] ) / 2 )) if img.size[0]> clip[0] + 2 else 0
            ry = int(( ( img.size[1]-clip[1] ) / 2 ) + eps * np.random.randint( ( img.size[1]-clip[0] ) / 2 )) if img.size[1]> clip[1] + 2 else 0

            print(f"|... img_to_sqr_pils tile {n}: {ry}, {ry+clip[1]}, {rx}, {rx+clip[0]}")

            sub = np.copy(img_data[ry:ry+clip[1], rx:rx+clip[0]]).astype(np.uint8) # y,x
            sub = sub[:,:,:3]

            # append
            newimg = Image.fromarray(sub)
            newimgs.append(newimg)
        return newimgs

    @staticmethod
    def img_to_tiled_pils(img, tile = (512, 512)):

        img_data = np.asarray(img.convert("RGB")).reshape( (img.size[1],img.size[0],-1) )
        shape = np.shape(img_data)

        hb,wb  = shape[0], shape[1]
        hb2,wb2 = int(hb/2), int(wb/2)
        hs, ws = tile[0], tile[1]
        hs2, ws2 = int(hs/2), int(ws/2)
        hr, wr = ((hb2-hs2))/hs, ((wb2-ws2))/ws
        hn, wn = math.floor(hr), math.floor(wr)

        newimgs=[]
        if wn >= 0 and hn >= 0:
            rx = wb2 + (0 - 1) * ws2
            ry = hb2 + (0 - 1) * hs2

            sub = np.copy(img_data[ry:ry+hs, rx:rx+ws]).astype(np.uint8) # y,x
            newimg = Image.fromarray(sub[:,:,:3])
            if 0:
                print(f"|... img_to_tiled_pils r0 ==> {0}, {0}: {[ry, rx]}: {np.shape(newimg)}")
            newimgs.append(newimg)

        for i in range(wn+1):
            for j in range(hn+1):
                wi = i
                hi = j
                if wi > 0:

                    # -x-|---
                    rx = wb2 + (-wi - 0) * ws2
                    ry = hb2 + (+hi - 0) * hs2

                    sub = np.copy(img_data[
                                ry:ry+hs, 
                                rx:rx+ws
                                ]).astype(np.uint8) # y,x
                    newimg = Image.fromarray(sub[:,:,:3])
                    # pil_show_rgb(newimg)
                    newimgs.append(newimg)
                    if 0:
                        print(f"|... img_to_tiled_pils r2 ==> {wi}, {hi}: {[ry, rx]}: {np.shape(newimg)}")

                    # ---|-x-
                    rx = wb2 + (+wi - 0) * ws2
                    ry = hb2 + (+hi - 0) * hs2

                    sub = np.copy(img_data[
                                ry:ry+hs, 
                                rx:rx+ws
                                ]).astype(np.uint8) # y,x
                    newimg = Image.fromarray(sub[:,:,:3])
                    # pil_show_rgb(newimg)
                    newimgs.append(newimg)
                    if 0:
                        print(f"|... img_to_tiled_pils r2 ==> {wi}, {hi}: {[ry, rx]}: {np.shape(newimg)}")

                if hi > 0:
                    #  ---^
                    rx = wb2 + (+wi - 0) * ws2
                    ry = hb2 + (-hi - 0) * hs2 

                    sub = np.copy(img_data[
                                ry:ry+hs, 
                                rx:rx+ws
                                ]).astype(np.uint8) # y,x
                    newimg = Image.fromarray(sub[:,:,:3])
                    # pil_show_rgb(newimg)
                    if 0:
                        print(f" r3 ==> {wi}, {hi}: {[ry, rx]}: {np.shape(newimg)}")
                    newimgs.append(newimg)

                    #  ---_
                    rx = wb2 + (+wi - 0) * ws2
                    ry = hb2 + (+hi - 0) * hs2 

                    sub = np.copy(img_data[
                                ry:ry+hs, 
                                rx:rx+ws
                                ]).astype(np.uint8) # y,x
                    newimg = Image.fromarray(sub[:,:,:3])
                    # pil_show_rgb(newimg)
                    if 0:
                        print(f"|... img_to_tiled_pils r3 ==> {wi}, {hi}: {[ry, rx]}: {np.shape(newimg)}")
                    newimgs.append(newimg)

                if wi > 0 and hi > 0:
                    rx = wb2 + (-wi - 0) * ws2
                    ry = hb2 + (-hi - 0) * hs2 

                    sub = np.copy(img_data[
                                ry:ry+hs, 
                                rx:rx+ws
                                ]).astype(np.uint8) # y,x
                    newimg = Image.fromarray(sub[:,:,:3])
                    if 0:
                        print(f"|... img_to_tiled_pils r4 ==> {wi}, {hi}: {[ry, rx]}: {np.shape(newimg)}")
                    # pil_show_rgb(newimg)
                    newimgs.append(newimg)

        print(f"|... img_to_tiled_pils new imgs q: {len(newimgs)}")
        return newimgs

    @staticmethod
    def random_jitter(img, throughsize=(286, 286), tosize=(256, 256)):
        # resizing to 286 x 286 x 3
        img = Ondata.resize_with_tf(img, throughsize[0], throughsize[1])

        # randomly cropping to 256 x 256 x 3
        img = Ondata.random_crop_with_tf(img, tosize[0], tosize[1])

        if np.random.uniform(()) > 0.5: # tf _e_
            # random mirroring
            img = tf.image.flip_left_right(img)
            real_image = tf.image.flip_left_right(real_image)
        
        return img

    @staticmethod
    def folder_to_formed_rgbs(
        src_dir,
        qimgs = -1,          # number of images to get from src
        scale = 1,          # will scale by scale
        ps = (-1, -1),      # will resize images if sizes in N
        qslices = 1,         # will generte slices per images
        eps = 0.0,          # will deviate from the center of th image
        clip = (512, 512),  # will generate slices of clip size   
    ):
        imgs = Ondata.folder_to_formed_pils(src_dir,qimgs,scale,ps,qslices,eps,clip)
        arrs = [np.array(img) for img in imgs]
        return arrs

    @staticmethod
    def folder_to_formed_nuas(
        src_dir,
        qimgs = -1,          # number of images to get from src
        scale = 1,          # will scale by scale
        ps = (-1, -1),      # will resize images if sizes in N
        qslices = 1,         # will generte slices per images
        eps = 0.0,          # will deviate from the center of th image
        clip = (512, 512),  # will generate slices of clip size   
    ):
        rgbs = Ondata.folder_to_formed_rgbs(src_dir,qimgs,scale,ps,qslices,eps,clip)
        narrs = [rgb / 255 for rgb in rgbs]
        return narrs

    @staticmethod
    def folder_to_formed_pils(
        src_dir,
        qimgs = -1,         # number of images to get from src
        scale = 1,          # will scale by scale
        ps = (-1, -1),      # will resize images if sizes in N
        qslices = 1,        # will generte slices per images
        eps = 0.0,          # will deviate from the center of th image
        clip = (512, 512),  # will generate slices of clip size   

    ):
        assert os.path.exists(src_dir), "src_dir does not exist"
        newimgs = []
        items = Onfile.folder_to_pils(src_dir, qimgs)
        for i,img in enumerate(items):
            img = Ondata.pil_scale(img, scale)
            print(f"|... folder_to_formed_pils after-scale img shape: {np.shape(img)}")
            img = Ondata.pil_resize(img, ps)
            print(f"|... folder_to_formed_pils alter-resize img shape: {np.shape(img)}")
            tiledimgs = Ondata.img_to_sqr_pils(img,qslices,eps,clip)
            newimgs += tiledimgs
        return newimgs

    @staticmethod
    def path_to_formed_pair(path, dim=512, do_crop=False, canny_thresh1=100, canny_thresh2=200):
        # https://github.com/memo/webcam-pix2pix-tensorflow/blob/master/preprocess.py

        print(f'|... path_to_formed_pair {path}')

        out_shape = (dim, dim)

        im = PIL.Image.open(path)
        im = im.convert('RGB')
        if do_crop:
            resize_shape = list(out_shape)
            if im.width < im.height:
                resize_shape[1] = int(round(float(im.height) / im.width * dim))
            else:
                resize_shape[0] = int(round(float(im.width) / im.height * dim))
            im = im.resize(resize_shape, PIL.Image.BICUBIC)
            hw = int(im.width / 2)
            hh = int(im.height / 2)
            hd = int(dim/2)
            area = (hw-hd, hh-hd, hw+hd, hh+hd)
            im = im.crop(area)            
                
        else:
            im = im.resize(out_shape, PIL.Image.BICUBIC)
            
        a1 = np.array(im) 
        a2 = cv2.Canny(a1, canny_thresh1, canny_thresh2)
        a2 = cv2.cvtColor(a2, cv2.COLOR_GRAY2RGB)                 
        a3 = np.concatenate((a1,a2), axis=1)
        im = PIL.Image.fromarray(a3)                     
                        
        # im.save(os.path.join(out_path, out_fname))
        # print(f"save {os.path.join(out_path, out_fname)})
        return im


    @staticmethod
    def path_to_formed_pil(path, dim=None, do_crop=False):
        # https://github.com/memo/webcam-pix2pix-tensorflow/blob/master/preprocess.py

        if 0: print(f'|... path_to_formed_pil {path}')


        im = PIL.Image.open(path)
        im = im.convert('RGB')
        if do_crop:
            assert dim, "dim must be defined"
            resize_shape = list((dim, dim))
            if im.width < im.height:
                resize_shape[1] = int(round(float(im.height) / im.width * dim))
            else:
                resize_shape[0] = int(round(float(im.width) / im.height * dim))
            im = im.resize(resize_shape, PIL.Image.BICUBIC)
            hw = int(im.width / 2)
            hh = int(im.height / 2)
            hd = int(dim/2)
            area = (hw-hd, hh-hd, hw+hd, hh+hd)
            im = im.crop(area)            
                
        elif dim:
            out_shape = (dim, dim)
            im = im.resize(out_shape, PIL.Image.BICUBIC)
            
        return im

    @staticmethod
    def folder_to_formed_data(
        src_dir,
        dest_dir,
        qimgs = -1,         # number of images to get from src
        scale = 1,          # will scale by scale
        ps = (-1, -1),      # will resize images if sizes in N
        qslices = 2,        # will generte slices per images
        eps = 0.5,          # will deviate from the center of th image
        clip = (512, 512),  # will generate slices of clip size   
        img_format = 'img{}.jpg', zfill=4,
        verbose=False
    ):
        assert os.path.exists(src_dir), "src_dir does not exist"
        newimgs = []
        imgs = []
        jpgs = glob.glob(os.path.join(src_dir, "*.jpg"))
        pngs = glob.glob(os.path.join(src_dir, "*.png"))
        fs = jpgs + pngs  # <class 'list'>
        newimgidx = 0
        for i,f in enumerate(fs):
            if (qimgs<0 or i<qimgs):
                img = Image.open(f)
                img = Ondata.pil_scale(img, scale)
                img = Ondata.pil_resize(img, ps)
                if verbose:
                    print(f"folder_to_formed_pils alter-resize img shape: {np.shape(img)}")
                tiledimgs = Ondata.img_to_sqr_pils(img,qslices,eps,clip)
                for newimg in tiledimgs:
                    img_name = img_format.format(str(newimgidx).zfill(zfill))                
                    outpath = os.path.join(dest_dir, img_name)
                    if verbose:
                        print(f"save image {newimgidx} to {outpath}")
                    newimg.save(outpath, 'JPEG', quality=90) 
                    newimgidx = newimgidx + 1

    @staticmethod
    def downsample(filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                kernel_initializer=initializer, use_bias=False))

        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())

        result.add(tf.keras.layers.LeakyReLU())
        return result

    @staticmethod
    def upsample(filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2DTranspose(filters, 
                size, 
                strides=2,
                padding='same',
                kernel_initializer=initializer,
                use_bias=False))

        result.add(tf.keras.layers.BatchNormalization())

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))

        result.add(tf.keras.layers.ReLU())
        return result

    @staticmethod
    def rgbs_to_nuas_batch(imgs, num, flip = True):
        idx = np.random.randint(0, imgs.shape[0] - 1, num)
        out = []

        for i in idx:
            out.append(imgs[i])
            if flip and np.random.uniform(()) < 0.5:  # flip as arg
                out[-1] = np.flip(out[-1], 1)

        return np.array(out).astype('float32') / 255.0


#   *******************
#   ONSET
#
#   *******************
class Onset:
# https://github.com/dvschultz/dataset-tools
# dataset-tools/dataset-tools.py
    @staticmethod
    def image_resize(image, width = None, height = None, max = None):
        inter = cv2.INTER_CUBIC

        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        if max is not None:
            if w > h:
                # produce
                r = max / float(w)
                dim = (max, int(h * r))
            elif h > w:
                r = max / float(h)
                dim = (int(w * r), max)
            else :
                dim = (max, max)

        else: 
            # if both the width and height are None, then return the
            # original image
            if width is None and height is None:
                return image

            # check to see if the width is None
            if width is None:
                # calculate the ratio of the height and construct the
                # dimensions
                r = height / float(h)
                dim = (int(w * r), height)

            # otherwise, the height is None
            else:
                # calculate the ratio of the width and construct the
                # dimensions
                r = width / float(w)
                dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(image, dim, interpolation = inter)

        # return the resized image
        return resized

    @staticmethod
    def image_scale(image, scalar = 1.0):
        (h, w) = image.shape[:2]
        dim = (int(w*scalar),int(h*scalar))
        # resize the image
        resized = cv2.resize(image, dim, interpolation = inter)
        
        # return the resized image
        return resized

    @staticmethod
    def arbitrary_crop(img, h_crop,w_crop, args=None):
        error = False
        bType = cv2.BORDER_REPLICATE
        if(args.border_type == 'solid'):
            bType = cv2.BORDER_CONSTANT
        elif (args.border_type == 'reflect'):
            bType = cv2.BORDER_REFLECT

        (h, w) = img.shape[:2]
        if(h>h_crop):
            hdiff = int((h-h_crop)/2) + args.shift_y

            if( ((hdiff+h_crop) > h) or (hdiff < 0)):
                print("error! crop settings are too much for this image")
                error = True
            else:
                img = img[hdiff:hdiff+h_crop,0:w]
        if(w>w_crop):
            wdiff = int((w-w_crop)/2) + args.shift_x
            
            if( ((wdiff+w_crop) > w) or (wdiff < 0) ):
                print("error! crop settings are too much for this image")
                error = True
            else:
                img = img[0:h_crop,wdiff:wdiff+w_crop]
        return img, error

    @staticmethod
    def crop_to_square(img, args=None):
        (h, w) = img.shape[:2]
        
        cropped = img.copy()
        if w > h:	
            if (args.h_align=='left'):
                print('here first')
                cropped = img[:h,:h]
            elif (args.h_align=='right'):
                cropped = img[0:h, w-h:w]
            else:
                diff = int((w-h)/2)
                cropped = img[0:h, diff:diff+h]
        elif h > w:
            if (args.v_align=='top'):
                cropped = img[:w, :w]
            elif (args.v_align=='bottom'):
                cropped = img[h-w:h, 0:w]
            else:
                diff = int((h-w)/2)
                cropped = img[diff:diff+w, 0:w]
            
        return cropped

    @staticmethod
    def crop_square_patch(img, imgSize):
        (h, w) = img.shape[:2]

        rH = random.randint(0,h-imgSize)
        rW = random.randint(0,w-imgSize)
        cropped = img[rH:rH+imgSize,rW:rW+imgSize]

        return cropped

    @staticmethod
    def processCanny(img, args=None):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if(args.blur_type=='gaussian'):
            gray = cv2.GaussianBlur(gray, (args.blur_amount, args.blur_amount), 0)
        elif(args.blur_type=='median'):
            gray = cv2.medianBlur(gray,args.blur_amount)
        gray = cv2.Canny(gray,100,300)

        return gray

    @staticmethod
    def makeResize(img,filename,scale, args=None):
        remakePath = args.output_folder + str(scale)+"/"
        if(args.keep_folder==True): remakePath = args.output_folder

        if not os.path.exists(remakePath):
            os.makedirs(remakePath)

        img_copy = img.copy()
        img_copy = Onset.image_resize(img_copy, max = scale)

        if(args.file_extension == "png"):
            new_file = os.path.splitext(filename)[0] + ".png"
            cv2.imwrite(os.path.join(remakePath, new_file), img_copy, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        elif(args.file_extension == "jpg"):
            new_file = os.path.splitext(filename)[0] + ".jpg"
            cv2.imwrite(os.path.join(remakePath, new_file), img_copy, [cv2.IMWRITE_JPEG_QUALITY, 90])

        if (args.mirror): Onset.flipImage(img_copy,new_file,remakePath, args)
        if (args.rotate): Onset.rotateImage(img_copy,new_file,remakePath, args)

    @staticmethod
    def makeDistance(img,filename,scale, args=None):
        makePath = args.output_folder + "distance-"+ str(args.max_size)+"/"
        if(args.keep_folder==True): makePath = args.output_folder

        if not os.path.exists(makePath):
            os.makedirs(makePath)

        img_copy = img.copy()
        img_copy = Onset.image_resize(img_copy, max = scale)

        BW = img_copy[:,:,0] > 127
        G_channel = pyimg.distance_transform_edt(BW)
        G_channel[G_channel>32]=32
        B_channel = pyimg.distance_transform_edt(1-BW)
        B_channel[B_channel>200]=200
        img_copy[:,:,1] = G_channel.astype('uint8')
        img_copy[:,:,0] = B_channel.astype('uint8')

        if(args.file_extension == "png"):
            new_file = os.path.splitext(filename)[0] + ".png"
            cv2.imwrite(os.path.join(makePath, new_file), img_copy, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        elif(args.file_extension == "jpg"):
            new_file = os.path.splitext(filename)[0] + ".jpg"
            cv2.imwrite(os.path.join(makePath, new_file), img_copy, [cv2.IMWRITE_JPEG_QUALITY, 90])

        if (args.mirror): Onset.flipImage(img_copy,new_file,makePath)
        if (args.rotate): Onset.rotateImage(img_copy,new_file,makePath)

    @staticmethod
    def makeScale(img,filename,scale, args=None):

        remakePath = args.output_folder + "scale_"+str(scale)+"/"
        if(args.keep_folder==True): remakePath = args.output_folder

        if not os.path.exists(remakePath):
            os.makedirs(remakePath)

        img_copy = img.copy()
        
        img_copy = image_scale(img_copy, scale)

        new_file = os.path.splitext(filename)[0] + ".png"
        cv2.imwrite(os.path.join(remakePath, new_file), img_copy, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        if (args.mirror): Onset.flipImage(img_copy,new_file,remakePath)
        if (args.rotate): Onset.rotateImage(img_copy,new_file,remakePath)

    @staticmethod
    def makeSquare(img,filename,scale, args=None):
        sqPath = args.output_folder + "sq-"+str(scale)+"/"
        if(args.keep_folder==True): sqPath = args.output_folder

        if not os.path.exists(sqPath):
            os.makedirs(sqPath)

        bType = cv2.BORDER_REPLICATE
        if(args.border_type == 'solid'):
            bType = cv2.BORDER_CONSTANT
        elif (args.border_type == 'reflect'):
            bType = cv2.BORDER_REFLECT
        img_sq = img.copy()
        (h, w) = img_sq.shape[:2]
        if((h < scale) and (w < scale)):
            if args.verbose > 1: print('skip resize')
        else:
            img_sq = Onset.image_resize(img_sq, max = scale)

        bColor = [int(item) for item in args.border_color.split(',')]

        (h, w) = img_sq.shape[:2]
        if(h > w):
            # pad left/right
            diff = h-w
            if(diff%2 == 0):
                img_sq = cv2.copyMakeBorder(img_sq, 0, 0, int(diff/2), int(diff/2), bType,value=bColor)
            else:
                img_sq = cv2.copyMakeBorder(img_sq, 0, 0, int(diff/2)+1, int(diff/2), bType,value=bColor)
        elif(w > h):
            # pad top/bottom
            diff = w-h
            if(diff%2 == 0):
                img_sq = cv2.copyMakeBorder(img_sq, int(diff/2), int(diff/2), 0, 0, bType,value=bColor)
            else:
                img_sq = cv2.copyMakeBorder(img_sq, int(diff/2), int(diff/2)+1, 0, 0, bType,value=bColor)
        else:
            diff = scale-h
            if(diff%2 == 0):
                img_sq = cv2.copyMakeBorder(img_sq, int(diff/2), int(diff/2), int(diff/2), int(diff/2), bType,value=bColor)
            else:
                img_sq = cv2.copyMakeBorder(img_sq, int(diff/2), int(diff/2)+1, int(diff/2), int(diff/2)+1, bType,value=bColor)

        if(args.file_extension == "png"):
            new_file = os.path.splitext(filename)[0] + ".png"
            cv2.imwrite(os.path.join(sqPath, new_file), img_sq, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        elif(args.file_extension == "jpg"):
            new_file = os.path.splitext(filename)[0] + ".jpg"
            cv2.imwrite(os.path.join(sqPath, new_file), img_sq, [cv2.IMWRITE_JPEG_QUALITY, 90])

        if (args.mirror): Onset.flipImage(img_sq,new_file,sqPath,args)
        if (args.rotate): Onset.rotateImage(img_sq,new_file,sqPath,args)

    @staticmethod        
    def makeCanny(img,filename,scale, args=None):
        make_path = args.output_folder + "canny-"+str(scale)+"/"
        if(args.keep_folder==True): make_path = args.output_folder

        if not os.path.exists(make_path):
            os.makedirs(make_path)

        img_copy = img.copy()
        img_copy = Onset.image_resize(img_copy, max = scale)
        gray = Onset.processCanny(img_copy, args)

        # save out
        if(args.file_extension == "png"):
            new_file = os.path.splitext(filename)[0] + ".png"
            cv2.imwrite(os.path.join(make_path, new_file), gray, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        elif(args.file_extension == "jpg"):
            new_file = os.path.splitext(filename)[0] + ".jpg"
            cv2.imwrite(os.path.join(make_path, new_file), gray, [cv2.IMWRITE_JPEG_QUALITY, 90])

        if (args.mirror): Onset.flipImage(img_copy,new_file,make_path,args)
        if (args.rotate): Onset.rotateImage(img_copy,new_file,make_path,args)

    @staticmethod
    def makeCrop(img,filename, args=None):
        make_path = args.output_folder + "crop-"+str(args.height)+"x"+str(args.width)+"/"
        if(args.keep_folder==True): make_path = args.output_folder
            
        if not os.path.exists(make_path):
            os.makedirs(make_path)

        img_copy = img.copy()
        img_copy,error = arbitrary_crop(img_copy,args.height,args.width)

        if (error==False):
            if(args.file_extension == "png"):
                new_file = os.path.splitext(filename)[0] + ".png"
                cv2.imwrite(os.path.join(make_path, new_file), img_copy, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            elif(args.file_extension == "jpg"):
                new_file = os.path.splitext(filename)[0] + ".jpg"
                cv2.imwrite(os.path.join(make_path, new_file), img_copy, [cv2.IMWRITE_JPEG_QUALITY, 90])

            if (args.mirror): Onset.flipImage(img_copy,new_file,make_path,args)
            if (args.rotate): Onset.rotateImage(img_copy,new_file,make_path,args)
        else:
            if(args.verbose): print(filename+" returned an error")

    @staticmethod
    def makeSquareCrop(img,filename,scale, args=None):
        make_path = args.output_folder + "sq-"+str(scale)+"/"
        if(args.keep_folder==True): make_path = args.output_folder    

        if not os.path.exists(make_path):
            os.makedirs(make_path)

        img_copy = img.copy()
        img_copy = Onset.crop_to_square(img_copy, args)
        img_copy = Onset.image_resize(img_copy, max = scale)

        if(args.file_extension == "png"):
            new_file = os.path.splitext(filename)[0] + ".png"
            cv2.imwrite(os.path.join(make_path, new_file), img_copy, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        elif(args.file_extension == "jpg"):
            new_file = os.path.splitext(filename)[0] + ".jpg"
            cv2.imwrite(os.path.join(make_path, new_file), img_copy, [cv2.IMWRITE_JPEG_QUALITY, 90])

        if (args.mirror): Onset.flipImage(img_copy,new_file,make_path,args)
        if (args.rotate): Onset.rotateImage(img_copy,new_file,make_path,args)

    @staticmethod
    def makeManySquares(img,filename,scale,args=None):
        make_path = args.output_folder + "many_squares-"+str(scale)+"/"
        if(args.keep_folder==True): make_path = args.output_folder    

        if not os.path.exists(make_path):
            os.makedirs(make_path)

        img_copy = img.copy()
        (h, w) = img_copy.shape[:2]
        img_ratio = h/w

        if(img_ratio >= 1.25):

            #crop images from top and bottom
            crop = img_copy[0:w,0:w]
            crop = Onset.image_resize(crop, max = scale)
            new_file = os.path.splitext(filename)[0] + "-1.png"
            cv2.imwrite(os.path.join(make_path, new_file), crop, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            if (args.mirror): Onset.flipImage(crop,new_file,make_path,args)
            if (args.rotate): Onset.rotateImage(crop,filename,make_path,args)

            crop = img_copy[h-w:h,0:w]
            crop = Onset.image_resize(crop, max = scale)
            new_file = os.path.splitext(filename)[0] + "-2.png"
            cv2.imwrite(os.path.join(make_path, new_file), crop, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            if (args.mirror): Onset.flipImage(crop,new_file,make_path,args)
            if (args.rotate): Onset.rotateImage(crop,filename,make_path,args)

        elif(img_ratio <= .8):
            #crop images from left and right
            print(os.path.splitext(filename)[0] + ': wide image')

            crop = img_copy[0:h,0:h]
            crop = Onset.image_resize(crop, max = scale)
            new_file = os.path.splitext(filename)[0] + "-wide1.png"
            cv2.imwrite(os.path.join(make_path, new_file), crop, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            if (args.mirror): Onset.flipImage(crop,new_file,make_path)
            if (args.rotate): Onset.rotateImage(crop,filename,make_path)

            crop = img_copy[0:h,w-h:w]
            crop = Onset.image_resize(crop, max = scale)
            new_file = os.path.splitext(filename)[0] + "-wide2.png"
            cv2.imwrite(os.path.join(make_path, new_file), crop, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            if (args.mirror): Onset.flipImage(crop,new_file,make_path)
            if (args.rotate): Onset.rotateImage(crop,filename,make_path)

        else:
            img_copy = Onset.crop_to_square(img_copy, args)
            img_copy = Onset.image_resize(img_copy, max = scale)
            new_file = os.path.splitext(filename)[0] + ".png"
            cv2.imwrite(os.path.join(make_path, new_file), img_copy, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            if (args.mirror): Onset.flipImage(img_copy,new_file,make_path)
            if(args.rotate): Onset.rotateImage(img_copy,filename,make_path)
            
    @staticmethod
    def makeSquareCropPatch(img,filename,scale,args=None):
        make_path = args.output_folder + "sq-"+str(scale)+"/"
        if(args.keep_folder==True): make_path = args.output_folder    

        if not os.path.exists(make_path):
            os.makedirs(make_path)

        img_copy = img.copy()
        img_copy = crop_square_patch(img_copy,args.max_size)

        new_file = os.path.splitext(filename)[0] + ".png"
        cv2.imwrite(os.path.join(make_path, new_file), img_copy, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        if (args.mirror): Onset.flipImage(img_copy,new_file,make_path,args)
        if (args.rotate): Onset.rotateImage(img_copy,new_file,make_path,args)

    @staticmethod
    def makePix2Pix(img,filename,scale,direction="BtoA",value=[0,0,0],args=None):
        img_p2p = img.copy()
        img_p2p = Onset.image_resize(img_p2p, max = scale)
        (h, w) = img_p2p.shape[:2]
        bType = cv2.BORDER_CONSTANT
        
        make_path = args.output_folder + "pix2pix-"+str(h)+"/"
        if(args.keep_folder==True): make_path = args.output_folder    

        if not os.path.exists(make_path):
            os.makedirs(make_path)

        canny = cv2.cvtColor(Onset.processCanny(img_p2p,args),cv2.COLOR_GRAY2RGB)
        
        if(direction=="BtoA"):
            img_p2p = cv2.copyMakeBorder(img_p2p, 0, 0, w, 0, bType, None, value)
            img_p2p[0:h,0:w] = canny
        
        if(args.file_extension == "png"):
            new_file = os.path.splitext(filename)[0] + ".png"
            cv2.imwrite(os.path.join(make_path, new_file), img_p2p, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        elif(args.file_extension == "jpg"):
            new_file = os.path.splitext(filename)[0] + ".jpg"
            cv2.imwrite(os.path.join(make_path, new_file), img_p2p, [cv2.IMWRITE_JPEG_QUALITY, 90])

    @staticmethod
    def flipImage(img,filename,path,args=None):
        flip_img = cv2.flip(img, 1)
        flip_file = os.path.splitext(filename)[0] + "-flipped.png"
        cv2.imwrite(os.path.join(path, flip_file), flip_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    @staticmethod
    def rotateImage(img,filename,path, args=None):
        r = img.copy() 
        r = imutils.rotate_bound(r, 90)

        if(args.file_extension == "png"):
            r_file = os.path.splitext(filename)[0] + "-rot90.png"
            cv2.imwrite(os.path.join(path, r_file), r, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        elif(args.file_extension == "jpg"):
            r_file = os.path.splitext(filename)[0] + "-rot90.jpg"
            cv2.imwrite(os.path.join(path, r_file), r, [cv2.IMWRITE_JPEG_QUALITY, 90])

        r = imutils.rotate_bound(r, 90)
        if(args.file_extension == "png"):
            r_file = os.path.splitext(filename)[0] + "-rot180.png"
            cv2.imwrite(os.path.join(path, r_file), r, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        elif(args.file_extension == "jpg"):
            r_file = os.path.splitext(filename)[0] + "-rot180.jpg"
            cv2.imwrite(os.path.join(path, r_file), r, [cv2.IMWRITE_JPEG_QUALITY, 90])

        r = imutils.rotate_bound(r, 90)
        if(args.file_extension == "png"):
            r_file = os.path.splitext(filename)[0] + "-rot270.png"
            cv2.imwrite(os.path.join(path, r_file), r, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        elif(args.file_extension == "jpg"):
            r_file = os.path.splitext(filename)[0] + "-rot270.jpg"
            cv2.imwrite(os.path.join(path, r_file), r, [cv2.IMWRITE_JPEG_QUALITY, 90])

    @staticmethod
    def processImage(img,filename, args=None):

        if args.process_type == "resize":	
            Onset.makeResize(img,filename,args.max_size, args)
        if args.process_type == "resize_pad":	
            Onset.makeResizePad(img,filename,args.max_size, args)
        if args.process_type == "square":
            Onset.makeSquare(img,filename,args.max_size, args)
        if args.process_type == "crop_to_square":
            Onset.makeSquareCrop(img,filename,args.max_size, args)
        if args.process_type == "canny":
            Onset.makeCanny(img,filename,args.max_size, args)
        if args.process_type == "canny-pix2pix":
            Onset.makePix2Pix(img,filename,args.max_size, args=args)
        if args.process_type == "crop_square_patch":
            Onset.makeSquareCropPatch(img,filename,args.max_size, args)
        if args.process_type == "scale":
            Onset.makeScale(img,filename,args.scale, args)
        if args.process_type == "many_squares":
            Onset.makeManySquares(img,filename,args.max_size, args)
        if args.process_type == "crop":
            Onset.makeCrop(img,filename, args)
        if args.process_type == "distance":
            Onset.makeDistance(img,filename,args.max_size, args)


    @staticmethod
    def processFolder(args=None):
        count = int(0)
        inter = cv2.INTER_CUBIC

        ''' filter files '''
        patt = re.compile(".*")
        try:
            if args.filepatt:
                patt = re.compile(args.filepatt)	# file name pattern
        except:
            pass


        for root, subdirs, files in os.walk(args.input_folder):
            if args.verbose > 0: print(f'processFolder {root}')

            for subdir in subdirs:
                if(args.verbose): print('\t- subdirectory ' + subdir)

            exclude = []
            try:
                exclude = args.exclude
            except:
                pass
            if 0: print(f"|.... exclude {exclude}")

            for filename in files:
                if patt.match(filename) and not filename in exclude: # file name pattern mach
                    if 0: print(f"|.... filename {filename} not in exclude")
                    file_path = os.path.join(root, filename)
                    if args.verbose > 0: print('\t- file %s (full path: %s)' % (filename, file_path))
                    
                    img = cv2.imread(file_path)

                    if hasattr(img, 'copy'):
                        if args.name:
                            if args.verbose > 1: 
                                print(f'|... processing image name: {filename}')
                            Onset.processImage(img,filename, args)
                        else:
                            if args.verbose > 1: 
                                print(f'|... processing image idx: {str(count)}')
                            zfill=0
                            try:
                                zfill = args.zfill
                            except:
                                pass
                            Onset.processImage(img,str(count).zfill(zfill), args)
                        count = count + int(1)
                else:
                    if 0: print(f"|.... filename {filename} EXCLUDED")

    # https://github.com/dvschultz/dataset-tools/dedupe.py

        # input_folder = './input/'
        # output_folder = './output/'
        # process_type = 'exclude'
        # file_extension = 'png'
        # avg_match = 1.0

        # absolute = True
        # relative = True

    @staticmethod
    def compare(img1,img2, args=None):
        test = False
        difference = cv2.absdiff(img1, img2)
        if(args.absolute):	
            return not np.any(difference)
        else:
            return np.divide(np.sum(difference),img1.shape[0]*img1.shape[1]) <= args.avg_match

            #way too greedy
            #return np.allclose(img1,img2,2,2)

    @staticmethod
    def exclude(imgs,args=None):
        expath = args.output_folder + "exclude/"
        if not os.path.exists(expath):
            os.makedirs(expath)

        i = 0
        print("avg_match" + str(args.avg_match))
        print("processing...")
        print("total images: " + str(len(imgs)))

        while i < len(imgs):
            img = imgs[i][0]
            filename = imgs[i][1]

            if 0:
                print( f"{str(i)}/{str(len(imgs))} matching to: {filename}")

            i2 = i+1
            while i2 < len(imgs):
                popped = False
                img2 = imgs[i2][0]
                filename2 = imgs[i2][1]

                # print ('comparing '+filename + " to " + filename2)
                if Onset.compare(img,img2, args):
                    print (f">>> {filename} matches {filename2}, pop <<<")
                    popped = True 
                    imgs.pop(i2)
                else:
                    pass
                    # print (f"imgs do not match")

                if not popped:
                    i2 += 1
                else:
                    # copy dup to exclude
                    if(args.file_extension == "png"):
                        new_file = os.path.splitext(filename2)[0] + ".png"
                        cv2.imwrite(os.path.join(expath, new_file), img2, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                    else:
                        new_file = os.path.splitext(filename2)[0] + ".jpg"
                        cv2.imwrite(os.path.join(expath, new_file), img2, [cv2.IMWRITE_JPEG_QUALITY, 90])

            i += 1
        
        return imgs

    @staticmethod
    def sort(imgs):
        #TODO
        print("skip")
        make_path1 = args.output_folder + "yes/"
        make_path2 = args.output_folder + "no/"
        if not os.path.exists(make_path1):
            os.makedirs(make_path1)
        if not os.path.exists(make_path2):
            os.makedirs(make_path2)

        (h, w) = img.shape[:2]
        ratio = h/w

        if(args.exact == True):
            if((ratio >= 1.0) and (h == args.max_size) and (w == args.min_size)):
                path = make_path1
            elif((ratio < 1.0) and (w == args.max_size) and (h == args.min_size)):
                path = make_path1
            else:
                path = make_path2
        else:
            #only works with ratio right now
            if(ratio>=args.min_ratio):
                path = make_path1
            else:
                path = make_path2

        if(args.file_extension == "png"):
            new_file = os.path.splitext(filename)[0] + ".png"
            cv2.imwrite(os.path.join(path, new_file), img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        else:
            new_file = os.path.splitext(filename)[0] + ".jpg"
            cv2.imwrite(os.path.join(path, new_file), img, [cv2.IMWRITE_JPEG_QUALITY, 90])

    @staticmethod
    def dedupe(imgs,filenames, args=None):
        if args.process_type == "exclude":	
            exclude(imgs,filenames)
        if args.process_type == "sort":	
            sort(imgs,filenames)

#   ******************
#   ONRECORD
#
#   ******************
class Onrecord:

    @staticmethod   
    def folder_to_tfrecord_paths(path):
        trf_paths = sorted(glob.glob(os.path.join(path, '*.tfrecords')))
        return trf_paths

    @staticmethod   
    def tfrecord_show(tfrpath):
        dataset = tf.data.TFRecordDataset(tfrpath) # <TFRecordDatasetV2 shapes: (), types: tf.string>
        print("dataset type", type(dataset)) #  <class 'tensorflow.python.

        image_feature_description = {
            'shape': tf.io.FixedLenFeature([3], tf.int64),
            'data': tf.io.FixedLenFeature([], tf.string),
        }

        # Parse the input tf.Example proto using the dictionary above
        # https://www.tensorflow.org/api_docs/python/tf/io/parse_single_example
        def _parse_image_function(example_proto):
            return tf.io.parse_single_example(example_proto, image_feature_description)

        # parse raw dataset
        parsed_dataset = dataset.map(_parse_image_function)
        print("parsed_dataset: ", parsed_dataset)

        for idx, dataitem in enumerate(parsed_dataset):
            print("image record id:", idx)

            tfshape = dataitem['shape']
            print("tfshape type: ", type(tfshape))
            print("tfshape shape: ", tfshape.shape)
            print("shape: ", tfshape) # tf.Tensor([  3 256 256], shape=(3,), dtype=int64)

            shape = tfshape.numpy() # [  3 256 256]
            tfdata = dataitem['data'] # () EagerTensor
            data = tfdata.numpy()
            # print("data type: ", type(data)) # <class 'bytes'>
            # display.display(display.Image(data=data))
            # image = Image.fromarray(data, 'RGB')
            # image = Image.open(StringIO.StringIO(data))

            # https://www.programcreek.com/python/example/89944/PIL.Image.frombytes
            # https://pillow.readthedocs.io/en/3.3.x/handbook/concepts.html#concept-modes
            size = (shape[1], shape[2])
            mode = 'RGB' if shape[0] == 3 else 'RGBA'
            img = np.array(data)

            channels = shape[0]
            width = shape[1]
            height = shape[2]
            bytes_needed = int(width * height * channels)  # tfr 9: 512x512x3
            bytesq = len(data) # tfr 9: 512x512x3
            # read from string buffer into int8 array
            b = np.frombuffer(data, dtype=np.uint8)
            # group int8 array to CHW format
            img_arr =  b.reshape(shape[0], shape[1], shape[2]) # 3x3 b/w
            # transpose int array to HWC, CHW => HWC
            img_arr = img_arr.transpose([1, 2, 0])
            cv_rgb(img_arr)

    @staticmethod
    def rgbs_to_npy(imgs,npyfolder,
            im_size=256,
            mss=(1024 ** 3),
            verbose=False,
            npy_file_frmt='data{}.npy',
            zfill=4
        ):

        segment_length = mss // (im_size * im_size * 3)
        np.random.shuffle(imgs)

        if verbose:
            print(f"{str(len(imgs))} imgs to npy")

        kn = 0  # image in segment
        sn = 0  # segment - segment_length = mss // (im_size * im_size * 3)
        segment = []
        for item in imgs:        
            if verbose:
                print('\r' + str(sn) + " // " + str(kn) + "\t", end = '\r')

            segment.append(item)
            kn = kn + 1
            if kn >= segment_length:
                npyfile = npy_file_frmt.format(str(sn).zfill(zfill))
                # npyfile = "data-"+str(sn)+".npy"

                npydst = os.path.join(npyfolder, npyfile)
                np.save(npydst, np.array(segment))
                segment = []
                kn = 0
                sn = sn + 1

        npyfile = npy_file_frmt.format(str(sn).zfill(zfill))
        npydst = os.path.join(npyfolder, npyfile)
        print("sn.kn %d %d" %(sn,kn))
        print(f"array_to_npy save to {npydst}")
        np.save(npydst, np.array(segment))

    @staticmethod
    def folder_to_npy(data_org, data_npy, im_size = 256, mss = (1024 ** 3), verbose = True):
    # https://github.com/manicman1999/StyleGAN2-Tensorflow-2.0/blob/master/datagen.py
        imgs = Onfile.folder_to_pils(data_org)
        Onrecord.pils_to_npy(imgs, data_npy, im_size, mss, verbose)

    @staticmethod
    def _folder_to_npy(data_dir, npy_folder, im_size, mss=(1024 ** 3), verbose=True):

        def snname(sn):
            return "data-"+str(sn)+".npy"

        os.makedirs(npy_folder, exist_ok=True)
        if verbose:
            print(f"Converting from images in {data_dir} to numpy files... onto {npy_folder}")

        names = []
        for dirpath, dirnames, filenames in os.walk(data_dir):
            for filename in [f for f in filenames if (f.endswith(".jpg") or f.endswith(".png") or f.endswith(".JPEG"))]:
                fname = os.path.join(dirpath, filename)
                names.append(fname)

        np.random.shuffle(names)

        if verbose:
            print(str(len(names)) + " images.")

        kn = 0
        sn = 0

        segment = []

        for fname in names:
            if verbose:
                print('\r' + str(sn) + " // " + str(kn) + "\t", end = '\r')

            try:
                temp = Image.open(fname).convert('RGB').resize((im_size, im_size), Image.BILINEAR)
            except:
                print("Importing image failed on", fname)
            temp = np.array(temp, dtype='uint8')
            segment.append(temp)
            kn = kn + 1

            segment_length = mss // (im_size * im_size * 3)

            if kn >= segment_length:
                
                target = os.path.join(npy_folder,snname(sn))
                np.save(target, np.array(segment))

                segment = []
                kn = 0
                sn = sn + 1

        target = os.path.join(npy_folder,snname(sn))
        np.save(target, np.array(segment))

    @staticmethod
    def pils_to_npy(pils, data_npy, im_size = 256, mss = (1024 ** 3), verbose = True):
        npyfolder = data_npy
        if verbose:
            print("datagene.folder_to_npy:images.q: %s" %(str(len(pils))))
        
        segment_length = mss // (im_size * im_size * 3)
        print(f"pils_to_npy segment_length: {segment_length}")

        np.random.shuffle(pils)

        kn = 0  # image in segment
        sn = 0  # segment where segment_length: mss // (im_size * im_size * 3)
        segment = []
        # for img in pils:
        for img in pils:        
            if 0:
                print(f"img type: {type(img)} {np.shape(img)}")
                print('\r' + str(sn) + " // " + str(kn) + "\t", end = '\r')

            img = img.convert('RGB').resize((im_size, im_size), Image.BILINEAR)
            rgb = np.array(img, dtype='uint8')
            
            # print("folder_to_npy img type", type(img)) # <class 'numpy.ndarray'>
            # print("folder_to_npy img shape", img.shape) # (256, 256, 3)
            segment.append(rgb)
            kn = kn + 1

            if kn >= segment_length:
                npyfile = "data-"+str(sn)+".npy"
                npydst = os.path.join(npyfolder, npyfile)
                np.save(npydst, np.array(segment))

                segment = []
                kn = 0
                sn = sn + 1

        npyfile = "data-"+str(sn)+".npy"
        npydst = os.path.join(npyfolder, npyfile)
        np.save(npydst, np.array(segment))


    @staticmethod
    def npys_folder_to_pils(folder):
    #  https://github.com/manicman1999/StyleGAN2-Tensorflow-2.0/blob/master/datagen.py
        pils=[]
        segments = []
        for dirpath, dirnames, filenames in os.walk(folder):
            for filename in [f for f in filenames if f.endswith(".npy")]:
                segments.append(os.path.join(dirpath, filename))

        for i in range(len(segments)):
            tmp = np.load(segments[i])
            for j in range(len(tmp)):
                rgb = tmp[j]
                pil = Image.fromarray(rgb)
                pils.append(pil)
        
        return pils

    @staticmethod
    def npys_folder_to_rgbs(folder):

        segments = []
        for dirpath, dirnames, filenames in os.walk(folder):
            for filename in [f for f in filenames if f.endswith(".npy")]:
                segments.append(os.path.join(dirpath, filename))

        import random
        segment_num = random.randint(0, len(segments) - 1)
        imgs = np.load(segments[segment_num])
        return imgs

    @staticmethod
    def load_from_npy(folder, segments):
        # for dirpath, dirnames, filenames in os.walk("data/" + folder + "-npy-" + str(self.im_size)):
        for dirpath, dirnames, filenames in os.walk(folder):
            for filename in [f for f in filenames if f.endswith(".npy")]:
                segments.append(os.path.join(dirpath, filename))
        return Onrecord.load_segment(segments)

    @staticmethod
    def load_segment(segments):
        segment_num =np.random.randint(0, len(segments))
        images = np.load(segments[segment_num])
        return images

    @staticmethod
    def folder_to_labels(data_dir='./', xSize=128, ySize=128, qslices=100, resize=0.75, eps = 1):
        
        arrs = folder_to_formed_nuas(data_dir, xSize, ySize, qslices, resize, eps)
        labels = [[arr[0], arr[1]] for arr in arrs]
        return labels

    @staticmethod
    def folder_to_tfrecords(src_dir, target_dir):
        assert os.path.exists(src_dir), f"src dir {src_dir} does not exist"
        assert os.path.exists(target_dir), f"src dir {target_dir} does not exist"
        imgs = Onfile.folder_to_rgbs(src_dir)
        Onrecord.imgs_to_tfrecords(imgs, target_dir)

    @staticmethod
    # relux
    def create_from_images(tfrecord_dir, image_dir, shuffle):
        print('Loading images from "%s"' % image_dir)
        image_filenames = sorted(glob.glob(os.path.join(image_dir, '*')))
        if len(image_filenames) == 0:
            Onutil.error('No input images found')

        img = np.asarray(PIL.Image.open(image_filenames[0]))
        resolution = img.shape[0]
        print("create_from_images resolution", resolution)
        channels = img.shape[2] if img.ndim == 3 else 1
        if img.shape[1] != resolution:
             Onutil.error('Input images must have the same width and height')
        if resolution != 2 ** int(np.floor(np.log2(resolution))):
             Onutil.error('Input image resolution must be a power-of-two')
        if channels not in [1, 3]:
             Onutil.error('Input images must be stored as RGB or grayscale')

        with TFRecordExporter(tfrecord_dir, len(image_filenames)) as tfr:
            order = tfr.choose_shuffled_order() if shuffle else np.arange(len(image_filenames))
            for idx in range(order.size):
                img = np.asarray(PIL.Image.open(image_filenames[order[idx]]))
                if channels == 1:
                    img = img[np.newaxis, :, :] # HW => CHW
                else:
                    img = img.transpose([2, 0, 1]) # HWC => CHW
                tfr.add_image(img)


    @staticmethod
    def imgs_to_tfrecords(imgs, target_dir, shuffle=False):
        assert os.path.exists(target_dir), f"src dir {target_dir} does not exist"

        # write imgs to tfrecord
        with TFRecordExporter(target_dir, len(imgs)) as tfr:
            order = tfr.choose_shuffled_order() if shuffle else np.arange(len(imgs))
            for idx in range(order.size):
                img = imgs[idx]
                channels = img.shape[2] if img.ndim == 3 else 1
                if channels == 1:
                    img = img[np.newaxis, :, :] # HW => CHW
                else:
                    img = img.transpose([2, 0, 1]) # HWC => CHW
                img = img / 255 # normalize
                tfr.add_image(img)

    @staticmethod
    def show_tfrecord_from_file(src_dir,tfr_file):
        images = []
        tfr_files = sorted(glob.glob(os.path.join(src_dir, '*.tfrecords')))
        dataset = tf.data.TFRecordDataset(src_dir, compression_type=None, buffer_size=None, num_parallel_reads=None)
        # tfrpath = os.path.join(src_dir, tfr_file)
        # dataset = tf.data.TFRecordDataset(tfrpath) # <TFRecordDatasetV2 shapes: (), types: tf.string>
        Onrecord.show_tfrecord(dataset)

    @staticmethod
    def show_tfrecord(dataset):

        image_feature_description = {
            'shape': tf.io.FixedLenFeature([3], tf.int64),
            'data': tf.io.FixedLenFeature([], tf.string),
        }
        def _parse_image_function(example_proto):
            return tf.io.parse_single_example(example_proto, image_feature_description)        

        parsed_dataset = dataset.map(_parse_image_function)
        for idx, dataitem in enumerate(parsed_dataset):
            tfshape = dataitem['shape']
            shape = tfshape.numpy()
            tfdata = dataitem['data']
            data = tfdata.numpy()
            size = (shape[1], shape[2])
            mode = 'RGB' if shape[0] == 3 else 'RGBA'
            img = np.array(data)
            channels = shape[0]
            width = shape[1]
            height = shape[2]
            bytes_needed = int(width * height * channels)  # tfr 9: 512x512x3
            bytesq = len(data) # tfr 9: 512x512x3
            b = np.frombuffer(data, dtype=np.uint8)
            img_arr =  b.reshape(shape[0], shape[1], shape[2]) # 3x3 b/w
            img_arr = img_arr.transpose([1, 2, 0])
            cv_rgb(img_arr)

    @staticmethod
    def get_dataset(tfrecord_base_dir, res, buffer_size, batch_size, epochs=None, name=None):
        # _p_ https://github.com/moono/stylegan2-tf-2.x/blob/master/dataset_ffhq.py
        fn_index = int(np.log2(res))
        if name:
            prefix=name
        else:
            prefix='*'
        # tfrecord_fn = os.path.join(tfrecord_base_dir, 'ffhq-r{:02d}.tfrecords'.format(fn_index))
        files = glob.glob(os.path.join(tfrecord_base_dir, prefix + '-r{:02d}.tfrecords'.format(fn_index)))
        tfrecord_fn = files[0]

        with tf.device('/cpu:0'):
            dataset = tf.data.TFRecordDataset(tfrecord_fn)
            dataset = dataset.map(map_func=Onrecord.parse_tfrecord_tf, num_parallel_calls=8)
            dataset = dataset.shuffle(buffer_size=buffer_size)
            dataset = dataset.repeat(epochs)
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset

    @staticmethod
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

    @staticmethod
    def tfts_to_files(imgs,dst_dir= './',img_name_frmt='img{}.jpg',zfill = 4,):
        for i,img in enumerate(imgs):
            img_name = img_name_frmt.format(str(i).zfill(zfill))
            path = os.path.join(dst_dir, img_name)
            tft = img # <class 'tensorflow.python.framework.ops.EagerTensor'> (256, 256, 3)
            arr = img.numpy().astype(np.uint8) # <class 'numpy.ndarray'> (256, 256, 3)
            img = Image.fromarray(arr)
            img = img.convert("RGB")
            img.save(path)

#   ******************
#   ONCHECK
#   ******************
class Oncheck(tf.keras.models.Model):
    def __init__(self,
                 model,
                 ckpt_dir = 'ckpt',
                 clear = True):
        
        super(Oncheck, self).__init__()
        
        self.clear = clear
        self.model = model
        self.ckpt_dir = ckpt_dir
        self.step = 1


    def getckptidx(self, ckptname): # ckpt-98, None, ckpt--1
        idx = None
        if ckptname:
            m = re.search(r'([a-zA-Z_]*)-(.*)', ckptname)
            if m and m.group(2):
                idx = m.group(2)
        return idx


    def ckpt_sample(self):
        ckpt = tf.train.Checkpoint(v=tf.Variable(0.))
        mgr = tf.train.CheckpointManager(ckpt, '/tmp/tfckpts', max_to_keep=5)

        @tf.function
        def train_fn():
            data = tf.data.Dataset.range(10)
            for item in data:
                ckpt.v.assign_add(tf.cast(item, tf.float32))
                tf.py_function(mgr.save, [], [tf.string])

        train_fn()

    def get_ckpt_prefix(self):
        ckpt_dir = self.ckpt_dir

        return ckpt_dir

    def get_checkpoint(self):
        model = self.model
        p = model.chkpt

        ckpt = tf.train.Checkpoint(**p)
        return ckpt

    def get_weights(self):
        model = self.model
        ws = model.ws
        wspath = model.wspath

        for w in ws:
            wfile = wspath + "/" + w
            if os.path.exists(wfile):
                print("load weights %s" %wfile)
                ws[w] = model[w]
                model[w].load_weights(wfile)
                model[w] = ws[w]
            else:
                print("weights not found: %s" %wfile)            

    def get_manager(self, ckpt):
        model = self.model
        ckpt_dir = model.ckpt_dir
        max_to_keep = 3

        if self.clear:
            if os.path.exists(ckpt_dir):
                for root, dirs, files in os.walk(ckpt_dir, topdown=False):
                    print("clear ckpt dir: %s" %root)
                    for name in files:
                        os.remove(os.path.join(root, name))
                    for name in dirs:
                        os.rmdir(os.path.join(root, name))
                
        manager = tf.train.CheckpointManager(ckpt, 
                                             ckpt_dir, 
                                             max_to_keep)
        return manager


    #   ******************
    #   restore
    #
    #   if a checkpoint exists, restore the latest checkpoint
    #   eg:
    #       ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    #       if ckpt_manager.latest_checkpoint:
    #           ckpt.restore(ckpt_manager.latest_checkpoint)
    #           print ('Latest checkpoint restored!!')
    #
    def restore(self, ckpt):
        model = self.model
        ckpt_dir = self.ckpt_dir
         
        print("checkpoint restore %s" %ckpt)
        res = ckpt.restore(tf.train.latest_checkpoint(ckpt_dir))
        print("res: in % s checkpoint: %s" %(ckpt_dir, res))


#   ******************
#   ONCUDA
#
#   ******************
class Oncuda:

    # Internal helper funcs.

    @staticmethod
    def _find_compiler_bindir():
        for compiler_path in compiler_bindir_search_path:
            if os.path.isdir(compiler_path):
                return compiler_path
        return None

    @staticmethod
    def _get_compute_cap(device):
        caps_str = device.physical_device_desc
        m = re.search('compute capability: (\\d+).(\\d+)', caps_str)
        major = m.group(1)
        minor = m.group(2)
        return (major, minor)

    @staticmethod
    def _get_cuda_gpu_arch_string():
        gpus = [x for x in device_lib.list_local_devices() if x.device_type == 'GPU']
        if len(gpus) == 0:
            raise RuntimeError('No GPU devices found')
        (major, minor) = Oncuda._get_compute_cap(gpus[0])
        return 'sm_%s%s' % (major, minor)

    @staticmethod
    def _run_cmd(cmd):
        with os.popen(cmd) as pipe:
            output = pipe.read()
            status = pipe.close()
        if status is not None:
            raise RuntimeError('NVCC returned an error. See below for full command line and output log:\n\n%s\n\n%s' % (cmd, output))

    @staticmethod
    def _prepare_nvcc_cli(opts):
        # cmd = 'nvcc --std=c++11 -DNDEBUG ' + opts.strip()
        cmd = 'nvcc -DNDEBUG ' + opts.strip()
        cmd += ' --disable-warnings'
        cmd += ' --include-path "%s"' % tf.sysconfig.get_include()
        cmd += ' --include-path "%s"' % os.path.join(tf.sysconfig.get_include(), 'external', 'protobuf_archive', 'src')
        cmd += ' --include-path "%s"' % os.path.join(tf.sysconfig.get_include(), 'external', 'com_google_absl')
        cmd += ' --include-path "%s"' % os.path.join(tf.sysconfig.get_include(), 'external', 'eigen_archive')

        compiler_bindir = Oncuda._find_compiler_bindir()
        if compiler_bindir is None:
            # Require that _find_compiler_bindir succeeds on Windows.  Allow
            # nvcc to use whatever is the default on Linux.
            if os.name == 'nt':
                raise RuntimeError('Could not find MSVC/GCC/CLANG installation on this computer. Check compiler_bindir_search_path list in "%s".' % __file__)
        else:
            cmd += ' --compiler-bindir "%s"' % compiler_bindir
        cmd += ' 2>&1'
        return cmd

    @staticmethod
    def get_plugin(cuda_file, verbose=False):
        # print("=========================> custom_ops.py get_plugin ", cuda_file)

        cuda_file_base = os.path.basename(cuda_file)
        cuda_file_name, cuda_file_ext = os.path.splitext(cuda_file_base)

        # Already in cache?
        if cuda_file in _plugin_cache:
            return _plugin_cache[cuda_file]

        # Setup plugin.
        if verbose:
            print('Setting up TensorFlow plugin "%s": ' % cuda_file_base, end='', flush=True)
        try:
            # -----------------------------------------------------
            # Hash CUDA source.

            with open(cuda_file, 'wb') as f:
                f.write(Oncuda.get_kernel_upfirdn().encode('utf-8'))

            md5 = hashlib.md5()
            with open(cuda_file, 'rb') as f:
                md5.update(f.read())
            md5.update(b'\n')
            # -----------------------------------------------------

            # Hash headers included by the CUDA code by running it through the preprocessor.
            if not do_not_hash_included_headers:
                if verbose:
                    print('Preprocessing... ', end='', flush=True)
                with tempfile.TemporaryDirectory() as tmp_dir:
                    os.makedirs(tmp_dir, exist_ok=True)
                    tmp_file = os.path.join(tmp_dir, cuda_file_name + '_tmp' + cuda_file_ext)
                    _run_cmd(Oncuda._prepare_nvcc_cli('"%s" --preprocess -o "%s" --keep --keep-dir "%s"' % (cuda_file, tmp_file, tmp_dir)))
                    with open(tmp_file, 'rb') as f:
                        bad_file_str = ('"' + cuda_file.replace('\\', '/') + '"').encode('utf-8') # __FILE__ in error check macros
                        good_file_str = ('"' + cuda_file_base + '"').encode('utf-8')
                        for ln in f:
                            if not ln.startswith(b'# ') and not ln.startswith(b'#line '): # ignore line number pragmas
                                ln = ln.replace(bad_file_str, good_file_str)
                                md5.update(ln)
                        md5.update(b'\n')


            # Select compiler options.
            compile_opts = ''
            if os.name == 'nt':
                compile_opts += '"%s"' % os.path.join(tf.sysconfig.get_lib(), 'python', '_pywrap_tensorflow_internal.lib')
            elif os.name == 'posix':
                compile_opts += '"%s"' % os.path.join(tf.sysconfig.get_lib(), 'python', '_pywrap_tensorflow_internal.so')
                compile_opts += ' --compiler-options \'-fPIC -D_GLIBCXX_USE_CXX11_ABI=0\''
            else:
                assert False # not Windows or Linux, w00t?
            compile_opts += ' --gpu-architecture=%s' % Oncuda._get_cuda_gpu_arch_string()
            compile_opts += ' --use_fast_math'

            nvcc_cmd = Oncuda._prepare_nvcc_cli(compile_opts)

            # Hash build configuration.
            md5.update(('nvcc_cmd: ' + nvcc_cmd).encode('utf-8') + b'\n')
            md5.update(('tf.VERSION: ' + tf.__version__).encode('utf-8') + b'\n')
            md5.update(('cuda_cache_version_tag: ' + cuda_cache_version_tag).encode('utf-8') + b'\n')

            # Compile if not already compiled.
            bin_file_ext = '.dll' if os.name == 'nt' else '.so'
            bin_file = os.path.join(cuda_cache_path, cuda_file_name + '_' + md5.hexdigest() + bin_file_ext)
            if not os.path.isfile(bin_file):
                if verbose:
                    print('Compiling... ', end='', flush=True)
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_file = os.path.join(tmp_dir, cuda_file_name + '_tmp' + bin_file_ext)

                    Oncuda._run_cmd(nvcc_cmd + ' "%s" --shared -o "%s" --keep --keep-dir "%s"' % (cuda_file, tmp_file, tmp_dir))

                    os.makedirs(cuda_cache_path, exist_ok=True)
                    intermediate_file = os.path.join(cuda_cache_path, cuda_file_name + '_' + uuid.uuid4().hex + '_tmp' + bin_file_ext)
                    shutil.copyfile(tmp_file, intermediate_file)
                    os.rename(intermediate_file, bin_file) # atomic

            # Load.
            if verbose:
                print('Loading... ', end='', flush=True)
            plugin = tf.load_op_library(bin_file)

            keys = plugin.__dict__.keys()        

            # Add to cache.
            _plugin_cache[cuda_file] = plugin
            if verbose:
                print('Done.', flush=True)
            return plugin

        except:
            if verbose:
                print('Failed!', flush=True)
            raise

    @staticmethod
    def _get_plugin(plg=None):
        # return custom_ops.get_plugin(os.path.splitext(__file__)[0] + '.cu')
        # upfirdn_2d.cu
        # cufile = os.path.splitext(__file__)[0] + '.cu'   # _e_
        cufile = plg + '.cu'   # _e_
        return Oncuda.get_plugin(cufile)

    @staticmethod
    def upfirdn_2d(x, k, upx=1, upy=1, downx=1, downy=1, padx0=0, padx1=0, pady0=0, pady1=0, impl='cuda', gpu=True):
        r"""Pad, upsample, FIR filter, and downsample a batch of 2D images.

        Accepts a batch of 2D images of the shape `[majorDim, inH, inW, minorDim]`
        and performs the following operations for each image, batched across
        `majorDim` and `minorDim`:

        1. Pad the image with zeros by the specified number of pixels on each side
        (`padx0`, `padx1`, `pady0`, `pady1`). Specifying a negative value
        corresponds to cropping the image.

        2. Upsample the image by inserting the zeros after each pixel (`upx`, `upy`).

        3. Convolve the image with the specified 2D FIR filter (`k`), shrinking the
        image so that the footprint of all output pixels lies within the input image.

        4. Downsample the image by throwing away pixels (`downx`, `downy`).

        This sequence of operations bears close resemblance to scipy.signal.upfirdn().
        The fused op is considerably more efficient than performing the same calculation
        using standard TensorFlow ops. It supports gradients of arbitrary order.

        Args:
            x:      Input tensor of the shape `[majorDim, inH, inW, minorDim]`.
            k:      2D FIR filter of the shape `[firH, firW]`.
            upx:    Integer upsampling factor along the X-axis (default: 1).
            upy:    Integer upsampling factor along the Y-axis (default: 1).
            downx:  Integer downsampling factor along the X-axis (default: 1).
            downy:  Integer downsampling factor along the Y-axis (default: 1).
            padx0:  Number of pixels to pad on the left side (default: 0).
            padx1:  Number of pixels to pad on the right side (default: 0).
            pady0:  Number of pixels to pad on the top side (default: 0).
            pady1:  Number of pixels to pad on the bottom side (default: 0).
            impl:   Name of the implementation to use. Can be `"ref"` or `"cuda"` (default).

        Returns:
            Tensor of the shape `[majorDim, outH, outW, minorDim]`, and same datatype as `x`.
        """

        impl_dict = {
            'ref':  Oncuda._upfirdn_2d_ref,
            'cuda': Oncuda._upfirdn_2d_cuda,
        }
        return impl_dict[impl](x=x, k=k, upx=upx, upy=upy, downx=downx, downy=downy, padx0=padx0, padx1=padx1, pady0=pady0, pady1=pady1, gpu=gpu)

    #----------------------------------------------------------------------------
    @staticmethod
    def _upfirdn_2d_ref(x, k, upx, upy, downx, downy, padx0, padx1, pady0, pady1, gpu=True):
        """Slow reference implementation of `upfirdn_2d()` using standard TensorFlow ops."""

        x = tf.convert_to_tensor(x)
        k = np.asarray(k, dtype=np.float32)
        assert x.shape.rank == 4
        inH = x.shape[1]
        inW = x.shape[2]
        minorDim = Oncuda._shape(x, 3)
        kernelH, kernelW = k.shape
        assert inW >= 1 and inH >= 1
        assert kernelW >= 1 and kernelH >= 1
        assert isinstance(upx, int) and isinstance(upy, int)
        assert isinstance(downx, int) and isinstance(downy, int)
        assert isinstance(padx0, int) and isinstance(padx1, int)
        assert isinstance(pady0, int) and isinstance(pady1, int)

        # Upsample (insert zeros).
        x = tf.reshape(x, [-1, inH, 1, inW, 1, minorDim])
        x = tf.pad(x, [[0, 0], [0, 0], [0, upy - 1], [0, 0], [0, upx - 1], [0, 0]])
        x = tf.reshape(x, [-1, inH * upy, inW * upx, minorDim])

        # Pad (crop if negative).
        x = tf.pad(x, [[0, 0], [max(pady0, 0), max(pady1, 0)], [max(padx0, 0), max(padx1, 0)], [0, 0]])
        x = x[:, max(-pady0, 0) : x.shape[1] - max(-pady1, 0), max(-padx0, 0) : x.shape[2] - max(-padx1, 0), :]

        # Convolve with filter.
        x = tf.transpose(x, [0, 3, 1, 2])
        x = tf.reshape(x, [-1, 1, inH * upy + pady0 + pady1, inW * upx + padx0 + padx1])
        w = tf.constant(k[::-1, ::-1, np.newaxis, np.newaxis], dtype=x.dtype)
        if gpu:
            x = tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='VALID', data_format='NCHW')
        else:
            x = tf.transpose(x, [0, 2, 3, 1])
            x = tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='VALID', data_format='NHWC')
            x = tf.transpose(x, [0, 3, 1, 2])     
        x = tf.reshape(x, [-1, minorDim, inH * upy + pady0 + pady1 - kernelH + 1, inW * upx + padx0 + padx1 - kernelW + 1])
        x = tf.transpose(x, [0, 2, 3, 1])

        # Downsample (throw away pixels).
        return x[:, ::downy, ::downx, :]

    #----------------------------------------------------------------------------
    @staticmethod
    def _upfirdn_2d_cuda(x, k, upx, upy, downx, downy, padx0, padx1, pady0, pady1, gpu=True):
        """Fast CUDA implementation of `upfirdn_2d()` using custom ops."""

        x = tf.convert_to_tensor(x)
        k = np.asarray(k, dtype=np.float32)
        majorDim, inH, inW, minorDim = x.shape.as_list()
        kernelH, kernelW = k.shape
        assert inW >= 1 and inH >= 1
        assert kernelW >= 1 and kernelH >= 1
        assert isinstance(upx, int) and isinstance(upy, int)
        assert isinstance(downx, int) and isinstance(downy, int)
        assert isinstance(padx0, int) and isinstance(padx1, int)
        assert isinstance(pady0, int) and isinstance(pady1, int)

        outW = (inW * upx + padx0 + padx1 - kernelW) // downx + 1
        outH = (inH * upy + pady0 + pady1 - kernelH) // downy + 1
        assert outW >= 1 and outH >= 1

        kc = tf.constant(k, dtype=x.dtype)
        gkc = tf.constant(k[::-1, ::-1], dtype=x.dtype)
        gpadx0 = kernelW - padx0 - 1
        gpady0 = kernelH - pady0 - 1
        gpadx1 = inW * upx - outW * downx + padx0 - upx + 1
        gpady1 = inH * upy - outH * downy + pady0 - upy + 1

        # add plugin _e_
        @tf.custom_gradient
        def func(x):
            y = Oncuda._get_plugin('upfirdn_2d').up_fir_dn2d(x=x, k=kc, upx=upx, upy=upy, downx=downx, downy=downy, padx0=padx0, padx1=padx1, pady0=pady0, pady1=pady1)
            y.set_shape([majorDim, outH, outW, minorDim])

            @tf.custom_gradient
            def grad(dy):
                dx = Oncuda._get_plugin('upfirdn_2d').up_fir_dn2d(x=dy, k=gkc, upx=downx, upy=downy, downx=upx, downy=upy, padx0=gpadx0, padx1=gpadx1, pady0=gpady0, pady1=gpady1)
                dx.set_shape([majorDim, inH, inW, minorDim])
                return dx, func
            return y, grad
        return func(x)

    #----------------------------------------------------------------------------
    @staticmethod
    def filter_2d(x, k, gain=1, data_format='NCHW', impl='cuda'):
        r"""Filter a batch of 2D images with the given FIR filter.

        Accepts a batch of 2D images of the shape `[N, C, H, W]` or `[N, H, W, C]`
        and filters each image with the given filter. The filter is normalized so that
        if the input pixels are constant, they will be scaled by the specified `gain`.
        Pixels outside the image are assumed to be zero.

        Args:
            x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W, C]`.
            k:            FIR filter of the shape `[firH, firW]` or `[firN]` (separable).
            gain:         Scaling factor for signal magnitude (default: 1.0).
            data_format:  `'NCHW'` or `'NHWC'` (default: `'NCHW'`).
            impl:         Name of the implementation to use. Can be `"ref"` or `"cuda"` (default).

        Returns:
            Tensor of the same shape and datatype as `x`.
        """

        k = Oncuda._setup_kernel(k) * gain
        p = k.shape[0] - 1
        return Oncuda._simple_upfirdn_2d(x, k, pad0=(p+1)//2, pad1=p//2, data_format=data_format, impl=impl)

    #----------------------------------------------------------------------------
    @staticmethod
    def upsample_2d(x, k=None, factor=2, gain=1, data_format='NCHW', impl='cuda', gpu=True):
        r"""Upsample a batch of 2D images with the given filter.

        Accepts a batch of 2D images of the shape `[N, C, H, W]` or `[N, H, W, C]`
        and upsamples each image with the given filter. The filter is normalized so that
        if the input pixels are constant, they will be scaled by the specified `gain`.
        Pixels outside the image are assumed to be zero, and the filter is padded with
        zeros so that its shape is a multiple of the upsampling factor.

        Args:
            x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W, C]`.
            k:            FIR filter of the shape `[firH, firW]` or `[firN]` (separable).
                        The default is `[1] * factor`, which corresponds to nearest-neighbor
                        upsampling.
            factor:       Integer upsampling factor (default: 2).
            gain:         Scaling factor for signal magnitude (default: 1.0).
            data_format:  `'NCHW'` or `'NHWC'` (default: `'NCHW'`).
            impl:         Name of the implementation to use. Can be `"ref"` or `"cuda"` (default).

        Returns:
            Tensor of the shape `[N, C, H * factor, W * factor]` or
            `[N, H * factor, W * factor, C]`, and same datatype as `x`.
        """

        assert isinstance(factor, int) and factor >= 1
        if k is None:
            k = [1] * factor
        k = Oncuda._setup_kernel(k) * (gain * (factor ** 2))
        p = k.shape[0] - factor
        return Oncuda._simple_upfirdn_2d(x, k, up=factor, pad0=(p+1)//2+factor-1, pad1=p//2, data_format=data_format, impl=impl, gpu=gpu)

    #----------------------------------------------------------------------------
    @staticmethod
    def downsample_2d(x, k=None, factor=2, gain=1, data_format='NCHW', impl='cuda'):
        r"""Downsample a batch of 2D images with the given filter.

        Accepts a batch of 2D images of the shape `[N, C, H, W]` or `[N, H, W, C]`
        and downsamples each image with the given filter. The filter is normalized so that
        if the input pixels are constant, they will be scaled by the specified `gain`.
        Pixels outside the image are assumed to be zero, and the filter is padded with
        zeros so that its shape is a multiple of the downsampling factor.

        Args:
            x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W, C]`.
            k:            FIR filter of the shape `[firH, firW]` or `[firN]` (separable).
                        The default is `[1] * factor`, which corresponds to average pooling.
            factor:       Integer downsampling factor (default: 2).
            gain:         Scaling factor for signal magnitude (default: 1.0).
            data_format:  `'NCHW'` or `'NHWC'` (default: `'NCHW'`).
            impl:         Name of the implementation to use. Can be `"ref"` or `"cuda"` (default).

        Returns:
            Tensor of the shape `[N, C, H // factor, W // factor]` or
            `[N, H // factor, W // factor, C]`, and same datatype as `x`.
        """

        assert isinstance(factor, int) and factor >= 1
        if k is None:
            k = [1] * factor
        k = Oncuda._setup_kernel(k) * gain
        p = k.shape[0] - factor
        return Oncuda._simple_upfirdn_2d(x, k, down=factor, pad0=(p+1)//2, pad1=p//2, data_format=data_format, impl=impl)

    #----------------------------------------------------------------------------
    @staticmethod
    def upsample_conv_2d(x, w, k=None, factor=2, gain=1, data_format='NCHW', impl='cuda', gpu=True):
        r"""Fused `upsample_2d()` followed by `tf.nn.conv2d()`.

        Padding is performed only once at the beginning, not between the operations.
        The fused op is considerably more efficient than performing the same calculation
        using standard TensorFlow ops. It supports gradients of arbitrary order.

        Args:
            x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W, C]`.
            w:            Weight tensor of the shape `[filterH, filterW, inChannels, outChannels]`.
                        Grouped convolution can be performed by `inChannels = x.shape[0] // numGroups`.
            k:            FIR filter of the shape `[firH, firW]` or `[firN]` (separable).
                        The default is `[1] * factor`, which corresponds to nearest-neighbor
                        upsampling.
            factor:       Integer upsampling factor (default: 2).
            gain:         Scaling factor for signal magnitude (default: 1.0).
            data_format:  `'NCHW'` or `'NHWC'` (default: `'NCHW'`).
            impl:         Name of the implementation to use. Can be `"ref"` or `"cuda"` (default).

        Returns:
            Tensor of the shape `[N, C, H * factor, W * factor]` or
            `[N, H * factor, W * factor, C]`, and same datatype as `x`.
        """

        assert isinstance(factor, int) and factor >= 1

        # Check weight shape.
        w = tf.convert_to_tensor(w)
        assert w.shape.rank == 4
        convH = w.shape[0]
        convW = w.shape[1]
        inC = Oncuda._shape(w, 2)
        outC = Oncuda._shape(w, 3)
        assert convW == convH

        # Setup filter kernel.
        if k is None:
            k = [1] * factor
        k = Oncuda._setup_kernel(k) * (gain * (factor ** 2))
        p = (k.shape[0] - factor) - (convW - 1)

        # Determine data dimensions.
        if data_format == 'NCHW':
            stride = [1, 1, factor, factor]
            output_shape = [Oncuda._shape(x, 0), outC, (Oncuda._shape(x, 2) - 1) * factor + convH, (Oncuda._shape(x, 3) - 1) * factor + convW]
            num_groups = Oncuda._shape(x, 1) // inC
        else:
            stride = [1, factor, factor, 1]
            output_shape = [Oncuda._shape(x, 0), (Oncuda._shape(x, 1) - 1) * factor + convH, (Oncuda._shape(x, 2) - 1) * factor + convW, outC]
            num_groups = Oncuda._shape(x, 3) // inC

        # Transpose weights.
        w = tf.reshape(w, [convH, convW, inC, num_groups, -1])
        w = tf.transpose(w[::-1, ::-1], [0, 1, 4, 3, 2])
        w = tf.reshape(w, [convH, convW, -1, num_groups * inC])

        # Execute.
        x = tf.nn.conv2d_transpose(x, w, output_shape=output_shape, strides=stride, padding='VALID', data_format=data_format)
        return Oncuda._simple_upfirdn_2d(x, k, pad0=(p+1)//2+factor-1, pad1=p//2+1, data_format=data_format, impl=impl, gpu=gpu)

    #----------------------------------------------------------------------------
    @staticmethod
    def conv_downsample_2d(x, w, k=None, factor=2, gain=1, data_format='NCHW', impl='cuda', gpu=True):
        r"""Fused `tf.nn.conv2d()` followed by `downsample_2d()`.

        Padding is performed only once at the beginning, not between the operations.
        The fused op is considerably more efficient than performing the same calculation
        using standard TensorFlow ops. It supports gradients of arbitrary order.

        Args:
            x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W, C]`.
            w:            Weight tensor of the shape `[filterH, filterW, inChannels, outChannels]`.
                        Grouped convolution can be performed by `inChannels = x.shape[0] // numGroups`.
            k:            FIR filter of the shape `[firH, firW]` or `[firN]` (separable).
                        The default is `[1] * factor`, which corresponds to average pooling.
            factor:       Integer downsampling factor (default: 2).
            gain:         Scaling factor for signal magnitude (default: 1.0).
            data_format:  `'NCHW'` or `'NHWC'` (default: `'NCHW'`).
            impl:         Name of the implementation to use. Can be `"ref"` or `"cuda"` (default).

        Returns:
            Tensor of the shape `[N, C, H // factor, W // factor]` or
            `[N, H // factor, W // factor, C]`, and same datatype as `x`.
        """

        assert isinstance(factor, int) and factor >= 1
        w = tf.convert_to_tensor(w)
        convH, convW, _inC, _outC = w.shape.as_list()
        assert convW == convH
        if k is None:
            k = [1] * factor
        k = Oncuda._setup_kernel(k) * gain
        p = (k.shape[0] - factor) + (convW - 1)
        if data_format == 'NCHW':
            s = [1, 1, factor, factor]
        else:
            s = [1, factor, factor, 1]
        x = Oncuda._simple_upfirdn_2d(x, k, pad0=(p+1)//2, pad1=p//2, data_format=data_format, impl=impl, gpu=gpu)
        return tf.nn.conv2d(x, w, strides=s, padding='VALID', data_format=data_format)

    #----------------------------------------------------------------------------
    # Internal helper funcs.

    @staticmethod
    def _shape(tf_expr, dim_idx):
        if tf_expr.shape.rank is not None:
            dim = tf_expr.shape[dim_idx]
            if dim is not None:
                return dim
        return tf.shape(tf_expr)[dim_idx]

    @staticmethod
    def _setup_kernel(k):
        k = np.asarray(k, dtype=np.float32)
        if k.ndim == 1:
            k = np.outer(k, k)
        k /= np.sum(k)
        assert k.ndim == 2
        assert k.shape[0] == k.shape[1]
        return k

    @staticmethod
    def _simple_upfirdn_2d(x, k, up=1, down=1, pad0=0, pad1=0, data_format='NCHW', impl='cuda', gpu=True):
        assert data_format in ['NCHW', 'NHWC']
        assert x.shape.rank == 4
        y = x
        if data_format == 'NCHW':
            y = tf.reshape(y, [-1, Oncuda._shape(y, 2), Oncuda._shape(y, 3), 1])
        y = Oncuda.upfirdn_2d(y, k, upx=up, upy=up, downx=down, downy=down, padx0=pad0, padx1=pad1, pady0=pad0, pady1=pad1, impl=impl, gpu=gpu)
        if data_format == 'NCHW':
            y = tf.reshape(y, [-1, Oncuda._shape(x, 1), Oncuda._shape(y, 1), Oncuda._shape(y, 2)])
        return y



    #   ******************
    #   dnnlib/ops/upfirdn_2d.cu - get raw
    #

    # _e_
    # ERROR in Eigen C++ file named Tensor
    # ref: https://github.com/tensorflow/tensorflow/issues/40148
    # ref: https://github.com/tensorflow/tensorflow/issues/39829
    # C:/Users/xxx/AppData/Local/Programs/Python/Python36/lib/site-packages/tensorflow/include/unsupported/Eigen/CXX11/Tensor    
    #     {port}/Anaconda3/envs/{env}/lib/site-packages/tensorflow/include/unsupported/Eigen/CXX11/Tensor(74): fatal error C1083: Cannot open include file: 'unistd.h': No such file or directory
    #     _pywrap_tensorflow_internal.lib
    #     upfirdn_2d.cu
    # python -c "import tensorflow as tf; print(tf.version.GIT_VERSION, tf.version.VERSION)" 

    @staticmethod    
    def get_kernel_upfirdn():
        res="""// Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/stylegan2/license.html

#define EIGEN_USE_GPU
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <stdio.h>

using namespace tensorflow;
using namespace tensorflow::shape_inference;

//------------------------------------------------------------------------
// Helpers.

#define OP_CHECK_CUDA_ERROR(CTX, CUDA_CALL) do { cudaError_t err = CUDA_CALL; OP_REQUIRES(CTX, err == cudaSuccess, errors::Internal(cudaGetErrorName(err))); } while (false)

static __host__ __device__ __forceinline__ int floorDiv(int a, int b)
{
    int c = a / b;
    if (c * b > a)
        c--;
    return c;
}

//------------------------------------------------------------------------
// CUDA kernel params.

template <class T>
struct UpFirDn2DKernelParams
{
    const T*    x;          // [majorDim, inH, inW, minorDim]
    const T*    k;          // [kernelH, kernelW]
    T*          y;          // [majorDim, outH, outW, minorDim]

    int         upx;
    int         upy;
    int         downx;
    int         downy;
    int         padx0;
    int         padx1;
    int         pady0;
    int         pady1;

    int         majorDim;
    int         inH;
    int         inW;
    int         minorDim;
    int         kernelH;
    int         kernelW;
    int         outH;
    int         outW;
    int         loopMajor;
    int         loopX;
};

//------------------------------------------------------------------------
// General CUDA implementation for large filter kernels.

template <class T>
static __global__ void UpFirDn2DKernel_large(const UpFirDn2DKernelParams<T> p)
{
    // Calculate thread index.
    int minorIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int outY = minorIdx / p.minorDim;
    minorIdx -= outY * p.minorDim;
    int outXBase = blockIdx.y * p.loopX * blockDim.y + threadIdx.y;
    int majorIdxBase = blockIdx.z * p.loopMajor;
    if (outXBase >= p.outW || outY >= p.outH || majorIdxBase >= p.majorDim)
        return;

    // Setup Y receptive field.
    int midY = outY * p.downy + p.upy - 1 - p.pady0;
    int inY = min(max(floorDiv(midY, p.upy), 0), p.inH);
    int h = min(max(floorDiv(midY + p.kernelH, p.upy), 0), p.inH) - inY;
    int kernelY = midY + p.kernelH - (inY + 1) * p.upy;

    // Loop over majorDim and outX.
    for (int loopMajor = 0, majorIdx = majorIdxBase; loopMajor < p.loopMajor && majorIdx < p.majorDim; loopMajor++, majorIdx++)
    for (int loopX = 0, outX = outXBase; loopX < p.loopX && outX < p.outW; loopX++, outX += blockDim.y)
    {
        // Setup X receptive field.
        int midX = outX * p.downx + p.upx - 1 - p.padx0;
        int inX = min(max(floorDiv(midX, p.upx), 0), p.inW);
        int w = min(max(floorDiv(midX + p.kernelW, p.upx), 0), p.inW) - inX;
        int kernelX = midX + p.kernelW - (inX + 1) * p.upx;

        // Initialize pointers.
        const T* xp = &p.x[((majorIdx * p.inH + inY) * p.inW + inX) * p.minorDim + minorIdx];
        const T* kp = &p.k[kernelY * p.kernelW + kernelX];
        int xpx = p.minorDim;
        int kpx = -p.upx;
        int xpy = p.inW * p.minorDim;
        int kpy = -p.upy * p.kernelW;

        // Inner loop.
        float v = 0.0f;
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                v += (float)(*xp) * (float)(*kp);
                xp += xpx;
                kp += kpx;
            }
            xp += xpy - w * xpx;
            kp += kpy - w * kpx;
        }

        // Store result.
        p.y[((majorIdx * p.outH + outY) * p.outW + outX) * p.minorDim + minorIdx] = (T)v;
    }
}

//------------------------------------------------------------------------
// Specialized CUDA implementation for small filter kernels.

template <class T, int upx, int upy, int downx, int downy, int kernelW, int kernelH, int tileOutW, int tileOutH>
static __global__ void UpFirDn2DKernel_small(const UpFirDn2DKernelParams<T> p)
{
    //assert(kernelW % upx == 0);
    //assert(kernelH % upy == 0);
    const int tileInW = ((tileOutW - 1) * downx + kernelW - 1) / upx + 1;
    const int tileInH = ((tileOutH - 1) * downy + kernelH - 1) / upy + 1;
    __shared__ volatile float sk[kernelH][kernelW];
    __shared__ volatile float sx[tileInH][tileInW];

    // Calculate tile index.
    int minorIdx = blockIdx.x;
    int tileOutY = minorIdx / p.minorDim;
    minorIdx -= tileOutY * p.minorDim;
    tileOutY *= tileOutH;
    int tileOutXBase = blockIdx.y * p.loopX * tileOutW;
    int majorIdxBase = blockIdx.z * p.loopMajor;
    if (tileOutXBase >= p.outW | tileOutY >= p.outH | majorIdxBase >= p.majorDim)
        return;

    // Load filter kernel (flipped).
    for (int tapIdx = threadIdx.x; tapIdx < kernelH * kernelW; tapIdx += blockDim.x)
    {
        int ky = tapIdx / kernelW;
        int kx = tapIdx - ky * kernelW;
        float v = 0.0f;
        if (kx < p.kernelW & ky < p.kernelH)
            v = (float)p.k[(p.kernelH - 1 - ky) * p.kernelW + (p.kernelW - 1 - kx)];
        sk[ky][kx] = v;
    }

    // Loop over majorDim and outX.
    for (int loopMajor = 0, majorIdx = majorIdxBase; loopMajor < p.loopMajor & majorIdx < p.majorDim; loopMajor++, majorIdx++)
    for (int loopX = 0, tileOutX = tileOutXBase; loopX < p.loopX & tileOutX < p.outW; loopX++, tileOutX += tileOutW)
    {
        // Load input pixels.
        int tileMidX = tileOutX * downx + upx - 1 - p.padx0;
        int tileMidY = tileOutY * downy + upy - 1 - p.pady0;
        int tileInX = floorDiv(tileMidX, upx);
        int tileInY = floorDiv(tileMidY, upy);
        __syncthreads();
        for (int inIdx = threadIdx.x; inIdx < tileInH * tileInW; inIdx += blockDim.x)
        {
            int relInY = inIdx / tileInW;
            int relInX = inIdx - relInY * tileInW;
            int inX = relInX + tileInX;
            int inY = relInY + tileInY;
            float v = 0.0f;
            if (inX >= 0 & inY >= 0 & inX < p.inW & inY < p.inH)
                v = (float)p.x[((majorIdx * p.inH + inY) * p.inW + inX) * p.minorDim + minorIdx];
            sx[relInY][relInX] = v;
        }

        // Loop over output pixels.
        __syncthreads();
        for (int outIdx = threadIdx.x; outIdx < tileOutH * tileOutW; outIdx += blockDim.x)
        {
            int relOutY = outIdx / tileOutW;
            int relOutX = outIdx - relOutY * tileOutW;
            int outX = relOutX + tileOutX;
            int outY = relOutY + tileOutY;

            // Setup receptive field.
            int midX = tileMidX + relOutX * downx;
            int midY = tileMidY + relOutY * downy;
            int inX = floorDiv(midX, upx);
            int inY = floorDiv(midY, upy);
            int relInX = inX - tileInX;
            int relInY = inY - tileInY;
            int kernelX = (inX + 1) * upx - midX - 1; // flipped
            int kernelY = (inY + 1) * upy - midY - 1; // flipped

            // Inner loop.
            float v = 0.0f;
            #pragma unroll
            for (int y = 0; y < kernelH / upy; y++)
                #pragma unroll
                for (int x = 0; x < kernelW / upx; x++)
                    v += sx[relInY + y][relInX + x] * sk[kernelY + y * upy][kernelX + x * upx];

            // Store result.
            if (outX < p.outW & outY < p.outH)
                p.y[((majorIdx * p.outH + outY) * p.outW + outX) * p.minorDim + minorIdx] = (T)v;
        }
    }
}

//------------------------------------------------------------------------
// TensorFlow op.

template <class T>
struct UpFirDn2DOp : public OpKernel
{
    UpFirDn2DKernelParams<T> m_attribs;

    UpFirDn2DOp(OpKernelConstruction* ctx) : OpKernel(ctx)
    {
        memset(&m_attribs, 0, sizeof(m_attribs));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("upx", &m_attribs.upx));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("upy", &m_attribs.upy));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("downx", &m_attribs.downx));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("downy", &m_attribs.downy));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("padx0", &m_attribs.padx0));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("padx1", &m_attribs.padx1));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("pady0", &m_attribs.pady0));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("pady1", &m_attribs.pady1));
        OP_REQUIRES(ctx, m_attribs.upx >= 1 && m_attribs.upy >= 1, errors::InvalidArgument("upx and upy must be at least 1x1"));
        OP_REQUIRES(ctx, m_attribs.downx >= 1 && m_attribs.downy >= 1, errors::InvalidArgument("downx and downy must be at least 1x1"));
    }

    void Compute(OpKernelContext* ctx)
    {
        UpFirDn2DKernelParams<T> p = m_attribs;
        cudaStream_t stream = ctx->eigen_device<Eigen::GpuDevice>().stream();

        const Tensor& x = ctx->input(0); // [majorDim, inH, inW, minorDim]
        const Tensor& k = ctx->input(1); // [kernelH, kernelW]
        p.x = x.flat<T>().data();
        p.k = k.flat<T>().data();
        OP_REQUIRES(ctx, x.dims() == 4, errors::InvalidArgument("input must have rank 4"));
        OP_REQUIRES(ctx, k.dims() == 2, errors::InvalidArgument("kernel must have rank 2"));
        OP_REQUIRES(ctx, x.NumElements() <= kint32max, errors::InvalidArgument("input too large"));
        OP_REQUIRES(ctx, k.NumElements() <= kint32max, errors::InvalidArgument("kernel too large"));

        p.majorDim  = (int)x.dim_size(0);
        p.inH       = (int)x.dim_size(1);
        p.inW       = (int)x.dim_size(2);
        p.minorDim  = (int)x.dim_size(3);
        p.kernelH   = (int)k.dim_size(0);
        p.kernelW   = (int)k.dim_size(1);
        OP_REQUIRES(ctx, p.kernelW >= 1 && p.kernelH >= 1, errors::InvalidArgument("kernel must be at least 1x1"));

        p.outW = (p.inW * p.upx + p.padx0 + p.padx1 - p.kernelW + p.downx) / p.downx;
        p.outH = (p.inH * p.upy + p.pady0 + p.pady1 - p.kernelH + p.downy) / p.downy;
        OP_REQUIRES(ctx, p.outW >= 1 && p.outH >= 1, errors::InvalidArgument("output must be at least 1x1"));

        Tensor* y = NULL; // [majorDim, outH, outW, minorDim]
        TensorShape ys;
        ys.AddDim(p.majorDim);
        ys.AddDim(p.outH);
        ys.AddDim(p.outW);
        ys.AddDim(p.minorDim);
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, ys, &y));
        p.y = y->flat<T>().data();
        OP_REQUIRES(ctx, y->NumElements() <= kint32max, errors::InvalidArgument("output too large"));

        // Choose CUDA kernel to use.
        void* cudaKernel = (void*)UpFirDn2DKernel_large<T>;
        int tileOutW = -1;
        int tileOutH = -1;
        if (p.upx == 1 && p.upy == 1 && p.downx == 1 && p.downy == 1 && p.kernelW <= 7 && p.kernelH <= 7) { cudaKernel = (void*)UpFirDn2DKernel_small<T, 1,1, 1,1, 7,7, 64,16>; tileOutW = 64; tileOutH = 16; }
        if (p.upx == 1 && p.upy == 1 && p.downx == 1 && p.downy == 1 && p.kernelW <= 6 && p.kernelH <= 6) { cudaKernel = (void*)UpFirDn2DKernel_small<T, 1,1, 1,1, 6,6, 64,16>; tileOutW = 64; tileOutH = 16; }
        if (p.upx == 1 && p.upy == 1 && p.downx == 1 && p.downy == 1 && p.kernelW <= 5 && p.kernelH <= 5) { cudaKernel = (void*)UpFirDn2DKernel_small<T, 1,1, 1,1, 5,5, 64,16>; tileOutW = 64; tileOutH = 16; }
        if (p.upx == 1 && p.upy == 1 && p.downx == 1 && p.downy == 1 && p.kernelW <= 4 && p.kernelH <= 4) { cudaKernel = (void*)UpFirDn2DKernel_small<T, 1,1, 1,1, 4,4, 64,16>; tileOutW = 64; tileOutH = 16; }
        if (p.upx == 1 && p.upy == 1 && p.downx == 1 && p.downy == 1 && p.kernelW <= 3 && p.kernelH <= 3) { cudaKernel = (void*)UpFirDn2DKernel_small<T, 1,1, 1,1, 3,3, 64,16>; tileOutW = 64; tileOutH = 16; }
        if (p.upx == 2 && p.upy == 2 && p.downx == 1 && p.downy == 1 && p.kernelW <= 8 && p.kernelH <= 8) { cudaKernel = (void*)UpFirDn2DKernel_small<T, 2,2, 1,1, 8,8, 64,16>; tileOutW = 64; tileOutH = 16; }
        if (p.upx == 2 && p.upy == 2 && p.downx == 1 && p.downy == 1 && p.kernelW <= 6 && p.kernelH <= 6) { cudaKernel = (void*)UpFirDn2DKernel_small<T, 2,2, 1,1, 6,6, 64,16>; tileOutW = 64; tileOutH = 16; }
        if (p.upx == 2 && p.upy == 2 && p.downx == 1 && p.downy == 1 && p.kernelW <= 4 && p.kernelH <= 4) { cudaKernel = (void*)UpFirDn2DKernel_small<T, 2,2, 1,1, 4,4, 64,16>; tileOutW = 64; tileOutH = 16; }
        if (p.upx == 2 && p.upy == 2 && p.downx == 1 && p.downy == 1 && p.kernelW <= 2 && p.kernelH <= 2) { cudaKernel = (void*)UpFirDn2DKernel_small<T, 2,2, 1,1, 2,2, 64,16>; tileOutW = 64; tileOutH = 16; }
        if (p.upx == 1 && p.upy == 1 && p.downx == 2 && p.downy == 2 && p.kernelW <= 8 && p.kernelH <= 8) { cudaKernel = (void*)UpFirDn2DKernel_small<T, 1,1, 2,2, 8,8, 32,8>;  tileOutW = 32; tileOutH = 8;  }
        if (p.upx == 1 && p.upy == 1 && p.downx == 2 && p.downy == 2 && p.kernelW <= 6 && p.kernelH <= 6) { cudaKernel = (void*)UpFirDn2DKernel_small<T, 1,1, 2,2, 6,6, 32,8>;  tileOutW = 32; tileOutH = 8;  }
        if (p.upx == 1 && p.upy == 1 && p.downx == 2 && p.downy == 2 && p.kernelW <= 4 && p.kernelH <= 4) { cudaKernel = (void*)UpFirDn2DKernel_small<T, 1,1, 2,2, 4,4, 32,8>;  tileOutW = 32; tileOutH = 8;  }
        if (p.upx == 1 && p.upy == 1 && p.downx == 2 && p.downy == 2 && p.kernelW <= 2 && p.kernelH <= 2) { cudaKernel = (void*)UpFirDn2DKernel_small<T, 1,1, 2,2, 2,2, 32,8>;  tileOutW = 32; tileOutH = 8;  }

        // Choose launch params.
        dim3 blockSize;
        dim3 gridSize;
        if (tileOutW > 0 && tileOutH > 0) // small
        {
            p.loopMajor = (p.majorDim - 1) / 16384 + 1;
            p.loopX = 1;
            blockSize = dim3(32 * 8, 1, 1);
            gridSize = dim3(((p.outH - 1) / tileOutH + 1) * p.minorDim, (p.outW - 1) / (p.loopX * tileOutW) + 1, (p.majorDim - 1) / p.loopMajor + 1);
        }
        else // large
        {
            p.loopMajor = (p.majorDim - 1) / 16384 + 1;
            p.loopX = 4;
            blockSize = dim3(4, 32, 1);
            gridSize = dim3((p.outH * p.minorDim - 1) / blockSize.x + 1, (p.outW - 1) / (p.loopX * blockSize.y) + 1, (p.majorDim - 1) / p.loopMajor + 1);
        }

        // Launch CUDA kernel.
        void* args[] = {&p};
        OP_CHECK_CUDA_ERROR(ctx, cudaLaunchKernel(cudaKernel, gridSize, blockSize, args, 0, stream));
    }
};

REGISTER_OP("UpFirDn2D")
    .Input      ("x: T")
    .Input      ("k: T")
    .Output     ("y: T")
    .Attr       ("T: {float, half}")
    .Attr       ("upx: int = 1")
    .Attr       ("upy: int = 1")
    .Attr       ("downx: int = 1")
    .Attr       ("downy: int = 1")
    .Attr       ("padx0: int = 0")
    .Attr       ("padx1: int = 0")
    .Attr       ("pady0: int = 0")
    .Attr       ("pady1: int = 0");
REGISTER_KERNEL_BUILDER(Name("UpFirDn2D").Device(DEVICE_GPU).TypeConstraint<float>("T"), UpFirDn2DOp<float>);
REGISTER_KERNEL_BUILDER(Name("UpFirDn2D").Device(DEVICE_GPU).TypeConstraint<Eigen::half>("T"), UpFirDn2DOp<Eigen::half>);

//------------------------------------------------------------------------"""
        return res


#   ******************
#   FUNS ONROSA
#   ******************
class Onrosa:

    # _p_ https://github.com/moono/stylegan2-tf-2.x/blob/master/stylegan2/utils.py

    @staticmethod
    def get_weight_initializer_runtime_coef(shape, gain=1, use_wscale=True, lrmul=1):
        """ get initializer and lr coef for different weights shapes"""
        fan_in = np.prod(shape[:-1]) # [kernel, kernel, fmaps_in, fmaps_out] or [in, out]
        he_std = gain / np.sqrt(fan_in) # He init

        # Equalized learning rate and custom learning rate multiplier.
        if use_wscale:
            init_std = 1.0 / lrmul
            runtime_coef = he_std * lrmul
        else:
            init_std = he_std / lrmul
            runtime_coef = lrmul
        
        return init_std, runtime_coef

    @staticmethod
    def convert_images_to_uint8(images, drange=[-1, 1], nchw_to_nhwc=False, shrink=1, uint8_cast=True):
        """Convert a minibatch of images from float32 to uint8 with configurable dynamic range.
        Can be used as an output transformation for Network.run().
        """
        images = tf.cast(images, tf.float32)
        if shrink > 1:
            ksize = [1, 1, shrink, shrink]
            images = tf.nn.avg_pool(images, ksize=ksize, strides=ksize, 
                                    padding="VALID", data_format="NCHW")
        if nchw_to_nhwc:
            images = tf.transpose(images, [0, 2, 3, 1])
        scale = 255 / (drange[1] - drange[0])
        images = images * scale + (0.5 - drange[0] * scale)
        if uint8_cast:
            images = tf.saturate_cast(images, tf.uint8)
        return images

    @staticmethod
    def nf(stage, fmap_base=16 << 10, fmap_decay=1.0, fmap_min=1, fmap_max=512): 
        return np.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max)
    # utils_stylegan2.py <=    


#   ******************
#   FUNS ONROLUX
#   /rolux/stylegan2encoder/dnnlib/util.py
#   ******************
class Onrolux:

    # Functionality to import modules/objects by name, and call functions by name
    # ------------------------------------------------------------------------------------------
    @staticmethod
    def get_module_from_obj_name(obj_name: str) -> Tuple[types.ModuleType, str]:
        """Searches for the underlying module behind the name to some python object.
        Returns the module and the object name (original name with module part removed)."""

        # allow convenience shorthands, substitute them by full names
        obj_name = re.sub("^np.", "numpy.", obj_name)
        obj_name = re.sub("^tf.", "tensorflow.", obj_name)

        # list alternatives for (module_name, local_obj_name)
        parts = obj_name.split(".")
        name_pairs = [(".".join(parts[:i]), ".".join(parts[i:])) for i in range(len(parts), 0, -1)]

        # try each alternative in turn
        for module_name, local_obj_name in name_pairs:
            try:
                module = importlib.import_module(module_name) # may raise ImportError
                get_obj_from_module(module, local_obj_name) # may raise AttributeError
                return module, local_obj_name
            except:
                pass

        # maybe some of the modules themselves contain errors?
        for module_name, _local_obj_name in name_pairs:
            try:
                importlib.import_module(module_name) # may raise ImportError
            except ImportError:
                if not str(sys.exc_info()[1]).startswith("No module named '" + module_name + "'"):
                    raise

        # maybe the requested attribute is missing?
        for module_name, local_obj_name in name_pairs:
            try:
                module = importlib.import_module(module_name) # may raise ImportError
                get_obj_from_module(module, local_obj_name) # may raise AttributeError
            except ImportError:
                pass

        # we are out of luck, but we have no idea why
        raise ImportError(obj_name)

    @staticmethod
    def get_obj_from_module(module: types.ModuleType, obj_name: str) -> Any:
        """Traverses the object name and returns the last (rightmost) python object."""
        if obj_name == '':
            return module
        obj = module
        for part in obj_name.split("."):
            obj = getattr(obj, part)
        return obj

    @staticmethod
    def get_obj_by_name(name: str) -> Any:
        """Finds the python object with the given name."""
        module, obj_name = Onrolux.get_module_from_obj_name(name)
        return get_obj_from_module(module, obj_name)

    @staticmethod
    def call_func_by_name(*args, func_name: str = None, **kwargs) -> Any:
        """Finds the python object with the given name and calls it as a function."""
        assert func_name is not None
        func_obj = get_obj_by_name(func_name)
        assert callable(func_obj)
        return func_obj(*args, **kwargs)

    @staticmethod
    def get_module_dir_by_obj_name(obj_name: str) -> str:
        """Get the directory path of the module containing the given object name."""
        module, _ = Onrolux.get_module_from_obj_name(obj_name)
        return os.path.dirname(inspect.getfile(module))

    @staticmethod
    def is_top_level_function(obj: Any) -> bool:
        """Determine whether the given object is a top-level function, i.e., defined at module scope using 'def'."""
        return callable(obj) and obj.__name__ in sys.modules[obj.__module__].__dict__

    @staticmethod
    def get_top_level_function_name(obj: Any) -> str:
        """Return the fully-qualified name of a top-level function."""
        assert is_top_level_function(obj)
        return obj.__module__ + "." + obj.__name__

    # rolux/stylegan2encoder/dnnlib/tflib/tfutil.py
    def set_vars(var_to_value_dict: dict) -> None:
        [tf.assign(var, value) for var, value in var_to_value_dict.items()]

    def create_var_with_large_initial_value(initial_value: np.ndarray, *args, **kwargs):
        """Create tf.Variable with large initial value without bloating the tf graph."""
        assert isinstance(initial_value, np.ndarray)
        zeros = tf.zeros(initial_value.shape, initial_value.dtype)
        var = tf.Variable(zeros, *args, **kwargs)
        # Onrolux.set_vars({var: initial_value})
        return var


#   ******************
#   ONMOONO
#
#   ******************
class Onmoono:

#   ******************
#
#    -> https://github.com/moono/stylegan2-tf-2.x/tree/master/stylegan2/upfirdn_2d.py <-
#
#   ******************
    @staticmethod
    def setup_resample_kernel(k):
        k = np.asarray(k, dtype=np.float32)
        if k.ndim == 1:
            k = np.outer(k, k)
        k /= np.sum(k)
        return k

    @staticmethod
    def upfirdn_ref(x, k, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1):
        in_height, in_width = tf.shape(x)[1], tf.shape(x)[2]
        minor_dim = tf.shape(x)[3]
        kernel_h, kernel_w = k.shape

        # Upsample (insert zeros).
        x = tf.reshape(x, [-1, in_height, 1, in_width, 1, minor_dim])
        x = tf.pad(x, [[0, 0], [0, 0], [0, up_y - 1], [0, 0], [0, up_x - 1], [0, 0]])
        x = tf.reshape(x, [-1, in_height * up_y, in_width * up_x, minor_dim])

        # Pad (crop if negative).
        x = tf.pad(x, [
            [0, 0], 
            [tf.math.maximum(pad_y0, 0), tf.math.maximum(pad_y1, 0)], 
            [tf.math.maximum(pad_x0, 0), tf.math.maximum(pad_x1, 0)], 
            [0, 0]
        ])
        x = x[:, tf.math.maximum(-pad_y0, 0): tf.shape(x)[1] - tf.math.maximum(-pad_y1, 0),
            tf.math.maximum(-pad_x0, 0): tf.shape(x)[2] - tf.math.maximum(-pad_x1, 0), :]

        # Convolve with filter.
        x = tf.transpose(x, [0, 3, 1, 2])
        x = tf.reshape(x, [-1, 1, in_height * up_y + pad_y0 + pad_y1, in_width * up_x + pad_x0 + pad_x1])
        w = tf.constant(k[::-1, ::-1, np.newaxis, np.newaxis], dtype=x.dtype)
        x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='VALID', data_format='NCHW')
        x = tf.reshape(x, [-1,
                        minor_dim,
                        in_height * up_y + pad_y0 + pad_y1 - kernel_h + 1,
                        in_width * up_x + pad_x0 + pad_x1 - kernel_w + 1])
        x = tf.transpose(x, [0, 2, 3, 1])

        # Downsample (throw away pixels).
        return x[:, ::down_y, ::down_x, :]

    @staticmethod
    def simple_upfirdn_2d(x, k, up=1, down=1, pad0=0, pad1=0):
        output_channel = tf.shape(x)[1]
        x = tf.reshape(x, [-1, tf.shape(x)[2], tf.shape(x)[3], 1])
        x = Onmoono.upfirdn_ref(x, k,
                        up_x=up, up_y=up, down_x=down, down_y=down, pad_x0=pad0, pad_x1=pad1, pad_y0=pad0, pad_y1=pad1)
        x = tf.reshape(x, [-1, output_channel, tf.shape(x)[1], tf.shape(x)[2]])
        return x

    @staticmethod
    def upsample_conv_2d(x, k, weight, factor, gain):
        x_height, x_width = tf.shape(x)[2], tf.shape(x)[3]
        w_height, w_width = tf.shape(weight)[0], tf.shape(weight)[1]
        w_ic, w_oc = tf.shape(weight)[2], tf.shape(weight)[3]

        # Setup filter kernel.
        k = k * (gain * (factor ** 2))
        p = (k.shape[0] - factor) - (w_width - 1)
        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2 + 1

        # Determine data dimensions.
        strides = [1, 1, factor, factor]
        output_shape = [1, w_oc, (x_height - 1) * factor + w_height, (x_width - 1) * factor + w_width]
        num_groups = tf.shape(x)[1] // w_ic

        # Transpose weights.
        weight = tf.reshape(weight, [w_height, w_width, w_ic, num_groups, -1])
        weight = tf.transpose(weight[::-1, ::-1], [0, 1, 4, 3, 2])
        weight = tf.reshape(weight, [w_height, w_width, -1, num_groups * w_ic])

        # Execute.
        x = tf.nn.conv2d_transpose(x, weight, output_shape, strides, padding='VALID', data_format='NCHW')
        x = Onmoono.simple_upfirdn_2d(x, k, pad0=pad0, pad1=pad1)
        return x

    @staticmethod
    def conv_downsample_2d(x, k, weight, factor, gain):
        w_height, w_width = tf.shape(weight)[0], tf.shape(weight)[1]

        # Setup filter kernel.
        k = k * gain
        p = (k.shape[0] - factor) + (w_width - 1)
        pad0 = (p + 1) // 2
        pad1 = p // 2

        strides = [1, 1, factor, factor]
        x = Onmoono.simple_upfirdn_2d(x, k, pad0=pad0, pad1=pad1)
        x = tf.nn.conv2d(x, weight, strides, padding='VALID', data_format='NCHW')
        return x

    @staticmethod
    def upsample_2d(x, k, factor, gain):
        # Setup filter kernel.
        k = k * (gain * (factor ** 2))
        p = k.shape[0] - factor
        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        x = Onmoono.simple_upfirdn_2d(x, k, up=factor, pad0=pad0, pad1=pad1)
        return x

    @staticmethod
    def downsample_2d(x, k, factor, gain):
        # Setup filter kernel.
        k = k * gain
        p = k.shape[0] - factor
        pad0 = (p + 1) // 2
        pad1 = p // 2

        x = Onmoono.simple_upfirdn_2d(x, k, down=factor, pad0=pad0, pad1=pad1)
        return x


#   ******************
#
#    -> https://github.com/moono/stylegan2-tf-2.x/tree/master/stylegan2/custom_layers.py <-
#
#   ******************
    @staticmethod
    def compute_runtime_coef(weight_shape, gain, lrmul):
        fan_in = np.prod(weight_shape[:-1])  # [kernel, kernel, fmaps_in, fmaps_out] or [in, out]
        he_std = gain / np.sqrt(fan_in)
        init_std = 1.0 / lrmul
        runtime_coef = he_std * lrmul
        return init_std, runtime_coef


#   ******************
#
#    -> https://github.com/moono/stylegan2-tf-2.x/blob/master/stylegan2/utils.py <-
#
#   ******************
    @staticmethod
    def lerp(a, b, t):
        out = a + (b - a) * t
        return out

    @staticmethod
    def lerp_clip(a, b, t):
        out = a + (b - a) * tf.clip_by_value(t, 0.0, 1.0)
        return out

    @staticmethod
    def adjust_dynamic_range(images, range_in, range_out, out_dtype):
        scale = (range_out[1] - range_out[0]) / (range_in[1] - range_in[0])
        bias = range_out[0] - range_in[0] * scale
        images = images * scale + bias
        images = tf.clip_by_value(images, range_out[0], range_out[1])
        images = tf.cast(images, dtype=out_dtype)
        return images

    @staticmethod
    def random_flip_left_right_nchw(images):
        s = tf.shape(images)
        mask = tf.random.uniform([s[0], 1, 1, 1], 0.0, 1.0)
        mask = tf.tile(mask, [1, s[1], s[2], s[3]])
        images = tf.where(mask < 0.5, images, tf.reverse(images, axis=[3]))
        return images

    @staticmethod
    def preprocess_fit_train_image(images, res):
        images = Onmoono.adjust_dynamic_range(images, range_in=(0.0, 255.0), range_out=(-1.0, 1.0), out_dtype=tf.dtypes.float32)
        images = Onmoono.random_flip_left_right_nchw(images)
        images.set_shape([None, 3, res, res])
        return images

    @staticmethod
    def postprocess_images(images):
        images = Onmoono.adjust_dynamic_range(images, range_in=(-1.0, 1.0), range_out=(0.0, 255.0), out_dtype=tf.dtypes.float32)
        images = tf.transpose(images, [0, 2, 3, 1])
        images = tf.cast(images, dtype=tf.dtypes.uint8)
        return images

    @staticmethod
    def merge_batch_images(images, res, rows, cols):
        batch_size = images.shape[0]
        assert rows * cols == batch_size
        canvas = np.zeros(shape=[res * rows, res * cols, 3], dtype=np.uint8)
        for row in range(rows):
            y_start = row * res
            for col in range(cols):
                x_start = col * res
                index = col + row * cols
                canvas[y_start:y_start + res, x_start:x_start + res, :] = images[index, :, :, :]
        return canvas

#   ******************
#
#    -> https://github.com/moono/stylegan2-tf-2.x/blob/master/train.py <-
#
#   ******************
    @staticmethod
    def filter_resolutions_featuremaps(resolutions, featuremaps, res):
        index = resolutions.index(res)
        filtered_resolutions = resolutions[:index + 1]
        filtered_featuremaps = featuremaps[:index + 1]
        return filtered_resolutions, filtered_featuremaps

#   ******************
#   Moono
#   https://github.com/moono/stylegan2-encoder/blob/master/utils.py
#   https://github.com/moono/stylegan2-tf-2.x/tree/master/stylegan2/tf_utils/utils.py
#
#   ******************
    @staticmethod
    def allow_memory_growth():
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
        return

    @staticmethod
    def adjust_dynamic_range(images, range_in, range_out, out_dtype):
        scale = (range_out[1] - range_out[0]) / (range_in[1] - range_in[0])
        bias = range_out[0] - range_in[0] * scale
        images = images * scale + bias
        images = tf.clip_by_value(images, range_out[0], range_out[1])
        images = tf.cast(images, dtype=out_dtype)
        return images


#   ******************
#   Moono
#   https://github.com/lllyasviel/DanbooRegion/
#   see licensing terms in https://github.com/lllyasviel/DanbooRegion
#
#   ******************
#
#   ******************
class Onlllyas:
    #   https://github.com/lllyasviel/DanbooRegion/blob/master/code/tricks.py
    @staticmethod
    def thinning(fillmap, max_iter=100):
        """Fill area of line with surrounding fill color.
        # Arguments
            fillmap: an image.
            max_iter: max iteration number.
        # Returns
            an image.
        """
        line_id = 0
        h, w = fillmap.shape[:2]
        result = fillmap.copy()

        for iterNum in range(max_iter):
            # Get points of line. if there is not point, stop.
            line_points = np.where(result == line_id)
            if not len(line_points[0]) > 0:
                break

            # Get points between lines and fills.
            line_mask = np.full((h, w), 255, np.uint8)
            line_mask[line_points] = 0
            line_border_mask = cv2.morphologyEx(line_mask, cv2.MORPH_DILATE,
                                                cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), anchor=(-1, -1),
                                                iterations=1) - line_mask
            line_border_points = np.where(line_border_mask == 255)

            result_tmp = result.copy()
            # Iterate over points, fill each point with nearest fill's id.
            for i, _ in enumerate(line_border_points[0]):
                x, y = line_border_points[1][i], line_border_points[0][i]

                if x - 1 > 0 and result[y][x - 1] != line_id:
                    result_tmp[y][x] = result[y][x - 1]
                    continue

                if x - 1 > 0 and y - 1 > 0 and result[y - 1][x - 1] != line_id:
                    result_tmp[y][x] = result[y - 1][x - 1]
                    continue

                if y - 1 > 0 and result[y - 1][x] != line_id:
                    result_tmp[y][x] = result[y - 1][x]
                    continue

                if y - 1 > 0 and x + 1 < w and result[y - 1][x + 1] != line_id:
                    result_tmp[y][x] = result[y - 1][x + 1]
                    continue

                if x + 1 < w and result[y][x + 1] != line_id:
                    result_tmp[y][x] = result[y][x + 1]
                    continue

                if x + 1 < w and y + 1 < h and result[y + 1][x + 1] != line_id:
                    result_tmp[y][x] = result[y + 1][x + 1]
                    continue

                if y + 1 < h and result[y + 1][x] != line_id:
                    result_tmp[y][x] = result[y + 1][x]
                    continue

                if y + 1 < h and x - 1 > 0 and result[y + 1][x - 1] != line_id:
                    result_tmp[y][x] = result[y + 1][x - 1]
                    continue

            result = result_tmp.copy()

        return result

    @staticmethod
    def topo_compute_normal(dist):
        c = cv2.filter2D(dist, cv2.CV_32F, np.array([[-1, +1]]))
        r = cv2.filter2D(dist, cv2.CV_32F, np.array([[-1], [+1]]))
        h = np.zeros_like(c + r, dtype=np.float32) + 0.75
        normal_map = np.stack([h, r, c], axis=2)
        normal_map /= np.sum(normal_map ** 2.0, axis=2, keepdims=True) ** 0.5
        return normal_map


    @staticmethod    
    #@njit
    def count_all(labeled_array, all_counts):
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
                if i > -1:
                    all_counts[i] = all_counts[i] + 1
        return


    @staticmethod    
    @njit
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

    @staticmethod
    def find_all(labeled_array):
        hist_size = int(np.max(labeled_array))
        if hist_size == 0:
            return []
        all_counts = [0 for _ in range(hist_size)]
        Onlllyas.count_all(labeled_array, all_counts)
        xs = [np.zeros(shape=(item, ), dtype=np.uint32) for item in all_counts]
        ys = [np.zeros(shape=(item, ), dtype=np.uint32) for item in all_counts]
        cs = [0 for item in all_counts]
        Onlllyas.trace_all(labeled_array, xs, ys, cs)
        filled_area = []
        for _ in range(hist_size):
            filled_area.append((xs[_], ys[_]))
        return filled_area

    @staticmethod
    def mk_resize(x, k):
        if x.shape[0] < x.shape[1]:
            s0 = k
            s1 = int(x.shape[1] * (k / x.shape[0]))
            s1 = s1 - s1 % 128
            _s0 = 32 * s0
            _s1 = int(x.shape[1] * (_s0 / x.shape[0]))
            _s1 = (_s1 + 64) - (_s1 + 64) % 128
        else:
            s1 = k
            s0 = int(x.shape[0] * (k / x.shape[1]))
            s0 = s0 - s0 % 128
            _s1 = 32 * s1
            _s0 = int(x.shape[0] * (_s1 / x.shape[1]))
            _s0 = (_s0 + 64) - (_s0 + 64) % 128
        new_min = min(_s1, _s0)
        raw_min = min(x.shape[0], x.shape[1])
        if new_min < raw_min:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_LANCZOS4
        y = cv2.resize(x, (_s1, _s0), interpolation=interpolation)
        return y

    @staticmethod
    def k_resize(x, k):
        if x.shape[0] < x.shape[1]:
            s0 = k
            s1 = int(x.shape[1] * (k / x.shape[0]))
            s1 = s1 - s1 % 64
            _s0 = 16 * s0
            _s1 = int(x.shape[1] * (_s0 / x.shape[0]))
            _s1 = (_s1 + 32) - (_s1 + 32) % 64
        else:
            s1 = k
            s0 = int(x.shape[0] * (k / x.shape[1]))
            s0 = s0 - s0 % 64
            _s1 = 16 * s1
            _s0 = int(x.shape[0] * (_s1 / x.shape[1]))
            _s0 = (_s0 + 32) - (_s0 + 32) % 64
        new_min = min(_s1, _s0)
        raw_min = min(x.shape[0], x.shape[1])
        if new_min < raw_min:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_LANCZOS4
        y = cv2.resize(x, (_s1, _s0), interpolation=interpolation)
        return y

    @staticmethod
    def sk_resize(x, k):
        if x.shape[0] < x.shape[1]:
            s0 = k
            s1 = int(x.shape[1] * (k / x.shape[0]))
            s1 = s1 - s1 % 16
            _s0 = 4 * s0
            _s1 = int(x.shape[1] * (_s0 / x.shape[0]))
            _s1 = (_s1 + 8) - (_s1 + 8) % 16
        else:
            s1 = k
            s0 = int(x.shape[0] * (k / x.shape[1]))
            s0 = s0 - s0 % 16
            _s1 = 4 * s1
            _s0 = int(x.shape[0] * (_s1 / x.shape[1]))
            _s0 = (_s0 + 8) - (_s0 + 8) % 16
        new_min = min(_s1, _s0)
        raw_min = min(x.shape[0], x.shape[1])
        if new_min < raw_min:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_LANCZOS4
        y = cv2.resize(x, (_s1, _s0), interpolation=interpolation)
        return y

    @staticmethod
    def d_resize(x, d, fac=1.0):
        new_min = min(int(d[1] * fac), int(d[0] * fac))
        raw_min = min(x.shape[0], x.shape[1])
        if new_min < raw_min:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_LANCZOS4
        y = cv2.resize(x, (int(d[1] * fac), int(d[0] * fac)), interpolation=interpolation)
        return y

    @staticmethod
    def n_resize(x, d):
        y = cv2.resize(x, (d[1], d[0]), interpolation=cv2.INTER_NEAREST)
        return y

    @staticmethod
    def s_resize(x, s):
        if x.shape[0] < x.shape[1]:
            s0 = x.shape[0]
            s1 = int(float(s0) / float(s[0]) * float(s[1]))
        else:
            s1 = x.shape[1]
            s0 = int(float(s1) / float(s[1]) * float(s[0]))
        new_max = max(s1, s0)
        raw_max = max(x.shape[0], x.shape[1])
        if new_max < raw_max:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_LANCZOS4
        y = cv2.resize(x, (s1, s0), interpolation=interpolation)
        return y

    @staticmethod
    def min_resize(x, m):
        if x.shape[0] < x.shape[1]:
            s0 = m
            s1 = int(float(m) / float(x.shape[0]) * float(x.shape[1]))
        else:
            s0 = int(float(m) / float(x.shape[1]) * float(x.shape[0]))
            s1 = m
        new_max = max(s1, s0)
        raw_max = max(x.shape[0], x.shape[1])
        if new_max < raw_max:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_LANCZOS4
        y = cv2.resize(x, (s1, s0), interpolation=interpolation)
        return y

    @staticmethod
    def n_min_resize(x, m):
        if x.shape[0] < x.shape[1]:
            s0 = m
            s1 = int(float(m) / float(x.shape[0]) * float(x.shape[1]))
        else:
            s0 = int(float(m) / float(x.shape[1]) * float(x.shape[0]))
            s1 = m
        new_max = max(s1, s0)
        raw_max = max(x.shape[0], x.shape[1])
        y = cv2.resize(x, (s1, s0), interpolation=cv2.INTER_NEAREST)
        return y

    @staticmethod
    def h_resize(x, m):
        s0 = m
        s1 = int(float(m) / float(x.shape[0]) * float(x.shape[1]))
        new_max = max(s1, s0)
        raw_max = max(x.shape[0], x.shape[1])
        if new_max < raw_max:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_LANCZOS4
        y = cv2.resize(x, (s1, s0), interpolation=interpolation)
        return y

    @staticmethod
    def w_resize(x, m):
        s0 = int(float(m) / float(x.shape[1]) * float(x.shape[0]))
        s1 = m
        new_max = max(s1, s0)
        raw_max = max(x.shape[0], x.shape[1])
        if new_max < raw_max:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_LANCZOS4
        y = cv2.resize(x, (s1, s0), interpolation=interpolation)
        return y

    @staticmethod
    def max_resize(x, m):
        if x.shape[0] > x.shape[1]:
            s0 = m
            s1 = int(float(m) / float(x.shape[0]) * float(x.shape[1]))
        else:
            s0 = int(float(m) / float(x.shape[1]) * float(x.shape[0]))
            s1 = m
        new_max = max(s1, s0)
        raw_max = max(x.shape[0], x.shape[1])
        if new_max < raw_max:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_LANCZOS4
        y = cv2.resize(x, (s1, s0), interpolation=interpolation)
        return y

    @staticmethod
    def vis(region_map, color_map):
        color = Onlllyas.d_resize(color_map, region_map.shape)
        indexs = (region_map.astype(np.float32)[:, :, 0] * 255 + region_map.astype(np.float32)[:, :, 1]) * 255 + region_map.astype(np.float32)[:, :, 2]
        result = np.zeros_like(color, dtype=np.uint8)
        for ids in [np.where(indexs == idsn) for idsn in np.unique(indexs).tolist()]:
            result[ids] = np.median(color[ids], axis=0)
        return result


    # https://github.com/lllyasviel/DanbooRegion/blob/master/code/skeletonize.py
    @staticmethod   
    def get_skeleton(region_map, filterstrength=5.0):
        from skimage.morphology import skeletonize, dilation    
        Xp = np.pad(region_map, [[0, 1], [0, 0], [0, 0]], 'symmetric').astype(np.float32)
        Yp = np.pad(region_map, [[0, 0], [0, 1], [0, 0]], 'symmetric').astype(np.float32)
        X = np.sum((Xp[1:, :, :] - Xp[:-1, :, :]) ** 2.0, axis=2) ** 0.5
        Y = np.sum((Yp[:, 1:, :] - Yp[:, :-1, :]) ** 2.0, axis=2) ** 0.5
        edge = np.zeros_like(region_map)[:, :, 0]
        edge[X > 0] = 255
        edge[Y > 0] = 255
        edge[0, :] = 255
        edge[-1, :] = 255
        edge[:, 0] = 255
        edge[:, -1] = 255
        skeleton = 1.0 - dilation(edge.astype(np.float32) / 255.0)
        skeleton = skeletonize(skeleton)
        skeleton = (skeleton * 255.0).clip(0, 255).astype(np.uint8)
        field = np.random.uniform(low=0.0, high=255.0, size=edge.shape).clip(0, 255).astype(np.uint8)
        field[skeleton > 0] = 255
        field[edge > 0] = 0
        filter = np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]],
            dtype=np.float32) / filterstrength
        height = np.random.uniform(low=0.0, high=255.0, size=field.shape).astype(np.float32)
        for _ in range(512):
            height = cv2.filter2D(height, cv2.CV_32F, filter)
            height[skeleton > 0] = 255.0
            height[edge > 0] = 0.0
        return height.clip(0, 255).astype(np.uint8)


    # https://github.com/lllyasviel/DanbooRegion/blob/master/code/skeleton2regions.py
    @staticmethod 
    def skeleton_to_regions(skeleton_map): # get_regions
        marker = skeleton_map[:, :, 0]
        normal = Onlllyas.topo_compute_normal(marker) * 127.5 + 127.5
        marker[marker > 100] = 255
        marker[marker < 255] = 0
        labels, nil = label(marker / 255)
        water = cv2.watershed(normal.clip(0, 255).astype(np.uint8), labels.astype(np.int32)) + 1
        water = Onlllyas.thinning(water)
        all_region_indices = Onlllyas.find_all(water)
        regions = np.zeros_like(skeleton_map, dtype=np.uint8)
        for region_indices in all_region_indices:
            regions[region_indices] = np.random.randint(low=0, high=255, size=(3,)).clip(0, 255).astype(np.uint8)
        return regions


    # https://github.com/lllyasviel/DanbooRegion/blob/master/code/rotate.py
    @staticmethod 
    def rotate_image(image, angle):
        """
        Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
        (in degrees). The returned image will be large enough to hold the entire
        new image, with a black background
        """

        # Get the image size
        # No that's not an error - NumPy stores image matricies backwards
        image_size = (image.shape[1], image.shape[0])
        image_center = tuple(np.array(image_size) / 2)

        # Convert the OpenCV 3x2 rotation matrix to 3x3
        rot_mat = np.vstack(
            [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
        )

        rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

        # Shorthand for below calcs
        image_w2 = image_size[0] * 0.5
        image_h2 = image_size[1] * 0.5

        # Obtain the rotated coordinates of the image corners
        rotated_coords = [
            (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
            (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
            (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
            (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
        ]

        # Find the size of the new image
        x_coords = [pt[0] for pt in rotated_coords]
        x_pos = [x for x in x_coords if x > 0]
        x_neg = [x for x in x_coords if x < 0]

        y_coords = [pt[1] for pt in rotated_coords]
        y_pos = [y for y in y_coords if y > 0]
        y_neg = [y for y in y_coords if y < 0]

        right_bound = max(x_pos)
        left_bound = min(x_neg)
        top_bound = max(y_pos)
        bot_bound = min(y_neg)

        new_w = int(abs(right_bound - left_bound))
        new_h = int(abs(top_bound - bot_bound))

        # We require a translation matrix to keep the image centred
        trans_mat = np.matrix([
            [1, 0, int(new_w * 0.5 - image_w2)],
            [0, 1, int(new_h * 0.5 - image_h2)],
            [0, 0, 1]
        ])

        # Compute the tranform for the combined rotation and translation
        affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

        # Apply the transform
        result = cv2.warpAffine(
            image,
            affine_mat,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR
        )

        return result

    @staticmethod 
    def largest_rotated_rect(w, h, angle):
        """
        Given a rectangle of size wxh that has been rotated by 'angle' (in
        radians), computes the width and height of the largest possible
        axis-aligned rectangle within the rotated rectangle.
        Original JS code by 'Andri' and Magnus Hoff from Stack Overflow
        Converted to Python by Aaron Snoswell
        """

        quadrant = int(math.floor(angle / (math.pi / 2))) & 3
        sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
        alpha = (sign_alpha % math.pi + math.pi) % math.pi

        bb_w = w * math.cos(alpha) + h * math.sin(alpha)
        bb_h = w * math.sin(alpha) + h * math.cos(alpha)

        gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

        delta = math.pi - alpha - gamma

        length = h if (w < h) else w

        d = length * math.cos(alpha)
        a = d * math.sin(alpha) / math.sin(delta)

        y = a * math.cos(gamma)
        x = y * math.tan(gamma)

        return (
            bb_w - 2 * x,
            bb_h - 2 * y
        )

    @staticmethod 
    def crop_around_center(image, width, height):
        """
        Given a NumPy / OpenCV 2 image, crops it to the given width and height,
        around it's centre point
        """

        image_size = (image.shape[1], image.shape[0])
        image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

        if(width > image_size[0]):
            width = image_size[0]

        if(height > image_size[1]):
            height = image_size[1]

        x1 = int(image_center[0] - width * 0.5)
        x2 = int(image_center[0] + width * 0.5)
        y1 = int(image_center[1] - height * 0.5)
        y2 = int(image_center[1] + height * 0.5)

        return image[y1:y2, x1:x2]


    # https://github.com/lllyasviel/DanbooRegion/blob/master/code/datasets.py

    @staticmethod 
    def np_RGB2GRAY(img):
        R = img[:, :, 0].astype(np.float32)
        G = img[:, :, 1].astype(np.float32)
        B = img[:, :, 2].astype(np.float32)
        r = np.random.rand()
        g = np.random.rand()
        b = np.random.rand()
        s = r + g + b
        r /= s
        g /= s
        b /= s
        light = R * r + G * g + B * b
        light -= np.min(light)
        light /= np.max(light)
        a = np.random.rand() * 0.4
        b = 1 - np.random.rand() * 0.4
        light = light.clip(a, b)
        light -= np.min(light)
        light /= np.max(light)
        light = light.clip(0, 1)
        light = (light * 255.0).astype(np.uint8)
        return light

    @staticmethod 
    def handle_next():
        indice = np.random.randint(low=0, high=3377)
        paint_r_mat = cv2.imread('./DanbooRegion2020/train/' + str(indice) + '.image.png')
        sketch_r_mat = cv2.imread('./DanbooRegion2020/train/' + str(indice) + '.skeleton.png')

        image_height, image_width = sketch_r_mat.shape[0:2]
        paint_r_mat = Onlllyas.d_resize(paint_r_mat, sketch_r_mat.shape)

        if np.random.rand() < 0.5:
            ri = np.random.rand() * 360.0
            paint_r_mat = crop_around_center(rotate_image(paint_r_mat, ri), *largest_rotated_rect(image_width, image_height, math.radians(ri)))
            sketch_r_mat = crop_around_center(rotate_image(sketch_r_mat, ri), *largest_rotated_rect(image_width, image_height, math.radians(ri)))
            kernel = np.random.randint(520, 650)
        else:
            kernel = np.random.randint(520, 1024)
        raw_s0 = float(paint_r_mat.shape[0])
        raw_s1 = float(paint_r_mat.shape[1])
        if raw_s0 < raw_s1:
            new_s0 = int(kernel)
            new_s1 = int(kernel / raw_s0 * raw_s1)
        else:
            new_s1 = int(kernel)
            new_s0 = int(kernel / raw_s1 * raw_s0)
        c0 = int(np.random.rand() * float(new_s0 - 512))
        c1 = int(np.random.rand() * float(new_s1 - 512))
        paint_mat = Onlllyas.d_resize(paint_r_mat, (new_s0, new_s1))[c0:c0 + 512, c1:c1 + 512, :]
        sketch_mat = Onlllyas.d_resize(sketch_r_mat, (new_s0, new_s1))[c0:c0 + 512, c1:c1 + 512, 0:1]

        if np.random.rand() < 0.5:
            sketch_mat = np.fliplr(sketch_mat)
            paint_mat = np.fliplr(paint_mat)

        if np.random.rand() < 0.5:
            sketch_mat = np.flipud(sketch_mat)
            paint_mat = np.flipud(paint_mat)

        if np.random.rand() < 0.5:
            paint_mat = np.stack([np_RGB2GRAY(paint_mat), np_RGB2GRAY(paint_mat), np_RGB2GRAY(paint_mat)], axis=2)

        if np.random.rand() < 0.5:
            for _ in range(int(np.random.randint(low=0, high=5))):
                paint_mat = cv2.GaussianBlur(paint_mat, (0, 0), 2.0)

        if np.random.rand() < 0.5:
            for _ in range(int(np.random.randint(low=0, high=5))):
                paint_mat = cv2.medianBlur(paint_mat, 3)

        return sketch_mat, paint_mat

    @staticmethod 
    def handle_batch(batch_size=4):
        sketch_batch = []
        paint_batch = []
        for _ in range(batch_size):
            sketch_mat, paint_mat = handle_next()
            sketch_batch.append(sketch_mat)
            paint_batch.append(paint_mat)
        sketch_batch = np.stack(sketch_batch, axis=0)
        paint_batch = np.stack(paint_batch, axis=0)
        return sketch_batch, paint_batch


    # https://github.com/lllyasviel/DanbooRegion/blob/master/code/segment.py
    @staticmethod     
    def go_flipped_vector(x):
        #a = go_vector(x)
        #b = np.fliplr(go_vector(np.fliplr(x)))
        #c = np.flipud(go_vector(np.flipud(x)))
        #d = np.flipud(np.fliplr(go_vector(np.flipud(np.fliplr(x)))))
        print(f'go_flipped_vector: {type(x)} {np.shape(x)}')
        a = x # <class 'numpy.ndarray'> (2944, 2048, 3)
        b = np.fliplr(np.fliplr(x))
        c = np.flipud(np.flipud(x))
        d = np.flipud(np.fliplr(np.flipud(np.fliplr(x))))
        return (a + b + c + d) / 4.0

    @staticmethod 
    def go_transposed_vector(x):
        a = Onlllyas.go_flipped_vector(x)
        b = np.transpose(Onlllyas.go_flipped_vector(np.transpose(x, [1, 0, 2])), [1, 0, 2])
        return (a + b) / 2.0


    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html
    from scipy.ndimage import label

    @staticmethod 
    def get_fill(image):
        labeled_array, num_features = label(image / 255)
        # num_features: 357
        # labeled_array: <class 'numpy.ndarray'> (2048, 2048, 3)
        # labeled_array: [[[0 0 1] [0 0 1] [0 0 1] ...
        filled_area = Onlllyas.find_all(labeled_array)
        return filled_area

    @staticmethod 
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

    @staticmethod 
    def segment(image):
        #raw_img = go_srcnn(Onlllyas.min_resize(image, 512)).clip(0, 255).astype(np.uint8)
        raw_img = image
        img_2048 = Onlllyas.min_resize(raw_img, 2048)
        height = Onlllyas.d_resize(Onlllyas.go_transposed_vector(Onlllyas.mk_resize(raw_img, 64)), img_2048.shape) * 255.0
        final_height = height.copy()
        height += (height - cv2.GaussianBlur(height, (0, 0), 3.0)) * 10.0
        height = height.clip(0, 255).astype(np.uint8)
        marker = height.copy()
        marker[marker > 135] = 255
        marker[marker < 255] = 0
        fills = Onlllyas.get_fill(marker / 255)
        for fill in fills:
            if fill[0].shape[0] < 64:
                marker[fill] = 0
        filter = np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]],
            dtype=np.uint8)
        big_marker = cv2.erode(marker, filter, iterations=5)
        fills = Onlllyas.get_fill(big_marker / 255)
        for fill in fills:
            if fill[0].shape[0] < 64:
                big_marker[fill] = 0
        big_marker = cv2.dilate(big_marker, filter, iterations=5)
        small_marker = marker.copy()
        small_marker[big_marker > 127] = 0
        fin_labels, nil = label(big_marker / 255)
        fin_labels = up_fill(Onlllyas.get_fill(small_marker), fin_labels)
        water = cv2.watershed(img_2048.clip(0, 255).astype(np.uint8), fin_labels.astype(np.int32)) + 1
        water = Onlllyas.thinning(water)
        all_region_indices = Onlllyas.find_all(water)
        regions = np.zeros_like(img_2048, dtype=np.uint8)
        for region_indices in all_region_indices:
            regions[region_indices] = np.random.randint(low=0, high=255, size=(3,)).clip(0, 255).astype(np.uint8)
        result = np.zeros_like(img_2048, dtype=np.uint8)
        for region_indices in all_region_indices:
            result[region_indices] = np.median(img_2048[region_indices], axis=0)
        return final_height.clip(0, 255).astype(np.uint8), regions.clip(0, 255).astype(np.uint8), result.clip(0, 255).astype(np.uint8)



#   ******************
#   CLS ONVGG
#
#   ******************
class Onvgg:


    @staticmethod
    def vgg_deprocess(img):
        VGG_RGB_MEAN = [123.68, 116.78, 103.94]
        vgg_rgb_mean_arr = np.array(VGG_RGB_MEAN)

        imgpost = np.copy(img)
        imgpost += vgg_rgb_mean_arr.reshape((1,1,1,3))
        
        imgpost = imgpost[0] # (1, h, w, d) +> (h, w, d)
        imgpost = np.clip(imgpost, 0, 255).astype('uint8')
        
        imgpost = imgpost[...,::-1] # RGB => BGR

        return imgpost

    @staticmethod
    def vgg_preprocess(img):
        VGG_RGB_MEAN = [123.68, 116.779, 103.939]
        vgg_rgb_mean_arr = np.array(VGG_RGB_MEAN).reshape((1,1,1,3))

        imgpre = np.copy(img)
        imgpre = imgpre[...,::-1] # BGR to RGB
        imgpre = imgpre.astype(np.float32) # _e_

        imgpre = imgpre[np.newaxis,:,:,:] # (h, w, d) => (1, h, w, d)
        imgpre -= vgg_rgb_mean_arr

        return imgpre


    @staticmethod
    def path_to_tvgg(image_path): # _e_
        # Util function to open, resize and format pictures into appropriate tensors
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_nrows, img_ncols))
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = vgg19.preprocess_input(img)
        return tf.convert_to_tensor(img)

    @staticmethod   
    def cv2_trgb(img, title="img", wait=3000):
        img = Onformat.vgg_deprocess(img)
        cv_img(img, title, wait)

    @staticmethod
    def cv2_path(path): #  _e_
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = Onformat.vgg_preprocess(img)
        img = Onformat.vgg_deprocess(img)
        cv2.imshow('img', img)
        cv2.waitKey(2000)   

    @staticmethod
    def bgr_to_rgb_vgg(img):
        VGG_RGB_MEAN = [123.68, 116.779, 103.939]
        vgg_rgb_mean_arr = np.array(VGG_RGB_MEAN)

        imgpre = np.copy(img)
        # bgr to rgb
        imgpre = imgpre[...,::-1]
        # shape (h, w, d) to (1, h, w, d)
        imgpre = imgpre[np.newaxis,:,:,:]
        imgpre -= vgg_rgb_mean_arr.reshape((1,1,1,3))
        return imgpre

    @staticmethod
    def tbgr_to_trgb_vgg(img):
        # Keras works with batches of images
        # So, the first dimension is used for the number of samples (or images)
        # vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((3,1,1))
        VGG_BGR_MEAN = [103.939, 116.779, 123.68]
        vgg_bgr_mean_arr = np.array(VGG_BGR_MEAN)

        # img = img[0] # np.squeeze(img, axis = 0) # 

        img = img[...,::-1] # bgr to rgb
        # img = img[np.newaxis,:,:,:] # shape (h, w, d) to (1, h, w, d)
        img -= vgg_bgr_mean_arr.reshape((1,1,1,3))

        img = tf.convert_to_tensor(img)
        return img

    @staticmethod
    def bgr_to_tnua_2_vgg(img):
        VGG_RGB_MEAN = [123.68, 116.779, 103.939]
        vgg_rgb_mean_arr = np.array(VGG_RGB_MEAN)

        # bgr to rgb
        img = img[...,::-1]
        # shape (h, w, d) to (1, h, w, d)
        img = img[np.newaxis,:,:,:]
        img -= vgg_rgb_mean_arr.reshape((1,1,1,3))
        return img

    @staticmethod
    def tbgr_to_tnua_2_vgg(img):
        VGG_RGB_MEAN = [123.68, 116.779, 103.939]
        vgg_rgb_mean_arr = np.array(VGG_RGB_MEAN)    

        img += vgg_rgb_mean_arr.reshape((1,1,1,3))
        # shape (1, h, w, d) to (h, w, d)
        img = img[0]
        img = np.clip(img, 0, 255).astype('uint8')
        # rgb to bgr
        img = img[...,::-1]
        return img

    @staticmethod
    def bgrs_to_tnuas(inputs):
        pretensors = []
        for item in inputs:
            preitem = Onformat.tbgr_to_trgb_vgg(item)
            pretensors.append(preitem)
        return pretensors

    @staticmethod
    def bgrs_to_tnuas(inputs):
        pretensors = []
        for item in inputs:
            preitem = Onformat.tbgr_to_trgb_vgg(item)
            pretensors.append(preitem)
        return pretensors

    @staticmethod
    def tnua_to_vgg(tnua, width=512, height=512):
        tnua = tf.image.resize(tnua, (width, height))
        bgr = tnua*255.0 
        vgg = Onvgg.tbgr_to_trgb_vgg(bgr)
        return vgg



#   ******************
#    TFRecordExporter: export images to tf records
#       relux
#       https://github.com/rolux/stylegan2encoder/dataset_tool.py
#
#   ******************
class TFRecordExporter:
    def __init__(self, 
        tfrecord_dir, 
        expected_images, 
        tfr_prefix='tfrecord', 
        progress_interval=10,
        verbose = True):

        self.tfrecord_dir       = tfrecord_dir
        self.tfr_prefix         = tfr_prefix
        self.expected_images    = expected_images
        self.cur_images         = 0
        self.shape              = None
        self.resolution_log2    = None
        self.tfr_writers        = []
        self.verbose     = verbose
        self.progress_interval  = progress_interval

        if self.verbose:
            print(f'Creating dataset in {tfrecord_dir}')
        os.makedirs(tfrecord_dir, exist_ok=True)

    def close(self):
        if self.verbose:
            print('%-40s\r' % 'Flushing data...', end='', flush=True)
        for tfr_writer in self.tfr_writers:
            tfr_writer.close()
        self.tfr_writers = []
        if self.verbose:
            print('%-40s\r' % '', end='', flush=True)
            print('Added %d image%s.' % (self.cur_images, 's'[:self.cur_images > 1]))

    def choose_shuffled_order(self): # Note: Images and labels must be added in shuffled order.
        order = np.arange(self.expected_images)
        np.random.RandomState(123).shuffle(order)
        return order

    def add_image(self, img):

        if self.verbose and self.cur_images % self.progress_interval == 0:
            print('%d / %d\r' % (self.cur_images, self.expected_images), end='', flush=True)
        if self.shape is None:
            self.shape = img.shape
            self.resolution_log2 = int(np.log2(self.shape[1]))

            # print("add_image shape[1]: %s" %self.shape[1])
            # print("add_image resolution_log2: %s" %self.resolution_log2)

            assert self.shape[0] in [1, 3], f"shape[0] assertion failed"
            assert self.shape[1] == self.shape[2], f"shape[1]:shape[2] assertion failed"
            assert self.shape[1] == 2**self.resolution_log2, f"shape is pow(2) assertion failed"
            tfr_opt = tf.io.TFRecordOptions(compression_type = None)
            for lod in range(self.resolution_log2 - 1):
                tfr_file = os.path.join(self.tfrecord_dir, self.tfr_prefix + '-r%02d.tfrecords' % (self.resolution_log2 - lod))
                self.tfr_writers.append(tf.io.TFRecordWriter(tfr_file, tfr_opt))

        assert img.shape == self.shape, "image shapes are not equal"
        for lod, tfr_writer in enumerate(self.tfr_writers):
            if lod:
                img = img.astype(np.float32)
                img = (img[:, 0::2, 0::2] + img[:, 0::2, 1::2] + img[:, 1::2, 0::2] + img[:, 1::2, 1::2]) * 0.25

            # assume normalized input
            quant = np.rint(img * 255).clip(0, 255).astype(np.uint8) # img * 255  # _e_

            ex = tf.train.Example(features=tf.train.Features(feature={
                'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=quant.shape)),
                'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[quant.tostring()]))}))
                
            # https://www.tensorflow.org/tutorials/load_data/tfrecord 
            feature = ex.SerializeToString()
            tfr_writer.write(feature)

 
            example_proto = tf.train.Example.FromString(feature)           

            # feature_description = {
            #     'shape': tf.io.FixedLenFeature([3], tf.int64),
            #     'data': tf.io.FixedLenFeature([], tf.string),
            # }
            # def _parse_function(example_proto):
            #     # Parse the input `tf.Example` proto using the dictionary above.
            #     return tf.io.parse_single_example(example_proto, feature_description)
            # parsed_dataset = example_proto.map(_parse_function)
            # print(parsed_dataset)

            examples = []
            data = example_proto.features.feature["data"]
            shape = example_proto.features.feature["shape"]

            shape = shape.int64_list.value


            ndata = np.frombuffer(data.bytes_list.value[0], dtype=np.uint8)
            im_arr =  ndata.reshape(shape[0], shape[1], shape[2])

            # im_arr = im_arr.reshape(shape[2], shape[0], shape[1])
            # im_arr = im_arr / 255

            im_arr = im_arr.transpose([1, 2, 0])
            imgstoplot = []
            imgstoplot.append(Image.fromarray(im_arr))

            # plt.imshow(im_arr)
            # image = Image.fromarray(im_arr)
            # image = im_arr
            # plt.show()

        self.cur_images += 1

    def add_labels(self, labels):
        if self.verbose:
            print('%-40s\r' % 'Saving labels...', end='', flush=True)
        assert labels.shape[0] == self.cur_images
        with open(self.tfr_prefix + '-rxx.labels', 'wb') as f:
            np.save(f, labels.astype(np.float32))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()



