import os
cmd = f'pwd & \
    ls -la '
os.system(cmd)


cmd = f'conda create -n tf2 pip python=3.7'
os.system(cmd)

cmd = f'conda activate tf2'
os.system(cmd)

cmd = f'conda list -n tf2'
os.system(cmd)

cmd = f'pip install --upgrade pip'
os.system(cmd)

cmd = f'pip install --upgrade tensorflow'
os.system(cmd)

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
