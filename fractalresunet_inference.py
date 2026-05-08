import os
import cv2
import glob
import warnings
import numpy as np
import time
from osgeo import gdal
from osgeo import ogr
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef
import pandas as pd
import sys

sys.path.append('/data/Aldhani/cv_fields/code/decode/')
sys.path.append('/data/Aldhani/cv_fields/code/eo-fields/decode_ssa')
sys.path.append('/data/Aldhani/cv_fields/code/eo-fields/')
sys.path.append('/data/Aldhani/cv_fields/code/')

import higra as hg
from tqdm import tqdm

import mxnet as mx
from mxnet import nd, gpu, gluon, autograd, npx, image
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.loss import Loss

from decode_ssa.helpers_io import *
from decode_ssa.helpers_model import *
from decode_ssa.helpers_labels import *
from decode.FracTAL_ResUNet.nn.loss.mtsk_loss import *
from decode.FracTAL_ResUNet.models.heads.head_cmtsk import *
from decode.FracTAL_ResUNet.models.semanticsegmentation.FracTAL_ResUNet_features import *
from decode.FracTAL_ResUNet.models.semanticsegmentation.FracTAL_ResUNet import FracTAL_ResUNet_cmtsk

sys.path.insert(0,'/data/Aldhani/cv_fields/code/')
sys.path.append('/data/Aldhani/cv_fields/code/eo-fields/field_delineation/src')
sys.path.append('/data/Aldhani/cv_fields/code/eo-fields/decode/FracTAL_ResUNet/models/semanticsegmentation')
sys.path.append('/data/Aldhani/cv_fields/code/eo-fields/waldnerf/decode/FracTAL_ResUNet/nn/loss')
#sys.path.append('../../')

# set seeds
mx.random.seed(42)
np.random.seed(42)

model = '/data/Aldhani/cv_fields/models/fractal-resunet_100_from-scratch_GE_rgb_nfilter-32_depth-6_bs-12_lr-0.001_trainval-70-30_norm-none_lossftnmt_masked_moz_tst_03'

# D6nf32 example
depth=6
NClasses=1
nfilters_init=32
linear_norm = False
norm = 'none'

# load params
net = FracTAL_ResUNet_cmtsk(nfilters_init=nfilters_init, NClasses=NClasses,depth=depth)
net.load_parameters(f'{model}/model.params')
net.collect_params().reset_ctx(gpu(0))

# define inputs
img_path = '/data/Aldhani/cv_fields/images/descartes_tiles/inference/mozambique/2017/'
prd_path = f'/data/Aldhani/cv_fields/preds/descartes_tiles/inference/mozambique/{os.path.basename(model)}/'

print(prd_path)
if not os.path.exists(prd_path): os.mkdir(prd_path)
in_files = sorted(glob.glob(img_path + '*.tif'))
len(in_files)

write=True

size=1024
step=512

for in_file in in_files:
  tic = time.time()
  out_file = prd_path + os.path.basename(in_file)[:-4]+'_preds.tif'
  # open
  if not os.path.isfile(out_file):
    input = open_tif_rgb_tc(in_file)
    
    # write output?
    if write:
      # create memory copy of ds
      mem_ds = create_mem_ds(in_file, 3, dtype=gdal.GDT_Int16)

    for y in np.arange(0, input.shape[1], step):   
      for x in np.arange(0, input.shape[2], step): 
        
        in_tns = input[None,:,y:y+size,x:x+size]
        
        # check if in_tns all zero?
        if nd.sum(in_tns[:,0,:,:])>0:
          # predict
          preds = net(in_tns)

          if write:
            # write outputs to bands
            for b in range(3):
              bnd = np.squeeze(preds[b]).asnumpy()
              bnd = (bnd*10000).astype(int)
              
              if (y==0) & (x==0):
                bnd = bnd[:int(step*1.5),:int(step*1.5)]
                mem_ds.GetRasterBand(b+1).WriteArray(bnd, xoff=x, yoff=y)

              if (y==0) & (x>0):
                bnd = bnd[:int(step*1.5),int(step*0.5):int(step*1.5)]
                mem_ds.GetRasterBand(b+1).WriteArray(bnd, xoff=x+step*0.5, yoff=y)

              if (y>0) & (x==0):
                bnd = bnd[int(step*0.5):int(step*1.5),:int(step*1.5)]
                mem_ds.GetRasterBand(b+1).WriteArray(bnd, xoff=x, yoff=y+step*0.5)

              if (y>0) & (x>0) & (x<input.shape[2]-step) & (y<input.shape[1]-step):
                bnd = bnd[int(step*0.5):int(step*1.5),int(step*0.5):int(step*1.5)]
                mem_ds.GetRasterBand(b+1).WriteArray(bnd, xoff=x+step*0.5, yoff=y+step*0.5)
              
              # last row
              if (x==input.shape[2]-step) & (y!=input.shape[1]-step):
                bnd = bnd[int(step*0.5):,int(step*0.5):int(step*1.5)]
                mem_ds.GetRasterBand(b+1).WriteArray(bnd, xoff=x+step*0.5, yoff=y+step*0.5)
              
              # last col
              if (x!=input.shape[2]-step) & (y==input.shape[1]-step):
                bnd = bnd[int(step*0.5):int(step*1.5),int(step*0.5):]
                mem_ds.GetRasterBand(b+1).WriteArray(bnd, xoff=x+step*0.5, yoff=y+step*0.5)
              
              # last tile
              if (x==input.shape[2]-step) & (y==input.shape[1]-step):
                bnd = bnd[int(step*0.5):,int(step*0.5):]
                mem_ds.GetRasterBand(b+1).WriteArray(bnd, xoff=x+step*0.5, yoff=y+step*0.5)

    # create physical copy of ds
    copy_mem_ds(out_file, mem_ds)
    print(out_file)
      
    print(time.time()-tic)