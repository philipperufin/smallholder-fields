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

os.chdir('/data/Aldhani/cv_fields/')

# set seeds
mx.random.seed(42)
np.random.seed(42)

img_folder = 'images/descartes_tiles/mozambique/chips'
lbl_folder = 'labels/descartes_tiles/mozambique/percentile_and_abs_095/mtsk/chips'

if False: # old way of splitting data
    # get tile ids
    tile_ids = sorted(glob.glob(lbl_folder + '/*.tif'))
    len(tile_ids)

    image_names = sorted([f"{img_folder}/{os.path.basename(tile_id)[7:-15]}_{os.path.basename(tile_id)[-9:]}" for tile_id in tile_ids])
    label_names = sorted([f"{lbl_folder}/{os.path.basename(tile_id)}" for tile_id in tile_ids])

    m = [print(image_names[i], label_names[i]) for i in range(0,10)]
    print(len(tile_ids), len(image_names), len(label_names))

    # define split and if split should be conducted across tiles
    # allowing for leakage leads to more stable behaviour, blocking it makes
    # validation loss jump eratically
    probs=[0.60, 0.20, 0.20]

    ### randomly select w leakage across tiles
    np.random.seed(42)
    split = np.random.choice(3, len(tile_ids), replace=True, p=probs)
    train_image_names = np.array(image_names)[split==0]
    val_image_names = np.array(image_names)[split==1]
    test_image_names = np.array(image_names)[split==2]

    train_label_names = np.array(label_names)[split==0]
    val_label_names = np.array(label_names)[split==1]
    test_label_names = np.array(label_names)[split==2]

    print('train')
    print(len(train_image_names))
    print(train_label_names[:5])
    print(train_image_names[:5])

    print('val')
    print(len(val_image_names))
    print(val_label_names[:5])
    print(val_image_names[:5])

    print('test')
    print(len(test_image_names))
    print(test_label_names[:5])
    print(test_image_names[:5])

    print('N train, val, test')
    print(len(train_image_names), len(train_label_names), len(val_image_names), len(val_label_names), len(test_image_names), len(test_label_names))

if True: 
    # select training data based on label presence / quantity
    res = pd.read_csv('/data/Aldhani/cv_fields/labels/descartes_tiles/mozambique/percentile_and_abs_095/mtsk/chips/labels_percentage.csv')

    # positive label files
    sub = res[res['field_ext']>0]
    print(len(sub))
    # negative label files w 50% labels
    add = res[(res['ncrop_ext']>0.5) & (res['field_ext']==0)]
    print(len(add))
    # sample N% of negative label files w 50% labels
    add = add.sample(int(len(add)*0.05), random_state=42)
    print(len(add))
    fs = add['file']
    [print(f) for f in fs]
    # add
    sub = pd.concat([sub,add])
    print(len(sub))
    tile_ids = sub['file']

    image_names = sorted([f"{img_folder}/{os.path.basename(tile_id)[7:-15]}_{os.path.basename(tile_id)[-9:]}" for tile_id in tile_ids])
    label_names = sorted([f"{lbl_folder}/{os.path.basename(tile_id)}" for tile_id in tile_ids])

    #m = [print(image_names[i], label_names[i]) for i in range(0,10)]
    #print(len(tile_ids), len(image_names), len(label_names))

    # define split and if split should be conducted across tiles
    # allowing for leakage leads to more stable behaviour, blocking it makes
    # validation loss jump eratically
    probs=[0.70, 0.30]

    ### randomly select w leakage across tiles
    np.random.seed(42)

    split = np.random.choice(2, len(tile_ids), replace=True, p=probs)
    train_image_names = np.array(image_names)[split==0]
    val_image_names = np.array(image_names)[split==1]

    train_label_names = np.array(label_names)[split==0]
    val_label_names = np.array(label_names)[split==1]

    print('train / val')
    print(len(train_image_names), len(train_label_names), len(val_image_names), len(val_label_names))

##############################################
# training params
trained_model = None
trained_model = 'models/airbus_france_india/model.params'

model_type = 'fractal-resunet'

epochs = 100
lr = 0.001 #changed this
lr_decay = None
n_filters = 32
depth = 6
n_classes = 1
batch_size = 12

month = 'GE'
bands='rgb'
train_val='70-30'
lossf='ftnmt_masked'
folder_suffix = 'moz_tst_05'#'_train-val'

ctx_name = 'gpu'
gpu_id = 0

model = run(train_image_names, val_image_names,
    train_label_names, val_label_names,
    trained_model=trained_model,
    epochs=epochs, lr=lr, lr_decay=lr_decay,
    model_type=model_type, n_filters=n_filters, depth=depth, n_classes=n_classes,
    batch_size=batch_size, month=month, bands=bands,
    train_val=train_val, normalize='none', maskedloss=True, otf_aug=True, lossf=lossf,
    ctx_name=ctx_name,
    gpu_id=gpu_id,
    folder_suffix=folder_suffix)