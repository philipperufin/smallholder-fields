import os
import numpy as np
import pandas as pd

import cv2
import higra as hg
from tqdm import tqdm

from osgeo import gdal
from osgeo import ogr, osr 

from sklearn.metrics import matthews_corrcoef
import matplotlib.pyplot as plt

import mxnet as mx
from mxnet import nd, gpu, gluon, autograd, npx, image
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.loss import Loss

from decode.FracTAL_ResUNet.nn.loss.mtsk_loss import *
from decode.FracTAL_ResUNet.models.heads.head_cmtsk import *
from decode.FracTAL_ResUNet.models.semanticsegmentation.FracTAL_ResUNet_features import *
from decode.FracTAL_ResUNet.models.semanticsegmentation.FracTAL_ResUNet import FracTAL_ResUNet_cmtsk

######################
# i/o

# read rgb tif 2 nd.array on gpu
def open_tif_rgb(pth):
  img = nd.array(gdal.Open(pth).ReadAsArray(), dtype='float32', ctx=gpu(0))[0:3,:,:]
  img = np.flip(img, axis=0) #bgr to rgb
  return img

# read rgb tif 2 numpy array
def open_tif_rgb_cpu(pth):
  img = gdal.Open(pth).ReadAsArray()[0:3,:,:]
  img = np.flip(img, axis=0) #bgr to rgb
  return img

# read rgb tif 2 nd.array on gpu
def open_tif_rgb_tc(pth):
  img = nd.array(gdal.Open(pth).ReadAsArray(), dtype='float32', ctx=gpu(0))[0:3,:,:]
  #img = np.flip(img, axis=0) #bgr to rgb
  return img

# read rgb tif 2 numpy array
def open_tif_rgb_tc_cpu(pth):
  img = gdal.Open(pth).ReadAsArray()[0:3,:,:]
  #img = np.flip(img, axis=0) #bgr to rgb
  return img

# read 3-band stack with masks as numpy array
def open_tif_mask_mtsk(pth):
  img = gdal.Open(pth).ReadAsArray()#[0:3,384:640,384:640]
  img[img==255] = 1
  img = np.nan_to_num(img, copy=False)
  return img

def get_srs(dataset):

    sr = osr.SpatialReference()
    sr.ImportFromWkt(dataset.GetProjection())
    # auto-detect epsg
    auto_detect = sr.AutoIdentifyEPSG()
    if auto_detect != 0:
        sr = sr.FindMatches()[0][0]  # Find matches returns list of tuple of SpatialReferences
        sr.AutoIdentifyEPSG()
    # assign input SpatialReference
    sr.ImportFromEPSG(int(sr.GetAuthorityCode(None)))
    return sr

######################
# scaling
def scale_255(array):
  return (array / 10000) * 255.

def scale_255_value(array, scalevalue):
  return (array / scalevalue) * 255.

def scale_value(array, scalevalue):
  return (array / scalevalue)

def linear_norm(array, in_min, in_max, out_min, out_max):

  in_min = np.expand_dims(in_min, axis = (1,2))
  in_max = np.expand_dims(in_max, axis = (1,2))
  out_min = np.expand_dims(out_min, axis = (1,2))
  out_max = np.expand_dims(out_max, axis = (1,2))

  return (array - in_min) * ((out_max - out_min) / (in_max - in_min)) + out_min

def normalize_bandwise(
    array: np.ndarray
):
  array_min, array_max = np.nanmin(array, axis = (0,1)), np.nanmax(array, axis = (0,1))
  array_min = np.expand_dims(array_min, axis = (0,1))
  array_max = np.expand_dims(array_max, axis = (0,1))

  return np.array((array - array_min) / (array_max - array_min), dtype=np.float32)

######################
# storing data

# create dataset in memory using geotransform specified in ref_pth
def create_mem_ds(ref_pth, n_bands, dtype=gdal.GDT_Float32):
  #print('creating empty raster \n copying geotransform of' + ref_pth)
  drvMemR = gdal.GetDriverByName('MEM')
  ds = gdal.Open(ref_pth)
  mem_ds = drvMemR.Create('', ds.RasterXSize, ds.RasterYSize, n_bands, dtype)
  mem_ds.SetGeoTransform(ds.GetGeoTransform())
  mem_ds.SetProjection(ds.GetProjection())
  return mem_ds

# create copy
def copy_mem_ds(pth, mem_ds):
  copy_ds = gdal.GetDriverByName("GTiff").CreateCopy(pth, mem_ds, 0, options=['COMPRESS=LZW'])
  copy_ds = None



######################
# model helpers
def dice_coef(x, y):
    if type(x).__module__ == 'numpy':
        intersection = np.logical_and(x, y)
        return 2. * np.sum(intersection) / (np.sum(x) + np.sum(y))
    else:
        intersection = mx.ndarray.op.broadcast_logical_and(x, y)
        return 2. * mx.nd.sum(intersection) / (mx.nd.sum(x) + mx.nd.sum(y))

# masked Fractal Tanimoto (with dual) loss
class ftnmt_loss_masked(Loss):
    """
    This function calculates the average fractal tanimoto similarity for d = 0...depth
    Masks out unlabeled areas
    """
    def __init__(self, depth=5, axis= [1,2,3], smooth = 1.0e-5, batch_axis=0, weight=None, **kwargs):
        super().__init__(batch_axis, weight, **kwargs)

        assert depth >= 0, ValueError("depth must be >= 0, aborting...")

        self.smooth = smooth
        self.axis=axis
        self.depth = depth

        if depth == 0:
            self.depth = 1
            self.scale = 1.
        else:
            self.depth = depth
            self.scale = 1./depth

    def inner_prod(self, F, prob, label):
        prod = F.broadcast_mul(prob,label)
        prod = F.sum(prod,axis=self.axis)

        return prod

    def tnmt_base(self, F, preds, labels):

        tpl  = self.inner_prod(F,preds,labels)
        tpp  = self.inner_prod(F,preds,preds)
        tll  = self.inner_prod(F,labels,labels)

        num = tpl + self.smooth
        scale = 1./self.depth
        denum = 0.0
        for d in range(self.depth):
            a = 2.**d
            b = -(2.*a-1.)

            denum = denum + F.reciprocal(F.broadcast_add(a*(tpp+tll), b *tpl) + self.smooth)

        result =  F.broadcast_mul(num,denum)*scale
        return  F.mean(result, axis=0,exclude=True)


    def hybrid_forward(self, F, preds, labels, mask):

        # zero out predictions where label is zero
        preds = F.broadcast_mul(preds, mask)

        preds_dual = 1. - preds
        labels_dual = 1. - labels
        preds_dual = F.broadcast_mul(preds_dual, mask)
        labels_dual = F.broadcast_mul(labels_dual, mask)

        l1 = self.tnmt_base(F, preds, labels)
        l2 = self.tnmt_base(F, preds_dual, labels_dual)

        result = 0.5 * (l1 + l2)

        return  1. - result


# plot with learning curves across train / val and metrics
def learning_curves(model, epochs=None):

  metrics = pd.read_csv(f'{model}/metrics.csv')

  #if epochs != None:
  #  fill = epochs - len(metrics)
  #  empty_df = pd.DataFrame(columns=metrics.columns)
  #  for _ in range(fill):
  #      metrics = metrics.concat(empty_df)
  #  metrics = metrics.reset_index(drop=True)
  #  print(metrics.shape)
  train_loss = metrics['train_loss']
  val_loss = metrics['val_loss']
  train_f1 = metrics['train_f1']
  val_f1 = metrics['val_f1']
  train_dice = metrics['train_dice']
  val_dice = metrics['val_dice']
  train_mcc = metrics['train_mcc']
  val_mcc = metrics['val_mcc']

  #if iter == None:
  epochs = range(0, len(train_loss))

  fig, ax = plt.subplots(1, 4, figsize=(20,4))

  ax[0].plot(epochs, train_loss, label='train')
  ax[0].plot(epochs, val_loss, label='val')
  ax[0].axvline(x = np.where(val_mcc==np.max(val_mcc))[0], color='black', label = 'max val mcc')
  ax[0].set_xlabel('epochs'), ax[0].set_ylabel('loss')
  ax[0].legend()

  ax[1].plot(epochs, train_f1, label='train')
  ax[1].plot(epochs, val_f1, label='val')
  ax[1].axvline(x = np.where(val_mcc==np.max(val_mcc))[0], color='black', label = 'max val mcc')
  ax[1].set_xlabel('epochs'), ax[1].set_ylabel('f1')
  ax[1].legend()

  ax[2].plot(epochs, train_dice, label='train')
  ax[2].plot(epochs, val_dice, label='val')
  ax[2].axvline(x = np.where(val_mcc==np.max(val_mcc))[0], color='black', label = 'max val mcc')
  ax[2].set_xlabel('epochs'), ax[2].set_ylabel('dice')
  ax[2].legend()

  ax[3].plot(epochs, train_mcc, label='train')
  ax[3].plot(epochs, val_mcc, label='val')
  ax[3].axvline(x = np.where(val_mcc==np.max(val_mcc))[0], color='black', label = 'max val mcc')
  ax[3].set_xlabel('epochs'), ax[3].set_ylabel('mcc')
  ax[3].legend()

  plt.show()

# NICFI 4-band data in order BGRN
# input images are scaled between 0 and 1
# augmentation hardcoded in external routine
# can handle float or 8bit scaling, depending on norm arg
class NICFI_rgb_mtsk_norm(gluon.data.Dataset):

    def __init__(self, norm='none', augment=True, image_names=None, label_names=None, in_min=None, in_max=None, out_min=None, out_max=None):

        self.image_names = image_names
        self.label_names = label_names
        self.augment = augment
        self.norm = norm


        if self.norm=='linear-norm-stats':
          self.in_min = in_min
          self.in_max = in_max
          self.out_min = out_min
          self.out_max = out_max

    def __getitem__(self, item):
        image_path = self.image_names[item]

        if self.norm=='none':
          image = np.moveaxis(open_tif_rgb_cpu(image_path), 0, 2)

        if self.norm=='float-to-8bit':
          image = np.moveaxis(open_tif_rgb_cpu(image_path), 0, 2) * 255
          image[image<0] == 0
          image[image>255] == 255

        if self.norm=='bandwise-min-max':
          image = np.moveaxis(normalize_bandwise(open_tif_rgb_cpu(image_path)), 0, 2)

        if self.norm=='linear-norm-stats':
          image = np.moveaxis(linear_norm(open_tif_rgb_cpu(image_path), self.in_min, self.in_max, self.out_min, self.out_max), 0, 2)

        extent_path = self.label_names[item]
        mask = np.moveaxis(open_tif_mask_mtsk(extent_path), 0, 2)
        mask[np.isnan(mask)] = 0

        nrow, ncol, nlabels = mask.shape
        nrow, ncol, nchannels = image.shape

        if self.augment:
          # brightness augmentation
          if self.norm=='float-to-8bit':
            image = np.random.uniform(low=0.8, high=1.25) * image
            image[image<0] == 0
            image[image>255] == 255
          if self.norm!='float-to-8bit':
            image = np.random.uniform(low=0.8, high=1.25) * image
            image[image<0.] == 0.
            image[image>1.] == 1.

          # rotation augmentation
          k = np.random.randint(low=0, high=4)
          image = np.rot90(image, k, axes=(0,1))
          mask = np.rot90(mask, k, axes=(0,1))

          # flip augmentation
          if np.random.uniform() > 0.5:
              image = np.flip(image, axis=0)
              mask = np.flip(mask, axis=0)

          if np.random.uniform() > 0.5:
              image = np.flip(image, axis=1)
              mask = np.flip(mask, axis=1)

        # datatype conversion depending on norm
        if self.norm=='float-to-8bit':
          image = np.array(image, dtype=np.uint8)

        if self.norm!='float-to-8bit':
          #image = image.astype(np.float32)
          image = np.array(image, dtype=np.float32)

        image = mx.nd.array(np.moveaxis(image, -1, 0))

        extent_mask = mx.nd.array(np.expand_dims(mask[:,:,0], 0), dtype='float32')
        boundary_mask = mx.nd.array(np.expand_dims(mask[:,:,1], 0), dtype='float32')
        distance_mask = mx.nd.array(np.expand_dims(mask[:,:,2], 0), dtype='float32')
        #mask_mask = mx.nd.array(np.sum(mask, axis=2)>0, dtype='float32')
        # for weak supervision without noncrop training, mask encoded in sum of band 1-3
        if (mask.shape[2] == 3):
          mask_mask = mx.nd.array(np.expand_dims(np.sum(mask, axis=2)>0, 0), dtype='float32')
        # for weak supervision with noncrop training, mask encoded in extra band 4
        if (mask.shape[2] == 4):
          mask_mask = mx.nd.array(np.expand_dims(mask[:,:,3]==1, 0), dtype='float32')
        return image, extent_mask, boundary_mask, distance_mask, mask_mask

    def __len__(self):
        return len(self.image_names)


def train_model(train_dataloader, model, tanimoto_dual, trainer, epoch, args, maskedloss=True):

    # initialize metrics
    cumulative_loss = 0
    accuracy = mx.metric.Accuracy()
    f1 = mx.metric.F1()
    mcc = mx.metric.MCC()
    dice = mx.metric.CustomMetric(feval=dice_coef, name="Dice")

    if args['ctx_name'] == 'cpu':
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(args['gpu'])

    # training set
    for batch_i, (img, extent, boundary, distance, mask) in enumerate(
        tqdm(train_dataloader, desc='Training epoch {}'.format(epoch))):

        with autograd.record():

            img = img.as_in_context(ctx)
            extent = extent.as_in_context(ctx)
            boundary = boundary.as_in_context(ctx)
            distance = distance.as_in_context(ctx)

            mask = mask.as_in_context(ctx)
            nonmask = mx.nd.ones(extent.shape).as_in_context(ctx)

            logits, bound, dist = model(img)

            # multi-task loss
            if maskedloss==True:
              # TODO: wrap this in a custom loss function / class
              loss_extent = mx.nd.sum(tanimoto_dual(logits, extent, mask))
              loss_boundary = mx.nd.sum(tanimoto_dual(bound, boundary, mask))
              loss_distance = mx.nd.sum(tanimoto_dual(dist, distance, mask))
            if maskedloss==False:
              # unmasked
              loss_extent = mx.nd.sum(tanimoto_dual(logits, extent))
              loss_boundary = mx.nd.sum(tanimoto_dual(bound, boundary))
              loss_distance = mx.nd.sum(tanimoto_dual(dist, distance))


            loss = 0.33 * (loss_extent + loss_boundary + loss_distance) # + loss_hsv)

        loss.backward()
        trainer.step(args['batch_size'])
        cumulative_loss += mx.nd.sum(loss).asscalar()

        logits_reshaped = logits.reshape((logits.shape[0], -1))
        extent_reshaped = extent.reshape((extent.shape[0], -1))
        mask_reshaped = mask.reshape((mask.shape[0], -1))

        if maskedloss == True:
          nonmask_idx = mx.np.nonzero(mask_reshaped.as_np_ndarray())
          nonmask_idx = mx.np.stack(nonmask_idx).as_nd_ndarray().as_in_context(ctx)
          
          logits_masked = mx.nd.gather_nd(logits_reshaped, nonmask_idx)
          extent_masked = mx.nd.gather_nd(extent_reshaped, nonmask_idx)

        if maskedloss == False:
          # unmasked
          logits_masked = logits_reshaped
          extent_masked = extent_reshaped


        # accuracy
        extent_predicted_classes = mx.nd.ceil(logits_masked - 0.5)
        accuracy.update(extent_masked, extent_predicted_classes)

        # f1 score
        probabilities = mx.nd.stack(1 - logits_masked, logits_masked, axis=1)
        f1.update(extent_masked, probabilities)

        # MCC metric
        mcc.update(extent_masked, probabilities)

        # Dice score
        dice.update(extent_masked, extent_predicted_classes)

    return cumulative_loss, accuracy, f1, mcc, dice


def evaluate_model(val_dataloader, model, tanimoto_dual, epoch, args, maskedloss=True):

    # initialize metrics
    cumulative_loss = 0
    accuracy = mx.metric.Accuracy()
    f1 = mx.metric.F1()
    mcc = mx.metric.MCC()
    dice = mx.metric.CustomMetric(feval=dice_coef, name="Dice")
    if args['ctx_name'] == 'cpu':
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(args['gpu'])

    # validation set
    for batch_i, (img, extent, boundary, distance, mask) in enumerate(
        tqdm(val_dataloader, desc='Validation epoch {}'.format(epoch))):

        img = img.as_in_context(ctx)
        extent = extent.as_in_context(ctx)
        boundary = boundary.as_in_context(ctx)
        distance = distance.as_in_context(ctx)

        mask = mask.as_in_context(ctx)
        nonmask = mx.nd.ones(extent.shape).as_in_context(ctx)

        # logits, bound, dist, convc = model(img)
        logits, bound, dist = model(img)

        # multi-task loss
        if maskedloss == True:
          loss_extent = mx.nd.sum(tanimoto_dual(logits, extent, mask))
          loss_boundary = mx.nd.sum(tanimoto_dual(bound, boundary, mask))
          loss_distance = mx.nd.sum(tanimoto_dual(dist, distance, mask))

        if maskedloss == False:
          # unmasked
          loss_extent = mx.nd.sum(tanimoto_dual(logits, extent))
          loss_boundary = mx.nd.sum(tanimoto_dual(bound, boundary))
          loss_distance = mx.nd.sum(tanimoto_dual(dist, distance))

        loss = 0.33 * (loss_extent + loss_boundary + loss_distance) # + loss_hsv)


        # update metrics based on every batch
        cumulative_loss += mx.nd.sum(loss).asscalar()

        # update metrics based on every batch
        # mask out unlabeled pixels
        logits_reshaped = logits.reshape((logits.shape[0], -1))
        extent_reshaped = extent.reshape((extent.shape[0], -1))
        mask_reshaped = mask.reshape((mask.shape[0], -1))

        if maskedloss == True:
          nonmask_idx = mx.np.nonzero(mask_reshaped.as_np_ndarray())
          nonmask_idx = mx.np.stack(nonmask_idx).as_nd_ndarray().as_in_context(ctx)
          
          logits_masked = mx.nd.gather_nd(logits_reshaped, nonmask_idx)
          extent_masked = mx.nd.gather_nd(extent_reshaped, nonmask_idx)

        if maskedloss == False:
          # unmasked
          logits_masked = logits_reshaped
          extent_masked = extent_reshaped

        # accuracy
        extent_predicted_classes = mx.nd.ceil(logits_masked - 0.5)
        accuracy.update(extent_masked, extent_predicted_classes)

        # f1 score
        probabilities = mx.nd.stack(1 - logits_masked, logits_masked, axis=1)
        f1.update(extent_masked, probabilities)

        # MCC metric
        mcc.update(extent_masked, probabilities)

        # Dice score
        dice.update(extent_masked, extent_predicted_classes)

    return cumulative_loss, accuracy, f1, mcc, dice


def run(train_names, val_names, #test_names,
        train_names_label, val_names_label, #test_names_label,
        trained_model=None,
        month='08-12',
        epochs=100, lr=0.0001, lr_decay=None, n_filters=32, batch_size=4,
        model_type='fractal-resunet', depth=6, n_classes=1,
        codes_to_keep=[1, 2],
        bands='rgb', train_val='70-30',
        normalize='float-to-8bit',
        otf_aug=True,
        maskedloss=True,
        lossf='ftnmt',
        folder_suffix='',
        ctx_name='gpu',
        gpu_id=0,
        visdom=False):
    '''
        explaining parameters:
        country, name used in save_path and visdom env
        train_names, val_names, test_names: file names of images, fed to DataSet function
        train_names_label, val_names_label, test_names_label: file names of labels, fed to DataSet function
        trained_model=None, if not none, loads corresponding weights
        month='Airbus', only namegiving, can be replaced/shortened
        epochs=100, lr=0.0001, lr_decay=None, n_filters=16, batch_size=8, # both namegiving and hyperparams for training loop
        model_type='fractal-resunet', depth=5, n_classes=1, # both namegiving and hyperparams for training loop
        codes_to_keep=[1, 2], # figure that out - i think sherrie used 0 for masking, so extent = 1, edge = 2?
        folder_suffix='', # additional namegiving suffix
        boundary_kernel_size=3, # kernel size for cv2 ops during dataset generation
        ctx_name='cpu', # defines gpu & id
        gpu_id=0):

    '''

    # Set MXNet ctx
    if ctx_name == 'cpu':
        ctx = mx.cpu()
    elif ctx_name == 'gpu':
        ctx = mx.gpu(gpu_id)

    # Set up names of directories and paths for saving
    if trained_model is None:
        model_name = model_type+'_'+folder_suffix
        if lr_decay:
            model_name = model_name + '_lrdecay-'+str(lr_decay)

        # define model
        model = FracTAL_ResUNet_cmtsk(nfilters_init=n_filters, depth=depth, NClasses=n_classes)
        model.initialize()
        model.hybridize()
        model.collect_params().reset_ctx(ctx)

    else:
        model_name = model_type+'_'+folder_suffix

        model = FracTAL_ResUNet_cmtsk(nfilters_init=n_filters, depth=depth, NClasses=n_classes)
        model.load_parameters(trained_model, ctx=ctx)

        # freeze params
        #for i,p in enumerate(model.collect_params().items()):
        #    if i < 396:
        #        p[1].grad_req = 'null'


    save_model_name = os.path.join('../model/', model_name + '.params')

    # Arguments
    args = {}
    args['batch_size'] = batch_size
    args['ctx_name'] = ctx_name
    args['gpu'] = gpu_id

    # Define train/val/test splits
    p01 = None
    p99 = None
    if not normalize == 'none':
      stats_read = pd.read_csv('image_train/image_train_stats.csv', sep=';')
      p01 = stats_read['p01']
      p99 = stats_read['p99']

    train_dataset = NICFI_rgb_mtsk_norm(
        image_names=train_names,
        label_names=train_names_label,
        norm=normalize,
        augment=otf_aug,
        in_min=p01, in_max=p99, out_min=[0,0,0,0], out_max=[1,1,1,1])

    val_dataset = NICFI_rgb_mtsk_norm(
        image_names=val_names,
        label_names=val_names_label,
        norm=normalize,
        augment=False,
        in_min=p01, in_max=p99, out_min=[0,0,0,0], out_max=[1,1,1,1])

    train_dataloader = gluon.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = gluon.data.DataLoader(val_dataset, batch_size=batch_size)

    # define loss function
    if maskedloss == True:
      print('maskedlossfct')
      tanimoto_dual = ftnmt_loss_masked(depth=0) # Tanimoto_with_dual_masked()

    if maskedloss == False:
      print('unmaskedlossfct')
      tanimoto_dual = ftnmt_loss(depth=0) # Tanimoto_with_dual_masked()

    if lr_decay:
        schedule = mx.lr_scheduler.FactorScheduler(step=5, factor=lr_decay, stop_factor_lr=1e-07)
        adam_optimizer = mx.optimizer.Adam(learning_rate=lr, lr_scheduler=schedule)
    else:
        adam_optimizer = mx.optimizer.Adam(learning_rate=lr)
    trainer = gluon.Trainer(model.collect_params(), optimizer=adam_optimizer)

    # containers for metrics to log
    train_metrics = {'train_loss': [], 'train_acc': [], 'train_f1': [],
                     'train_mcc': [], 'train_dice': []}
    val_metrics = {'val_loss': [], 'val_acc': [], 'val_f1': [],
                   'val_mcc': [], 'val_dice': []}
    best_loss = 100.0

    # training loop
    for epoch in range(1, epochs+1):

        # training set
        train_loss, train_accuracy, train_f1, train_mcc, train_dice = train_model(
            train_dataloader, model, tanimoto_dual, trainer, epoch, args, maskedloss=maskedloss)

        # training set metrics
        train_loss_avg = train_loss / len(train_dataset)
        train_metrics['train_loss'].append(train_loss_avg)
        train_metrics['train_acc'].append(train_accuracy.get()[1])
        train_metrics['train_f1'].append(train_f1.get()[1])
        train_metrics['train_mcc'].append(train_mcc.get()[1])
        train_metrics['train_dice'].append(train_dice.get()[1])

        # validation set
        val_loss, val_accuracy, val_f1, val_mcc, val_dice = evaluate_model(
            val_dataloader, model, tanimoto_dual, epoch, args, maskedloss=maskedloss)

        # validation set metrics
        val_loss_avg = val_loss / len(val_dataset)
        val_metrics['val_loss'].append(val_loss_avg)
        val_metrics['val_acc'].append(val_accuracy.get()[1])
        val_metrics['val_f1'].append(val_f1.get()[1])
        val_metrics['val_mcc'].append(val_mcc.get()[1])
        val_metrics['val_dice'].append(val_dice.get()[1])

        print("Epoch {}:".format(epoch))
        print("    Train loss {:0.3f}, accuracy {:0.3f}, F1-score {:0.3f}, MCC: {:0.3f}, Dice: {:0.3f}".format(
            train_loss_avg, train_accuracy.get()[1], train_f1.get()[1], train_mcc.get()[1], train_dice.get()[1]))
        print("    Val loss {:0.3f}, accuracy {:0.3f}, F1-score {:0.3f}, MCC: {:0.3f}, Dice: {:0.3f}".format(
            val_loss_avg, val_accuracy.get()[1], val_f1.get()[1], val_mcc.get()[1], val_dice.get()[1]))

        # save model based on min validation loss
        if val_loss_avg < best_loss:
            model.save_parameters(save_model_name)
            best_loss = val_loss_avg

        # save model based on best MCC metric
        #if val_mcc.get()[1] > best_mcc:
        #    model.save_parameters(save_model_name)
        #    best_mcc = val_mcc.get()[1]

        # save metrics
        metrics = pd.concat([pd.DataFrame(train_metrics), pd.DataFrame(val_metrics)], axis=1)
        metrics.to_csv(os.path.join('../model/', model_name + '_metrics.csv'), index=False)

    #return model

def InstSegm(extent, boundary, t_ext=0.4, t_bound=0.2):
    """
    INPUTS:
    extent : extent prediction
    boundary : boundary prediction
    t_ext : threshold for extent
    t_bound : threshold for boundary
    OUTPUT:
    instances
    """

    # Threshold extent mask
    ext_binary = np.uint8(extent >= t_ext)

    # Artificially create strong boundaries for
    # pixels with non-field labels
    input_hws = np.copy(boundary)
    input_hws[ext_binary == 0] = 1

    # Create the directed graph
    size = input_hws.shape[:2]
    graph = hg.get_8_adjacency_graph(size)
    edge_weights = hg.weight_graph(
        graph,
        input_hws,
        hg.WeightFunction.mean
    )

    tree, altitudes = hg.watershed_hierarchy_by_dynamics(
        graph,
        edge_weights
    )

    # Get individual fields
    # by cutting the graph using altitude
    instances = hg.labelisation_horizontal_cut_from_threshold(
        tree,
        altitudes,
        threshold=t_bound)

    instances[ext_binary == 0] = -1

    return instances

def InstScores(instances, extent):
    iids = np.unique(instances)
    iids = iids[(iids>0)]

    ###################################################################
    # instance level scores
    inst_scores = np.float32(np.copy(instances))
    for iid in iids:
        score = np.nanmedian(extent[(instances==iid)])
        inst_scores[(instances==iid)] = score
    return inst_scores
    
    ######################
# multi-taks labels from boundaries
def get_boundary(label, kernel_size = (2,2)):
    tlabel = label.astype(np.uint8)
    temp = cv2.Canny(tlabel,0,1)
    tlabel = cv2.dilate(
        temp,
        cv2.getStructuringElement(
            cv2.MORPH_CROSS,
            kernel_size),
        iterations = 1)
    tlabel = tlabel.astype(np.float32)
    tlabel /= 255.
    return tlabel

def get_distance(label):
    tlabel = label.astype(np.uint8)
    dist = cv2.distanceTransform(tlabel,
                                 cv2.DIST_L2,
                                 0)

    # get unique objects
    output = cv2.connectedComponentsWithStats(label, 4, cv2.CV_32S)
    num_objects = output[0]
    labels = output[1]

    # min/max normalize dist for each object
    for l in range(num_objects):
        dist[labels==l] = (dist[labels==l]) / (dist[labels==l].max())

    return dist

def get_crop(image, kernel_size = (2,2)):

    im_floodfill = image.copy()
    h, w = image.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    # floodfill
    cv2.floodFill(im_floodfill, mask, (0,0), 1);

    # invert
    im_floodfill = cv2.bitwise_not(im_floodfill)

    # kernel size
    kernel = np.ones(kernel_size, np.uint8)

    # erode & dilate
    img_erosion = cv2.erode(im_floodfill, kernel, iterations=1)
    return cv2.dilate(img_erosion, kernel, iterations=1) - 254
