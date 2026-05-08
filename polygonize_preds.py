import os
import cv2
import glob
import warnings
import numpy as np
import time
from osgeo import gdal
from osgeo import ogr, osr

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

path = '/data/Aldhani/cv_fields/preds/descartes_tiles/inference/mozambique/2023/fractal-resunet'
prd_fls = glob.glob(path+'/*inst_ext020_bnd020.tif')

print(len(prd_fls))


for prd_fl in prd_fls:
  dst_layername = prd_fl[:-4]+'.gpkg'
  #dst_layername.replace('-', 'n')
  if not os.path.isfile(dst_layername):
    
    print(prd_fl)
    src_ds = gdal.Open(prd_fl)
    #sr = osr.SpatialReference()
    #sr.ImportFromWkt(dataset.GetProjection())
    srcBand = src_ds.GetRasterBand(1)

    drv = ogr.GetDriverByName("GPKG")
    dst_ds = drv.CreateDataSource(dst_layername)
    dst_layer = dst_ds.CreateLayer(os.path.basename(dst_layername), srs = get_srs(src_ds))

    gdal.Polygonize(srcBand, srcBand, dst_layer, -1, [], callback=None)

    dst_layername = None
    src_ds = None
    srcBand = None
    drv = None
    dst_ds = None
    dst_layer = None