
'''
This script generates DSMs from Lidar data inconjunction to a building footprint vector layer.
It also generates a LAI dataset (hopefully)
This script is based on Swedish national lidar point cloud
'''

# Initialisation
from osgeo import gdal
from qgis.core import QgsApplication, QgsVectorLayer, QgsCoordinateReferenceSystem
import numpy as np
from osgeo.gdalconst import GDT_Float32, GA_ReadOnly
import sys, os
import shutil
import matplotlib.pylab as plt
# from pathlib import Path

# Initiating a QGIS application and connect to processing
qgishome = 'C:/OSGeo4W/apps/qgis/'
QgsApplication.setPrefixPath(qgishome, True)
app = QgsApplication([], False)
app.initQgis()

sys.path.append(r'C:\OSGeo4W\apps\qgis\python\plugins')
sys.path.append(r'C:\Users\xlinfr\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins')
sys.path.append(r'C:\FUSION')
import os, subprocess

def saveraster(gdal_data, filename, raster):
    rows = gdal_data.RasterYSize
    cols = gdal_data.RasterXSize

    outDs = gdal.GetDriverByName("GTiff").Create(filename, cols, rows, int(1), GDT_Float32)
    outBand = outDs.GetRasterBand(1)

    # write the data
    outBand.WriteArray(raster, 0, 0)
    # flush data to disk, set the NoData value and calculate stats
    outBand.FlushCache()
    outBand.SetNoDataValue(-9999)

    # georeference the image and set the projection
    outDs.SetGeoTransform(gdal_data.GetGeoTransform())
    outDs.SetProjection(gdal_data.GetProjection())

import processing
from processing_umep.processing_umep_provider import ProcessingUMEPProvider
umep_provider = ProcessingUMEPProvider()
QgsApplication.processingRegistry().addProvider(umep_provider)

from processing_fusion.processing_fusion_provider import ProcessingFUSIONProvider
fusion_provider = ProcessingFUSIONProvider()
QgsApplication.processingRegistry().addProvider(fusion_provider)

from processing.core.Processing import Processing
Processing.initialize()

# from processing_fusion import fusionUtils
# from processing.core.ProcessingConfig import ProcessingConfig, Setting

# Input data paths and settings
workingpath = 'C:/Users/xlinfr/Documents/Undervisning/GIS/LidarQGISFUSION/tempfromscript/'
domain = r'C:\Users\xlinfr\Documents\Undervisning\GIS\LidarQGISFUSION\FastighetskartanVektor_1501_3006\rutnat_get.shp'
# buildingFootprint = r'C:\Users\xlinfr\Documents\Undervisning\GIS\LidarQGISFUSION\FastighetskartanVektor_1501_3006\by_get.shp'
buildingFootprint = 'C:/temp/by_get.shp'
lidardata = r'C:\Users\xlinfr\Documents\Undervisning\GIS\LidarQGISFUSION\Laserdata_1501_3006\09B002_63975_3175_25\09B002_63975_3175_25.las'

intensitylimit = 175 # Threshold to identify grass surfaces from Lidar
cellsize = 2         # Cellsize for output raster
buildingbuffer = cellsize * 1.5 # Buffersize from building footprints
zfilter = 2.5        # Height above ground to filter out low vegetation

outputs = {}

if os.path.exists(workingpath):
     shutil.rmtree(workingpath)
os.mkdir(workingpath)

vlayer = QgsVectorLayer(domain, "polygon", "ogr")
extent = vlayer.extent()
projwin = str(extent.xMinimum()) + ',' + str(extent.xMaximum()) + ',' + str(extent.yMinimum()) +',' + str(extent.yMaximum()) + ' [EPSG:3006]'

print('Clipping LAS-file')
alg_params = {
    'INPUT':lidardata,
    'EXTENT':projwin,
    'SHAPE':0,
    'VERSION64':True,
    'OUTPUT': workingpath + 'clippedLidar.las',
    'DTM':'',
    'HEIGHT':False,
    'IGNOREOVERLAP':False,
    'CLASS':'',
    'ADVANCED_MODIFIERS':''
}
outputs['ClipLidardata'] = processing.run("fusion:clipdata", alg_params)

print('Gereate ground elevation model')
alg_params = {
    'INPUT': outputs['ClipLidardata']['OUTPUT'], 
    'CELLSIZE':cellsize,
    'XYUNITS':0,
    'ZUNITS':0,
    'VERSION64':True,
    'OUTPUT_DTM':workingpath + 'ground.dtm',
    'SPIKE':'',
    'MEDIAN':'',
    'SMOOTH':'',
    'MINIMUM':'',
    'CLASS':'2',
    'SLOPE':'',
    'ADVANCED_MODIFIERS':''}
    
processing.run("fusion:gridsurfacecreate", alg_params)

#TODO polyclipdata dose not work as processing alg. Running with os.system() here instead. Seems to be issues with /\ and long search paths
print('Clipping out vegetation points')
alg_params = 'C:/FUSION/PolyClipData64.exe /outside /class:1 "'  + buildingFootprint + '" "' + workingpath + 'veg.las" "' + outputs['ClipLidardata']['OUTPUT'] + '"'
os.system(alg_params)

print('Clipping out building points')
alg_params = 'C:/FUSION/PolyClipData64.exe /class:1 "'  + buildingFootprint + '" "' + workingpath + 'building.las" "' + outputs['ClipLidardata']['OUTPUT'] + '"'
os.system(alg_params)

print('Clipping out ground points')
alg_params = 'C:/FUSION/PolyClipData64.exe /outside /class:2 "'  + buildingFootprint + '" "' + workingpath + 'ground.las" "' + outputs['ClipLidardata']['OUTPUT'] + '"'
os.system(alg_params)

print('Make DSM from ground and building points')
alg_params = {
     'INPUT': workingpath + 'building.las;' + workingpath + 'ground.las',
     'CELLSIZE':cellsize,
     'XYUNITS':0,
     'ZUNITS':0,
     'VERSION64':True,
     'OUTPUT':workingpath + 'dsm.dtm',
     'GROUND':'','MEDIAN':'','SMOOTH':'','CLASS':'','SLOPE':False,'ASCII':False,
     'ADVANCED_MODIFIERS':'/ascii'
}
outputs['DSMCanopyModel'] = processing.run("fusion:canopymodel",alg_params)

alg_params = {
    'INPUT':outputs['DSMCanopyModel']['OUTPUT'],
    'CSV':False,
    'RASTER':True,
    'MULTIPLIER':None,
    'OUTPUT': workingpath + 'dsm.asc'
}
outputs['DSMDTMtoASCII'] = processing.run("fusion:dtm2ascii", alg_params)


print('Removing vegpoints lower than 2.5 meter above ground')
alg_params = {
    'INPUT':workingpath + 'veg.las',
    'EXTENT':projwin,
    'SHAPE':0,
    'VERSION64':True,
    'OUTPUT':workingpath + 'veg_filt.las',
    'GROUND':workingpath + 'ground.dtm',
    'HEIGHT':False,
    'IGNOREOVERLAP':False,
    'CLASS':'',
    'ADVANCED_MODIFIERS':
    '/zmin:' + str(zfilter) 
}
outputs['FilteredVeg2point5meters'] = processing.run("fusion:clipdata", alg_params)


print('Make CDSM of filtered vegetation points')
alg_params = {
     'INPUT':workingpath + 'veg_filt.las;' + workingpath + 'ground.las',
     'CELLSIZE':cellsize,
     'XYUNITS':0,
     'ZUNITS':0,
     'VERSION64':True,
     'OUTPUT':workingpath + 'cdsm.dtm',
     'GROUND':workingpath + 'ground.dtm', # using DTM switch is not working 
     'MEDIAN':'','SMOOTH':'','CLASS':'','SLOPE':False,'ASCII':False,
     'ADVANCED_MODIFIERS':'/ascii'
}
outputs['CDSMCanopyModel'] = processing.run("fusion:canopymodel",alg_params)

alg_params = {
    'INPUT':outputs['CDSMCanopyModel']['OUTPUT'],
    'CSV':False,
    'RASTER':True,
    'MULTIPLIER':None,
    'OUTPUT': workingpath + 'cdsmraw.asc'
}
outputs['CDSMDTMtoASCII'] = processing.run("fusion:dtm2ascii", alg_params)

print('Buffering building footprints')
alg_params = {
    'INPUT': buildingFootprint,
    'DISTANCE': buildingbuffer,
    'SEGMENTS':1,
    'END_CAP_STYLE':1,
    'JOIN_STYLE':1,
    'MITER_LIMIT':2,
    'DISSOLVE':True,
    'OUTPUT': workingpath + 'by_buff.shp'}

outputs['BufferedBuildingsTif'] = processing.run("native:buffer", alg_params)

#fix new projwin to allign rasters
data = gdal.Open(outputs['DSMDTMtoASCII']['OUTPUT'], GA_ReadOnly)
geoTransform = data.GetGeoTransform()
minx = geoTransform[0]
maxy = geoTransform[3]
maxx = minx + geoTransform[1] * data.RasterXSize
miny = maxy + geoTransform[5] * data.RasterYSize
data = None
projwinrasterize = str(minx) + ',' + str(maxx) + ',' + str(miny) +',' + str(maxy) + ' [EPSG:3006]'

alg_params = {
    'INPUT':outputs['BufferedBuildingsTif']['OUTPUT'],
    'FIELD':'',
    'BURN':0,
    'USE_Z':False,
    'UNITS':1,
    'WIDTH':cellsize,
    'HEIGHT':cellsize,
    'EXTENT':projwinrasterize,
    'NODATA':None,
    'OPTIONS':'',
    'DATA_TYPE':5,
    'INIT':1,
    'INVERT':False,
    'EXTRA':'',
    'OUTPUT': workingpath + 'buff_bolean.tif'
}
outputs['BufferedBuildings'] = processing.run("gdal:rasterize", alg_params)

alg_params = {
    'INPUT_A':outputs['CDSMDTMtoASCII']['OUTPUT'],
    'BAND_A':1,
    'INPUT_B':outputs['BufferedBuildings']['OUTPUT'],
    'BAND_B':1,'INPUT_C':None,'BAND_C':None,'INPUT_D':None,'BAND_D':None,'INPUT_E':None,'BAND_E':None,'INPUT_F':None,'BAND_F':None,
    'FORMULA':'A * B','NO_DATA':None,'PROJWIN':None,'RTYPE':5,'OPTIONS':'','EXTRA':'',
    'OUTPUT':workingpath + 'cdsm_filt.tif'}

outputs['RemoveBufferedBuildings'] = processing.run("gdal:rastercalculator", alg_params )

alg_params = {
    'INPUT_A':outputs['RemoveBufferedBuildings']['OUTPUT'],
    'BAND_A':1,
    'INPUT_B':None,
    'BAND_B':None,
    'INPUT_C':None,'BAND_C':None,'INPUT_D':None,'BAND_D':None,'INPUT_E':None,'BAND_E':None,'INPUT_F':None,'BAND_F':None,
    'FORMULA':'(A > 0.5) * A',
    'NO_DATA':None,'PROJWIN':None,'RTYPE':5,'OPTIONS':'','EXTRA':'',
    'OUTPUT':workingpath + 'cdsm_filt2.tif'
}
outputs['RemovedSmallNumbersCDSM'] = processing.run("gdal:rastercalculator", alg_params)

print('vegetation filtering/refinements')
# filters to be included (remove holes and fill holes (and maybe linear also))
data = gdal.Open(outputs['RemovedSmallNumbersCDSM']['OUTPUT'], GA_ReadOnly)
x = data.ReadAsArray().astype(float)
nd = data.GetRasterBand(1).GetNoDataValue() # one line of nodata to the left is removed
x[x==nd] = 0
col = x.shape[1]
row = x.shape[0]

# fill holes in vegetation
y=np.copy(x)
z=np.copy(x)
z[z>0]=1
for i in np.arange(1, row-1):
    for j in np.arange(1, col-1):
        dom = z[i-1:i+2, j-1:j+2]
        if (z[i, j] == 0) and (np.sum(dom) >= 6):
            y[i, j] = np.median(x[i-1:i+2, j-1:j+2]) 
        
# Remove small vegetation units 1
sur = 2 # number of surrounding vegetation pixels
for i in np.arange(1, row-1):
    for j in np.arange(1, col-1):
        dom = z[i-1:i+2, j-1:j+2]
        if (z[i, j] == 1) and (np.sum(dom) <= sur):
            y[i, j] = 0 

# Remove small vegetation units 2
z=np.copy(y)
y2=np.copy(y)
z[z>0]=1
for i in np.arange(1, row-1):
    for j in np.arange(1, col-1):
        dom = z[i-1:i+2, j-1:j+2]
        if (z[i, j] == 1) and (np.sum(dom) <= 1):
            y2[i, j] = 0 

saveraster(data,workingpath + 'cdsm.tif', y2)

# remove notataline frpm dsm
data2 = gdal.Open(outputs['RemovedSmallNumbersCDSM']['OUTPUT'], GA_ReadOnly)
dsm = data2.ReadAsArray().astype(float)
nd = data2.GetRasterBand(1).GetNoDataValue() # one line of nodata to the left is removed
dsm[dsm==nd] = 0
saveraster(data, workingpath + 'dsm.tif', dsm)

# land cover
print('Making landcover from lidar')
alg_params = {
    'INPUT':buildingFootprint,
    'FIELD':'',
    'BURN':0,
    'USE_Z':False,
    'UNITS':1,
    'WIDTH':cellsize,
    'HEIGHT':cellsize,
    'EXTENT':projwinrasterize,
    'NODATA':None,
    'OPTIONS':'',
    'DATA_TYPE':5,
    'INIT':1,
    'INVERT':False,
    'EXTRA':'',
    'OUTPUT': workingpath + 'build_bolean.tif'
}
outputs['BuildingsBoolean'] = processing.run("gdal:rasterize", alg_params)

alg_params = {
    'INPUT_A':workingpath + 'cdsm.tif',
    'BAND_A':1,
    'INPUT_B':None,
    'BAND_B':None,
    'INPUT_C':None,'BAND_C':None,'INPUT_D':None,'BAND_D':None,'INPUT_E':None,'BAND_E':None,'INPUT_F':None,'BAND_F':None,
    'FORMULA':'(A > 0)',
    'NO_DATA':None,'PROJWIN':None,'RTYPE':5,'OPTIONS':'','EXTRA':'',
    'OUTPUT':workingpath + 'veg_bolean.tif'
}
outputs['vegBoolean'] = processing.run("gdal:rastercalculator", alg_params)

alg_params = {
    'INPUT':workingpath + 'ground.las',
    'ALLRET':False,
    'LOWEST':False,
    'HIST':False,
    'PIXEL':cellsize,
    'SWITCH':1, # bug in FUSION. Bitmap is jpg
    'OUTPUT': workingpath + 'intensity',
    'ADVANCED_MODIFIERS':''
}
outputs['IntensityRaster'] = processing.run("fusion:intensityimage", alg_params)

alg_params = {
    'INPUT_A':workingpath + 'intensity.jpg',
    'BAND_A':1,
    'INPUT_B':None,
    'BAND_B':None,
    'INPUT_C':None,'BAND_C':None,'INPUT_D':None,'BAND_D':None,'INPUT_E':None,'BAND_E':None,'INPUT_F':None,'BAND_F':None,
    'FORMULA':'A * 1',
    'NO_DATA':None,'PROJWIN':None,'RTYPE':5,'OPTIONS':'','EXTRA':'',
    'OUTPUT':workingpath + 'intensity1.tif'
}
outputs['Intensity1Raster'] = processing.run("gdal:rastercalculator", alg_params)
alg_params = {
    'INPUT_A':outputs['Intensity1Raster']['OUTPUT'],
    'BAND_A':1,
    'INPUT_B':None,
    'BAND_B':None,
    'INPUT_C':None,'BAND_C':None,'INPUT_D':None,'BAND_D':None,'INPUT_E':None,'BAND_E':None,'INPUT_F':None,'BAND_F':None,
    'FORMULA':'(A < 255) * A',   #((“Intensity1@1” < 255) * “Intensity1@1”)
    'NO_DATA':None,'PROJWIN':None,'RTYPE':5,'OPTIONS':'','EXTRA':'',
    'OUTPUT':workingpath + 'intensity1nodata.tif'
}
outputs['Intensity1RasterNodata'] = processing.run("gdal:rastercalculator", alg_params)

alg_params = {
    'INPUT_A':outputs['Intensity1RasterNodata']['OUTPUT'],
    'BAND_A':1,
    'INPUT_B':None,
    'BAND_B':None,
    'INPUT_C':None,'BAND_C':None,'INPUT_D':None,'BAND_D':None,'INPUT_E':None,'BAND_E':None,'INPUT_F':None,'BAND_F':None,
    'FORMULA':'(A >= ' + str(intensitylimit) + ')',  
    'NO_DATA':None,'PROJWIN':None,'RTYPE':5,'OPTIONS':'','EXTRA':'',
    'OUTPUT':workingpath + 'lc_bolean.tif'
}
outputs['Intensity1RasterFinal'] = processing.run("gdal:rastercalculator", alg_params)

alg_params = {
    'INPUT_A':outputs['BuildingsBoolean']['OUTPUT'],
    'BAND_A':1,
    'INPUT_B':outputs['vegBoolean']['OUTPUT'],
    'BAND_B':1,
    'INPUT_C':outputs['Intensity1RasterFinal']['OUTPUT'],
    'BAND_C':1,
    'INPUT_D':None,'BAND_D':None,'INPUT_E':None,'BAND_E':None,'INPUT_F':None,'BAND_F':None,
    'FORMULA':'A + (C * 2) + ((A * B) * 3)',   #(“build_bolean@1”) + (“lc_bolean@1” * 2) + ((“build_bolean@1” * “veg_bolean@1”) * 3)
    'NO_DATA':None,'PROJWIN':None,'RTYPE':5,'OPTIONS':'','EXTRA':'',
    'OUTPUT':workingpath + 'lcunclassed.tif'
}
outputs['LandCoverUnclassedRaster'] = processing.run("gdal:rastercalculator", alg_params)


lctable = [
    0,1,1, #paved
    -1,0,2, #build
    3,8,4, # decid
    1,3,5, # grass
]
alg_params = {
    'INPUT_RASTER':outputs['LandCoverUnclassedRaster']['OUTPUT'],
    'RASTER_BAND':1,
    'TABLE':lctable,
    'NO_DATA':-9999,
    'RANGE_BOUNDARIES':0,
    'NODATA_FOR_MISSING':False,
    'DATA_TYPE':3,
    'OUTPUT': workingpath + 'lc.tif'
}
outputs['LandCoverRaster'] = processing.run("native:reclassifybytable", alg_params)

# plt.figure()
# ax1 = plt.subplot(1,2,1)
# ax1.imshow(x)
# ax2 = plt.subplot(1,2,2)
# ax2.imshow(y2)
# plt.show()

print('LAI estimation from Lidar')
print('Clipping out vegetation points with buffered buildings')
alg_params = 'C:/FUSION/PolyClipData64.exe /outside /class:1 "'  + outputs['BufferedBuildingsTif']['OUTPUT'] + '" "' + workingpath + 'vegbuff.las" "' + outputs['ClipLidardata']['OUTPUT'] + '"'
os.system(alg_params)

alg_params = {
    'INPUT':workingpath + 'vegbuff.las;' + workingpath + 'ground.las',
    'OUTPUT':workingpath + 'lai.las',
    'ADVANCED_MODIFIERS':''
}
outputs['LAIcloud'] = processing.run("fusion:mergelasfiles", alg_params)

alg_params = {
    'INPUT':outputs['LAIcloud']['OUTPUT'],
    'CELLSIZE':10,
    'VERSION64':True,
    'OUTPUT':workingpath + 'lai_grounddensity.dtm',
    'FIRST':False,
    'ASCII':True,
    'CLASS':'2'
}
processing.run("fusion:returndensity", alg_params)

alg_params = {
    'INPUT':outputs['LAIcloud']['OUTPUT'],
    'CELLSIZE':10,
    'VERSION64':True,
    'OUTPUT':workingpath + 'lai_density.dtm',
    'FIRST':False,
    'ASCII':True,
    'CLASS':''
}
processing.run("fusion:returndensity", alg_params)

print('Calculating LAI from point density [-1.94 * ln(Rground/Rtotal)]')
alg_params = {
    'INPUT_A':workingpath + 'lai_grounddensity.asc',
    'BAND_A':1,
    'INPUT_B':workingpath + 'lai_density.asc',
    'BAND_B':1,
    'INPUT_C':None,'BAND_C':None,'INPUT_D':None,'BAND_D':None,'INPUT_E':None,'BAND_E':None,'INPUT_F':None,'BAND_F':None,
    'FORMULA':'log ( A / B ) * -1.94',
    'NO_DATA':None,'PROJWIN':None,'RTYPE':5,'OPTIONS':'','EXTRA':'',
    'OUTPUT':workingpath + 'lai_10m_nodata.tif'
}
outputs['LAINodata'] = processing.run("gdal:rastercalculator", alg_params)

alg_params = {
    'INPUT':outputs['LAINodata']['OUTPUT'],
    'BAND':1,
    'FILL_VALUE':0,
    'OUTPUT': workingpath + 'lai_10m.tif'
}
outputs['LAI'] = processing.run("native:fillnodata", alg_params)

# Assign CRS to LAI-Rasters
alg_params = {
    'INPUT': outputs['LAI']['OUTPUT'],
    'CRS':QgsCoordinateReferenceSystem('EPSG:3006')}

processing.run("gdal:assignprojection", alg_params)

alg_params = {
    'INPUT': outputs['LAINodata']['OUTPUT'],
    'CRS':QgsCoordinateReferenceSystem('EPSG:3006')}
processing.run("gdal:assignprojection", alg_params)

test = 4