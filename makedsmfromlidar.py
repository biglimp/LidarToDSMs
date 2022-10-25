
'''
This script generates DSMs from Lidar data inconjunction to a building footprint vector layer.
It also generates a LAI dataset
This script is based on Swedish national lidar point cloud but any point cloud could be used if correct classes are specified
Output:
dem.tif - Digital elevation model (masl)
dsm.tif - Digital Surface Model (masl)
cdsm.tif - Canopy Digital Surface Model (magl)
lc.tif - land cover 
lai_10m.tif - Leaf Area Index
'''

# Initialisation
# from doctest import OutputChecker
# # from tkinter.messagebox import NO
from osgeo import gdal
from qgis.core import QgsApplication, QgsVectorLayer, QgsCoordinateReferenceSystem
import numpy as np
from osgeo.gdalconst import GDT_Float32, GA_ReadOnly
import sys, os
import shutil
# import matplotlib.pylab as plt
import time
# from pathlib import Path

start = time.time()

### Input data paths and settings ###
windowsuser = 'xlinfr'
workingpath = 'D:/LidarQGISFUSION/tempfromscript/'
domain = 'D:/LidarQGISFUSION/FastighetskartanVektor_1501_3006/rutnat_get.shp'
buildingFootprint = 'D:/LidarQGISFUSION/FastighetskartanVektor_1501_3006/by_get.shp'
lidardata = 'D:/LidarQGISFUSION/Laserdata_1501_3006/09B002_63975_3175_25/09B002_63975_3175_25.las'
outputfolder = 'D:/temp/'

#test
#TODO: bara_bygg is not tested proparly. Didnt fint a good testing area
# domain = 'D:/LidarQGISFUSION/Probs/bara_bygg.shp' #inga_byggnader.shp' #ingen_highveg.shp'
# buildingFootprint = 'D:/LidarQGISFUSION/Probs/byggnader.shp'
# lidardata = 'D:/LidarQGISFUSION/Probs/09B002_639_32_2500.las'



EPSGnum = 3006          # Target CRS. Make sure all data is in same CRS
intensitylimit = 175    # to identify grass surfaces from Lidar
cellsize = 2            # Cellsize for output raster
buildingbuffer = 2.5    # Buffersize from building footprints
zfilter = 2.5           # height above ground to filter out low vegetation
outputs = {}
#TODO include classes in las-file


# Initiating a QGIS application and connect to processing
qgishome = 'C:/OSGeo4W/apps/qgis/'
QgsApplication.setPrefixPath(qgishome, True)
app = QgsApplication([], False)
app.initQgis()

sys.path.append(r'C:\OSGeo4W\apps\qgis\python\plugins')
sys.path.append('C:/Users/' + windowsuser + '/AppData/Roaming/QGIS/QGIS3/profiles/default/python/plugins') # path to third party plugins
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

def dtm2ascii(input, output):
    alg_params = {
    'INPUT': input, 
    'CSV':False,
    'RASTER':True, #TEsting with FALSE
    'MULTIPLIER':None,
    'OUTPUT': output}
    return processing.run("fusion:dtm2ascii", alg_params)

def clipdata (input, output, dtm,adv_modifier):
    alg_params = {
        'INPUT':input,
        'EXTENT':projwin,
        'SHAPE':0,
        'VERSION64':True,
        'OUTPUT': output, 
        'DTM': dtm,
        'HEIGHT':False,
        'IGNOREOVERLAP':False,
        'CLASS':'',
        'ADVANCED_MODIFIERS': adv_modifier
    }
    return processing.run("fusion:clipdata", alg_params)

def gdal_rasterize(input, output):

    alg_params = {
        'INPUT': input,
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
        'OUTPUT': output}
    return processing.run("gdal:rasterize", alg_params)

def rastercalculator(output,calc_formula, A, B = None, C = None): 
    if B == None:
        Bband = 0
    else:
        Bband = 1
    if C == None:
        Cband = 0
    else:
        Cband = 1

    alg_params = {
        'INPUT_A': A,
        'BAND_A':1,
        'INPUT_B': B,
        'BAND_B':Bband,
        'INPUT_C':C,
        'BAND_C':Cband,
        'INPUT_D':None,'BAND_D':None,'INPUT_E':None,'BAND_E':None,'INPUT_F':None,'BAND_F':None,
        'FORMULA': calc_formula,
        'NO_DATA':None,'PROJWIN':projwinrasterize,'RTYPE':5,'OPTIONS':'','EXTRA':'',
        'OUTPUT': output
        } 
    return processing.run("gdal:rastercalculator", alg_params)

import processing
from processing_umep.processing_umep_provider import ProcessingUMEPProvider
umep_provider = ProcessingUMEPProvider()
QgsApplication.processingRegistry().addProvider(umep_provider)

from processing_fusion.processing_fusion_provider import ProcessingFUSIONProvider
fusion_provider = ProcessingFUSIONProvider()
QgsApplication.processingRegistry().addProvider(fusion_provider)

from processing.core.Processing import Processing
Processing.initialize()

from QuickOSM.quick_osm_processing.provider import Provider
quickOSM_provider = Provider()
QgsApplication.processingRegistry().addProvider(quickOSM_provider)

### Start of process ###
if os.path.exists(workingpath):
     shutil.rmtree(workingpath)
os.mkdir(workingpath)

vlayer = QgsVectorLayer(domain, "polygon", "ogr")
extent = vlayer.extent()
projwin = str(extent.xMinimum()) + ',' + str(extent.xMaximum()) + ',' + str(extent.yMinimum()) +',' + str(extent.yMaximum()) + ' [EPSG:' + str(EPSGnum) + ']'

print('Clipping LAS-file')

outputs['ClipLidardata']  = clipdata(lidardata, workingpath + 'clippedLidar.las', '','')

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

outputs['DEMDTMtoASCII'] = dtm2ascii(workingpath + 'ground.dtm', workingpath + 'dem.asc')

# dem ascii to tif
data2 = gdal.Open(workingpath + 'dem.asc', GA_ReadOnly)
dem = data2.ReadAsArray().astype(float)
nd = data2.GetRasterBand(1).GetNoDataValue() # one line of nodata to the left is removed
dem[dem==nd] = 0
saveraster(data2, workingpath + 'dem.tif', dem)

#TODO polyclipdata dose not work as processing alg. Running with os.system() here instead. Seems to be issues with /\ and long paths. Issue has been raised to FUSION developers.
#TODO Seems to work now. Strange... Keeping os.system() processing.run("fusion:polyclipdata", {'INPUT':'D:\\LidarQGISFUSION\\tempfromscript\\clippedLidar.las','MASK':'D:\\LidarQGISFUSION\\FastighetskartanVektor_1501_3006\\by_get.shp','VERSION64':True,'OUTPUT':'D:/LidarQGISFUSION/tempfromscript/test4.las','SHAPE':False,'FIELD':'','VALUE':'','ADVANCED_MODIFIERS':'/class:1'})

print('Clipping out building points')
alg_params = 'C:/FUSION/PolyClipData.exe /class:1 "'  + buildingFootprint + '" "' + workingpath + 'building.las" "' + outputs['ClipLidardata']['OUTPUT'] + '"'
os.system(alg_params)

print('Clipping out ground points')
alg_params = 'C:/FUSION/PolyClipData64.exe /outside /class:2 "'  + buildingFootprint + '" "' + workingpath + 'ground.las" "' + outputs['ClipLidardata']['OUTPUT'] + '"'
os.system(alg_params)

if os.path.exists(workingpath + 'ground.las'):
    pass
else:
    alg_params = {
        'INPUT':outputs['ClipLidardata']['OUTPUT'],
        'OUTPUT': workingpath + 'ground.las',
        'ADVANCED_MODIFIERS':'/class:2'}
    processing.run("fusion:mergelasfiles", alg_params)

if os.path.exists(workingpath + 'building.las'):
    building_exist = 1
    print('Make DSM from ground and building points')

    alg_params = {
    'INPUT': workingpath + 'building.las;' + workingpath + 'ground.las',
    'CELLSIZE':cellsize,
    'XYUNITS':0,
    'ZUNITS':0,
    'VERSION64':True,
    'OUTPUT':workingpath + 'dsm.dtm',
    'GROUND':'','MEDIAN':'','SMOOTH':'','CLASS':'','SLOPE':False,'ASCII':False,
    'ADVANCED_MODIFIERS':''
}
    outputs['DSMCanopyModel'] = processing.run("fusion:canopymodel",alg_params)
    
    outputs['DSMDTMtoASCII'] = dtm2ascii(outputs['DSMCanopyModel']['OUTPUT'], workingpath + 'dsm.asc')

else :
    print('No Buildings found in the LiDAR Pointcloud')
    building_exist = 0
    # create empty dsm
    data2 = gdal.Open(workingpath + 'dem.tif', GA_ReadOnly)
    dsm = data2.ReadAsArray().astype(float)
    saveraster(data2, workingpath + 'dsm.tif', dsm)
    
    outputs['DSMDTMtoASCII'] = {}
    outputs['DSMDTMtoASCII']['OUTPUT'] = workingpath + 'dem.tif' #issue #1

#fix new projwin to allign rasters
data = gdal.Open(outputs['DSMDTMtoASCII']['OUTPUT'], GA_ReadOnly)
geoTransform = data.GetGeoTransform()
minx = geoTransform[0]
maxy = geoTransform[3]
maxx = minx + geoTransform[1] * data.RasterXSize
miny = maxy + geoTransform[5] * data.RasterYSize
data = None
projwinrasterize = str(minx) + ',' + str(maxx) + ',' + str(miny) +',' + str(maxy) + ' [EPSG:' + str(3006) + ']'

print('Clipping out vegetation points')
if building_exist == 1:
    alg_params = 'C:/FUSION/PolyClipData64.exe /outside /class:1 "'  + buildingFootprint + '" "' + workingpath + 'veg.las" "' + outputs['ClipLidardata']['OUTPUT'] + '"'
    os.system(alg_params)
else:
    alg_params = {
        'INPUT':outputs['ClipLidardata']['OUTPUT'],
        'OUTPUT': workingpath + 'veg.las',
        'ADVANCED_MODIFIERS':'/class:1'}
    processing.run("fusion:mergelasfiles", alg_params)

#check if veg.las includes any points
alg_params = {
    'INPUT':workingpath + 'veg.las','OUTPUT':workingpath + 'check_veg.csv','DENSITY':'','FIRSTDENSITY':'','INTENSITY':'','ADVANCED_MODIFIERS':''}
processing.run("fusion:catalog", alg_params)
a = np.genfromtxt(workingpath + 'check_veg.csv', skip_header=1, delimiter=',',missing_values='**********', filling_values=-9999)
if a[9] == 0:
    veg_exist = 0
else: 
    veg_exist = 1

#TODO if no High vegeation is present in LidarPointcloud, no CDSM will be created wich will create problems further down when making landcover.tif
if veg_exist == 1:
    print('Make CDSM of filtered vegetation points')
    alg_params = {
    'INPUT':workingpath + 'veg.las;' + workingpath + 'ground.las',
    'CELLSIZE': cellsize,
    'XYUNITS':0,
    'ZUNITS':0,
    'VERSION64':True,
    'OUTPUT':workingpath + 'cdsm.dtm',
    'GROUND':workingpath + 'ground.dtm', # using DTM switch is not working 
    'MEDIAN':'','SMOOTH':'','CLASS':'','SLOPE':False,'ASCII':False,
    'ADVANCED_MODIFIERS':' '
    }
    outputs['CDSMCanopyModel'] = processing.run("fusion:canopymodel",alg_params)

    outputs['CDSMDTMtoASCII'] = dtm2ascii(outputs['CDSMCanopyModel']['OUTPUT'], workingpath + 'cdsmraw.asc')

    # csdm ascii to tif
    data3 = gdal.Open(workingpath + 'cdsmraw.asc', GA_ReadOnly)
    cdsmtemp = data3.ReadAsArray().astype(float)
    nd = data3.GetRasterBand(1).GetNoDataValue() # one line of nodata to the left is removed
    cdsmtemp[cdsmtemp==nd] = 0
    saveraster(data3, workingpath + 'cdsm.tif', cdsmtemp)

else:
    print('No High vegetation found in the LiDAR Pointcloud')
    veg_exist = 0
    data3 = gdal.Open(workingpath + 'dem.asc', GA_ReadOnly)
    demo = data3.ReadAsArray().astype(float)
    cdsm = np.copy(demo) * 0.0
    saveraster(data3, workingpath + 'cdsm.tif', cdsm)
    outputs['CDSMDTMtoASCII'] = {}
    outputs['CDSMDTMtoASCII']['OUTPUT'] = workingpath + 'cdsm.tif' #issue #2

if building_exist == 1:
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

    outputs['BufferedBuildings'] = gdal_rasterize(outputs['BufferedBuildingsTif']['OUTPUT'],workingpath + 'buff_bolean.tif')

    outputs['RemoveBufferedBuildings'] = rastercalculator(workingpath + 'cdsm_filt.tif','A * B', outputs['CDSMDTMtoASCII']['OUTPUT'],outputs['BufferedBuildings']['OUTPUT'])
else:
    outputs['RemoveBufferedBuildings'] = {}
    outputs['RemoveBufferedBuildings']['OUTPUT'] = workingpath + 'cdsm.tif'

if veg_exist == 1:
    print('Removing vegpoints lower than 2.5 meter above ground')
    alg_params = {
        'INPUT_A':outputs['RemoveBufferedBuildings']['OUTPUT'],
        'BAND_A':1,
        'INPUT_B':None,
        'BAND_B':None,
        'INPUT_C':None,'BAND_C':None,'INPUT_D':None,'BAND_D':None,'INPUT_E':None,'BAND_E':None,'INPUT_F':None,'BAND_F':None,
        'FORMULA':'(A > ' + str(zfilter) + ') * A',
        'NO_DATA':None,'PROJWIN':projwinrasterize,'RTYPE':5,'OPTIONS':'','EXTRA':'',
        'OUTPUT':workingpath + 'cdsm_2point5meter.tif'
    }
    outputs['RemovedLOWHEIGHTSCDSM'] = processing.run("gdal:rastercalculator", alg_params)

    print('vegetation filtering/refinements')
    # filters to be included (remove holes and fill holes (and maybe linear also))
    data = gdal.Open(outputs['RemovedLOWHEIGHTSCDSM']['OUTPUT'], GA_ReadOnly)
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

    saveraster(data,workingpath + 'cdsm_temp.tif', y2)

    # Clip CDSM to fit with DSM
    bigraster = gdal.Open(workingpath + 'cdsm_temp.tif')
    bbox = (minx, maxy, maxx, miny) 
    gdal.Translate(workingpath + 'cdsm.tif', bigraster, projWin=bbox) # Clip raster

# remove nodataline from dsm
data2 = gdal.Open(outputs['DSMDTMtoASCII']['OUTPUT'], GA_ReadOnly)
dsm = data2.ReadAsArray().astype(float)
nd = data2.GetRasterBand(1).GetNoDataValue() # one line of nodata to the left is removed
dsm[dsm==nd] = 0
saveraster(data2, workingpath + 'dsm.tif', dsm)

# land cover
print('Making landcover from lidar')
outputs['BuildingsBoolean'] = gdal_rasterize(buildingFootprint, workingpath + 'build_bolean.tif')

if veg_exist == 1:
    # If there is no high vegetation, this will not work. Therefore try: statement
    alg_params = {
        'INPUT_A': workingpath + 'cdsm.tif',
        'BAND_A':1,
        'INPUT_B':None,
        'BAND_B':None,
        'INPUT_C':None,'BAND_C':None,'INPUT_D':None,'BAND_D':None,'INPUT_E':None,'BAND_E':None,'INPUT_F':None,'BAND_F':None,
        'FORMULA':'(A > 0)',
        'NO_DATA':None,'PROJWIN':'','RTYPE':5,'OPTIONS':'','EXTRA':'',
        'OUTPUT':workingpath + 'veg_bolean.tif'
    }
    outputs['vegBoolean'] = processing.run("gdal:rastercalculator", alg_params)

else:
    outputs['vegBoolean'] = {}
    outputs['vegBoolean']['OUTPUT'] = workingpath + 'cdsm.tif' 

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

outputs['Intensity1Raster'] = rastercalculator(workingpath + 'intensity1.tif','A * 1',workingpath + 'intensity.jpg',)

outputs['Intensity1RasterNodata'] = rastercalculator(workingpath + 'intensity1nodata.tif','(A < 255) * A', outputs['Intensity1Raster']['OUTPUT'])
#((“Intensity1@1” < 255) * “Intensity1@1”)

outputs['Intensity1RasterFinal'] = rastercalculator(workingpath + 'lc_boleantemp.tif','(A >= ' + str(intensitylimit) + ')', outputs['Intensity1RasterNodata']['OUTPUT'],)

# Clip Intensity to fit with CDSM
bigraster = gdal.Open(workingpath + 'lc_boleantemp.tif')
bbox = (minx, maxy, maxx, miny) 
gdal.Translate(workingpath + 'lc_bolean.tif', bigraster, projWin=bbox) # Clip raster

print('extracting grass raster')
alg_params = {
    'INPUT_A':outputs['BuildingsBoolean']['OUTPUT'],
    'BAND_A':1,
    'INPUT_B':outputs['vegBoolean']['OUTPUT'],
    'BAND_B':1,
    'INPUT_C':workingpath + 'lc_bolean.tif',
    'BAND_C':1,
    'INPUT_D':None,'BAND_D':None,'INPUT_E':None,'BAND_E':None,'INPUT_F':None,'BAND_F':None,
    'FORMULA':'(B * -1 + 1) * C * A',   #("veg_bolean@1" * -1 + 1) * "lc_bolean@1" * "buff_bolean@1"
    'NO_DATA':None,'PROJWIN':None,'RTYPE':5,'OPTIONS':'','EXTRA':'',
    'OUTPUT': workingpath + 'grass.tif'
}

outputs['GrassRaster'] = processing.run("gdal:rastercalculator", alg_params)


alg_params = {
    'INPUT_A':outputs['BuildingsBoolean']['OUTPUT'],
    'BAND_A':1,
    'INPUT_B':outputs['vegBoolean']['OUTPUT'],
    'BAND_B':1,
    'INPUT_C':workingpath + 'grass.tif',
    'BAND_C':1,
    'INPUT_D':None,'BAND_D':None,'INPUT_E':None,'BAND_E':None,'INPUT_F':None,'BAND_F':None,
    'FORMULA':'A + (C * 2) + ((A * B) * 3)',   #(“build_bolean@1”) + (“lc_bolean@1” * 2) + ((“build_bolean@1” * “veg_bolean@1”) * 3)
    'NO_DATA':None,'PROJWIN':None,'RTYPE':5,'OPTIONS':'','EXTRA':'',
    'OUTPUT': workingpath + 'lcunclassed.tif'
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

print('Download Water Polygons from Open Street Map')

try:
    alg_params = {   
        'KEY':'water',
        'VALUE':'',
        'EXTENT':projwin,
        'TIMEOUT':25,
        'SERVER':'https://lz4.overpass-api.de/api/interpreter',
        'FILE': workingpath + 'water.gpkg'}

    outputs['WaterPolygons'] = processing.run("quickosm:downloadosmdataextentquery", alg_params)

    #Reproject to Correct CRS
    print('Reproject to selected CRS')
    alg_params = {
        'INPUT': workingpath + '/water.gpkg|layername=water_multipolygons',
        'TARGET_CRS':QgsCoordinateReferenceSystem.fromEpsgId(EPSGnum),
        'OPERATION':'+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad +step +proj=utm +zone=33 +ellps=GRS80',
        'OUTPUT': workingpath + 'water_reproject.shp' }

    outputs['ReprojecterWaterPolygons'] = processing.run("native:reprojectlayer", alg_params )

    # Clip to extent/domain
    print('Clip to domain')
    alg_params = {
        'INPUT' : workingpath + 'water_reproject.shp',
        'OVERLAY': domain,
        'OUTPUT': workingpath + 'water_reproject_clip.shp'}

    outputs['ClippedWaterPolygons'] = processing.run("native:clip", alg_params) 

    print('Brun 7 (water) into LC where there are water polygons')
    alg_params = {
        'INPUT':workingpath + 'water_reproject_clip.shp',
        'INPUT_RASTER': workingpath + 'lc.tif',
        'BURN':7, # 7 = Water
        'ADD':False,
        'EXTRA':''}

    outputs['LandCoverRasterWater'] = processing.run("gdal:rasterize_over_fixed_value", alg_params) 
    
except:
    print('no water found in domain')



print('LAI estimation from Lidar')
if veg_exist == 1:
    if building_exist == 1:
        print('Clipping out vegetation points with buffered buildings')
        alg_params = 'C:/FUSION/PolyClipData64.exe /outside /class:1 "'  + outputs['BufferedBuildingsTif']['OUTPUT'] + '" "' + workingpath + 'vegbuff.las" "' + outputs['ClipLidardata']['OUTPUT'] + '"'
        os.system(alg_params)

        alg_params = {
            'INPUT':workingpath + 'vegbuff.las;' + workingpath + 'ground.las',
            'OUTPUT':workingpath + 'lai.las',
            'ADVANCED_MODIFIERS':''
        }
        outputs['LAIcloud'] = processing.run("fusion:mergelasfiles", alg_params)
    else:
        alg_params = {
            'INPUT':workingpath + 'veg.las;' + workingpath + 'ground.las',
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
        'CRS':QgsCoordinateReferenceSystem.fromEpsgId(EPSGnum)}

    processing.run("gdal:assignprojection", alg_params)
    shutil.copyfile(outputs['LAI']['OUTPUT'], outputfolder + 'lai_10m.tif')
else:
    print('No vegetation in domain. No LAI created')

# Assign CRS to Rasters
alg_params = {'INPUT': workingpath + 'dem.tif', 'CRS':QgsCoordinateReferenceSystem.fromEpsgId(EPSGnum)}
processing.run("gdal:assignprojection", alg_params)
alg_params = {'INPUT': workingpath + 'dsm.tif', 'CRS':QgsCoordinateReferenceSystem.fromEpsgId(EPSGnum)}
processing.run("gdal:assignprojection", alg_params)
alg_params = {'INPUT': workingpath + 'cdsm.tif', 'CRS':QgsCoordinateReferenceSystem.fromEpsgId(EPSGnum)}
processing.run("gdal:assignprojection", alg_params)

shutil.copyfile(workingpath + 'lc.tif', outputfolder + 'lc.tif')
shutil.copyfile(workingpath + 'dem.tif', outputfolder + 'dem.tif')
shutil.copyfile(workingpath + 'dsm.tif', outputfolder + 'dsm.tif')
shutil.copyfile(workingpath + 'cdsm.tif', outputfolder + 'cdsm.tif')

end = time.time()
total_time = end - start
print('Script finished in ' + str(total_time) + ' seconds' )