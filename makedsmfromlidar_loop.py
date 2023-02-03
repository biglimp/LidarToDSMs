
'''
This script generates DSMs from Lidar data inconjunction to a building footprint vector layer.
It also generates a LAI dataset
This script is currently based on Swedish national lidar point cloud but any point cloud could be used if correct classes are specified
This script is set up as a loop to process many tiles of Lidar-data.
laz and well as las could be used as input.
The Script is set for Windows but if correct paths is specified, any OS should be possible.

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
import matplotlib.pylab as plt
import time
# from pathlib import Path
# import subprocess
# *****************************************************************************
import warnings
warnings.filterwarnings("ignore")
# *****************************************************************************

### Input data paths and settings ###
windowsuser = 'xlinfr'                          # set username
infolder = 'C:/temp/Sandra/'        # inputfolder for all input data
outfolder = 'C:/temp/Sandra/out/'              # outputfolder
mergeoutput = outfolder + 'mergeoutput/'        # Outfolder for merged dsm, dem, cdsm & lc
workingpath = outfolder + 'tempdata/'           # Path to temp-folder DO NOT CHANGE
lidar_folder = 'Skogsdata/'                  # Set to '' if lidar files in infolder
lidarExtension = '.laz'                         # or .las
updateLAS = 'no'                               # Set to yes if maxPoints is changed
maxPoints = 2                               # maximum number of pulses per square meter to include (if possible)
NH_lidar = 'yes'                                # yes if Nationell höjddata from Lantmäteriet is used

domain = None                                    # polygon to clip with. if using whole .las file(s), set this to None
buildingFootprint = infolder + 'by_gbg.shp'      # Buildings polygons 
waterpolygon = infolder + 'vattenGöteborg.shp'    # if no water polygon Set to None. Then OSMdata will be used TODO: water.gpdb from osm is not deleted
bridges = None #infolder + 'bridges_poly.shp'    # if no bridges polygon set to None
railway = None #infolder + 'rail.shp'            # if no railway polygon set to None

#TODO: only build with no high veg is not tested proparly. Didnt fint a good testing area

EPSGnum = 3006          # Target CRS. Make sure all data is in same CRS
intensitylimit = 115    # to identify grass surfaces from Lidar. Values above will be classified as grass [0 to 255]
cellsize = 2            # Cellsize for output raster in meters
buildingbuffer = 2.2    # Buffersize from building footprints in meters
zfilter = 2.5           # min height above ground to filter out low vegetation (meter) 
zlim = 35               # max height above ground to filer out (items such as bulding cranes and other stuff that may interfere) (meter) 
lowLimit = 0            # low relative limit for DSM (meter) 
highLimit = 120         # high relative limit for DSM (meter)
calculate_LAI = 'no'   # Calculate LAI? yes or no

# Set names for lidarfiles to use as list without .las. Set to None to automatically read file list
lidar_list = None
if lidar_list is None:
    lidar_list = [] 
    for file in os.listdir(infolder + lidar_folder):
        if file.endswith(lidarExtension):
            lidar_list.append(file[:-4])

#TODO include classes from standard las-file. 
if NH_lidar.lower() == 'yes':
    building_class = 1
    ground_class = 2
    unclassified = 1
    veg_class = None # not used yet
else: # standard LAS
    building_class = 6
    ground_class = 2
    unclassified = 1
    veg_class = [3,4,5] # not used yet    

# Initiating a QGIS application and connect to processing
qgishome = 'C:/OSGeo4W/apps/qgis/'
QgsApplication.setPrefixPath(qgishome, True)
app = QgsApplication([], False)
app.initQgis()

sys.path.append('C:/OSGeo4W/apps/qgis/python/plugins')
sys.path.append('C:/Users/' + windowsuser + '/AppData/Roaming/QGIS/QGIS3/profiles/default/python/plugins') # path to third party plugins
sys.path.append('C:/FUSION')

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

# Internal functions
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
    'RASTER':False, #TEsting with FALSE
    'MULTIPLIER':None,
    'OUTPUT': output
    }
    return processing.run("fusion:dtm2ascii", alg_params)

def clipdata (input, output,projwin,  dtm, adv_modifier):
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

def gdal_rasterize(input, output, projwinrasterize):

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
        'OUTPUT': output
        }
    return processing.run("gdal:rasterize", alg_params)

def rastercalculator(output, projwinrasterize,  calc_formula, A, B = None, C = None, ): 
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

# Function to overwrite raster with shapefile. for instance set all pixels within water polygon to 7 in lc.
def overwrite_raster(input_raster, input_vector, value):
    '''
    input_vector = polygonlayer
    input_raster = raster that is going to be overwritten
    value = value that is to be written into the pixels inside the polygons
    '''
    alg_params = {
        'INPUT':  input_vector,
        'INPUT_RASTER': input_raster,
        'BURN': value, # 7 = Water
        'ADD':False,
        'EXTRA':''
        }
    return processing.run("gdal:rasterize_over_fixed_value", alg_params) 


def makedsmfromlidar(filename, lidardata, outputfolder, building_clip_path, building_buff_path):    
    print('\n***************************************\nProcessing LASfile: ', filename, '\n***************************************')
    
    outputs = {}
    ### Start of process ###
    # TODO errors here with trying to remove building clip shp. Tested with making it temporary output,
    # but it gets error with FUSION as the temporary file is a .gpkg file, and Fusion requires .shp
    # temporary fix using different file names
    
    if os.path.exists(workingpath):
        shutil.rmtree(workingpath, ignore_errors= True) # ignore error cleans the folder except for 2 building_clip files.
        if not os.path.exists(workingpath):
            os.mkdir(workingpath)
    else:
        os.mkdir(workingpath)

    if os.path.exists(outputfolder):
        shutil.rmtree(outputfolder)
        os.mkdir(outputfolder)
    else:
        os.mkdir(outputfolder)

    # converts laz ot las (if nececcary)
    if lidarExtension == '.laz':
        # if not os.path.exists(lidardata):
        if updateLAS == 'yes':
            print("Convert laz to las and perform thinning to remove stripes")
            if os.path.exists(lidardata):
                os.remove(lidardata)
            alg_params = {
            'INPUT':lidardata[:-4] + '.laz',
            'DENSITY': maxPoints,
            'CELLSIZE':10,
            'RSEED':None,
            'CLASS':'',
            'IGNOREOVERLAP':False,
            'VERSION64':True,
            'OUTPUT':lidardata}
            processing.run("fusion:thindata", alg_params)

    if domain == None:
        # Get extent coordinates for .las file (minx, miny, maxx, maxy)
        if NH_lidar == 'yes':
            minx = lidardata[-11:-9] + lidardata[-6:-4] + '00.0'
            miny = lidardata[-15:-12] + lidardata[-8:-6] + '00.0'
            maxx = str(float(minx) + 2500)
            maxy = str(float(miny) + 2500)
        else:
            alg_params = {
                'INPUT': lidardata,
                'OUTPUT': workingpath + 'check_inputlas.csv',
                'DENSITY':'',
                'FIRSTDENSITY':'',
                'INTENSITY':'',
                'ADVANCED_MODIFIERS':''
                }
            processing.run("fusion:catalog", alg_params)

            las_catalog = np.genfromtxt(workingpath + 'check_inputlas.csv', skip_header=1, delimiter=',',missing_values='**********', filling_values=-9999)
            minx, miny, maxx, maxy = las_catalog[2], las_catalog[3], las_catalog[5], las_catalog[6] 
        
        projwin = str(minx) + ',' + str(maxx) + ',' + str(miny) +',' + str(maxy) + ' [EPSG:' + str(EPSGnum) + ']'
        outputs['ClipLidardata']  = {'OUTPUT' : lidardata}# = clipdata(lidardata, workingpath + 'clippedLidar.las', dtm='', adv_modifier = '/zmax:100')

        # Clip Building polygons to extent
        alg_params =  {
            'INPUT': buildingFootprint,
            'EXTENT': projwin,
            'OPTIONS':'',
            'OUTPUT': building_clip_path
            }
        outputs['BuildingsClip'] = processing.run("gdal:clipvectorbyextent", alg_params)
        # outputs['BuildingsClip'] = {'OUTPUT' : buildingFootprint}

    else:
        vlayer = QgsVectorLayer(domain, "polygon", "ogr")
        extent = vlayer.extent()
        projwin = str(extent.xMinimum()) + ',' + str(extent.xMaximum()) + ',' + str(extent.yMinimum()) +',' + str(extent.yMaximum()) + ' [EPSG:' + str(EPSGnum) + ']'
        
        alg_params =  {
            'INPUT': buildingFootprint,
            'EXTENT': projwin,
            'OPTIONS':'',
            'OUTPUT': 'TEMPORARY_OUTPUT'
            }
        outputs['BuildingsClip'] = processing.run("gdal:clipvectorbyextent", alg_params)

        print('Clipping LAS-file')
        outputs['ClipLidardata']  = clipdata(lidardata, workingpath + 'clippedLidar.las', projwin,  dtm='')

    print('Generate ground elevation model (DEM)')
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
        'CLASS': str(ground_class),
        'SLOPE':'',
        'ADVANCED_MODIFIERS':''}
        
    processing.run("fusion:gridsurfacecreate", alg_params)

    outputs['DEMDTMtoASCII'] = dtm2ascii(workingpath + 'ground.dtm', workingpath + 'dem.asc')

    # alligning DEM and fill no data in water
    alg_params = {'INPUT':workingpath + 'dem.asc','PROJWIN':projwin,'OVERCRS':False,'NODATA':None,'OPTIONS':'','DATA_TYPE':0,'EXTRA':'','OUTPUT':workingpath + 'demtemp.tif'}
    outputs['DEMDTMtoASCII'] = processing.run("gdal:cliprasterbyextent", alg_params)
    processing.run("native:fillnodata", {'INPUT':workingpath + 'demtemp.tif','BAND':1,'FILL_VALUE':0,'OUTPUT':workingpath + 'dem.tif'})

    print('Clipping out building points')
    alg_params  = {
        'INPUT': outputs['ClipLidardata']['OUTPUT'],
        'MASK': outputs['BuildingsClip']['OUTPUT'] ,
        'VERSION64':True,
        'OUTPUT': workingpath + 'building.las',
        'SHAPE':False,
        'FIELD':'',
        'VALUE':'',
        'ADVANCED_MODIFIERS':'/class:' + str(building_class)}

    outputs['BuildingLAS'] = processing.run("fusion:polyclipdata", alg_params)

    print('Clipping out ground points')
    alg_params  = {
        'INPUT': outputs['ClipLidardata']['OUTPUT'],
        'MASK':outputs['BuildingsClip']['OUTPUT'] ,
        'VERSION64':True,
        'OUTPUT': workingpath + 'ground.las',
        'SHAPE':False,
        'FIELD':'',
        'VALUE':'',
        'ADVANCED_MODIFIERS':'/outside /class:' + str(ground_class)}

    outputs['GroundLAS'] = processing.run("fusion:polyclipdata", alg_params)

    if os.path.exists(outputs['GroundLAS']['OUTPUT']):
        pass
    else:
        alg_params = {
            'INPUT':outputs['ClipLidardata']['OUTPUT'],
            'OUTPUT': workingpath + 'ground.las',
            'ADVANCED_MODIFIERS':'/class:' + str(ground_class)
            }
        outputs['GroundLAS'] = processing.run("fusion:mergelasfiles", alg_params)

    if os.path.exists(workingpath + 'building.las'):
        building_exist = 1
        print('Make DSM from ground and building points')

        alg_params = {
            'INPUT':  outputs['BuildingLAS']['OUTPUT'] + ';' + outputs['GroundLAS']['OUTPUT'],
            'CELLSIZE':cellsize,
            'XYUNITS':0,
            'ZUNITS':0,
            'VERSION64':True,
            'OUTPUT':workingpath + 'dsm.dtm',
            'GROUND':'','MEDIAN':'','SMOOTH':'','CLASS':'','SLOPE':False,'ASCII':False,
            'ADVANCED_MODIFIERS': '/ground:' + workingpath + 'ground.dtm /outlier:' + str(lowLimit) + ',' + str(highLimit)
        }
        outputs['DSMCanopyModel'] = processing.run("fusion:canopymodel",alg_params)
        outputs['DSMDTMtoASCII'] = dtm2ascii(outputs['DSMCanopyModel']['OUTPUT'], workingpath + 'dsm.asc')
        # alligning DSM and fill no data in water
        
        processing.run("native:fillnodata", {'INPUT':workingpath + 'dsm.asc','BAND':1,'FILL_VALUE':0,'OUTPUT':workingpath + 'dsmtemp.tif'})
        # add ground heights to dsm
        outputs['DSMDTMtoTIFF'] = rastercalculator(workingpath + 'dsmtemp2.tif', projwin ,'A + B', workingpath + 'dem.tif',workingpath + 'dsmtemp.tif')
        alg_params = {'INPUT':workingpath + 'dsmtemp2.tif','PROJWIN':projwin,'OVERCRS':False,'NODATA':None,'OPTIONS':'','DATA_TYPE':0,'EXTRA':'','OUTPUT':workingpath + 'dsm.tif'}
        outputs['DSMDTMtoTIFF'] = processing.run("gdal:cliprasterbyextent", alg_params)

    else :
        print('No Buildings found in the LiDAR Pointcloud')
        building_exist = 0
        # create empty dsm
        data2 = gdal.Open(workingpath + 'dem.tif', GA_ReadOnly)
        dsm = data2.ReadAsArray().astype(float)
        saveraster(data2, workingpath + 'dsm.tif', dsm)
        
        outputs['DSMDTMtoASCII'] = {}
        outputs['DSMDTMtoASCII']['OUTPUT'] = workingpath + 'dem.tif' #issue #1

    print('Clipping out vegetation points')
    if building_exist == 1:
        alg_params  = {
            'INPUT': outputs['ClipLidardata']['OUTPUT'],
            'MASK': outputs['BuildingsClip']['OUTPUT'],#buildingFootprint ,
            'VERSION64':True,
            'OUTPUT': workingpath + 'veg.las',
            'SHAPE':False,
            'FIELD':'',
            'VALUE':'',
            'ADVANCED_MODIFIERS':'/outside /class:' + str(unclassified)}

        processing.run("fusion:polyclipdata", alg_params)

    else:
        alg_params = {
            'INPUT':outputs['ClipLidardata']['OUTPUT'],
            'OUTPUT': workingpath + 'veg.las',
            'ADVANCED_MODIFIERS':'/class:1'}
        processing.run("fusion:mergelasfiles", alg_params)

    # check if veg.las includes any points
    alg_params = {
        'INPUT':workingpath + 'veg.las',
        'OUTPUT':workingpath + 'check_veg.csv',
        'DENSITY':'',
        'FIRSTDENSITY':'',
        'INTENSITY':'',
        'ADVANCED_MODIFIERS':''}
    processing.run("fusion:catalog", alg_params)
    a = np.genfromtxt(workingpath + 'check_veg.csv', skip_header=1, delimiter=',',missing_values='**********', filling_values=-9999)

    if a[9] == 0:
        veg_exist = 0
    else: 
        veg_exist = 1

    #TODO if no High vegeation is present in LidarPointcloud, no CDSM will be created wich will create problems further down when making landcover.tif. fixed?
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
        'ADVANCED_MODIFIERS':''
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
            'INPUT': outputs['BuildingsClip']['OUTPUT'] ,
            'DISTANCE': buildingbuffer,
            'SEGMENTS':1,
            'END_CAP_STYLE':1,
            'JOIN_STYLE':1,
            'MITER_LIMIT':2,
            'DISSOLVE':True,
            'OUTPUT': building_buff_path}

        outputs['BufferedBuildingsTif'] = processing.run("native:buffer", alg_params)
        outputs['BufferedBuildings'] = gdal_rasterize(outputs['BufferedBuildingsTif']['OUTPUT'],workingpath + 'buff_bolean.tif', projwin)
        outputs['RemoveBufferedBuildings'] = rastercalculator(workingpath + 'cdsm_filt.tif', projwin ,'A * B', outputs['CDSMDTMtoASCII']['OUTPUT'],outputs['BufferedBuildings']['OUTPUT'])
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
            'NO_DATA':None,'PROJWIN':projwin,'RTYPE':5,'OPTIONS':'','EXTRA':'',
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

        print('remove points higher than ' , zlim, 'm')
        y2[np.where(y2 > zlim)] = 0

        saveraster(data,workingpath + 'cdsm_temp.tif', y2)

        # Clip CDSM to fit with DSM
        bigraster = gdal.Open(workingpath + 'cdsm_temp.tif')
        bbox = (minx, maxy, maxx, miny) 
        gdal.Translate(workingpath + 'cdsmtemp2.tif', bigraster, projWin=bbox) # Clip raster
        processing.run("native:fillnodata", {'INPUT':workingpath + 'cdsmtemp2.tif','BAND':1,'FILL_VALUE':0,'OUTPUT':workingpath + 'cdsm.tif'})

    # land cover
    print('Making landcover from lidar')
    outputs['BuildingsBoolean'] = gdal_rasterize(buildingFootprint, workingpath + 'build_bolean.tif', projwin)

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
            'OUTPUT':workingpath + 'veg_bolean.tif'}
        outputs['vegBoolean'] = processing.run("gdal:rastercalculator", alg_params)

    else:
        outputs['vegBoolean'] = {}
        outputs['vegBoolean']['OUTPUT'] = workingpath + 'cdsm.tif' 

    alg_params = {
        'INPUT':workingpath + 'ground.las',
        'ALLRET':False,
        'LOWEST':False,
        'HIST':False,
        'PIXEL':cellsize / 2.,
        'SWITCH':1, # bug in FUSION. Bitmap is jpg
        'OUTPUT': workingpath + 'intensity',
        'ADVANCED_MODIFIERS':''
        }
    outputs['IntensityRaster'] = processing.run("fusion:intensityimage", alg_params)

    outputs['Intensity1Raster'] = rastercalculator(workingpath + 'intensity1.tif', projwin, 'A * 1',workingpath + 'intensity.jpg')

    alg_params = {'INPUT':outputs['Intensity1Raster']['OUTPUT'],
        'SOURCE_CRS':QgsCoordinateReferenceSystem('EPSG:' + str(EPSGnum)),
        'TARGET_CRS':QgsCoordinateReferenceSystem('EPSG:' + str(EPSGnum)),
        'RESAMPLING':0,'NODATA':None,
        'TARGET_RESOLUTION':cellsize,
        'OPTIONS':'','DATA_TYPE':1,
        'TARGET_EXTENT':projwin,'TARGET_EXTENT_CRS':None,'MULTITHREADING':False,'EXTRA':'','OUTPUT':'TEMPORARY_OUTPUT'}
    outputs['IntensityWarp'] = processing.run("gdal:warpreproject", alg_params)
    
    shutil.copyfile(outputs['IntensityWarp']['OUTPUT'], outputfolder + 'intensity.tif')
    outputs['Intensity1RasterNodata'] = rastercalculator(workingpath + 'intensity1nodata.tif',None ,'(A < 255) * A', outputs['IntensityWarp']['OUTPUT'])
    outputs['Intensity1RasterFinal'] = rastercalculator(workingpath + 'lc_bolean.tif', None ,'(A >= ' + str(intensitylimit) + ')', outputs['Intensity1RasterNodata']['OUTPUT'])

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
        'OUTPUT': workingpath + 'grass.tif'}

    outputs['GrassRaster'] = processing.run("gdal:rastercalculator", alg_params)

    alg_params = {
        'INPUT_A':outputs['BuildingsBoolean']['OUTPUT'],
        'BAND_A':1,
        'INPUT_B':outputs['vegBoolean']['OUTPUT'],
        'BAND_B':1,
        'INPUT_C':outputs['GrassRaster']['OUTPUT'],#workingpath + 'grass.tif',
        'BAND_C':1,
        'INPUT_D':None,'BAND_D':None,'INPUT_E':None,'BAND_E':None,'INPUT_F':None,'BAND_F':None,
        'FORMULA':'A + (C * 2) + ((A * B) * 3)',   #(“build_bolean@1”) + (“lc_bolean@1” * 2) + ((“build_bolean@1” * “veg_bolean@1”) * 3)
        'NO_DATA':None,'PROJWIN':None,'RTYPE':5,'OPTIONS':'','EXTRA':'',
        'OUTPUT': workingpath + 'lcunclassed.tif'}

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
        'OUTPUT': workingpath + 'lc.tif'}

    outputs['LandCoverRaster'] = processing.run("native:reclassifybytable", alg_params)

    if waterpolygon == None:
        print('No Water polygon provided, trying to download Water Polygons from Open Street Map')

        try:
            alg_params = {   
                'KEY':'water',
                'VALUE':'',
                'EXTENT':projwin,
                'TIMEOUT':25,
                'SERVER':'https://lz4.overpass-api.de/api/interpreter',
                'FILE': workingpath + 'water.gpkg'}

            outputs['WaterPolygons'] = processing.run("quickosm:downloadosmdataextentquery", alg_params)
            print('Water Polygon Downloaded')

            #Reproject to Correct CRS
            print('Reproject to selected CRS')

            if EPSGnum == 3007:
                operation = '+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad +step +proj=tmerc +lat_0=0 +lon_0=12 +k=1 +x_0=150000 +y_0=0 +ellps=GRS80'
            elif EPSGnum == 3006:
                operation = '+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad +step +proj=utm +zone=33 +ellps=GRS80'
            else:
                #TODO any other CRS?
                return
            
            alg_params = {
                'INPUT': workingpath + '/water.gpkg|layername=water_multipolygons',
                'TARGET_CRS':QgsCoordinateReferenceSystem.fromEpsgId(EPSGnum),
                'OPERATION': operation,
                'OUTPUT': workingpath + 'water_reproject.shp' }

            outputs['ReprojecterWaterPolygons'] = processing.run("native:reprojectlayer", alg_params )

            # Clip to extent/domain
            print('Clip to domain')

            alg_params =  {
                'INPUT':   outputs['ReprojecterWaterPolygons']['OUTPUT'],
                'EXTENT': projwin,
                'OPTIONS':'',
                'OUTPUT': 'TEMPORARY_OUTPUT',# workingpath + 'water_reproject_clip.shp'
            }
            outputs['ClippedWaterPolygons'] = processing.run("gdal:clipvectorbyextent", alg_params)

            print('Burn 7 (water) into LC where there are water polygons')

            overwrite_raster(outputs['LandCoverRaster']['OUTPUT'], outputs['ClippedWaterPolygons']['OUTPUT'], 7)
            
            overwrite_raster(workingpath + 'cdsm.tif', outputs['ClippedWaterPolygons']['OUTPUT'], 0)
            
        except:
            print('no water found in domain')
    else:
        print('Burn 7 (water) into LC where there are water polygons') 

        # Set area within water polygon to 7 in lc
        overwrite_raster(workingpath + 'lc.tif', waterpolygon, 7)
        # Remove veg inside water polygon for cdsm
        overwrite_raster(workingpath + 'cdsm.tif', waterpolygon, 0)

    if bridges is not None:
        # remove veg on bridges 
        overwrite_raster(workingpath + 'cdsm.tif', bridges, 0)
        # set areas of bridges to paved 1 in lc             
        print('Burn 1 (paved) into LC where there are bridges') 
        overwrite_raster(workingpath + 'lc.tif', bridges, 1)

    if railway is not None:
        # remove veg on railway in cdsm
        overwrite_raster(workingpath + 'cdsm.tif', railway, 0)
        # set areas within railway to paved 1 in lc
        overwrite_raster(workingpath + 'lc.tif', railway, 1)

    alg_params = {'INPUT':workingpath + 'lc.tif','PROJWIN':projwin,'OVERCRS':False,'NODATA':None,'OPTIONS':'','DATA_TYPE':0,'EXTRA':'','OUTPUT':workingpath + 'landcover.tif'}
    processing.run("gdal:cliprasterbyextent", alg_params)

    if calculate_LAI.lower() == 'yes':
        print('LAI estimation from Lidar')
        if veg_exist == 1:
            if building_exist == 1:
                print('Clipping out vegetation points with buffered buildings')
                alg_params  = {
                'INPUT': outputs['ClipLidardata']['OUTPUT'],
                'MASK': outputs['BufferedBuildingsTif']['OUTPUT'],#buildingFootprint ,
                'VERSION64':True,
                'OUTPUT': workingpath + 'vegbuff.las',
                'SHAPE':False,
                'FIELD':'',
                'VALUE':'',
                'ADVANCED_MODIFIERS':'/outside /class:' + str(unclassified)}

                processing.run("fusion:polyclipdata", alg_params)

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
                'CLASS': str(ground_class)
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
                'OUTPUT': workingpath + 'lai_10m_nodata.tif'
            }
            outputs['LAINodata'] = processing.run("gdal:rastercalculator", alg_params)

            alg_params = {
                'INPUT_A':outputs['LAINodata']['OUTPUT'],
                'BAND_A':1,
                'INPUT_B':None,
                'BAND_B':None,
                'INPUT_C':None,'BAND_C':None,'INPUT_D':None,'BAND_D':None,'INPUT_E':None,'BAND_E':None,'INPUT_F':None,'BAND_F':None,
                'FORMULA':'( A < 100 ) * A ',
                'NO_DATA':None,'PROJWIN':None,'RTYPE':5,'OPTIONS':'','EXTRA':'',
                'OUTPUT': workingpath + 'lai_10m_nodatainf.tif'
            }
            outputs['LAINodataInf'] = processing.run("gdal:rastercalculator", alg_params)

            alg_params = {'INPUT':outputs['LAINodataInf']['OUTPUT'],'PROJWIN':projwin,'OVERCRS':False,'NODATA':None,'OPTIONS':'','DATA_TYPE':0,'EXTRA':'','OUTPUT':workingpath + 'laiclip.tif'}
            outputs['LAINodataClip'] = processing.run("gdal:cliprasterbyextent", alg_params)

            alg_params = {
                'INPUT':outputs['LAINodataClip']['OUTPUT'],
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

    for raster in ['dem.tif','dsm.tif','cdsm.tif', 'lc.tif',]:
        # Fill nodata-holes in rasters
        alg_params = {
            'INPUT': workingpath + raster,
            'BAND':1,
            'DISTANCE':10,
            'ITERATIONS':0,
            'NO_MASK':False,
            'MASK_LAYER':None,
            'OPTIONS':'',
            'EXTRA':'',
            'OUTPUT':outputfolder + raster}

        processing.run("gdal:fillnodata", alg_params)

    # shutil.copyfile(workingpath + 'lc.tif', outputfolder + 'lc.tif')
    # shutil.copyfile(workingpath + 'dem.tif', outputfolder + 'dem.tif')
    # shutil.copyfile(workingpath + 'dsm.tif', outputfolder + 'dsm.tif')
    # shutil.copyfile(workingpath + 'cdsm.tif', outputfolder + 'cdsm.tif')

    

##########################################################
###                 Loop Start                         ###
##########################################################

start = time.time()

# Create emppy lists to fill with rasternames used later to merge
lc_list = []
dem_list = []
dsm_list = []
cdsm_list = []
lai_list = []

if os.path.exists(mergeoutput):
    shutil.rmtree(mergeoutput)
    os.mkdir(outfolder + 'mergeoutput')
else:
    os.mkdir(outfolder + 'mergeoutput')

clipindex = 0

for filename in lidar_list:

    lidardata = infolder + lidar_folder + filename + '.las'
    outputfolder = infolder + filename + '/'

    lc_list.append(outputfolder + 'lc.tif')
    dem_list.append(outputfolder + 'dem.tif')
    dsm_list.append(outputfolder + 'dsm.tif')
    cdsm_list.append(outputfolder + 'cdsm.tif')
    lai_list.append(outputfolder + 'lai_10m.tif')

    building_clip_path = workingpath + 'building_clip' + str(clipindex) + '.shp'
    building_buff_path = workingpath + 'building_buff' + str(clipindex) + '.shp'

    makedsmfromlidar(filename=filename, lidardata=lidardata, outputfolder=outputfolder, building_clip_path=building_clip_path, building_buff_path=building_buff_path)
    clipindex = clipindex + 1

print('Merge rasters from all used LiDAR-Squares in to one .tif')
for raster_list, raster_name in zip([lc_list, dem_list, dsm_list, cdsm_list, lai_list], ['lc', 'dem', 'dsm', 'cdsm', 'lai']):
    
    if raster_name == 'cdsm': #TODO: remove possible stripe in merged cdsm (as gound is zero) using mosiac or something...
        alg_params = {
            'INPUT': raster_list,
            'PCT':False,
            'SEPARATE':False,
            'NODATA_INPUT':None,
            'NODATA_OUTPUT':None,
            'OPTIONS':'',
            'EXTRA':'',
            'DATA_TYPE':5,
            'OUTPUT': mergeoutput + raster_name + 'temp.tif' }
    else:
        alg_params = {
            'INPUT': raster_list,
            'PCT':False,
            'SEPARATE':False,
            'NODATA_INPUT':0,
            'NODATA_OUTPUT':None,
            'OPTIONS':'',
            'EXTRA':'',
            'DATA_TYPE':5,
            'OUTPUT': mergeoutput + raster_name + 'temp.tif' }

    processing.run("gdal:merge", alg_params)

# set nodata to -9999
for raster_name in ['lc', 'dem', 'dsm', 'cdsm', 'lai']:
    data3 = gdal.Open(mergeoutput + raster_name + 'temp.tif', GA_ReadOnly)
    raster = data3.ReadAsArray().astype(float)
    saveraster(data3, mergeoutput + raster_name + '.tif', raster)
    data3 = None
    os.remove(mergeoutput + raster_name + 'temp.tif')

end = time.time()
total_time = end - start

print('Script finished in ' + str(total_time / 60.) + ' minutes' )
outputs = None

app.exitQgis()
if os.path.exists(workingpath):
    shutil.rmtree(workingpath, ignore_errors= True) # ignore error cleans the folder except for 2 building_clip files.

