# LidarToDSMs

This code generates high resolution surface models and other data from a LiDAR point cloud making use of FUSION LDV (available only for Windows) and the OsGeo/QGIS python environment.


**Input data required:**
1. Lidar point cloud (currently NH - Nationell höjddatabas)
2. Building vector foot print (polygon) in shapefile format
3. Vector polygon representing domain (shapefile) [optional]
4. Water, downloaded from Open Street Map - (Quick OSM plugin required) or as a polygon vector layer [optional]
5. Elevated rail structures and bridges as polygon layers [optional]


**Output data generated:**
1. DSM: Digital surface model (ground and building heights - masl)
2. CDSM: Canopy Digital Surface Model (tall vegetation - magl)
3. DEM: Digital Elevation Model (ground heights only - masl)
4. Landcover (7-classes) [optional]
5. LAI (Leaf Area Index) [optional]

Main file: makedsmfromlidar_loop.py

**Remarks**
1. If you have .laz-files you need to make sure that FUSION can read .laz-files. Go to the FUSION-manual to see how.
2. If you have .laz-files, you need to set the parameter updateLAS to 'yes' the first time you run the code. This will unpack your .laz-files to .las.
