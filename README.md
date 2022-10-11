# LidarToDSMs

This code generates high resolution surface models and other data from a LiDAR point cloud making use of FUSION LDV (available only for Windows) and the OsGeo/QGIS python environment.


**Input data required:**
1. Lidar point cloud
2. Building vector foot print (polygon) and shapefile format
3. Vector polygon representing extent (shapefile)
4. Water (downloaded from Open Street Map - Quick OSM plugin required)


**Output data generated:**
1. DSM: Digital surface model (ground and building heights - masl)
2. CDSM: Canopy Digital Surface Model (tall vegetation - magl)
3. Landcover (7-classes)
4. LAI (Leaf Area Index)

Main file: makedsmfromlidar.py
