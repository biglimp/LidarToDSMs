# LidarToDSMs

This code generates high resolution surface models and other data from a LiDAR point cloud making use of FUSION LDV (available only for Windows) and the OsGeo/QGIS python environment.


**Input data required:**
1. Lidar point cloud (currently NH - Nationell höjddatabas)
2. Building vector foot print (polygon) in shapefile format
3. Vector polygon representing domain (shapefile)
4. Water (downloaded from Open Street Map - Quick OSM plugin required)


**Output data generated:**
1. DSM: Digital surface model (ground and building heights - masl)
2. CDSM: Canopy Digital Surface Model (tall vegetation - magl)
3. DEM: Digital Elevation Model (ground heights only - masl)
4. Landcover (7-classes)
5. LAI (Leaf Area Index)

Main file: makedsmfromlidar.py
