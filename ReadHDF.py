# -*- coding: utf-8 -*-
import numpy as np
import scipy
from osgeo import gdal, osr
import os
# from skimage import transform
from matplotlib import pyplot as plt
import h5py
from scipy import ndimage


def geoMERSI2(file, geoFile, outdir, scale, band):
    '''

    Parameters
    ----------
    file : 文件绝对路径
        需要校正的文件.
    geoFile : 文件绝对路径
        地理坐标存在的文件.
    afterGeoFile : tif文件绝对路径
        经过校正后的tif格式 Albers投影 250m空间分辨率.
    Returns
    -------
    None.

    '''
    os.chdir(r'F:\project\xv\work\data\FY3D\hdf')
    dataset = gdal.Open(file)
    # geoDataset = gdal.Open(geoFile)
    # print(gdal.Info(dataset)) # 查看元数据
    # print(gdal.Info(geoDataset))
    # subdataset = dataset.GetSubDatasets() # 查看子数据集
    # print(subdataset)
    '''
    查看子数据集位置：
       SUBDATASET_6_NAME=HDF5:"F:\FY3D_MERSI_data\FY3D_MERSI_GBAL_L1_20190403_0445_0250M_MS.hdf"://Data/EV_250_Emissive_b24
       SUBDATASET_6_DESC=[8000x8192] //Data/EV_250_Emissive_b24 (16-bit unsigned integer) 
    '''
    '''
    地理校正数据的位置：
      SUBDATASET_1_NAME=HDF5:"F:\FY3D_MERSI_data\FY3D_MERSI_GBAL_L1_20190403_0445_GEOQK_MS.hdf"://Latitude
      SUBDATASET_1_DESC=[8000x8192] //Latitude (32-bit floating-point)
      SUBDATASET_2_NAME=HDF5:"F:\FY3D_MERSI_data\FY3D_MERSI_GBAL_L1_20190403_0445_GEOQK_MS.hdf"://Longitude
      SUBDATASET_2_DESC=[8000x8192] //Longitude (32-bit floating-point)  
    '''
    vrtDir = os.path.splitext(file)[0] + '.vrt'
    subDataset = dataset.GetSubDatasets()[band][0]
    vrtFile = gdal.Translate(vrtDir,
                             subDataset,
                             format='vrt')
    '''
    需要写入描述GEOLOCATION元数据域的信息：
        <Metadata domain="GEOLOCATION">
         <MDI key="LINE_OFFSET">1</MDI>
         <MDI key="LINE_STEP">1</MDI>
         <MDI key="PIXEL_OFFSET">1</MDI>
         <MDI key="PIXEL_STEP">1</MDI>
         <MDI key="SRS">GEOGCS["WGS84",DATUM["WGS_1984",SPHEROID["WGS84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],TOWGS84[0,0,0,0,0,0,0],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9108"]],AUTHORITY["EPSG","4326"]]</MDI>
         <MDI key="X_BAND">1</MDI>
         <MDI key="X_DATASET">HDF5:"F:\FY3D_MERSI_data\FY3D_MERSI_GBAL_L1_20190403_0445_GEOQK_MS.hdf"://Longitude</MDI>
         <MDI key="Y_BAND">1</MDI>
         <MDI key="Y_DATASET">HDF5:"F:\FY3D_MERSI_data\FY3D_MERSI_GBAL_L1_20190403_0445_GEOQK_MS.hdf"://Latitude</MDI>
        </Metadata> 
    '''
    srs = osr.SpatialReference()
    srs.ImportFromProj4(
        '+proj=aea +lat_0=0 +lon_0=105 +lat_1=25 +lat_2=47 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs')

    lines = []
    with open(vrtDir, 'r') as f:
        for line in f:
            lines.append(line)
    lines.insert(1, '<Metadata domain="GEOLOCATION">\n')
    lines.insert(2, ' <MDI key="LINE_OFFSET">1</MDI>\n')
    lines.insert(3, ' <MDI key="LINE_STEP">' + str(scale) + '</MDI>\n')
    lines.insert(4, ' <MDI key="PIXEL_OFFSET">1</MDI>\n')
    lines.insert(5, ' <MDI key="PIXEL_STEP">' + str(scale) + '</MDI>\n')
    lines.insert(6,
                 ' <MDI key="SRS">GEOGCS["WGS84",DATUM["WGS_1984",SPHEROID["WGS84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],TOWGS84[0,0,0,0,0,0,0],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9108"]],AUTHORITY["EPSG","4326"]]</MDI>\n')
    lines.insert(7, ' <MDI key="X_BAND">1</MDI>\n')
    lines.insert(8, ' <MDI key="X_DATASET">HDF5:"{}"://Geolocation/Longitude</MDI>\n'.format(geoFile.split('\\')[-1]))
    lines.insert(9, ' <MDI key="Y_BAND">1</MDI>\n')
    lines.insert(10, ' <MDI key="Y_DATASET">HDF5:"{}"://Geolocation/Latitude</MDI>\n'.format(geoFile.split('\\')[-1]))
    lines.insert(11, '</Metadata>\n')
    with open(vrtDir, 'w') as f:
        for line in lines:
            f.writelines(line)
    '''
    geoData = gdal.Warp('F:\Py_project\gdalDealMERSI2\Warp_B24_WGS.tif',  
              'F:\Py_project\gdalDealMERSI2\MERSI2_0403_B24_vrt.vrt', 
              format='GTiff', geoloc=True, dstSRS="EPSG:4326",
              xRes=0.25, yRes=0.25)
    '''
    geoData = gdal.Warp(os.path.join(outdir, subDataset.split('/')[-1] + '.tif'), vrtDir,
                        format='GTiff', geoloc=True, dstSRS=srs,
                        resampleAlg=gdal.GRIORA_Bilinear, xRes=250, yRes=250, dstNodata=0)

    os.remove(vrtDir)
    if geoData is None:
        print('deal failure!')
    del geoData
    print('{} finish Geo\n'.format(subDataset))


def hdf2tiff(file, geoFile, outdir):
    if file[-12:-8] == '0250':
        for band in [7, 8, 9, 10]:
            geoMERSI2(file, geoFile, outdir, 4, band)
    elif file[-12:-8] == '1000':
        for band in [6]:
            geoMERSI2(file, geoFile, outdir, 1, band)
    else:
        print('Error FY-3 file!')
        return

# =====================================================================
# 辐射定标
# 计算公式
# dn=DN*Slope +Intercept 给定的官方文档说明了 反射通道中Slope值为1 Intercept值为0 所以这里省略该步骤
# Ref=Cal_2*dn2+Cal_1*dn+Cal_0
# 输入参数
# Cal：定标系数  HDF5 dataset
# DN：维度与定标系数的维度对应（注意不是一致是对应） HDF5 dataset
# sza：太阳天顶角
# 对于MersiⅡ 1000m Cal:shape(19,3) ----> DN:shape(19,c,r)
def calibration(Cal,DN,sza):
    DN2 = np.square(DN)
    Ref = np.ones(DN.shape)
    # 按通道分别进行计算
    for i in range(Cal.shape[0]):
        Ref[i,:,:] = (Cal[i,2] * DN2[i,:,:] + Cal[i,1] * DN[i,:,:] + Cal[i,0])/(100*np.cos(np.radians(sza)))
    return Ref

def fy_calibration(file_path, geofile_path, flag):
    # 由于250m和1000m数据文件hdf格式有区别这里 同时250m无自身的地理文件 需要进行插值处理 所以这里区分
    # 250m分辨率数据
    if flag == 0:
        with h5py.File(file_path, 'r') as hf:
            # 包含了19个可见光通道的定标系数 只取前4个通道
            VIS_Cal_Coeff = np.array(hf['Calibration']['VIS_Cal_Coeff'][0:4, :])
            # 250m数据文件的通道较少 这里直接拼接成数组进行处理
            EV_Ref = np.array(
                [hf['Data']['EV_250_RefSB_b1'], hf['Data']['EV_250_RefSB_b2'], hf['Data']['EV_250_RefSB_b3'],
                 hf['Data']['EV_250_RefSB_b4']])
        with h5py.File(geofile_path, 'r') as ghf:
            # 地理文件为1000m分辨率 对其进行插值
            # Bilinear interpolation would be order = 1,
            # nearest is order = 0,
            # and cubic is the default(order=3).
            Latitude = ndimage.zoom(ghf['Geolocation']['Latitude'], 4, order=1)
            Longitude = ndimage.zoom(ghf['Geolocation']['Longitude'], 4, order=1)
            SolarZenith = ndimage.zoom(ghf['Geolocation']['SolarZenith'], 4) * 0.01  # 太阳天顶角用于计算反射率
            SolarAzimuth = ndimage.zoom(ghf['Geolocation']['SolarAzimuth'], 4) * 0.01
            SensorZenith = ndimage.zoom(ghf['Geolocation']['SensorZenith'], 4) * 0.01
            SensorAzimuth = ndimage.zoom(ghf['Geolocation']['SensorAzimuth'], 4) * 0.01

    # 1000m分辨率数据
    elif flag == 1:
        with h5py.File(file_path, 'r') as hf:
            VIS_Cal_Coeff = np.array(hf['Calibration']['VIS_Cal_Coeff'])  # 定标参数
            EV_Ref = np.concatenate((np.array(hf['Data']['EV_250_Aggr.1KM_RefSB']), np.array(hf['Data']['EV_1KM_RefSB'])))  # 1000m反射通道dn值  250m反射通道dn值 聚合到1000m
        with h5py.File(geofile_path, 'r') as ghf:
            Latitude = np.array(ghf['Geolocation']['Latitude'])
            Longitude = np.array(ghf['Geolocation']['Longitude'])
            SolarZenith = np.array(ghf['Geolocation']['SolarZenith']) * 0.01  # 太阳天顶角用于计算反射率
            SolarAzimuth = np.array(ghf['Geolocation']['SolarAzimuth']) * 0.01
            SensorZenith = np.array(ghf['Geolocation']['SensorZenith']) * 0.01
            SensorAzimuth = np.array(ghf['Geolocation']['SensorAzimuth']) * 0.01
    else:
        print("标识值错误：必须为0或1")
        return 0

    # 进行辐射定标
    Ref = calibration(VIS_Cal_Coeff, EV_Ref, SolarZenith)

    # 存储结果
    # with h5py.File(os.path.join(r'F:\project\xv\work\data\FY3D\res', file_path.split('\\')[-1]), 'w') as ohf:
    #     geo = ohf.create_group("Geolocation")
    #     dt = ohf.create_group("Data")
    #     ohf.create_dataset('Geolocation/Latitude', data=Latitude)
    #     ohf.create_dataset('Geolocation/Longitude', data=Longitude)
    #     ohf.create_dataset('Geolocation/SolarZenith', data=SolarZenith)
    #     ohf.create_dataset('Data/EV_Ref', data=EV_Ref)
    #     ohf.create_dataset('Data/TOA_Ref', data=Ref)
    #
    # return
    return Ref


if __name__ == '__main__':
    # file0250 = r'F:\project\xv\work\data\FY3D\hdf\FY3D_MERSI_GBAL_L1_20200727_0615_0250M_MS.HDF'
    # file1000 = r'F:\project\xv\work\data\FY3D\hdf\FY3D_MERSI_GBAL_L1_20200727_0615_1000M_MS.HDF'
    # geoFile = r'F:\project\xv\work\data\FY3D\hdf\FY3D_MERSI_GBAL_L1_20200727_0615_GEO1K_MS.HDF'
    # outdir = r'F:\project\xv\work\data\FY3D\tiff'
    # # hdf2tiff(file0250, geoFile, outdir)
    # hdf2tiff(file1000, geoFile, outdir)

    # fy_dir = r'F:\project\xv\work\data\FY3D\hdf'
    # fy_band = []
    #
    # bands = [1, 2, 3, 4]
    #
    # file = r'FY3D_MERSI_GBAL_L1_20200727_0615_0250M_MS.HDF'
    # ds = gdal.Open(os.path.join(fy_dir, file))
    # subdatasets = ds.GetSubDatasets()
    #
    # for i in range(len(bands)):
    #     band = gdal.Open(subdatasets[bands[i] + 6][0])  # band1
    #     band_band = band.GetRasterBand(1)
    #     band_array = band_band.ReadAsArray()
    #     fy_band.append(band_array)
    #
    # bands = [5, 6, 7, 9]
    #
    # file = r'FY3D_MERSI_GBAL_L1_20200727_0615_1000M_MS.HDF'
    # ds = gdal.Open(os.path.join(fy_dir, file))
    # subdatasets = ds.GetSubDatasets()
    #
    # for i in range(len(bands)):
    #     band = gdal.Open(subdatasets[6][0])  # band1
    #     geoTrans = band.GetGeoTransform()
    #     proj = band.GetProjection()
    #     band_band = band.GetRasterBand(bands[i] - 4)
    #     band_array = band_band.ReadAsArray()
    #     band_array = transform.resize(band_array, (8000, 8192), order=0, preserve_range=True)
    #     plt.imshow(band_array)
    #     plt.show()
    #     fy_band.append(band_array)
    #
    #
    # fy_band = np.array(fy_band)
    # pass

    file0250 = r'F:\project\xv\work\data\FY3D\hdf\FY3D_MERSI_GBAL_L1_20200727_0615_0250M_MS.HDF'
    file1000 = r'F:\project\xv\work\data\FY3D\hdf\FY3D_MERSI_GBAL_L1_20200727_0615_1000M_MS.HDF'
    geoFile = r'F:\project\xv\work\data\FY3D\hdf\FY3D_MERSI_GBAL_L1_20200727_0615_GEO1K_MS.HDF'

    # fy_calibration(file0250, geoFile, 0)
    # fy_calibration(file1000, geoFile, 1)

    toa = []
    toa.append(fy_calibration(file0250, geoFile, 0))
    toa.append(fy_calibration(file1000, geoFile, 1))
    pass


