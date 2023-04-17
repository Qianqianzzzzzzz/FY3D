#by Qianqian Jiang
#20230406
#从FY3D的MERSI数据中读取出需要的信息，只处理250m的1-4波段和1km的1-7波段
import h5py
import numpy as np
import os
import scipy.ndimage
import matplotlib.pyplot as plt
import pandas as pd
from osgeo import gdal
import tkinter.filedialog
import math
def calibration(DN,Slope,Intercept,Cal,Sun_dist,SZA):
    TOA_ref=[]
    for i in range(DN.shape[0]):
        dn=DN[i,:,:]*Slope[i]+Intercept[i]
        Ref =(Cal[i,2] * np.square(dn) + Cal[i,1] * dn + Cal[i,0])/100
        TOA_ref.append(Sun_dist**2*Ref/np.cos(np.radians(SZA)))
    TOA_ref=np.array(TOA_ref)
    return TOA_ref
def read_rad_info(path,flag):
    files=os.listdir(path)
    for file in files:
        if file[-4:]=='.HDF':
            if file.split('_')[4]=='CLM':
                clm_path=os.path.join(path,file)
            if file.split('_')[6]=='GEO1K':
                date = file.split('_')[4]
                hhmm=file.split('_')[5]
                geofile_path=os.path.join(path,file)
                if flag==0:
                    file_path = os.path.join(path, file.replace('GEO1K', '0250M'))
                elif flag==1:
                    file_path=os.path.join(path,file.replace('GEO1K','1000M'))

    with h5py.File(clm_path, 'r') as chf:
        Cloud_Mask = np.array(chf['Cloud_Mask'])[0, :, :]
    # 对250m分辨率的数据进行辐射定标，并读取相应地理信息和云掩膜信息。
    if flag==0:
        # 首先对云检测文件进行最邻近插值 注意其它插值方法不可行
        Cloud_Mask = scipy.ndimage.zoom(Cloud_Mask, 4, order=0)
        with h5py.File(file_path, 'r') as hf:
            SolarDis = hf.attrs.get("EarthSun Distance Ratio")
            # 包含了19个可见光通道的定标系数 只取前4个通道
            VIS_Cal_Coeff = np.array(hf['Calibration']['VIS_Cal_Coeff'][0:4, :])
            # 250m数据文件的通道较少 这里直接拼接成数组进行处理
            EV_Ref = np.array(
                [hf['Data']['EV_250_RefSB_b1'][:], hf['Data']['EV_250_RefSB_b2'][:], hf['Data']['EV_250_RefSB_b3'][:],
                 hf['Data']['EV_250_RefSB_b4'][:]])
            EV_slop = np.hstack(
                (hf['Data']['EV_250_RefSB_b1'].attrs.get("Slope"), hf['Data']['EV_250_RefSB_b2'].attrs.get("Slope"),
                 hf['Data']['EV_250_RefSB_b3'].attrs.get("Slope"), hf['Data']['EV_250_RefSB_b4'].attrs.get("Slope")))
            EV_intercept = np.hstack(
                (hf['Data']['EV_250_RefSB_b1'].attrs.get("Intercept"),hf['Data']['EV_250_RefSB_b2'].attrs.get("Intercept"),
            hf['Data']['EV_250_RefSB_b3'].attrs.get("Intercept"),hf['Data']['EV_250_RefSB_b4'].attrs.get("Intercept")))
        with h5py.File(geofile_path, 'r') as ghf:
            # 地理文件为1000m分辨率 对其进行插值
            # Bilinear interpolation would be order = 1,
            # nearest is order = 0,
            # and cubic is the default(order=3).
            Latitude = scipy.ndimage.zoom(ghf['Geolocation']['Latitude'], 4, order=1)
            Longitude = scipy.ndimage.zoom(ghf['Geolocation']['Longitude'], 4, order=1)
            SolarZenith = scipy.ndimage.zoom(ghf['Geolocation']['SolarZenith'], 4) * 0.01  # 太阳天顶角用于计算反射率
            ViewZenith = scipy.ndimage.zoom(ghf['Geolocation']['SensorZenith'], 4) * 0.01  # 卫星天顶角用于计算反射率
    elif flag==1:
        #对1000m分辨率的数据进行辐射定标，并读取相应地理信息和云掩膜信息
        with h5py.File(file_path, 'r') as hf:
            SolarDis=hf.attrs.get("EarthSun Distance Ratio")
            VIS_Cal_Coeff = np.array(hf['Calibration']['VIS_Cal_Coeff'][:])  # 定标参数
            EV_Ref=np.vstack((hf['Data']['EV_250_Aggr.1KM_RefSB'][:],hf['Data']['EV_1KM_RefSB'][:]))# 1000m反射通道dn值  250m反射通道dn值 聚合到1000m
            EV_slop=np.hstack((hf['Data']['EV_250_Aggr.1KM_RefSB'].attrs.get("Slope"),hf['Data']['EV_1KM_RefSB'].attrs.get("Slope")))
            EV_intercept=np.hstack([hf['Data']['EV_250_Aggr.1KM_RefSB'].attrs.get("Intercept"),hf['Data']['EV_1KM_RefSB'].attrs.get("Intercept")])
        with h5py.File(geofile_path, 'r') as ghf:
            Latitude = np.array(ghf['Geolocation']['Latitude'])
            Longitude = np.array(ghf['Geolocation']['Longitude'])
            SolarZenith = np.array(ghf['Geolocation']['SolarZenith']) * 0.01  # 太阳天顶角用于计算反射率
            ViewZenith = np.array(ghf['Geolocation']['SensorZenith']) * 0.01  # 卫星天顶角用于计算反射率

    #辐射定标
    TOA_Ref=calibration(EV_Ref,EV_slop,EV_intercept,VIS_Cal_Coeff,SolarDis,SolarZenith)*10000

    # 识别有云位置
    clm = np.bitwise_and(Cloud_Mask, 7) != 7
    cloud_mask = np.zeros(clm.shape)
    cloud_mask[clm]=1
    return TOA_Ref,SolarZenith,ViewZenith,Latitude,Longitude,cloud_mask,date,hhmm,geofile_path
def read_spec_info(pathDir):
    #前7个波段
    BandID=[3,6,7,11,17,18,19]#在光谱响应函数文件夹里的对应编号
    band_num=7
    srcfiles=os.listdir(pathDir)
    name=srcfiles[0][:17]
    #读取中心波长和光谱响应函数
    Center_Wave=np.zeros(band_num)
    Spec=np.zeros([2200-390+1,band_num])#390-2200nm的范围
    for i in range(band_num):
        filename=name+str(BandID[i])+'.txt'
        filepath=os.path.join(pathDir,filename)
        fp=open(filepath,'r')
        band_wavelength=[]
        spec_response=[]
        datalist=fp.readlines()
        for line in datalist:
            band_wavelength.append(float(line.strip('\n').split(' ')[0])*1000)
            spec_response.append(float(line.strip('\n').split(' ')[1]))
        spec_response=np.array(spec_response)
        id=spec_response.argsort()
        Center_Wave[i]=band_wavelength[id[-1]]
        for j in range(390,2201,1):
            for k in band_wavelength:
                if j==k:
                    Spec[j-390,i]=spec_response[band_wavelength.index(k)]
                    break
    return Spec,Center_Wave



if __name__=='__main__':
    default_dir = r"文件路径"
    inpath = tkinter.filedialog.askdirectory(title=u'选择FY3D文件夹', initialdir=(os.path.expanduser((default_dir))))
    # inpath=r"F:\jqq\高光谱项目\ac_code\ac_code\FY-3D\DATA\FY3D\hdf\2"
    flag = 1  # 为0是250m,为1是1km
    #读取影像辐射信息
    Ref, SolarZenith, ViewZenith, Latitude, Longitude, cloud_mask,date,geopath=read_rad_info(inpath,flag)
    band_num = Ref.shape[0]
    row = Ref.shape[1]
    col = Ref.shape[2]
    #读取影像光谱信息
    fltbpath=r"F:\jqq\高光谱项目\ac_code\ac_code\FY-3D\DATA\rtcoef_fy3_4_mersi2_srf"
    Spec,Center_Wave=read_spec_info(fltbpath)
    if flag == 0:
        outfile = geopath.replace('GEO1K', '0250M').replace('.HDF', '.tiff')
    else:
        outfile = geopath.replace('GEO1K', '1000M').replace('.HDF', '.tiff')
    driver = gdal.GetDriverByName('GTiff')
    outraster = driver.Create(outfile, col, row, band_num, gdal.GDT_Int16)
    for i in range(band_num):
        outband = outraster.GetRasterBand(i + 1).WriteArray(Ref[i, :, :])
    outraster.FlushCache()
    if flag == 0:
        clmpath = geopath.replace('GEO1K', '0250M').replace('.HDF', '_clm.tiff')
    else:
        clmpath = geopath.replace('GEO1K', '1000M').replace('.HDF', '_clm.tiff')
    driver = gdal.GetDriverByName('GTiff')
    outraster = driver.Create(clmpath, col, row, 1, gdal.GDT_Int16)
    outband = outraster.GetRasterBand(1).WriteArray(cloud_mask)
    outraster.FlushCache()

