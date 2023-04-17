#by Qianqian Jiang
#20230406
#参考深蓝算法反演AOD
import h5py
import numpy as np
import os
import scipy.ndimage
import matplotlib.pyplot as plt
import pandas as pd
from osgeo import gdal,gdal_array,gdalconst
from pyproj import Proj
from scipy import optimize
from scipy.optimize import minimize
import time
import cv2
from scipy.interpolate import interp1d, griddata
import xml.dom.minidom
import tkinter.filedialog
import math
from read_FY3D import read_spec_info,read_rad_info

def calu_popt(y_array,x_array,weight_array):
    popt_c = np.sum(y_array/x_array*weight_array)/np.sum(weight_array)
    return popt_c

def read_tiff(inpath):
    ds = gdal.Open(inpath)
    if ds is None:
        print("Can't Open {}".format(inpath))
    else:
        row=ds.RasterYSize
        col=ds.RasterXSize
        band=ds.RasterCount
        projection=ds.GetProjection()
        geoTransform=ds.GetGeoTransform()
        data=ds.ReadAsArray(0, 0, col, row)
    ds=None
    return data,row,col,band,projection,geoTransform

def Read_LUT(MOD,SZA,VZA,AOD):
    MOD_List=[0,1,2,3,5]
    MOD=MOD_List.index(MOD)
    AOD_LIST = np.array([0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 0.80, 1.00, 1.20, 1.40,1.60,1.80,2.00])
    FILE_PATH =r"F:\jqq\高光谱项目\ac_code\ac_code\FY-3D\DATA\fy3d_LUT_6S.txt"
    Bandnum = 7
    para_SZA = SZA/10 - int(SZA / 10)
    para_VZA = VZA/10 - int(VZA / 10)
    AOD_Left = np.max(np.where(AOD_LIST <= AOD))
    AOD_Right = AOD_Left+1
    if AOD_Right >= len(AOD_LIST):
        VIS = AOD_LIST[-1]
        AOD_Right = len(AOD_LIST)-1
        para_AOD = 0
    else:
        para_AOD = (AOD - AOD_LIST[AOD_Left])/(AOD_LIST[AOD_Right] - AOD_LIST[AOD_Left])

    ID_start_vza1 = (MOD*8*8*len(AOD_LIST) + int(SZA / 10) *8*len(AOD_LIST) + int(VZA / 10) *len(AOD_LIST))*(Bandnum+1)+3
    ID_end_vza1 = (MOD*8*8*len(AOD_LIST) + int(SZA / 10) *8*len(AOD_LIST) + int(VZA / 10) *len(AOD_LIST) + 2*len(AOD_LIST))*(Bandnum+1)+3
    ID_start_vza2 = (MOD*8*8*len(AOD_LIST) + int(SZA / 10 + 1) *8*len(AOD_LIST) + int(VZA / 10) * len(AOD_LIST))*(Bandnum+1)+3
    ID_end_vza2 = (MOD*8*8*len(AOD_LIST) + int(SZA / 10 + 1) *8*len(AOD_LIST) + int(VZA / 10) *len(AOD_LIST) + 2*len(AOD_LIST))*(Bandnum+1)+3
    # 读取查找表
    FILE_HANDLER = open(FILE_PATH, encoding="utf-8")
    dataList = FILE_HANDLER.readlines()[ID_start_vza1:ID_end_vza2]
    i = 0

    Trans_LUT_1 = np.zeros((Bandnum, 2*len(AOD_LIST)))
    Ref_LUT_1 = np.zeros((Bandnum, 2*len(AOD_LIST)))
    Sph_albedo_LUT_1 = np.zeros((Bandnum, 2*len(AOD_LIST)))
    for line in dataList[0:ID_end_vza1 - ID_start_vza1]:
        if "id:" in line:
            continue
        Trans_LUT_1[i % Bandnum, i // Bandnum] = float(line.split('      ')[1])
        Ref_LUT_1[i % Bandnum, i // Bandnum] = float(line.split('      ')[3].strip('\n'))
        Sph_albedo_LUT_1[i % Bandnum, i // Bandnum] = float(line.split('      ')[2])
        i = i + 1
    i = 0
    Trans_LUT_2 = np.zeros((Bandnum, 2*len(AOD_LIST)))
    Ref_LUT_2 = np.zeros((Bandnum, 2*len(AOD_LIST)))
    Sph_albedo_LUT_2 = np.zeros((Bandnum, 2*len(AOD_LIST)))
    for line in dataList[ID_start_vza2 - ID_start_vza1:ID_end_vza2 - ID_start_vza1]:
        if "id:" in line:
            continue
        Trans_LUT_2[i % Bandnum, i // Bandnum] =float( line.split('      ')[1])
        Ref_LUT_2[i % Bandnum, i // Bandnum] = float(line.split('      ')[3].strip('\n'))
        Sph_albedo_LUT_2[i % Bandnum, i // Bandnum] =float( line.split('      ')[2])
        i = i + 1
    Trans_LUT = (1 - para_SZA) * ((1 - para_VZA) *Trans_LUT_1[:,0:len(AOD_LIST)] + para_VZA *Trans_LUT_1[:,len(AOD_LIST):2*len(AOD_LIST)]) + para_SZA * ((1 - para_VZA) *Trans_LUT_2[:,0:len(AOD_LIST)] + para_VZA *Trans_LUT_2[:,len(AOD_LIST):2*len(AOD_LIST)])
    Ref_LUT = (1 - para_SZA) * ((1 - para_VZA) *Ref_LUT_1[:,0:len(AOD_LIST)] + para_VZA *Ref_LUT_1[:,len(AOD_LIST):2*len(AOD_LIST)]) + para_SZA * ((1 - para_VZA) *Ref_LUT_2[:,0:len(AOD_LIST)] + para_VZA *Ref_LUT_2[:,len(AOD_LIST):2*len(AOD_LIST)])
    Sph_albedo_LUT = (1 - para_SZA) * ((1 - para_VZA) *Sph_albedo_LUT_1[:,0:len(AOD_LIST)] + para_VZA *Sph_albedo_LUT_1[:,len(AOD_LIST):2*len(AOD_LIST)]) + para_SZA * ((1 - para_VZA) *Sph_albedo_LUT_2[:,0:len(AOD_LIST)] + para_VZA *Sph_albedo_LUT_2[:,len(AOD_LIST):2*len(AOD_LIST)])
    FILE_HANDLER.close()
    del Trans_LUT_1, Trans_LUT_2, Sph_albedo_LUT_1, Sph_albedo_LUT_2, Ref_LUT_1, Ref_LUT_2
    Trans_LUT_aod = (1-para_AOD)*Trans_LUT[:,AOD_Left]+(para_AOD)*Trans_LUT[:,AOD_Right]
    Ref_LUT_aod = (1 - para_AOD) * Ref_LUT[:, AOD_Left] + (para_AOD) * Ref_LUT[:, AOD_Right]
    Sph_albedo_LUT_aod = (1 - para_AOD) * Sph_albedo_LUT[:, AOD_Left] + (para_AOD) * Sph_albedo_LUT[:, AOD_Right]
    return Trans_LUT_aod, Ref_LUT_aod, Sph_albedo_LUT_aod

def AOD_mat_filter(AOD_MAT):
    wins = 1
    AOD_MAT_NEW = AOD_MAT
    for i in range(0,AOD_MAT.shape[0],1):
        for j in range(0,AOD_MAT.shape[1],1):
            if (i >= wins)and(i < AOD_MAT.shape[0]-wins)and(j >= wins)and(j < AOD_MAT.shape[1]-wins):
                windows = AOD_MAT[i-wins:i+wins+1,j-wins:j+wins+1]
                AOD_MAT_NEW[i,j] = np.mean(windows[np.where(windows>0)])
            elif (i < wins)and(j >= wins)and(j < AOD_MAT.shape[1]-wins):
                windows = AOD_MAT[i:i+wins+1,j-wins:j+wins+1]
                AOD_MAT_NEW[i,j] = np.mean(windows[np.where(windows>0)])
            elif (i >= AOD_MAT.shape[0]-wins)and(j >= wins)and(j < AOD_MAT.shape[1]-wins):
                windows = AOD_MAT[i-wins:i,j-wins:j+wins+1]
                AOD_MAT_NEW[i,j] = np.mean(windows[np.where(windows>0)])
            elif (j < wins)and(i >= wins)and(i < AOD_MAT.shape[0]-wins):
                windows = AOD_MAT[i-wins:i+wins+1,j:j+wins+1]
                AOD_MAT_NEW[i,j] = np.mean(windows[np.where(windows>0)])
            elif (j >= AOD_MAT.shape[1]-wins)and(i >= wins)and(i < AOD_MAT.shape[0]-wins):
                windows = AOD_MAT[i-wins:i+wins+1,j-wins:j]
                AOD_MAT_NEW[i,j] = np.mean(windows[np.where(windows>0)])
            elif (i < wins) and (j<wins):
                windows = AOD_MAT[i:i + wins + 1, j:j+wins+1]
                AOD_MAT_NEW[i, j] = np.mean(windows[np.where(windows > 0)])
            elif (i < wins) and (j >= AOD_MAT.shape[1]-wins):
                windows = AOD_MAT[i:i + wins + 1, j-wins:j]
                AOD_MAT_NEW[i, j] = np.mean(windows[np.where(windows > 0)])
            elif (i >= AOD_MAT.shape[0]-wins) and (j<wins):
                windows = AOD_MAT[i-wins:i, j:j+wins+1]
                AOD_MAT_NEW[i, j] = np.mean(windows[np.where(windows > 0)])
            elif (i >= AOD_MAT.shape[0] - wins) and (j >= AOD_MAT.shape[1]-wins):
                windows = AOD_MAT[i - wins:i, j-wins:j]
                AOD_MAT_NEW[i, j] = np.mean(windows[np.where(windows > 0)])
            else:
                pass

    return AOD_MAT_NEW

def get_sr_from_MODIS_BlueSRD(TOA_Ref_img,Latitude,Longitude,date,inpath):
    month=int(date[4:6])
    # month=5
    modispath = r"I:\AOD\AOD\MOD09A1\global\1km"
    pathDir = os.listdir(modispath)
    for s in pathDir:
        if s[-4:] == '.tif':
            if int(s.split('_')[2]) == month:
                LSRPath = modispath + r"/" + s
                data_band, LSR_row, LSR_col, LSR_band, LSR_projection, LSR_geoTransform = read_tiff(LSRPath)
                Lat_data = np.array(data_band)
                break
    LSR_data = Lat_data

    # 影像匹配
    Row = TOA_Ref_img.shape[1]
    Col = TOA_Ref_img.shape[2]
    LSR_data_cal = np.zeros([Row, Col])
    p = Proj(LSR_projection)
    xy = p(Longitude, Latitude)
    LSR_pixel_col = (xy[0] - LSR_geoTransform[0]) / LSR_geoTransform[1]
    LSR_pixel_row = (xy[1] - LSR_geoTransform[3]) / LSR_geoTransform[5]
    LSR_pixel_col = np.array(LSR_pixel_col).astype(np.int_)
    LSR_pixel_row = np.array(LSR_pixel_row).astype(np.int_)
    LSR_data_cal = LSR_data[LSR_pixel_row, LSR_pixel_col]

    # outpath = inpath + r"\modis_bluesr_1km.tiff"
    # driver = gdal.GetDriverByName('GTiff')
    # outraster = driver.Create(outpath, Col, Row, 1, gdal.GDT_Int16)
    # outraster.GetRasterBand(1).WriteArray(LSR_data_cal)
    # outraster.FlushCache()

    return LSR_data_cal

def get_AOD_from_MCD19A2(TOA_Ref_img,Latitude,Longitude,date,MCD19A2,inpath):
    data_band, LSR_row, LSR_col, LSR_band, LSR_projection, LSR_geoTransform = read_tiff(MCD19A2)
    data_aod = np.array(data_band).astype(np.float_)
    # 对MCD19A2进行有效值合成
    data_aod[np.where(data_aod <= 0)[0], np.where(data_aod <= 0)[1], np.where(data_aod <= 0)[2]] = np.nan
    # 有效值合成
    data = np.nanmean(data_aod, axis=0)

    # 影像匹配
    Row = TOA_Ref_img.shape[1]
    Col = TOA_Ref_img.shape[2]
    LSR_data_cal = np.zeros([Row, Col])
    p = Proj(LSR_projection)
    xy = p(Longitude, Latitude)
    LSR_pixel_col = (xy[0] - LSR_geoTransform[0]) / LSR_geoTransform[1]
    LSR_pixel_row = (xy[1] - LSR_geoTransform[3]) / LSR_geoTransform[5]
    LSR_pixel_col = np.array(LSR_pixel_col).astype(np.int_)
    LSR_pixel_col[LSR_pixel_col>=LSR_col]=0
    LSR_pixel_col[LSR_pixel_col <0]=0
    LSR_pixel_row = np.array(LSR_pixel_row).astype(np.int_)
    LSR_pixel_row[LSR_pixel_row>=LSR_row]=0
    LSR_pixel_row[LSR_pixel_row <0] = 0
    LSR_data_cal = data[LSR_pixel_row, LSR_pixel_col]
    outpath = inpath + r"\MCD19A2_1.tiff"
    driver = gdal.GetDriverByName('GTiff')
    outraster = driver.Create(outpath, Col, Row, 1, gdal.GDT_Int16)
    outraster.GetRasterBand(1).WriteArray(LSR_data_cal)
    outraster.FlushCache()

    return LSR_data_cal

def get_AOD_from_MOD09CMA(TOA_Ref_img,Latitude,Longitude,date,cloudmask,MOD09CMA,inpath):
    data_band, LSR_row, LSR_col, LSR_band, LSR_projection, LSR_geoTransform = read_tiff(MOD09CMA)
    data_aod = np.array(data_band)
    # 影像匹配
    Row = TOA_Ref_img.shape[1]
    Col = TOA_Ref_img.shape[2]
    LSR_data_cal = np.zeros([Row, Col])
    p = Proj(LSR_projection)
    xy = p(Longitude, Latitude)
    LSR_pixel_col = (xy[0] - LSR_geoTransform[0]) / LSR_geoTransform[1]
    LSR_pixel_row = (xy[1] - LSR_geoTransform[3]) / LSR_geoTransform[5]
    LSR_pixel_col = np.array(LSR_pixel_col).astype(np.int_)
    LSR_pixel_col[LSR_pixel_col>=LSR_col]=0
    LSR_pixel_col[LSR_pixel_col <0]=0
    LSR_pixel_row = np.array(LSR_pixel_row).astype(np.int_)
    LSR_pixel_row[LSR_pixel_row>=LSR_row]=0
    LSR_pixel_row[LSR_pixel_row <0] = 0
    LSR_data_cal = data_aod[LSR_pixel_row, LSR_pixel_col]
    LSR_data_cal[cloudmask==1]=0
    outpath = inpath + r"\MOD09CMA.tiff"
    driver = gdal.GetDriverByName('GTiff')
    outraster = driver.Create(outpath, Col, Row, 1, gdal.GDT_Int16)
    outraster.GetRasterBand(1).WriteArray(LSR_data_cal)
    outraster.FlushCache()

    return LSR_data_cal

def get_pixelAOD(TOA_Ref,SR,AOD_List,Trans_LUT_List330,Ref_LUT_List330,Sph_albedo_LUT_List330,BlueBand):
    NIR_SR=TOA_Ref[5]
    TOA_List=[]
    for aod in range(0, len(AOD_List), 1):
        Trans_LUT = Trans_LUT_List330[aod][BlueBand]
        Ref_LUT = Ref_LUT_List330[aod][BlueBand]
        Sph_albedo_LUT = Sph_albedo_LUT_List330[aod][BlueBand]
        a = 1 - SR * (Sph_albedo_LUT)
        # 计算样本TOA辐亮度
        TOA_simu = Ref_LUT + Trans_LUT * SR / a
        TOA_List.append(TOA_simu)
    Derta_List = abs((np.array(TOA_List) - TOA_Ref[BlueBand])) #
    sort_sum = (Derta_List).argsort()
    VIS_Pred = 0
    num_list = np.arange(0, len(AOD_List), 1).astype(int)

    if (sort_sum[0] == 0) or (sort_sum[0] == len(num_list) - 1):
        TOA_aID = sort_sum[0]
        TOA_bID = sort_sum[1]
    else:
        TOA_aID = sort_sum[0] + 1
        TOA_bID = sort_sum[0] - 1
    aID = num_list[TOA_aID]
    bID = num_list[TOA_bID]
    AOD_A = AOD_List[aID]
    AOD_B = AOD_List[bID]

    y = TOA_Ref[BlueBand] - TOA_List[TOA_aID]
    x = TOA_List[TOA_bID] - TOA_List[TOA_aID]
    weight_array = 1
    popt = calu_popt(y, x, weight_array)
    # popt, pcov = optimize.curve_fit(func_linear,x,y)
    if (popt > 1) or (popt < 0):
        aID = num_list[sort_sum[0]]
        bID = num_list[sort_sum[1]]
        AOD_A = AOD_List[aID]
        AOD_B = AOD_List[bID]

        Trans_LUT = Trans_LUT_List330[aID][BlueBand]
        Ref_LUT = Ref_LUT_List330[aID][BlueBand]
        Sph_albedo_LUT = Sph_albedo_LUT_List330[aID][BlueBand]
        a = 1 - SR * (Sph_albedo_LUT)
        # 计算样本TOA辐亮度
        TOA_A = Ref_LUT + Trans_LUT * SR / a

        Trans_LUT = Trans_LUT_List330[bID][BlueBand]
        Ref_LUT = Ref_LUT_List330[bID][BlueBand]
        Sph_albedo_LUT = Sph_albedo_LUT_List330[bID][BlueBand]
        a = 1 - SR * (Sph_albedo_LUT)
        # 计算样本TOA辐亮度
        TOA_B = Ref_LUT + Trans_LUT * SR / a

        Derta_A = abs((TOA_A - TOA_Ref[BlueBand]))
        Derta_B = abs((TOA_B - TOA_Ref[BlueBand]))
        Derta = min([Derta_A, Derta_B])
        if Derta >= abs(TOA_Ref[BlueBand]) * 0.2:
            AOD_Pred_res = 0
            error_bands_res = 0
        else:
            if Derta_A > Derta_B:
                AOD_Pred = AOD_B
                TOA = TOA_B
            else:
                AOD_Pred = AOD_A
                TOA = TOA_A
            AOD_Pred_res = AOD_Pred
            error_bands_res = abs(TOA_Ref[BlueBand]-TOA)
    else:
        AOD_Pred = popt * (AOD_B - AOD_A) + AOD_A  # (1-popt[0])*AOD_A+popt[0]*AOD_B

        Trans_LUT = (1 - popt) * Trans_LUT_List330[aID][BlueBand] + popt * Trans_LUT_List330[bID][BlueBand]
        Ref_LUT = (1 - popt) * Ref_LUT_List330[aID][BlueBand] + popt * Ref_LUT_List330[bID][BlueBand]
        Sph_albedo_LUT = (1 - popt) * Sph_albedo_LUT_List330[aID][BlueBand] + popt * Sph_albedo_LUT_List330[bID][BlueBand]
        a = 1 - SR * (Sph_albedo_LUT)
        # 计算样本TOA辐亮度
        TOA = Ref_LUT + Trans_LUT * SR / a

        Derta = abs(TOA_Ref[BlueBand] - TOA)
        if Derta >= abs(TOA_Ref[BlueBand]) * 0.2:
            AOD_Pred_res = 0
            error_bands_res = 0
        else:
            # 计算surf
            AOD_Pred_res = AOD_Pred
            error_bands_res = abs(TOA_Ref[BlueBand]-TOA)

    #对初始的AOD值进行迭代
    BandID=[0,1,2,3,5,6]
    AOD_List=np.array(AOD_List)
    first_AOD=AOD_Pred_res
    aID = len(np.where(AOD_List <= first_AOD)[0]) - 1
    if aID == 0:
        bID = aID + 1
        Max_AOD = AOD_List[aID]
    elif aID == len(AOD_List) - 1:
        aID = len(AOD_List) - 2
        bID = aID + 1
        Max_AOD = AOD_List[bID]
    else:
        bID = aID + 1
    max_ID = aID
    para = (AOD_List[aID] - first_AOD) / ((AOD_List[aID] - AOD_List[bID]))
    Trans_LUT_23km = Trans_LUT_List330[aID] * (1 - para) + Trans_LUT_List330[bID] * para
    Ref_LUT_23km = Ref_LUT_List330[aID] * (1 - para) + Ref_LUT_List330[bID] * para
    Sph_albedo_LUT_23km = Sph_albedo_LUT_List330[aID] * (1 - para) + Sph_albedo_LUT_List330[bID] * para
    spec_Initial = (TOA_Ref - Ref_LUT_23km) / (Trans_LUT_23km + Sph_albedo_LUT_23km * (TOA_Ref - Ref_LUT_23km))

    TOA_List = []
    for aod in range(0, len(AOD_List), 1):
        Trans_LUT = Trans_LUT_List330[aod]
        Ref_LUT = Ref_LUT_List330[aod]
        Sph_albedo_LUT = Sph_albedo_LUT_List330[aod]
        a = 1 - spec_Initial * (Sph_albedo_LUT)
        # 计算样本TOA辐亮度
        TOA_simu = Ref_LUT + Trans_LUT * spec_Initial / a
        TOA_List.append(TOA_simu[BandID])
    Derta_List = np.sum(abs((np.array(TOA_List) - TOA_Ref[BandID])),axis=1 ) #
    sort_sum = (Derta_List).argsort()
    VIS_Pred = 0
    num_list = np.arange(0, len(AOD_List), 1).astype(int)

    if (sort_sum[0] == 0) or (sort_sum[0] == len(num_list) - 1):
        TOA_aID = sort_sum[0]
        TOA_bID = sort_sum[1]
    else:
        TOA_aID = sort_sum[0] + 1
        TOA_bID = sort_sum[0] - 1
    aID = num_list[TOA_aID]
    bID = num_list[TOA_bID]
    AOD_A = AOD_List[aID]
    AOD_B = AOD_List[bID]

    y = TOA_Ref[BandID] - TOA_List[TOA_aID]
    x = TOA_List[TOA_bID] - TOA_List[TOA_aID]
    weight_array = 1
    popt = calu_popt(y, x, weight_array)
    # popt, pcov = optimize.curve_fit(func_linear,x,y)
    if (popt > 1) or (popt < 0):
        aID = num_list[sort_sum[0]]
        bID = num_list[sort_sum[1]]
        AOD_A = AOD_List[aID]
        AOD_B = AOD_List[bID]

        Trans_LUT = Trans_LUT_List330[aID]
        Ref_LUT = Ref_LUT_List330[aID]
        Sph_albedo_LUT = Sph_albedo_LUT_List330[aID]
        a = 1 - spec_Initial * (Sph_albedo_LUT)
        # 计算样本TOA辐亮度
        TOA_A = Ref_LUT + Trans_LUT * spec_Initial / a

        Trans_LUT = Trans_LUT_List330[bID]
        Ref_LUT = Ref_LUT_List330[bID]
        Sph_albedo_LUT = Sph_albedo_LUT_List330[bID]
        a = 1 - spec_Initial * (Sph_albedo_LUT)
        # 计算样本TOA辐亮度
        TOA_B = Ref_LUT + Trans_LUT * spec_Initial / a

        Derta_A = np.sum(abs((TOA_A[BandID] - TOA_Ref[BandID])))
        Derta_B = np.sum(abs((TOA_B[BandID] - TOA_Ref[BandID])))
        Derta = min([Derta_A, Derta_B])
        if Derta >= abs(TOA_Ref[BlueBand]) * 0.2:
            AOD_Pred_res = 0
            error_bands_res = 0
        else:
            if Derta_A > Derta_B:
                AOD_Pred = AOD_B
                TOA = TOA_B
            else:
                AOD_Pred = AOD_A
                TOA = TOA_A
            AOD_Pred_res = AOD_Pred
            error_bands_res =  np.sum(abs(TOA_Ref - TOA))
    else:
        AOD_Pred = popt * (AOD_B - AOD_A) + AOD_A  # (1-popt[0])*AOD_A+popt[0]*AOD_B

        Trans_LUT = (1 - popt) * Trans_LUT_List330[aID] + popt * Trans_LUT_List330[bID]
        Ref_LUT = (1 - popt) * Ref_LUT_List330[aID] + popt * Ref_LUT_List330[bID]
        Sph_albedo_LUT = (1 - popt) * Sph_albedo_LUT_List330[aID] + popt * Sph_albedo_LUT_List330[bID]
        a = 1 - spec_Initial * (Sph_albedo_LUT)
        # 计算样本TOA辐亮度
        TOA = Ref_LUT + Trans_LUT * spec_Initial / a

        Derta = np.sum(abs(TOA_Ref[BandID] - TOA[BandID]))
        if Derta >= abs(TOA_Ref[BlueBand]) * 0.2:
            AOD_Pred_res = 0
            error_bands_res = 0
        else:
            # 计算surf
            AOD_Pred_res = AOD_Pred
            error_bands_res =np.sum(abs(TOA_Ref[BandID] - TOA[BandID]))
    return AOD_Pred_res,error_bands_res

def get_imgAOD(TOA_Ref,Lat,Long,date,CLM,SZA,VZA,Row,Col,band_num,spec,centerwave,inpath):
    time1=time.time()
    MODIS_SR_orginal = get_sr_from_MODIS_BlueSRD(TOA_Ref, Lat, Long, date,inpath)
    MODIS_SR_orginal = MODIS_SR_orginal / 10000
    #进行波段转换
    MODIS_SR=MODIS_SR_orginal*1.0041+0.0007
    # MODIS_SR = MODIS_SR_orginal
    # mcd19a2=get_AOD_from_MCD19A2(TOA_Ref, Lat, Long,date,'I:\FY03D\FY3D\hdf\\3\MCD19A2.A2020121.h25v04.006.2020125224624modis_AOD.tif',inpath)
    # mod09cma = get_AOD_from_MOD09CMA(TOA_Ref, Lat, Long, date,CLM,'I:\FY03D\MOD09CMA.A2020209.061.2020338103709modis_AOD.tif')
    time2 = time.time()
    print("读取地表反射率数据库用时：%.2fs" % (time2 - time1))
    # 筛选用于计算的像元
    Blue_Band = 0
    Green_Band = 1
    Red_Band = 2
    NIR840_Band = 3
    NDVI_img = (TOA_Ref[NIR840_Band, :, :] - TOA_Ref[Red_Band, :, :]) / (
                TOA_Ref[NIR840_Band, :, :] + TOA_Ref[Red_Band, :, :])
    NDWI_img = (TOA_Ref[Green_Band, :, :] - TOA_Ref[NIR840_Band, :, :]) / (
                TOA_Ref[Green_Band, :, :] + TOA_Ref[NIR840_Band, :, :])
    dev_Vindow = np.zeros([Row, Col])
    for i in range(band_num):
        mean_img = cv2.blur(TOA_Ref[i, :, :], (3, 3))
        res_lut1 = mean_img ** 2
        res_lut2 = cv2.blur((TOA_Ref[i, :, :]) ** 2, (3, 3))
        dev_Vindow += np.sqrt(np.maximum(res_lut2 - res_lut1, 0))
    dev_Vindow[np.where(dev_Vindow == 0)] = 9999
    dev_Vindow[np.where(TOA_Ref[0, :, :] == 9999)] = 9999
    step = 5  # 筛选像元步长
    rowww = math.ceil(Row / step)
    colll = math.ceil(Col / step)
    pyramid_Ref = np.zeros([band_num, rowww, colll])
    pyramid_LSR = np.zeros([rowww, colll])
    pyramid_pt = np.zeros([2, rowww, colll])
    pyramid_cloud_mask = np.zeros([rowww, colll])
    # 筛选计算像元
    lines = 0
    while lines < Row:
        samples = 0
        while samples < Col:
            # print(str(lines)+","+str(samples))
            # 选择窗口内异质性最小的9个像元和他周围的四个像元
            lines_end = min(step + lines, Row)
            samples_end = min(step + samples, Col)
            # 在窗口内查找NDVI最大值
            NDVI_Window = NDVI_img[lines:lines_end, samples:samples_end]
            NDWI_Window = NDWI_img[lines:lines_end, samples:samples_end]
            Max_NDVI = np.max(NDVI_Window)
            Min_DEV = np.min(dev_Vindow[lines:lines_end, samples:samples_end])
            TOA_Win = np.zeros(TOA_Ref[:, lines:lines_end, samples:samples_end].shape)
            TOA_Win[:, :, :] = TOA_Ref[:, lines:lines_end, samples:samples_end]
            # TOA_Win[np.where(TOA_Win <= 0)] = np.nan
            # 优先选择植被像元 进行光谱库匹配
            if Max_NDVI > 0.5:
                pt = np.where(NDVI_Window[lines:lines_end, samples:samples_end] == Max_NDVI)
                if len(pt[0]) > 1:
                    pt_x = pt[0][0] + lines
                    pt_y = pt[1][0] + samples
                    if NDWI_Window[pt[0][0], pt[1][0]] < 0:
                        pyramid_pt[0, int(lines / step), int(samples / step)] = pt[0][0] + lines
                        pyramid_pt[1, int(lines / step), int(samples / step)] = pt[1][0] + samples
                        pyramid_Ref[:, int(lines / step), int(samples / step)] = TOA_Ref[:, pt[0][0] + lines,
                                                                                 pt[1][0] + samples].reshape(band_num)

                        pyramid_LSR[int(lines / step), int(samples / step)] = MODIS_SR[
                            pt[0][0] + lines, pt[1][0] + samples]
                        pyramid_cloud_mask[int(lines / step), int(samples / step)] = CLM[
                            pt[0][0] + lines, pt[1][0] + samples]
                else:
                    if NDWI_Window[pt[0], pt[1]] < 0:
                        pyramid_pt[0, int(lines / step), int(samples / step)] = pt[0] + lines
                        pyramid_pt[1, int(lines / step), int(samples / step)] = pt[1] + samples
                        pyramid_Ref[:, int(lines / step), int(samples / step)] = TOA_Ref[:, pt[0] + lines,
                                                                                 pt[1] + samples].reshape(band_num)
                        pyramid_LSR[int(lines / step), int(samples / step)] = MODIS_SR[pt[0] + lines, pt[1] + samples]
                        pyramid_cloud_mask[int(lines / step), int(samples / step)] = CLM[pt[0] + lines, pt[1] + samples]
            elif Min_DEV < 0.2:
            # if Min_DEV < 0.2:
                pt = np.where(dev_Vindow[lines:lines_end, samples:samples_end] == Min_DEV)
                if len(pt[0]) > 1:
                    pt_x = pt[0][0] + lines
                    pt_y = pt[1][0] + samples
                    if NDWI_Window[pt[0][0], pt[1][0]] < 0:
                        pyramid_pt[0, int(lines / step), int(samples / step)] = pt[0][0] + lines
                        pyramid_pt[1, int(lines / step), int(samples / step)] = pt[1][0] + samples
                        pyramid_Ref[:, int(lines / step), int(samples / step)] = TOA_Ref[:, pt[0][0] + lines,
                                                                                 pt[1][0] + samples].reshape(band_num)
                        pyramid_LSR[int(lines / step), int(samples / step)] = MODIS_SR[
                            pt[0][0] + lines, pt[1][0] + samples]
                        pyramid_cloud_mask[int(lines / step), int(samples / step)] = CLM[
                            pt[0][0] + lines, pt[1][0] + samples]
                else:
                    if NDWI_Window[pt[0], pt[1]] < 0:
                        pyramid_pt[0, int(lines / step), int(samples / step)] = pt[0] + lines
                        pyramid_pt[1, int(lines / step), int(samples / step)] = pt[1] + samples
                        pyramid_Ref[:, int(lines / step), int(samples / step)] = TOA_Ref[:, pt[0] + lines,
                                                                                 pt[1] + samples].reshape(band_num)
                        pyramid_LSR[int(lines / step), int(samples / step)] = MODIS_SR[pt[0] + lines, pt[1] + samples]
                        pyramid_cloud_mask[int(lines / step), int(samples / step)] = CLM[pt[0] + lines, pt[1] + samples]
            else:
                pass
            try:
                # 如果前面的条件都不满足，则取30*30窗口内蓝绿波段反射率最小的像元，进行光谱库匹配
                if (pyramid_Ref[:, int(lines / step), int(samples / step)] == 0).all():
                    min_sum = np.min(TOA_Win[Blue_Band, :, :])
                    pt = np.where(TOA_Win[Blue_Band, :, :] == min_sum)
                    if len(pt[0]) > 1:
                        pyramid_pt[0, int(lines / step), int(samples / step)] = pt[0][0] + lines
                        pyramid_pt[1, int(lines / step), int(samples / step)] = pt[1][0] + samples
                        pyramid_Ref[:, int(lines / step), int(samples / step)] = TOA_Ref[:, pt[0][0] + lines,
                                                                                 pt[1][0] + samples].reshape(band_num)
                        pyramid_LSR[int(lines / step), int(samples / step)] = MODIS_SR[
                            pt[0][0] + lines, pt[1][0] + samples]
                        pyramid_cloud_mask[int(lines / step), int(samples / step)] = CLM[
                            pt[0][0] + lines, pt[1][0] + samples]
                    else:
                        pyramid_pt[0, int(lines / step), int(samples / step)] = pt[0] + lines
                        pyramid_pt[1, int(lines / step), int(samples / step)] = pt[1] + samples
                        pyramid_Ref[:, int(lines / step), int(samples / step)] = TOA_Ref[:, pt[0] + lines,
                                                                                 pt[1] + samples].reshape(band_num)
                        pyramid_LSR[int(lines / step), int(samples / step)] = MODIS_SR[pt[0] + lines, pt[1] + samples]
                        pyramid_cloud_mask[int(lines / step), int(samples / step)] = CLM[pt[0] + lines, pt[1] + samples]
            except:
                a = 0
            samples += step
        lines += step
    time3 = time.time()
    print("筛选计算像元用时：%.2fs" % (time3 - time2))
    # 截取影像中间一块
    # Ref=Ref[:,4311:4697,3135:3554]
    # SolarZenith=SolarZenith[4311:4697,3135:3554]
    # ViewZenith=ViewZenith[4311:4697,3135:3554]
    # Latitude=Latitude[4311:4697,3135:3554]
    # Longitude=Longitude[4311:4697,3135:3554]
    # cloud_mask=cloud_mask[4311:4697,3135:3554]
    ###############################对整景影像分幅进行计算，分成1行5列################################
    AOD_pyramid_img = np.zeros([2, rowww, colll])
    n = 10
    width = int(Col / n)
    for i in range(n):
        # i=6
        start_col = i * width
        end_col = min((i + 1) * width, Col)
        SolarZenith = SZA[:, start_col:end_col]
        ViewZenith = VZA[:, start_col:end_col]
        Latitude = Lat[:, start_col:end_col]
        Longitude = Long[:, start_col:end_col]
        minLat, maxLat, meanLat = np.min(Latitude), np.max(Latitude), np.mean(Latitude)
        minLong, maxLong, meanLong = np.min(Longitude), np.max(Longitude), np.mean(Longitude)
        minSZA, maxSZA, meanSZA = np.min(SolarZenith), np.max(SolarZenith), np.median(SolarZenith)
        minVZA, maxVZA, meanVZA = np.min(ViewZenith), np.max(ViewZenith), np.median(ViewZenith)
        print("最小经纬:%f,最低纬度:%f" % (minLong, minLat))
        print("最大经纬:%f,最高纬度:%f" % (maxLong, maxLat))
        print("中心经纬:%f,中心纬度:%f" % (maxLong, maxLat))
        print("最小SZA:%f,最大SZA:%f,中位数SZA：%f" % (minSZA, maxSZA, meanSZA))
        print("最小VZA:%f,最大VZA:%f,中位数VZA：%f" % (minVZA, maxVZA, meanVZA))
        Month = int(date[4:6])
        # 确定气溶胶类型
        MOD = 1

        # 查找表确定AOD初值
        AOD_List = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 0.80, 1.00, 1.20, 1.40, 1.60, 1.80, 2.00]

        Trans_LUT_List330 = []
        Ref_LUT_List330 = []
        Sph_albedo_LUT_List330 = []
        # 读取 VIS LUT列表（GF5B对应LUT)
        for aod in AOD_List:
            Trans_LUT, Ref_LUT, Sph_albedo_LUT = Read_LUT(MOD, meanSZA, meanVZA, aod)
            Trans_LUT_List330.append(Trans_LUT)
            Ref_LUT_List330.append(Ref_LUT)
            Sph_albedo_LUT_List330.append(Sph_albedo_LUT)

        # 把筛选的像元也分成10份
        width_pyramid = int(colll / 10)
        start_col_pyramid = i * width_pyramid
        end_col_pyramid = min((i + 1) * width_pyramid, colll)
        Ref = pyramid_Ref[:, :, start_col_pyramid:end_col_pyramid]
        cloud_mask = pyramid_cloud_mask[:, start_col_pyramid:end_col_pyramid]
        modis_sr = pyramid_LSR[:, start_col_pyramid:end_col_pyramid]

        for lines in range(Ref.shape[1]):
            for samples in range(Ref.shape[2]):
                # lines=258
                if cloud_mask[lines, samples] == 1:
                    AOD_pyramid_img[0, lines, samples + start_col_pyramid], AOD_pyramid_img[
                        1, lines, samples + start_col_pyramid] = 0, 0
                else:
                    AOD_pyramid_img[0, lines, samples + start_col_pyramid], AOD_pyramid_img[
                        1, lines, samples + start_col_pyramid] = get_pixelAOD(
                        Ref[:, lines, samples], modis_sr[lines, samples], AOD_List, Trans_LUT_List330,
                        Ref_LUT_List330, Sph_albedo_LUT_List330, Blue_Band)
    time4 = time.time()
    print("获取整景影像金字塔AOD用时：%.2fs" % (time4 - time3))
    # 整景影像进行插值
    try:
        # 剔除误差大的点
        valid_id = AOD_pyramid_img[0, :, :] > 0
        AOD_pyramid_img[:, np.where(AOD_pyramid_img[0, :, :] == 0)[0],
        np.where(AOD_pyramid_img[0, :, :] == 0)[1]] = np.nan
        AOD_pyramid_img[:, np.where(AOD_pyramid_img[0, :, :] == np.nan)[0],
        np.where(AOD_pyramid_img[0, :, :] == np.nan)[1]] = np.nan
        dev_Std = np.nanstd(AOD_pyramid_img[1, :, :], ddof=1)
        dev_id2 = np.where(abs(AOD_pyramid_img[1, :, :] - np.nanmin(AOD_pyramid_img[1, :, :])) >= dev_Std * 3)
        dev_ID = np.where(abs(AOD_pyramid_img[1, :, :] - np.nanmin(AOD_pyramid_img[1, :, :])) < dev_Std * 3)
        dev_id = abs(AOD_pyramid_img[1, :, :] - np.nanmin(AOD_pyramid_img[1, :, :])) < dev_Std * 3
        dev_id = valid_id & dev_id
        AOD_pyramid_img[0, dev_id2[0], dev_id2[1]] = -999

        AOD_pyramid_img[0, :, :] = AOD_mat_filter(AOD_pyramid_img[0, :, :])
        grid_x, grid_y = np.mgrid[0:Row:1, 0:Col:1]
        if (pyramid_pt != 0).any():
            grid_z0 = griddata(pyramid_pt[:, dev_id].T, AOD_pyramid_img[0, dev_id],
                               (grid_x, grid_y), method='nearest')
            pt_List = np.array([[0, 0], [0, Col - 1], [Row - 1, 0], [Row - 1, Col - 1]])
            points_arr = np.zeros([2, len(dev_ID[0]) + 4]).astype(int)
            points_arr[:, :4] = pt_List.T
            points_arr[:, 4:] = pyramid_pt[:, dev_id]
            grid_aod = griddata(points_arr.T, grid_z0[points_arr[0, :], points_arr[1, :]], (grid_x, grid_y),
                                method='linear')
            AOD_Img_3d = grid_aod.reshape(Row, Col) * 1000
            print("finished AOD calculation")
            return AOD_Img_3d,True
    except:
        print("ERROR in AOD retrieval!")
        return -1,False

if __name__=='__main__':
    t1 = time.time()
    default_dir = r"文件路径"
    inpath = tkinter.filedialog.askdirectory(title=u'选择FY3D文件夹', initialdir=(os.path.expanduser((default_dir))))
    # inpath = r"F:\jqq\高光谱项目\ac_code\ac_code\FY-3D\DATA\FY3D\hdf\3"
    flag=1 #flag=0是250m,flag=1是1km
    # 读取影像辐射信息
    TOA_Ref_img, SZA_img, VZA_img, Lat_img, Long_img, CLM_img, date,hhmm,geopath = read_rad_info(inpath, flag)
    TOA_Ref_img=TOA_Ref_img/10000
    # 读取影像光谱信息
    fltbpath = r"F:\jqq\高光谱项目\ac_code\ac_code\FY-3D\DATA\rtcoef_fy3_4_mersi2_srf"
    Spec, Center_Wave = read_spec_info(fltbpath)
    t2=time.time()
    print("读取影像信息用时：%.2fs"%(t2-t1))
    if flag==0:
        band_num=4
    else:
        band_num=7
    TOA_Ref_img=TOA_Ref_img[:band_num,:,:]
    Row = TOA_Ref_img.shape[1]
    Col = TOA_Ref_img.shape[2]

    AOD_img,flag=get_imgAOD(TOA_Ref_img,Lat_img,Long_img,date,CLM_img,SZA_img,VZA_img,Row,Col,band_num,Spec,Center_Wave,inpath)
    AOD_img=np.array(AOD_img)
    if flag:
        AOD_img[CLM_img==1]=0
        if flag==0:
            outfile=geopath.replace('GEO1K','250M').replace('.HDF','_AOD.tiff')
        else:
            outfile=geopath.replace('GEO1K','1000M').replace('.HDF','_AOD.tiff')
        driver = gdal.GetDriverByName('GTiff')
        outraster = driver.Create(outfile, Col, Row, 1, gdal.GDT_Int16)
        outband = outraster.GetRasterBand(1).WriteArray(AOD_img)
        outraster.FlushCache()
        t3=time.time()
        print("AOD空间插值用时：%.2f秒" % (t3 - t2))
        print("反演AOD总用时：%.2f秒"%(t3-t1))
    else:
        print("AOD反演失败！")

