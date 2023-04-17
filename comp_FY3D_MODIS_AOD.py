#by Qianqian Jiang
#202304
#比较FY3D的AOD反演结果和MODIS AOD产品

import numpy as np
import random
from osgeo import gdal,gdal_array,gdalconst
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from matplotlib.colors import LogNorm
from matplotlib.pyplot import MultipleLocator
from AOD_func_FY3D import get_AOD_from_MOD09CMA,read_rad_info
import time
import os
import tkinter.filedialog

font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 20,
             }

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

def scatter_fy_with_modsi(fy_AOD,modis_AOD):
    fy_row=fy_AOD.shape[0]
    fy_col=fy_AOD.shape[1]
    i = 0
    fy_list = []
    modis_list = []
    while i < 5000:
        line = random.randint(1, fy_row - 2)
        sample = random.randint(1, fy_col - 2)
        if fy_AOD[line, sample] > 0 and modis_AOD[line, sample] > 0:
            fy_list.append(fy_AOD[line, sample] / 1000)
            modis_list.append(modis_AOD[line, sample] / 1000)
            i = i + 1
    fy_list = np.array(fy_list)
    modis_list = np.array(modis_list)
    X = np.array(modis_list).reshape(-1, 1)
    Y = np.array(fy_list).reshape(-1, 1)
    reg = LinearRegression()
    reg.fit(X, Y)
    r2 = reg.score(X, Y)
    a = reg.coef_
    b = reg.intercept_
    rmse = np.mean((modis_list - fy_list) ** 2) ** (0.5)

    bias = np.mean(-modis_list + fy_list)
    plt.hist2d(modis_list, fy_list, bins=100, norm=LogNorm())
    plt.colorbar()
    x1 = np.linspace(0, 1, 100)
    y1 = np.linspace(0, 1, 100)
    plt.plot(x1, y1, color='b')
    # y2 = a[0, 0] * x1 + b[0]
    # plt.plot(x1, y2, color='r')
    # popt, pcov= optimize.curve_fit(func_linear, lst_ref_list[:, i], gf_ref_list[:, i])
    # A1 = popt[0]
    # B1 = popt[1]
    # x2 = np.arange(0.001, 1, 0.001)
    # y2 = (A1*x2 + B1)
    # pl.plot(x2, y2, color='k')

    # 计算EE
    EE = abs(0.05 + 0.15 * x1)
    y2 = x1 - EE
    y3 = x1 + EE
    plt.plot(x1, y2, color='y', ls='--')
    plt.plot(x1, y3, color='y', ls='--')

    EE_num = 0
    uplimit = modis_list + abs(0.05 + 0.15 * modis_list)
    downlimit = modis_list - abs(0.05 + 0.15 * modis_list)
    for i in range(len(fy_list)):
        if uplimit[i] - fy_list[i] >= 0 and fy_list[i] - downlimit[i] >= 0:
            EE_num = EE_num + 1
    EE_num_per = EE_num / len(fy_list) * 100

    plt.ylim(0, 0.8)
    plt.xlim(0, 0.8)
    plt.title("Compare FY-3D AOD and MOD09CMA", font1)
    plt.xlabel("MOD09CMA", font1)
    plt.ylabel("FY-3D retrieved", font1)
    x_major_locator = MultipleLocator(0.2)
    # 把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator = MultipleLocator(0.2)
    # 把y轴的刻度间隔设置为10，并存在变量里
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    plt.grid()
    plt.text(0.05, 0.76, 'N = %d' % i, font1)
    # if b > 0:
    #     plt.text(0.05, 0.7, 'y=%.4f * x + %.4f' % (a, b), font1)
    # else:
    #     plt.text(0.05, 0.7, 'y=%.4f * x - %.4f' % (a, abs(b)), font1)
    # plt.text(0.05, 0.64, 'R2 = %.4f' % r2, font1)
    plt.text(0.05, 0.7, 'RMSE = %.4f' % rmse, font1)
    plt.text(0.05, 0.64, 'bias = %.4f' % bias, font1)
    plt.text(0.05, 0.58, '=EE : {:.2f}%'.format(EE_num_per), font1)
    plt.show()

if __name__=='__main__':
    t1 = time.time()
    default_dir = r"文件路径"
    inpath = tkinter.filedialog.askdirectory(title=u'选择FY3D文件夹', initialdir=(os.path.expanduser((default_dir))))
    # inpath = r"F:\jqq\高光谱项目\ac_code\ac_code\FY-3D\DATA\FY3D\hdf\3"
    flag = 1  # flag=0是250m,flag=1是1km
    # 读取影像辐射信息
    TOA_Ref_img, SZA_img, VZA_img, Lat_img, Long_img, CLM_img, date, geopath = read_rad_info(inpath, flag)
    if flag == 0:
        FY_AOD_path = geopath.replace('GEO1K', '250M').replace('.HDF', '_AOD.tiff')
    else:
        FY_AOD_path = geopath.replace('GEO1K', '1000M').replace('.HDF', '_AOD.tiff')
    FY_AOD, FY_row, FY_col, FY_band, FY_projection, FY_geoTransform = read_tiff(FY_AOD_path)
    default_dir = r"文件路径"
    MODIS_AOD_path = tkinter.filedialog.askopenfilename(title=u'选择MODIS AOD产品', initialdir=(os.path.expanduser((default_dir))))
    MODIS_AOD=get_AOD_from_MOD09CMA(TOA_Ref_img, Lat_img, Long_img, date, CLM_img, MODIS_AOD_path,inpath)
    FY_AOD_subset=FY_AOD[1056:1569,1193:1742]
    MODIS_AOD_subset=MODIS_AOD[1056:1569,1193:1742]
    scatter_fy_with_modsi(FY_AOD_subset,MODIS_AOD_subset)
