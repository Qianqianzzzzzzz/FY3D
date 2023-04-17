import numpy as np
from osgeo import gdal
from read_FY3D import read_spec_info,read_rad_info
from AOD_func_FY3D import get_AOD_from_MOD09CMA
import time
import tkinter.filedialog
import os

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

def extract_AOD_from_AERONET(siteloc,sitename,FY_AOD,FY_lat,FY_long,MODIS_AOD,AERONET_FILE):
    dis = np.array((FY_long - siteloc[0]) ** 2 + (FY_lat - siteloc[1]) ** 2)
    pt = np.where(dis == np.min(dis))
    line, sample = pt[0][0], pt[1][0]
    print(line, sample)
    print(SZA_img[line, sample], VZA_img[line, sample])

    time_img = int(date[-2:]) * 24 * 3600 + int(hhmm[:2]) * 3600 + int(hhmm[2:]) * 60
    if os.path.exists(AERONET_FILE):
        FILE_HANDLER = open(AERONET_FILE, 'r', encoding='utf-8', errors='ignore')
        dataList = FILE_HANDLER.readlines()
        aerofile_name_list = dataList[6].split(",")
        AOD_675_id = aerofile_name_list.index("AOD_675nm")
        AOD_500_id = aerofile_name_list.index("AOD_500nm")
        Ang_id = aerofile_name_list.index("440-675_Angstrom_Exponent")
        time_site_List = []
        file_line_num = []
        num = 7
        for lines in dataList[7:]:
            date_aeronet = lines.split(",")[0]
            if int(date_aeronet.split(":")[0]) == int(date[-2:]) and int(date_aeronet.split(":")[1]) == int(
                    date[4:6]) and int(date_aeronet.split(":")[2]) == int(date[:4]):
                time_aeronet = lines.split(",")[1]
                time_site = float(date_aeronet.split(":")[0]) * 24 * 3600 + (
                    float(time_aeronet.split(":")[0])) * 3600 + float(
                    time_aeronet.split(":")[1]) * 60 + float(time_aeronet.split(":")[2])
                time_site_List.append(time_site)
                file_line_num.append(int(num))
            num += 1
        time_site_List = abs(np.array(time_site_List) - time_img)
        nearest_time_id = np.where(time_site_List == np.min(time_site_List))[0]
        aod_500 = []
        aod_675 = []
        AngExp = []
        for i in range(len(nearest_time_id)):
            AOT_500 = float(dataList[int(file_line_num[nearest_time_id[i]])].split(",")[AOD_500_id])
            AOT_675 = float(dataList[int(file_line_num[nearest_time_id[i]])].split(",")[AOD_675_id])
            Ang_Ex = float(dataList[int(file_line_num[nearest_time_id[i]])].split(",")[Ang_id])
            if (AOT_500 >= 0) and (AOT_675 >= 0):
                # AngExp.append(math.log((AOT_675/AOT_500),675/500))
                AngExp.append(Ang_Ex)
                aod_500.append(AOT_500)
                aod_675.append(AOT_675)
            else:
                pass
        aod_675 = np.array(aod_675)
        aod_500 = np.array(aod_500)
        AngExp = np.array(AngExp)
        aod_550 = np.mean(aod_500 * pow(1.1, -AngExp))
    FILE_HANDLER.close()

    print("sitename AERONET FY3D MOD09CMA")
    print("%s %f %f %f\n" % (sitename,aod_550, FY_AOD[line, sample], MODIS_AOD[line, sample]))

if __name__=='__main__':
    t1 = time.time()
    default_dir = r"文件路径"
    inpath = tkinter.filedialog.askdirectory(title=u'选择FY3D文件夹', initialdir=(os.path.expanduser((default_dir))))
    # inpath = r"F:\jqq\高光谱项目\ac_code\ac_code\FY-3D\DATA\FY3D\hdf\3"
    flag = 1  # flag=0是250m,flag=1是1km
    # 读取影像辐射信息
    TOA_Ref_img, SZA_img, VZA_img, Lat_img, Long_img, CLM_img, date,hhmm, geopath = read_rad_info(inpath, flag)
    TOA_Ref_img = TOA_Ref_img / 10000

    if flag == 0:
        FY_AOD_path = geopath.replace('GEO1K', '250M').replace('.HDF', '_AOD.tiff')
    else:
        FY_AOD_path = geopath.replace('GEO1K', '1000M').replace('.HDF', '_AOD.tiff')
    FY_AOD, FY_row, FY_col, FY_band, FY_projection, FY_geoTransform = read_tiff(FY_AOD_path)

    default_dir = r"文件路径"
    MODIS_AOD_path = tkinter.filedialog.askopenfilename(title=u'选择MODIS AOD产品',
                                                        initialdir=(os.path.expanduser((default_dir))))
    MODIS_AOD = get_AOD_from_MOD09CMA(TOA_Ref_img, Lat_img, Long_img, date, CLM_img, MODIS_AOD_path, inpath)

    #站点经纬度
    # Beijing=[116.381,39.977]
    Beijing_CAMS = [116.317, 39.933]
    Beijing_RADI=[116.379,40.005]
    # AOE_Baotou=[109.629,40.852]
    Dalanzadgad=[104.419,43.577]

    AERONET_FILEPATH=r"I:\FY03D\FY3D\hdf\20200401_20200430_Beijing-CAMS.lev15"
    extract_AOD_from_AERONET(Beijing_CAMS,'Beijing_CAMS',FY_AOD,Lat_img,Long_img,MODIS_AOD,AERONET_FILEPATH)
    AERONET_FILEPATH = r"I:\FY03D\FY3D\hdf\20200401_20200430_Beijing_RADI.lev15"
    extract_AOD_from_AERONET(Beijing_RADI, 'Beijing_RADI', FY_AOD, Lat_img, Long_img, MODIS_AOD,AERONET_FILEPATH )
    AERONET_FILEPATH = r"I:\FY03D\FY3D\hdf\20200401_20200430_Dalanzadgad.lev15"
    extract_AOD_from_AERONET(Dalanzadgad, 'Dalanzadgad', FY_AOD, Lat_img, Long_img, MODIS_AOD, AERONET_FILEPATH)
